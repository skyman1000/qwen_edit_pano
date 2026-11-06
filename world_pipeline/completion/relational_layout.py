"""8B scene semantics -> bounded object-relative CPU search -> Adapter boxes.

This borrows constraint/search separation, not the full Holodeck implementation.
No room reconstruction, measured floor, meshes, new model or observed-box repair.
"""
import argparse
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import ELIGIBLE, finite_vector, new_output, read, sha, write
from .category_bridge import category as bridge_category, key, official_labels
from .geometry import CANONICAL_CATEGORIES, parse_json
from .placement_program import aligned_y
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .semantic_layout import build_states, diagnostics
from .semantic_layout_visual import render
from .semantic_probe import DEFAULT_BATCH, load_cases

PROTOCOL = 'object_relative_bounded_search_v2'
REVISION_PROTOCOL = 'sparse_id_keyed_placement_patch_v2'
RELATIONS = ('near', 'left_of', 'right_of', 'in_front_of', 'behind', 'on_top')
CAMERA_ID = '__camera__'
PLACEMENT_FIELDS = frozenset(('anchor_id', 'relation', 'gap_m', 'height_anchor_id',
                              'height_alignment', 'height_offset_m', 'reason'))


def prompt(case, capacity):
    context = [{**o, 'center': [round(v, 3) for v in o['center']],
                'full_aabb_size': [round(v, 3) for v in o['full_aabb_size']]} for o in case['context']]
    return (
        'Propose ONE plausible complete surrounding scene extending this partial RGB photograph. '
        'The fixed observations are visible evidence, not a complete room inventory. '
        'Infer the room purpose and choose useful distinct NEW objects beyond this view. '
        'Preserve the photograph and every observed object; do not redecorate visible empty areas '
        'or propose components already belonging to visible objects as new furniture. '
        'Additional instances of existing categories are allowed with a scene-specific purpose. '
        'Only structured additions commit existence; prose cannot add unlisted objects. '
        'Output a small RELATIVE placement specification, not numeric object centres or offsets. '
        'CPU search will calculate xyz from the chosen reference, sizes and relationship. '
        'Return JSON with scene_type, scene_hypothesis, additions. Each addition contains: '
        'id, name, raw_category (simple actual object noun), category (allowed canonical name or null), '
        'size (3 positive FULL camera-axis AABB lengths in metres), anchor_id, relation, gap_m, '
        'height_anchor_id, height_alignment, height_offset_m, reason. '
        'anchor_id is an observed ID or an EARLIER new ID. For a new functional area that is not '
        'adjacent to visible furniture, anchor_id may be __camera__ with relation=near; its gap_m '
        'then sets a preferred distance beyond the new box from the camera. This permits a separate '
        'surrounding area, not only accessories beside visible objects. Chains terminate at an '
        'observed ID or __camera__; no self references or cycles. Choose meaningful references. '
        'relation is near, left_of, right_of, in_front_of, behind, or on_top. Directions use CAMERA axes, '
        'not an object\'s facing direction. left/right order box edges along x; in_front_of means '
        'smaller z than the reference; behind means larger z. near allows any surrounding angle. '
        'gap_m>=0 is a preferred ray separation after the boxes stop overlapping horizontally; '
        'the CPU searches a bounded neighborhood and reports its actual displacement and deviation. '
        'Avoid distant placements. '
        'For on_top (e.g. a lamp on a newly listed bedside table), the CPU searches x/z positions '
        'within the anchor box footprint and puts the new bottom at its CAMERA-y top plus gap_m; use gap_m=0 '
        'for approximate contact. This is an AABB-top placement prior, not a measured support '
        'surface. No search slack is added to this vertical gap. The following height reference '
        'fields are unnecessary and unused for on_top. '
        'height_anchor_id must be OBSERVED. height_alignment is bottom, center, or top: align '
        'the same CAMERA-y box edge/center, then add the signed height_offset_m (usually 0). '
        'For floor-standing furniture use a relevant visible floor-standing object as the height '
        'reference, not wall art or a ceiling light. Small supported objects need a credible '
        'relative height explanation. Edge alignment is only a coarse prior, not physical support '
        'or a measured floor; tilted cameras and inflated AABBs can make it approximate. '
        'Axes: x right, y IMAGE-UP, z forward, camera-local metres. Gravity, floor, walls and '
        'room boundaries are unknown. No fixed y=-1.5 floor, region labels or direction quotas. '
        'The CPU tests candidates around your reference over all 360 degrees, requiring the whole '
        'new box to be outside the photograph and avoiding positive-volume AABB intersections. '
        'This version completes OFFSCREEN objects, not in-view objects hidden by occluders. '
        'The final RGB target is a 2:1 ERP panorama, not a long flat room. '
        'Use category=null for objects genuinely outside the Adapter vocabulary. A rug/carpet '
        'must not be relabelled furniture merely to enter the vocabulary. Keep its actual noun. '
        'No face/yaw/shape/support graph/room bounds, independent center, or xyz offset fields. '
        f'At most {capacity} additions, not a target count. No required direction coverage. '
        f'Input horizontal FOV={case["hfov"]:g} degrees. Return ONLY JSON. '
        '\nALLOWED CATEGORIES: '+', '.join(CANONICAL_CATEGORIES.values())
        +'\nFIXED OBSERVED: '+json.dumps(context, ensure_ascii=False, separators=(',', ':')))


def validate_spec(answer, observed, capacity):
    if (not isinstance(answer, dict) or any(not isinstance(answer.get(k), str) or not answer[k].strip()
            for k in ('scene_type', 'scene_hypothesis')) or not isinstance(answer.get('additions'), list)):
        raise ValueError('Need scene_type, scene_hypothesis and additions')
    items = deepcopy(answer['additions'])
    if len(items)>capacity:
        raise ValueError('Object budget exceeded; no truncation')
    fixed = {o['track_id'] for o in observed['objects']}; seen = fixed | {CAMERA_ID}
    labels, digest = official_labels()
    for obj in items:
        if (not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) or not obj[k].strip()
                for k in ('id','name','raw_category','anchor_id','reason')) or obj['id'] in seen):
            raise ValueError('Need distinct IDs, object name/noun, references and reason')
        seen.add(obj['id'])
        if obj.get('relation') not in RELATIONS:
            raise ValueError('Use a supported relation and camera-y alignment')
        if obj['anchor_id']==CAMERA_ID and obj['relation']!='near':
            raise ValueError('Camera roots use near; object-relative front/behind is not camera-facing language')
        if obj['relation']!='on_top':
            if (obj.get('height_alignment') not in ('bottom','center','top')
                    or not isinstance(obj.get('height_anchor_id'),str) or obj['height_anchor_id'] not in fixed):
                raise ValueError('Need a camera-y alignment and fixed observed height reference')
        else:
            obj['unused_height_fields'] = [k for k in ('height_anchor_id','height_alignment','height_offset_m') if k in obj]
        if not finite_vector(obj.get('size'),3) or min(obj['size'])<=0:
            raise ValueError('Need positive full camera-axis AABB size')
        for k in (('gap_m',) if obj['relation']=='on_top' else ('gap_m','height_offset_m')):
            if type(obj.get(k)) not in (int,float) or not math.isfinite(obj[k]) or (k=='gap_m' and obj[k]<0):
                raise ValueError('Need finite gap_m>=0 and signed height_offset_m')
        if any(k in obj for k in ('center','offset_m','region','center_y_m')):
            raise ValueError('Relative specification cannot also contain a numeric placement')
        if 'category' not in obj or (obj['category'] is not None and not isinstance(obj['category'],str)):
            raise ValueError('Category must be a string or null')
        raw_ids = labels.get(key(obj['raw_category']), set())
        obj['category_review'] = dict(model_category=obj['category'], raw_category=obj['raw_category'],
                                     semantics_verified=False)
        # An exact known floor/wall/etc noun cannot be forced into an object ID.
        # Unknown nouns are NOT mapped by fuzzy matching; the model's explicit
        # coarse canonical selection remains an unverified semantic prediction.
        if len(raw_ids)==1 and not raw_ids <= ELIGIBLE:
            obj['category_review'].update(reason='exact_official_noun_outside_adapter_vocabulary',
                official_mpcat40_id=next(iter(raw_ids)), mapping_sha256=digest)
            obj['category'] = None
        elif obj['category'] is not None:
            try:
                canonical, mapping = bridge_category(obj['category'])
                obj['category'] = canonical
                if mapping: obj['category_review']['mapping'] = mapping
            except ValueError:
                # An out-of-vocabulary proposal is a reviewable candidate,
                # not grounds for losing every otherwise supported object.
                obj['category_review']['reason'] = 'unresolved_category_excluded_from_adapter'
                obj['category'] = None
    pending = list(items); ordered = []; resolved = fixed | {CAMERA_ID}
    while pending:
        ready = [o for o in pending if o['anchor_id'] in resolved]
        if not ready:
            raise ValueError('Unknown/self/cyclic placement reference; cannot reach observed geometry')
        for obj in ready:
            ordered.append(obj); resolved.add(obj['id']); pending.remove(obj)
    return dict(answer, additions=ordered)


def intersect(a, b):
    return bool(np.all((np.array(a['size'])+b['size'])/2-abs(np.array(a['center'])-b['center'])>1e-6))


def relation_holds(obj, reference, relation):
    if relation=='near': return True
    axis = 1 if relation=='on_top' else 0 if relation in ('left_of','right_of') else 2
    sign = -1 if relation in ('left_of','in_front_of') else 1
    return sign*(obj['center'][axis]-reference['center'][axis]) >= (obj['size'][axis]+reference['size'][axis])/2-1e-7


def candidates(obj, reference, height_reference, fixed, hfov, image_size, angle_step, gap_slack, rng, counts=None):
    size = np.array(obj['size']); ref_size = np.array(reference['size']); c = np.array(reference['center'])
    atop = obj['relation']=='on_top'
    y = c[1]+(ref_size[1]+size[1])/2+obj['gap_m'] if atop else aligned_y(height_reference, size, obj['height_alignment'])+obj['height_offset_m']
    th = math.tan(math.radians(hfov)/2); tv = th*image_size[1]/image_size[0]
    normals = np.array([[1,0,th],[-1,0,th],[0,1,tv],[0,-1,tv],[0,0,1]])
    half = (size+ref_size)[[0,2]]/2
    gaps = sorted(set([obj['gap_m'], *np.linspace(max(0,obj['gap_m']-gap_slack),obj['gap_m']+gap_slack,5)]))
    options = []
    counts = counts if counts is not None else {}
    counts.update(sampled=0,relation=0,input_fov=0,observed_aabb_overlap=0,support_footprint=0,accepted=0)
    positions = []
    if atop:
        extent = (ref_size-size)[[0,2]]/2
        if min(extent)<-1e-7:
            counts['support_footprint'] = 1
            return []
        extent = np.maximum(extent,0)
        # on_top is a relation, not an additional centre-alignment constraint.
        # Sample the finite parent AABB footprint, including its centre/edges.
        # This is NOT a claim that an inflated camera AABB is a real tabletop.
        for dx in sorted(set(np.linspace(-extent[0],extent[0],5))):
            for dz in sorted(set(np.linspace(-extent[1],extent[1],5))):
                positions.append((c[0]+dx,c[2]+dz,None,obj['gap_m']))
    else:
        for angle in np.arange(-180,180,angle_step):
            u = np.array([math.sin(math.radians(angle)),math.cos(math.radians(angle))])
            # Ray exits the Minkowski rectangle of the two camera-x/z AABBs.
            touching_distance = min(half[i]/abs(u[i]) for i in range(2) if abs(u[i])>1e-10)
            for gap in gaps:
                x,z = c[[0,2]]+u*(touching_distance+gap)
                positions.append((x,z,float(angle),gap))
    for x,z,angle,gap in positions:
        center = np.array([x,y,z]); box = dict(id=obj['id'],center=center.tolist(),size=obj['size'])
        counts['sampled'] += 1
        if not relation_holds(box,reference,obj['relation']): counts['relation'] += 1; continue
        if not np.any(normals@center+abs(normals)@(size/2)<-1e-6): counts['input_fov'] += 1; continue
        if any(intersect(box,other) for other in fixed): counts['observed_aabb_overlap'] += 1; continue
        counts['accepted'] += 1
        cost = abs(gap-obj['gap_m']) + .05*np.linalg.norm(center[[0,2]]-c[[0,2]]) + rng.uniform(0,1e-9)
        options.append(dict(box,score=float(cost),sample_angle_about_anchor_deg=angle,
            sampled_ray_gap_m=float(gap),gap_deviation_m=float(abs(gap-obj['gap_m'])),
            position_domain='anchor_camera_aabb_top_proxy' if atop else 'object_relative_xz_rays',
            anchor_id=obj['anchor_id'],height_anchor_id=obj.get('height_anchor_id') if not atop else None,
            relation=obj['relation']))
    return sorted(options,key=lambda p:p['score'])


class NoLayout(ValueError):
    def __init__(self,message,report):
        super().__init__(message)
        self.report = report


def solve(spec, observed, hfov, image_size, seed=3407, angle_step=15, gap_slack=1., beam_width=32):
    fixed = [dict(id=o['track_id'],center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value'])
             for o in observed['objects']]
    by_fixed = {o['id']:o for o in fixed}
    by_fixed[CAMERA_ID] = dict(id=CAMERA_ID,center=[0.,0.,0.],size=[0.,0.,0.])
    beam = [(0., [])]; stages = []; cache = {}
    rng = np.random.default_rng(seed)
    for obj in spec['additions']:
        expanded = []; tested = 0; rejections = []
        for score, placed in beam:
            boxes = {**by_fixed, **{p['id']:p for p in placed}}
            reference = boxes[obj['anchor_id']]
            cache_key = (obj['id'],tuple(reference['center']))
            if cache_key not in cache:
                counts = {}
                height_ref = None if obj['relation']=='on_top' else by_fixed[obj['height_anchor_id']]
                cache[cache_key] = candidates(obj,reference,height_ref,fixed,
                    hfov,image_size,angle_step,gap_slack,rng,counts)
                rejections.append(dict(anchor_center=reference['center'],counts=counts))
            for option in cache[cache_key]:
                tested += 1
                if not any(intersect(option,old) for old in placed):
                    expanded.append((score+option['score'],placed+[option]))
        stages.append(dict(id=obj['id'],tested_candidates=tested,feasible_extensions=len(expanded),candidate_pools=rejections))
        beam = sorted(expanded,key=lambda item:item[0])[:beam_width]
        if not beam:
            raise NoLayout(f'{obj["id"]}: no joint placement in finite search; specification preserved, no object deleted. '
                           'This does not prove that no real room layout exists.',
                           dict(status='no_layout_in_search',failed_object=obj['id'],stages=stages,
                                angle_step_deg=angle_step,gap_slack_m=gap_slack,beam_width=beam_width))
    selected = beam[0][1]
    report = dict(protocol=PROTOCOL,selected=selected,stages=stages,angle_step_deg=angle_step,
        gap_slack_m=gap_slack,beam_width=beam_width,observed_state_preserved=True,
        floor_verified=False,gravity_verified=False,semantics_verified=False,geometry_verified=False,
        constraints_satisfied_by_construction=True,
        limitations='Finite camera-axis AABB search, not room reconstruction, mesh collision or physical support. '
                    'Camera-y alignment is a prior, not measured gravity. No in-view occlusion completion.')
    return selected, report


def revision_prompt(case, spec, failure):
    stage = failure.report['stages'][-1]
    counts = {k:sum(p['counts'].get(k,0) for p in stage['candidate_pools'])
              for k in ('sampled','relation','input_fov','observed_aabb_overlap','support_footprint','accepted')}
    feedback = dict(failed_object=failure.report['failed_object'],candidate_counts=counts,
                    tested_joint_candidates=stage['tested_candidates'],error=str(failure))
    # This is a different operation from proposing a scene. Repeating the initial
    # generation prompt asks the model both to choose content and not to choose it.
    # Keep the object registry in Python and ask only for changes keyed by its IDs.
    template = dict(placements={},withdrawals={})
    context = [{**o, 'center':[round(v,3) for v in o['center']],
                'full_aabb_size':[round(v,3) for v in o['full_aabb_size']]} for o in case.get('context',[])]
    return ('Review ONE proposed extension of the attached partial RGB photograph. '
        'The fixed observations are visible evidence and must never change. The new objects '
        'are hypotheses, not observations. The '
        'CPU found no joint layout for its declared placements in the finite candidate domain. '
        'Return ONLY JSON with two dictionaries: placements and withdrawals. '
        'This is a SPARSE patch: omit unchanged IDs; Python retains those objects and placements. '
        'placements maps an existing new ID to CHANGED placement fields only. An empty placements '
        'dictionary changes no placements. Only IDs explicitly named in withdrawals are removed. '
        'Do not renumber IDs or put the same ID in both dictionaries. '
        'Allowed patch fields are anchor_id, relation, gap_m, height_anchor_id, height_alignment, '
        'height_offset_m, reason. Names, categories and sizes are supplied by Python: do NOT '
        'repeat or change them and do NOT return a full additions list. No numeric centres/offsets. '
        'Normally retain the proposed objects and repair their references or relationships. '
        'If the image and room function make a hypothesized object redundant or inappropriate, '
        'withdrawals may map that original ID to an explicit semantic reason. A finite search '
        'failure alone is NOT evidence that an object is inappropriate. Do not withdraw merely '
        'to clear a geometric test or reduce the count. Withdrawals are retained in a separate '
        'review record, not counted as successful placements. No replacement or extra object '
        'is requested in this placement repair. A retained object must not reference a withdrawn one. '
        'Use the original image to choose coherent positions. '
        'A reference to a visible fixture does not make the new object offscreen. If a prescribed '
        'side/top placement cannot be offscreen, consider another meaningful observed or PLANNED '
        'reference, or a separate functional group rooted at __camera__ with relation=near. '
        'Do not merely increase all distances, duplicate an object already visible, or claim '
        'occlusion: this stage places offscreen objects. Respect room function, not only tests. '
        'Axes are camera-local metres: x right, y IMAGE-UP, z forward; floor, gravity and walls '
        'are unknown. Relations: near, left_of, right_of, in_front_of (smaller camera z), behind '
        '(larger camera z), on_top. Left/right and front/behind order full AABB edges, not facing '
        'directions. gap_m>=0 is preferred ray separation; horizontal search adds bounded slack. '
        'on_top searches the parent camera-AABB footprint; it is not restricted to its centre, '
        'and puts the new bottom at its camera-y top plus EXACT gap_m (0 for contact), ignoring '
        'height_* fields. This is a box-top prior, not a measured surface. Other relations need '
        'height_anchor_id referencing an observed object, height_alignment=bottom/center/top '
        '(same box edge/center) and signed height_offset_m. For floor-standing furniture align '
        'bottoms with a relevant visible floor-standing object, not centers or wall decorations; '
        'this is only a prior, not gravity calibration. '
        'Referencing another retained planned object must not create a cycle. '
        'The CPU searches over 360 degrees, requiring the whole new AABB outside the input '
        'photograph and no positive-volume AABB overlaps. There are no direction quotas. '
        'Counts describe a bounded search failure, not proof a real room is impossible. '
        f'Input horizontal FOV={case["hfov"]:g} degrees; final RGB target is a 2:1 ERP panorama. '
        +'\nREPLY TEMPLATE (include only changes; withdrawal needs a semantic reason): '
        +json.dumps(template,separators=(',',':'))
        +'\nFIXED OBSERVED: '+json.dumps(context,ensure_ascii=False,separators=(',',':'))
        +'\nPREVIOUS VALIDATED SPECIFICATION: '+json.dumps(spec,ensure_ascii=False,separators=(',',':'))
        +'\nCOMPUTED FAILURE FEEDBACK: '+json.dumps(feedback,separators=(',',':')))


def validate_revision(answer, initial, observed, capacity):
    """Read older full-spec replies without silently reconciling renamed IDs."""
    revised = validate_spec(answer,observed,capacity)
    identity = lambda s: {o['id']:{k:o[k] for k in ('name','category','size')} for o in s['additions']}
    before,after = identity(initial),identity(revised)
    if before!=after:
        delta = dict(missing_ids=sorted(before.keys()-after.keys()),new_ids=sorted(after.keys()-before.keys()),
            changed_identity={oid:[k for k in before[oid] if before[oid][k]!=after[oid][k]]
                              for oid in before.keys() & after.keys() if before[oid]!=after[oid]})
        raise ValueError('Placement revision changed the fixed object list, category or size; '
                         'raw reply retained. '+json.dumps(delta,sort_keys=True))
    return revised


def apply_revision(answer, initial, observed, capacity):
    """Merge an ID-keyed placement patch; every content withdrawal is explicit.

    Omitted IDs/fields are unchanged, as in a sparse update, NOT deleted. Only
    an explicit withdrawal removes a new hypothesis. Observations cannot be
    patched. A withdrawal remains an unverified semantic decision, never a
    successful geometric repair.
    """
    if not isinstance(answer,dict) or 'placements' not in answer:
        # Replay compatibility for already accepted v2 full-spec replies.
        return validate_revision(answer,initial,observed,capacity)
    if set(answer)!= {'placements','withdrawals'} or not all(isinstance(answer[k],dict) for k in answer):
        raise ValueError('Revision needs only placements and withdrawals dictionaries')
    patches, withdrawals = answer['placements'],answer['withdrawals']
    ids = {o['id'] for o in initial['additions']}
    if set(patches)&set(withdrawals) or not (set(patches)|set(withdrawals))<=ids:
        raise ValueError('Patch IDs must be existing new hypotheses, with no overlap between placements '
                         'and withdrawals; no renamed, extra or observed IDs')
    for oid,patch in patches.items():
        if not isinstance(patch,dict) or not set(patch)<=PLACEMENT_FIELDS:
            raise ValueError(f'{oid}: patch only placement fields; identity/category/size are Python-owned')
    if any(not isinstance(reason,str) or not reason.strip() for reason in withdrawals.values()):
        raise ValueError('Each withdrawn hypothesis needs an explicit semantic reason')
    revised = deepcopy(initial)
    revised['additions'] = [dict(obj,**deepcopy(patches.get(obj['id'],{})))
                           for obj in initial['additions'] if obj['id'] not in withdrawals]
    retained_withdrawals = deepcopy(initial.get('withdrawn_hypotheses',[]))
    retained_withdrawals += [dict(hypothesis=deepcopy(obj),withdrawal_reason=withdrawals[obj['id']],
                                 semantics_verified=False)
                            for obj in initial['additions'] if obj['id'] in withdrawals]
    revised['withdrawn_hypotheses'] = retained_withdrawals
    for obj in revised['additions']:
        if 'category_review' in obj:
            obj['category'] = obj.pop('category_review')['model_category']
    return validate_spec(revised,observed,capacity)


def saved_spec(source, case, dest, initial_only=False):
    previous = source/case['id']; old_input = read(previous/'input.json')
    # Compiler/prompt improvements must not force regeneration of an unchanged
    # RGB-conditioned semantic specification. Bind the actual old request too.
    evidence = lambda c: {k:v for k,v in c.items() if k!='prompt'}
    if evidence(old_input)!=evidence(case): raise ValueError('Saved specification uses different image/state/context')
    if read(previous/'G_obs.json')!=read(case['state']):
        raise ValueError('Saved observed state differs from current G_obs')
    path = previous/'relative_scene.json'
    extra_paths = []
    if (not initial_only and (previous/'revision_report.json').exists()
            and read(previous/'revision_report.json').get('status')=='accepted'):
        # New replies are patches, not complete scene specifications. Reuse the
        # exact materialized result; keep both raw proposal and patch provenance.
        path = previous/'spec_validated.json'
        extra_paths = [previous/p for p in ('relative_scene.json','placement_revision.json','revision_report.json')]
    write(dest/'source_input.json',old_input)
    paths = [previous/'input.json',path,*extra_paths]
    request = (previous/'placement_revision.request.json') if extra_paths else path.with_suffix('.request.json')
    if request.exists():
        paths.append(request)
        write(dest/'source_request.json',read(request))
    write(dest/'replay_source.json',{str(p.resolve()):sha(p) for p in paths})
    answer = read(path)
    if extra_paths:
        # Restore the original model labels before validation, so canonicalizing
        # a replay does not erase alias/out-of-vocabulary review provenance.
        for obj in answer['additions']:
            if 'category_review' in obj:
                obj['category'] = obj.pop('category_review')['model_category']
    return answer


def saved_revision(source, case, initial, dest):
    """Read a cached reply explicitly, bound to the same evidence and proposal.

    saved_spec already checked the image/context and observed state. This also
    checks the proposal embedded in the actual request and the raw reply, so a
    rejected patch can be recompiled without another model call or hand editing.
    """
    previous = source/case['id']
    paths = [previous/name for name in ('spec_initial_validated.json',
        'placement_revision.json','placement_revision.txt','placement_revision.request.json')]
    hashes = {str(p.resolve()):sha(p) for p in paths}
    if read(paths[0])!=initial:
        raise ValueError('Cached revision belongs to a different initial hypothesis registry')
    request = read(paths[3]); text = request.get('prompt','')
    marker = '\nPREVIOUS VALIDATED SPECIFICATION: '
    if marker not in text:
        raise ValueError('Cached revision request does not identify its original specification')
    requested_spec = text.split(marker,1)[1].split('\nCOMPUTED FAILURE FEEDBACK:',1)[0]
    if json.loads(requested_spec)!=initial:
        raise ValueError('Cached revision request and initial specification differ')
    answer = read(paths[1])
    if parse_json(paths[2].read_text())!=answer:
        raise ValueError('Cached parsed revision differs from its raw model response')
    (dest/'placement_revision.txt').write_text(paths[2].read_text())
    write(dest/'placement_revision.request.json',request)
    generation = previous/'placement_revision.generation.json'
    if generation.exists():
        hashes[str(generation.resolve())] = sha(generation)
        write(dest/generation.name,read(generation))
    write(dest/'cached_revision_source.json',hashes)
    if any(sha(Path(p))!=h for p,h in hashes.items()):
        raise ValueError('Cached revision changed while being read')
    return answer


def export(observed, spec, selected):
    by_id = {o['id']:o for o in spec['additions']}
    plan = dict(scene_hypothesis=spec['scene_hypothesis'],additions=[dict(id=o['id'],name=o['name'],
        reason=o['reason'],placement=f'{o["relation"]} {o["anchor_id"]}') for o in spec['additions']])
    items = [dict(id=p['id'],plan_id=p['id'],category=by_id[p['id']]['category'],center=p['center'],
        size=p['size'],unseen_reason='outside_fov',reason=by_id[p['id']]['reason']) for p in selected]
    all_candidates, full, unmapped = build_states(observed,plan,items)
    for graph in (all_candidates,full):
        for obj in graph['objects'][len(observed['objects']):]:
            obj['center_local_m'].update(source='cpu_relation_search',method=PROTOCOL)
            obj['size_aabb_local_m'].update(source='qwen3vl_size_hypothesis',method='full_camera_axis_aabb')
            obj['category_id'].update(source='qwen3vl_category_hypothesis' if obj['category_id']['valid'] else 'unknown')
            obj['placement_specification'] = deepcopy(by_id[obj['track_id']])
    unmapped = [deepcopy(o) for o in all_candidates['objects'][len(observed['objects']):] if not o['category_id']['valid']]
    return all_candidates, full, unmapped


def run(model, cases, output, capacity=6, seed=3407, angle_step=15, gap_slack=1., replay=None,
        apply_saved_revision=False):
    if apply_saved_revision and (not replay or model is not None):
        raise ValueError('Applying a saved revision requires a CPU-only replay source')
    records = []
    for case in cases:
        dest = output/case['id']; dest.mkdir()
        with Image.open(case['image']) as handle: image = handle.convert('RGB')
        image.thumbnail((768,768)); image.save(dest/'local.png')
        observed = read(case['state']); write(dest/'G_obs.json',observed); write(dest/'input.json',case)
        record = dict(id=case['id'],status='failed',observed=len(observed['objects']),
                      semantics_verified=False,geometry_verified=False,panorama_generated=False,
                      raw_spec_reused=bool(replay),placement_revision_used=False,placement_revision_reused=False)
        try:
            if replay:
                answer = saved_spec(replay,case,dest,initial_only=apply_saved_revision)
            else:
                answer = model.ask(image,case['prompt'],dest/'relative_scene.txt',seed=seed,
                    sample=True,profile='official',max_tokens=4096,guard_repetition=True)
            write(dest/'relative_scene.json',answer)
            spec = validate_spec(answer,observed,min(capacity,160-len(observed['objects'])))
            write(dest/'spec_validated.json',spec)
            record['planned'] = len(spec['additions'])
            record['planned_before_revision'] = len(spec['additions'])+len(spec.get('withdrawn_hypotheses',[]))
            failure = None
            if not apply_saved_revision:
                try:
                    selected, report = solve(spec,observed,case['hfov'],image.size,seed,angle_step,gap_slack)
                except NoLayout as exc:
                    failure = exc
                    write(dest/'initial_search_report.json',failure.report)
                    if model is None: raise
            if apply_saved_revision or failure is not None:
                record['placement_revision_used'] = True
                record['placement_revision_reused'] = apply_saved_revision
                write(dest/'spec_initial_validated.json',spec)
                revision = dict(status='requested',protocol=REVISION_PROTOCOL,
                                original_hypothesis_ids=[o['id'] for o in spec['additions']])
                write(dest/'revision_report.json',revision)
                try:
                    if apply_saved_revision:
                        corrected = saved_revision(replay,case,spec,dest)
                    else:
                        corrected = model.ask(image,revision_prompt(case,spec,failure),dest/'placement_revision.txt',
                            seed=seed,sample=True,profile='official',max_tokens=4096,guard_repetition=True)
                    write(dest/'placement_revision.json',corrected)
                    revised = apply_revision(corrected,spec,observed,capacity)
                    before = {o['id']:o for o in spec['additions']}
                    revision['changed_fields'] = {o['id']:[k for k in o if before[o['id']].get(k)!=o[k]]
                                                  for o in revised['additions']}
                    revision['withdrawn_hypotheses'] = revised.get('withdrawn_hypotheses',[])
                    selected, report = solve(revised,observed,case['hfov'],image.size,seed,angle_step,gap_slack)
                    spec = revised
                    write(dest/'spec_validated.json',spec)
                    revision['status'] = 'accepted'
                except Exception as exc:
                    revision.update(status='failed',error=str(exc))
                    write(dest/'revision_report.json',revision)
                    raise
                write(dest/'revision_report.json',revision)
            write(dest/'search_report.json',report)
            withdrawn = spec.get('withdrawn_hypotheses',[])
            write(dest/'withdrawn_hypotheses.json',withdrawn)
            candidates_state,full,unmapped = export(observed,spec,selected)
            for name,value in [('G_full_candidates.json',candidates_state),('G_full.json',full),('unmapped_candidates.json',unmapped)]:
                write(dest/name,value)
            checks = diagnostics([dict(id=p['id'],center=p['center'],size=p['size'],category=None,unseen_reason='outside_fov')
                                  for p in selected],case['hfov'],image.size)
            write(dest/'geometry_review.json',checks)
            adapter_ids = {o['track_id'] for o in full['objects'][len(observed['objects']):]}
            record.update(status='needs_content_review' if withdrawn else 'needs_category_review' if unmapped else 'requires_visual_review',
                planned=len(spec['additions']),hypothesized=len(selected)-len(unmapped),unmapped=len(unmapped),
                hypotheses_withdrawn=len(withdrawn),
                withdrawal_reasons={o['hypothesis']['id']:o['withdrawal_reason'] for o in withdrawn},
                full_objects=len(full['objects']),outside_fov_certified=sum(c['outside_input_fov_certified'] for c in checks),
                outside_fov_adapter_objects=sum(c['outside_input_fov_certified'] for c in checks if c['id'] in adapter_ids),
                observed_state_preserved=full['objects'][:len(observed['objects'])]==observed['objects'],
                completion_outcome='objects_added' if len(full['objects'])>len(observed['objects']) else 'no_adapter_additions',
                constraints_satisfied_by_construction=True)
            render(candidates_state,image,case['hfov'],dest,record,
                position_note='新增物体与尺寸由 Qwen3-VL-8B 提出；位置由参照对象、相对关系及 CPU 有限搜索计算，不是 WildDet3D 对未知区域的测量。')
        except Exception as exc:
            if isinstance(exc,NoLayout): write(dest/'search_report.json',exc.report)
            record.update(error=str(exc),status='no_layout_in_search' if isinstance(exc,NoLayout) else 'failed')
        records.append(record); write(dest/'summary.json',record)
        write(output/'summary.json',dict(protocol=PROTOCOL,records=records,scene_gt_used=False))
        cards = ''.join('<p>'+escape(r['id'])+': '+escape(r['status'])+' '
            +('<a href="'+r['id']+'/index.html">蓝色观测／橙色补全／红色未映射候选</a>'
              if (output/r['id']/'index.html').exists() else escape(r.get('error','')))+'</p>' for r in records)
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>8B 语义与 CPU 相对布局</h1>'
            '<p>位置由CPU搜索产生；视野外与AABB检查通过是构造结果，不证明场景语义正确。图中y是图像上方向，未标定地板与重力。</p>'+cards)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    if any(sha(Path(p))!=h for c in cases for p,h in c['source_sha256'].items()):
        raise ValueError('Source image/observed state changed during run')
    for case in cases:
        bound = output/case['id']/'cached_revision_source.json'
        if bound.exists() and any(sha(Path(p))!=h for p,h in read(bound).items()):
            raise ValueError('Source cached revision changed during run')
    return all(r['status'] not in ('failed','no_layout_in_search') for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--only-id',action='append')
    parser.add_argument('--max-added',type=int,default=6)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--angle-step',type=float,default=15.)
    parser.add_argument('--gap-slack',type=float,default=1.)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--apply-saved-revision',action='store_true',
        help='With --replay: CPU-only apply the saved raw correction, including a previously rejected patch')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--replay',type=Path,help='CPU-only re-export of saved specification into a new output')
    source.add_argument('--continue-from',type=Path,help='Reuse saved specifications; use 8B only for one failed-placement revision per case')
    args = parser.parse_args()
    if args.apply_saved_revision and not args.replay:
        parser.error('--apply-saved-revision requires --replay; no model is loaded')
    if not 0<=args.max_added<=32 or not 0<=args.seed<2**63 or not 1<=args.angle_step<=90 or not 0<=args.gap_slack<=5:
        parser.error('Invalid finite search settings')
    cases = load_cases(args.batch.resolve(),args.only_id)
    for case in cases:
        if len(case['context'])>160: parser.error('Observed state exceeds Adapter capacity')
        if any(o[k] is None for o in case['context'] for k in ('center','full_aabb_size')):
            parser.error('Need complete fixed observed geometry')
        case['prompt'] = prompt(case,min(args.max_added,160-len(case['context'])))
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',cases=len(cases),model_loaded=False,
            observed_source='fixed_predicted_G_obs',new_requests_per_case_max=0 if args.replay else 1 if args.continue_from else 2,
            position_source='object_relative_cpu_search',scene_gt_used=False)))
        return
    output = new_output(args.output)
    write(output/'config.json',dict(protocol=PROTOCOL,inputs=cases,model=str(args.model.resolve()),
        revision_protocol=REVISION_PROTOCOL,
        seed=args.seed,max_added=args.max_added,angle_step=args.angle_step,gap_slack=args.gap_slack,
        replay=str(args.replay.resolve()) if args.replay else None,
        continue_from=str(args.continue_from.resolve()) if args.continue_from else None,
        apply_saved_revision=args.apply_saved_revision,
        code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('qwen_upstream.py'),
            Path(__file__).with_name('semantic_layout.py'),Path(__file__).with_name('semantic_layout_visual.py'),
            Path(__file__).with_name('placement_program.py'),Path(__file__).with_name('category_bridge.py'))}))
    model = None if args.replay else Qwen(args.model)
    if not run(model,cases,output,args.max_added,args.seed,args.angle_step,args.gap_slack,
               args.replay or args.continue_from,args.apply_saved_revision):
        raise SystemExit('Some cases need review; raw replies retained, no forced placement or silent object deletion')


if __name__=='__main__': main()
