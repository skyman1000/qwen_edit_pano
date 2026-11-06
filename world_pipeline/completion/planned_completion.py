"""Image-conditioned room contents -> discrete layout choices -> coarse state.

No assets, learned weights or room reconstruction. The search region is an
explicit prior, not recovered architecture. Geometry does not delete objects.
"""
from copy import deepcopy
import json
import math

import numpy as np

from ..common import finite_vector, write
from .box_layout import frustum_margin
from .geometry import CANONICAL_CATEGORIES
from .scene_geometry import camera_geometry, ground_center, half_extent
from .scene_relations import TYPES, validate_relations, ordered_objects, supported_pair, check_relation, relation_report

CONTENT_PROMPT_VERSION = 'partial_view_room_hypothesis_v5'


def content_prompt(observed, capacity, hfov=90):
    return (
        'Generate ONE plausible COMPLETE ROOM hypothesis from this PARTIAL photograph. '
        f'The camera sees only {hfov:g} degrees horizontally; other directions are UNOBSERVED, not empty. '
        'The supplied inventory is only the visible subset of the room, not its full object list. '
        'Use the room purpose, visible furnishings and style to infer suitable OFFSCREEN contents. '
        'You may hypothesize objects without direct visual evidence: this is plausible scene generation, '
        'not a claim to recover the unique true hidden room. '
        'Consider what functions the unobserved part could serve beyond those already visible. '
        'Choose a restrained, coherent arrangement; do not merely caption the photograph. '
        'Keep all observed IDs unchanged. List only NEW instances in additions, '
        'with a scene-specific purpose and a placement intent in the SAME room outside the photo. '
        'Do not duplicate existing instances or remodel the visible region. '
        'Only structured object IDs commit an object to the scene; prose cannot add unlisted furniture. '
        'Return JSON {"room_type":"...","scene_hypothesis":"brief full-room purpose/layout hypothesis", '
        '"additions":[{"id":"new_01",'
        '"category":"allowed category","description":"object appearance/function",'
        '"reason":"scene-specific purpose","placement_intent":"desired relation to visible objects '
        'or other proposed IDs"}], "relations":[]}. '
        'Optional key relationships use {"subject":"new_ID","type":"near","target":"existing_or_new_ID",'
        f'"max_gap_m":1.0}}. Supported types: {", ".join(TYPES)}. '
        'Relations must reference existing or proposed IDs; directions use camera axes. '
        'Resting on another object requires an on relation to its ID. face_to is an unverified intent. '
        'This stage plans contents ONLY; geometry is estimated later. '
        f'At most {capacity} additions, not a target count. '
        'If additions is empty, explain in scene_hypothesis why your full-room hypothesis needs no '
        'offscreen contents. Lack of visual evidence for offscreen objects is not itself a reason to omit them. '
        'Allowed categories: '+', '.join(CANONICAL_CATEGORIES.values())
        +'\nVisible inventory: '+json.dumps(observed, ensure_ascii=False))


def validate_contents(answer, observed, capacity):
    if not isinstance(answer, dict) or any(not isinstance(answer.get(k), str) or not answer[k].strip()
                                          for k in ('room_type', 'scene_hypothesis')):
        raise ValueError('Contents need room_type and scene_hypothesis')
    items = answer.get('additions')
    if not isinstance(items, list) or len(items) > capacity:
        raise ValueError('Contents additions exceed capacity or are not a list')
    ids = {o['id'] for o in observed}
    for obj in items:
        if not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) or not obj[k].strip()
                for k in ('id', 'category', 'description', 'reason', 'placement_intent')):
            raise ValueError('Each planned object needs ID/category/description/reason/placement_intent')
        if obj['id'] in ids or obj['category'] not in CANONICAL_CATEGORIES.values():
            raise ValueError('Planned IDs must be unique and categories must use the vocabulary')
        ids.add(obj['id'])

    validate_relations(answer, observed)
    return deepcopy(answer)


def validate_geometry(obj):
    oid = obj.get('id', '<missing ID>')
    missing = [key for key in ('size', 'bottom_height_m') if key not in obj]
    if missing:
        raise ValueError(f'{oid}: missing geometry fields: {missing}')
    if not finite_vector(obj['size'], 3) or min(obj['size']) <= 0:
        raise ValueError(f'{oid}: size must contain 3 finite positive full camera-axis lengths; got {obj["size"]!r}')
    bottom = obj['bottom_height_m']
    if type(bottom) not in (int, float) or not math.isfinite(bottom) or bottom < 0:
        raise ValueError(f'{oid}: bottom_height_m must be finite and nonnegative; got {bottom!r}')


def complete_geometry(contents, observed_layout, camera_height, ask, geometry=None):
    """Estimate absent numeric fields in one batch; never invent default sizes.

    Existing valid values are retained. A malformed value is not treated as a
    missing field. Semantic records cannot be rewritten by this numeric stage.
    """
    result = deepcopy(contents)
    support_ids = {r['target'] for r in contents.get('relations', []) if r['type']=='on'}
    pending = []
    for obj in result['additions']:
        if 'size' in obj and (not finite_vector(obj['size'], 3) or min(obj['size']) <= 0):
            raise ValueError(f'{obj["id"]}: invalid provided size: {obj["size"]!r}')
        bottom = obj.get('bottom_height_m')
        if 'bottom_height_m' in obj and (type(bottom) not in (int, float) or not math.isfinite(bottom) or bottom < 0):
            raise ValueError(f'{obj["id"]}: invalid provided bottom_height_m: {bottom!r}')
        if 'size' not in obj or 'bottom_height_m' not in obj:
            pending.append(obj)
    if pending or support_ids:
        prompt = ('Estimate approximate geometry for these FIXED planned objects using the original photo '
                  'and observed scale context. This is a plausible hypothesis, not metric reconstruction. '
                  'Return ONLY JSON {"objects":[{"id":"...","size":[0.6,0.8,0.6],"bottom_height_m":0}]}. '
                  'Include each requested ID exactly once. size is FULL camera-axis x/y/z lengths in metres '
                  '(x right,y up,z forward), not intrinsic width/height/thickness or half lengths. '
                  'bottom_height_m is the object bottom above the floor; 0 for floor-standing. '
                  'The example numbers are not required sizes. No categories, descriptions, centers, '
                  'new objects or deletions. Preserve numeric fields already provided. '
                  'If SUPPORT TARGET IDs are listed, also return top-level support_surfaces: '
                  '[{"id":"target_ID","offset_m":0.05}]. offset_m is the actual supporting '
                  'surface displacement along the working up direction from the target box CENTER '
                  '(a seat/tabletop, not automatically the whole AABB top/headboard). '
                  'These offsets are estimates, not measured surfaces. '
                  f'Assumed camera height={camera_height}m. '
                  '\nWORKING CAMERA/GRAVITY PRIORS: '+json.dumps(geometry or camera_geometry(camera_height=camera_height))
                  +'\nREQUESTED OBJECTS: '+json.dumps(pending, ensure_ascii=False)
                  +'\nSUPPORT TARGET IDs: '+json.dumps(sorted(support_ids))
                  +'\nALL PLANNED OBJECTS: '+json.dumps(contents['additions'],ensure_ascii=False)
                  +'\nOBSERVED COARSE BOXES: '+json.dumps(observed_layout['objects']))
        reply = ask(prompt, 'room_geometry', 2048)
        items = reply if isinstance(reply, list) else reply.get('objects') if isinstance(reply, dict) else None
        if not isinstance(items, list) or any(not isinstance(o, dict) or not isinstance(o.get('id'), str) for o in items):
            raise ValueError('room_geometry requires an objects list with string IDs')
        if len(items) != len(pending) or {o['id'] for o in items} != {o['id'] for o in pending}:
            raise ValueError('room_geometry must return every requested ID exactly once')
        by_id = {o['id']: o for o in items}
        for obj in pending:
            numeric = by_id[obj['id']]
            validate_geometry(numeric)
            for key in ('size', 'bottom_height_m'):
                if key in obj and obj[key] != numeric[key]:
                    raise ValueError(f'{obj["id"]}: room_geometry attempted to change provided {key}')
                obj[key] = deepcopy(numeric[key])
        if support_ids:
            surfaces = reply.get('support_surfaces') if isinstance(reply, dict) else None
            if not isinstance(surfaces,list) or any(not isinstance(s,dict) or not isinstance(s.get('id'),str) for s in surfaces):
                raise ValueError('on relations require estimated support_surfaces with ID and offset_m')
            if len(surfaces)!=len(support_ids) or {s['id'] for s in surfaces}!=support_ids:
                raise ValueError('Support surfaces must match the referenced target IDs exactly')
            if any(type(s.get('offset_m')) not in (int,float) or not math.isfinite(s['offset_m']) for s in surfaces):
                raise ValueError('Support surface offset_m must be finite')
            result['support_surfaces']={s['id']:s['offset_m'] for s in surfaces}
    for obj in result['additions']:
        validate_geometry(obj)
    return result


def overlaps(a, b):
    # Conservative box overlap, not mesh collision/support certification.
    return bool(np.all((np.array(a['size'])+b['size'])/2
                       - np.abs(np.array(a['center'])-b['center']) > 1e-6))


def candidate_positions(obj, observed_layout, hfov, image_size, camera_height, geometry=None,
                        parent_id=None, parent_options=None, support_offset=None):
    tan_h = math.tan(math.radians(hfov)/2)
    tan_v = tan_h*image_size[1]/image_size[0]
    options = []
    geometry = camera_geometry(camera_height=camera_height) if geometry is None else geometry
    if parent_id is not None:
        up = np.array(geometry['up_local'])
        for parent in parent_options:
            center=(np.array(parent['center'])+up*(support_offset+half_extent(obj['size'],up))).tolist()
            box=dict(center=center,size=obj['size'])
            if frustum_margin(np.array(center),np.array(obj['size']),tan_h,tan_v)>=-1e-6:
                continue
            if any(old['id']!=parent_id and overlaps(box,old) for old in observed_layout['objects']):
                continue
            options.append(dict(candidate_id=f'p{len(options):02d}',center=center,
                                parent_id=parent_id,parent_candidate_id=parent['candidate_id']))
        return options
    # A compact explicit search prior. No expansion to an unbounded room.
    for radius in (1.5, 2.5, 3.5, 4.5):
        for angle in range(-180, 180, 30):
            center = ground_center(obj['size'], obj['bottom_height_m'], radius, angle, geometry)
            box = dict(center=center, size=obj['size'])
            if frustum_margin(np.array(center), np.array(obj['size']), tan_h, tan_v) >= -1e-6:
                continue
            if any(overlaps(box, old) for old in observed_layout['objects']):
                continue
            options.append(dict(candidate_id=f'p{len(options):02d}', center=center,
                                azimuth_deg=angle, distance_m=radius))
    return options


def select_layout(contents, candidates, reply, beam_width=256, observed_layout=None, geometry=None):
    """Choose a non-overlapping combination from the model's ranked alternatives.

    Never invent a choice, drop an object, or move observed boxes. Bounded beam
    search can miss a solution; failure is not proof the room is impossible.
    """
    choices = reply.get('placements') if isinstance(reply, dict) else None
    ids = [o['id'] for o in contents['additions']]
    if not isinstance(choices, list) or any(not isinstance(o, dict) or not isinstance(o.get('id'), str) for o in choices):
        raise ValueError('Placement response needs a placements list')
    if len(choices) != len(ids) or {o['id'] for o in choices} != set(ids):
        raise ValueError('Placement must include every planned ID exactly once; no deletion')
    by_id = {o['id']: o for o in choices}
    beam = [((0,0), [], [])]
    geometry = camera_geometry() if geometry is None else geometry
    relations=contents.get('relations',[])
    observed_boxes={o['id']:dict(o) for o in (observed_layout or {}).get('objects',[])}
    for oid,offset in contents.get('support_surfaces',{}).items():
        if oid in observed_boxes:
            observed_boxes[oid]['support_surface_offset_m']=offset
    for obj in ordered_objects(contents):
        choice = by_id[obj['id']]
        preferences = choice.get('candidate_ids')
        available = {o['candidate_id']: o for o in candidates[obj['id']]}
        if not isinstance(preferences, list) or not 1 <= len(preferences) <= 5 or any(
                not isinstance(p, str) or p not in available for p in preferences) or len(set(preferences)) != len(preferences):
            raise ValueError('Choose 1..5 distinct supplied candidate IDs per planned object')
        expanded = []
        for score, placed, selected in beam:
            for rank, cid in enumerate(preferences):
                option = available[cid]
                box = dict(id=obj['id'], center=option['center'], size=obj['size'])
                if obj['id'] in contents.get('support_surfaces',{}):
                    box['support_surface_offset_m']=contents['support_surfaces'][obj['id']]
                if option.get('parent_id') and option['parent_id'] not in observed_boxes:
                    parent = next((p for p in selected if p['id']==option['parent_id']),None)
                    if parent is None or parent['candidate_id']!=option['parent_candidate_id']:
                        continue
                if any(overlaps(box, other) and not supported_pair(obj['id'],other['id'],relations) for other in placed):
                    continue
                by_id_boxes={**observed_boxes, **{p['id']:p for p in placed}, obj['id']:box}
                checks=[check_relation(r,by_id_boxes,geometry) for r in relations]
                if any(r['type']=='on' and r['status']=='violated' for r in checks):
                    continue
                # Soft spatial wishes cannot make all otherwise usable layouts
                # disappear. Prefer fewer violations, then better model ranks;
                # report remaining violations rather than claiming success.
                new_score=(sum(r['status']=='violated' for r in checks),score[1]+rank)
                expanded.append((new_score, placed+[box], selected+[dict(
                    candidate_id=cid, preference_rank=rank+1, **box)]))
        beam = sorted(expanded, key=lambda state: state[0])[:beam_width]
        if not beam:
            raise ValueError('No joint layout found among ranked choices; content plan preserved, no objects deleted')
    by_selected={p['id']:p for p in beam[0][2]}
    selected=[by_selected[oid] for oid in ids]
    answer = deepcopy(contents)
    for obj, position in zip(answer['additions'], selected):
        obj.update(center=position['center'], unseen_reason='outside_fov',
                   placement=f"selected candidate {position['candidate_id']}")
        if any(r['type']=='on' and r['subject']==obj['id'] for r in relations):
            up=np.array(geometry['up_local'])
            obj['requested_bottom_height_m']=obj['bottom_height_m']
            obj['bottom_height_m']=float(np.dot(up,obj['center'])+geometry['floor_plane_local'][3]
                                         -half_extent(obj['size'],up))
            obj['bottom_height_source']='derived_from_parent_support_hypothesis'
    return answer, dict(protocol='ranked_candidate_layout_v1', selected=selected,
                        planned_count=len(ids), placed_count=len(selected), beam_width=beam_width,
                        note='Ranked coarse layout, not wall/support/semantic certification')


def complete(ask, observed, observed_layout, hfov, image_size, camera_height, capacity, output, geometry=None):
    geometry = camera_geometry(camera_height=camera_height) if geometry is None else geometry
    contents = validate_contents(ask(content_prompt(observed, capacity, hfov), 'room_contents', 4096), observed, capacity)
    write(output/'room_contents_validated.json', contents)
    write(output/'object_registry.json',dict(
        observed=[dict(id=o['id'],category=o['category']) for o in observed],
        planned=[dict(id=o['id'],category=o['category']) for o in contents['additions']],
        commitment='Only these structured object IDs commit existence; prose is descriptive, not extra state',
        relations=contents.get('relations',[])))
    contents = complete_geometry(contents, observed_layout, camera_height, ask, geometry)
    write(output/'room_contents_with_geometry.json', contents)
    candidates = {}
    parents={r['subject']:r['target'] for r in contents.get('relations',[]) if r['type']=='on'}
    old_boxes={o['id']:o for o in observed_layout['objects']}
    for obj in ordered_objects(contents):
        parent=parents.get(obj['id'])
        parent_options = ([dict(candidate_id='observed',center=old_boxes[parent]['center'])]
                          if parent in old_boxes else candidates.get(parent))
        candidates[obj['id']]=candidate_positions(obj,observed_layout,hfov,image_size,camera_height,geometry,
                    parent_id=parent,parent_options=parent_options,
                    support_offset=contents.get('support_surfaces',{}).get(parent))
    write(output/'placement_candidates.json', dict(candidates=candidates,camera_geometry=geometry,
          search_prior='horizontal radii [1.5,2.5,3.5,4.5]m; 30-degree angular grid; not measured room',
          checks=['whole_AABB_outside_FOV', 'no_positive_volume_overlap_with_observed_AABBs']))
    if any(not options for options in candidates.values()):
        raise ValueError('A planned object has no candidates in the finite search region; content plan preserved')
    if not contents['additions']:
        reply = dict(placements=[])
    else:
        # Short candidate display; exact floats are retained in the saved artifact.
        display = {oid: [dict(p,center=[round(v, 3) for v in p['center']]) for p in options]
                   for oid, options in candidates.items()}
        context = [dict(**o, center=b['center'], size=b['size'])
                   for o, b in zip(observed, observed_layout['objects'])]
        prompt = ('Arrange the fixed room content plan into ONE coherent room around the original photo. '
                  'Choose from the supplied candidate positions for each new object, using the original image, '
                  'visible furniture, intended functions and relationships. These are possible positions, '
                  'not measured empty space or known room walls. Consider all additions together. '
                  'Prefer nearby functional arrangements and avoid unnecessary separation. '
                  'Every planned object must remain; do not delete, add, resize or rename objects. '
                  'All explicit ID relations are checked numerically; near/directional wishes are '
                  'soft preferences with remaining violations reported. Candidates of supported children '
                  'reference a particular parent candidate: select matching pairs. Parent and child '
                  'AABBs may overlap; this is not a collision rejection for an on relation. '
                  'For each ID rank up to 5 ACCEPTABLE candidate IDs from best to worst, '
                  'allowing the CPU to choose a non-overlapping combination. Do not invent candidate IDs. '
                  'Return ONLY JSON {"placements":[{"id":"new_01","candidate_ids":["p00","p01"],'
                  '"reason":"why these positions suit the room"}]}. No coordinates or revised objects. '
                  '\nROOM CONTENT PLAN: '+json.dumps(contents, ensure_ascii=False)
                  +'\nFIXED OBSERVED STATE: '+json.dumps(context, ensure_ascii=False)
                  +'\nCANDIDATE POSITIONS (camera x right,y up,z forward; metres): '+json.dumps(display))
        reply = ask(prompt, 'room_placement', 4096)
    answer, report = select_layout(contents, candidates, reply, observed_layout=observed_layout,geometry=geometry)
    boxes={o['id']:dict(o) for o in observed_layout['objects']}
    boxes.update({o['id']:dict(o) for o in report['selected']})
    for oid,offset in contents.get('support_surfaces',{}).items():
        boxes[oid]['support_surface_offset_m']=offset
    relation_diagnostic=relation_report(contents,boxes,geometry)
    write(output/'relation_diagnostic.json',relation_diagnostic)
    report['relation_diagnostic']=relation_diagnostic
    write(output/'placement_selection.json', report)
    return answer, dict(report, protocol='content_geometry_id_relations_camera_v3', legacy_region_ids=[])
