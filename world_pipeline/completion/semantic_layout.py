"""Saved scene contents -> Qwen coarse numeric boxes -> colored full-state review.

No retraining, GT, floor snapping, direction quotas, solver or automatic repair.
All candidates remain in G_full_candidates; G_full is the explicit in-vocabulary
Adapter subset. Unsupported proposals are reported, never silently relabelled.
"""
import argparse
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import ROOT, field, finite_vector, new_output, read, sha, validate_graph, write
from .geometry import CANONICAL_CATEGORIES
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .semantic_layout_visual import render
from .semantic_probe import validate_answer

PROTOCOL='fixed_semantic_plan_camera_boxes_v1'
DEFAULT_SOURCE=ROOT/'qwen_edit_pano/outputs/scene_content_probe_val6_run002'
CATEGORY_IDS={v:k for k,v in CANONICAL_CATEGORIES.items()}


def prompt(case,plan):
    return (
        'Convert the FIXED scene-content hypothesis into ONE coarse surrounding scene layout. '
        'The RGB image and observed camera boxes are evidence. The proposed unseen objects are '
        'hypotheses, not observations. Do not add new object types or change observed objects. '
        'For each planned item return one independent instance, or multiple separate instances '
        'if it explicitly names a group. Every item must have a plan_id matching its source id. '
        'Each instance has a distinct id different from observed ids. '
        'Choose a category from ALLOWED CATEGORIES for the actual object, preserving fine identity '
        'in the source name. If no honest supported category exists, use null; do not invent an ID '
        'or force it into misc. Such candidates remain visible in the review but are not Adapter inputs. '
        'Return JSON {"objects":[...]} with each object containing id, plan_id, category, center, '
        'size, unseen_reason, reason. center=[x,y,z] is in camera-local METRES, x right, '
        'y IMAGE-UP, z forward. Negative z is behind this SAME camera. '
        'size=[sx,sy,sz] means positive FULL camera-axis AABB lengths, not half sizes or OBB dimensions. '
        'Use the supplied observed boxes as approximate scale and spatial context. '
        'Choose positions anywhere around the camera, including intermediate angles, both sides, '
        'above/below and behind; there are no fixed rays, sides, room bounds or direction quotas. '
        'Keep visible objects and visible empty space consistent with the photograph. '
        'Use the proposed qualitative placements when consistent with the image; if they conflict, '
        'choose a reasonable unobserved location for that same object and explain the change in reason. '
        'Do not blindly flip signs or move every object far away to pass an offscreen check. '
        'Treat a placement that duplicates an already visible object as an unresolved proposal and '
        'state that in reason; numeric placement does not prove semantic correctness. '
        'unseen_reason is outside_fov, occluded, or uncertain. Occluded requires a credible visible '
        'occluder explanation; do not use it as a workaround for visible empty space. '
        'Gravity and floor are not measured: do not snap to a guessed y=-1.5 plane. '
        'Reason jointly about dimensions, centre spacing and rough furniture heights. '
        'No mesh, rotation, support graph, floor or room boundary is required. '
        'The final panorama is a 2:1 ERP encoding directions, not a long flat room. '
        f'Input horizontal FOV is {case["hfov"]:g} degrees. Return ONLY JSON. '
        '\nALLOWED CATEGORIES: '+', '.join(CATEGORY_IDS)
        +'\nFIXED OBSERVED: '+json.dumps(case['context'],ensure_ascii=False,separators=(',',':'))
        +'\nFIXED CONTENT PLAN: '+json.dumps(plan,ensure_ascii=False,separators=(',',':')))


def validate_boxes(answer,plan,observed,max_objects=160):
    items=answer.get('objects') if isinstance(answer,dict) else None
    if not isinstance(items,list):
        raise ValueError('Expected objects array with numeric scene instances')
    if len(items)+len(observed['objects'])>max_objects:
        raise ValueError('Encoder capacity exceeded; no silent truncation')
    planned={o['id'] for o in plan['additions']}
    used={o['track_id'] for o in observed['objects']};covered=set();by_plan={}
    for obj in items:
        if (not isinstance(obj,dict) or not isinstance(obj.get('id'),str) or not obj['id'].strip()
                or obj['id'] in used or not isinstance(obj.get('plan_id'),str) or obj['plan_id'] not in planned):
            raise ValueError('Each numeric instance needs a distinct id and an existing plan_id')
        if obj.get('category') is not None and (not isinstance(obj['category'],str) or obj['category'] not in CATEGORY_IDS):
            raise ValueError('Use a canonical category or null; no guessed string alias conversion')
        if 'category' not in obj or not finite_vector(obj.get('center'),3) or not finite_vector(obj.get('size'),3) or min(obj['size'])<=0:
            raise ValueError('Need explicit category/null, finite center and positive full AABB size')
        if obj.get('unseen_reason') not in ('outside_fov','occluded','uncertain') or not isinstance(obj.get('reason'),str) or not obj['reason'].strip():
            raise ValueError('Need unseen_reason and a placement reason')
        used.add(obj['id']);covered.add(obj['plan_id'])
        by_plan.setdefault(obj['plan_id'],set()).add(obj['category'])
    if covered!=planned:
        raise ValueError('Every semantic item must remain accounted for, including unmappable items')
    if any(len(cats)>1 for cats in by_plan.values()):
        raise ValueError('Split instances of one semantic item must have one category')
    return deepcopy(items)


def build_states(observed,plan,items):
    source={obj['id']:obj for obj in plan['additions']}
    candidates=deepcopy(observed);candidates.update(scope='full',stage='prediction',
        quality_status='requires_visual_review',scene_hypothesis=plan['scene_hypothesis'])
    for obj in items:
        semantic=source[obj['plan_id']]
        candidates['objects'].append(dict(track_id=obj['id'],description=semantic['name'],
            category_id=field(CATEGORY_IDS.get(obj['category']),'qwen3vl_layout_hypothesis',
                'model_selected_canonical_category',reason='no_supported_category'),
            center_local_m=field(obj['center'],'qwen3vl_layout_hypothesis','camera_xyz_hypothesis'),
            size_aabb_local_m=field(obj['size'],'qwen3vl_layout_hypothesis','full_camera_axis_aabb'),
            bbox2d_visible_xyxy_norm=field(None,reason='unobserved_hypothesis'),
            in_primary_room=field(None,reason='room_extent_not_measured'),
            local_evidence='unobserved',completion_role='hypothesized',unseen_reason=obj['unseen_reason'],
            reason=obj['reason'],semantic_plan_id=obj['plan_id'],semantic_reason=semantic['reason'],
            semantic_placement=semantic['placement'],geometry_quality='hypothesized_not_reconstructed'))
    full=deepcopy(candidates)
    unmapped=[deepcopy(o) for o in candidates['objects'][len(observed['objects']):] if not o['category_id']['valid']]
    full['objects']=[o for o in full['objects'] if o['category_id']['valid'] or o['local_evidence']=='visible']
    full['unmapped_candidate_ids']=[o['track_id'] for o in unmapped]
    full['semantic_plan_fully_exported']=not unmapped
    for state in (candidates,full):
        validate_graph(state)
        if state['objects'][:len(observed['objects'])]!=observed['objects']:
            raise AssertionError('Observed state was changed')
    return candidates,full,unmapped


def diagnostics(items,hfov,image_size):
    th=math.tan(math.radians(hfov)/2);tv=th*image_size[1]/image_size[0]
    normals=np.array([[1,0,th],[-1,0,th],[0,1,tv],[0,-1,tv],[0,0,1]])
    result=[]
    for obj in items:
        c=np.array(obj['center']);s=np.array(obj['size'])
        outside=bool(np.any(normals@c+np.abs(normals)@(s/2)<=0))
        result.append(dict(id=obj['id'],unseen_reason=obj['unseen_reason'],
            outside_input_fov_certified=outside,category=obj['category'],
            azimuth_deg=float(np.degrees(np.arctan2(c[0],c[2]))),
            camera_origin_inside_aabb=bool(np.all(np.abs(c)<=s/2)),
            note='Half-space separation is sufficient, not necessary; failed certification does not prove visibility. Occlusion is unverified.'))
    return result


def load_cases(source,only_ids):
    config=read(source/'config.json');cases=[]
    found={c['id'] for c in config['inputs']}
    if only_ids and set(only_ids)-found:
        raise ValueError('Unknown case IDs')
    for saved in config['inputs']:
        if only_ids and saved['id'] not in only_ids:continue
        case=deepcopy(saved);folder=source/case['id']
        if read(folder/'input.json')!=saved:
            raise ValueError('Saved semantic input differs from source config')
        graph=validate_graph(read(case['state']))
        if graph['stage']!='prediction' or graph['scope']!='observed':
            raise ValueError('Need predicted G_obs, not GT')
        if any(not o[k]['valid'] for o in graph['objects'] for k in ('category_id','center_local_m','size_aabb_local_m')):
            raise ValueError('Fixed observed boxes must have category, center and full size')
        for path,digest in case['source_sha256'].items():
            if sha(Path(path))!=digest:raise ValueError('Source observed image/state changed')
        plan_path=folder/'image_state/answer.json'
        plan=read(plan_path);validate_answer(plan,case['context'])
        case.update(plan=plan,observed=graph,prompt=prompt(case,plan))
        case['source_sha256'].update({str(p.resolve()):sha(p) for p in (source/'config.json',folder/'input.json',plan_path)})
        cases.append(case)
    if not cases:raise ValueError('No saved image_state content plans selected')
    return cases


def run(model,cases,output,seed=3407,max_tokens=4096,postprocess_only=False):
    records=[]
    for case in cases:
        dest=output/case['id'];dest.mkdir(exist_ok=True)
        with Image.open(case['image']) as handle:image=handle.convert('RGB')
        image.thumbnail((768,768));image.save(dest/'local.png')
        write(dest/'config.json',dict(protocol=PROTOCOL,image=case['image'],image_sha256=sha(Path(case['image'])),
            hfov_deg=case['hfov'],source_sha256=case['source_sha256']))
        write(dest/'G_obs.json',case['observed']);write(dest/'semantic_plan.json',case['plan'])
        record=dict(id=case['id'],status='failed')
        try:
            if postprocess_only:
                if read((dest/'layout.request.json'))['prompt']!=case['prompt']:
                    raise ValueError('Saved numeric answer belongs to another prompt')
                answer=read(dest/'layout.json')
            else:
                answer=model.ask(image,case['prompt'],dest/'layout.txt',seed=seed,sample=True,
                    profile='official',max_tokens=max_tokens,guard_repetition=True)
            items=validate_boxes(answer,case['plan'],case['observed'])
            candidates,full,unmapped=build_states(case['observed'],case['plan'],items)
            checks=diagnostics(items,case['hfov'],image.size)
            write(dest/'G_full_candidates.json',candidates);write(dest/'G_full.json',full)
            write(dest/'unmapped_candidates.json',unmapped);write(dest/'geometry_review.json',checks)
            record.update(status='artifacts_written_requires_review',observed=len(case['observed']['objects']),
                planned_items=len(case['plan']['additions']),new_numeric_instances=len(items),
                new_adapter_instances=len(items)-len(unmapped),unmapped_instances=len(unmapped),
                all_semantic_candidates_exported=not unmapped,
                outside_claims_certified=sum(c['outside_input_fov_certified'] for c in checks if c['unseen_reason']=='outside_fov'),
                outside_claims_not_certified=sum(not c['outside_input_fov_certified'] for c in checks if c['unseen_reason']=='outside_fov'),
                occlusion_unverified=sum(c['unseen_reason']=='occluded' for c in checks),
                uncertain_instances=sum(c['unseen_reason']=='uncertain' for c in checks),
                semantics_verified=False,geometry_verified=False,panorama_generated=False)
            render(candidates,image,case['hfov'],dest,record)
        except Exception as exc:
            record['error']=str(exc)
        write(dest/'summary.json',record);records.append(record)
        write(output/'summary.json',dict(protocol=PROTOCOL,records=records,scene_gt_used=False))
        cards=[]
        for r in records:
            cid=r['id'];cards.append('<section><h2>'+escape(cid)+'</h2><p>'+escape(str(r))+'</p>'
                +('<a href="'+cid+'/index.html"><img style="max-width:100%" src="'+cid+'/erp_state_map.png"></a>' if r['status']!='failed' else '')+'</section>')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>蓝色观测／橙色补全／红色未映射候选</h1><p>2:1图是状态方向图，不是生成的ERP照片。未进行自动几何修正。</p>'+''.join(cards))
        print(json.dumps(record,ensure_ascii=False),flush=True)
    if any(sha(Path(p))!=h for case in cases for p,h in case['source_sha256'].items()):
        raise ValueError('Source inputs changed during layout generation')
    return all(r['status']!='failed' for r in records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--only-id',action='append')
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--max-tokens',type=int,default=4096)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--postprocess-only',action='store_true')
    args=parser.parse_args()
    if not 0<=args.seed<2**63 or args.max_tokens<1:parser.error('Invalid generation settings')
    cases=load_cases(args.source.resolve(),args.only_id)
    config=dict(protocol=PROTOCOL,source=str(args.source.resolve()),cases=cases,model=str(args.model.resolve()),
        seed=args.seed,max_tokens=args.max_tokens,code_sha256={p.name:sha(p) for p in
        (Path(__file__),Path(__file__).with_name('qwen_upstream.py'))})
    output=args.output.resolve()
    if output.exists() and (not (args.resume or args.postprocess_only) or read(output/'run_config.json')!=config):
        parser.error('Use a new output or resume unchanged inputs and settings')
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed',cases=len(cases),planned_items=sum(len(c['plan']['additions']) for c in cases),
            model_loaded=False,scene_gt_used=False,arbitrary_camera_xyz=True,canonical_categories=len(CATEGORY_IDS))))
        return
    if args.postprocess_only and not output.exists():parser.error('Postprocess needs existing numeric responses')
    if not output.exists():output=new_output(output);write(output/'run_config.json',config)
    model=None if args.postprocess_only else Qwen(args.model)
    if not run(model,cases,output,args.seed,args.max_tokens,args.postprocess_only):
        raise SystemExit('Some layouts failed; inspect saved raw responses and summary.json')


if __name__=='__main__':
    main()
