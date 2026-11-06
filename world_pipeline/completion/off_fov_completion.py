"""New off-FOV completion pilot. Fixed Qwen2D/WildDet3D observations; no GT input.

Writes hypotheses/proposals even when constraints fail. Produces world states
and visualizations only; no Adapter connection or panorama model is constructed.
Unmapped fine names are retained. No GPU jobs are submitted by this module.
Semantic/physical/causal quality is never self-certified.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil

from PIL import Image

from ..common import FRAME, new_output, read, sha, validate_graph, write
from .batch_world_review import inventory_image
from .causal_contract import execute, typed_graph, validate_candidates, validate_factors
from .causal_world_completion import (check_hashes, graph_svg,
                                     load_pilot, selected_plan)
from .geometry import CANONICAL_CATEGORIES, parse_json
from .off_fov_geometry import REGIONS, layout_answer, solve
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .scene_completion import export_states, layout_prompt, map_plan
from .semantic_layout_visual import render

PROTOCOL='outside_input_frustum_world_v2'


def code_hashes():
    from .causal_world_completion import code_hashes as shared_hashes
    result=shared_hashes()
    paths=[Path(__file__),Path(__file__).with_name('off_fov_geometry.py')]
    result.update({str(p.resolve()):sha(p) for p in paths})
    return result


def context_prompt(case, reasoning='hypothesis'):
    from .causal_world_completion import visible_inventory
    if reasoning=='hypothesis':
        return ('Infer one plausible existing indoor world around this SAME camera from this partial '
            'photograph. The prediction target is objects OUTSIDE the input camera field of view. '
            'First infer scene context and possible latent activities/environmental conditions. '
            'Explain evidence versus uncertainty; do not assume hidden conditions are observed. '
            'Do not design, decorate or improve the photographed room. Aesthetic preferences do not '
            'create a need to add ornaments, and functions do not imply mandatory equipment. '
            'All visible instances remain fixed even when partially cropped. Omitted detections do '
            'not make a visible object hidden. Do not enumerate new objects or geometry yet. '
            'Return ONLY JSON with scene_type, scene_hypothesis (strings), factors (array). '
            'Each factor has id, kind (context/activity/environment/preference), statement, assumed '
            '(boolean), evidence_ids (observed IDs or []), reason, optional parents (factor IDs). '
            'Parents only express a hypothesized enabling dependence, not co-occurrence or evidence. '
            'Factors form a DAG; value=assumed AND all(parent values). Empty factors are allowed. '
            'This is a declared hypothesis prior, not discovered causality. '
            f'Input horizontal FOV is {case["hfov"]:g} degrees. OBSERVED: '
            +json.dumps(visible_inventory(case),ensure_ascii=False))
    if reasoning!='plain':raise ValueError('Unknown reasoning mode')
    return ('Describe one plausible surrounding indoor scene compatible with this partial image. '
        'Infer the scene type from observations. Do not design a room or demand missing functions. '
        'Do not enumerate additional objects or numeric geometry yet. Return ONLY JSON with '
        'scene_type, scene_hypothesis (strings), factors (exactly an empty array). '
        'This ordinary baseline does not construct explicit activity/dependency factors. '
        'OBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False))


def contents_prompt(case,factors,reasoning='hypothesis'):
    from .causal_world_completion import visible_inventory
    question=('Infer the existing surrounding objects OUTSIDE THE INPUT CAMERA FIELD OF VIEW '
        'for ONE plausible world consistent with the image and fixed observations. This is '
        'hidden-world prediction, not equipment needed to make the room functional or an interior '
        'design request. Do not propose enhancements to visible surfaces. Do not add visible objects '
        'omitted by the detector, components or duplicates of observed instances. A fully occluded '
        'new object within the original cone is not this task. Object count is free; no direction quota. '
        'Return ONLY JSON {"additions":[...],"relations":[...]}. Each addition has exactly id, name, '
        'object_type (ordinary noun), reason, placement, off_fov_region, grounds and optional requires. '
        'off_fov_region must be one of left/right/behind_camera/above/below. left/right mean beyond '
        'the input viewing cone on that camera side, not merely on that side WITHIN the photograph. '
        'above/below also mean beyond the camera cone, not above/below furniture in the photograph. '
        'behind_camera means the full new object is at negative camera z. Choose a plausible region '
        'first, consistent with placement. No coordinates, dimensions or numeric search bounds here. '
        'grounds is OR of AND lists of listed factor IDs; [[]] is an unconditional scene prior. '
        'requires lists observed/new IDs whose existence this hypothesis requires, not mere neighbours. '
        'Do not claim every plausible co-occurrence is necessary. Selection follows these declared '
        'gates, not a learned causal model. Each optional relation has source, target (instance IDs), '
        'kind and reason. Allowed kinds: left_of/right_of/above/below/in_front_of/behind/near/'
        'supported_by/on_top_of/inside. near additionally requires a positive max_distance_m. '
        'All directions are camera axes: x right, y image-up, z forward; in_front_of means smaller '
        'z, behind means larger z, distinct from behind_camera. Support/inside point to a support/'
        'container. Supports must exist among observations or explicitly proposed instances. '
        'A support relation must be geometrically compatible with the selected off-FOV region; '
        'do not place new objects on visible surfaces just to decorate them. '
        'These are spatial constraints, not identified causal edges. Room bounds and gravity are unknown. '
        '\nOBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False)
        +'\nHYPOTHESES: '+json.dumps(factors,ensure_ascii=False))
    if reasoning=='plain':
        question += ('\nORDINARY BASELINE: use grounds=[[]], requires=[], relations=[] for every '
                     'instance. No explicit mechanism or spatial-graph reasoning in this baseline.')
    return question


def boxes_prompt(case,plan,max_shift):
    return layout_prompt(case,plan)+(
        '\nSTRICT OFF-FOV LAYOUT: every NEW full AABB must be disjoint from the input pinhole '
        'viewing cone. Occluded or uncertain in-FOV objects are not valid substitutes. Only new '
        'objects are to be placed; existing instances and their full extents remain frozen. '
        'Return ONLY {"objects":[{"id":...,"center":[x,y,z],"size":[sx,sy,sz],'
        '"reason":...}]}. Do NOT return center_bounds_m, off_fov_region, unseen_reason or category. '
        'Use the fixed off_fov_region from each semantic instance. A program will create a local '
        f'trust region of +/-{max_shift:g} metres per axis around your centre, and search it without '
        'changing sizes, identity, selected region or observations. This trust region is not a measured '
        'room extent and cannot turn an in-FOV semantic design into an off-FOV prediction. '
        'For right, the full box needs xmin>0 and xmin>tan(hfov/2)*zmax; for left, xmax<0 and '
        '-xmax>tan(hfov/2)*zmax. Behind_camera needs zmax<0. Above/below use the analogous vertical '
        'cone, where tan(vfov/2)=tan(hfov/2)*height/width. Do not just move the centre out of frame. '
        'Choose appropriate metric sizes before the search; do not shrink objects to pass FOV. '
        'Do not use arbitrary far-away locations. Distances/floor/room extent are hypotheses. '
        'Respect the fixed relations and dependency order. Camera in_front_of means smaller z, '
        'not nearer along every panoramic viewing ray. Negative z is valid behind the camera. '
        f'The centred input has width/height={case["image_size"][0]}/{case["image_size"][1]}, '
        f'and horizontal FOV={case["hfov"]:g} degrees. This is full-frame, not a cropped image.')


def normalized_context(answer,case,reasoning):
    result=validate_factors(answer,[o['id'] for o in case['context']])
    if reasoning=='plain' and result['factors']:
        raise ValueError('Plain baseline must not introduce explicit factors')
    return result


def normalized_contents(answer,factors,case,reasoning):
    plain=deepcopy(answer);regions={}
    if not isinstance(plain,dict) or not isinstance(plain.get('additions'),list):
        raise ValueError('Contents need additions array')
    for i,obj in enumerate(plain['additions']):
        if not isinstance(obj,dict) or obj.get('off_fov_region') not in REGIONS:
            raise ValueError(f'additions[{i}].off_fov_region must be one of {REGIONS}')
        regions[obj.get('id')]=obj.pop('off_fov_region')
    result=validate_candidates(plain,factors,[o['id'] for o in case['context']],max_candidates=256)
    for obj in result['additions']:obj['off_fov_region']=regions[obj['id']]
    if any(r['kind']=='near' and 'max_distance_m' not in r for r in result['relations']):
        raise ValueError('near relation needs explicit positive max_distance_m')
    if reasoning=='plain' and (result['relations'] or any(
            o['grounds'] != [[]] or o['requires'] for o in result['additions'])):
        raise ValueError('Plain baseline must have no explicit graph/gated selection')
    return result


def load_semantics(source,case,max_objects,reasoning):
    source=Path(source);config=read(source/'config.json');folder=source/case['id']
    if (config['protocol']!=PROTOCOL or config['case']!=case or
            config['max_objects']!=max_objects or config['reasoning']!=reasoning or
            config['code_sha256']!=code_hashes() or read(folder/'input.json')!=case):
        raise ValueError('Semantic input/code/capacity/scope differs; do not reuse legacy in-FOV semantics')
    paths=[source/'config.json',folder/'input.json'];values=[]
    for name in ('context','contents'):
        raw=folder/(name+'.txt');parsed=folder/(name+'.json')
        request=folder/(name+'.request.json');generation=folder/(name+'.generation.json')
        value=read(parsed)
        expected=(context_prompt(case,reasoning) if name=='context' else
                  contents_prompt(case,values[0],reasoning))
        if parse_json(raw.read_text())!=value or read(request)['prompt']!=expected:
            raise ValueError('Semantic raw/parsed/request binding differs: '+name)
        if read(generation).get('finish_reason') in ('length','repeated_objects'):
            raise ValueError('Semantic generation truncated: '+name)
        values.append(normalized_context(value,case,reasoning) if name=='context' else
                      normalized_contents(value,values[0],case,reasoning))
        paths.extend((raw,parsed,request,generation))
    return (*values,{str(p.resolve()):sha(p) for p in paths})


def report(folder,record):
    body=('<!doctype html><meta charset="utf-8"><title>Outside-FOV world</title>'
          '<style>body{max-width:1400px;margin:24px auto;font:16px/1.6 sans-serif}img{max-width:100%}'
          'pre{white-space:pre-wrap;overflow-wrap:anywhere}iframe{width:100%;height:700px}</style>'
          '<h1>固定局部观测 → 视野外世界状态</h1><p>新增对象只允许在输入视锥之外。'
          '假设图不是已识别的因果模型，方向图不是生成照片；数值通过不等于语义/物理验证。</p>'
          '<pre>'+escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre><img width="640" src="local.png">')
    if (folder/'qwen_2d.png').exists():
        body+='<h2>已运行的Qwen二维观测</h2><img width="800" src="qwen_2d.png">'
    if (folder/'wilddet_projection.png').exists():
        body+='<h2>已运行的WildDet3D三维观测重投影</h2><img width="800" src="wilddet_projection.png">'
    if (folder/'typed_graph.svg').exists():
        body+='<h2>实际假设图</h2><object style="width:100%;height:700px" data="typed_graph.svg"></object>'
    if (folder/'world/scene_3d.html').exists():
        body+='<h2>完整世界候选（含失败候选，不能据此认定通过）</h2><a href="world/scene_3d.html">打开交互3D图</a>'
        body+='<iframe src="world/scene_3d.html"></iframe><img src="world/erp_state_map.png">'
        body+='<h2>原图投影：新增对象不应进入原图</h2><img src="world/local_projection.png">'
    for path in sorted(folder.glob('*.json')):
        body+='<p><a href="'+path.name+'">'+path.name+'</a></p>'
    (folder/'index.html').write_text(body)


def validate_world(observed,full,items,max_objects):
    validate_graph(observed);validate_graph(full)
    n=len(observed['objects'])
    if full['objects'][:n]!=observed['objects']:
        raise ValueError('Observed prefix changed')
    if len(full['objects'])!=n+len(items):
        raise ValueError('New objects were dropped or extra instances were added')
    if len(full['objects'])>max_objects:
        raise ValueError('World safety capacity exceeded; no truncation')
    for obj,item in zip(full['objects'][n:],items):
        if (obj['track_id']!=item['id'] or obj['center_local_m']['value']!=item['center']
                or obj['size_aabb_local_m']['value']!=item['size']):
            raise ValueError('World/layout identity or geometry differs')
        if not obj['center_local_m']['valid'] or not obj['size_aabb_local_m']['valid']:
            raise ValueError('Selected object has missing geometry')
    return full


def generate_layout(model,image,case,observed,plan,mappings,folder,options,max_shift,sweeps,
                    max_distance,layout_retries):
    """One initial proposal and a bounded numeric-feedback correction, all saved.

    A correction may reconsider a centre in its fixed semantic off-FOV region.
    Local search stays within each proposal's program-owned trust region.
    Categories, selected regions/instances, observed fields and first valid sizes
    remain fixed. The final valid-schema attempt is saved, never a dropped subset.
    """
    base=boxes_prompt(case,plan,max_shift);prompt=base
    attempts=[];last_valid=None;sizes=None
    count=1+layout_retries if plan['additions'] else 1
    for attempt in range(count):
        name='boxes' if attempt==0 else f'boxes_retry{attempt}'
        answer=None;diagnostic={}
        try:
            answer=(model.ask(image,prompt,folder/(name+'.txt'),**{**options,'seed':options['seed']+attempt})
                    if plan['additions'] else dict(objects=[]))
            write(folder/(f'layout_attempt{attempt}_proposal.json'),answer)
            if attempt==0:write(folder/'layout_proposal.json',answer)
            items=layout_answer(answer,plan,mappings,max_shift)
            current_sizes={o['id']:o['size'] for o in items}
            if sizes is not None and sizes!=current_sizes:
                raise ValueError('Correction changed fixed sizes; do not shrink/change objects to pass FOV')
            sizes=deepcopy(current_sizes)
            items,search=solve(items,observed,plan['relations'],case['hfov'],list(image.size),sweeps,max_distance)
            diagnostic=search
            last_valid=(items,search,attempt,answer)
            attempts.append(dict(attempt=attempt,source=name,numeric_constraints_passed=
                                 search['after']['numeric_constraints_passed']))
            write(folder/(f'layout_attempt{attempt}_review.json'),search)
            if search['after']['numeric_constraints_passed']:break
        except ValueError as exc:
            diagnostic=dict(error=f'{type(exc).__name__}: {exc}')
            attempts.append(dict(attempt=attempt,source=name,numeric_constraints_passed=False,**diagnostic))
            write(folder/(f'layout_attempt{attempt}_review.json'),diagnostic)
        if attempt+1<count:
            prompt=(base+'\nNUMERIC FEEDBACK: the previous attempt below did not satisfy constraints. '
                'Correct metric placement in the SAME declared off_fov_region of each instance. '
                'Do not use a visible surface as a substitute, change semantic selection/region, '
                'change sizes, delete instances or edit observations. If the world is infeasible, '
                'do not fake success; the program retains all failed proposals. Return the same '
                'four-field box schema only.\nFIXED SIZES: '+json.dumps(sizes)
                +'\nPREVIOUS ANSWER: '+json.dumps(answer,ensure_ascii=False)
                +'\nCOMPUTED DIAGNOSTICS: '+json.dumps(diagnostic,ensure_ascii=False))
    write(folder/'layout_attempts.json',dict(attempts=attempts,layout_calls=len(attempts) if plan['additions'] else 0,
          selected_attempt=None if last_valid is None else last_valid[2],selection='last_valid_schema_attempt',
          correction_sizes_fixed=True,semantic_regions_fixed=True))
    if last_valid is None:
        raise ValueError('No structurally valid layout after bounded attempts; inspect layout_attempt*_review.json and raw replies')
    items,search,attempt,answer=last_valid
    write(folder/'layout_selected_proposal.json',answer)
    return items,search,attempt


def run(model,case,output,max_objects=256,stage='full',semantic_source=None,seed=3407,max_tokens=4096,
        reasoning='hypothesis',max_shift=.75,sweeps=3,max_distance=20.,layout_retries=1):
    folder=output/case['id'];folder.mkdir()
    record=dict(id=case['id'],protocol=PROTOCOL,status='failed',world_state_ready=False,
                observed_instances=len(case['context']),semantics_verified=False,geometry_verified=False,
                causal_gain_validated=False,panorama_generated=False,scene_gt_used=False)
    try:
        check_hashes(case['source_sha256']);write(folder/'input.json',case)
        observed=validate_graph(read(case['state']));write(folder/'G_obs.json',observed)
        shutil.copyfile(case['image'],folder/'local.png')
        inventory_image(case['image'],read(case['inventory']),folder/'qwen_2d.png')
        projection=Path(case['state']).with_name('projection_boxes.png')
        if projection.is_file():shutil.copyfile(projection,folder/'wilddet_projection.png')
        with Image.open(case['image']) as handle:image=handle.convert('RGB')
        image.thumbnail((768,768))
        write(folder/'inference_image.json',dict(size=list(image.size),hfov=case['hfov'],
                                               camera='centred_pinhole',full_frame_preserved=True))
        options=dict(seed=seed,sample=True,profile='official',max_tokens=max_tokens,guard_repetition=True)
        if semantic_source is None:
            factors=normalized_context(model.ask(image,context_prompt(case,reasoning),folder/'context.txt',**options),case,reasoning)
            contents=normalized_contents(model.ask(image,contents_prompt(case,factors,reasoning),folder/'contents.txt',**options),
                                         factors,case,reasoning)
        else:
            factors,contents,provenance=load_semantics(semantic_source,case,max_objects,reasoning)
            for stem in ('context','contents'):
                for suffix in ('.txt','.json','.request.json','.generation.json'):
                    shutil.copyfile(semantic_source/case['id']/(stem+suffix),folder/(stem+suffix))
            write(folder/'semantic_source.json',provenance)
        ids=[o['id'] for o in case['context']]
        execution=execute(factors,contents,ids);plan=selected_plan(factors,execution)
        write(folder/'mechanism_execution.json',execution);write(folder/'semantic_plan.json',plan)
        graph=typed_graph(factors,contents,ids,execution)
        write(folder/'typed_graph.json',graph);graph_svg(graph,folder/'typed_graph.svg')
        mappings=map_plan(plan,'semantic_lexicon_v1');write(folder/'category_mapping.json',mappings)
        record.update(status='semantic_hypothesis_saved_needs_review',proposed_instances=len(contents['additions']),
                      selected_instances=len(plan['additions']),inactive_instances=len(contents['additions'])-len(plan['additions']))
        if stage!='semantic':
            items,search,attempt=generate_layout(model,image,case,observed,plan,mappings,folder,options,
                                                max_shift,sweeps,max_distance,layout_retries)
            record['selected_layout_attempt']=attempt
            write(folder/'layout_solved.json',dict(objects=items));write(folder/'numeric_review.json',search)
            checks=search['after']
            semantic,_,unmapped=export_states(observed,plan,items,mappings,checks)
            semantic.update(completion_protocol=PROTOCOL,target='outside_input_frustum',
                            world_relations=deepcopy(plan['relations']),typed_hypothesis_graph=graph,
                            causal_model_identified=False,quality_status='candidate_not_semantically_verified',
                            category_policy='open_names_with_optional_existing_coarse_mapping',
                            input_camera=dict(model='centred_pinhole',hfov_deg=case['hfov'],
                                              image_size_wh=case['image_size'],origin_local_m=[0,0,0],
                                              y_axis='image_up_not_verified_gravity'),
                            adapter_connected=False)
            for obj,item in zip(semantic['objects'][len(observed['objects']):],items):
                obj.update(geometry_proposal_center_m=deepcopy(item['proposal_center']),
                           center_search_bounds_m=deepcopy(item['center_bounds_m']),
                           search_bounds_source=item['bounds_source'],off_fov_region=item['off_fov_region'],
                           geometry_method='qwen_proposal_then_bounded_coordinate_search')
            validate_world(observed,semantic,items,max_objects)
            write(folder/'unmapped_candidates.json',unmapped)
            # Keep all selected candidates on failure. Do not export a successful subset.
            blockers=[]
            if not items:blockers.append('no_selected_outside_instances')
            if not checks['numeric_constraints_passed']:blockers.append('numeric_constraints_failed')
            if execution['inactive_relations']:blockers.append('relations_reference_inactive_instances')
            record.update(status='world_blocked' if blockers else 'numeric_world_ready_needs_semantic_review',
                          geometry_generated=bool(items),numeric_constraints_passed=checks['numeric_constraints_passed'],
                          hard_violations=checks['hard_violations'],unverified_relations=checks['unverified_relations'],
                          observed_state_preserved=True,blockers=blockers,
                          world_state_ready=not blockers,unmapped_new=len(unmapped),
                          category_mapping_blocks_world=False,adapter_connected=False)
            semantic['quality_status']=record['status']
            semantic['numeric_constraints_passed']=checks['numeric_constraints_passed']
            write(folder/'G_full.json',semantic)
            hidden=dict(schema_version=semantic['schema_version'],frame=semantic['frame'],
                        scope='hidden',stage='prediction',target='outside_input_frustum',
                        completion_protocol=PROTOCOL,input_camera=deepcopy(semantic['input_camera']),
                        quality_status=record['status'],numeric_constraints_passed=checks['numeric_constraints_passed'],
                        objects=semantic['objects'][len(observed['objects']):])
            validate_graph(hidden);write(folder/'G_out.json',hidden)
            world=folder/'world';world.mkdir()
            render(semantic,image,case['hfov'],world,record,
                   position_note='所有新增目标均为输入视野之外。失败候选仍展示；仅以数值审计和summary判定是否通过。')
            page=world/'index.html'
            page.write_text(page.read_text().replace(
                '红色候选保留在G_full_candidates中，不进入Adapter子集G_full。',
                '红色对象仅表示没有现有粗类别映射，仍保留在完整G_full中；本阶段不连接Adapter。'))
        check_hashes(case['source_sha256'])
        if semantic_source is not None:check_hashes(provenance)
        files=[p for p in folder.iterdir() if p.is_file() and p.name not in ('summary.json','index.html')]
        record['export_sha256']={str(p.resolve()):sha(p) for p in files}
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',world_state_ready=False)
    write(folder/'summary.json',record);report(folder,record)
    write(output/'summary.json',dict(protocol=PROTOCOL,records=[record],scene_gt_used=False))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="'+case['id']+'/index.html">'
                                   '打开视野外世界补全与诊断</a><pre>'+escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre>')
    print(json.dumps({k:v for k,v in record.items() if k!='export_sha256'},ensure_ascii=False),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('semantic','full','layout'),default='full')
    p.add_argument('--semantic-from',type=Path,help='Only NEW strict off-FOV semantic run root')
    p.add_argument('--max-objects',type=int,default=256,help='World safety capacity, not an Adapter setting')
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--reasoning',choices=('plain','hypothesis'),default='hypothesis')
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--max-tokens',type=int,default=6144)
    p.add_argument('--max-shift',type=float,default=.75)
    p.add_argument('--sweeps',type=int,default=3)
    p.add_argument('--max-distance',type=float,default=20.)
    p.add_argument('--layout-retries',type=int,default=1,help='At most this many numeric-feedback corrections, 0..2')
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if (a.output.exists() or not 0<=a.seed<2**63 or not 1<=a.max_tokens<=16384 or
            not 0<=a.max_shift<=1.5 or not 1<=a.sweeps<=10 or not 0<a.max_distance<=100 or
            not 1<=a.max_objects<=256 or not 0<=a.layout_retries<=2):
        p.error('Need NEW output and valid bounded settings')
    if a.stage=='layout' and a.semantic_from is None:p.error('layout needs --semantic-from')
    case=load_pilot()
    with Image.open(case['image']) as im:case['image_size']=list(im.size)
    # This entry deliberately supports the audited centred validation face only.
    # Arbitrary smartphone intrinsics/crops need a separate camera-aware observer.
    if case['hfov']!=90 or case['image_size'][0]!=case['image_size'][1]:
        p.error('This pilot requires the audited full-frame centred square 90-degree validation input')
    if len(case['context'])>a.max_objects:p.error('Observations exceed safety capacity')
    if not a.model.is_dir():p.error('Local Qwen model missing; no downloads')
    source=a.semantic_from.resolve() if a.semantic_from else None
    if source:_,_,source_hashes=load_semantics(source,case,a.max_objects,a.reasoning)
    config=dict(protocol=PROTOCOL,case=case,frame=FRAME,max_objects=a.max_objects,stage=a.stage,
                reasoning=a.reasoning,seed=a.seed,max_tokens=a.max_tokens,max_shift=a.max_shift,
                sweeps=a.sweeps,max_distance=a.max_distance,model=str(a.model.resolve()),
                layout_retries=a.layout_retries,
                semantic_source=str(source) if source else None,code_sha256=code_hashes(),scene_gt_used=False)
    if a.preflight:
        print(json.dumps(dict(status='preflight_passed',id=case['id'],observed=len(case['context']),
              target='outside_input_frustum',reasoning=a.reasoning,
              semantic_calls=0 if source else 2,maximum_layout_calls=(1+a.layout_retries)*int(a.stage!='semantic'),
              coarse_mapping_classes=len(CANONICAL_CATEGORIES),world_safety_capacity=a.max_objects,
              category_mapping_blocks_world=False,adapter_connected=False,
              model_loaded=False,gpu_started=False,scene_gt_used=False),ensure_ascii=False));return
    output=new_output(a.output);write(output/'config.json',config)
    if source:check_hashes(source_hashes)
    model=None if source and a.stage=='semantic' else Qwen(a.model)
    if config['code_sha256']!=code_hashes():raise ValueError('Code changed while loading model')
    record=run(model,case,output,a.max_objects,a.stage,source,a.seed,a.max_tokens,a.reasoning,
               a.max_shift,a.sweeps,a.max_distance,a.layout_retries)
    if record['status'] in ('failed','world_blocked'):
        raise SystemExit('Completion blocked; inspect summary.json, raw replies and numeric_review.json. '
                         'All candidates preserved; this is world-state completion only, no Adapter.')


if __name__=='__main__':main()
