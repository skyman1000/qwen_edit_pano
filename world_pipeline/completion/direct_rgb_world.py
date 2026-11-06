"""RGB-only direct world baseline: no observer network, SCM or layout solver."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

import numpy as np
from PIL import Image

from ..common import ROOT,new_output,read,rows,sha,write,field,validate_graph
from .geometry import graph,CANONICAL_CATEGORIES
from .semantic_categories import resolve_identity
from .world_generation import recipe,image_bytes
from .qwen_upstream import DEFAULT_MODEL

DEFAULT_INPUTS=ROOT/'qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl'
KIND='direct_RGB_structured_world_v1'
RELATIONS={'on','inside','attached_to','near'}


def code_hashes():
    names=('direct_rgb_world','direct_rgb_review','world_generation','world_api','qwen_upstream','geometry',
        'semantic_categories','category_bridge','assembly_presentation_visual','assembly_pilot_visual','semantic_layout_visual','batch_world_review','scene_completion')
    files=[Path(__file__).with_name(n+'.py') for n in names]+[ROOT/'qwen_edit_pano/world_pipeline/common.py',ROOT/'qwen_edit_pano/scripts/direct_rgb_world.sh']
    return {str(p.resolve()):sha(p) for p in files}


def pixel_sha(image):return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def prompt_for(scene,max_objects):
    return ('From the attached PARTIAL indoor RGB photograph, directly predict ONE plausible complete surrounding 360-degree world state. '
        'Include visible object instances and plausible objects OUTSIDE the original image frustum, including possible side and rear space. '
        'Each separate physical instance appears once; do not duplicate visible objects as hidden ones, or list parts again as whole objects. '
        'Do not list floor, wall or ceiling surfaces as object instances; the task is the same object-state inventory used by the full method. '
        'Include suitable ordinary belongings as well as furniture; infer their number from this image without a count or direction quota. '
        'Keep the visible content consistent with the photograph. Hidden content is uncertain. Do not infer resident biography. '
        'Directly estimate object categories, metric dimensions, centres and orientations together. Use coherent scales and spatial arrangements. '
        'Do not output function hypotheses, scene programs, probability tables, a causal graph or an explanation. '
        'Return ONE JSON object with exactly objects and relations. objects is an array of compact rows. '
        'Each row has EXACTLY ELEVEN columns: [id, name, category, evidence, cx, cy, cz, width, height, depth, yaw_deg]. '
        'id is a unique string, name a specific English noun phrase, category an ordinary English object noun. '
        'evidence is exactly visible or outside_view. The remaining seven entries must be finite numbers, not null or text. '
        'Full intrinsic dimensions width,height,depth must be positive metres, not half sizes. '
        'Camera frame: origin at original input camera, X right, Y IMAGE-UP, Z forward. Rear has Z<0. '
        'yaw_deg is absolute rotation about camera Y; intrinsic x is width and z is depth: '
        'x_cam=cos(yaw)*x_obj+sin(yaw)*z_obj; z_cam=-sin(yaw)*x_obj+cos(yaw)*z_obj. '
        'Centres are ABSOLUTE camera coordinates, not offsets from a host or group. '
        'Relations are optional descriptive spatial/support information, encoded as rows [source_id, kind, target_id]; '
        'kind must be on, inside, attached_to or near. Refer only to object IDs in this same response. '
        'Use an empty relations array when no relation is asserted. These are spatial descriptions, not causal mechanisms. '
        'Do not invent walls or known room dimensions. Never copy empty numeric placeholders. '
        'Return a compact, closed JSON object. '+str(max_objects)+' total instances is only a computation limit, not a target. '
        '\nORIGINAL_CAMERA: '+json.dumps(dict(hfov_degrees=scene['hfov'],image_size=scene['image_size'],frame='X_right_Y_image_up_Z_forward_metres')))


def numeric(v):return type(v) in (int,float) and math.isfinite(v)


def validate_answer(value,max_objects=128):
    if not isinstance(value,dict) or set(value)!={'objects','relations'}:raise ValueError('Return exactly objects and relations')
    objects=value['objects'];relations=value['relations']
    if not isinstance(objects,list) or len(objects)>max_objects:raise ValueError('objects must be an array within the computation capacity; no truncation')
    ids=[]
    for i,r in enumerate(objects):
        if not isinstance(r,list) or len(r)!=11:raise ValueError('objects['+str(i)+'] requires exactly eleven ordered columns')
        if any(not isinstance(s,str) or not s.strip() for s in r[:4]):raise ValueError('id, name, category and evidence must be nonempty strings')
        if r[3] not in ('visible','outside_view'):raise ValueError('evidence must be visible or outside_view')
        if any(not numeric(v) for v in r[4:]):raise ValueError('All seven geometry fields must be finite numbers, not null/bool/text')
        if min(r[7:10])<=0:raise ValueError('Full width,height,depth must be positive')
        if r[0] in ids:raise ValueError('Duplicate instance ID: '+r[0])
        ids.append(r[0])
        camera_aabb(r[7:10],r[10])  # representation overflow is a format error
    if not isinstance(relations,list):raise ValueError('relations must be an array')
    seen=set()
    for r in relations:
        if not isinstance(r,list) or len(r)!=3 or any(not isinstance(v,str) for v in r):raise ValueError('Relation rows are [source_id, kind, target_id]')
        if r[0] not in ids or r[2] not in ids or r[0]==r[2] or r[1] not in RELATIONS:raise ValueError('Unknown/self relation endpoint or unsupported relation kind')
        if tuple(r) in seen:raise ValueError('Duplicate relation row')
        seen.add(tuple(r))
    return deepcopy(value)


def camera_aabb(size,yaw):
    a=math.radians(math.remainder(yaw,360.));c=abs(math.cos(a));s=abs(math.sin(a));w,h,d=size
    result=[c*w+s*d,h,s*w+c*d]
    if not all(numeric(x) and x>0 for x in result):raise ValueError('Dimensions overflow the camera-AABB representation')
    return result


def export_state(answer,scene,backend):
    from .assembly_taxonomy import vocabulary
    vocab=vocabulary();objects=[];mapping=[]
    for row in answer['objects']:
        oid,name,noun,evidence=row[:4];center=row[4:7];intrinsic=row[7:10];yaw=row[10];size=camera_aabb(intrinsic,yaw)
        if not np.isfinite(np.asarray(center)+np.asarray(size)/2).all() or not np.isfinite(np.asarray(center)-np.asarray(size)/2).all():raise ValueError('Coordinates overflow representable box corners')
        resolution=resolve_identity(noun,name);canonical=resolution['category']
        cid=next((k for k,v in CANONICAL_CATEGORIES.items() if v==canonical),None)
        objects.append(dict(track_id=oid,raw_category=noun,description=name,
            category_id=field(cid,'semantic_lexicon_v1',reason='unmapped_raw_noun'),
            center_local_m=field(list(center),'direct_RGB_model_prediction','absolute_camera_metres'),
            size_aabb_local_m=field(size,'deterministic_representation','intrinsic_box_yaw_to_camera_AABB'),
            intrinsic_size_m=field(list(intrinsic),'direct_RGB_model_prediction','full_intrinsic_dimensions'),
            yaw_camera_deg=field(yaw,'direct_RGB_model_prediction','camera_Y_yaw_degrees'),
            bbox2d_visible_xyxy_norm=field(None,reason='not_requested_by_direct_world_contract'),
            in_primary_room=field(None,reason='room_membership_unknown'),local_evidence='visible' if evidence=='visible' else 'unobserved',
            visibility_claim_source='direct_model_not_independently_verified'))
        mapping.append(dict(id=oid,raw_category=noun,resolution=resolution))
    full=graph(scene['id'],scene.get('split','unspecified'),'full',objects,vocab['category_mapping_sha256'])
    full.update(method=KIND,backend=backend,relations=[dict(source=s,kind=k,target=t) for s,k,t in answer['relations']],
        scene_gt_used=False,wilddet_used=False,explicit_causal_model_used=False,layout_solver_used=False,
        geometry_modified_after_model=False,geometry_verified=False,semantics_verified=False,
        output_scope='direct_RGB_world_hypothesis_not_certified_reconstruction')
    return validate_graph(full),mapping


def prepare(root,inputs=DEFAULT_INPUTS,seed=3407,max_tokens=8192,max_objects=128,reference=None):
    if type(max_tokens) is not int or max_tokens<256:raise ValueError('At least 256 output tokens required')
    if type(max_objects) is not int or not 1<=max_objects<=512:raise ValueError('Computation capacity must be 1..512')
    if reference:
        old=read(Path(reference)/'suite.json');records=[dict(id=r['id'],local_rgb=r['image'],local_sha256=r['image_sha256'],hfov_degrees=r['hfov'],split=r.get('split','unspecified')) for r in old['scenes']]
    else:records=rows(inputs)
    if not records:raise ValueError('Empty RGB input list')
    scenes=[]
    for r in records:
        if not re.fullmatch('[A-Za-z0-9_-]+',r['id']):raise ValueError('Unsafe scene ID')
        path=Path(r['local_rgb']).resolve();fov=r['hfov_degrees']
        if not numeric(fov) or not 0<fov<180:raise ValueError('Finite original-camera HFOV required')
        if r.get('local_sha256') and sha(path)!=r['local_sha256']:raise ValueError('Input RGB changed: '+r['id'])
        with Image.open(path) as im:
            im=im.convert('RGB');size=list(im.size);pixels=pixel_sha(im);jpeg=hashlib.sha256(image_bytes(im)).hexdigest()
        scenes.append(dict(id=r['id'],image_source=str(path),source_sha256=sha(path),pixel_sha256=pixels,canonical_image_sha256=jpeg,hfov=fov,image_size=size,split=r.get('split','unspecified')))
    if len({r['id'] for r in scenes})!=len(scenes):raise ValueError('Duplicate scene IDs')
    root=new_output(root);(root/'scenes').mkdir()
    config=dict(kind=KIND,scenes=scenes,seed=seed,max_tokens=max_tokens,max_objects=max_objects,worlds_per_image_backend=1,
        local_model=str(Path(DEFAULT_MODEL).resolve()),api_model='deepseek-ai/DeepSeek-V4.1-Flash',api_base='https://api.siliconflow.cn/v1',generation=recipe(),
        initial_calls_per_image_backend=1,maximum_calls_per_image_backend=2,transport_retries=0,
        reference_suite=str(Path(reference).resolve()) if reference else None,reference_data_used='RGB_ID_HFOV_only' if reference else None,
        wilddet_used=False,observer_stage_used=False,explicit_causal_model_used=False,program_library_used=False,layout_solver_used=False,training_performed=False,scene_GT_used=False,code_sha256=code_hashes())
    write(root/'direct_experiment.json',config)
    for scene in scenes:
        folder=root/'scenes'/scene['id'];folder.mkdir()
        with Image.open(scene['image_source']) as im:im.convert('RGB').save(folder/'input.png')
        write(folder/'scene.json',scene);(folder/'prompt.txt').write_text(prompt_for(scene,max_objects))
    return config


def verify(root):
    config=read(root/'direct_experiment.json')
    if config.get('kind')!=KIND:raise ValueError('Not a direct RGB experiment; original causal runs are read-only')
    changed=[p for p,h in config['code_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    if changed:raise ValueError('Code changed after preparation: '+str(changed))
    for scene in config['scenes']:
        with Image.open(root/'scenes'/scene['id']/'input.png') as im:
            if list(im.size)!=scene['image_size'] or pixel_sha(im)!=scene['pixel_sha256']:raise ValueError('Prepared RGB changed')
        if (root/'scenes'/scene['id']/'prompt.txt').read_text()!=prompt_for(scene,config['max_objects']):raise ValueError('Shared prompt changed')
    return config


def provider(config,backend):
    if backend=='qwen':
        from .world_generation import MatchedLocalJSON
        return MatchedLocalJSON(config['local_model'])
    if backend=='api':
        from .world_api import SiliconFlowJSON
        return SiliconFlowJSON(model=config['api_model'],base=config['api_base'],thinking='off',generation_profile='matched_v1',max_calls=2*len(config['scenes']))
    raise ValueError('Unknown backend')


def generate(root,backend,p=None):
    root=Path(root).resolve();config=verify(root)
    if backend not in ('qwen','api'):raise ValueError('Unknown backend')
    pending=[s for s in config['scenes'] if not (root/'scenes'/s['id']/backend).exists()]
    if pending and p is None:p=provider(config,backend)
    results=[]
    for scene in config['scenes']:
        folder=root/'scenes'/scene['id'];out=folder/backend
        if (out/'summary.json').exists():results.append(read(out/'summary.json'));continue
        if out.exists():
            interrupted=dict(scene=scene['id'],backend=backend,status='failed',error='Interrupted prior attempt retained; no automatic extra model draw',model_calls=None)
            write(out/'summary.json',interrupted);results.append(interrupted);continue
        out.mkdir();(out/'calls').mkdir();summary=dict(scene=scene['id'],backend=backend,status='failed',model_calls=0,
            wilddet_used=False,explicit_causal_model_used=False,layout_solver_used=False)
        try:
            base=(folder/'prompt.txt').read_text();error='';answer=None
            with Image.open(folder/'input.png') as im:image=im.convert('RGB')
            for attempt in range(2):
                path=out/'calls'/('direct_world'+('_repair' if attempt else '')+'.txt');summary['model_calls']+=1
                prompt=base+ ('\nFORMAT CORRECTION: '+error[:1200]+' Return the complete closed objects/relations JSON. No explanation or placeholder values.' if attempt else '')
                try:
                    value=p.ask(image,prompt,path,max_tokens=config['max_tokens'],seed=config['seed']+attempt,profile='completion',sample=True)
                    answer=validate_answer(value,config['max_objects']);full,mapping=export_state(answer,scene,backend);break
                except ValueError as exc:
                    error=str(exc);write(path.with_suffix('.validation.json'),dict(error=error))
                    if attempt:raise
            write(out/'direct_answer.json',answer);write(out/'G_full.json',full);write(out/'mapping.json',mapping)
            hidden=deepcopy(full);hidden.update(scope='hidden',objects=[o for o in hidden['objects'] if o['local_evidence']!='visible']);write(out/'G_out.json',hidden)
            summary.update(status='empty_world' if not full['objects'] else 'schema_valid_world_needs_review',schema_valid=True,
                visible_objects=sum(o['local_evidence']=='visible' for o in full['objects']),hidden_objects=len(hidden['objects']),geometry_optimized=False)
        except Exception as exc:
            summary.update(error=type(exc).__name__+': '+str(exc));(out/'error_traceback.txt').write_text(traceback.format_exc())
        write(out/'summary.json',summary);results.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return dict(status='complete' if all(r['status']!='failed' for r in results) else 'partial_or_failed',results=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','generate','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS);p.add_argument('--reference-suite',type=Path)
    p.add_argument('--backend',choices=('qwen','api'));p.add_argument('--seed',type=int,default=3407);p.add_argument('--max-tokens',type=int,default=8192);p.add_argument('--max-objects',type=int,default=128)
    p.add_argument('--out',type=Path);p.add_argument('--causal-suite',type=Path);a=p.parse_args()
    try:
        if a.action=='prepare':r=prepare(a.root,a.inputs,a.seed,a.max_tokens,a.max_objects,a.reference_suite);print(json.dumps(dict(root=str(a.root),inputs=len(r['scenes']),wilddet_used=False,SCM_used=False)))
        elif a.action=='generate':
            if not a.backend:p.error('--backend required')
            r=generate(a.root,a.backend);print(json.dumps(dict(status=r['status'])));sys.exit(0 if r['status']=='complete' else 1)
        else:
            from .direct_rgb_review import report
            print(report(a.root,a.out,a.causal_suite))
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
