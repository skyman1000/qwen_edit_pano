"""Fresh, backend-independent RGB -> 2D inventory contract."""
from copy import deepcopy
from pathlib import Path
import math

from PIL import Image,ImageDraw

from ..common import ROOT,field,read,sha,validate_graph,write
from .category_bridge import category
from .geometry import CANONICAL_CATEGORIES,graph
from .scene_program import text
from .world_scm_runner import ask_valid

PROMPT=('Inspect this partial indoor RGB photograph. Return the VISIBLE object instances, not unseen surroundings. '
    'Include furniture and identifiable everyday belongings/decor. Treat each separate instance separately; '
    'do not list an object again under its part, synonym or another overlapping label. Do not add floor/wall/ceiling surfaces. '
    'Return ONE JSON object {"objects":[{"name":"specific noun","category":"ordinary English noun",'
    '"bbox_1000":[x1,y1,x2,y2],"on_floor":true}],"evidence_description":"joint description of visible objects and their relationships"}. '
    'bbox_1000 uses the VISIBLE image rectangle normalized to 0..1000, x right/y down, not pixel coordinates. '
    '0<=x1<x2<=1000 and 0<=y1<y2<=1000. Partly cropped objects get only the visible extent. '
    'on_floor means the whole object rests directly on the floor, excluding table/shelf-supported objects. '
    'No 3D coordinates, hidden objects, room redesign or claims about objects absent outside the photograph. '
    'Use 0..64 records as a computation limit, without a target count. No restricted Adapter vocabulary.')


def answer(value):
    rows=value.get('objects')
    if not isinstance(rows,list) or len(rows)>64:raise ValueError('objects must contain 0..64 records')
    result=[]
    for i,r in enumerate(rows):
        if not isinstance(r,dict):raise ValueError(f'objects[{i}] must be an object')
        b=r.get('bbox_1000')
        if (not isinstance(b,list) or len(b)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b)
                or not 0<=b[0]<b[2]<=1000 or not 0<=b[1]<b[3]<=1000):
            raise ValueError(f'objects[{i}].bbox_1000 requires four ordered finite numbers in [0,1000]')
        if type(r.get('on_floor')) is not bool:raise ValueError(f'objects[{i}].on_floor must be boolean')
        result.append(dict(name=text(r.get('name'),'name'),category=text(r.get('category'),'category'),
            bbox_1000=b,on_floor=r['on_floor']))
    return dict(objects=result,evidence_description=text(value.get('evidence_description'),'evidence_description'))


def state_from_answer(value,experiment,backend):
    vocab=read(ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json')
    objects=[];mapping=[]
    for i,r in enumerate(value['objects']):
        raw=r['category'];cid=None;audit=None
        try:
            canonical,audit=category(raw);cid=next(k for k,v in CANONICAL_CATEGORIES.items() if v==canonical)
        except ValueError as exc:
            audit=dict(unmapped_reason=str(exc),object_preserved=True)
        oid=f'obs_{i:03d}'
        objects.append(dict(track_id=oid,category_id=field(cid,'noun_category_bridge',reason='unmapped_raw_label'),
            raw_category=raw,description=r['name'],center_local_m=field(None,reason='not_predicted_yet'),
            size_aabb_local_m=field(None,reason='not_predicted_yet'),
            bbox2d_visible_xyxy_norm=field([v/1000 for v in r['bbox_1000']],'model_prediction','2d_grounding'),
            in_primary_room=field(None,reason='room_membership_not_inferred'),local_evidence='visible'))
        mapping.append(dict(id=oid,raw_label=raw,category_id=cid,mapping=audit))
    state=graph(experiment['id'],'validation','observed',objects,vocab['category_mapping_sha256'])
    state.update(observer=backend+'_fresh_2d',scene_gt_used=False)
    return validate_graph(state),mapping


def observe(provider,root,backend,max_tokens=8192):
    config=read(root/'experiment.json');out=root/backend/'observer2d';out.mkdir(parents=True,exist_ok=False)
    with Image.open(root/'input.png') as im:image=im.convert('RGB')
    try:
        value=ask_valid(provider,image,PROMPT,out/'answer.txt',answer,max_tokens)
        state,mapping=state_from_answer(value,config,backend)
        write(out/'G_2d.json',state);write(out/'labels.json',mapping)
        write(out/'grounding.json',dict(instances=[dict(id=o['track_id'],variant=r['name'],on_floor=r['on_floor'])
            for o,r in zip(state['objects'],value['objects'])],evidence_description=value['evidence_description']))
        canvas=image.copy();canvas.thumbnail((1200,1200));draw=ImageDraw.Draw(canvas)
        for obj in state['objects']:
            b=obj['bbox2d_visible_xyxy_norm']['value'];rect=[b[0]*canvas.width,b[1]*canvas.height,b[2]*canvas.width,b[3]*canvas.height]
            draw.rectangle(rect,outline='red',width=2);draw.text(tuple(rect[:2]),obj['track_id']+' '+obj['raw_category'],fill='red')
        canvas.save(out/'boxes2d.png')
        summary=dict(status='completed',objects=len(state['objects']),backend=backend,image_sha256=sha(root/'input.png'),
            unmapped=sum(not o['category_id']['valid'] for o in state['objects']),old_observations_reused=False)
        write(out/'summary.json',summary);return summary
    except Exception as exc:
        write(out/'summary.json',dict(status='failed',error=type(exc).__name__+': '+str(exc)));raise


def load_case(root,backend):
    config=read(root/'experiment.json');state_path=root/backend/'observer3d'/'G_obs.json'
    state=validate_graph(read(state_path));context=[]
    for o in state['objects']:
        if not o['center_local_m']['valid'] or not o['size_aabb_local_m']['valid']:
            raise ValueError('Every retained 2D instance needs a WildDet3D result; no silent dropping')
        context.append(dict(id=o['track_id'],category=o.get('raw_category') or CANONICAL_CATEGORIES.get(o['category_id']['value'],'unknown'),
            description=o.get('description',''),visible_bbox_xyxy_norm=o['bbox2d_visible_xyxy_norm']['value']))
    return dict(id=config['id'],image=str(root/'input.png'),state=str(state_path),hfov=config['hfov'],
        image_size=config['image_size'],context=context,
        source_sha256={str(root/'input.png'):sha(root/'input.png'),str(state_path):sha(state_path)})
