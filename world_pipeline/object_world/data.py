"""Minimal completed records; no legacy graph conversion or content hashing."""
from pathlib import Path
import json

from ..build_polished_world import ROOT, FRAME, read, read_rows, validate_objects

DEFAULT_DATA = ROOT/'qwen_edit_pano/data/world_polished_repair_v2'
DEFAULT_PACK = ROOT/'qwen_edit_pano/data/object_world_polished_v1'
DEFAULT_BASE = ROOT/'qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch015'


def identity(row):
    return dict(id=row['id'], split=row['split'], scene_id=row['scene_id'], source_view_id=row['source_view_id'],
        local_condition={k:row['local_condition'][k] for k in ('archive','member','face_index')},
        image={k:row['image'][k] for k in ('arrow_file','batch_index','row_index')},
        target_alignment={k:row['target_alignment'][k] for k in ('mirror','roll_fraction')})


def dataset(root):
    root=Path(root)
    result=read(root/'BUILD_RESULT.json')
    if not result['all_candidates_built'] or result['failures'] or result['missing']:
        raise ValueError('Dataset build is incomplete')
    names=read(root/'category_names.json')
    records=[row for split in ('train','val') for row in read_rows(root/f'{split}.jsonl')]
    ids=set();houses={}
    for row in records:
        if row['id'] in ids or houses.setdefault(row['scene_id'],row['split'])!=row['split']:
            raise ValueError('Duplicate ID or train/val building overlap')
        ids.add(row['id'])
        if row['frame']!=FRAME or row['local_condition']['face_index']!=2:
            raise ValueError('Expected single-face2 camera-local objects')
        validate_objects(row['objects'],names)
        if not Path(row['object_region_supervision']['path']).is_file():
            raise ValueError('Missing instance supervision: '+row['id'])
    counts={split:sum(row['split']==split for row in records) for split in ('train','val')}
    if counts!=result['counts'] or len(records)!=result['completed']:
        raise ValueError('Successful manifests disagree with BUILD_RESULT')
    return records,names


def tensorize(row, max_objects=256):
    import torch
    if len(row['objects'])>max_objects:
        raise ValueError('Object capacity exceeded; truncation is forbidden')
    # Dynamic padding: slot zero is global, all other slots are actual objects.
    count=len(row['objects'])+1
    category=torch.zeros(1,count,dtype=torch.long)
    geometry=torch.zeros(1,count,8)
    mask=torch.ones(1,count,dtype=torch.bool)
    for index,obj in enumerate(row['objects'],1):
        category[0,index]=obj['category_id']
        for offset,key,valid in ((0,'center_local_m','center_valid'),(3,'size_aabb_local_m','size_valid')):
            if obj[valid]:
                geometry[0,index,offset:offset+3]=torch.tensor(obj[key])
                geometry[0,index,6+offset//3]=1
    return category,geometry,mask


def target_image(row):
    from qwen_pano.cached_data import ArrowImages
    from qwen_edit_pano.pair_alignment import apply_alignment
    with ArrowImages().open(row['image']) as image:
        if image.width!=2*image.height:
            raise ValueError('Expected polished ERP target')
        return apply_alignment(image.convert('RGB'),row['target_alignment'])


def load_pack(root):
    root=Path(root).resolve()
    plan=read(root/'PLAN.json')
    if plan['version']!='object_world_pack_v1':
        raise ValueError('Wrong training pack version')
    records=read_rows(root/'samples.jsonl')
    if [identity(row) for row in records]!=plan['sources']:
        raise ValueError('Pack record identities differ from PLAN')
    names=plan['names'];houses={};ids=set()
    for row in records:
        if row['split'] not in ('train','val'):
            raise ValueError('Unknown split in training pack')
        if row['id'] in ids or houses.setdefault(row['scene_id'],row['split'])!=row['split']:
            raise ValueError('Duplicate ID or building split overlap in pack')
        ids.add(row['id']);validate_objects(row['objects'],names)
        if row['frame']!=FRAME or len(row['objects'])>plan['max_objects']:
            raise ValueError('Pack geometry frame or capacity mismatch')
        if row['local_condition']['face_index']!=2:
            raise ValueError('Training pack requires a single face2 Local image')
    counts={split:sum(row['split']==split for row in records) for split in ('train','val')}
    if counts!={'train':plan['train_count'],'val':plan['val_count']}:
        raise ValueError('Training pack split counts differ from PLAN')
    return plan,records


def atomic_tensors(path, values):
    import torch
    from safetensors.torch import save_file
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.safetensors')
    values={k:v.detach().cpu().contiguous() for k,v in values.items()}
    if not all(torch.isfinite(v).all() for v in values.values()):
        raise FloatingPointError('Refusing to save nonfinite tensors: '+str(path))
    save_file(values,str(tmp));tmp.replace(path)


def write_rows(path, rows):
    path=Path(path);tmp=path.with_suffix('.tmp.jsonl')
    tmp.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows));tmp.replace(path)
