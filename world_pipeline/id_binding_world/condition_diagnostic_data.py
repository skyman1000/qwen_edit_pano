"""CPU panel selection and immutable condition permutations for diagnostics."""
import random
import re
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import identity
from ..object_world.evaluation import object_report

PANEL_VERSION='id_world_fixed_noise_panel_v3'
MODES=('correct','repeat_correct','wrong_category','wrong_position','world_off')
SID_PATTERN=re.compile(r'^[A-Za-z0-9]{11}_[0-9a-f]{32}$')
SALIENT={3,5,7,10,11,14,18,19,21,22,23,25,26,31,34,37}


def history_ids(root=ROOT):
    """Bounded provenance scan, excluding full training-pack source lists/caches.

    Prior panorama-only benchmark exposure is recorded separately; it does not
    disqualify a NEW World diagnostic. No claim of untouched final-test status.
    """
    world={};panorama={}
    def add(target,value,path):
        if isinstance(value,str) and SID_PATTERN.fullmatch(value):target.setdefault(value,[]).append(str(path))
    def config_ids(data,target,path):
        if not isinstance(data,dict):return
        add(target,data.get('id'),path)
        for value in data.get('ids',[]) if isinstance(data.get('ids'),list) else []:add(target,value,path)
        for key in ('samples','cases'):
            for value in data.get(key,[]) if isinstance(data.get(key),list) else []:
                if isinstance(value,dict):add(target,value.get('id'),path)
        if isinstance(data.get('plan'),dict):config_ids(data['plan'],target,path)
    output_root=root/'qwen_edit_pano/outputs'
    for directory in sorted(output_root.iterdir()):
        if not directory.is_dir():continue
        is_world=any(s in directory.name for s in ('world','oracle','fourier','state_probe','gated','gt_text','scene_box','panorama_qwen_probe'))
        target=world if is_world else panorama
        for filename in ('run_config.json','PLAN.json','diagnostic_panel.json','plan.json'):
            p=directory/filename
            if p.is_file():config_ids(read(p),target,p)
        stack=[(directory,0)]
        while stack:
            current,depth=stack.pop()
            if depth:
                for filename in ('run_config.json','PLAN.json','diagnostic_panel.json','plan.json'):
                    p=current/filename
                    if p.is_file():config_ids(read(p),target,p)
            for child in current.iterdir():
                if SID_PATTERN.fullmatch(child.stem):add(target,child.stem,child)
                if child.is_dir() and depth<(4 if is_world else 2) and not any(s in child.name.lower() for s in
                    ('checkpoint','cache','posterior','condition','regions','target_post','latent','model')):
                    stack.append((child,depth+1))
    for p in sorted((root/'research/audits').glob('*/SAMPLES.json')):config_ids(read(p),world,p)
    return world,panorama


def checkpoint_exposures(checkpoint):
    """Follow saved resume ancestry and discard all unsaved/replayed log tails."""
    checkpoint=Path(checkpoint);step=read(checkpoint/'COMPLETE.json')['step'];run=checkpoint.parent
    resume=read(run/'run_plan.json').get('resume')
    counts=Counter();start=0
    if resume:
        parent=Path(resume);counts.update(checkpoint_exposures(parent));start=read(parent/'COMPLETE.json')['step']
    observed=[]
    with (run/'train_log.jsonl').open() as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line)
            if start<row['step']<=step:
                observed.append(row['step']);counts.update(row['sample_ids'])
    if observed!=list(range(start+1,step+1)):raise ValueError('Saved checkpoint ancestry has incomplete optimization logs')
    return counts


def ensure_panel(path,plan,records,train_count=4,val_count=6,seed=20261025,seen_train_ids=None):
    if path.exists():
        panel=read(path)
        if (panel['version']!=PANEL_VERSION or panel['train_count']!=train_count or panel['val_count']!=val_count
            or panel['seed']!=seed):raise ValueError('Existing panel settings differ; choose another --panel')
        byid={r['id']:r for r in records}
        for case in panel['cases']:
            if identity(byid[case['id']])!=case['source']:raise ValueError('Frozen panel image identity changed')
        return panel
    old_world,old_panorama=history_ids()
    selected=[];rng=random.Random(seed)
    for split,count in [('train',train_count),('val',val_count)]:
        candidates=[r for r in records if r['split']==split and (split!='val' or r['id'] not in old_world)
                    and (split!='train' or seen_train_ids is None or r['id'] in seen_train_ids)]
        rng.shuffle(candidates);houses=set();chosen=[]
        for row in candidates:
            if row['scene_id'] in houses:continue
            if len({o['category_id'] for o in row['objects']})<3:continue
            with np.load(row['object_region_supervision']['path'],allow_pickle=False) as masks:
                report=object_report(row,plan['names'],masks['instance'],masks['instance_valid'],masks['object_loss_valid'])
            eligible=sorted((o for o in report['objects'] if o['eligible']),
                key=lambda o:(o['category_id'] not in SALIENT,-o['outside_local_fraction'],-o['sphere_fraction'],o['instance_id']))[:6]
            salient=[o for o in eligible if o['category_id'] in SALIENT]
            if (len(salient)<3 or len({o['category_id'] for o in salient})<2 or
                not any(o['outside_local_fraction']>=.5 for o in salient)):continue
            chosen.append(dict(id=row['id'],split=split,scene_id=row['scene_id'],source=identity(row),
                latent_seed=seed+10007*len(selected)+10007*len(chosen),objects=eligible,
                prior_world_diagnostic=bool(row['id'] in old_world),
                prior_panorama_only_outputs=old_panorama.get(row['id'],[])[:8]))
            houses.add(row['scene_id'])
            if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError(f'Need {count} distinct-house {split} diagnostic images; found {len(chosen)}')
        selected.extend(chosen)
    panel=dict(version=PANEL_VERSION,train_count=train_count,val_count=val_count,seed=seed,cases=selected,
        scope='New World diagnostics, not a new final test split. Prior panorama-only output exposure is recorded.',
        selection='seeded shuffle; train exposed by earliest checkpoint; distinct houses per split; >=3 eligible salient objects, >=2 categories, >=1 mostly outside Local; no outcome selection',
        history_excluded_val_ids=sorted(set(old_world)&{r['id'] for r in records if r['split']=='val'}),
        hashes_computed=False)
    path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,panel)
    return panel


def condition_variant(tensors,mode):
    """Wrong controls preserve category histogram / center-coordinate multiset."""
    import torch
    if mode not in MODES:raise ValueError('Unknown diagnostic condition')
    if mode=='world_off':return None,dict(operation='disable World')
    category,geometry,mask=(t.clone() for t in tensors)
    metadata=dict(operation=mode,changed_objects=0)
    if mode=='wrong_category':
        slots=torch.nonzero(mask[0],as_tuple=False).flatten();slots=slots[slots>0]
        old=category[0,slots].clone()
        if len(old)<2 or len(old.unique())<2:raise ValueError('Wrong-category control needs distinct categories')
        scores=[int((old!=old.roll(i)).sum()) for i in range(1,len(old))]
        shift=1+scores.index(max(scores));category[0,slots]=old.roll(shift)
        metadata.update(shift=shift,changed_objects=max(scores),total_objects=len(slots),category_histogram_preserved=True,
            changed_slots=slots[old!=old.roll(shift)].cpu().tolist())
    elif mode=='wrong_position':
        slots=torch.nonzero(mask[0]&(geometry[0,:,6]>.5),as_tuple=False).flatten();slots=slots[slots>0]
        if len(slots)<2:raise ValueError('Wrong-position control needs two valid centers')
        old=geometry[0,slots,:3].clone();shift=max(1,len(slots)//2)
        geometry[0,slots,:3]=old.roll(shift,0)
        distance=(old-geometry[0,slots,:3]).float().norm(dim=-1)
        if not bool((distance>1e-6).any()):raise ValueError('Position perturbation is a no-op')
        metadata.update(shift=shift,changed_objects=int((distance>1e-6).sum()),
            mean_center_displacement_m=float(distance.mean()),centers_multiset_preserved=True,sizes_fixed=True,
            changed_slots=slots[distance>1e-6].cpu().tolist())
    return (category,geometry,mask),metadata


def case_support(row,case,height,padding):
    """Fixed selected-instance latent weights and fractional-support query pairs."""
    import torch
    from torch.nn import functional as F
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as arrays:
        instance=arrays['instance'];valid=arrays['object_loss_valid']
    th,tw=height//16,height//8
    if instance.shape[0]%th or instance.shape[1]%tw:raise ValueError('Invalid instance/token grid')
    fy,fx=instance.shape[0]//th,instance.shape[1]//tw
    positions=[];slots=[];latents=[];objects=[]
    lookup={o['instance_id']:(i,o) for i,o in enumerate(row['objects'],1)}
    for entry in case['objects']:
        iid=entry['instance_id'];slot,obj=lookup[iid];support=(instance==int(iid))&valid
        if not support.any():raise ValueError('Selected instance lost all support')
        mass=support.reshape(th,fy,tw,fx).sum((1,3)).astype(float)
        picks=np.argsort(-mass.ravel(),kind='stable')[:8];picks=picks[mass.ravel()[picks]>0]
        y,x=np.divmod(picks,tw);positions.extend((y*(tw+2*padding)+x+padding).tolist());slots.extend([slot]*len(picks))
        a=F.interpolate(torch.tensor(support,dtype=torch.float32)[None,None],size=(height//8,height//4),mode='area')[0,0]
        latents.append(a/a.sum());objects.append(dict(instance_id=iid,category_id=obj['category_id'],slot=slot,
            name=entry['name'],outside_local_fraction=entry['outside_local_fraction']))
    # Uniform grid probes include both support and background for variation tests.
    uniform=[int(y)*(tw+2*padding)+int(x)+padding for y in np.linspace(0,th-1,8).astype(int) for x in np.linspace(0,tw-1,8).astype(int)]
    queries=sorted(set(positions+uniform));lookup_q={q:i for i,q in enumerate(queries)}
    pairs=[(lookup_q[q],slot) for q,slot in zip(positions,slots)]
    return torch.stack(latents),dict(queries=queries,pairs=pairs,objects=objects,grid=[th,tw],padding=padding)
