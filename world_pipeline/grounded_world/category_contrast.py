"""One offscreen category mismatch; labels are losses, never generator inputs."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import tensorize
from ..spatial_world.controls import fully_outside
from .data import source_meta,load_support,DEFAULT_SUPPORT,inputs,check_rows
from .train import configuration as pilot_configuration

EXPERIMENT='grounded_category_contrast_continuation_v1'
INITIAL=ROOT/'qwen_edit_pano/outputs/object_world_grounded_v2/pilot/B/run-1793029346114145904/checkpoint-step000200'
OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_category_contrast_v1'
TARGETS=ROOT/'qwen_edit_pano/data/grounded_category_targets_v1.json'
POLICY=dict(version='single_supported_offscreen_category_pair_v1',minimum_pixels=256,minimum_packed_cells=4,
    excluded_categories=[34,36,37,39,40],ambiguous_groups=[[3,19,34,36],[7,13,31,36],[9,12,32],[5,26]],
    selection='uniform eligible category then uniform instance within the current image; independent NumPy RNG',
    sigma_range=[.5,.9],temperature=.25,weight=.01,ramp_new_steps=50,maximum_pairs_per_optimizer_step=1,
    target='same original GT velocity and instance support; negative condition is not a reconstruction-positive pair',
    negative='one legal distinct category only; same Local/geometry/mask/other objects/noise/t',
    inference_GT_masks=False,score='exp(-instance velocity MSE / temperature)',loss='softplus(score_negative-score_positive)',
    normalization='one auxiliary pair per eligible optimizer step, outside the four-main-microbatch average')


def source_contract(plan,rows,source=INITIAL):
    from types import SimpleNamespace
    from .data import DEFAULT_PACK,DEFAULT_PANEL
    from .runtime import VERSION
    c=read(source/'config.json');marker=read(source/'COMPLETE.json')
    expected=pilot_configuration(plan,SimpleNamespace(pack=DEFAULT_PACK,panel=DEFAULT_PANEL,support=DEFAULT_SUPPORT,
        group='B',smoke=False,seed=0),rows)
    if c!=expected or marker!=dict(version=VERSION,step=200):raise ValueError('Use the bound, unchanged real B200 checkpoint/configuration')
    if not all((source/n).is_file() for n in ('world.safetensors','optimizer.pt')):raise ValueError('Missing source weights/optimizer')
    return c


def target_entries(row,items):
    _,geometry,_=tensorize(row);outside=fully_outside(geometry)[0]
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as f:
        instance=f['instance'];valid=f['object_loss_valid']
    result=[]
    for item in items:
        slot=item['slot'];obj=row['objects'][slot-1]
        if obj['category_id'] in POLICY['excluded_categories'] or not bool(outside[slot]):continue
        pixels=int(((instance==int(obj['instance_id']))&valid).sum())
        if pixels<POLICY['minimum_pixels'] or len(item['queries'])<POLICY['minimum_packed_cells']:continue
        result.append(dict(slot=slot,instance_id=obj['instance_id'],category_id=obj['category_id'],pixels=pixels,packed_cells=len(item['queries'])))
    return result


def prepare(path=TARGETS):
    plan,_,rows,_=inputs();source_contract(plan,rows);check_rows(rows,plan)
    sources={r['id']:source_meta(r,plan) for r in rows}
    for row in rows:
        if read((DEFAULT_SUPPORT/(row['id']+'.json')))['source']!=sources[row['id']]:
            raise ValueError('Existing sparse binding labels no longer match input/GT source')
    expected=dict(experiment=EXPERIMENT,policy=POLICY,sources=sources)
    if Path(path).is_file():
        saved=read(path)
        if {k:saved[k] for k in expected}!=expected:raise ValueError('Category target policy/source changed; use a separate target file')
        return saved
    targets={r['id']:target_entries(r,load_support(DEFAULT_SUPPORT,r['id'])) for r in rows}
    if not any(targets.values()):raise ValueError('No eligible training targets')
    result=dict(**expected,targets=targets,images=len(rows),eligible_images=sum(bool(v) for v in targets.values()),
        eligible_instances=sum(len(v) for v in targets.values()),eligible_TVs=sum(t['category_id']==22 for v in targets.values() for t in v),
        cpu_only=True,hashes_computed=False,GT_masks_enter_forward=False)
    Path(path).parent.mkdir(parents=True,exist_ok=True);atomic_json(path,result);return result


def select_target(entries,names,seed):
    if not entries:return None
    rng=np.random.default_rng(int(seed));categories=sorted({e['category_id'] for e in entries})
    category=int(rng.choice(categories));same=[e for e in entries if e['category_id']==category]
    item=dict(same[int(rng.integers(len(same)))]);excluded=set(POLICY['excluded_categories'])|{category}
    for group in POLICY['ambiguous_groups']:
        if category in group:excluded.update(group)
    alternatives=sorted(int(k) for k in names if int(k) not in excluded)
    if not alternatives:raise ValueError('No non-overlapping legal category alternative')
    item['wrong_category_id']=int(rng.choice(alternatives));return item


def negative_state(state,entry):
    cat,geometry,mask=state;slot=entry['slot']
    if slot<1 or slot>=cat.shape[1] or not bool(mask[0,slot]) or int(cat[0,slot])!=entry['category_id']:
        raise ValueError('Category target slot no longer matches actual input')
    if entry['wrong_category_id']==entry['category_id']:raise ValueError('Negative category must differ')
    changed=cat.clone();changed[0,slot]=entry['wrong_category_id']
    return changed,geometry,mask


def instance_weights(row,entry,height):
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as f:
        mask=(f['instance']==int(entry['instance_id']))&f['object_loss_valid']
    if int(mask.sum())!=entry['pixels']:raise ValueError('Instance support changed')
    w=F.interpolate(torch.tensor(mask,dtype=torch.float32)[None,None],size=(height//8,height//4),mode='area')[0,0]
    if not w.sum()>0:raise ValueError('Empty target support')
    return w/w.sum()


def regional_error(prediction,target,weights):
    if prediction.shape!=target.shape or prediction.ndim!=5 or prediction.shape[0]!=1 or weights.shape!=prediction.shape[-2:]:
        raise ValueError('Category loss expects one unpadded ERP prediction/target/support')
    error=(prediction.float()-target.float()).square().mean((0,1,2))
    return (error*weights.float()).sum()


def objective(positive,negative,tau=POLICY['temperature']):
    if tau<=0:raise ValueError('Temperature must be positive')
    return F.softplus(torch.exp(-negative.float()/tau)-torch.exp(-positive.float()/tau))


@torch.no_grad()
def coefficients(positive,negative,tau=POLICY['temperature']):
    """Exact chain-rule coefficients for serial positive/negative backward."""
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all() or positive<0 or negative<0 or tau<=0:
        raise ValueError('Invalid contrast errors/temperature')
    a=torch.exp(-positive.float()/tau);b=torch.exp(-negative.float()/tau);factor=(b-a).sigmoid()
    return objective(positive,negative,tau),factor*a/tau,-factor*b/tau


def weight(step,smoke=False):
    return POLICY['weight']*(1. if smoke else min(1.,(step-200+1)/POLICY['ramp_new_steps']))
