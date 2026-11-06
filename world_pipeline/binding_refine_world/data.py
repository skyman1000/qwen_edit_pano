"""CPU-only source contracts, cache checks, and independent continuation lineage."""
from pathlib import Path
import torch
from safetensors.torch import load_file
from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import DEFAULT_PACK,load_pack
from ..object_world.cache import check_condition,check_posterior
from .supervision import POLICY,sample_binding

VERSION='id_A10000_train_only_binding_refine_v1'
SOURCE=ROOT/'qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-1792820440973994739/checkpoint-step010000'
OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_binding_refine_v1'


def stat(path):
    p=Path(path).resolve();s=p.stat()
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def inputs():
    plan,rows=load_pack(DEFAULT_PACK);old=read(SOURCE/'config.json');marker=read(SOURCE/'COMPLETE.json')
    if (old['group']!='A' or old['binding_weight']!=0 or old['region_weight']!=.25
            or old['category_encoding']!='learned_ID_42_slots' or old['blocks']!=[20,40,59]
            or old['width']!=256 or old['accumulation']!=4 or old['pano_lora_scale']!=.5
            or marker['step']!=10000 or old['pack_plan']!=plan or old['smoke']):
        raise ValueError('Use the exact completed A10000 and unchanged original pack')
    pano=read(Path(plan['base_checkpoint'])/'pano_config.json')
    if any(pano[k]!=plan[k] for k in ('base_snapshot','height','padding_columns','prompt')):
        raise ValueError('Frozen Panorama and cache pack differ')
    if (plan['train_count'],plan['val_count'])!=(7390,923):raise ValueError('Unexpected data split')
    return plan,[r for r in rows if r['split']=='train'],old


def configuration(plan,old,group,smoke=False):
    if group not in ('control','binding'):raise ValueError('Unknown arm')
    return dict(version=VERSION,group=group,smoke=smoke,pack=str(DEFAULT_PACK),
        pack_plan=stat(DEFAULT_PACK/'PLAN.json'),pack_samples=stat(DEFAULT_PACK/'samples.jsonl'),
        source={name:stat(SOURCE/name) for name in ('config.json','COMPLETE.json','world.safetensors','optimizer.pt')},
        source_step=10000,base_checkpoint=plan['base_checkpoint'],base_snapshot=plan['base_snapshot'],
        height=plan['height'],padding_columns=plan['padding_columns'],prompt=plan['prompt'],
        category_encoding=old['category_encoding'],blocks=old['blocks'],width=old['width'],
        binding_block=40,pano_lora_scale=.5,region_weight=.25,
        coverage_weight=.01 if group=='binding' else 0.,concentration_weight=.001 if group=='binding' else 0.,
        binding_ramp_steps=200,policy=POLICY,lr=old['lr'],seed=old['seed'],accumulation=4,
        betas=[.9,.999],weight_decay=1e-5,adam_eps=1e-6,clip_norm=1.,save_every=100,
        forward='exact original ID A: additive fusion, learned global, unrestricted object cross-attention',
        loss='flow+.5cube+.5yaw+.25region+coverage_weight*coverage+concentration_weight*reliable_other_BCE',
        sampling='A full-set EpochSampler continued at absolute micro-index step*4; per-microbatch RNG for both single-GPU arms',
        train_count=7390,val_count=923,caption=False,spatial_routing=False,
        hashes_computed=False,automatic_evaluation=False)


def smoke_rows(rows):
    result=[]
    for row in (max(rows,key=lambda r:len(r['objects'])),rows[0]):
        if row['id'] not in [r['id'] for r in result]:result.append(row)
    return result


def optimizer_step(optimizer):
    values=[float(v['step']) for v in optimizer.state.values()]
    if not values or len(set(values))!=1 or values[0]!=int(values[0]):raise ValueError('Adam steps disagree')
    return int(values[0])


def restore(world,opt,checkpoint):
    world.load_state_dict(load_file(str(checkpoint/'world.safetensors'),device='cpu'),strict=True)
    opt.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True))
    # load_state_dict places moments on their parameter devices.
    expected=read(checkpoint/'COMPLETE.json')['step']
    if optimizer_step(opt)!=expected:raise ValueError('Adam/checkpoint step mismatch')
    if any(p.dtype!=torch.float32 for p in world.parameters()):raise ValueError('World masters must be FP32')
    for state in opt.state.values():
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].dtype!=torch.float32 or not torch.isfinite(state[key]).all():raise ValueError('Invalid Adam moments')
    return expected


def check(root):
    plan,rows,old=inputs()
    missing={key:0 for key in ('condition_cache','posterior_cache','region_cache','object_region_supervision')}
    for index,row in enumerate(rows,1):
        for key in missing:
            path=row[key]['path'] if key=='object_region_supervision' else row[key]
            if not Path(path).is_file():missing[key]+=1
        if index%1000==0:print(dict(stage='CPU_cache_presence',checked=index,total=len(rows)),flush=True)
    if any(missing.values()):raise ValueError('Missing existing caches: '+str(missing))
    representatives=smoke_rows(rows);support=[]
    for row in representatives:
        check_condition(row['condition_cache']);check_posterior(row['posterior_cache'],plan['height'])
        labels,meta=sample_binding(row,plan['height'],plan['padding_columns'],0)
        if not meta['positive_pairs'] or not meta['negative_pairs']:raise ValueError('Smoke needs both positive and reliable negative labels')
        support.append(dict(id=row['id'],objects=len(row['objects']),**meta))
    state=load_file(str(SOURCE/'world.safetensors'),device='cpu')
    if not all(torch.isfinite(t).all() for t in state.values()):raise ValueError('Nonfinite source World')
    opt=torch.load(SOURCE/'optimizer.pt',map_location='cpu',weights_only=True)
    if {float(v['step']) for v in opt['state'].values()}!={10000.}:raise ValueError('Invalid source Adam steps')
    if not all(torch.isfinite(v[k]).all() for v in opt['state'].values() for k in ('exp_avg','exp_avg_sq')):
        raise ValueError('Nonfinite source Adam')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    result=dict(version=VERSION,train=len(rows),val=plan['val_count'],missing=missing,smoke_support=support,
        source_optimizer_step=10000,condition='original fixed instruction + Local only',
        source=configuration(plan,old,'control')['source'],hashes_computed=False,GPU=False,
        note='All cache paths present; representative headers/support checked, not full polished pixel certification')
    atomic_json(root/'CPU_CHECK.json',result);print(result,flush=True);return result
