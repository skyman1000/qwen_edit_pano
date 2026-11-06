"""Existing pack/caches and an explicit 128-image train-only panel; no hashes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import DEFAULT_PACK,load_pack,atomic_tensors
from ..object_world.cache import check_condition,check_posterior,semantic_phrases
from . import supervision

DEFAULT_PANEL=ROOT/'research/audits/spatial_generation_results_2026-10-26/TRAIN128.json'
DEFAULT_SUPPORT=ROOT/'qwen_edit_pano/data/grounded_world_v2_support'
DEFAULT_ROOT=ROOT/'qwen_edit_pano/outputs/object_world_grounded_v2'
EVAL_PANEL=ROOT/'research/audits/id_condition_diagnostic_2026-10-25/PANEL.json'
AUDIT=ROOT/'research/audits/grounded_world_v2_2026-10-26'


def contract(plan):return {k:v for k,v in plan.items() if k!='sources'}


def smoke_rows(rows):
    largest=max(rows,key=lambda r:len(r['objects']))
    first=next(r for r in rows if r['id']!=largest['id'])
    return [first,largest]


def inputs(pack=DEFAULT_PACK,panel=DEFAULT_PANEL):
    plan,rows=load_pack(pack);selection=read(panel);ids=selection['ids'];byid={r['id']:r for r in rows}
    if len(ids)!=128 or len(set(ids))!=128 or any(s not in byid or byid[s]['split']!='train' for s in ids):
        raise ValueError('Use the explicit 128 unique training IDs, never validation images')
    features=load_file(str(Path(pack)/'class_features.safetensors'),device='cpu')['features']
    meta=read(Path(pack)/'class_features.json')
    expected=dict(policy=plan['semantic_policy'],names=plan['names'],phrases=semantic_phrases(plan['names']),
        base_snapshot=plan['base_snapshot'],pooling='mean of name tokens only; padding/prefix excluded')
    if meta!=expected or features.shape!=(42,3584) or not torch.isfinite(features).all():raise ValueError('Semantic cache mismatch')
    if any(features[int(k)].norm()==0 for k in plan['names']):raise ValueError('Missing class semantics')
    return plan,rows,[byid[s] for s in ids],features


def source_meta(row,plan):
    path=Path(row['object_region_supervision']['path']);stat=path.stat()
    return dict(id=row['id'],objects=row['objects'],label_source=str(path),source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
        height=plan['height'],padding=plan['padding_columns'],max_objects=plan['max_objects'],policy=supervision.POLICY)


def support_path(root,sid):return Path(root)/(sid+'.safetensors')


def ensure_support(row,plan,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=support_path(root,row['id']);meta_path=path.with_suffix('.json')
    expected=source_meta(row,plan)
    if path.is_file() and meta_path.is_file():
        meta=read(meta_path)
        if meta['source']!=expected:raise ValueError('Stale reachable-label cache; use a separate support directory')
        return meta['summary']
    items,summary=supervision.prepare(row,plan['height'],plan['padding_columns'],plan['max_objects'])
    offsets=np.cumsum([0]+[len(r['queries']) for r in items])
    arrays=dict(offsets=torch.tensor(offsets,dtype=torch.int64),slots=torch.tensor([r['slot'] for r in items],dtype=torch.int64),
        instance_ids=torch.tensor([r['instance_id'] for r in items],dtype=torch.int64),
        retention=torch.tensor([r['retention'] for r in items],dtype=torch.float64),
        queries=torch.tensor(np.concatenate([r['queries'] for r in items]) if items else [],dtype=torch.int64),
        probability=torch.tensor(np.concatenate([r['probability'] for r in items]) if items else [],dtype=torch.float64))
    atomic_tensors(path,arrays);atomic_json(meta_path,dict(source=expected,summary=summary))
    return summary


def load_support(root,sid):
    arrays=load_file(str(support_path(root,sid)),device='cpu');offsets=arrays['offsets'].tolist();items=[]
    if len(offsets)!=len(arrays['slots'])+1:raise ValueError('Invalid sparse label offsets')
    for i,slot in enumerate(arrays['slots'].tolist()):
        a,b=offsets[i:i+2];p=arrays['probability'][a:b].numpy();queries=arrays['queries'][a:b].numpy()
        if a>=b or not np.isfinite(p).all() or (p<=0).any() or not np.isclose(p.sum(),1):raise ValueError('Invalid sparse probabilities')
        items.append(dict(slot=slot,instance_id=int(arrays['instance_ids'][i]),queries=queries,probability=p,retention=float(arrays['retention'][i])))
    return items


def check_rows(rows,plan):
    for row in rows:
        check_condition(row['condition_cache']);check_posterior(row['posterior_cache'],plan['height'])
        weights=load_file(row['region_cache'],device='cpu')['weights']
        if weights.shape!=(plan['height']//8,plan['height']//4) or not torch.isfinite(weights).all() or (weights<0).any():
            raise ValueError('Invalid existing region weights')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pack',type=Path,default=DEFAULT_PACK)
    p.add_argument('--panel',type=Path,default=DEFAULT_PANEL);p.add_argument('--support',type=Path,default=DEFAULT_SUPPORT)
    args=p.parse_args();torch.set_num_threads(4);plan,_,rows,features=inputs(args.pack,args.panel);check_rows(rows,plan)
    summaries={}
    for i,row in enumerate(rows,1):
        summaries[row['id']]=ensure_support(row,plan,args.support)
        if i%16==0:print(dict(stage='cpu_reachable_labels',completed=i,total=len(rows)),flush=True)
    if not all(summaries[r['id']]['eligible_instances'] for r in smoke_rows(rows)):raise ValueError('Smoke scenes require reachable object labels')
    from .runtime import WorldBranch
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json')
    model=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],features,height=plan['height'],padding=plan['padding_columns'])
    result=dict(train=len(rows),buildings=len({r['scene_id'] for r in rows}),class_features_shape=list(features.shape),
        trainable_parameters=sum(p.numel() for p in model.parameters()),old_world_loaded=False,base_trainable=0,
        eligible_instances=sum(v['eligible_instances'] for v in summaries.values()),support_cache=str(args.support),
        filtered_reasons={reason:sum(sum(r['reason']==reason for r in v['filtered']) for v in summaries.values())
            for reason in ('generic_category','too_small_or_no_valid_support','insufficient_reachable_support')},
        all_caches_present=True,cpu_only=True,hashes_computed=False,rows=summaries)
    result['smoke_ids']=[r['id'] for r in smoke_rows(rows)]
    result['smoke_maximum_objects']=max(len(r['objects']) for r in smoke_rows(rows))
    AUDIT.mkdir(parents=True,exist_ok=True);atomic_json(AUDIT/'PREFLIGHT.json',result)
    atomic_json(args.support/'PREPARED.json',{k:v for k,v in result.items() if k!='rows'})
    print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
