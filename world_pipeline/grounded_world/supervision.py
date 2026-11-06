"""Sparse, reachable train-only labels, not generator conditioning."""
import numpy as np
import torch
from ..spatial_world.projection import erp_support
from ..object_world.data import tensorize

POLICY=dict(version='reachable_instance_coverage_v2',minimum_supported_pixels=64,
    excluded_categories=[39,40],minimum_reachable_fraction=.5,maximum_instances=32,queries_per_instance=8,
    labels='existing mesh instance + object_loss_valid; no edited GT targets',
    normalization='equal weight per selected instance, queries sampled by fractional reachable support',
    candidates='input AABB projection only, computed identically on CPU in train/inference',
    gt_masks_enter_forward=False)


def prepare(row,height,padding,max_objects=256):
    _,geometry,mask=tensorize(row,max_objects)
    routing,meta=erp_support(geometry,mask,height//16,height//8,padding)
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as f:
        instance=f['instance'];valid=f['object_loss_valid'];hit=f['instance_valid']
    th,tw=height//16,height//8
    if (instance.ndim!=2 or valid.shape!=instance.shape or hit.shape!=instance.shape
            or valid.dtype!=bool or hit.dtype!=bool or (valid&~hit).any()
            or instance.shape[0]%th or instance.shape[1]%tw):raise ValueError('Invalid instance labels/grid')
    ids=[int(o['instance_id']) for o in row['objects']]
    if len(set(ids))!=len(ids) or not np.isin(instance[valid],ids).all():raise ValueError('Instance labels do not match condition objects')
    fy,fx=instance.shape[0]//th,instance.shape[1]//tw;items=[];filtered=[]
    for slot,obj in enumerate(row['objects'],1):
        iid=int(obj['instance_id']);support=valid&(instance==iid);pixels=int(support.sum())
        reason=None
        if obj['category_id'] in POLICY['excluded_categories']:reason='generic_category'
        elif pixels<POLICY['minimum_supported_pixels']:reason='too_small_or_no_valid_support'
        if reason:filtered.append(dict(slot=slot,instance_id=iid,reason=reason));continue
        mass=support.reshape(th,fy,tw,fx).sum((1,3)).astype(np.float64)
        # Query positions use canonical columns; do not double-count seam padding.
        query=(np.arange(th)[:,None]*(tw+2*padding)+np.arange(tw)[None,:]+padding).ravel()
        reachable=routing[0,query,slot].numpy()>0
        probability=mass.ravel()*reachable;fraction=float(probability.sum()/mass.sum())
        if fraction<POLICY['minimum_reachable_fraction']:
            filtered.append(dict(slot=slot,instance_id=iid,reason='insufficient_reachable_support',retention=fraction));continue
        chosen=probability>0
        items.append(dict(slot=slot,instance_id=iid,queries=query[chosen],probability=probability[chosen]/probability.sum(),retention=fraction))
    items.sort(key=lambda r:r['instance_id'])
    return items,dict(eligible_instances=len(items),filtered=filtered,routing=meta)


def sample(items,seed):
    rng=np.random.default_rng(int(seed));selected=list(items)
    if len(selected)>POLICY['maximum_instances']:
        keep=sorted(rng.choice(len(selected),POLICY['maximum_instances'],replace=False).tolist());selected=[selected[i] for i in keep]
    positions=[];slots=[]
    for item in selected:
        picks=rng.choice(item['queries'],POLICY['queries_per_instance'],replace=True,p=item['probability'])
        positions.extend(picks.tolist());slots.extend([item['slot']]*len(picks))
    queries,pairs=np.unique(np.asarray(positions,dtype=np.int64),return_inverse=True);n=len(slots)
    labels=(torch.from_numpy(queries),torch.from_numpy(pairs.astype(np.int64)),torch.tensor(slots,dtype=torch.long),
        torch.full((n,),1./n if n else 0.,dtype=torch.float32))
    return labels,dict(selected_instances=len(selected),sampled_pairs=n,unique_queries=len(queries))
