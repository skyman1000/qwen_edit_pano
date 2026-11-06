"""Instance-balanced positive coverage and reliable-other-instance negatives.

No target mask enters the forward attention. Negatives ignore unknown support,
generic categories, and a one-token guard around each target (periodic ERP x).
Both terms are sample means, not ratios of Monte Carlo estimates.
"""
import math
import numpy as np
import torch

POLICY = dict(version='instance_positive_and_reliable_other_bce_v1',
    minimum_supported_pixels=64, excluded_categories=[39, 40], maximum_instances=32,
    queries_per_instance=8, negatives_per_instance=8, negative_guard_tokens=1,
    negative_source='eligible other instances within object_loss_valid only; never unknown/background complement',
    pooling='fractional area mass, no positive threshold', heads='mean probabilities',
    spatial_routing=False, normalization='mean per instance separately for positives and available negatives')


def guard(support):
    """One-token Chebyshev guard, periodic longitude and clamped latitude."""
    out=np.zeros_like(support,dtype=bool)
    for dy in (-1,0,1):
        source=np.clip(np.arange(support.shape[0])+dy,0,support.shape[0]-1)
        for dx in (-1,0,1):out |= np.roll(support[source],dx,axis=1)
    return out


def sample_binding(row,height,padding,seed,policy=POLICY):
    th,tw=height//16,height//8
    if height<16 or height%16 or padding<0 or policy['negative_guard_tokens']!=1:
        raise ValueError('Invalid ERP grid/guard')
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as f:
        instance=f['instance'];valid=f['object_loss_valid'];hit=f['instance_valid']
    if (instance.ndim!=2 or instance.shape!=valid.shape or hit.shape!=instance.shape
            or valid.dtype!=bool or hit.dtype!=bool or (valid&~hit).any()
            or instance.shape[0]%th or instance.shape[1]%tw):
        raise ValueError('Invalid instance supervision')
    ids=[int(o['instance_id']) for o in row['objects']]
    if len(set(ids))!=len(ids) or not np.isin(instance[valid],ids).all():
        raise ValueError('Instance IDs must map uniquely to input slots')
    unique,counts=np.unique(instance[valid],return_counts=True);sizes=dict(zip(unique.tolist(),counts.tolist()))
    candidates=sorted((int(o['instance_id']),slot) for slot,o in enumerate(row['objects'],1)
        if o['category_id'] not in policy['excluded_categories']
        and sizes.get(int(o['instance_id']),0)>=policy['minimum_supported_pixels'])
    fy,fx=instance.shape[0]//th,instance.shape[1]//tw
    def pool(pixels):return pixels.reshape(th,fy,tw,fx).sum((1,3)).astype(np.float64)
    masses={iid:pool(valid&(instance==iid)) for iid,_ in candidates}
    total=sum(masses.values(),np.zeros((th,tw),dtype=np.float64))
    rng=np.random.default_rng(int(seed));eligible=len(candidates)
    if eligible>policy['maximum_instances']:
        chosen=rng.choice(eligible,policy['maximum_instances'],replace=False)
        candidates=[candidates[i] for i in sorted(chosen.tolist())]
    positions=[];slots=[];positive=[];negative=[];negative_instances=0
    def append(mass,iid_slot,n,is_positive):
        picks=rng.choice(th*tw,n,replace=True,p=mass.ravel()/mass.sum())
        y,x=np.divmod(picks,tw)
        positions.extend((y*(tw+2*padding)+x+padding).tolist());slots.extend([iid_slot]*n)
        positive.extend([1./n if is_positive else 0.]*n)
        negative.extend([0. if is_positive else 1./n]*n)
    for iid,slot in candidates:
        mass=masses[iid];append(mass,slot,policy['queries_per_instance'],True)
        other=np.maximum(total-mass,0.);other[guard(mass>0)]=0.
        if other.sum()>0:
            append(other,slot,policy['negatives_per_instance'],False);negative_instances+=1
    queries,pairs=np.unique(np.asarray(positions,dtype=np.int64),return_inverse=True)
    pos=torch.tensor(positive,dtype=torch.float32)/max(1,len(candidates))
    neg=torch.tensor(negative,dtype=torch.float32)/max(1,negative_instances)
    tensors=(torch.from_numpy(queries),torch.from_numpy(pairs.astype(np.int64)),
        torch.tensor(slots,dtype=torch.long),pos,neg)
    return tensors,dict(eligible_instances=eligible,selected_instances=len(candidates),negative_instances=negative_instances,
        positive_pairs=sum(v>0 for v in positive),negative_pairs=sum(v>0 for v in negative),unique_queries=len(queries))


def binding_terms(q,k,mask,labels):
    """FP32 true SDPA probabilities; stable log(1-p) via other-key logsumexp."""
    indices,pairs,slots,pos,neg=labels
    if q.ndim!=4 or k.ndim!=4 or q.shape[0]!=1 or k.shape[0]!=1 or mask.shape!=(1,k.shape[-2]):
        raise ValueError('Expected single-scene Q/K and object mask')
    if not len(slots):
        zero=q[...,:1,:1].float().sum()*0+k[...,:1,:1].float().sum()*0
        return zero,zero
    if (indices.ndim!=1 or pairs.shape!=slots.shape or pos.shape!=slots.shape or neg.shape!=slots.shape
            or indices.min()<0 or indices.max()>=q.shape[-2] or pairs.min()<0 or pairs.max()>=len(indices)
            or slots.min()<1 or slots.max()>=k.shape[-2] or not mask[0,slots].all()
            or any(not torch.isfinite(w).all() or (w<0).any() for w in (pos,neg))
            or not torch.isclose(pos.sum(),pos.new_tensor(1.))
            or not (torch.isclose(neg.sum(),neg.new_tensor(0.)) or torch.isclose(neg.sum(),neg.new_tensor(1.)))
            or (pos*neg).any() or not mask[0,0]):
        raise ValueError('Invalid binding labels/weights')
    with torch.autocast(q.device.type,enabled=False):
        logits=(q[0,:,indices].float()@k[0].float().transpose(-1,-2))/math.sqrt(q.shape[-1])
        logits=logits.masked_fill(~mask[0][None,None,:],-torch.inf)
        logp=logits.log_softmax(-1)
        positive=pos>0
        lp=logp[:,pairs[positive],slots[positive]]
        coverage=-((torch.logsumexp(lp,0)-math.log(q.shape[1]))*pos[positive]).sum()
        negative=neg>0
        if negative.any():
            # log(1-mean_h p_h) = logmeanexp_h(log(1-p_h)); avoid
            # subtracting probabilities near 1 or all-inf invalid padding.
            selected=logits[:,pairs[negative],:]
            other=selected.clone()
            other.scatter_(-1,slots[negative][None,:,None].expand(q.shape[1],-1,1),-torch.inf)
            log_other=torch.logsumexp(other,-1)-torch.logsumexp(selected,-1)
            concentration=-((torch.logsumexp(log_other,0)-math.log(q.shape[1]))*neg[negative]).sum()
        else:concentration=coverage*0
    return coverage,concentration
