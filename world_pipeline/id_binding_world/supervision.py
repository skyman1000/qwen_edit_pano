"""Instance-balanced sampled coverage CE, from existing mesh-derived support.

Masks are targets only, never attention masks or generator input features.
The local NumPy RNG does not alter posterior/noise/timestep RNG in either arm.
"""
import numpy as np
import torch

POLICY = dict(version='instance_coverage_sampled_ce_v1', minimum_supported_pixels=64,
              excluded_categories=[39, 40], maximum_instances=32, queries_per_instance=8,
              label_source='existing_instance_and_object_loss_valid',
              boundary_policy='fractional_area_pool_no_binarization',
              visibility='mesh_support_not_polished_pixel_certification',
              outside_loss=False, spatial_bias=False)


def sample_binding(row, height, padding, seed, policy=POLICY):
    """Return query indices + pair/query/slot/weight tensors on CPU and metadata.

    Sampling an object's queries proportional to soft area coverage estimates
    mean_i sum_q M_i(q)*[-log A(q,i)] / sum_q M_i(q).
    All condition objects remain present, including unselected/generic objects.
    """
    th, tw = height//16, height//8
    if height < 16 or height % 16 or padding < 0:
        raise ValueError('Invalid target/token grid')
    with np.load(row['object_region_supervision']['path'], allow_pickle=False) as arrays:
        instance = arrays['instance']; valid = arrays['object_loss_valid']
        hit = arrays['instance_valid']
    if (instance.ndim != 2 or valid.shape != instance.shape or hit.shape != instance.shape
            or valid.dtype != bool or hit.dtype != bool or (valid & ~hit).any()
            or instance.shape[0] % th or instance.shape[1] % tw):
        raise ValueError('Invalid source instance/token supervision')
    ids = [int(o['instance_id']) for o in row['objects']]
    if len(set(ids)) != len(ids) or not np.isin(instance[valid], ids).all():
        raise ValueError('Instance support does not match condition slots')
    unique, counts = np.unique(instance[valid], return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))
    # Sort by instance ID so input token order cannot change the sampled objects.
    candidates = sorted([(int(o['instance_id']), slot) for slot,o in enumerate(row['objects'],1)
        if o['category_id'] not in policy['excluded_categories']
        and sizes.get(int(o['instance_id']), 0) >= policy['minimum_supported_pixels']])
    rng = np.random.default_rng(int(seed))
    if len(candidates) > policy['maximum_instances']:
        chosen = rng.choice(len(candidates), policy['maximum_instances'], replace=False)
        candidates = [candidates[i] for i in sorted(chosen.tolist())]
    positions=[]; slots=[]
    fy, fx = instance.shape[0]//th, instance.shape[1]//tw
    for iid, slot in candidates:
        support = valid & (instance == iid)
        mass = support.reshape(th,fy,tw,fx).sum((1,3)).astype(np.float64).ravel()
        if mass.sum() <= 0:raise ValueError('Selected instance has no support')
        picks = rng.choice(th*tw, policy['queries_per_instance'], replace=True, p=mass/mass.sum())
        y,x = np.divmod(picks,tw)
        positions.extend((y*(tw+2*padding)+x+padding).tolist())
        slots.extend([slot]*len(picks))
    queries, pair_queries = np.unique(np.asarray(positions, dtype=np.int64), return_inverse=True)
    count = len(positions)
    tensors = (torch.from_numpy(queries), torch.from_numpy(pair_queries.astype(np.int64)),
        torch.tensor(slots,dtype=torch.long), torch.full((count,),1./count if count else 0.,dtype=torch.float32))
    return tensors, dict(eligible_instances=len([o for o in row['objects']
        if o['category_id'] not in policy['excluded_categories'] and sizes.get(int(o['instance_id']),0)>=policy['minimum_supported_pixels']]),
        selected_instances=len(candidates), sampled_pairs=count, unique_queries=len(queries))


def attention_coverage_loss(q, k, mask, supervision):
    """Real pre-SDPA Q/K; FP32 log mean probability over heads, stable gradients."""
    import math
    query_indices, pair_queries, slots, weights = supervision
    if q.shape[0] != 1 or k.shape[0] != 1 or q.ndim != 4 or k.ndim != 4:
        raise ValueError('Binding supports a single scene')
    if not len(slots):
        # A differentiable finite zero, without retaining a full attention map.
        return q[..., :1, :1].float().sum()*0 + k[..., :1, :1].float().sum()*0
    if (query_indices.ndim!=1 or pair_queries.shape!=slots.shape or weights.shape!=slots.shape
            or query_indices.min()<0 or query_indices.max()>=q.shape[-2]
            or pair_queries.min()<0 or pair_queries.max()>=len(query_indices)
            or slots.min()<1 or slots.max()>=k.shape[-2] or not mask[0,slots].all()
            or not torch.isfinite(weights).all() or (weights<0).any()
            or not torch.isclose(weights.sum(),weights.new_tensor(1.))):
        raise ValueError('Invalid instance binding indices/weights')
    with torch.autocast(device_type=q.device.type,enabled=False):
        logits=(q[0,:,query_indices].float() @ k[0].float().transpose(-1,-2))/math.sqrt(q.shape[-1])
        logits=logits.masked_fill(~mask[0][None,None,:],-torch.inf)
        logp=logits.log_softmax(-1)
        # Gather VALID target slots before head aggregation. logsumexp on an
        # all--inf padding column has undefined gradients, even if later unused.
        log_mean_p=torch.logsumexp(logp[:,pair_queries,slots],dim=0)-math.log(q.shape[1])
        return -(log_mean_p*weights.float()).sum()
