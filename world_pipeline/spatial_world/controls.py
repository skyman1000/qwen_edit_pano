"""Diagnostic controls only change AABBs wholly outside the fixed Local FOV."""
import torch
from ..id_binding_world.condition_diagnostic_data import MODES


def fully_outside(geometry):
    c=geometry[...,:3];h=geometry[...,3:6]/2;zmax=c[...,2]+h[...,2]
    known=(geometry[...,6]>.5)&(geometry[...,7]>.5)&(h>0).all(-1)
    # Every corner is outside at least one same 90deg perspective frustum plane.
    return known&((zmax<0)|(c[...,0]-h[...,0]>zmax)|(-c[...,0]-h[...,0]>zmax)|
                  (c[...,1]-h[...,1]>zmax)|(-c[...,1]-h[...,1]>zmax))


def condition_variant(tensors,mode):
    if mode not in MODES:raise ValueError('Unknown control')
    if mode=='world_off':return None,dict(operation=mode,changed_objects=0)
    category,geometry,mask=(t.clone() for t in tensors)
    if mode in ('correct','repeat_correct'):return (category,geometry,mask),dict(operation=mode,changed_objects=0)
    eligible=fully_outside(geometry)[0]&mask[0];eligible[0]=False
    slots=eligible.nonzero().flatten()
    if len(slots)<2:raise ValueError('Need at least two fully-offscreen objects for paired controls')
    metadata=dict(operation=mode,scope='input_AABB_fully_outside_fixed_90deg_Local',eligible_objects=len(slots),
                  local_intersecting_objects_unchanged=True)
    if mode=='wrong_category':
        original=category[0,slots].clone()
        shift=max(range(1,len(slots)),key=lambda s:int((original.roll(s)!=original).sum()))
        category[0,slots]=original.roll(shift)
        changed=slots[category[0,slots]!=original]
        metadata.update(category_histogram_preserved=True,shift=shift)
    else:
        original=geometry[0,slots,:3].clone();best=None
        for shift in range(1,len(slots)):
            trial=geometry[0,slots].clone();trial[:,:3]=original.roll(shift,0)
            if fully_outside(trial).all():
                distance=(trial[:,:3]-original).norm(dim=-1).sum()
                if best is None or distance>best[0]:best=(float(distance),trial[:,:3])
        if best is None:
            # Keep the center multiset even when sizes preclude a full cycle.
            for a in range(len(slots)):
                for b in range(a):
                    trial=geometry[0,slots].clone();trial[a,:3]=original[b];trial[b,:3]=original[a]
                    if fully_outside(trial).all():
                        distance=float((trial[:,:3]-original).norm(dim=-1).sum())
                        if best is None or distance>best[0]:best=(distance,trial[:,:3].clone())
        if best is None:raise ValueError('Cannot permute offscreen centers without conflicting with Local')
        geometry[0,slots,:3]=best[1]
        changed=slots[(geometry[0,slots,:3]!=original).any(-1)]
        metadata.update(centers_multiset_preserved=True,sizes_fixed=True,
            mean_center_displacement_m=float((geometry[0,slots,:3]-original).norm(dim=-1).mean()))
    if not len(changed):raise ValueError('Offscreen perturbation changed no objects')
    metadata.update(changed_objects=len(changed),changed_slots=changed.cpu().tolist())
    return (category,geometry,mask),metadata
