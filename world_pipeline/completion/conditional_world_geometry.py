"""Compile bearing/range/support into original camera-axis full AABBs.

No nearest-frustum teleportation. Ground support uses a declared camera-Y-level
prior fitted to observed object bases, never claimed as measured gravity/floor.
Unsupported, wall-mounted and suspended objects use explicit camera-Y estimates.
"""
from copy import deepcopy
import math
import numpy as np

from ..common import finite_vector
from .frozen_causal_layout import diagnostics


def support_plane(observed, anchor_ids):
    boxes={o['track_id']:o for o in observed['objects']}
    if len(set(anchor_ids))!=len(anchor_ids) or any(i not in boxes for i in anchor_ids):
        raise ValueError('Unknown or duplicate floor anchor')
    bases=[boxes[i]['center_local_m']['value'][1]-boxes[i]['size_aabb_local_m']['value'][1]/2 for i in anchor_ids]
    return dict(kind='camera_Y_level_prior_from_predicted_object_bases',anchor_ids=list(anchor_ids),
        base_y_values_m=bases,floor_y_m=float(np.median(bases)) if bases else None,
        base_spread_m=max(bases)-min(bases) if bases else None,
        gravity_measured=False,floor_measured=False,assumption='camera_Y_approximates_vertical_for_this_hypothesis',
        floor_uncertainty='observer_error_and_unmeasured_camera_tilt_not_calibrated')


def compile_layout(answer, plan, mappings, observed, plane, hfov, image_size, fixed_sizes=None):
    if not isinstance(answer,dict) or set(answer)!={'objects'} or not isinstance(answer['objects'],list):
        raise ValueError('Need exactly objects array')
    expected={o['id'] for o in plan['additions']};raw={}
    if len(plan['placement_dependency_order'])!=len(expected) or set(plan['placement_dependency_order'])!=expected:
        raise ValueError('Placement order must contain each selected ID exactly once')
    for o in answer['objects']:
        if not isinstance(o,dict) or not isinstance(o.get('id'),str) or o['id'] not in expected or o['id'] in raw:
            raise ValueError('Unknown/duplicate layout ID')
        angle,distance=o.get('azimuth_deg'),o.get('range_m')
        if type(angle) not in (int,float) or not math.isfinite(angle) or not -180<=angle<=180:
            raise ValueError('Need finite azimuth_deg in [-180,180]')
        if type(distance) not in (int,float) or not math.isfinite(distance) or distance<0:
            raise ValueError('Need nonnegative horizontal range_m')
        if not finite_vector(o.get('size'),3) or min(o['size'])<=0:
            raise ValueError('Need positive FULL camera-axis size')
        if fixed_sizes is not None and fixed_sizes[o['id']]!=o['size']:
            raise ValueError('Geometric replan may not resize objects')
        if not isinstance(o.get('reason'),str) or not o['reason'].strip(): raise ValueError('Need placement reason')
        if any(k in o for k in ('center','category','center_bounds_m')):
            raise ValueError('Use bearing/range, not free xyz or search bounds')
        raw[o['id']]=o
    if set(raw)!=expected: raise ValueError('Every selected instance is required')
    old={o['track_id']:dict(id=o['track_id'],center=o['center_local_m']['value'],
                          size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    mapped={o['id']:o['category'] for o in mappings}
    byid={o['id']:o for o in plan['additions']};solved={};support_checks=[]
    for oid in plan['placement_dependency_order']:
        if oid not in raw: raise ValueError('Placement order differs from inventory')
        o=raw[oid];s=o['size']; angle=math.radians(o['azimuth_deg']);r=o['range_m']
        x,z=r*math.sin(angle),r*math.cos(angle)
        support=byid[oid]['support'];kind=support['kind']
        if kind=='floor':
            if plane['floor_y_m'] is None: raise ValueError('Floor-supported layout needs explicit observed base anchors')
            y=plane['floor_y_m']+s[1]/2
            support_checks.append(dict(id=oid,kind=kind,proxy_passed=True,
                interpretation='contact_with_declared_level_prior_only',physical_support_verified=False))
        elif kind=='object':
            target={**old,**solved}.get(support['target'])
            if target is None: raise ValueError('Support must be placed before its dependent')
            y=target['center'][1]+target['size'][1]/2+s[1]/2
            overlap=[min(a+sa/2,b+sb/2)-max(a-sa/2,b-sb/2)
                     for a,sa,b,sb in zip([x,z],[s[0],s[2]],
                         [target['center'][0],target['center'][2]],[target['size'][0],target['size'][2]])]
            support_checks.append(dict(id=oid,kind=kind,target=support['target'],
                horizontal_overlap_m=overlap,proxy_passed=min(overlap)>0,
                interpretation='AABB_top_contact_under_camera_Y_level_prior_only',physical_support_verified=False))
        else:
            y=o.get('center_y_m')
            if type(y) not in (int,float) or not math.isfinite(y): raise ValueError('Unspecified support requires signed center_y_m')
            support_checks.append(dict(id=oid,kind=kind,proxy_passed=None,
                interpretation='support_not_modeled',physical_support_verified=False))
        solved[oid]=dict(id=oid,plan_id=oid,center=[x,y,z],size=deepcopy(s),category=mapped[oid],
            unseen_reason='outside_fov',reason=o['reason'],bearing_proposal_deg=o['azimuth_deg'],
            horizontal_range_m=r,declared_support=deepcopy(support),center_method='bearing_range_and_shared_support')
    if set(solved)!=expected: raise ValueError('Incomplete placement dependency order')
    items=[solved[o['id']] for o in plan['additions']]
    review=diagnostics(items,observed,plan['relations'],hfov,image_size)
    review.update(support_checks=support_checks,declared_support_plane=plane,
        support_proxy_failures=sum(c['proxy_passed'] is False for c in support_checks),
        position_was_projected=False,sizes_changed=False,observed_state_preserved=True)
    review['geometry_contract_passed']=review['numeric_constraints_passed'] and review['support_proxy_failures']==0
    return items,review
