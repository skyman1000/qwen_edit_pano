"""Conservative off-FOV AABB constraints and bounded, deterministic local search.

No gravity, walls, mesh collision or physical support certification is inferred.
The camera is a centred pinhole, matching the frozen validation face2 input.
"""
from copy import deepcopy
import itertools
import math

import numpy as np

from ..common import finite_vector
from .causal_world_completion import relation_checks
from .scene_completion import validate_layout, view_check

REGIONS=('left','right','behind_camera','above','below')


def layout_answer(answer, plan, mappings, max_shift=.75):
    if not math.isfinite(max_shift) or not 0<=max_shift<=1.5:
        raise ValueError('max_shift must be finite and within 0..1.5 metres')
    if not isinstance(answer, dict) or set(answer) != {'objects'}:
        raise ValueError('Layout requires exactly objects array')
    if not isinstance(answer['objects'], list):
        raise ValueError('objects must be an array')
    regions={o['id']:o.get('off_fov_region') for o in plan['additions']}
    if any(r not in REGIONS for r in regions.values()):
        raise ValueError('Each semantic instance needs a valid off_fov_region')
    plain=[]
    for obj in answer['objects']:
        if not isinstance(obj, dict) or set(obj) != {'id','center','size','reason'}:
            raise ValueError('v2 box needs exactly id, center, size, reason; search bounds are program-owned')
        plain.append(deepcopy(obj))
    items = validate_layout({'objects':plain}, plan, mappings)
    for obj in items:
        c=np.array(obj['center'],dtype=float)
        obj['center_bounds_m']=[(c-max_shift).tolist(),(c+max_shift).tolist()]
        obj['bounds_source']='program_trust_region_not_scene_measurement'
        obj['off_fov_region']=regions[obj['id']]
        obj['proposal_center'] = deepcopy(obj['center'])
    return items


def region_margin(center,size,region,hfov,image_size):
    """Whole-box half-space certificate, matching the declared camera region."""
    c,s=np.asarray(center),np.asarray(size)
    tx=math.tan(math.radians(hfov)/2);ty=tx*image_size[1]/image_size[0]
    if region=='behind_camera':return float(-c[2]-s[2]/2)
    if region in ('left','right'):
        sign=-1 if region=='left' else 1
        return float(min(sign*c[0]-s[0]/2,sign*c[0]-tx*c[2]-(s[0]+tx*s[2])/2))
    if region in ('above','below'):
        sign=1 if region=='above' else -1
        return float(min(sign*c[1]-s[1]/2,sign*c[1]-ty*c[2]-(s[1]+ty*s[2])/2))
    raise ValueError('Unknown off-FOV region: '+str(region))


def check(items, observed, relations, hfov, image_size, max_distance=20.):
    if (not math.isfinite(hfov) or not 0 < hfov < 180 or len(image_size) != 2
            or any(type(v) is not int or v <= 0 for v in image_size)):
        raise ValueError('Need valid centred-pinhole FOV and image [width,height]')
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError('Invalid distance guard')
    old = [dict(id=o['track_id'],center=o['center_local_m']['value'],
                size=o['size_aabb_local_m']['value']) for o in observed['objects']]
    all_boxes = old+items
    new_ids = {o['id'] for o in items}
    by_id = {o['id']:o for o in all_boxes}
    rows = []
    for obj in items:
        if (not finite_vector(obj['center'],3) or not finite_vector(obj['size'],3)
                or min(obj['size']) <= 0):
            raise ValueError('Need finite geometry and positive full size')
        v = view_check(obj['center'], obj['size'], hfov, image_size)
        d = float(np.linalg.norm(obj['center']))
        margin=region_margin(obj['center'],obj['size'],obj['off_fov_region'],hfov,image_size)
        rows.append(dict(id=obj['id'], intersects_input_frustum=v['intersects_input_box_proxy'],
                         frustum_witness_m=v['frustum_witness_m'],
                         camera_inside_box=v['camera_inside_box_proxy'], distance_m=d,
                         exceeds_distance_guard=d>max_distance,
                         intended_region=obj['off_fov_region'],region_margin_m=margin,
                         intended_region_passed=margin>1e-6))
    spatial = relation_checks(relations, observed, items)
    # Physical contact requires nearby bounding boxes irrespective of gravity.
    # This necessary condition can reject an impossible support; passing it does
    # not establish contact, top surfaces or physical stability.
    for r in spatial:
        if r['kind'] in ('supported_by','on_top_of'):
            a,b=by_id[r['source']],by_id[r['target']]
            gap=np.maximum(np.abs(np.array(a['center'])-b['center'])-
                           (np.array(a['size'])+b['size'])/2,0.)
            r['aabb_contact_gap_m']=gap.tolist()
            r['proxy_passed']=False if np.any(gap>.05) else None
            r['interpretation']='necessary_AABB_contact_with_5cm_tolerance_not_support_verification'
    # Overlap with an explicitly hypothesized support/container is reported,
    # rather than being mislabelled a proven solid-shape collision.
    structured = {frozenset((r['source'],r['target'])) for r in relations
                  if r['kind'] in ('inside','supported_by','on_top_of')}
    overlaps = []
    for a,b in itertools.combinations(all_boxes,2):
        if not ({a['id'],b['id']} & new_ids):
            continue
        extent = (np.minimum(np.array(a['center'])+np.array(a['size'])/2,
                             np.array(b['center'])+np.array(b['size'])/2)
                  -np.maximum(np.array(a['center'])-np.array(a['size'])/2,
                              np.array(b['center'])-np.array(b['size'])/2))
        if np.all(extent > 1e-5):
            overlaps.append(dict(a=a['id'],b=b['id'],volume_m3=float(np.prod(extent)),
                                 declared_support_or_container=frozenset((a['id'],b['id'])) in structured))
    failed = sum(r['proxy_passed'] is False for r in spatial)
    unrelated = sum(not r['declared_support_or_container'] for r in overlaps)
    violations = (sum(r['intersects_input_frustum'] or not r['intended_region_passed'] or r['camera_inside_box'] or
                      r['exceeds_distance_guard'] for r in rows)+failed+unrelated)
    # Continuous residual helps the finite search cross a sequence of bad states;
    # it never overrides a hard violation count.
    residual = sum(r['volume_m3'] for r in overlaps if not r['declared_support_or_container'])
    residual += sum(max(0.,1e-5-r['region_margin_m']) for r in rows)
    signs = {'left_of':(0,-1),'right_of':(0,1),'above':(1,1),'below':(1,-1),
             'in_front_of':(2,-1),'behind':(2,1)}
    for r in spatial:
        if r['kind'] in signs:
            axis,sign=signs[r['kind']]
            delta=sign*(by_id[r['source']]['center'][axis]-by_id[r['target']]['center'][axis])
            residual += max(0.,1e-5-delta)
        elif r['kind']=='near' and 'max_distance_m' in r:
            residual += max(0.,float(np.linalg.norm(np.array(by_id[r['source']]['center'])-
                                                  by_id[r['target']]['center']))-r['max_distance_m'])
        elif 'aabb_contact_gap_m' in r:
            residual += sum(max(0.,g-.05) for g in r['aabb_contact_gap_m'])
    return dict(objects=rows, relations=spatial, overlaps=overlaps,
                hard_violations=int(violations), numeric_constraints_passed=violations==0,
                failed_relation_proxies=failed, unrelated_aabb_overlaps=unrelated,
                unverified_relations=sum(r['proxy_passed'] is None for r in spatial),
                residual=float(residual), physical_geometry_verified=False,
                note='Conservative AABB proxies; support, gravity, room boundaries and actual visibility unverified')


def solve(items, observed, relations, hfov, image_size, sweeps=3, max_distance=20.):
    """Search each explicit small domain; never change size, identity or selection.

    This finite coordinate search is not a complete solver: failure means no
    solution found in this bounded search, not proof that the scene is impossible.
    """
    if not 1 <= sweeps <= 10:
        raise ValueError('sweeps must be 1..10')
    result, trace = deepcopy(items), []
    def evaluate():
        report=check(result,observed,relations,hfov,image_size,max_distance)
        movement=sum(float(np.linalg.norm(np.array(o['center'])-o['proposal_center'])) for o in result)
        return (report['hard_violations'],report['residual'],movement),report
    before=check(result,observed,relations,hfov,image_size,max_distance)
    for sweep in range(sweeps):
        changed=False
        for obj in result:
            original=deepcopy(obj['center']);best=original;best_score,_=evaluate()
            lo,hi=np.asarray(obj['center_bounds_m'],dtype=float)
            axes=[sorted(set([float(lo[k]),float(hi[k]),float((lo[k]+hi[k])/2),
                              float(obj['proposal_center'][k])])) for k in range(3)]
            for candidate in itertools.product(*axes):
                obj['center']=list(candidate)
                score,_=evaluate()
                if score < best_score:
                    best_score,best=score,list(candidate)
            obj['center']=best
            if best != original:
                changed=True;trace.append(dict(sweep=sweep,id=obj['id'],before=original,after=best))
        if not changed:
            break
    _,after=evaluate()
    return result,dict(before=before,after=after,changes=trace,
                       method='finite_bounded_coordinate_search',complete_solver=False,
                       sizes_changed=False,selection_changed=False)
