"""Off-frustum box layout: no fixed movement radius, no asserted room geometry.

Hard contract: finite positive boxes outside the input cone, fixed identities.
Relations and AABB overlaps are soft diagnostics, not verified shape physics.
Finite coordinate refinement is not a complete global layout solver.
"""
from copy import deepcopy
import itertools
import math

import numpy as np

from .causal_world_completion import relation_checks
from .off_fov_geometry import REGIONS
from .region_projection import closest_region_center
from .scene_completion import validate_layout, view_check


def diagnostics(items, observed, relations, hfov, image_size):
    boxes = [dict(id=o['track_id'], center=o['center_local_m']['value'],
                  size=o['size_aabb_local_m']['value']) for o in observed['objects']]+items
    new = {o['id'] for o in items}
    checks = [dict(id=o['id'], **view_check(o['center'], o['size'], hfov, image_size)) for o in items]
    overlaps = []
    for a, b in itertools.combinations(boxes, 2):
        if not {a['id'], b['id']} & new:
            continue
        extent = np.minimum(np.array(a['center'])+np.array(a['size'])/2,
                            np.array(b['center'])+np.array(b['size'])/2)-np.maximum(
                            np.array(a['center'])-np.array(a['size'])/2,
                            np.array(b['center'])-np.array(b['size'])/2)
        if np.all(extent > 0):
            volume = float(np.prod(extent))
            overlaps.append(dict(a=a['id'], b=b['id'], volume_m3=volume,
                fraction_of_smaller_box=volume/min(math.prod(a['size']), math.prod(b['size'])),
                actual_shape_collision_verified=False))
    spatial = relation_checks(relations, observed, items)
    violations = sum(r['intersects_input_box_proxy'] or r['camera_inside_box_proxy'] for r in checks)
    return dict(objects=checks, overlaps=overlaps, relations=spatial,
        hard_violations=violations, numeric_constraints_passed=violations == 0,
        soft_relation_violations=sum(r['proxy_passed'] is False for r in spatial),
        unverified_relations=sum(r['proxy_passed'] is None for r in spatial),
        physical_geometry_verified=False, room_bounds_known=False,
        note='AABB overlaps/axis relations are review signals, not physical collision or measured support.')


def solve_layout(answer, plan, mappings, observed, hfov, image_size, sweeps=3):
    if type(sweeps) is not int or not 0 <= sweeps <= 10:
        raise ValueError('Need 0..10 coordinate sweeps')
    if not isinstance(answer, dict) or set(answer) != {'objects'}:
        raise ValueError('Layout needs exactly objects array')
    result = validate_layout(answer, plan, mappings)
    proposal = {o['id']: deepcopy(o['center']) for o in result}
    before = diagnostics(result, observed, plan['relations'], hfov, image_size)
    fixed = [dict(id=o['track_id'], center=o['center_local_m']['value'],
                  size=o['size_aabb_local_m']['value']) for o in observed['objects']]
    trace = []
    def objective():
        review = diagnostics(result, observed, plan['relations'], hfov, image_size)
        movement = sum(float(np.linalg.norm(np.array(o['center'])-proposal[o['id']])) /
                       max(float(np.linalg.norm(o['size'])), 1e-6) for o in result)
        # Declared heuristic, not a causal probability or a binary physics test.
        return (review['hard_violations'],
                sum(o['fraction_of_smaller_box'] for o in review['overlaps'])+
                review['soft_relation_violations']+0.05*movement)
    for sweep in range(max(1, sweeps)):
        changed = False
        for obj in result:
            start = deepcopy(obj['center'])
            proposals = [start]
            # Minimum translations to each off-frustum convex disjunct. There
            # is no semantic left/right quota or artificial trust cube.
            if sweep < sweeps:
                for other in fixed+result:
                    if other['id'] == obj['id']:
                        continue
                    for axis in range(3):
                        for sign in (-1, 1):
                            p = start.copy()
                            p[axis] = other['center'][axis]+sign*(obj['size'][axis]+other['size'][axis])/2
                            proposals.append(p)
                by_id = {o['id']: o for o in fixed+result}
                signs = {'left_of':(0,-1), 'right_of':(0,1), 'above':(1,1),
                         'below':(1,-1), 'in_front_of':(2,-1), 'behind':(2,1)}
                for relation in plan['relations']:
                    if relation['kind'] not in signs or obj['id'] not in (relation['source'], relation['target']):
                        continue
                    other_id = relation['target'] if obj['id'] == relation['source'] else relation['source']
                    axis, sign = signs[relation['kind']]
                    if obj['id'] == relation['target']: sign *= -1
                    p = start.copy(); p[axis] = by_id[other_id]['center'][axis]+sign*1e-4
                    proposals.append(p)
            best, best_score, best_region = None, None, None
            for p in proposals:
                for region in REGIONS:
                    projected = closest_region_center(p, obj['size'], region, hfov, image_size, epsilon=1e-4)
                    obj['center'] = projected['center']
                    score = objective()
                    if best_score is None or score < best_score:
                        best, best_score, best_region = deepcopy(obj['center']), score, region
            obj['center'] = best
            obj['off_fov_certificate_region'] = best_region
            if start != best:
                changed = True
                trace.append(dict(sweep=sweep, id=obj['id'], before=start, after=best))
        if not changed:
            break
    for obj in result:
        obj['proposal_center'] = proposal[obj['id']]
        obj['translation_m'] = (np.array(obj['center'])-proposal[obj['id']]).tolist()
    return result, dict(before=before, after=diagnostics(result, observed, plan['relations'], hfov, image_size),
        changes=trace, method='off_frustum_halfspace_projection_with_soft_coordinate_refinement',
        complete_solver=False, sizes_changed=False, selection_changed=False,
        movement_radius_m=None, soft_objective='overlap_fraction + failed_axis_relations + 0.05*size_normalized_translation')
