"""Bounded joint placement and original-camera checks using explicit box proxies.

No inventory, host, intrinsic size, envelope or yaw changes in optimization.
Ray tests use a finite grid; transparent/unknown observed boxes do not certify
occlusion. Results are proxy compatibility, never physical/semantic ground truth.
"""
from copy import deepcopy
import itertools
import json
import math

import numpy as np
from scipy.optimize import minimize

from .global_world_model import vector, number

SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def rotation(yaw):
    a = math.radians(yaw)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def aabb_size(size, yaw):
    return (np.abs(rotation(yaw)) @ np.asarray(size)).tolist()


def placement_prompt(hypothesis, program, packet):
    """Legacy absolute-coordinate interface, unused by the v3 production pipeline."""
    return ('Jointly place ALL fixed new instances under the supplied whole-world program and room envelope. '
        'Return a JSON object with placements (array), each row with an EXACT requested id, '
        'center_m (three finite numbers) and yaw_deg (finite number). '
        'Every fixed NEW ID once. Camera x-right/y-IMAGE-UP/z-forward, metres. yaw=0 local +z is camera +z; '
        'positive yaw rotates local +z towards +x. New object dimensions/hosts are already frozen. '
        'Do not add/drop/resize instances or change their support, program, room or visibility hypothesis. '
        'Arrange the whole inventory together, respecting shared resources, relations and access zones. '
        'For top/floor/ceiling items the compiler derives support Y; supply plausible XZ centres. '
        'For inside items choose a centre inside the host. For wall/free items choose all XYZ. '
        'Box extents, including after yaw, must fit the envelope. Newly visible objects cannot be called hidden. '
        'No minimum object count or rear quota. '
        '\nHYPOTHESIS: ' + json.dumps(hypothesis) + '\nJOINT_PROGRAM: ' + json.dumps(program) +
        '\nFROZEN_OBSERVATIONS: ' + json.dumps(packet))


def placement_answer(value, program):
    rows = value.get('placements') if isinstance(value, dict) else None
    expected = {o['id'] for o in program['objects']}
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('placements list required')
    ids = [r.get('id') for r in rows]
    if any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError('Exactly every NEW ID once required')
    for r in rows:
        vector(r.get('center_m'), 3, 'center_m')
        if not number(r.get('yaw_deg')):
            raise ValueError('Finite yaw_deg required')
    return {r['id']: dict(center_m=list(r['center_m']), yaw_deg=float(r['yaw_deg'])) for r in rows}


def solve_layout(hypothesis, program, packet, proposal, seed, starts=3, maxiter=180, tolerance=.015, camera=None):
    obs = {o['id']: dict(center_m=o['center_m'], size=o['full_camera_aabb_m']) for o in packet['objects']}
    ids = [o['id'] for o in program['objects']]
    specs = {o['id']: o for o in program['objects']}
    sizes = {i: np.array(aabb_size(specs[i]['size_m'], proposal[i]['yaw_deg'])) for i in ids}
    sizes.update({i: np.asarray(o['size']) for i, o in obs.items()})
    room = hypothesis['room']
    lo = np.array([room['x_min'], room['floor_y'], room['z_min']])
    hi = np.array([room['x_max'], room['ceiling_y'], room['z_max']])
    if not ids:
        return {}, dict(passed=True, starts=0, minimum_constraint_margin_m=None,
            inventory_changed=False, intrinsic_dimensions_changed=False, issues=[])
    lower = np.array([lo + sizes[i]/2 for i in ids])
    upper = np.array([hi - sizes[i]/2 for i in ids])
    if np.any(lower > upper):
        return deepcopy(proposal), dict(passed=False, issues=['Frozen size exceeds hypothesized room envelope'])
    original = np.array([proposal[i]['center_m'] for i in ids])
    initial = original.copy()
    # Topological support-Y compilation only. Observed hosts remain untouched.
    done = {i: np.asarray(o['center_m']) for i, o in obs.items()}
    pending = list(ids)
    while pending:
        for i in list(pending):
            support = specs[i]['support'];host = support['host'];kind = support['kind'];k = ids.index(i)
            if host and host not in done:
                continue
            if kind == 'floor':initial[k, 1] = room['floor_y'] + sizes[i][1]/2
            elif kind == 'ceiling':initial[k, 1] = room['ceiling_y'] - sizes[i][1]/2
            elif kind == 'top':initial[k, 1] = done[host][1] + (sizes[host][1]+sizes[i][1])/2
            done[i] = initial[k].copy();pending.remove(i)
            break
        else:
            raise ValueError('Unresolved support graph')
    initial = np.clip(initial, lower, upper)
    allids = list(obs) + ids

    def contained_pair(a, b):
        # Explicit containment exempts the enclosing host box from collision.
        i = a
        while i in specs:
            s = specs[i]['support']
            if s['kind'] != 'inside':return False
            if s['host'] == b:return True
            i = s['host']
        return False

    pairs = [(a, b) for a, b in itertools.combinations(allids, 2)
             if (a in specs or b in specs) and not contained_pair(a, b) and not contained_pair(b, a)]

    def constraints(flat, named=False):
        centres = {i: np.asarray(o['center_m']) for i, o in obs.items()}
        centres.update(zip(ids, flat.reshape(-1, 3)))
        values = [];names = []
        def add(name, val):
            array = np.atleast_1d(val).ravel()
            values.extend(array.tolist());names.extend([name]*len(array))
        for i in ids:
            c = centres[i];size = sizes[i];s = specs[i]['support'];host = s['host'];kind = s['kind']
            add(i+':room', c-size/2-lo);add(i+':room', hi-c-size/2)
            if kind in ('floor', 'ceiling', 'top'):
                y = room['floor_y']+size[1]/2 if kind == 'floor' else room['ceiling_y']-size[1]/2
                if kind == 'top':y = centres[host][1]+(sizes[host][1]+size[1])/2
                add(i+':support_contact', tolerance-abs(c[1]-y))
            if kind in ('top', 'inside'):
                axes = [0, 2] if kind == 'top' else [0, 1, 2]
                add(i+':support_extent', (sizes[host][axes]-size[axes])/2 - np.abs((c-centres[host])[axes]))
            if kind == 'wall':
                gap = min(c[0]-size[0]/2-lo[0], hi[0]-c[0]-size[0]/2,
                          c[2]-size[2]/2-lo[2], hi[2]-c[2]-size[2]/2)
                add(i+':wall_contact', .08-gap)
            add(i+':camera_clear', np.max(np.abs(c)-size/2)-.05)
            if camera and specs[i]['visibility'] == 'outside_fov':
                th = math.tan(math.radians(camera['hfov'])/2)
                tv = th*camera['image_size'][1]/camera['image_size'][0]
                normals = np.array([[0,0,1],[1,0,th],[-1,0,th],[0,1,tv],[0,-1,tv]])
                hidden = np.array([.01,0,0,0,0])-normals@c-np.abs(normals)@size/2-.02
                region = specs[i].get('camera_region', 'outside_fov')
                if region in ('rear','left','right'):
                    add(i+':camera_region', hidden[{'rear':0,'left':1,'right':2}[region]])
                    if region in ('left','right'):
                        add(i+':camera_side', (-1 if region=='left' else 1)*c[0]-size[0]/2)
                else:add(i+':outside_frustum', np.max(hidden))
        for a, b in pairs:
            add(a+':collision:'+b, np.max(np.abs(centres[a]-centres[b])-(sizes[a]+sizes[b])/2))
        for r in program['relations']:
            a, b = r['source'], r['target'];delta = centres[b]-centres[a]
            if r['kind'] == 'near':
                gap = (np.maximum(np.abs(delta[[0,2]])-(sizes[a][[0,2]]+sizes[b][[0,2]])/2,0)
                    if r.get('measure','center') == 'footprint_gap' else delta[[0,2]])
                add(a+':near:'+b, r['distance_m']-np.linalg.norm(gap))
            elif r['kind'] == 'separate':
                gap = np.maximum(np.abs(delta[[0, 2]])-(sizes[a][[0, 2]]+sizes[b][[0, 2]])/2, 0)
                add(a+':separate:'+b, np.linalg.norm(gap)-r['distance_m'])
            else:
                direction = rotation(proposal[a]['yaw_deg'])[:, 2]
                add(a+':facing:'+b, np.dot(delta[[0, 2]], direction[[0, 2]]) - .5*np.linalg.norm(delta[[0, 2]]))
        for n, z in enumerate(program['access_zones']):
            owner = z['owner'];rot = rotation(proposal[owner]['yaw_deg'])
            size = np.abs(rot)@np.asarray(z['size_m'])
            if z.get('base') == 'floor':
                c = centres[owner]+rot@np.array([z['offset_xz_m'][0],0.,z['offset_xz_m'][1]])
                c[1] = lo[1]+size[1]/2
            else:c = centres[owner]+rot@np.asarray(z['offset_m'])
            add('access_'+str(n)+':room', c-size/2-lo);add('access_'+str(n)+':room', hi-c-size/2)
            for other in allids:
                add('access_'+str(n)+':occupied:'+other, np.max(np.abs(c-centres[other])-(size+sizes[other])/2))
        return (np.asarray(values), names) if named else np.asarray(values)

    def objective(flat):
        violation = np.minimum(constraints(flat), 0)
        return float(np.square(flat-initial.ravel()).sum() + 1000*np.square(violation).sum())

    rng = np.random.default_rng(seed);best = None;attempts = []
    def rank_candidate(flat):
        margins = constraints(flat)
        return (bool(margins.min() < -tolerance-1e-5),
            float(np.square(np.minimum(margins+tolerance,0)).sum()), objective(flat))
    from .global_world_feasibility import seed_layout
    feasible, feasibility = seed_layout(ids,specs,obs,sizes,room,proposal,initial,lower,upper,
        pairs,program['relations'],program['access_zones'],camera,rotation)
    if feasible is not None:
        best = (rank_candidate(feasible),feasible.copy())
        attempts.append(dict(start='MILP',solver_success=not best[0][0],
            message='Independent exact constraint evaluation of linear seed',
            minimum_margin_m=float(constraints(feasible).min())))
    bounds = list(zip(lower.ravel(), upper.ravel()))
    for k in range(starts):
        if best is not None and not best[0][0]:break
        x = initial.copy()
        if k:x[:, [0, 2]] += rng.normal(0, .5*k, (len(ids), 2))
        x = np.clip(x, lower, upper)
        answer = minimize(objective, x.ravel(), method='SLSQP', bounds=bounds,
            constraints=[dict(type='ineq', fun=lambda v: constraints(v)+tolerance)],
            options=dict(maxiter=maxiter, ftol=1e-7))
        margins = constraints(answer.x);minimum = float(margins.min())
        rank = rank_candidate(answer.x)
        attempts.append(dict(start=k, solver_success=bool(answer.success), message=str(answer.message), minimum_margin_m=minimum))
        if best is None or rank < best[0]:best = (rank, answer.x.copy())
        if not rank[0]:break
    margins, names = constraints(best[1], named=True)
    failed = sorted({name for name, margin in zip(names, margins) if margin < -tolerance-1e-5})
    layout = {i: dict(center_m=best[1].reshape(-1, 3)[k].tolist(), yaw_deg=proposal[i]['yaw_deg'],
        aabb_size_m=sizes[i].tolist()) for k, i in enumerate(ids)}
    return layout, dict(passed=not failed, issues=failed, attempts=attempts,
        feasibility_search=feasibility,camera_constraints_applied=camera is not None,
        minimum_constraint_margin_m=float(margins.min()), tolerance_m=tolerance,
        displacement_m=float(np.linalg.norm(best[1]-original.ravel())),
        inventory_changed=False, intrinsic_dimensions_changed=False, yaw_changed=False,
        scope='conservative_camera_AABB_support_collision_access_proxies; level_camera_assumed')


def ray_hits(centres, sizes, directions):
    """Entry distance per box/ray; forward +z directions need not be normalized."""
    centres = np.asarray(centres).reshape(-1, 3);sizes = np.asarray(sizes).reshape(-1, 3)
    result = []
    for c, size in zip(centres, sizes):
        lo, hi = c-size/2, c+size/2
        parallel = np.abs(directions) < 1e-12
        safe = np.where(parallel, 1., directions)
        t0, t1 = lo/safe, hi/safe
        lower = np.where(parallel, -np.inf, np.minimum(t0, t1))
        upper = np.where(parallel, np.inf, np.maximum(t0, t1))
        entry = np.maximum(lower.max(1), .01);exit_ = upper.min(1)
        outside = (parallel & ((lo > 0) | (hi < 0))).any(1)
        result.append(np.where((exit_ >= entry) & ~outside, entry, np.inf))
    return np.asarray(result).reshape(len(centres), len(directions))


def intersects_frustum(center, size, hfov, image_size):
    th = math.tan(math.radians(hfov)/2);tv = th*image_size[1]/image_size[0]
    # Conservative plane test, including boxes crossing the near plane.
    normals = np.array([[0, 0, 1], [1, 0, th], [-1, 0, th], [0, 1, tv], [0, -1, tv]])
    limits = np.array([.01, 0, 0, 0, 0])
    return bool(np.all(normals@np.asarray(center) + np.abs(normals)@np.asarray(size)/2 >= limits))


def visibility_audit(state, hfov, image_size, grid=96, visible_limit=.03):
    objects = state['objects'];th = math.tan(math.radians(hfov)/2);tv = th*image_size[1]/image_size[0]
    x, y = np.meshgrid((np.arange(grid)+.5)/grid*2-1, 1-(np.arange(grid)+.5)/grid*2)
    directions = np.column_stack([x.ravel()*th, y.ravel()*tv, np.ones(grid*grid)])
    hits = ray_hits([o['center_local_m']['value'] for o in objects],
                    [o['size_aabb_local_m']['value'] for o in objects], directions)
    observed = [i for i, o in enumerate(objects) if o['local_evidence'] == 'visible']
    new = [i for i in range(len(objects)) if i not in observed]
    def opaque(o):
        name = (str(o.get('raw_category', ''))+' '+str(o.get('description', ''))).lower()
        if any(w in name for w in ('glass', 'mirror', 'window', 'curtain', 'railing', 'plant', 'chair', 'stool', 'shelf', 'table', 'desk', 'gym', 'stair')):
            return False
        if o['local_evidence'] == 'visible':
            return any(w in name for w in ('cabinet', 'wardrobe', 'chest', 'drawer', 'sofa', 'bed', 'fridge', 'refrigerator', 'counter', 'island', 'door'))
        return o['opaque_proxy']
    can_occlude = np.array([opaque(o) for o in objects])
    opaque_hits = np.where(can_occlude[:, None], hits, np.inf)
    issues = [];records = []
    for i in new:
        others = opaque_hits.copy();others[i] = np.inf
        blocker = others.min(0) if len(objects) else np.full(grid*grid, np.inf)
        projected = np.isfinite(hits[i]);front = projected & (hits[i] <= blocker+.01)
        pixels = int(projected.sum());fraction = float(front.sum()/pixels) if pixels else None
        intersects = intersects_frustum(objects[i]['center_local_m']['value'], objects[i]['size_aabb_local_m']['value'], hfov, image_size)
        visibility = objects[i]['visibility_hypothesis'];problems = []
        if visibility == 'outside_fov' and intersects:
            problems.append('declared_outside_fov_but_box_intersects_frustum')
        if visibility == 'occluded':
            if pixels < 4:problems.append('occlusion_unresolved_at_grid_resolution')
            elif fraction > visible_limit:problems.append('declared_occluded_but_proxy_rays_are_visible')
        # Hidden nearer boxes must not cover photographed visible rectangles.
        occluded_observed = []
        for j in observed:
            b = objects[j]['bbox2d_visible_xyxy_norm']['value']
            u, v = (x.ravel()+1)/2, (1-y.ravel())/2
            mask = (u >= b[0]) & (u <= b[2]) & (v >= b[1]) & (v <= b[3])
            blocked = mask & front & np.isfinite(hits[j]) & (hits[i]+.03 < hits[j])
            if mask.any() and blocked.sum()/mask.sum() > visible_limit:
                occluded_observed.append(objects[j]['track_id'])
        if occluded_observed:problems.append('addition_occludes_frozen_visible_instance')
        records.append(dict(id=objects[i]['track_id'], hypothesis=visibility, projected_grid_rays=pixels,
            visible_fraction_of_projected_rays=fraction, frustum_intersection_proxy=intersects,
            occludes_observed=occluded_observed, issues=problems))
        issues.extend(dict(id=objects[i]['track_id'], issue=p) for p in problems)
    return dict(passed=not issues, objects=records, issues=issues, grid=[grid, grid],
        scope='finite_grid_and_conservative_AABB_proxy; not_image_likelihood_or_verified_occlusion',
        transparent_and_open_structure_boxes_excluded_from_occlusion=True)
