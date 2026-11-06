"""Relative scene-program geometry; fixed sizes/observations, off-frustum boxes.

Enumerate several view certificates per group, then jointly repair AABB overlap
with a lexicographic numerical proxy. This is an approximate layout compiler,
not a sampler with a metric probability density or a physical collision proof.
"""
from copy import deepcopy
import itertools
import math

import numpy as np
from scipy.optimize import LinearConstraint,minimize

from .scm_joint_geometry import REGIONS,linear_feasible,overlaps,system
from .scm_typed_layout import initial_boxes


def relative_quadratic(ids,bodies,old):
    index = {i:j for j,i in enumerate(ids)}; matrix = np.zeros((2*len(ids),2*len(ids))); target = np.zeros(2*len(ids)); terms = []
    for oid in ids:
        body = bodies[oid]; anchor = body['anchor']
        for axis,dim in ((0,0),(1,2)):
            a = np.zeros(2*len(ids)); a[2*index[oid]+axis] = 1.
            value = body['offset_m'][dim]
            if anchor in index: a[2*index[anchor]+axis] -= 1.
            elif anchor != 'camera': value += old[anchor]['center'][dim]
            matrix += np.outer(a,a); target += value*a; terms.append((a,value))
    def energy(x): return float(sum((a@x-b)**2 for a,b in terms))
    start = np.linalg.solve(matrix,target)
    return matrix,target,energy,start


def solve(ids,plan,bodies,old,heights,case,regions,separations=()):
    A,b = system(ids,plan,bodies,old,heights,case,regions,separations)
    if A is None: return None
    matrix,target,energy,start = relative_quadratic(ids,bodies,old)
    if np.min(A@start-b,initial=0.) < -1e-8:
        start = linear_feasible(A,b)
        if start is None: return None
    fit = minimize(lambda x:float(x@matrix@x-2*target@x),start,
        jac=lambda x:2*(matrix@x-target),constraints=LinearConstraint(A,b,np.inf),
        method='SLSQP',options=dict(maxiter=200,ftol=1e-10))
    valid = np.isfinite(fit.x).all() and np.min(A@fit.x-b,initial=0.) >= -1e-7
    x = fit.x if valid else start
    return dict(x=x,energy=energy(x),converged=bool(fit.success and valid),
        residual_min=float(np.min(A@x-b,initial=0.)),separations=tuple(separations),regions=dict(regions))


def compile_program(case,observed,plan,bodies,groups,placements,region_keep=3,beam_width=4,rounds=8):
    ids = [o['id'] for o in plan['additions']]; old = initial_boxes(case,observed)
    if not ids:
        return dict(objects=[]),dict(method='relative_group_program_compiler_v1',regions={},
            initial_overlaps=[],final_overlaps=[],relative_energy=0.,mean_relative_energy=0.,solver_calls=0,
            solver_converged=True,displacement_XZ_m={},constraint_residual_min=0.,
            metric_probability_density_defined=False,physical_geometry_verified=False)
    if set(ids)!=set(bodies) or set(ids)!=set(groups) or set(ids)!=set(placements): raise ValueError('Inventory mismatch')
    if not 1<=region_keep<=5 or beam_width<1 or rounds<0: raise ValueError('Invalid layout search budget')
    heights = {i:placements[i][1] for i in ids}; bygroup = {}
    for oid in ids: bygroup.setdefault(groups[oid],[]).append(oid)
    alternatives = []; calls = 0
    # Recipe references are local to a group, or refer to frozen observations.
    # Therefore each group's feasible certificates can be checked independently.
    for group,local in bygroup.items():
        local_plan = dict(additions=[o for o in plan['additions'] if o['id'] in local])
        local_bodies = {i:bodies[i] for i in local}; fits = []
        for region in REGIONS:
            calls += 1
            fit = solve(local,local_plan,local_bodies,old,heights,case,{i:region for i in local})
            if fit is not None: fits.append(fit)
        if not fits:
            raise ValueError('No off-frustum layout for complete group '+group+'; recipe preserved, no objects deleted')
        alternatives.append(sorted(fits,key=lambda f:(f['energy'],tuple(f['regions'].values())))[:region_keep])
    def annotate(fit):
        fit['overlaps'] = overlaps(ids,fit['x'],bodies,old,heights)
        fit['rank'] = (len(fit['overlaps']),sum(p['fraction_of_smaller_box'] for p in fit['overlaps']),fit['energy'])
        return fit
    seeds = []
    for combination in itertools.product(*alternatives):
        regions = {i:r for fit in combination for i,r in fit['regions'].items()}; calls += 1
        fit = solve(ids,plan,bodies,old,heights,case,regions)
        if fit is not None: seeds.append(annotate(fit))
    if not seeds: raise RuntimeError('Independent feasible group certificates lost joint feasibility')
    seeds.sort(key=lambda f:f['rank']); best = seeds[0]; initial = deepcopy(best['overlaps'])
    beam = seeds[:beam_width]; seen = set()
    for _ in range(rounds):
        expanded = []
        for candidate in beam:
            if not candidate['overlaps']: continue
            pair = candidate['overlaps'][0]
            for axis,sign in ((0,-1),(0,1),(1,-1),(1,1)):
                extra = tuple(sorted(candidate['separations']+((pair['a'],pair['b'],axis,sign),)))
                signature = (tuple(sorted(candidate['regions'].items())),extra)
                if signature in seen: continue
                seen.add(signature); calls += 1
                fit = solve(ids,plan,bodies,old,heights,case,candidate['regions'],extra)
                if fit is None: continue
                fit = annotate(fit); expanded.append(fit)
                if fit['rank'] < best['rank']: best = fit
        if not expanded: break
        beam = sorted(expanded,key=lambda f:f['rank'])[:beam_width]
    raw = []; displacement = {}; relative_residuals = []
    index = {oid:i for i,oid in enumerate(ids)}
    for i,oid in enumerate(ids):
        x,z = map(float,best['x'][2*i:2*i+2]); body=bodies[oid]; anchor=body['anchor']
        ref = [0.,0.] if anchor=='camera' else ([old[anchor]['center'][d] for d in (0,2)] if anchor in old else
            best['x'][2*index[anchor]:2*index[anchor]+2])
        residual = [x-ref[0]-body['offset_m'][0],z-ref[1]-body['offset_m'][2]]
        relative_residuals.append(dict(id=oid,anchor=anchor,residual_XZ_m=list(map(float,residual))))
        displacement[oid] = math.dist([x,z],[placements[oid][0],placements[oid][2]])
        raw.append(dict(id=oid,azimuth_deg=math.degrees(math.atan2(x,z)),range_m=math.hypot(x,z),
            size=deepcopy(body['size']),center_y_m=heights[oid],reason=body['reason']))
    review = dict(method='relative_group_program_compiler_v1',regions=best['regions'],
        initial_overlaps=initial,final_overlaps=best['overlaps'],relative_energy=best['energy'],
        mean_relative_energy=best['energy']/len(ids),relative_residuals=relative_residuals,
        displacement_XZ_m=displacement,solver_calls=calls,solver_converged=best['converged'],
        constraint_residual_min=best['residual_min'],selected_separations=[list(s) for s in best['separations']],
        candidate_region_combinations=len(seeds),region_keep=region_keep,beam_width=beam_width,rounds=rounds,
        rank_policy='lexicographic_AABB_overlap_pairs_then_overlap_fraction_then_relative_energy',
        complete_search=False,room_bounds_known=False,gravity_measured=False,
        metric_probability_density_defined=False,physical_geometry_verified=False)
    return dict(objects=raw),review
