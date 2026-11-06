"""Deterministic joint XZ layout of a fixed inventory; no learned parameters.

Hard constraints: frozen observations/sizes, declared support height, assumed
central support point, and one sufficient off-frustum certificate per support
component. AABB repulsion is SOFT and is never a physical collision certificate.
No room boundary is invented. Support/region search is approximate and recorded.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import LinearConstraint,linprog,minimize

from .scm_typed_layout import initial_boxes
from .scm_world_layout import support_answer

REGIONS=('left','right','behind_camera','above','below')
REPULSION_WEIGHT=4.
CENTERING_WEIGHT=.25


class LayoutInfeasible(ValueError):
    def __init__(self,message,component_ids):
        super().__init__(message);self.component_ids=tuple(component_ids)


def support_heights(plan,bodies,old,plane):
    y={};byid={o['id']:o for o in plan['additions']}
    for oid in plan['placement_dependency_order']:
        obj=byid[oid];s=bodies[oid]['size'];edge=obj['support']
        if edge['kind']=='floor':
            if plane['floor_y_m'] is None:raise ValueError('Floor prior unavailable')
            value=plane['floor_y_m']+s[1]/2
        elif edge['kind']=='object':
            target=edge['target']
            hy=y[target] if target in y else old[target]['center'][1]
            hs=bodies[target]['size'] if target in bodies else old[target]['size']
            value=hy+(hs[1]+s[1])/2
        else:
            value=bodies[oid]['center_y_m']
            if value is None:raise ValueError('Unmodeled installation requires explicit signed camera-Y')
        y[oid]=float(value)
    return y


def system(ids,plan,bodies,old,heights,case,regions,separations=()):
    """All inequalities are A @ flattened_XZ >= b."""
    index={oid:i for i,oid in enumerate(ids)};rows=[];rhs=[]
    byid={o['id']:o for o in plan['additions']}
    tx=math.tan(math.radians(case['hfov'])/2);ty=tx*case['image_size'][1]/case['image_size'][0]
    def add(terms,bound):
        a=np.zeros(2*len(ids))
        for oid,axis,coefficient in terms:a[2*index[oid]+axis]+=coefficient
        rows.append(a);rhs.append(bound)
    for oid in ids:
        sx,sy,sz=bodies[oid]['size'];region=regions[oid];edge=byid[oid]['support']
        if region=='behind_camera':add([(oid,1,-1)],sz/2+1e-4)
        elif region in ('left','right'):
            sign=-1 if region=='left' else 1
            add([(oid,0,sign)],sx/2+1e-4)
            add([(oid,0,sign),(oid,1,-tx)],(sx+tx*sz)/2+1e-4)
        elif region in ('above','below'):
            sign=-1 if region=='below' else 1
            if sign*heights[oid]<sy/2+1e-4:return None,None
            add([(oid,1,-ty)],(sy+ty*sz)/2+1e-4-sign*heights[oid])
        else:raise ValueError('Unknown view certificate')
        if edge['kind']=='object':
            host=edge['target']
            for axis,dim in ((0,0),(1,2)):
                for sign in (-1,1):
                    terms=[(oid,axis,sign)]
                    if host in index:
                        terms.append((host,axis,-sign));bound=-bodies[host]['size'][dim]/2
                    else:
                        bound=sign*old[host]['center'][dim]-old[host]['size'][dim]/2
                    # Central contact point in the host footprint, NOT full
                    # child-box containment or a measured center-of-mass test.
                    add(terms,bound)
    for a,b,axis,sign in separations:
        dim=0 if axis==0 else 2;terms=[(a,axis,sign)]
        bs=bodies[b]['size'] if b in index else old[b]['size']
        bound=(bodies[a]['size'][dim]+bs[dim])/2+1e-5
        if b in index:terms.append((b,axis,-sign))
        else:bound+=sign*old[b]['center'][dim]
        add(terms,bound)
    return np.asarray(rows).reshape(-1,2*len(ids)),np.asarray(rhs)


def quadratic(ids,plan,bodies,old):
    index={oid:i for i,oid in enumerate(ids)}
    byid={o['id']:o for o in plan['additions']};centers={i:[b['center'][0],b['center'][2]] for i,b in old.items()}
    for oid in plan['placement_dependency_order']:
        if oid not in index:continue
        edge=byid[oid]['support'];body=bodies[oid]
        # Accessory proposals were elicited BEFORE host selection. They are
        # not a conditional global-location prior and must not drag the host
        # across the room. Compose the child's initial center from its host.
        centers[oid]=(list(centers[edge['target']]) if edge['kind']=='object' else
            [body['range_m']*math.sin(math.radians(body['canonical_azimuth_deg'])),
             body['range_m']*math.cos(math.radians(body['canonical_azimuth_deg']))])
    proposal=np.array([centers[i] for i in ids]).ravel()
    weights=np.repeat([float(byid[i]['support']['kind']!='object') for i in ids],2)
    matrix=np.diag(weights);target=weights*proposal
    for obj in plan['additions']:
        oid=obj['id'];edge=obj['support']
        if oid not in index or edge['kind']!='object':continue
        host=edge['target']
        for axis,dim in ((0,0),(1,2)):
            a=np.zeros(len(proposal));a[2*index[oid]+axis]=1.
            if host in index:a[2*index[host]+axis]=-1.;value=0.
            else:value=old[host]['center'][dim]
            matrix+=CENTERING_WEIGHT*np.outer(a,a);target+=CENTERING_WEIGHT*value*a
    def energy(x):return float((weights*(x-proposal))@(x-proposal)+
        sum(CENTERING_WEIGHT*((x[2*index[o['id']]+ax]-(x[2*index[o['support']['target']]+ax]
            if o['support']['target'] in index else old[o['support']['target']]['center'][dim]))**2)
            for o in plan['additions'] if o['id'] in index and o['support']['kind']=='object'
            for ax,dim in ((0,0),(1,2))))
    return proposal,matrix,target,energy


def solve_qp(ids,plan,bodies,old,heights,case,regions,separations=()):
    A,b=system(ids,plan,bodies,old,heights,case,regions,separations)
    if A is None:return None
    proposal,matrix,target,energy=quadratic(ids,plan,bodies,old)
    start=proposal
    if np.min(A@start-b,initial=0.) < -1e-8:
        start=linear_feasible(A,b)
        if start is None:return None
    fit=minimize(lambda x:float(x@matrix@x-2*target@x),start,
        jac=lambda x:2*(matrix@x-target),constraints=LinearConstraint(A,b,np.inf),
        method='SLSQP',options=dict(maxiter=200,ftol=1e-10))
    valid=np.isfinite(fit.x).all() and np.min(A@fit.x-b,initial=0.)>=-1e-7
    # Optimization failure is not evidence that the support is impossible.
    # Retain the independently feasible start and explicitly require review.
    chosen=fit.x if valid else start
    return dict(x=chosen,energy=energy(chosen),converged=bool(fit.success and valid),
        residual_min=float(np.min(A@chosen-b,initial=0.)),separations=tuple(separations))


def linear_feasible(A,b):
    result=linprog(np.zeros(A.shape[1]),A_ub=-A,b_ub=-b,bounds=[(None,None)]*A.shape[1],method='highs')
    if result.status==2:return None
    if not result.success:raise RuntimeError('LP feasibility solver failed: '+str(result.message))
    if not np.isfinite(result.x).all() or np.min(A@result.x-b,initial=0.) < -1e-7:
        raise RuntimeError('LP solver returned an invalid feasibility certificate')
    return result.x


def overlaps(ids,x,bodies,old,heights):
    new={oid:dict(center=[float(x[2*i]),heights[oid],float(x[2*i+1])],size=bodies[oid]['size']) for i,oid in enumerate(ids)}
    boxes={**old,**new};pairs=[]
    for i,a in enumerate(ids):
        for b in list(old)+ids[i+1:]:
            aa,bb=boxes[a],boxes[b]
            lengths=[min(aa['center'][d]+aa['size'][d]/2,bb['center'][d]+bb['size'][d]/2)-
                max(aa['center'][d]-aa['size'][d]/2,bb['center'][d]-bb['size'][d]/2) for d in range(3)]
            if min(lengths)<=1e-8:continue
            penetration=[(aa['size'][d]+bb['size'][d])/2-abs(aa['center'][d]-bb['center'][d]) for d in (0,2)]
            pairs.append(dict(a=a,b=b,volume_m3=math.prod(lengths),
                fraction_of_smaller_box=math.prod(lengths)/min(math.prod(aa['size']),math.prod(bb['size'])),
                repulsion_cost=min(penetration)**2))
    return sorted(pairs,key=lambda p:(-p['repulsion_cost'],p['a'],p['b']))


def compile_joint(case,observed,plan,plane,bodies,selections,beam_width=4,rounds=10):
    ids=[o['id'] for o in plan['additions']]
    if set(bodies)!=set(ids) or set(selections)!=set(ids):raise ValueError('Fixed inventory differs')
    rows=[dict(id=i,placement='Outside original input frustum; joint layout.',
        support={k:selections[i][k] for k in ('kind','target')}) for i in ids]
    out=support_answer(dict(objects=rows),plan,case);old=initial_boxes(case,observed)
    if not ids:return out,dict(objects=[]),dict(method='joint_XZ_soft_AABB_repulsion_v1',regions={},
        initial_overlaps=[],final_overlaps=[],energy_before=0.,energy_after=0.,solver_calls=0,
        physical_geometry_verified=False,room_bounds_known=False)
    heights=support_heights(out,bodies,old,plane);edges={o['id']:o['support'] for o in out['additions']}
    roots={}
    for oid in ids:
        current=oid
        while edges[current]['target'] in edges:current=edges[current]['target']
        roots.setdefault(current,[]).append(oid)
    regions={};calls=0
    for root,component in roots.items():
        possible=[]
        for region in REGIONS:
            calls+=1;chosen={i:region for i in component}
            fit=solve_qp(component,out,bodies,old,heights,case,chosen)
            if fit is not None:possible.append((fit['energy'],REGIONS.index(region),chosen))
        if not possible:raise LayoutInfeasible('No central-support-point off-view layout for support component',component)
        regions.update(min(possible,key=lambda r:(r[0],r[1]))[2])
    start=solve_qp(ids,out,bodies,old,heights,case,regions);calls+=1
    if start is None:raise LayoutInfeasible('Independent feasible components failed joint feasibility',ids)
    def score(candidate):
        candidate['overlaps']=overlaps(ids,candidate['x'],bodies,old,heights)
        candidate['score']=candidate['energy']+REPULSION_WEIGHT*sum(p['repulsion_cost'] for p in candidate['overlaps'])
        return candidate
    start=score(start);best=start;beam=[start];seen={()}
    # Search separating halfspaces as candidate moves. The unchanged candidate
    # remains eligible; unresolved AABB overlap is exported as a review signal.
    for _ in range(rounds):
        expanded=[]
        for candidate in beam:
            if not candidate['overlaps']:continue
            pair=candidate['overlaps'][0]
            for axis,sign in ((0,-1),(0,1),(1,-1),(1,1)):
                extra=tuple(sorted(candidate['separations']+((pair['a'],pair['b'],axis,sign),)))
                if extra in seen:continue
                seen.add(extra);calls+=1
                fit=solve_qp(ids,out,bodies,old,heights,case,regions,extra)
                if fit is None:continue
                fit=score(fit);expanded.append(fit)
                if fit['score']<best['score']-1e-10:best=fit
        if not expanded:break
        beam=sorted(expanded,key=lambda c:(c['score'],c['separations']))[:beam_width]
    raw=[]
    for i,oid in enumerate(ids):
        x,z=best['x'][2*i:2*i+2]
        raw.append(dict(id=oid,azimuth_deg=math.degrees(math.atan2(x,z)),range_m=math.hypot(x,z),
            size=deepcopy(bodies[oid]['size']),center_y_m=heights[oid],reason=bodies[oid]['reason']))
    for obj in out['additions']:obj['placement']='Outside original input frustum: '+regions[obj['id']]
    review=dict(method='joint_XZ_soft_AABB_repulsion_v1',regions=regions,
        initial_overlaps=start['overlaps'],final_overlaps=best['overlaps'],
        energy_before=start['score'],energy_after=best['score'],
        proposal_and_support_energy=best['energy'],repulsion_weight=REPULSION_WEIGHT,
        support_centering_weight=CENTERING_WEIGHT,central_contact_point_assumed=True,
        solver_converged=best['converged'],constraint_residual_min=best['residual_min'],
        selected_separations=[list(s) for s in best['separations']],solver_calls=calls,
        beam_width=beam_width,rounds=rounds,all_discrete_regions_searched_jointly=False,
        proposal_contract='root_global_proposals_and_support_relative_child_centers_v1',
        child_global_proposals_not_used_as_absolute_priors=[i for i in ids if edges[i]['kind']=='object'],
        physical_geometry_verified=False,room_bounds_known=False)
    return out,dict(objects=raw),review


def observed_host_feasible(oid,body,host,old,plane,case):
    plan=dict(additions=[dict(id=oid,support=dict(kind='object',target=host))],placement_dependency_order=[oid])
    bodies={oid:body};heights=support_heights(plan,bodies,old,plane)
    for region in REGIONS:
        A,b=system([oid],plan,bodies,old,heights,case,{oid:region})
        if A is not None and linear_feasible(A,b) is not None:return True
    return False
