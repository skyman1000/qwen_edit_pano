"""Explicit offset frames and articulated support-tree layout refinement.

Each object's local XZ adjustment propagates to its descendants. Other roots
in the same semantic group may move softly; they are not welded together.
Only AABB contact/overlap proxies are checked, not mesh-level stability.
"""
from copy import deepcopy
import json
import math

import numpy as np
from scipy.optimize import LinearConstraint,minimize,linprog

from . import assembly_layout as base
from .fresh_layout import relation_objective


def layout_prompt(case,observed,world,plane):
    predicted={o['id'] for o in world['objects']}
    template=dict(groups=[dict(id=gid,azimuth_deg=None,range_m=None,objects=[dict(id=o['id'],
        width_m=None,height_m=None,depth_m=None,yaw_deg=0,offset_m=[None,None,None],
        offset_frame='host_local' if o['attachment']['host'] in predicted else 'camera',
        surface_fraction=None,hanging_drop_m=None) for o in objects]) for gid,objects in base.grouped(world).items()])
    return ('Give metric attributes for ALL FIXED surrounding objects using the exact TEMPLATE. '
        'azimuth_deg: 0=original camera forward, +90=right, -90=left, 180=behind. '
        f'Choose group bearings outside the original photographed sector +/-{case["hfov"]/2:g} degrees. '
        'range_m is a positive horizontal camera distance. No room walls have been measured. '
        'width_m/height_m/depth_m are FULL intrinsic dimensions, metres. yaw_deg is ABSOLUTE about camera Y. '
        'Yaw transforms local X/Z to camera X/Z by x_camera=cos(yaw)*x_local+sin(yaw)*z_local; '
        'z_camera=-sin(yaw)*x_local+cos(yaw)*z_local. '
        'Every object explicitly declares offset_frame: camera or host_local. '
        'An unhosted object uses camera: offset_m is its relative displacement from this group centre in original camera axes. '
        'For a hosted item use host_local for displacement from the HOST CENTRE along the HOST intrinsic axes; '
        'the program rotates this offset by the HOST absolute yaw. This is distinct from the child own yaw. '
        'camera is also allowed for hosted offsets if you explicitly calculate camera-axis displacements. '
        'Observed hosts have only camera AABBs, no known yaw, so use camera for their children. '
        'Never use a global position as a relative offset. Keep support contact and account for other items sharing a surface. '
        'Floor Y uses the supplied prior; top Y uses host top + half child height. '
        'seat/shelf needs surface_fraction in [0,1] from host bottom to top; hang needs nonnegative hanging_drop_m to child TOP. '
        'Wall/ceiling/free root offset_m[1] is a signed camera-Y hypothesis; no measured mounting plane. '
        'Unused surface_fraction/hanging_drop_m may be null. Preserve every fixed identity and observed dimension. '
        'No new object, no minimum counts. Numeric refinement can move support trees and adjust hosted item contact.\n'
        'TEMPLATE: '+json.dumps(template)+'\nFIXED SEMANTIC WORLD: '+json.dumps(dict(configuration=world['configuration'],
            hypothesis=world['hypothesis'],objects=[{k:o[k] for k in ('id','name','variant','attachment','placement_hint','group')} for o in world['objects']],
            relations=world['relations']),ensure_ascii=False)+'\nFROZEN OBSERVATIONS: '+json.dumps([
                dict(id=o['track_id'],name=o.get('description'),center_m=o['center_local_m']['value'],
                    full_AABB_size_m=o['size_aabb_local_m']['value']) for o in observed['objects']],ensure_ascii=False)+
        '\nLEVEL PRIOR: '+json.dumps(plane))


def layout_answer(value,world,hfov):
    clean=base.layout_answer(value,world,hfov);contracts={o['id']:o for o in world['objects']}
    raw={o['id']:o for g in value['groups'] for o in g['objects']}
    for group in clean.values():
        for row in group['objects']:
            oid=row['id'];frame=raw[oid].get('offset_frame');host=contracts[oid]['attachment']['host']
            if frame not in ('camera','host_local') or (frame=='host_local' and host not in contracts):
                raise ValueError(oid+'.offset_frame must be explicit camera/host_local; host_local requires a predicted host with yaw. Roots and observed-host children use camera.')
            row['offset_frame']=frame
    return clean


def instantiate(world,layout,observed,plane,hfov):
    converted=deepcopy(layout);rows={o['id']:o for g in layout.values() for o in g['objects']};contracts={o['id']:o for o in world['objects']}
    for group in converted.values():
        for row in group['objects']:
            if row['offset_frame']=='host_local':
                parent=rows[contracts[row['id']]['attachment']['host']];a=math.radians(parent['yaw_deg']);x,y,z=row['offset_m']
                row['offset_m']=[math.cos(a)*x+math.sin(a)*z,y,-math.sin(a)*x+math.cos(a)*z]
    return base.instantiate(world,converted,observed,plane,hfov)


def articulated_repair(case,observed,boxes,contracts,certificates,fixed_groups,relations=(),*,hard_ordering=False,surface_proximity=False,fixed_regions=None,support_centres=False,collision_rounds=None):
    old={o['track_id']:dict(id=o['track_id'],center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    ids=list(boxes);index={oid:i for i,oid in enumerate(ids)};n=len(ids)
    # A round adds at most one separating half-space along each beam path.
    # Eight rounds cannot finish worlds needing ten independent separations.
    # This is a bounded numeric budget, not an object-count or room-size prior.
    round_budget=min(128,max(8,2*n)) if collision_rounds is None else collision_rounds
    if type(round_budget) is not int or not 1<=round_budget<=128:raise ValueError('collision_rounds must be an integer in 1..128')
    fixed_regions=fixed_regions or {}
    if set(fixed_regions)-set(ids) or any(r not in ('left','right','rear') for r in fixed_regions.values()):raise ValueError('Invalid fixed spatial-program regions')
    if not ids:return boxes,dict(method='articulated_support_tree',initial_overlaps=[],final_overlaps=[],
        intentional_host_envelope_overlaps=[],local_adjustments_XZ_m={},global_adjustments_XZ_m={},solver_converged=True)
    # Influence matrix: parent displacement transports ALL descendants, with an
    # additional local displacement variable for each child on its support.
    lift=np.zeros((2*n,2*n));remaining=set(ids)
    while remaining:
        ready=[oid for oid in ids if oid in remaining and contracts[oid]['attachment']['host'] not in remaining]
        if not ready:raise ValueError('Support-tree cycle in numeric solver')
        for oid in ready:
            j=index[oid];lift[2*j:2*j+2,2*j:2*j+2]=np.eye(2);host=contracts[oid]['attachment']['host']
            if host in index:
                k=index[host];lift[2*j:2*j+2]+=lift[2*k:2*k+2]
            remaining.remove(oid)
    initial=np.array([[boxes[oid]['center'][d] for d in (0,2)] for oid in ids]);tx=math.tan(math.radians(case['hfov'])/2)
    rows=[];rhs=[];fixed_trees={};host_region_search={}
    def influence(oid):return lift[2*index[oid]:2*index[oid]+2] if oid in index else np.zeros((2,2*n))
    def add(row,b):rows.append(row);rhs.append(b)
    for oid,b in boxes.items():
        dx,dz=influence(oid);x,_,z=b['center'];sx,_,sz=b['size'];region=certificates[b['group']]['kind']
        if oid in fixed_regions:region='behind_camera' if fixed_regions[oid]=='rear' else fixed_regions[oid]
        if region=='fixed_observed_host':
            root=oid
            while contracts[root]['attachment']['host'] in index:root=contracts[root]['attachment']['host']
            fixed_trees.setdefault(root,[]).append(oid)
        elif region=='behind_camera':add(-dz,z+sz/2+1e-5)
        else:
            sign=1 if region=='right' else -1
            add(sign*dx,sx/2-sign*x+1e-5);add(sign*dx-tx*dz,(sx+tx*sz)/2-sign*x+tx*z+1e-5)
        a=contracts[oid]['attachment'];host=a['host']
        if host and a['kind'] in ('top','seat','shelf'):
            parent=(old|boxes)[host];diff=influence(oid)-influence(host)
            for axis,dim in enumerate((0,2)):
                # Positive overlap of projected footprints. This permits
                # overhang; it does not assert full containment or stability.
                bound=(parent['size'][dim]/2 if support_centres and contracts[oid].get('support_model','rigid_box')=='rigid_box'
                    else (b['size'][dim]+parent['size'][dim])/2)-1e-5
                delta=b['center'][dim]-parent['center'][dim]
                add(diff[axis],-bound-delta);add(-diff[axis],-bound+delta)
    def feasible(matrix,target):
        fit=linprog(np.zeros(2*n),A_ub=-matrix,b_ub=-target,bounds=[(None,None)]*(2*n),method='highs')
        return fit.x if fit.success else None
    if hard_ordering:
        for relation in relations:
            kind=relation['kind']
            if kind not in ('left_of','right_of','in_front_of','behind'):continue
            source,target=relation['source'],relation['target'];axis=0 if kind in ('left_of','right_of') else 1
            dimension=(0,2)[axis];sign=1 if kind in ('right_of','in_front_of') else -1
            if source not in old|boxes or target not in old|boxes:continue
            diff=influence(source)-influence(target)
            delta=(old|boxes)[source]['center'][dimension]-(old|boxes)[target]['center'][dimension]
            add(sign*diff[axis],1e-5-sign*delta)
    # Existing hosts cannot be transported. Search the five actual frustum
    # faces at the supplied attachment height instead of assuming the sign of
    # the current centre is the only possible escape direction.
    ty=tx*case.get('image_size',[1,1])[1]/case.get('image_size',[1,1])[0]
    for root,tree in fixed_trees.items():
        options=[]
        for region in ('left','right','rear','above','below'):
            rr=[];bb=[]
            for oid in tree:
                dx,dz=influence(oid);x,y,z=boxes[oid]['center'];sx,sy,sz=boxes[oid]['size']
                if region=='rear':row,b=-dz,z+sz/2+1e-5
                elif region in ('left','right'):
                    sign=1 if region=='right' else -1
                    row,b=sign*dx-tx*dz,(sx+tx*sz)/2-sign*x+tx*z+1e-5
                else:
                    sign=1 if region=='above' else -1
                    row,b=-ty*dz,(sy+ty*sz)/2-sign*y+ty*z+1e-5
                rr.append(row);bb.append(b)
            matrix=np.array(rows+rr);target=np.array(rhs+bb);start=feasible(matrix,target)
            if start is None:continue
            fit=minimize(lambda d:float(d@d),start,jac=lambda d:2*d,constraints=LinearConstraint(matrix,target,np.inf),
                method='SLSQP',options=dict(maxiter=100,ftol=1e-9))
            d=fit.x if np.isfinite(fit.x).all() and np.min(matrix@fit.x-target,initial=0)>=-1e-6 else start
            options.append((float(d@d),region,rr,bb))
        if options:
            _,chosen,rr,bb=min(options,key=lambda p:p[0]);rows.extend(rr);rhs.extend(bb)
            host_region_search[root]=dict(selected=chosen,feasible_faces=[p[1] for p in options],
                observed_host_moved=False)
        else:
            # Preserve the contradictory candidate for diagnosis. No fallback
            # to an unhosted object or a silently relocated observed anchor.
            host_region_search[root]=dict(selected=None,feasible_faces=[],observed_host_moved=False,
                reason='No tested frustum face is compatible with the declared support tree and metric sizes.')
            for oid in tree:
                dx,dz=influence(oid);x,_,z=boxes[oid]['center'];sx,_,sz=boxes[oid]['size'];sign=1 if x>=0 else -1
                add(sign*dx-tx*dz,(sx+tx*sz)/2-sign*x+tx*z+1e-5)
    A=np.array(rows);bounds=np.array(rhs);offset=initial.ravel();rel=relation_objective(ids,old,relations,weight=.25)
    def proximity(d):
        positions=initial+(lift@d).reshape(n,2);value=0.;grad=np.zeros(2*n)
        for relation in relations:
            if relation['kind']!='near':continue
            source,target=relation['source'],relation['target']
            if source not in old|boxes or target not in old|boxes:continue
            aa,bb=(old|boxes)[source],(old|boxes)[target]
            ca=positions[index[source]] if source in index else np.array(aa['center'])[[0,2]]
            cb=positions[index[target]] if target in index else np.array(bb['center'])[[0,2]]
            delta=ca-cb;extent=(np.array(aa['size'])+np.array(bb['size']))[[0,2]]/2
            gap=np.maximum(np.abs(delta)-extent,0.);value+=float(gap@gap)
            if source in index:grad[2*index[source]:2*index[source]+2]+=2*np.sign(delta)*gap
            if target in index:grad[2*index[target]:2*index[target]+2]-=2*np.sign(delta)*gap
        return value,lift.T@grad
    def objective(d):return float(d@d)+rel(offset+lift@d)[0]
    def gradient(d):return 2*d+lift.T@rel(offset+lift@d)[1]
    def annotate(d,extra=(),converged=True):
        result=deepcopy(boxes);positions=initial+(lift@d).reshape(n,2)
        for oid,b in result.items():j=index[oid];b['center'][0]=float(positions[j,0]);b['center'][2]=float(positions[j,1])
        overlaps,intentional=base.overlap_pairs(old|result,set(ids),contracts)
        violations=int(np.count_nonzero(A@d-bounds < -1e-6))
        score=(proximity(d)[0],float(d@d)) if surface_proximity else (objective(d),)
        return dict(x=d,boxes=result,overlaps=overlaps,intentional=intentional,extra=extra,converged=converged,
            rank=(violations,len(overlaps),sum(r['volume_m3'] for r in overlaps),*score))
    solver_calls=0
    def solve(start,extra):
        nonlocal solver_calls
        solver_calls+=1
        matrix=np.vstack([A]+[r[None] for r,v in extra]);target=np.r_[bounds,[v for r,v in extra]]
        if np.min(matrix@start-target,initial=0)<-1e-6:
            start=feasible(matrix,target)
            if start is None:return None
        if surface_proximity:
            # Relations are the primary metric objective, rather than a weak
            # spring against the original guessed coordinates. No near cutoff.
            primary=minimize(lambda d:proximity(d)[0],start,jac=lambda d:proximity(d)[1],
                constraints=LinearConstraint(matrix,target,np.inf),method='SLSQP',options=dict(maxiter=200,ftol=1e-9))
            valid=np.isfinite(primary.x).all() and np.min(matrix@primary.x-target,initial=0)>=-1e-6
            if not valid:return None
            limit=proximity(primary.x)[0]+1e-7
            fit=minimize(lambda d:float(d@d),primary.x,jac=lambda d:2*d,
                constraints=[LinearConstraint(matrix,target,np.inf),dict(type='ineq',fun=lambda d:limit-proximity(d)[0],jac=lambda d:-proximity(d)[1])],
                method='SLSQP',options=dict(maxiter=150,ftol=1e-9))
            if not np.isfinite(fit.x).all() or np.min(matrix@fit.x-target,initial=0)<-1e-6 or proximity(fit.x)[0]>limit+1e-6:fit=primary
        else:
            fit=minimize(objective,start,jac=gradient,constraints=LinearConstraint(matrix,target,np.inf),method='SLSQP',
                options=dict(maxiter=200,ftol=1e-9))
        if not np.isfinite(fit.x).all() or np.min(matrix@fit.x-target,initial=0)<-1e-6:return None
        return annotate(fit.x,extra,bool(fit.success))
    zero=np.zeros(2*n);best=annotate(zero,converged=False);first=deepcopy(best);fit=solve(zero,())
    if fit is not None and fit['rank']<=best['rank']:best=fit
    beam=[best]
    if surface_proximity and fit is not None and fit is not best:
        # A closer initial solution can overlap temporarily. It must enter
        # collision repair even if the distant original is collision-free.
        beam.append(fit)
    rounds_used=0;termination='round_budget_exhausted'
    for _ in range(round_budget):
        rounds_used+=1
        expanded=[]
        for candidate in beam:
            if not candidate['overlaps']:continue
            pair=candidate['overlaps'][0];a,bid=pair['a'],pair['b'];aa=(old|boxes)[a];bb=(old|boxes)[bid]
            diff=influence(a)-influence(bid)
            for axis,dim in enumerate((0,2)):
                for sign in (-1,1):
                    row=sign*diff[axis]
                    if not np.any(row):continue
                    bound=(aa['size'][dim]+bb['size'][dim])/2+1e-4-sign*(aa['center'][dim]-bb['center'][dim])
                    proposal=solve(candidate['x'],candidate['extra']+((row,bound),))
                    if proposal is None:continue
                    expanded.append(proposal)
                    if proposal['rank']<best['rank']:best=proposal
        if not expanded:
            termination='collision_free' if not best['overlaps'] else 'frontier_exhausted';break
        if not surface_proximity and not best['overlaps']:termination='collision_free';break
        if surface_proximity and all(not c['overlaps'] for c in expanded):termination='collision_free';break
        beam=sorted(expanded,key=lambda p:p['rank'])[:4]
    if not best['overlaps']:termination='collision_free'
    return best['boxes'],dict(method='articulated_support_tree',initial_overlaps=first['overlaps'],final_overlaps=best['overlaps'],
        intentional_host_envelope_overlaps=best['intentional'],contact_and_frustum_linear_violations=best['rank'][0],
        local_adjustments_XZ_m={oid:best['x'][2*j:2*j+2].tolist() for oid,j in index.items()},
        global_adjustments_XZ_m={oid:(lift@best['x'])[2*j:2*j+2].tolist() for oid,j in index.items()},
        solver_converged=best['converged'],base_linear_system_feasible=fit is not None,
        continuous_solver_converged=best['converged'],collision_search_round_budget=round_budget,
        collision_search_rounds_used=rounds_used,collision_search_solver_calls=solver_calls,
        collision_search_termination=termination,collision_search_budget_exhausted=bool(best['overlaps']) and rounds_used>=round_budget,
        accepted_separation_constraints=len(best['extra']),
        observed_host_region_search=host_region_search,complete_search=False,observed_objects_movable=False,
        support_contact_is_AABB_proxy=True,relative_positions_are_soft=True)


def compile_layout(case,observed,world,layout,plane):
    if world.get('unresolved_installations'):raise ValueError('Unresolved installation requires a new explicit placement hypothesis; objects were not deleted')
    result=base.compile_layout(case,observed,world,layout,plane,instantiator=instantiate,repairer=articulated_repair)
    result['review'].update(offset_frames_explicit=True,host_offsets_rotated_by_host_yaw=True,
        relation_ordering_optimized_in_XZ=True,support_trees_transport_descendants=True)
    return result
