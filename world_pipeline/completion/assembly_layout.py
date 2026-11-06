"""Assembly bearings + relative poses; attachment-specific Y; rigid XZ repair.

The active v2 path in assembly_spatial.py supplies frame-aware instantiation
and an articulated repairer through compile_layout's explicit callbacks.

Only entire assembly translations are optimized. The compiler cannot separate
chairs from their table or put a hanging object above its declared anchor.
All boxes and floor priors remain camera-axis hypotheses, not measured physics.
"""
from copy import deepcopy
import itertools
import json
import math

import numpy as np
from scipy.optimize import LinearConstraint,minimize

from ..common import finite_vector
from .scene_completion import map_plan,view_check
from .fresh_layout import review_relations,relation_objective


def grouped(world):
    return {g['key']:[o for o in world['objects'] if o['group']==g['key']] for g in world['groups']
        if any(o['group']==g['key'] for o in world['objects'])}


def layout_prompt(case,observed,world,plane):
    template=dict(groups=[dict(id=gid,azimuth_deg=None,range_m=None,objects=[dict(id=o['id'],
        width_m=None,height_m=None,depth_m=None,yaw_deg=0,offset_m=[None,None,None],
        surface_fraction=None,hanging_drop_m=None) for o in objects]) for gid,objects in grouped(world).items()])
    return ('Give metric attributes for these FIXED surrounding assemblies. Return every group/object in the template exactly once. '
        'Place each assembly by azimuth_deg and range_m from the ORIGINAL camera: 0=forward, +90=right, -90=left, 180=behind. '
        f'The original photographed sector is [-{case["hfov"]/2:g},+{case["hfov"]/2:g}] degrees. '
        'Choose assembly bearings OUTSIDE that sector, preferably side/back space rather than its edge. '
        'range_m is positive horizontal camera distance; no room boundaries have been measured. '
        'width/height/depth are positive intrinsic FULL dimensions in metres; yaw_deg rotates footprint about camera Y. '
        'For root furniture, offset_m is a SMALL local displacement from the assembly centre, in camera axes x-right/y-image-up/z-forward. '
        'For a hosted object, offset_m is instead relative to its exact host centre. It is NOT a global position; '
        'keep chairs around their table and other functionally associated furniture together. '
        'Floor Y is computed from the declared prior. top Y is host top + half child height. '
        'For seat/shelf fill surface_fraction in [0,1], locating its supporting horizontal level from host bottom to top; '
        'the code adds half child height. For hang fill nonnegative hanging_drop_m from host centre to child TOP. '
        'Wall/ceiling/free roots use offset_m[1] as an explicit signed camera Y hypothesis; no mounting plane is asserted. '
        'Unused surface_fraction/hanging_drop_m may remain null. Do not add, remove, rename or resize observed objects. '
        'No category restriction. The compiler adjusts whole assemblies, preserving internal relative geometry.\n'
        'TEMPLATE: '+json.dumps(template)+'\nFIXED SEMANTIC WORLD: '+json.dumps(dict(configuration=world['configuration'],
            hypothesis=world['hypothesis'],objects=[{k:o[k] for k in ('id','name','variant','attachment','placement_hint','group')} for o in world['objects']],
            relations=world['relations']),ensure_ascii=False)+
        '\nFROZEN OBSERVATIONS: '+json.dumps([dict(id=o['track_id'],name=o.get('description'),
            center_m=o['center_local_m']['value'],full_AABB_size_m=o['size_aabb_local_m']['value']) for o in observed['objects']],ensure_ascii=False)+
        '\nLEVEL PRIOR: '+json.dumps(plane))


def layout_answer(value,world,hfov):
    expected=grouped(world);rows=value.get('groups')
    if not isinstance(rows,list) or len(rows)!=len(expected) or any(not isinstance(r,dict) for r in rows):raise ValueError('Need every group once')
    if {r.get('id') for r in rows}!=set(expected):raise ValueError('Group IDs differ from template')
    result={}
    for r in rows:
        gid=r['id'];a=r.get('azimuth_deg');distance=r.get('range_m')
        if type(a) not in (int,float) or not math.isfinite(a) or not -180<=a<=180 or abs(a)<=hfov/2+1e-6:
            raise ValueError(f'{gid}.azimuth_deg must be OUTSIDE the original photographed sector +/-{hfov/2:g}; 0=front, +/-90=side, 180=back')
        if type(distance) not in (int,float) or not math.isfinite(distance) or distance<=0:raise ValueError('range_m must be finite positive')
        objects=r.get('objects');contracts={o['id']:o for o in expected[gid]}
        if not isinstance(objects,list) or len(objects)!=len(contracts) or any(not isinstance(o,dict) for o in objects):raise ValueError('Every selected object required')
        if {o.get('id') for o in objects}!=set(contracts):raise ValueError('Copy object IDs exactly; no truncation or extra instances')
        clean=[]
        for o in objects:
            dims=[o.get(k) for k in ('width_m','height_m','depth_m')];yaw=o.get('yaw_deg',0)
            if not finite_vector(dims,3) or min(dims)<=0 or not finite_vector(o.get('offset_m'),3):raise ValueError(o['id']+' needs positive full dimensions and finite relative offsets')
            if type(yaw) not in (int,float) or not math.isfinite(yaw):raise ValueError('Invalid yaw')
            kind=contracts[o['id']]['attachment']['kind'];fraction=o.get('surface_fraction');drop=o.get('hanging_drop_m')
            if kind in ('seat','shelf') and (type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<=fraction<=1):raise ValueError('seat/shelf needs surface_fraction in [0,1]')
            if kind=='hang' and (type(drop) not in (int,float) or not math.isfinite(drop) or drop<0):raise ValueError('hang needs nonnegative hanging_drop_m')
            angle=math.radians(yaw);w,h,d=dims
            clean.append(dict(id=o['id'],size=[abs(math.cos(angle))*w+abs(math.sin(angle))*d,h,
                abs(math.sin(angle))*w+abs(math.cos(angle))*d],offset_m=list(o['offset_m']),
                surface_fraction=fraction if kind in ('seat','shelf') else None,hanging_drop_m=drop if kind=='hang' else None,
                intrinsic_size_m=dims,yaw_deg=yaw))
        result[gid]=dict(id=gid,azimuth_deg=float(a),range_m=float(distance),objects=clean)
    return result


def instantiate(world,layout,observed,plane,hfov):
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    contracts={o['id']:o for o in world['objects']};boxes={};reviews=[];certificates={};fixed_groups=set();tx=math.tan(math.radians(hfov)/2)
    for gid,group in layout.items():
        remaining={o['id']:o for o in group['objects']};local={}
        while remaining:
            ready=[oid for oid in remaining if contracts[oid]['attachment']['host'] is None or contracts[oid]['attachment']['host'] in old|local]
            if not ready:raise ValueError('Missing active host or cross-assembly attachment')
            for oid in ready:
                row=remaining.pop(oid);semantic=contracts[oid];a=semantic['attachment'];host=a['host'];kind=a['kind'];size=row['size'];offset=row['offset_m']
                center=np.array(offset,dtype=float)
                attached_to_observed=False
                if host:
                    parent=(old|local)[host];center+=parent['center']
                    attached_to_observed=host in old or parent.get('fixed_observed_host',False)
                    if attached_to_observed:fixed_groups.add(gid)
                    py,ph=parent['center'][1],parent['size'][1]
                    if kind=='top':center[1]=py+(ph+size[1])/2
                    elif kind in ('seat','shelf'):center[1]=py-ph/2+row['surface_fraction']*ph+size[1]/2
                    elif kind=='hang':center[1]=py-row['hanging_drop_m']-size[1]/2
                elif kind=='floor' and plane['floor_y_m'] is not None:center[1]=plane['floor_y_m']+size[1]/2
                local[oid]=dict(id=oid,center=center.tolist(),size=deepcopy(size),group=gid,attachment=deepcopy(a),fixed_observed_host=attached_to_observed)
                reviews.append(dict(id=oid,kind=kind,host=host,physical_attachment_verified=False,
                    height_method='host_top' if kind=='top' else 'host_fraction' if kind in ('seat','shelf') else
                    'below_hanging_anchor' if kind=='hang' else 'predicted_floor_level_prior' if kind=='floor' and plane['floor_y_m'] is not None else 'explicit_camera_y_unverified'))
        angle=math.radians(group['azimuth_deg']);sx,cz=math.sin(angle),math.cos(angle);minimum=0.
        movable={oid:b for oid,b in local.items() if not b['fixed_observed_host']}
        if movable:
            if cz<0:
                region='behind_camera'
                minimum=max([(b['size'][2]/2+b['center'][2]+1e-4)/(-cz) for b in movable.values()]+[0.])
            else:
                sign=1 if sx>0 else -1;region='right' if sign>0 else 'left';den=sign*sx-tx*cz
                if den<=0:raise ValueError('Bearing cannot certify off-frustum assembly')
                for b in movable.values():
                    x,_,z=b['center'];wx,_,wz=b['size']
                    minimum=max(minimum,(wx/2+tx*wz/2-sign*x+tx*z+1e-4)/den,(wx/2-sign*x+1e-4)/abs(sx))
            radius=max(group['range_m'],minimum)
            for b in movable.values():b['center'][0]+=radius*sx;b['center'][2]+=radius*cz
            certificates[gid]=dict(kind=region,azimuth_deg=group['azimuth_deg'],requested_range_m=group['range_m'],
                minimum_range_for_complete_assembly_m=minimum,used_range_m=radius)
        if gid in fixed_groups:
            certificates[gid]=dict(kind='fixed_observed_host',requested_range_m=group['range_m'],used_range_m=None,
                frozen_existing_host_connection=True)
        boxes.update(local)
    return boxes,reviews,certificates,fixed_groups


def overlap_pairs(boxes,new_ids,contracts):
    rows=[];intentional=[]
    for a,b in itertools.combinations(boxes,2):
        if a not in new_ids and b not in new_ids:continue
        aa,bb=boxes[a],boxes[b];extent=np.minimum(np.array(aa['center'])+np.array(aa['size'])/2,np.array(bb['center'])+np.array(bb['size'])/2)-np.maximum(
            np.array(aa['center'])-np.array(aa['size'])/2,np.array(bb['center'])-np.array(bb['size'])/2)
        if min(extent)<=1e-8:continue
        rec=dict(a=a,b=b,volume_m3=float(np.prod(extent)))
        child=contracts.get(a) if contracts.get(a,{}).get('attachment',{}).get('host')==b else contracts.get(b) if contracts.get(b,{}).get('attachment',{}).get('host')==a else None
        if child and child['attachment']['kind'] in ('seat','shelf','hang'):
            intentional.append(dict(**rec,reason='host_AABB_can_include_seat_shelf_or_hanging_contents; shape occupancy unverified'))
        else:rows.append(rec)
    return rows,intentional


def rigid_repair(case,observed,boxes,contracts,certificates,fixed_groups,relations=()):
    gids=list(certificates);index={g:i for i,g in enumerate(gids)};old={o['track_id']:dict(id=o['track_id'],center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    if not boxes:return boxes,dict(method='rigid_assembly_translation',initial_overlaps=[],final_overlaps=[],intentional_host_envelope_overlaps=[],translations_XZ_m={},internal_geometry_preserved=True,solver_converged=True)
    tx=math.tan(math.radians(case['hfov'])/2);rows=[];rhs=[]
    for oid,b in boxes.items():
        gid=b['group'];j=index[gid];x,y,z=b['center'];sx,sy,sz=b['size'];region=certificates[gid]['kind']
        def add(dx,dz,v):
            row=np.zeros(2*len(gids));row[2*j:2*j+2]=[dx,dz];rows.append(row);rhs.append(v)
        if region=='behind_camera':add(0,-1,z+sz/2+1e-5)
        elif region in ('left','right'):
            sign=1 if region=='right' else -1
            add(sign,0,sx/2-sign*x+1e-5);add(sign,-tx,(sx+tx*sz)/2-sign*x+tx*z+1e-5)
        else:
            # An existing-host component cannot be translated independently of
            # that frozen observed host. Its original box is checked below.
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):add(dx,dz,0)
    A=np.array(rows);b=np.array(rhs)
    ids=list(boxes);offset=np.array([[boxes[i]['center'][d] for d in (0,2)] for i in ids]).ravel()
    lift=np.zeros((2*len(ids),2*len(gids)))
    for i,oid in enumerate(ids):
        j=index[boxes[oid]['group']];lift[2*i:2*i+2,2*j:2*j+2]=np.eye(2)
    relation_cost=relation_objective(ids,old,relations,weight=.25)
    def objective(d):return float(d@d)+relation_cost(offset+lift@d)[0]
    def gradient(d):return 2*d+lift.T@relation_cost(offset+lift@d)[1]
    def shifted(d):
        result=deepcopy(boxes)
        for item in result.values():j=index[item['group']];item['center'][0]+=float(d[2*j]);item['center'][2]+=float(d[2*j+1])
        return result
    def annotate(d,extra=()):
        candidate=shifted(d);overlap,intentional=overlap_pairs(old|candidate,set(candidate),contracts)
        return dict(x=d,boxes=candidate,overlaps=overlap,intentional=intentional,extra=extra,
            rank=(len(overlap),sum(r['volume_m3'] for r in overlap),objective(d)))
    zero=np.zeros(2*len(gids));best=annotate(zero);initial=deepcopy(best['overlaps']);beam=[best];converged=True
    initial_fit=minimize(objective,zero,jac=gradient,constraints=LinearConstraint(A,b,np.inf),
        method='SLSQP',options=dict(maxiter=150,ftol=1e-9))
    if np.isfinite(initial_fit.x).all() and np.min(A@initial_fit.x-b,initial=0)>=-1e-6:
        fit_candidate=annotate(initial_fit.x)
        if fit_candidate['rank']<best['rank']:best=fit_candidate;beam=[best];converged=bool(initial_fit.success)
    for _ in range(6):
        expanded=[]
        for candidate in beam:
            movable=[p for p in candidate['overlaps'] if not (p['a'] in boxes and p['b'] in boxes and boxes[p['a']]['group']==boxes[p['b']]['group'])]
            if not movable:continue
            pair=movable[0];a,bid=pair['a'],pair['b'];aa=(old|boxes)[a];bb=(old|boxes)[bid]
            for axis,dim in ((0,0),(1,2)):
                for sign in (-1,1):
                    row=np.zeros(2*len(gids));bound=(aa['size'][dim]+bb['size'][dim])/2+1e-4-sign*(aa['center'][dim]-bb['center'][dim])
                    if a in boxes:row[2*index[boxes[a]['group']]+axis]+=sign
                    if bid in boxes:row[2*index[boxes[bid]['group']]+axis]-=sign
                    extra=candidate['extra']+((row,bound),);mat=np.vstack([A]+[r[None] for r,v in extra]);vec=np.r_[np.array(rhs),[v for r,v in extra]]
                    fit=minimize(objective,candidate['x'],jac=gradient,
                        constraints=LinearConstraint(mat,vec,np.inf),method='SLSQP',options=dict(maxiter=150,ftol=1e-9))
                    if not np.isfinite(fit.x).all() or np.min(mat@fit.x-vec,initial=0)<-1e-6:continue
                    proposal=annotate(fit.x,extra);expanded.append(proposal)
                    if proposal['rank']<best['rank']:best=proposal;converged=bool(fit.success)
        if not expanded:break
        beam=sorted(expanded,key=lambda c:c['rank'])[:4]
    return best['boxes'],dict(method='rigid_assembly_translation',initial_overlaps=initial,final_overlaps=best['overlaps'],
        intentional_host_envelope_overlaps=best['intentional'],translations_XZ_m={g:best['x'][2*j:2*j+2].tolist() for g,j in index.items()},
        internal_geometry_preserved=True,solver_converged=converged,complete_search=False)


def compile_layout(case,observed,world,layout,plane,*,instantiator=None,repairer=None):
    contracts={o['id']:o for o in world['objects']}
    boxes,attachments,certificates,fixed=(instantiator or instantiate)(world,layout,observed,plane,case['hfov'])
    boxes,solver=(repairer or rigid_repair)(case,observed,boxes,contracts,certificates,fixed,world['relations'])
    plan=dict(scene_type='conditional existing surrounding world',scene_hypothesis=world['configuration'],additions=[
        dict(id=o['id'],name=o['name'],object_type=o['category'],variant=o['variant'],reason=o['distinct_instance_reason'],placement=o['placement_hint']) for o in world['objects']])
    mappings=map_plan(plan,'semantic_lexicon_v1');mapping={m['id']:m['category'] for m in mappings}
    items=[dict(id=o['id'],plan_id=o['id'],center=boxes[o['id']]['center'],size=boxes[o['id']]['size'],category=mapping[o['id']],
        reason=o['variant'],unseen_reason='outside_fov') for o in world['objects']]
    checks=[dict(id=o['id'],**view_check(o['center'],o['size'],case['hfov'],case['image_size'])) for o in items]
    relations=review_relations(world['relations'],items,observed)
    hard=sum(c['intersects_input_box_proxy'] or c['camera_inside_box_proxy'] for c in checks)
    unresolved=[r['id'] for r in attachments if r['height_method']=='explicit_camera_y_unverified' or r['kind'] in ('wall','ceiling','seat','shelf','hang')]
    byid={o['id']:o for o in items};old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    for row in attachments:
        row['horizontal_contact_proxy_passed']=None
        if row['host'] and row['kind'] in ('top','seat','shelf'):
            child=byid[row['id']];parent=(old|byid)[row['host']]
            extent=np.minimum(np.array(child['center'])+np.array(child['size'])/2,np.array(parent['center'])+np.array(parent['size'])/2)-np.maximum(
                np.array(child['center'])-np.array(child['size'])/2,np.array(parent['center'])-np.array(parent['size'])/2)
            row['horizontal_contact_proxy_passed']=bool(min(extent[[0,2]])>0)
    contact_failures=sum(r['horizontal_contact_proxy_passed'] is False for r in attachments)
    review=dict(geometry_contract_passed=hard==0,hard_violations=hard,objects=checks,attachment_review=attachments,
        unresolved_installations=unresolved,declared_relation_review=relations,assembly_certificates=certificates,
        program_solver=solver,observed_state_preserved=True,physical_geometry_verified=False,
        layout_proxy_passed=hard==0 and not solver['final_overlaps'] and not unresolved and not contact_failures and all(r['ordering_or_presence_satisfied'] is not False for r in relations),
        attachment_contact_failures=contact_failures,relation_ordering_optimized_in_XZ=True,near_has_no_invented_hard_threshold=True,declared_support_plane=plane)
    return dict(plan=plan,items=items,mappings=mappings,review=review)
