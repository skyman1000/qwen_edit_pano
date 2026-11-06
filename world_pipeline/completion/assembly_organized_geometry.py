"""Frozen dimensions, typed host coordinates, and joint sibling packing.

MILP changes only XZ offsets of siblings under the existing support-box proxy.
Dimensions, yaw, identities, hosts and sampled camera regions stay fixed.
"""
from copy import deepcopy
from collections import defaultdict
import itertools
import json
import math

import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint

from ..common import read,write
from . import assembly_compositional_geometry as base,assembly_parametric as parametric

PARAMETERIZATION='frozen_dimensions_typed_host_coordinates_joint_packing_v13'
layout_prompt=base.layout_prompt
layout_answer=base.layout_answer
annotate_state=base.annotate_state


def number(v):return type(v) in (int,float) and math.isfinite(v)
def vector(v,n):return isinstance(v,list) and len(v)==n and all(number(x) for x in v)


def dimensions_prompt(world):
    return ('Estimate intrinsic FULL dimensions for EVERY fixed object. '
        'This stage has no positions to fit: do not shrink objects to eliminate overlap or satisfy visibility. '
        'Use each concrete variant, not a generic category cube. Metres, order [width,height,depth]. '
        'For an open laptop, include its raised screen height and base depth; for a writing desk, include usable desk height. '
        'Dimensions will be frozen before joint position/orientation inference. Do not add/remove/rename objects. '
        'Return only the exact JSON template with finite positive dimensions. '
        '\nDIMENSION_TEMPLATE: '+json.dumps(dict(objects=[dict(id=o['id'],full_size_m=[None,None,None]) for o in world['objects']]))+
        '\nFIXED_OBJECTS: '+json.dumps([{k:o[k] for k in ('id','name','category','variant','attachment','placement_hint')} for o in world['objects']],ensure_ascii=False))


def dimensions_answer(value,world):
    rows=value.get('objects') if isinstance(value,dict) else None;expected={o['id'] for o in world['objects']}
    if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):raise ValueError('objects must contain every fixed ID once')
    ids=[r.get('id') for r in rows]
    if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids) or set(ids)!=expected:raise ValueError('Exact fixed dimension identities required')
    errors=[]
    for r in rows:
        if not vector(r.get('full_size_m'),3) or min(r['full_size_m'])<=0:errors.append(r['id']+'.full_size_m: positive FULL metric vector required')
    if errors:raise ValueError(json.dumps(errors))
    return {r['id']:dict(id=r['id'],full_size_m=list(r['full_size_m'])) for r in rows}


def placement_template(world):
    groups=[]
    for g in world['groups']:
        records=[]
        for o in world['objects']:
            if o['group']!=g['key']:continue
            a=o['attachment'];r=dict(id=o['id'],yaw_deg=None)
            if a['host']:
                r['host_offset_uv']=[None,None]
                if a['kind'] not in ('top','seat','shelf','hang'):r['host_offset_y_m']=None
            else:r['root_offset_m']=[None,None,None]
            if a['kind'] in ('seat','shelf'):r['surface_fraction']=None
            if a['kind']=='hang':r['hanging_drop_m']=None
            records.append(r)
        groups.append(dict(id=g['key'],range_m=None,objects=records))
    return dict(groups=groups)


def placement_prompt(case,observed,world,plane,dimensions):
    hosts=defaultdict(list)
    for o in world['objects']:
        if o['attachment']['host']:hosts[o['attachment']['host']].append(o['id'])
    return ('Place and orient the ENTIRE fixed world inventory jointly, using already frozen dimensions. '
        'Return exactly TEMPLATE; no dimensions, new identities, or region choices. '
        'yaw_deg is absolute camera-Y yaw; x=cos(yaw)*u+sin(yaw)*v, z=-sin(yaw)*u+cos(yaw)*v. '
        'Choose furniture and belongings orientations jointly with their positions and functional relations. '
        'For a ROOT, root_offset_m is a displacement from its group origin in camera axes, not absolute XYZ. '
        'For a HOSTED object, host_offset_uv is dimensionless [u,v]: dx=u*HOST_WIDTH/2, dz=v*HOST_DEPTH/2. '
        'Zero is the HOST centre, 1 is its positive footprint edge, -1 its negative edge. '
        'Use the predicted host intrinsic axes; the compiler applies host yaw. Observed host yaw is unknown: use its camera AABB axes. '
        'Do not paste absolute observed coordinates into u,v. Arrange ALL items sharing a host together; do not put all at [0,0]. '
        'Overhang is not automatically forbidden or physically certified. Preserve each declared host. '
        'The compiler derives floor/top Y; shelf/seat require surface_fraction in [0,1], hang requires nonnegative hanging_drop_m. '
        'Other hosted kinds require host_offset_y_m. Root wall/ceiling/free Y is a signed camera-Y hypothesis, not a measured wall. '
        'range_m is positive camera distance, not known room extent. Respect the fixed ORIGINAL CAMERA region for every group. '
        'Functional service and camera-depth ordering are distinct. Keep concrete furniture-use relations coherent. '
        '\nTEMPLATE: '+json.dumps(placement_template(world))+
        '\nFROZEN_DIMENSIONS: '+json.dumps(list(dimensions.values()))+
        '\nSAME_HOST_ITEMS: '+json.dumps(dict(hosts))+
        '\nFIXED_SEMANTIC_WORLD: '+json.dumps(dict(hypothesis=world['hypothesis'],objects=world['objects'],relations=world['relations'],spatial_program=world.get('spatial_program',{})),ensure_ascii=False)+
        '\nFROZEN_OBSERVATIONS: '+json.dumps([dict(id=o['track_id'],name=o['description'],predicted_center_m=o['center_local_m']['value'],predicted_full_camera_AABB_m=o['size_aabb_local_m']['value']) for o in observed['objects']],ensure_ascii=False)+
        '\nLEVEL_PRIOR: '+json.dumps(plane))


def _placement_parts(value,world,observed,dimensions):
    rows=value.get('groups') if isinstance(value,dict) else None;template=placement_template(world)
    if not isinstance(rows,list) or any(not isinstance(g,dict) for g in rows):raise ValueError('All fixed groups required')
    gids=[g.get('id') for g in rows];expected={g['id']:{o['id'] for o in g['objects']} for g in template['groups']}
    if any(not isinstance(g,str) for g in gids) or len(set(gids))!=len(gids) or set(gids)!=set(expected):raise ValueError('Exact group IDs required')
    contracts={o['id']:o for o in world['objects']};obs={o['track_id']:o['size_aabb_local_m']['value'] for o in observed['objects']};converted=[];errors=[];missing=[]
    for g in rows:
        if not number(g.get('range_m')) or g['range_m']<=0:missing.append(g['id'])
        items=g.get('objects')
        if not isinstance(items,list) or any(not isinstance(o,dict) for o in items):errors.append(g['id']+': object list required');continue
        ids=[o.get('id') for o in items]
        if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids) or set(ids)!=expected[g['id']]:errors.append(g['id']+': exact object membership required');continue
        values=[]
        for r in items:
            oid=r['id'];a=contracts[oid]['attachment'];host=a['host'];kind=a['kind'];row=deepcopy(dimensions[oid])
            if not number(r.get('yaw_deg')):errors.append(oid+'.yaw_deg: finite absolute orientation required');continue
            row['yaw_deg']=r['yaw_deg']
            if host:
                uv=r.get('host_offset_uv')
                if not vector(uv,2):errors.append(oid+'.host_offset_uv: two finite dimensionless coordinates required');continue
                size=dimensions[host]['full_size_m'] if host in dimensions else obs.get(host)
                if size is None:errors.append(oid+': missing frozen host dimensions');continue
                y=r.get('host_offset_y_m',0.)
                if kind not in ('top','seat','shelf','hang') and not number(r.get('host_offset_y_m')):errors.append(oid+'.host_offset_y_m required');continue
                if not number(y):errors.append(oid+'.host_offset_y_m must be finite when supplied');continue
                row['offset_m']=[uv[0]*size[0]/2,y,uv[1]*size[2]/2]
            else:
                if not vector(r.get('root_offset_m'),3):errors.append(oid+'.root_offset_m: three finite relative metres required');continue
                row['offset_m']=list(r['root_offset_m'])
            if kind in ('seat','shelf'):
                v=r.get('surface_fraction')
                if not number(v) or not 0<=v<=1:errors.append(oid+'.surface_fraction in [0,1] required')
                row['surface_fraction']=v
            if kind=='hang':
                v=r.get('hanging_drop_m')
                if not number(v) or v<0:errors.append(oid+'.hanging_drop_m must be nonnegative')
                row['hanging_drop_m']=v
            values.append(row)
        converted.append(dict(id=g['id'],range_m=g.get('range_m'),objects=values))
    if errors:raise ValueError('Repair all typed placement errors without changing dimensions or inventory: '+json.dumps(errors))
    return converted,missing


def placement_answer(value,world,observed,dimensions,hfov):
    converted,missing=_placement_parts(value,world,observed,dimensions)
    if missing:raise ValueError('Finite positive range_m required for groups: '+json.dumps(missing))
    return base.layout_answer(dict(groups=converted),world,hfov)


def placement_capture(value,world,observed,dimensions,hfov):
    """Accept an otherwise valid partial reply without fabricating coordinates.

    This tagged result must never reach the geometry compiler. Only a separate,
    strictly validated range patch can complete it. Other field errors still fail.
    """
    _,missing=_placement_parts(value,world,observed,dimensions)
    if missing:return dict(pending_ranges=deepcopy(value),missing_group_ids=missing)
    return placement_answer(value,world,observed,dimensions,hfov)


def ranges_prompt(case,observed,world,plane,dimensions,partial,missing):
    return ('Complete ONLY the requested group distances in the preserved placement. '
        'Return exactly RANGE_TEMPLATE, replacing null with finite positive metres. '
        'Do not return groups, objects, yaw, offsets, sizes or explanations. '
        'range_m is the positive horizontal distance of a group origin from the ORIGINAL input camera; it is not room extent. '
        'Choose distances jointly with the frozen original-camera regions, offsets, dimensions, hosts and functional relations. '
        'Do not relocate or remove objects. A range cannot establish room boundaries or certify physical feasibility. '
        '\nRANGE_TEMPLATE: '+json.dumps(dict(ranges=[dict(id=g,range_m=None) for g in missing]))+
        '\nCAMERA: '+json.dumps(dict(hfov=case['hfov'],image_size=case['image_size']))+
        '\nPRESERVED_PLACEMENT: '+json.dumps(partial,ensure_ascii=False)+
        '\nFROZEN_DIMENSIONS: '+json.dumps(list(dimensions.values()))+
        '\nFIXED_SEMANTIC_WORLD: '+json.dumps(world,ensure_ascii=False)+
        '\nFROZEN_OBSERVATIONS: '+json.dumps(observed,ensure_ascii=False)+
        '\nLEVEL_PRIOR: '+json.dumps(plane))


def ranges_answer(value,missing):
    if not isinstance(value,dict) or set(value)!={'ranges'} or not isinstance(value['ranges'],list):raise ValueError('Return only {ranges:[{id,range_m}]}')
    rows=value['ranges']
    if any(not isinstance(r,dict) or set(r)!={'id','range_m'} for r in rows):raise ValueError('Each range row must contain only id and range_m')
    ids=[r['id'] for r in rows]
    if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids) or set(ids)!=set(missing):raise ValueError('Exact requested group IDs once required: '+json.dumps(missing))
    if any(not number(r['range_m']) or r['range_m']<=0 for r in rows):raise ValueError('Every range_m must be finite positive metres, not null/bool')
    return {r['id']:r['range_m'] for r in rows}


def apply_ranges(partial,ranges,world,observed,dimensions,hfov):
    _,missing=_placement_parts(partial,world,observed,dimensions)
    patch=ranges_answer(dict(ranges=[dict(id=k,range_m=v) for k,v in ranges.items()]),missing)
    completed=deepcopy(partial)
    for g in completed['groups']:
        if g['id'] in patch:g['range_m']=patch[g['id']]
    # All other fields, including already valid group distances, stay exact.
    stripped=deepcopy(completed)
    for a,b in zip(partial['groups'],stripped['groups']):
        if a['id'] in patch:
            if 'range_m' in a:b['range_m']=a['range_m']
            else:b.pop('range_m',None)
    if stripped!=partial:raise AssertionError('Range patch changed a frozen placement field')
    return placement_answer(completed,world,observed,dimensions,hfov),completed


def infer_layout(case,observed,world,plane,board,ask,dest,index):
    dims=ask(f'layout_{index:02d}_dimensions',None,dimensions_prompt(world),lambda v:dimensions_answer(v,world))
    write(dest/'dimension_stage.json',dict(dimensions=dims,dimensions_frozen=True))
    layout=ask(f'layout_{index:02d}_placement',board,placement_prompt(case,observed,world,plane,dims),
        lambda v:placement_capture(v,world,observed,dims,case['hfov']))
    if 'pending_ranges' in layout:
        partial=layout['pending_ranges'];missing=layout['missing_group_ids']
        write(dest/'range_pending.json',dict(placement=partial,missing_group_ids=missing))
        patch=ask(f'layout_{index:02d}_ranges',board,ranges_prompt(case,observed,world,plane,dims,partial,missing),lambda v:ranges_answer(v,missing))
        layout,completed=apply_ranges(partial,patch,world,observed,dims,case['hfov'])
        write(dest/'range_patch.json',dict(ranges=patch,completed_placement=completed,only_requested_ranges_changed=True))
    for o in (o for g in layout.values() for o in g['objects']):
        assert o['intrinsic_size_m']==dims[o['id']]['full_size_m']
    return layout


def pack_rectangles(centres,sizes,lower,upper,vertical_intervals):
    """Joint camera-XZ AABB separation with minimal L1 centre displacement."""
    n=len(centres);pairs=[(i,j) for i,j in itertools.combinations(range(n),2)
        if min(vertical_intervals[i][1],vertical_intervals[j][1])-max(vertical_intervals[i][0],vertical_intervals[j][0])>1e-8]
    if not pairs:return None,dict(status='no_vertical_conflict')
    if np.any(np.asarray(lower)>np.asarray(upper)):return None,dict(status='infeasible',reason='Existing support-centre bounds are inconsistent',solver_status=2)
    initial_ok=all(all(lower[i][k]-1e-7<=centres[i][k]<=upper[i][k]+1e-7 for k in range(2)) for i in range(n))
    separated=lambda i,j:any(abs(centres[i][k]-centres[j][k])>=(sizes[i][k]+sizes[j][k])/2+1e-4 for k in range(2))
    if initial_ok and all(separated(i,j) for i,j in pairs):return None,dict(status='already_separated')
    nv=4*n+4*len(pairs);c=np.zeros(nv);c[2*n:4*n]=1.;lo=np.full(nv,-np.inf);hi=np.full(nv,np.inf)
    lo[:2*n]=np.asarray(lower).ravel();hi[:2*n]=np.asarray(upper).ravel();lo[2*n:4*n]=0.;lo[4*n:]=0.;hi[4*n:]=1.
    integer=np.zeros(nv);integer[4*n:]=1.;matrix=[];bounds=[]
    def add(row,b):matrix.append(row);bounds.append(b)
    for i in range(n):
        for k in range(2):
            j=2*i+k
            for sign in (-1,1):
                row=np.zeros(nv);row[j]=sign;row[2*n+j]=1.;add(row,sign*centres[i][k])
    for p,(i,j) in enumerate(pairs):
        for d,(axis,sign) in enumerate(((0,-1),(0,1),(1,-1),(1,1))):
            sep=(sizes[i][axis]+sizes[j][axis])/2+1e-4
            smallest=lower[i][axis]-upper[j][axis] if sign==1 else lower[j][axis]-upper[i][axis]
            big_m=sep-smallest;row=np.zeros(nv);row[2*i+axis]=sign;row[2*j+axis]=-sign;row[4*n+4*p+d]=-big_m
            add(row,sep-big_m)
        row=np.zeros(nv);row[4*n+4*p:4*n+4*p+4]=1.;add(row,1.)
    A=np.asarray(matrix);b=np.asarray(bounds)
    fit=milp(c,integrality=integer,bounds=Bounds(lo,hi),constraints=LinearConstraint(A,b,np.inf),options=dict(time_limit=3.,node_limit=10000,mip_rel_gap=0.))
    x=fit.x;valid=x is not None and np.isfinite(x).all() and np.min(A@x-b,initial=0.)>=-1e-6 and np.min(x-lo,initial=0.)>=-1e-6 and np.min(hi-x,initial=0.)>=-1e-6 and np.max(abs(x[4*n:]-np.rint(x[4*n:])),initial=0.)<1e-6
    report=dict(status='feasible' if valid else 'infeasible' if fit.status==2 else 'no_feasible_incumbent',solver_status=int(fit.status),optimal=bool(fit.success),
        pairs=len(pairs),objective_L1_m=float(fit.fun) if valid else None,search_time_limit_s=3.,physical_stability_verified=False)
    return (x[:2*n].reshape(n,2) if valid else None),report


def joint_host_pack(case,observed,world,layout,plane):
    revised=deepcopy(layout);contracts={o['id']:o for o in world['objects']};rows={o['id']:o for g in revised.values() for o in g['objects']}
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']};hosts=defaultdict(list)
    for o in world['objects']:
        if o['attachment']['host'] and o['attachment']['kind'] in ('top','seat','shelf'):hosts[o['attachment']['host']].append(o['id'])
    def depth(h):
        value=0
        while h in contracts:h=contracts[h]['attachment']['host'];value+=1
        return value
    reports=[]
    for host in sorted(hosts,key=lambda h:(depth(h),h)):
        ids=sorted(hosts[host])
        if len(ids)<2:continue
        boxes,_,_,_=parametric.instantiate(world,revised,observed,plane,case['hfov']);parent=(old|boxes)[host];pc=np.asarray(parent['center']);ps=parent['size']
        centres=[(np.asarray(boxes[i]['center'])-pc)[[0,2]].tolist() for i in ids];sizes=[[boxes[i]['size'][k] for k in (0,2)] for i in ids]
        limits=[]
        for i in ids:
            rigid=contracts[i].get('support_model','rigid_box')=='rigid_box'
            limits.append([(ps[k]/2 if rigid else (ps[k]+boxes[i]['size'][k])/2)-1e-5 for k in (0,2)])
        intervals=[(boxes[i]['center'][1]-boxes[i]['size'][1]/2,boxes[i]['center'][1]+boxes[i]['size'][1]/2) for i in ids]
        packed,report=pack_rectangles(centres,sizes,(-np.asarray(limits)).tolist(),limits,intervals);report.update(host=host,children=ids)
        if packed is not None:
            yaw=math.radians(rows[host]['yaw_deg']) if host in rows else 0.;co,si=math.cos(yaw),math.sin(yaw)
            for oid,p in zip(ids,packed):
                x,z=p;previous=rows[oid]['offset_m'];rows[oid]['offset_m']=[float(co*x-si*z),previous[1],float(si*x+co*z)]
            report['camera_relative_centres_before_m']=centres;report['camera_relative_centres_after_m']=packed.tolist()
        reports.append(report)
    assert {o['id']:(o['intrinsic_size_m'],o['yaw_deg']) for g in revised.values() for o in g['objects']}=={o['id']:(o['intrinsic_size_m'],o['yaw_deg']) for g in layout.values() for o in g['objects']}
    return revised,reports


def compile_layout(case,observed,world,layout,plane):
    packed,packing=joint_host_pack(case,observed,world,layout,plane)
    result=base.compile_layout(case,observed,world,packed,plane);selected='joint_host_packing';attempts=[]
    def rank(s):
        r=s['review'];p=r['program_solver'];return (not r['geometry_contract_passed'],r['hard_violations'],len(p['final_overlaps']),sum(v['volume_m3'] for v in p['final_overlaps']))
    attempts.append(dict(layout=selected,rank=list(rank(result))))
    if packed!=layout and not result['review']['geometry_contract_passed']:
        raw=base.compile_layout(case,observed,world,layout,plane);attempts.append(dict(layout='original_offsets',rank=list(rank(raw))))
        if rank(raw)<rank(result):result=raw;selected='original_offsets'
    result['review'].update(geometry_parameterization=PARAMETERIZATION,joint_host_packing=packing,numeric_candidates=attempts,
        selected_numeric_candidate=selected,inventory_pruned=False,dimensions_changed=False,host_identity_changed=False,
        packing_scope='Joint sibling camera-AABB separation under existing support-centre bounds; no mesh or usability certification')
    return result
