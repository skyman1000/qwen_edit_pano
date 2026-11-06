"""Closest feasible off-frustum initialization of a fixed support assembly.

The LM origin is a numeric proposal, not a certificate. Project only the group
translation onto the union of left/right/rear AABB half-spaces. No object,
dimension, orientation or relative support offset is changed here.
"""
from itertools import combinations
from copy import deepcopy
import math

import numpy as np


def closest_halfspaces(point,A,b):
    A=np.asarray(A,dtype=float);b=np.asarray(b,dtype=float);p=np.asarray(point,dtype=float)
    candidates=[]
    if np.all(A@p>=b-1e-9):candidates.append(p)
    for n in range(1,min(2,len(b))+1):
        for subset in combinations(range(len(b)),n):
            matrix=A[list(subset)];rhs=b[list(subset)]
            if np.linalg.matrix_rank(matrix)<n:continue
            x=p+matrix.T@np.linalg.solve(matrix@matrix.T,rhs-matrix@p)
            if np.all(A@x>=b-1e-8):candidates.append(x)
    if not candidates:raise ValueError('No finite feasible assembly initialization')
    return min(candidates,key=lambda x:float((x-p)@(x-p)))


def initialize(origin,rows,contracts,observed,hfov):
    old={o['track_id']:dict(center=np.array(o['center_local_m']['value'])[[0,2]],fixed=True)
        for o in observed['objects']}
    byid={r['id']:r for r in rows};local={};remaining=list(byid)
    while remaining:
        ready=[oid for oid in remaining if not contracts[oid]['attachment']['host'] or contracts[oid]['attachment']['host'] in old|local]
        if not ready:raise ValueError('Missing support-tree parent in numeric initialization')
        for oid in ready:
            row=byid[oid];host=contracts[oid]['attachment']['host'];offset=np.array(row['offset_m'])[[0,2]]
            if row['offset_frame']=='host_local':
                angle=math.radians(byid[host]['yaw_deg']);x,z=offset
                offset=np.array([math.cos(angle)*x+math.sin(angle)*z,-math.sin(angle)*x+math.cos(angle)*z])
            parent=(old|local).get(host);center=offset+(parent['center'] if parent else 0.)
            angle=math.radians(row['yaw_deg']);w,d=row['width_m'],row['depth_m']
            local[oid]=dict(center=center,size=np.array([abs(math.cos(angle))*w+abs(math.sin(angle))*d,
                abs(math.sin(angle))*w+abs(math.cos(angle))*d]),fixed=bool(parent and parent['fixed']))
            remaining.remove(oid)
    moving=[v for v in local.values() if not v['fixed']]
    if not moving:
        # The legacy compiler ignores bearing/range for fixed-observed-host
        # trees. These adapter values do not place or move any physical object.
        return dict(azimuth_deg=90.,range_m=1.),dict(method='fixed_observed_host_origin_unused',
            requested_center_xz_m=origin['center_xz_m'],objects_moved=False)
    tx=math.tan(math.radians(hfov)/2);epsilon=1e-3;options=[];p=np.array(origin['center_xz_m'])
    for name,sign in [('right',1.),('left',-1.)]:
        b1=max([0.]+[v['size'][0]/2-sign*v['center'][0] for v in moving])+epsilon
        b2=max([0.]+[(v['size'][0]+tx*v['size'][1])/2-sign*v['center'][0]+tx*v['center'][1] for v in moving])+epsilon
        point=closest_halfspaces(p,[[sign,0.],[sign,-tx]],[b1,b2]);options.append((name,point))
    bz=max([0.]+[v['size'][1]/2+v['center'][1] for v in moving])+epsilon
    options.append(('rear',closest_halfspaces(p,[[0.,-1.]],[bz])))
    region,point=min(options,key=lambda v:float((v[1]-p)@(v[1]-p)))
    record=dict(method='closest_union_of_off_frustum_assembly_halfspaces',region=region,
        requested_center_xz_m=p.tolist(),feasible_center_xz_m=point.tolist(),translation_xz_m=(point-p).tolist(),
        changed=bool(np.linalg.norm(point-p)>1e-8),dimensions_changed=False,relative_offsets_changed=False,
        alternatives=[dict(region=n,squared_translation_m2=float((x-p)@(x-p))) for n,x in options],
        scope='initialization_only_final_collision_contact_and_original_camera_checks_still_required')
    return dict(azimuth_deg=math.degrees(math.atan2(point[0],point[1])),range_m=float(np.linalg.norm(point))),record


def instantiate(world,layout,observed,plane,hfov):
    """Use the chosen Cartesian point AND half-space without a second projection.

    A negative origin Z does not imply that the whole assembly was placed in
    the rear half-space: positive relative Z offsets can put its objects to the
    side. Reinterpreting that point as a rear ray caused 30--115 metre layouts.
    """
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    contracts={o['id']:o for o in world['objects']};boxes={};attachments=[];certificates={};fixed_groups=set()
    rows={o['id']:o for g in layout.values() for o in g['objects']}
    for gid,group in layout.items():
        proof=group.get('origin_initialization')
        if not proof:raise ValueError('Missing Cartesian initialization certificate: '+gid)
        remaining={o['id']:o for o in group['objects']};local={}
        while remaining:
            ready=[oid for oid in remaining if not contracts[oid]['attachment']['host'] or contracts[oid]['attachment']['host'] in old|local]
            if not ready:raise ValueError('Missing active host or support-tree cycle')
            for oid in ready:
                row=remaining.pop(oid);a=contracts[oid]['attachment'];host=a['host'];kind=a['kind'];size=row['size']
                center=np.asarray(row['offset_m'],dtype=float).copy();fixed=False
                if row['offset_frame']=='host_local':
                    yaw=math.radians(rows[host]['yaw_deg']);x,y,z=center
                    center=np.array([math.cos(yaw)*x+math.sin(yaw)*z,y,-math.sin(yaw)*x+math.cos(yaw)*z])
                if host:
                    parent=(old|local)[host];center+=parent['center'];fixed=host in old or parent.get('fixed_observed_host',False)
                    py,ph=parent['center'][1],parent['size'][1]
                    if kind=='top':center[1]=py+(ph+size[1])/2
                    elif kind in ('seat','shelf'):center[1]=py-ph/2+row['surface_fraction']*ph+size[1]/2
                    elif kind=='hang':center[1]=py-row['hanging_drop_m']-size[1]/2
                elif kind=='floor' and plane['floor_y_m'] is not None:center[1]=plane['floor_y_m']+size[1]/2
                local[oid]=dict(id=oid,center=center.tolist(),size=deepcopy(size),group=gid,attachment=deepcopy(a),fixed_observed_host=fixed)
                if fixed:fixed_groups.add(gid)
                attachments.append(dict(id=oid,kind=kind,host=host,physical_attachment_verified=False,
                    height_method='host_top' if kind=='top' else 'host_fraction' if kind in ('seat','shelf') else
                    'below_hanging_anchor' if kind=='hang' else 'predicted_floor_level_prior' if kind=='floor' and plane['floor_y_m'] is not None else 'explicit_camera_y_unverified'))
        moving=[b for b in local.values() if not b['fixed_observed_host']]
        if moving:
            point=proof['feasible_center_xz_m']
            for b in moving:b['center'][0]+=point[0];b['center'][2]+=point[1]
            certificates[gid]=dict(kind='behind_camera' if proof['region']=='rear' else proof['region'],
                azimuth_deg=group['azimuth_deg'],requested_range_m=group['range_m'],used_range_m=group['range_m'],
                cartesian_origin_xz_m=list(point),coordinate_transport='direct_cartesian_no_radial_reinterpretation')
        if gid in fixed_groups:certificates[gid]=dict(kind='fixed_observed_host',frozen_existing_host_connection=True,
            coordinate_transport='direct_cartesian_no_radial_reinterpretation')
        boxes.update(local)
    return boxes,attachments,certificates,fixed_groups
