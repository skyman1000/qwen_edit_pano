"""Continuous group origins; frozen inventory and dimensions; strict proxy review."""
from copy import deepcopy
from functools import partial
import json
import math

from . import assembly_layout as base, assembly_spatial as spatial
from . import assembly_named_geometry as named, assembly_organized_geometry as organized
from .joint_world_model import finite,concrete
from .joint_world_initialization import initialize,instantiate as cartesian_instantiate
from .conditional_world_geometry import support_plane


def floor_answer(value,observed,require_resolved=False):
    ids=value.get('anchor_ids');floor=value.get('floor_y_m');basis=value.get('basis')
    known={o['track_id'] for o in observed['objects']}
    if not isinstance(ids,list) or len(ids)!=len(set(ids)) or not set(ids)<=known:raise ValueError('Use distinct exact observed IDs as floor anchors')
    if not isinstance(basis,str) or not basis.strip():raise ValueError('Explain uncertain floor evidence in basis')
    if require_resolved:concrete(basis)
    if ids:
        if floor is not None:raise ValueError('With observed anchors, floor_y_m must be null; the program computes their median box bottom')
        plane=support_plane(observed,ids)
    else:
        if floor is not None and (not finite(floor) or floor>=0):raise ValueError('Explicit floor hypothesis needs a finite camera-Y below the camera, or null')
        plane=dict(kind='explicit_unmeasured_camera_Y_floor_hypothesis',floor_y_m=floor,anchor_ids=[],floor_measured=False,gravity_measured=False)
    if plane['floor_y_m'] is not None and plane['floor_y_m']>=0:raise ValueError('Selected box bottoms do not define a floor below the camera')
    plane.update(basis=basis,floor_measured=False,gravity_measured=False,uncertainty='predicted_boxes_camera_tilt_and_floor_assumptions_not_calibrated')
    if require_resolved and plane['floor_y_m'] is None:
        raise ValueError('Unresolved floor: this fixed inventory has floor roots. Select defensible exact observed floor anchors, or state a finite negative unmeasured camera-Y floor hypothesis with its basis. If genuinely unknown, keep null and this world stays blocked; never invent a measurement.')
    return plane


def compact_world(world):
    active={o['group'] for o in world['objects']}
    return dict(hypothesis=world['hypothesis'],configuration=world['configuration'],
        groups=[dict(key=g['key'],function=g['function']) for g in world['groups'] if g['key'] in active],
        objects=[{k:o[k] for k in ('id','category','variant','attachment','placement_hint','group')} for o in world['objects']],
        relations=world['relations'])


def metric_observations(case):
    return [{k:r[k] for k in ('id','category','description','center','size') if k in r} for r in case['context']]


def floor_candidates(observed):
    """Finite predicted-bottom hypotheses, without a furniture/category whitelist."""
    result={}
    for obj in observed['objects']:
        oid=obj['track_id'];plane=support_plane(observed,[oid]);y=plane['floor_y_m']
        if finite(y) and y<0:
            result['base_'+oid]=dict(anchor_ids=[oid],floor_y_m=y,
                object=obj.get('description',obj.get('raw_category','')),
                assumption='This observed object bottom directly contacts an approximately camera-Y-level floor.')
    return result


def floor_choice_answer(value,observed,candidates):
    choice=value.get('choice');basis=value.get('basis')
    if not isinstance(basis,str) or not basis.strip():raise ValueError('Give concrete evidence and uncertainty for floor selection')
    concrete(basis)
    if choice=='unresolved':raise ValueError('Unresolved floor selection; floor-supported world remains blocked, not assigned a default camera height')
    if not isinstance(choice,str) or choice not in candidates:
        raise ValueError('Choose one exact FLOOR_CHOICES ID or unresolved: '+json.dumps(list(candidates)))
    plane=floor_answer(dict(anchor_ids=candidates[choice]['anchor_ids'],floor_y_m=None,basis=basis),observed,require_resolved=True)
    plane.update(selection=choice,selection_scope='finite_predicted_bottom_hypotheses_not_measured_floor',
        candidate_count=len(candidates))
    return plane


def floor_hypothesis(case,observed,world,board,ask):
    if not any(o['attachment']['kind']=='floor' for o in world['objects']):
        return dict(floor_y_m=None,kind='no_new_floor_roots',anchor_ids=[])
    candidates=floor_candidates(observed)
    if candidates:
        prompt=('Resolve a camera-Y-level FLOOR HYPOTHESIS for this fixed world. '
            'Select one exact FLOOR_CHOICES ID for a predicted object bottom plausibly resting DIRECTLY on the floor. '
            'Judge from RGB and object function: table-supported items and wall-mounted items are not floor anchors. '
            'The code already computed each candidate height from frozen 3D boxes; do not predict new coordinates or modify measurements. '
            'Every candidate is an uncertain approximately level-floor assumption, not a floor measurement or known camera height. '
            'Return JSON {"choice":"","basis":""}, filling both strings with an exact candidate ID and concrete evidence/uncertainty. '
            'If none is defensible choose "unresolved"; this blocks the floor-supported world rather than inventing a measurement. '
            '\nFLOOR_CHOICES: '+json.dumps(candidates,ensure_ascii=False)+
            '\nOBSERVED: '+json.dumps(metric_observations(case),ensure_ascii=False)+
            '\nHYPOTHESIS: '+json.dumps(world['hypothesis'],ensure_ascii=False))
        return ask('floor',board,prompt,lambda v:floor_choice_answer(v,observed,candidates),2048)
    prompt=('Resolve a camera-Y-level FLOOR HYPOTHESIS for this fixed world. This is metric inference AFTER semantic inventory, not a measured floor. '
        'Return {"anchor_ids":[exact selected observed IDs],"floor_y_m":null,"basis":a concrete explanation} '
        'OR {"anchor_ids":[],"floor_y_m":a finite negative number,"basis":a concrete uncertain-height assumption}. '
        'Prefer exact observed IDs for objects whose BOTTOM actually rests on the floor, e.g. furniture legs, not items supported by a table. '
        'When anchor_ids is nonempty leave floor_y_m null: the program computes the median predicted box bottom. '
        'If no observed bottom is a defensible anchor, supply an explicit uncertain finite negative floor_y_m based on RGB and metric boxes, '
        'or leave null when unresolved. Do not assume camera Y=0 is the floor or invent a standard camera height. '
        'Camera is origin, Y up, metres. Do not add objects, repair geometry, or change observed measurements. '
        '\nOBSERVED: '+json.dumps(metric_observations(case),ensure_ascii=False)+
        '\nNEW_FLOOR_ROOTS: '+json.dumps([dict(id=o['id'],variant=o['variant']) for o in world['objects'] if o['attachment']['kind']=='floor']))
    return ask('floor',board,prompt,lambda v:floor_answer(v,observed,require_resolved=True),2048)


def origins_answer(value,world,hfov,requested=None):
    expected=set(requested) if requested is not None else {o['group'] for o in world['objects']};rows=value.get('groups')
    if not isinstance(rows,list) or len(rows)!=len(expected) or {r.get('id') for r in rows}!=expected:
        raise ValueError('Return exactly these group IDs once: '+json.dumps(sorted(expected)))
    result={}
    for r in rows:
        v=r.get('center_xz_m')
        if not isinstance(v,list) or len(v)!=2 or not all(map(finite,v)):
            raise ValueError('Finite numeric XZ origin required')
        angle=math.degrees(math.atan2(v[0],v[1]))
        result[r['id']]=dict(id=r['id'],center_xz_m=list(v),azimuth_deg=angle,range_m=math.hypot(*v))
    return result


def placements_answer(value,objects):
    from .assembly_coordinate_transport import response
    if set(value)=={'placements'}:value=dict(objects=value['placements'])
    result=response(value,dict(key='requested_poses'),objects)
    expected={o['id'] for o in objects}
    if result['errors'] or set(result['rows'])!=expected:
        raise ValueError('Exact frozen pose IDs required: '+json.dumps(dict(expected=sorted(expected),accepted=sorted(result['rows']),errors=result['errors'])))
    return result['rows']


def convert(world,observed,dimensions,origins,poses,hfov):
    old={o['track_id']:o['size_aabb_local_m']['value'] for o in observed['objects']};groups=[];proofs={}
    contracts={o['id']:o for o in world['objects']}
    for gid,origin in origins.items():
        rows=[]
        for obj in world['objects']:
            if obj['group']!=gid:continue
            oid=obj['id'];r=poses[oid];host=obj['attachment']['host'];w,h,d=dimensions[oid]['full_size_m']
            if host:
                size=dimensions[host]['full_size_m'] if host in dimensions else old[host]
                u,v=r['host_offset_uv'];offset=[u*size[0]/2,r.get('host_offset_y_m',0.),v*size[2]/2]
            else:offset=r['root_offset_m']
            row=dict(id=oid,width_m=w,height_m=h,depth_m=d,yaw_deg=r['yaw_deg'],offset_m=offset,
                offset_frame='host_local' if host in dimensions else 'camera')
            for k in ('surface_fraction','hanging_drop_m'):
                if k in r:row[k]=r[k]
            rows.append(row)
        feasible,proof=initialize(origin,rows,contracts,observed,hfov);proofs[gid]=proof
        groups.append(dict(id=gid,**feasible,objects=rows))
    layout=spatial.layout_answer(dict(groups=groups),world,hfov)
    for gid in layout:layout[gid]['origin_initialization']=proofs[gid]
    return layout


def infer(case,observed,world,dimensions,plane,board,ask):
    prompt=('Propose a continuous origin for the REQUESTED_GROUP in this ONE complete world. Return '
        'the exact ORIGIN_TEMPLATE with both null values replaced by finite numbers. '
        'Continuous metres, original camera X right, Z forward. Do not snap to cardinal angles. '
        'Prefer unobserved space. This is an initialization; a geometric solver, not this point, certifies whole-box visibility. '
        'Keep functional clusters; no direction coverage quota or uniform distribution objective. '
        'Use the global spatial hypothesis, frozen observed positions and dimensions. '
        'Do not invent measured room boundaries. All active groups coexist. '
        '\nHFOV: '+str(case['hfov'])+'\nWORLD: '+json.dumps(compact_world(world),ensure_ascii=False)+
        '\nDIMENSIONS: '+json.dumps(dimensions)+'\nOBSERVED: '+json.dumps(metric_observations(case)))
    origins={}
    for gid in dict.fromkeys(o['group'] for o in world['objects']):
        request=prompt+'\nREQUESTED_GROUP: '+gid+'\nORIGIN_TEMPLATE: '+json.dumps(dict(groups=[dict(id=gid,center_xz_m=[None,None])]))
        origins.update(ask('origins_'+gid,board,request,lambda v,g=gid:origins_answer(v,world,case['hfov'],[g]),2048))
    poses={};roots=[o for o in world['objects'] if not o['attachment']['host']]
    # Root poses precede hosted poses. Batches retain the entire world and all
    # accepted poses; transport capacity does not split the semantic program.
    batches=[roots[i:i+6] for i in range(0,len(roots),6)]
    hosted=[o for o in world['objects'] if o['attachment']['host']]
    for gid in origins:
        rows=[o for o in hosted if o['group']==gid]
        for i in range(0,len(rows),6):batches.append(rows[i:i+6])
    for i,objects in enumerate(batches):
        prompt=('Joint metric poses for the requested IDs in a fixed surrounding world. Return '
            '{"placements":[records with EXACT required fields]}. No dimensions or identities may change. '
            'yaw_deg is absolute camera-Y yaw. root_offset_m is [dx,dy,dz] from its zone origin, '
            'camera axes, not absolute XYZ. floor Y is derived from the explicit floor hypothesis. '
            'host_offset_uv are normalized intrinsic host offsets: dx=u*host_width/2, dz=v*host_depth/2; '
            'host yaw rotates these offsets. Observed hosts use camera AABB axes. '
            'Top height is derived; shelf/seat needs surface_fraction, hang needs hanging_drop_m. '
            'Choose functional orientation and spacing jointly across zones; zero yaw/offset is allowed only '
            'if appropriate. Preserve same-host packing, relations and fixed observed geometry. '
            '\nREQUIRED_FIELDS: '+json.dumps({o['id']:named.contract(o) for o in objects})+
            '\nGLOBAL_WORLD: '+json.dumps(compact_world(world),ensure_ascii=False)+'\nORIGINS: '+json.dumps(origins)+
            '\nFROZEN_DIMENSIONS: '+json.dumps(dimensions)+'\nFROZEN_POSES: '+json.dumps(poses)+
            '\nFLOOR_HYPOTHESIS: '+json.dumps(plane)+'\nOBSERVED: '+json.dumps(metric_observations(case)))
        poses.update(ask('poses_'+str(i),board,prompt,lambda v:placements_answer(v,objects)))
    layout=convert(world,observed,dimensions,origins,poses,case['hfov'])
    return layout,dict(origins=origins,poses=poses,origin_initialization={g:v['origin_initialization'] for g,v in layout.items()},
        dimensions_frozen=True,inventory_changed=False)


def compile_layout(case,observed,world,layout,plane,policy='approximate'):
    if policy not in ('approximate','strict'):raise ValueError('Unknown layout policy')
    # Packing adjusts positions, never population, host identities or frozen sizes.
    packed,packing=organized.joint_host_pack(case,observed,world,layout,plane)
    solved=base.compile_layout(case,observed,world,packed,plane,instantiator=cartesian_instantiate,
        repairer=partial(spatial.articulated_repair,hard_ordering=policy=='strict',surface_proximity=policy=='strict',support_centres=True))
    review=solved['review'];review['host_packing']=packing;errors=[]
    if not review['geometry_contract_passed']:errors.append('original_camera_visibility_or_camera_intersection')
    if review['program_solver']['final_overlaps']:errors.append('remaining_AABB_collisions')
    if review['attachment_contact_failures']:errors.append('host_contact_failure')
    boxes={o['id']:o for o in solved['items']}
    boxes.update({o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']})
    for obj in world['objects']:
        a=obj['attachment'];b=boxes[obj['id']]
        if a['host'] and a['kind'] in ('top','seat','shelf'):
            parent=boxes[a['host']]
            if any(abs(b['center'][axis]-parent['center'][axis])>parent['size'][axis]/2+2e-5 for axis in (0,2)):
                errors.append('unsupported_box_center:'+obj['id'])
        if a['kind']=='floor' and (plane['floor_y_m'] is None or abs(b['center'][1]-b['size'][1]/2-plane['floor_y_m'])>1e-5):
            errors.append('unresolved_floor:'+obj['id'])
        if a['kind'] in ('wall','ceiling'):errors.append('unverified_architectural_support:'+obj['id'])
        bounds=world['hypothesis'].get('bounds_xz_m')
        if bounds:
            x,_,z=b['center'];sx,_,sz=b['size']
            if x-sx/2<bounds[0] or x+sx/2>bounds[1] or z-sz/2<bounds[2] or z+sz/2>bounds[3]:
                errors.append('outside_hypothesized_extent:'+obj['id'])
    for r in world['relations']:
        a,b=boxes[r['source']],boxes[r['target']];k=r['kind'];ok=True
        if k=='above':ok=a['center'][1]-a['size'][1]/2>=b['center'][1]+b['size'][1]/2-1e-5
        elif k=='below':ok=a['center'][1]+a['size'][1]/2<=b['center'][1]-b['size'][1]/2+1e-5
        elif k in ('left_of','right_of','in_front_of','behind'):
            axis=0 if k in ('left_of','right_of') else 2
            sign=1 if k in ('right_of','in_front_of') else -1
            ok=sign*(a['center'][axis]-b['center'][axis])>=-1e-5
        if not ok:errors.append('unsatisfied_declared_relation:'+json.dumps(r))
    review.update(geometry_contract_passed=not errors,joint_contract_errors=errors,
        continuous_origins_preserved=not any(v.get('origin_initialization',{}).get('changed',False) for v in layout.values()),
        origin_initialization={g:v.get('origin_initialization') for g,v in layout.items()},room_extent_is_hypothesis=True,
        geometry_scope='AABB_visibility_contact_collision_floor_and_declared_order_only_not_RGB_or_physical_certification')
    if policy=='approximate':
        from .joint_world_acceptance import evaluate
        review.update(evaluate(solved,observed,world,plane,errors))
    return solved
