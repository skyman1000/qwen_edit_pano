"""Finite hidden regions and parent-relative metric attributes.

The LM never needs to invent an off-frustum boundary angle, nor restate the
observed absolute positions. The compiler owns camera projection and frames.
"""
import json
import math
import itertools
from copy import deepcopy

import numpy as np

from ..common import finite_vector
from . import assembly_layout as groups
from . import assembly_spatial as spatial

PARAMETERIZATION='hidden_region_domains_parent_relative_v2'
REGIONS={'left','right','rear'}


def bearings(hfov):
    # Sideways and backward initial directions; the feasible half-spaces also
    # cover unphotographed front-side space. No room boundary is inferred.
    return dict(left=-90.,right=90.,rear=180.)


def layout_prompt(case,observed,world,plane):
    templates=[];predicted={o['id'] for o in world['objects']}
    for gid,objects in groups.grouped(world).items():
        rows=[]
        for o in objects:
            kind=o['attachment']['kind']
            rows.append(dict(id=o['id'],full_size_m=[None,None,None],yaw_deg=0,offset_m=[0,0,0],
                surface_fraction=.5 if kind in ('seat','shelf') else None,
                hanging_drop_m=0 if kind=='hang' else None))
        templates.append(dict(id=gid,region='unspecified',range_m=3.,objects=rows))
    observed_hosts={o['attachment']['host'] for o in world['objects'] if o['attachment']['host'] not in predicted}
    anchors=[dict(id=o['track_id'],name=o.get('description'),
        **(dict(full_camera_AABB_size_m=o['size_aabb_local_m']['value']) if o['track_id'] in observed_hosts else {})) for o in observed['objects']]
    return ('Supply relative metric attributes for this FIXED HIDDEN-WORLD DELTA. Return exactly the group/object IDs in TEMPLATE. '
        'The attached observation board shows the original photographed view, not a layout canvas for the added objects. '
        'For each group choose region from left,right,rear relative to the original camera; '
        'use unspecified when the evidence/placement description does not determine an off-view direction. '
        'The compiler will consider multiple hidden directions for unspecified; it never treats a central visible area as hidden. '
        'The program constructs a valid hidden-direction bearing and accounts for complete object extents; do not output azimuth_deg. '
        'range_m is a positive horizontal distance hypothesis, not a measured room boundary. '
        'full_size_m=[width,height,depth] gives positive FULL intrinsic dimensions in metres. '
        'yaw_deg is absolute camera-Y yaw: x_camera=cos(yaw)*x_local+sin(yaw)*z_local, '
        'z_camera=-sin(yaw)*x_local+cos(yaw)*z_local. '
        'offset_m is always RELATIVE, never an absolute camera position. '
        'For an unhosted root: small displacement from the group centre in camera axes. '
        'For a predicted-host item: displacement from its HOST centre in HOST intrinsic axes, rotated by HOST yaw by the program. '
        'For an observed-host item: relative displacement in camera axes, because observed yaw is unknown. '
        'Keep hosted belongings on the actual host surface and functionally associated furniture together. '
        'The compiler computes floor/top Y. seat/shelf use surface_fraction [0,1] from host bottom to top; '
        'hang uses nonnegative hanging_drop_m from host centre to child TOP. The displayed .5/0 are unverified initialization priors. '
        'Wall/ceiling/free root offset Y is a signed camera-Y hypothesis, not a recovered mounting plane. '
        'Unused height parameters may be null. No observed object is a new instance; never copy a visible object location into offset_m. '
        'Do not add/remove/rename instances.\nTEMPLATE: '+json.dumps(dict(groups=templates))+
        '\nFIXED SEMANTIC WORLD: '+json.dumps(dict(configuration=world['configuration'],hypothesis=world['hypothesis'],
            objects=[{k:o[k] for k in ('id','name','category','variant','attachment','placement_hint','group')} for o in world['objects']],
            relations=world['relations'],realized_requirements=world.get('realized_requirements',[])),ensure_ascii=False)+
        '\nFROZEN OBSERVATIONS: '+json.dumps(anchors,ensure_ascii=False)+'\nLEVEL PRIOR: '+json.dumps(plane))


def layout_answer(value,world,hfov):
    expected=groups.grouped(world);rows=value.get('groups');contracts={o['id']:o for o in world['objects']}
    if not isinstance(rows,list) or len(rows)!=len(expected) or any(not isinstance(g,dict) or not isinstance(g.get('id'),str) for g in rows):
        raise ValueError('Return every group exactly once: '+str(list(expected)))
    if {g['id'] for g in rows}!=set(expected):raise ValueError('Copy group IDs exactly: '+str(list(expected)))
    converted=[];angles=bearings(hfov)
    for g in rows:
        region=g.get('region')
        # "center" was the actual failing response in two real Qwen worlds.
        # It supplies no outside-view bearing. Preserve that uncertainty as a
        # domain of hypotheses, never silently reinterpret center as rear.
        if region in (None,'unspecified','center','centre'):
            domain=['left','right','rear']
        elif isinstance(region,str) and region in REGIONS:domain=[region]
        else:raise ValueError(g['id']+'.region must be left/right/rear/unspecified; center supplies no hidden bearing')
        records=g.get('objects')
        if not isinstance(records,list) or any(not isinstance(o,dict) for o in records):raise ValueError(g['id']+'.objects must be records')
        clean=[]
        for r in records:
            oid=r.get('id');dims=r.get('full_size_m')
            if not isinstance(oid,str) or oid not in contracts:raise ValueError('Unknown object ID '+repr(oid)+'; copy TEMPLATE identities')
            if not finite_vector(dims,3) or min(dims)<=0:raise ValueError(oid+'.full_size_m must be three finite positive FULL dimensions')
            kind=contracts[oid]['attachment']['kind'];host=contracts[oid]['attachment']['host']
            clean.append(dict(id=oid,width_m=dims[0],height_m=dims[1],depth_m=dims[2],yaw_deg=r.get('yaw_deg',0),
                offset_m=r.get('offset_m'),offset_frame='host_local' if host in contracts else 'camera',
                surface_fraction=r.get('surface_fraction',.5) if kind in ('seat','shelf') else None,
                hanging_drop_m=r.get('hanging_drop_m',0) if kind=='hang' else None))
        converted.append(dict(id=g['id'],azimuth_deg=angles[domain[0]],range_m=g.get('range_m'),objects=clean,
            region_domain=domain,region_response=region))
    result=spatial.layout_answer(dict(groups=converted),world,hfov)
    for g in converted:
        result[g['id']].update(region_domain=g['region_domain'],region_response=g['region_response'])
    return result


def instantiate(world,layout,observed,plane,hfov):
    boxes,attachments,certificates,fixed=spatial.instantiate(world,layout,observed,plane,hfov)
    for gid,cert in certificates.items():
        if cert['kind']=='fixed_observed_host':continue
        angle=layout[gid]['azimuth_deg']
        cert['kind']='behind_camera' if abs(angle-180)<1e-8 else 'left' if angle<0 else 'right'
        cert['region_from_finite_parameterization']=True
    return boxes,attachments,certificates,fixed


def compile_layout(case,observed,world,layout,plane,*,max_direction_previews=None,repairer=None,refinement_score=None,include_default_preview=False):
    if world.get('unresolved_installations'):raise ValueError('Unresolved host installation; no completed geometry exported')
    # Preview every direction combination (at most 3**4) cheaply, then refine
    # a bounded beam. This is geometric search, not posterior/causal ranking.
    gids=list(layout);angles=bearings(case['hfov']);previews=[];contracts={o['id']:o for o in world['objects']}
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    domains=[];fixed_direction_groups=[]
    for gid in gids:
        domain=layout[gid]['region_domain']
        # A group connected to fixed observed hosts has no movable group
        # origin. Its per-support escape faces are searched by the solver;
        # repeated nominal bearings must not consume the refinement beam.
        roots=[]
        for row in layout[gid]['objects']:
            oid=row['id']
            while contracts[oid]['attachment']['host'] in contracts:oid=contracts[oid]['attachment']['host']
            roots.append(contracts[oid]['attachment']['host'])
        if roots and all(host in old for host in roots):domain=domain[:1];fixed_direction_groups.append(gid)
        domains.append(domain)
    combinations=math.prod(len(d) for d in domains)
    if max_direction_previews is not None and combinations>max_direction_previews:
        if max_direction_previews<1:raise ValueError('max_direction_previews must be positive')
        # Balanced deterministic candidates avoid enumerating 3**many-groups.
        # Include the default and single-coordinate changes, then shared seeded
        # proposals. This is bounded numeric search, not probability inference.
        selected_choices=[tuple(d[0] for d in domains)];seen={selected_choices[0]}
        for i,domain in enumerate(domains):
            for value in domain[1:]:
                trial=list(selected_choices[0]);trial[i]=value;trial=tuple(trial)
                if len(selected_choices)<max_direction_previews and trial not in seen:selected_choices.append(trial);seen.add(trial)
        rng=np.random.default_rng(3407)
        while len(selected_choices)<max_direction_previews:
            trial=tuple(d[int(rng.integers(len(d)))] for d in domains)
            if trial not in seen:selected_choices.append(trial);seen.add(trial)
    else:selected_choices=itertools.product(*domains)
    for choices in selected_choices:
        candidate=deepcopy(layout)
        for gid,region in zip(gids,choices):candidate[gid]['azimuth_deg']=angles[region]
        boxes,_,_,_=instantiate(world,candidate,observed,plane,case['hfov'])
        overlaps,_=groups.overlap_pairs(old|boxes,set(boxes),contracts)
        ids=list(boxes);x=np.array([[boxes[oid]['center'][d] for d in (0,2)] for oid in ids]).ravel()
        cost=spatial.relation_objective(ids,old,world['relations'],weight=.25)(x)[0]
        previews.append(dict(layout=candidate,regions=dict(zip(gids,choices)),rank=(len(overlaps),sum(o['volume_m3'] for o in overlaps),cost)))
    results=[]
    finalists=sorted(previews,key=lambda p:p['rank'])[:4]
    if include_default_preview and previews and not any(p is previews[0] for p in finalists):finalists=[previews[0]]+finalists[:3]
    for preview in finalists:
        solved=groups.compile_layout(case,observed,world,preview['layout'],plane,instantiator=instantiate,repairer=repairer or spatial.articulated_repair)
        r=solved['review'];solver=r['program_solver']
        rank=(r['hard_violations'],r['attachment_contact_failures'],len(solver['final_overlaps']),
            sum(o['volume_m3'] for o in solver['final_overlaps']),sum(v['ordering_or_presence_satisfied'] is False for v in r['declared_relation_review']))
        if refinement_score is not None:rank=refinement_score(solved)
        results.append(dict(solved=solved,regions=preview['regions'],rank=rank))
    selected=min(results,key=lambda p:p['rank']);result=selected['solved']
    result['review'].update(geometry_parameterization=PARAMETERIZATION,host_offsets_rotated_by_host_yaw=True,
        support_trees_transport_descendants=True,observed_absolute_positions_sent_to_metric_LM=False,
        regions_are_directions_not_measured_room_boundaries=True,
        direction_search=dict(response={g:layout[g].get('region_response') for g in gids},
            domains={g:layout[g].get('region_domain') for g in gids},combinations_previewed=len(previews),
            total_combinations_in_domain=combinations,preview_budget=max_direction_previews,
            refined=[dict(regions=p['regions'],numeric_rank=list(p['rank'])) for p in results],
            selected=selected['regions'],ranking_scope='bounded_numeric_search_not_image_posterior',
            fixed_observed_groups_direction_not_used=fixed_direction_groups,
            inventory_changed=False,complete_search=False))
    return result
