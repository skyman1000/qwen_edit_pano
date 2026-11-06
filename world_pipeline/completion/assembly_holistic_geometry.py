"""Protected sampled camera regions and explicitly limited support proxies."""
from . import assembly_panorama_geometry as base

PARAMETERIZATION='sampled_joint_regions_rigid_box_center_support_v11'
layout_answer=base.layout_answer


def layout_prompt(case,observed,world,plane):
    return base.layout_prompt(case,observed,world,plane)+(
        '\nV11 METRIC CONTRACT: widths/heights/depths are intrinsic FULL dimensions of each concrete variant, '
        'not guessed camera positions or copies of an observed same-category object. '
        'offset_m is RELATIVE displacement; do not paste absolute observed XYZ as offsets. '
        'For rigid_box top/seat/shelf support, put the CHILD BOX CENTRE projection inside the HOST footprint. '
        'This is a uniform-box/quasistatic proxy, not measured mesh stability. contact_only objects retain contact '
        'with unverified mass/support. Account for coexisting contents on each host. '
        'Respect the sampled whole-world sector assignments: geometry does not choose another sector to improve near. '
        'For functional service, choose usable configurations rather than just non-colliding boxes. '
        'No hidden walls, gravity direction or room boundary were measured.')


def compile_layout(case,observed,world,layout,plane):
    solved=base.compile_layout(case,observed,world,layout,plane,support_centres=True)
    boxes={o['id']:o for o in solved['items']}
    boxes.update({o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']})
    rows=[]
    for o in world['objects']:
        a=o['attachment'];host=a['host']
        if not host or a['kind'] not in ('top','seat','shelf'):continue
        child=boxes[o['id']];parent=boxes[host];rigid=o.get('support_model','rigid_box')=='rigid_box'
        inside=all(abs(child['center'][d]-parent['center'][d])<=parent['size'][d]/2+2e-5 for d in (0,2))
        overlap=[max(0.,min(child['center'][d]+child['size'][d]/2,parent['center'][d]+parent['size'][d]/2)-max(child['center'][d]-child['size'][d]/2,parent['center'][d]-parent['size'][d]/2)) for d in (0,2)]
        rows.append(dict(child=o['id'],host=host,support_model=o.get('support_model','rigid_box'),
            box_center_inside_host_XZ=inside,child_footprint_overlap_fraction=overlap[0]*overlap[1]/(child['size'][0]*child['size'][2]),
            required_proxy_failed=rigid and not inside))
    failures=sum(r['required_proxy_failed'] for r in rows)
    solved['review'].update(geometry_parameterization=PARAMETERIZATION,support_projection_review=rows,
        support_proxy_assumption='Uniform rigid box, one horizontal host plane; not mesh/COM/gravity measurement',
        physical_geometry_verified=False)
    if failures:
        solved['review']['geometry_contract_passed']=False;solved['review']['hard_violations']+=failures
    return solved


def annotate_state(state,layout):
    dimensions={o['id']:o for g in layout.values() for o in g['objects']}
    for obj in state['objects']:
        if obj['track_id'] not in dimensions:continue
        row=dimensions[obj['track_id']]
        obj['orientation_yaw_camera_deg']=dict(value=row['yaw_deg'],valid=True,source='layout_hypothesis',
            note='Absolute camera-Y rotation; canonical object forward direction unverified')
        obj['intrinsic_size_m']=dict(value=row['intrinsic_size_m'],valid=True,source='layout_hypothesis')
    return state
