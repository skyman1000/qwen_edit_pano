"""Approximate world hypotheses: contradictions vs. unverified assumptions.

The user requests plausible approximate layouts, not precision reconstruction.
Keep visibility, identity and real object support constraints. Qualitative
directions and unmeasured architectural surfaces remain reviewable assumptions.
"""
import math

import numpy as np

POLICY=dict(name='approximate_plausibility_v1',max_shallow_penetration_m=.03,
    max_smaller_volume_overlap_fraction=.05,floor_contact_tolerance_m=.03,
    thresholds_scope='engineering_tolerances_for_uncertain_AABBs_not_measured_accuracy')


def evaluate(solved,observed,world,plane,strict_errors):
    review=solved['review'];errors=[];warnings=[];assumptions=[]
    boxes={o['id']:o for o in solved['items']}
    boxes.update({o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']})
    if review['hard_violations']:errors.append('original_camera_visibility_or_camera_intersection')
    for oid,b in boxes.items():
        if not all(type(v) in (int,float) and math.isfinite(v) for v in b['center']+b['size']) or min(b['size'])<=0:
            errors.append('invalid_numeric_box:'+oid)
    for pair in review['program_solver']['final_overlaps']:
        a,b=(boxes[pair[k]] for k in ('a','b'))
        depth=np.minimum(np.array(a['center'])+np.array(a['size'])/2,np.array(b['center'])+np.array(b['size'])/2)-np.maximum(
            np.array(a['center'])-np.array(a['size'])/2,np.array(b['center'])-np.array(b['size'])/2)
        volume=float(np.prod(np.maximum(depth,0.)));fraction=volume/min(float(np.prod(a['size'])),float(np.prod(b['size'])))
        small=float(np.min(depth))<=POLICY['max_shallow_penetration_m'] and fraction<=POLICY['max_smaller_volume_overlap_fraction']
        if small:warnings.append(dict(code='minor_AABB_overlap',objects=[pair['a'],pair['b']],overlap_fraction=fraction,penetration_m=float(np.min(depth))))
        else:errors.append('substantial_AABB_collision:'+pair['a']+':'+pair['b'])
    for obj in world['objects']:
        oid=obj['id'];a=obj['attachment'];b=boxes[oid]
        if a['host'] and a['kind'] in ('top','seat','shelf'):
            parent=boxes[a['host']]
            footprint=[min(b['center'][i]+b['size'][i]/2,parent['center'][i]+parent['size'][i]/2)-max(
                b['center'][i]-b['size'][i]/2,parent['center'][i]-parent['size'][i]/2) for i in (0,2)]
            if min(footprint)<=0:errors.append('host_contact_failure:'+oid)
            elif any(abs(b['center'][i]-parent['center'][i])>parent['size'][i]/2 for i in (0,2)):
                warnings.append(dict(code='support_overhang',object=oid,host=a['host']))
        if a['kind']=='floor':
            if plane['floor_y_m'] is None:errors.append('unresolved_floor:'+oid)
            else:
                residual=abs(b['center'][1]-b['size'][1]/2-plane['floor_y_m'])
                if residual>POLICY['floor_contact_tolerance_m']:errors.append('floor_contact_failure:'+oid)
                elif residual>1e-5:warnings.append(dict(code='approximate_floor_contact',object=oid,residual_m=residual))
        if a['kind'] in ('wall','ceiling'):
            record=dict(object=oid,support_kind=a['kind'],verified=False,
                assumption='An unobserved mounting surface exists at the hypothesized fixture; its plane and contact have not been reconstructed.')
            assumptions.append(record);warnings.append(dict(code='unmeasured_architectural_support',**record))
    for error in strict_errors:
        if error.startswith(('unsatisfied_declared_relation:','outside_hypothesized_extent:')):
            warnings.append(dict(code='qualitative_layout_preference',detail=error))
    if not review['program_solver'].get('continuous_solver_converged',True):
        warnings.append(dict(code='solver_not_converged',detail='Final numeric constraints are evaluated independently of the solver termination flag.'))
    return dict(geometry_contract_passed=not errors,joint_contract_errors=errors,layout_policy=dict(POLICY),
        strict_geometry_contract_passed=not strict_errors,strict_contract_errors=list(strict_errors),
        soft_layout_warnings=warnings,environment_support_assumptions=assumptions,
        physical_geometry_verified=False,semantic_plausibility_verified=False,
        geometry_scope='approximate_hypothesis_visibility_support_and_major_collision_checks_not_precision_or_mesh_certification')
