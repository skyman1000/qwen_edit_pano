"""Preserve sampled functional-spatial program regions during strict geometry."""
from functools import partial
import json
import math
import numpy as np

from . import assembly_mechanism_geometry as numeric,assembly_realization_geometry as v7
from . import assembly_activation_geometry as base,assembly_spatial as spatial

PARAMETERIZATION='protected_camera_region_realizations_v9'
layout_answer=numeric.layout_answer


def layout_prompt(case,observed,world,plane):
    return numeric.layout_prompt(case,observed,world,plane)+(
        '\nV9 FIXED SPATIAL PROGRAM: '+json.dumps(world.get('spatial_program',{}),ensure_ascii=False)+
        '\nFUNCTIONAL LINKS (not near/facing/support constraints): '+json.dumps(world.get('functional_links',[]),ensure_ascii=False)+
        '\nA sampled rear program means the COMPLETE added boxes must lie behind the ORIGINAL CAMERA, Z<0. '
        'It is not behind a sofa. The program region is part of the sampled world hypothesis and will NOT be changed '
        'by proximity optimization. Use compatible relative numeric offsets. unspecified still allows all hidden domains. '
        'Do not output a region field or reinterpret service as adjacency. No direction has a required object count.')


def region_satisfied(item,region,hfov,tolerance=2e-5):
    x,_,z=item['center'];sx,_,sz=item['size'];tx=math.tan(math.radians(hfov)/2)
    if region=='unspecified':return True
    if region=='rear':return z+sz/2<=tolerance
    sign=1 if region=='right' else -1
    return sign*x-sx/2>=-tolerance and sign*x-tx*z-(sx+tx*sz)/2>=-tolerance


def compile_layout(case,observed,world,layout,plane,*,support_centres=False):
    numeric_world,revised,mapping=v7.support_tree_groups(case,observed,world,layout)
    groups={g['key']:g for g in world['groups']};fixed={}
    for o in numeric_world['objects']:
        original=mapping[o['id']]['semantic_group'];r=groups[original].get('layout_region','unspecified')
        if r!='unspecified':
            revised[o['group']]['region_domain']=[r];fixed[o['id']]=r
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    def score(solved):
        review=solved['review'];solver=review['program_solver'];boxes={o['id']:o for o in solved['items']}|old
        wrong=sum(not region_satisfied(boxes[oid],r,case['hfov']) for oid,r in fixed.items());near=0.
        for relation in world['relations']:
            if relation['kind']!='near':continue
            a,b=(boxes[relation[k]] for k in ('source','target'))
            gap=np.maximum(0,np.abs(np.asarray(a['center'])-np.asarray(b['center']))-(np.asarray(a['size'])+np.asarray(b['size']))/2)
            near+=float(gap@gap)
        return (review['hard_violations']+wrong+sum(v['ordering_or_presence_satisfied'] is False for v in review['declared_relation_review']),
            solver.get('contact_and_frustum_linear_violations',0),review['attachment_contact_failures'],len(solver['final_overlaps']),
            sum(v['volume_m3'] for v in solver['final_overlaps']),near,
            sum(float(np.asarray(d)@np.asarray(d)) for d in solver.get('local_adjustments_XZ_m',{}).values()))
    solved=base.compile_layout(case,observed,numeric_world,revised,plane,
        repairer=partial(spatial.articulated_repair,hard_ordering=True,surface_proximity=True,fixed_regions=fixed,support_centres=support_centres),
        refinement_score=score,include_default_preview=True)
    r=solved['review'];boxes={o['id']:o for o in solved['items']}
    violations=[dict(id=oid,required_region=region) for oid,region in fixed.items() if not region_satisfied(boxes[oid],region,case['hfov'])]
    r.update(geometry_parameterization=PARAMETERIZATION,spatial_program=world.get('spatial_program',{}),
        fixed_region_by_object=fixed,spatial_program_violations=violations,
        functional_links_are_metric_near=False,semantic_to_support_tree_groups=mapping,
        region_scope='Sampled camera-region state is protected; unspecified retains bounded hidden-direction search',geometry_search_complete=False)
    if violations or r['program_solver'].get('contact_and_frustum_linear_violations',0):
        r['geometry_contract_passed']=False;r['layout_proxy_passed']=False;r['hard_violations']+=len(violations)
    return solved
