"""Vertical order projection for free mounting roots, then existing XYZ audits."""
from copy import deepcopy
import json

import numpy as np
from scipy.optimize import linprog,minimize,LinearConstraint

from . import assembly_sdscene_geometry as base,assembly_parametric as parametric

PARAMETERIZATION='functional_assemblies_with_vertical_order_projection_v5'
layout_answer=base.layout_answer


def layout_prompt(case,observed,world,plane):
    return (base.layout_prompt(case,observed,world,plane)+
        '\nV5 GEOMETRY CONTRACT: above/below order complete AABB vertical intervals along original camera Y, '
        'not a relation along a wall normal. Respect these together with floor/top support. '
        'Root wall/ceiling/free offset_m[1] is a signed camera Y hypothesis, not a measured mounting plane. '
        'The solver may project free mounting-root Y and transport its descendants to satisfy declared vertical order; '
        'it will not move observed objects, floor-root heights, or change inventory/dimensions. '
        'Use the following frozen metric anchors to infer relative proximity. They are predicted camera AABBs, '
        'not ground-truth room geometry. Their absolute centres are READ ONLY; never copy an absolute centre into offset_m. '
        'Root offset remains relative to the group centre; hosted offset remains relative to its own host. '
        '\nREAD_ONLY_METRIC_ANCHORS: '+json.dumps([dict(id=o['track_id'],name=o.get('description'),
            center_camera_m=o['center_local_m']['value'],full_AABB_size_m=o['size_aabb_local_m']['value'])
            for o in observed['objects']],ensure_ascii=False))


def vertical_projection(case,observed,world,layout,plane):
    layout=deepcopy(layout);boxes,_,_,_=parametric.instantiate(world,layout,observed,plane,case['hfov'])
    contracts={o['id']:o for o in world['objects']};rows={o['id']:o for g in layout.values() for o in g['objects']}
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    roots={}
    for oid in contracts:
        root=oid
        while contracts[root]['attachment']['host'] in contracts:root=contracts[root]['attachment']['host']
        roots[oid]=root
    movable=sorted({r for r in roots.values() if contracts[r]['attachment']['kind'] in ('wall','ceiling','free')})
    index={r:i for i,r in enumerate(movable)};a=[];b=[];queries=[]
    def y(oid):return (boxes|old)[oid]['center'][1]
    for relation in world['relations']:
        if relation['kind'] not in ('above','below'):continue
        source,target=relation['source'],relation['target'];sign=1. if relation['kind']=='above' else -1.
        vector=np.zeros(len(movable))
        for oid,coefficient in ((source,sign),(target,-sign)):
            root=roots.get(oid)
            if root in index:vector[index[root]]+=coefficient
        clearance=sum((boxes|old)[oid]['size'][1] for oid in (source,target))/2
        rhs=clearance-sign*(y(source)-y(target));a.append(vector);b.append(rhs)
        queries.append(dict(source=source,target=target,kind=relation['kind'],before_satisfied=rhs<=0))
    report=dict(method='minimum_root_Y_projection',queries=queries,adjustments_Y_m={},feasible=True,
        fixed_observations_moved=False,floor_roots_moved=False,dimensions_changed=False,
        mounting_surfaces_recovered=False,scope='AABB_vertical_interval_order_not_mounting_or_stability_certification')
    if not b or max(b)<=0:return layout,report
    if not movable:report['feasible']=False;return layout,report
    matrix=np.array(a);rhs=np.array(b)
    fit=linprog(np.zeros(len(movable)),A_ub=-matrix,b_ub=-rhs,bounds=[(None,None)]*len(movable),method='highs')
    if not fit.success:report['feasible']=False;return layout,report
    result=minimize(lambda x:float(x@x),fit.x,jac=lambda x:2*x,
        constraints=LinearConstraint(matrix,rhs,np.inf),method='SLSQP',options=dict(maxiter=100,ftol=1e-10))
    adjustment=result.x if np.isfinite(result.x).all() and np.min(matrix@result.x-rhs,initial=0.)>=-1e-7 else fit.x
    for root,delta in zip(movable,adjustment):
        rows[root]['offset_m'][1]+=float(delta);report['adjustments_Y_m'][root]=float(delta)
    return layout,report


def compile_layout(case,observed,world,layout,plane,*,repairer=None,refinement_score=None,include_default_preview=False):
    revised,projection=vertical_projection(case,observed,world,layout,plane)
    result=parametric.compile_layout(case,observed,world,revised,plane,max_direction_previews=81,
        repairer=repairer,refinement_score=refinement_score,include_default_preview=include_default_preview);review=result['review']
    items={o['id']:o for o in result['items']};items.update({o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']})
    for relation in review['declared_relation_review']:
        if relation['kind'] in ('above','below'):
            s,t=items[relation['source']],items[relation['target']];sign=1 if relation['kind']=='above' else -1
            gap=sign*(s['center'][1]-t['center'][1])-(s['size'][1]+t['size'][1])/2
            relation.update(vertical_interval_gap_m=gap,ordering_or_presence_satisfied=gap>=-1e-6,
                ordering_scope='full_AABB_vertical_intervals_v5')
    unsatisfied=[r for r in review['declared_relation_review'] if r['ordering_or_presence_satisfied'] is False]
    review.update(vertical_order_projection=projection,geometry_parameterization=PARAMETERIZATION,
        unsatisfied_declared_ordering=unsatisfied,observed_absolute_positions_sent_to_metric_LM=True)
    if unsatisfied:
        review['geometry_contract_passed']=False;review['hard_violations']+=len(unsatisfied)
        review['layout_proxy_passed']=False
    return result
