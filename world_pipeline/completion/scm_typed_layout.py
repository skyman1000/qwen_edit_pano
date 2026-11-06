"""Program-owned identities, finite support choices, and level-preserving layout.

This backend places a fixed sampled inventory. Ordering/support are layout
proposals, not existence mechanisms or identified causal effects. Unknown room
bounds and gravity remain unknown. All numerical adjustments are recorded.
"""
from copy import deepcopy
import itertools
import json
import math

import numpy as np

from ..common import finite_vector,write
from .conditional_world_geometry import compile_layout
from .frozen_causal_world import ask_validated
from .scm_typed_queries import finite_choice
from .scm_world_layout import support_answer
from .scene_completion import map_plan


def canonical_angle(angle):
    if type(angle) not in (int,float) or not math.isfinite(angle):raise ValueError('Need finite azimuth')
    # Periodic coordinate conversion, not clipping or moving an object.
    return (angle+180.)%360.-180.


def numeric_body(value):
    fields={'azimuth_deg','range_m','size','center_y_m','reason'}
    required={'azimuth_deg','range_m','size'}
    if not isinstance(value,dict) or not required<=set(value) or set(value)-fields:
        raise ValueError('Geometry needs azimuth_deg, range_m and size; optional center_y_m/reason; no IDs/arrays')
    angle=canonical_angle(value['azimuth_deg']);r=value['range_m'];y=value.get('center_y_m')
    if type(r) not in (int,float) or not math.isfinite(r) or r<0:raise ValueError('Need nonnegative finite horizontal range')
    if y is not None and (type(y) not in (int,float) or not math.isfinite(y)):raise ValueError('Need finite signed camera-Y when supplied')
    if not finite_vector(value['size'],3) or min(value['size'])<=0:raise ValueError('Need positive FULL camera-axis sizes')
    reason=value.get('reason')
    if not isinstance(reason,str) or not reason.strip():
        reason='Model metric initial proposal; support and off-camera position computed by program. No model rationale supplied.'
    return dict(azimuth_deg=value['azimuth_deg'],range_m=r,size=deepcopy(value['size']),center_y_m=y,
        reason=reason,canonical_azimuth_deg=angle)


def project_2d(point,A,b):
    """Exact Euclidean projection onto a 2D polyhedron, or None if infeasible."""
    p=np.asarray(point,dtype=float);A=np.asarray(A,dtype=float).reshape(-1,2);b=np.asarray(b,dtype=float)
    candidates=[]
    for count in range(min(2,len(b))+1):
        for active in itertools.combinations(range(len(b)),count):
            if not active:x=p.copy()
            else:
                a=A[list(active)];rhs=b[list(active)]-a@p
                try:lam=np.linalg.solve(a@a.T,rhs)
                except np.linalg.LinAlgError:continue
                if (lam < -1e-9).any():continue
                x=p+a.T@lam
            if np.all(A@x>=b-1e-9):candidates.append(x)
    return min(candidates,key=lambda x:float(np.sum((x-p)**2))) if candidates else None


def solve_center(body,support,plane,boxes,hfov,image_size,epsilon=1e-4,regions=None):
    """Keep size and support Y fixed; solve five sufficient convex regions.

    Positive footprint overlap is the existing AABB support proxy. Neither
    overlap nor a level plane is a verified physical contact/gravity estimate.
    """
    angle=math.radians(body['canonical_azimuth_deg']);r=body['range_m'];s=body['size']
    proposed=[r*math.sin(angle),body['center_y_m'],r*math.cos(angle)]
    kind=support['kind'];target=None;A0=[];b0=[]
    if kind=='floor':
        if plane['floor_y_m'] is None:raise ValueError('No declared floor prior exists')
        y=plane['floor_y_m']+s[1]/2
    elif kind=='object':
        target=boxes[support['target']];y=target['center'][1]+(target['size'][1]+s[1])/2
        # Necessary positive XZ overlap; do not impose full footprint containment.
        for axis,dim in ((0,0),(1,2)):
            extent=(target['size'][dim]+s[dim])/2-epsilon
            if extent<=0:raise ValueError('Support footprint cannot have positive overlap')
            for sign in (-1,1):
                a=[0.,0.];a[axis]=sign;A0.append(a);b0.append(sign*target['center'][dim]-extent)
    else:
        y=body['center_y_m']
        if y is None:raise ValueError('Unmodeled support needs an explicit camera-Y proposal')
    if proposed[1] is None:proposed[1]=y
    tx=math.tan(math.radians(hfov)/2);ty=tx*image_size[1]/image_size[0]
    proposals=[]
    allowed=('left','right','behind_camera','above','below')
    if regions is not None and (not regions or any(r not in allowed for r in regions)):raise ValueError('Unknown certificate region')
    for region in (allowed if regions is None else regions):
        A=list(A0);b=list(b0)
        if region=='behind_camera':A.append([0.,-1.]);b.append(s[2]/2+epsilon)
        elif region in ('left','right'):
            sign=-1 if region=='left' else 1
            A.extend([[sign,0.],[sign,-tx]])
            b.extend([s[0]/2+epsilon,(s[0]+tx*s[2])/2+epsilon])
        else:
            sign=-1 if region=='below' else 1
            if sign*y<s[1]/2+epsilon:continue
            A.append([0.,-ty]);b.append((s[1]+ty*s[2])/2+epsilon-sign*y)
        xz=project_2d([proposed[0],proposed[2]],A,b)
        if xz is not None:
            center=[float(xz[0]),float(y),float(xz[1])]
            displacement=float(np.linalg.norm(np.array(center)-proposed))
            proposals.append((displacement,region,center))
    if not proposals:raise ValueError('No off-frustum centre exists with the fixed support and size; candidate retained')
    displacement,region,center=min(proposals,key=lambda v:v[0])
    return dict(center=center,proposal_center=proposed,off_fov_certificate_region=region,
        translation_m=(np.array(center)-proposed).tolist(),translation_norm_m=displacement,
        support_height_programmed=kind in ('floor','object'),sizes_changed=False,
        method='nearest_in_five_sufficient_regions_at_fixed_support_Y_no_room_bounds',
        semantic_validity_verified=False)


def support_choices(boxes,plane):
    choices=[]
    if plane['floor_y_m'] is not None:choices.append(dict(description='standing directly on the floor',kind='floor',target=None))
    for oid,box in boxes.items():
        choices.append(dict(description='supported on top of '+box['name'],kind='object',target=oid))
    choices.extend([dict(description=d,kind=k,target=None) for k,d in
        (('wall','wall-mounted; wall geometry unmeasured'),('ceiling','suspended from ceiling; geometry unmeasured'),
         ('unspecified','support unresolved; explicit camera-Y needed'))])
    return choices


def initial_boxes(case,observed):
    context={o['id']:o for o in case['context']}
    return {o['track_id']:dict(id=o['track_id'],name=context[o['track_id']].get('description') or
        context[o['track_id']].get('category','observed object'),center=o['center_local_m']['value'],
        size=o['size_aabb_local_m']['value']) for o in observed['objects']}


def creates_cycle(child,parent,selected):
    seen=set()
    while parent is not None:
        if parent==child:return True
        if parent in seen:raise ValueError('Existing support graph already cyclic')
        seen.add(parent);edge=selected.get(parent)
        parent=edge['target'] if edge and edge['kind']=='object' else None
    return False


def feasible_supports(oid,bodies,plan,selected,observed_boxes,plane,case):
    """Mask observed supports that cannot satisfy the actual outside-view task.

    Future sampled hosts remain available; they will be placed first by the DAG.
    No object is invented to provide a missing support.
    """
    choices=[];rejected=[]
    for choice in support_choices(observed_boxes,plane):
        support={k:choice[k] for k in ('kind','target')}
        if support['kind'] in ('wall','ceiling'):support['kind']='unspecified'
        try:solve_center(bodies[oid],support,plane,observed_boxes,case['hfov'],case['image_size'])
        except ValueError as exc:
            rejected.append(dict(choice=choice,reason=str(exc)));continue
        choices.append(choice)
    for obj in plan['additions']:
        target=obj['id']
        if target==oid or creates_cycle(oid,target,selected):continue
        choices.append(dict(description='supported on top of NEW '+obj['name'],kind='object',target=target))
    if not choices:raise ValueError('No admissible support; missing height or incompatible fixed observations. No objects were discarded.')
    return choices,rejected


def compile_dependency_layout(case,observed,plan,plane,bodies,selections):
    """Jointly place support components using a common sufficient view region.

    Trying five certificates for an entire component avoids freezing a host at
    a local optimum that makes its unseen children impossible. This remains a
    conservative AABB layout approximation, not a complete room/physics solver.
    """
    new_ids={o['id'] for o in plan['additions']};byid={o['id']:o for o in plan['additions']}
    if set(bodies)!=new_ids or set(selections)!=new_ids:raise ValueError('Missing fixed sampled instance data')
    rows=[dict(id=o['id'],placement='Outside original input frustum; component layout pending.',
        support={k:selections[o['id']][k] for k in ('kind','target')}) for o in plan['additions']]
    out=support_answer(dict(objects=rows),plan,case)
    edges={o['id']:o['support'] for o in out['additions']};children={i:[] for i in new_ids}
    roots=[]
    for oid in out['placement_dependency_order']:
        target=edges[oid]['target']
        if target in new_ids:children[target].append(oid)
        else:roots.append(oid)
    boxes=initial_boxes(case,observed);certificates={};components=[]
    for root in roots:
        candidates=[]
        for region in ('left','right','behind_camera','above','below'):
            local_boxes=deepcopy(boxes);local_cert={}
            def place(oid):
                cert=solve_center(bodies[oid],edges[oid],plane,local_boxes,case['hfov'],case['image_size'],regions=(region,))
                local_cert[oid]=cert;local_boxes[oid]=dict(id=oid,name=byid[oid]['name'],center=cert['center'],size=bodies[oid]['size'])
                for child in children[oid]:place(child)
            try:place(root)
            except ValueError:continue
            cost=sum(c['translation_norm_m']**2 for c in local_cert.values())
            candidates.append((cost,region,local_boxes,local_cert))
        if not candidates:raise ValueError('Support component has no solution in the five sufficient regions; all candidates retained')
        cost,region,boxes,chosen=min(candidates,key=lambda c:c[0]);certificates.update(chosen)
        components.append(dict(root=root,ids=list(chosen),region=region,squared_translation_cost=cost))
    raw=[]
    for oid in out['placement_dependency_order']:
        cert=certificates[oid];c=cert['center'];body=bodies[oid]
        raw.append(dict(id=oid,azimuth_deg=math.degrees(math.atan2(c[0],c[2])),range_m=math.hypot(c[0],c[2]),
            size=body['size'],center_y_m=c[1],reason=body['reason']))
        obj=next(o for o in out['additions'] if o['id']==oid)
        obj['placement']='Outside original input frustum: '+cert['off_fov_certificate_region']
    return out,dict(objects=raw),certificates,components


def solve_typed(model,image,case,observed,plan,plane,dest,options,choose=None):
    choose=choose or finite_choice
    observed_boxes=initial_boxes(case,observed);bodies={};metric_records=[];support_records=[];selected={}
    common=('Infer a layout of this FIXED sampled existing off-camera inventory around the original camera. '
        'Do not redesign the room or add required equipment. All observed boxes are frozen. '+
        '\nSCENE: '+plan['scene_hypothesis']+'\nFIXED OBSERVED: '+json.dumps(observed_boxes,ensure_ascii=False)+
        '\nSAMPLED INVENTORY: '+json.dumps([dict(id=o['id'],name=o['name']) for o in plan['additions']],ensure_ascii=False))
    # Get geometry essentials first. Explanatory prose is never a numeric gate.
    for obj in plan['additions']:
        oid=obj['id'];stem='metric_'+oid
        prompt=(common+'\nONLY TARGET: '+json.dumps(obj,ensure_ascii=False)+
            '\nReturn ONE numeric JSON BODY {"azimuth_deg":number,"range_m":number,"size":[sx,sy,sz],'
            '"center_y_m":number}. No id or objects array. reason is optional and unnecessary. '
            'Metres, FULL original camera-axis AABB lengths. x right, y image-up, z forward. '
            'Azimuth 0 forward, +90 right, -90/270 left, 180 back; range is horizontal sqrt(x*x+z*z). '
            'center_y_m is optional for floor/object support and signed relative to the camera, not height '
            'above floor. This is an initial proposal; support selection and final layout follow in code.')
        body=ask_validated(model,image,prompt,dest/(stem+'.txt'),numeric_body,
            {**options,'seed':options['seed']+len(metric_records)})
        bodies[oid]=body;metric_records.append(dict(id=oid,stem=stem,body=body))
        write(dest/'typed_layout_progress.json',dict(stage='metric_proposals',metric_records=metric_records,
            remaining_ids=[o['id'] for o in plan['additions'] if o['id'] not in bodies]))
    # A future NEW host may be chosen. Cycle masks and topological ordering are
    # program-owned; observed infeasible hosts are rejected before LM selection.
    for obj in plan['additions']:
        oid=obj['id'];stem='support_'+oid
        choices,rejected=feasible_supports(oid,bodies,plan,selected,observed_boxes,plane,case)
        choice=choose(model,image,common+'\nONLY TARGET: '+json.dumps(obj,ensure_ascii=False)+
            '\nSIZE AND INITIAL PROPOSAL: '+json.dumps(bodies[oid])+ 
            '\nChoose actual supporting contact, not adjacency. A NEW sampled host may be placed later '
            'by dependency order. Known observed supports that cannot put this box OUTSIDE the original '
            'input view have already been excluded. Preserve the sampled objects; never add a host.',choices,dest/(stem+'.txt'))
        if choice not in choices:raise ValueError('Support selection outside admissible finite domain')
        selected[oid]=choice;support_records.append(dict(id=oid,stem=stem,choice=choice,rejected_observed_or_unmodeled=rejected))
        write(dest/'typed_layout_progress.json',dict(stage='support_graph',metric_records=metric_records,
            support_records=support_records,remaining_ids=[o['id'] for o in plan['additions'] if o['id'] not in selected]))
    out,raw,certificates,components=compile_dependency_layout(case,observed,plan,plane,bodies,selected)
    steps=[dict(id=oid,body=bodies[oid],support=selected[oid],certificate=certificates[oid]) for oid in out['placement_dependency_order']]
    saved=dict(schema_version=2,flow='metrics_then_feasible_support_DAG_then_component_layout',
        metric_records=metric_records,support_records=support_records,steps=steps,components=components)
    mappings=map_plan(out,'semantic_lexicon_v1')
    items,review=compile_layout(raw,out,mappings,observed,plane,case['hfov'],case['image_size'])
    decorate_review(review,saved)
    write(dest/'geometry_raw.json',raw);write(dest/'typed_layout.json',saved)
    return out,mappings,items,review


def decorate_review(review,saved):
    review.update(position_was_projected=any(s['certificate']['translation_norm_m']>1e-9 for s in saved['steps']),
        layout_backend='metric_first_feasible_support_DAG_component_solver',layout_adjustments=saved['steps'],
        layout_components=saved['components'],complete_global_layout_solver=False)
    return review


def replay_typed(case,observed,plan,plane,saved):
    if saved.get('schema_version')!=2:raise ValueError('Need dependency-layout schema version 2')
    ids={o['id'] for o in plan['additions']};bodies={};selected={};boxes=initial_boxes(case,observed)
    for row in saved['metric_records']:
        oid=row['id']
        if oid not in ids or oid in bodies:raise ValueError('Typed object identity changed')
        body=numeric_body({k:v for k,v in row['body'].items() if k!='canonical_azimuth_deg'})
        if body!=row['body']:raise ValueError('Typed numeric normalization changed')
        bodies[oid]=body
    if set(bodies)!=ids:raise ValueError('Missing typed metric instances')
    for row in saved['support_records']:
        oid=row['id']
        if oid not in ids or oid in selected:raise ValueError('Typed support identity changed')
        choices,rejected=feasible_supports(oid,bodies,plan,selected,boxes,plane,case)
        if row['choice'] not in choices or row['rejected_observed_or_unmodeled']!=rejected:raise ValueError('Support feasibility or cycle mask differs')
        selected[oid]=row['choice']
    out,raw,certificates,components=compile_dependency_layout(case,observed,plan,plane,bodies,selected)
    steps=[dict(id=oid,body=bodies[oid],support=selected[oid],certificate=certificates[oid]) for oid in out['placement_dependency_order']]
    if steps!=saved['steps'] or components!=saved['components']:raise ValueError('Typed component solution differs')
    return out,raw
