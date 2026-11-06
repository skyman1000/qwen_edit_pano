"""Installation roles -> admissible hosts -> joint layout of fixed new objects.

These are conditional layout hypotheses, not new existence-causal mechanisms.
Every model choice and program resolution is saved; no sampled object is lost.
"""
from copy import deepcopy
import json

from ..common import write
from .conditional_world_geometry import compile_layout
from .frozen_causal_world import ask_validated
from .scm_joint_geometry import LayoutInfeasible,compile_joint,observed_host_feasible
from .scm_typed_layout import creates_cycle,initial_boxes,numeric_body
from .scm_typed_queries import finite_choice
from .scene_completion import map_plan

PROFILES=[dict(installation=k,surface=s) for k,s in (
    ('floor','top'),('floor','seat'),('floor','none'),('surface','top'),('surface','none'),
    ('seat','none'),('wall','top'),('wall','none'),('ceiling','none'),('unresolved','unknown'))]


def choose_value(model,image,prompt,candidates,path,choose):
    if len(candidates)==1:
        value=deepcopy(candidates[0]);path.write_text(json.dumps(value,ensure_ascii=False))
        write(path.with_suffix('.json'),value)
        write(path.with_suffix('.request.json'),dict(method='program_singleton_v1',prompt=prompt,
            candidates=candidates,image_used=False))
        return value
    return choose(model,image,prompt,candidates,path)


def host_choices(oid,profiles,bodies,plan,selected,old,plane,case):
    mode=profiles[oid]['installation'];rejected=[]
    if mode=='floor':
        if plane['floor_y_m'] is not None:
            return [dict(kind='floor',target=None,description='standing on declared floor prior')],rejected
        return [dict(kind='unspecified',target=None,description='floor installation intended; floor height unresolved')],rejected
    if mode in ('wall','ceiling','unresolved'):
        kind=mode if mode in ('wall','ceiling') else 'unspecified'
        return [dict(kind=kind,target=None,description='installation geometry unmeasured')],rejected
    choices=[];new={o['id']:o for o in plan['additions']}
    needed='seat' if mode=='seat' else 'top'
    for target in list(old)+list(new):
        if target==oid or creates_cycle(oid,target,selected):continue
        if profiles[target]['surface'] not in (needed,'unknown'):continue
        if target in old and not observed_host_feasible(oid,bodies[oid],target,old,plane,case):
            rejected.append(dict(target=target,reason='No off-view central-contact-point placement on frozen host'))
            continue
        name=old[target]['name'] if target in old else new[target]['name']
        choices.append(dict(kind='object',target=target,description='supported by '+name))
    choices.append(dict(kind='unspecified',target=None,description='support unresolved; preserve object for review'))
    return choices,rejected


def metric_validator(value,profile,plane):
    body=numeric_body(value)
    if (profile['installation']!='floor' or plane['floor_y_m'] is None) and body['center_y_m'] is None:
        raise ValueError('This installation needs an explicit signed center_y_m for unmeasured support alternatives')
    return body


def resolve_geometry(case,observed,plan,plane,bodies,selected):
    resolved=deepcopy(selected);changes=[]
    for _ in range(len(selected)+1):
        try:
            result=compile_joint(case,observed,plan,plane,bodies,resolved)
            return (*result,resolved,changes)
        except LayoutInfeasible as exc:
            # Resolve ONE edge in the failing component only. Do not withdraw
            # unrelated support hypotheses to make the whole world export.
            component=set(exc.component_ids);local=deepcopy(plan)
            local['additions']=[o for o in plan['additions'] if o['id'] in component]
            local_bodies={i:bodies[i] for i in component}
            local_selected={i:resolved[i] for i in component};alternatives=[]
            for oid in sorted(component):
                if resolved[oid]['kind']!='object':continue
                trial=deepcopy(local_selected)
                trial[oid]=dict(kind='unspecified',target=None,description='joint support layout infeasible; unresolved')
                try:_,_,check=compile_joint(case,observed,local,plane,local_bodies,trial,rounds=0)
                except LayoutInfeasible:continue
                alternatives.append((check['energy_after'],oid,trial[oid]))
            if not alternatives:raise
            _,oid,edge=min(alternatives,key=lambda a:(a[0],a[1]))
            changes.append(dict(id=oid,requested_support=deepcopy(resolved[oid]),
                failing_component_ids=sorted(component),reason=str(exc)))
            resolved[oid]=edge
    raise RuntimeError('Support-resolution progress invariant violated')


def decorate_joint(review,saved,items,observed):
    boxes={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    boxes.update({o['id']:o for o in items});checks=[];unresolved=[]
    for obj in saved['resolved_plan']['additions']:
        oid=obj['id'];support=obj['support']
        if support['kind']=='object':
            a,b=boxes[oid],boxes[support['target']]
            inside=all(abs(a['center'][d]-b['center'][d]) <= b['size'][d]/2+1e-7 for d in (0,2))
            checks.append(dict(id=oid,target=support['target'],central_contact_point_in_host=inside,
                physical_stability_verified=False))
            if saved['profiles'][support['target']]['surface']=='unknown':unresolved.append(oid)
        elif support['kind']=='unspecified':unresolved.append(oid)
    failures=sum(not c['central_contact_point_in_host'] for c in checks)
    remaining=saved['solver']['final_overlaps']
    review.update(layout_backend='installation_roles_joint_XZ_soft_repulsion',
        support_point_checks=checks,central_contact_failures=failures,
        unresolved_installations=sorted(set(unresolved)),joint_layout=saved['solver'],
        program_support_resolutions=saved['program_resolutions'],
        layout_proxy_passed=not remaining and not unresolved and not failures and
            saved['solver'].get('solver_converged',True),
        complete_global_layout_solver=False,physical_geometry_verified=False,
        sizes_changed=False,observed_state_preserved=True,position_was_projected=bool(items))
    review['geometry_contract_passed']=review['geometry_contract_passed'] and failures==0
    return review


def solve_relational(model,image,case,observed,plan,plane,dest,options,choose=None):
    choose=choose or finite_choice;old=initial_boxes(case,observed)
    profiles={};profile_records=[];metric_records=[];support_records=[];bodies={};selected={}
    context=('Infer the plausible EXISTING world around the ORIGINAL camera, not a room redesign. '
        'Observed objects are frozen. No objects may be added, withdrawn or renamed. '
        'Every new instance belongs outside the ORIGINAL input frustum, on the sides or behind the camera as appropriate. '
        '\nSCENE HYPOTHESIS: '+plan['scene_hypothesis']+
        '\nOBSERVED: '+json.dumps(old,ensure_ascii=False)+
        '\nFIXED OUTSIDE INVENTORY: '+json.dumps(plan['additions'],ensure_ascii=False))
    if plan['additions']:
        for oid,name in [(i,b['name']) for i,b in old.items()]+[(o['id'],o['name']) for o in plan['additions']]:
            stem='profile_'+oid
            prompt=(context+'\nONLY TARGET: '+json.dumps(dict(id=oid,name=name),ensure_ascii=False)+
                '\nChoose its ordinary intended installation and usable support surface. '
                'installation floor: standing on floor; surface: a small item on a horizontal host; '
                'seat: cushion/pillow on a seating/bedding surface; wall/ceiling: mounted or suspended; '
                'unresolved: evidence is insufficient. surface describes WHAT THIS TARGET CAN SUPPORT: '
                'top=usable horizontal surface, seat=seating/bedding, none=no usable support, unknown=unresolved. '
                'A floor table/desk is a floor object with a top; a tray is a surface object with a top. '
                'A sofa is a floor object with a seat. A floor lamp and a table lamp have different installations. '
                'These are layout hypotheses, not observed category changes or causal facts.')
            value=choose_value(model,image,prompt,PROFILES,dest/(stem+'.txt'),choose)
            if value not in PROFILES:raise ValueError('Installation role outside finite domain')
            profiles[oid]=value;profile_records.append(dict(id=oid,stem=stem,choice=value))
        for obj in plan['additions']:
            oid=obj['id'];stem='metric_'+oid;profile=profiles[oid]
            prompt=(context+'\nONLY TARGET: '+json.dumps(obj,ensure_ascii=False)+
                '\nINSTALLATION ROLE: '+json.dumps(profile)+
                '\nReturn ONE JSON body {"azimuth_deg":number,"range_m":number,"size":[sx,sy,sz],"center_y_m":number}. '
                'No id or objects array. Metres, FULL original camera-axis AABB lengths. '
                'x right, y image-up, z forward; azimuth 0 forward, +90 right, -90 left, 180 back. '
                'range_m=sqrt(x*x+z*z). center_y_m is signed relative to camera, not height above floor. '
                'Supply a plausible signed center_y_m even if support is uncertain. '
                'This is an initial position proposal; later joint solving preserves sizes and all observed boxes. '
                'reason is optional.')
            body=ask_validated(model,image,prompt,dest/(stem+'.txt'),
                lambda v:metric_validator(v,profile,plane),{**options,'seed':options['seed']+len(metric_records)})
            bodies[oid]=body;metric_records.append(dict(id=oid,stem=stem,body=body))
        # Fixed floor/mounting roots are recorded before dependent accessories.
        ordered=sorted(plan['additions'],key=lambda o:(profiles[o['id']]['installation'] in ('surface','seat'),
            profiles[o['id']]['surface']=='none',o['id']))
        for obj in ordered:
            oid=obj['id'];stem='support_'+oid
            candidates,rejected=host_choices(oid,profiles,bodies,plan,selected,old,plane,case)
            prompt=(context+'\nONLY TARGET: '+json.dumps(obj,ensure_ascii=False)+
                '\nINSTALLATION ROLES: '+json.dumps(profiles)+
                '\nFULL SIZES AND INITIAL POSES: '+json.dumps(bodies)+
                '\nChoose the actual host for this fixed instance. Use the role-compatible finite domain. '
                'Consider all objects, including NEW hosts. Geometrically incompatible visible hosts are excluded. '
                'Do not confuse adjacency with support. If a plausible host cannot be established, choose unresolved.')
            choice=choose_value(model,image,prompt,candidates,dest/(stem+'.txt'),choose)
            if choice not in candidates:raise ValueError('Support outside role-compatible finite domain')
            selected[oid]=choice;support_records.append(dict(id=oid,stem=stem,choice=choice,rejected=rejected))
    out,raw,solver,resolved,changes=resolve_geometry(case,observed,plan,plane,bodies,selected)
    saved=dict(schema_version=1,flow='installation_roles_then_metrics_then_hosts_then_joint_layout',
        profiles=profiles,profile_records=profile_records,metric_records=metric_records,support_records=support_records,
        resolved_supports=resolved,program_resolutions=changes,resolved_plan=out,solver=solver)
    mappings=map_plan(out,'semantic_lexicon_v1')
    items,review=compile_layout(raw,out,mappings,observed,plane,case['hfov'],case['image_size'])
    decorate_joint(review,saved,items,observed)
    write(dest/'geometry_raw.json',raw);write(dest/'joint_layout.json',saved)
    return out,mappings,items,review


def replay_relational(case,observed,plan,plane,saved):
    if saved.get('schema_version')!=1:raise ValueError('Unknown joint layout schema')
    ids=[o['id'] for o in plan['additions']];old=initial_boxes(case,observed);profiles={};bodies={};selected={}
    expected=set(ids)|set(old) if ids else set()
    for r in saved['profile_records']:
        if r['id'] not in expected or r['id'] in profiles or r['choice'] not in PROFILES:raise ValueError('Profile identity or role changed')
        profiles[r['id']]=r['choice']
    if set(profiles)!=expected or profiles!=saved['profiles']:raise ValueError('Missing installation roles')
    for r in saved['metric_records']:
        oid=r['id']
        if oid not in ids or oid in bodies:raise ValueError('Metric identity changed')
        body=metric_validator({k:v for k,v in r['body'].items() if k!='canonical_azimuth_deg'},profiles[oid],plane)
        if body!=r['body']:raise ValueError('Metric normalization changed')
        bodies[oid]=body
    if set(bodies)!=set(ids):raise ValueError('Missing fixed metric objects')
    for r in saved['support_records']:
        oid=r['id']
        if oid not in ids or oid in selected:raise ValueError('Support identity changed')
        candidates,rejected=host_choices(oid,profiles,bodies,plan,selected,old,plane,case)
        if r['choice'] not in candidates or r['rejected']!=rejected:raise ValueError('Host domain changed')
        selected[oid]=r['choice']
    if set(selected)!=set(ids):raise ValueError('Missing support assignments')
    out,raw,solver,resolved,changes=resolve_geometry(case,observed,plan,plane,bodies,selected)
    from .scm_world_audit import close
    if not close(dict(resolved_plan=out,solver=solver,resolved_supports=resolved,program_resolutions=changes),
        {k:saved[k] for k in ('resolved_plan','solver','resolved_supports','program_resolutions')}):
        raise ValueError('Joint layout replay differs')
    return out,raw
