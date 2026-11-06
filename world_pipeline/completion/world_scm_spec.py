"""Semantic domains and parent-only mechanism elicitation for the world SCM."""
from copy import deepcopy
import math

from ..common import finite_vector
from .scene_program import KEY,SUPPORTS,dependency_order,text
from .world_scm import PROTOCOL,distribution,key,parent_rows,validate


def grounding_answer(value,case):
    rows=value.get('instances',[]);expected={o['id'] for o in case['context']}
    if len(rows)!=len(expected) or {r.get('id') for r in rows}!=expected:
        raise ValueError('instances must bind every original observed ID once')
    out=[]
    for row in rows:
        if type(row.get('on_floor')) is not bool:raise ValueError('on_floor must be a boolean')
        out.append(dict(id=row['id'],variant=text(row.get('variant'),'variant'),on_floor=row['on_floor']))
    return dict(instances=out,evidence_description=text(value.get('evidence_description'),'positive visual evidence'))


def domain_answer(value,case,max_groups=4,max_new=24):
    result={}
    for name,prefix in [('contexts','c'),('organizations','s')]:
        rows=value.get(name)
        if not isinstance(rows,list) or not 1<=len(rows)<=3:raise ValueError(name+' must contain 1..3 hypotheses')
        result[name]=[dict(id=prefix+str(i),description=text(r.get('description'),name)) for i,r in enumerate(rows)]
    observed={o['id'] for o in case['context']};groups=value.get('groups');maximum=0
    if not isinstance(groups,list) or not 0<=len(groups)<=max_groups:raise ValueError('Invalid group budget')
    result['groups']=[]
    for gi,g in enumerate(groups):
        parents=g.get('parents',[])
        if (not isinstance(parents,list) or len(set(parents))!=len(parents) or
                any(type(v) is not int or not 0<=v<gi for v in parents)):raise ValueError('parents must refer to earlier group indices')
        modes=g.get('modes',[])
        if not 1<=len(modes)<=2:raise ValueError('Each group needs 1..2 nonempty joint alternatives')
        normalized=[dict(id=0,description='This optional group has no additional unseen instances.',objects=[],observed_role_ids=[])]
        for mi,m in enumerate(modes,1):
            objects=m.get('objects',[]);existing=m.get('observed_role_ids',[])
            if not isinstance(objects,list) or not 1<=len(objects)<=max_new:raise ValueError('Need nonempty semantic objects')
            if not isinstance(existing,list) or any(i not in observed for i in existing):raise ValueError('Unknown reused observed ID')
            records=[]
            for obj in objects:
                oid=obj.get('key');support=obj.get('support',{});kind=support.get('kind');target=support.get('target')
                if not isinstance(oid,str) or not KEY.fullmatch(oid) or oid in observed or oid=='camera':raise ValueError('Invalid new object key')
                if obj.get('identity')!='new_unobserved':raise ValueError('New objects require identity=new_unobserved; reuse original IDs via observed_role_ids')
                if any(k in obj for k in ['center','size_m','offset_m','width_m']):raise ValueError('Semantic mechanism domain must not mix metric geometry into objects')
                if kind not in SUPPORTS:raise ValueError('Invalid support kind')
                if kind=='object':
                    if not isinstance(target,str) or target=='camera':raise ValueError('object support requires a local or observed instance target')
                elif target is not None:raise ValueError(f'{oid}.support.target must be null for {kind}; location anchors are separate')
                records.append(dict(key=oid,name=text(obj.get('name'),'name'),category=text(obj.get('category'),'category'),
                    variant=text(obj.get('variant'),'variant'),identity='new_unobserved',
                    distinct_instance_reason=text(obj.get('distinct_instance_reason'),'independent hidden identity'),
                    placement_hint=text(obj.get('placement_hint'),'qualitative off-camera placement'),
                    support=dict(kind=kind,target=target)))
            if len({o['key'] for o in records})!=len(records):raise ValueError('Duplicate new identity')
            dependency_order([{**o,'anchor':o['support']['target'] or 'camera'} for o in records],observed)
            normalized.append(dict(id=mi,description=text(m.get('description'),'mode'),objects=records,observed_role_ids=existing))
        maximum+=max(len(m['objects']) for m in normalized)
        result['groups'].append(dict(id=f'g{gi}',name=text(g.get('name'),'group'),parents=sorted(parents),
            mechanism_reason=text(g.get('mechanism_reason'),'mechanism rationale'),modes=normalized))
    if maximum>max_new:raise ValueError('Largest complete program exceeds requested capacity; do not drop objects')
    result.update(maximum_new_instances=maximum,protocol=PROTOCOL)
    return result


def table_answer(value,node,rows):
    records=value.get('rows',[])
    if len(records)!=len(rows):raise ValueError('CPT must contain every requested parent assignment exactly once')
    expected={key([r[p] for p in node['parents']]) for r in rows};table={}
    for row in records:
        given=row.get('given',{})
        if set(given)!=set(node['parents']):raise ValueError('CPT row has wrong parent variables')
        k=key([given[p] for p in node['parents']])
        if k not in expected or k in table:raise ValueError('Unknown or duplicate CPT row')
        table[k]=distribution(row.get('p'),len(node['states']))
    return table


def build_model(domain,evidence,ask):
    """E is a joint positive semantic event; no absent/unseen objects counted as negative evidence."""
    nodes=[];catalog={}
    def add(name,states,parents,descriptions,kind):
        node=dict(id=name,states=states,parents=parents)
        rows=parent_rows(nodes,parents)
        table={}
        # Split large CPTs instead of truncating relationships or asking for a
        # single answer longer than the provider's output budget.
        for start in range(0,len(rows),24):
            chunk=rows[start:start+24]
            payload=dict(variable=name,mechanism_kind=kind,states=descriptions,
                declared_parent_domains={p:catalog[p] for p in parents},required_given_rows=chunk,
                response_schema={'rows':[{'given':r,'p':['one probability per state, sums to 1']} for r in chunk]})
            query=name if len(rows)<=24 else name+f'_part{start//24:02d}'
            value=ask(query,payload,lambda v,chunk=chunk:table_answer(v,node,chunk))
            table.update(value)
        node['cpt']=table;nodes.append(node);catalog[name]=descriptions
    add('C',[c['id'] for c in domain['contexts']],[],domain['contexts'],'prior over surrounding-room organization/preferences')
    add('S',[s['id'] for s in domain['organizations']],['C'],domain['organizations'],
        'generative selection of spatial/activity organization, not posterior classification of the photograph')
    event=dict(id='signature',description=evidence)
    add('E',['other','signature'],['C','S'],[dict(id='other',description='The joint positive local evidence signature is not recorded.'),event],
        'subjective likelihood of jointly recording the stated positive local semantic signature with the given camera. '
        'Marginalize unobserved outside contents; not an RGB likelihood, no missing-detection count model. '
        'All correlated visible facts form ONE event, not a product of independent object confidences.')
    for group in domain['groups']:
        add(group['id'],[str(m['id']) for m in group['modes']],['C','S']+[f'g{p}' for p in group['parents']],
            [dict(id=str(m['id']),description=m['description'],joint_instances=m['objects'],
                  reused_observed_roles=m['observed_role_ids']) for m in group['modes']],
            'generative joint hidden-group variant, including optional absence. '
            'Condition ONLY on declared parents. Infer existing surrounding contents, not required purchases or redesign.')
    return validate(dict(protocol=PROTOCOL,nodes=nodes,mechanism_source='LM_elicited_subjective_CPT_not_token_logprobs',
        domain_selected_with_evidence=True,RGB_likelihood_defined=False,real_causal_structure_identified=False,
        observation_model='single_positive_semantic_signature',geometry_density_defined=False))


def geometry_answer(value,domain,state,case):
    selected=[(g,g['modes'][int(state[g['id']])]) for g in domain['groups'] if state[g['id']]!='0']
    expected={g['id']+'_'+o['key']:(g,o) for g,m in selected for o in m['objects']}
    rows=value.get('objects',[])
    if len(rows)!=len(expected) or {r.get('id') for r in rows}!=set(expected):raise ValueError('Geometry must cover every selected new instance once')
    indexed={r['id']:r for r in rows};out=[];conversions={}
    for group,mode in selected:
        objects=[]
        for semantic in mode['objects']:
            oid=group['id']+'_'+semantic['key'];row=indexed[oid]
            dims=[row.get(k) for k in ['width_m','height_m','depth_m']];yaw=row.get('yaw_deg')
            if not finite_vector(dims,3) or min(dims)<=0 or type(yaw) not in (int,float) or not math.isfinite(yaw):
                raise ValueError(oid+' requires positive width/height/depth and finite camera-Y yaw')
            angle=math.radians(yaw);w,h,d=dims
            size=[abs(math.cos(angle))*w+abs(math.sin(angle))*d,h,abs(math.sin(angle))*w+abs(math.cos(angle))*d]
            support=deepcopy(semantic['support']);anchor=row.get('anchor');offset=row.get('offset_m')
            if not finite_vector(offset,3) or not isinstance(anchor,str):raise ValueError('Need signed relative offset and anchor')
            # Semantic host references stay fixed. Dimensions are converted once
            # to full CAMERA AABBs; camera-Y-level is explicitly only a prior.
            objects.append(dict(key=semantic['key'],name=semantic['name'],category=semantic['category'],variant=semantic['variant'],
                size_m=size,anchor=anchor,offset_m=offset,support=support))
            conversions[oid]=dict(intrinsic_dimensions_m=dims,yaw_about_camera_Y_deg=yaw,camera_AABB_size_m=size)
        out.append(dict(name=group['name'],explanation=group['mechanism_reason'],parents=[],
                        modes=[dict(description=mode['description'],objects=objects)]))
    return dict(groups=out),conversions
