"""Frozen-LM, finite joint-world hypothesis SCM; no per-group absence sampling.

C -> Z -> E and Z -> H. H expands a complete joint scene program. E is ONE
semantic observation event, not an RGB likelihood. The domain is proposed using
the image; all probabilities are subjective and conditional on that domain.
"""
from copy import deepcopy
import json

from .scene_program import text
from .world_scm import parent_rows,key,validate
from .world_scm_spec import table_answer
from .world_scm_runner import domain_response

PROTOCOL='fresh_joint_configuration_scm_v1'
RELATIONS={'requires','near','left_of','right_of','above','below','in_front_of','behind'}


def registry_answer(value,limit=3):
    if not isinstance(value,dict):raise ValueError('Need contexts and configurations object')
    result={}
    for field,prefix in [('contexts','c'),('configurations','z')]:
        rows=value.get(field)
        if not isinstance(rows,list) or not 1<=len(rows)<=limit or any(not isinstance(r,dict) for r in rows):
            raise ValueError(field+f' needs 1..{limit} records')
        result[field]=[]
        for i,row in enumerate(rows):
            record=dict(id=prefix+str(i),description=text(row.get('description'),field+'.description'))
            if field=='configurations':
                record['visible_expectations']=text(row.get('visible_expectations'),'configuration.visible_expectations')
            result[field].append(record)
        if len({r['description'] for r in result[field]})!=len(rows):raise ValueError('Duplicate '+field+' descriptions')
    return result


def registry_prompt(case,grounding,limit):
    return ('Infer alternative EXISTING whole-room configurations consistent with this partial indoor view. '
        'The target is hidden surroundings of this same camera for a 360-degree world, not improvements to the photographed arrangement. '
        'Configurations can contain several coexisting functional areas; a seating area and lighting area are not competing whole rooms. '
        'Return {"contexts":[{"description":"possible prior room use / occupant preference"}],'
        '"configurations":[{"description":"one possible overall organization, including surrounding unseen space",'
        '"visible_expectations":"what local positive object/relationship evidence this organization tends to produce"}]}. '
        f'Use 1..{limit} distinct contexts and 1..{limit} distinct whole-room alternatives. This is a hypothesis budget, not a furniture count. '
        'Alternative arrangements may share objects. Do not invent alternatives solely to fill the budget. '
        'Do not give xyz, sizes or a generic shopping list. Observed items remain fixed. No scene GT is supplied.\n'
        'POSITIVE LOCAL OBSERVATION: '+json.dumps(grounding,ensure_ascii=False)+'\nOBSERVED INVENTORY: '+
        json.dumps([{k:o[k] for k in ['id','category','description','visible_bbox_xyxy_norm']} for o in case['context']],ensure_ascii=False))


def program_prompt(case,grounding,configuration,budgets):
    return ('Describe ONE plausible COMPLETE inventory of the EXISTING unphotographed surroundings under the configuration below. '
        'Jointly choose furniture, storage, everyday belongings and decor that make a plausible surrounding world, including optional contents. '
        'Preserve all observed instances. Do not merely add cushions or accessories to the photographed view, finish visible shapes, redesign a room, '
        'or produce only the minimum equipment for an activity. Unphotographed is unknown, not absent. '
        'Each group is a spatially coherent assembly in this same whole-room hypothesis; ALL groups and their objects coexist. '
        'Support-connected objects belong to the same group. There is NO automatic per-group absent option or independent object selection. '
        f'Return 0..{budgets["groups"]} groups, at most {budgets["max_new"]} total new objects; these are computation limits, no minimum or direction quotas. '
        'An empty inventory is allowed only as your actual hypothesis, not because evidence outside the photograph is unavailable. '
        'Return {"groups":[{"name":"assembly","reason":"why it may already exist in the unseen surrounding room",'
        '"objects":[{"key":"globally_unique_local_key","name":"specific noun","category":"ordinary noun",'
        '"variant":"specific type","identity":"new_unobserved","distinct_instance_reason":"separate unseen instance, not the visible one moved",'
        '"placement_hint":"qualitative location OUTSIDE input view in original camera coordinates",'
        '"support_target":"floor"}]}],"relations":[{"source":"new key or observed ID","target":"new key or observed ID",'
        '"kind":"requires","reason":"specific declared dependence or spatial rationale"}]}. '
        'support_target is ONE literal floor/wall/ceiling/unspecified, exact observed ID, or exact new host key in the SAME group. '
        'Do not use a category noun as an instance ID. All new keys are unique across this whole hypothesis. '
        'relations kinds: requires, near, left_of, right_of, above, below, in_front_of, behind. '
        'requires means logical presence dependence of this specific variant; do not declare mere typical co-occurrence necessary. '
        'Support is separately encoded by support_target; a table lamp may require its supporting table, a floor lamp does not. '
        'Directions mean centre ordering in CAMERA axes x right/y image-up/z optical forward, not furniture-facing direction. '
        'A relation need not be added to every pair. No metric geometry in this stage. No Adapter class restriction.\n'
        'CONFIGURATION: '+json.dumps(configuration,ensure_ascii=False)+'\nOBSERVED: '+json.dumps(grounding,ensure_ascii=False)+
        '\nORIGINAL CAMERA: '+json.dumps(dict(hfov=case['hfov'],image_size=case['image_size'])))


def program_answer(value,case,configuration,budgets):
    if not isinstance(value,dict) or not isinstance(value.get('groups'),list):raise ValueError('Need groups and relations object')
    groups=[];keys=[]
    for i,g in enumerate(value['groups']):
        if not isinstance(g,dict) or not isinstance(g.get('objects'),list) or not g['objects']:
            raise ValueError(f'groups[{i}] needs a nonempty objects array; omit a genuinely absent assembly')
        if any(not isinstance(o,dict) or not isinstance(o.get('key'),str) for o in g['objects']):
            raise ValueError(f'groups[{i}].objects need string keys')
        keys.extend(o['key'] for o in g['objects'])
        groups.append(dict(name=text(g.get('name'),'group.name'),mechanism_reason=text(g.get('reason'),'group.reason'),parents=[],
            modes=[dict(description=configuration['description'],observed_role_ids=[],objects=g['objects'])]))
    if len(set(keys))!=len(keys):raise ValueError('New keys must be unique across the whole joint configuration')
    ambiguous=set(keys)&{'floor','wall','ceiling','unspecified'}
    if ambiguous:raise ValueError('Rename new instance keys that collide with surface tokens: '+', '.join(sorted(ambiguous)))
    # Reuse only the typed domain compiler. Its internal zero modes are NEVER
    # assigned probabilities or sampled; every group selects mode 1 together.
    bindings=[]
    domain=domain_response(dict(contexts=[dict(description='Compiler-only fixed context')],
        organizations=[dict(description=configuration['description'])],groups=groups),case,budgets,bindings=bindings)
    allowed=set(keys)|{o['id'] for o in case['context']};relations=value.get('relations',[]);normalized=[];seen=set()
    if not isinstance(relations,list):raise ValueError('relations must be an array')
    for i,r in enumerate(relations):
        if not isinstance(r,dict):raise ValueError(f'relations[{i}] must be a record')
        a,b,k=r.get('source'),r.get('target'),r.get('kind')
        if not isinstance(a,str) or not isinstance(b,str) or a not in allowed or b not in allowed or a==b or k not in RELATIONS:
            raise ValueError(json.dumps(dict(path=f'relations[{i}]',error='Exact existing endpoints, no self edge, and a supported relation kind are required',
                allowed_endpoints=sorted(allowed),allowed_kinds=sorted(RELATIONS))))
        triple=(a,b,k)
        if triple in seen:raise ValueError(f'Duplicate relation at relations[{i}]')
        seen.add(triple);normalized.append(dict(source=a,target=b,kind=k,reason=text(r.get('reason'),'relation.reason')))
    return dict(configuration=deepcopy(configuration),geometry_domain=domain,relations=normalized,bindings=bindings,
        new_instances=len(keys),all_groups_jointly_present=True)


def geometry_state(program):return {g['id']:'1' for g in program['geometry_domain']['groups']}


def translated_relations(program):
    ids={o['key']:f'sp_g{gi:02d}_'+o['key'] for gi,g in enumerate(program['geometry_domain']['groups']) for o in g['modes'][1]['objects']}
    return [{**r,'source':ids.get(r['source'],r['source']),'target':ids.get(r['target'],r['target'])} for r in program['relations']]


def build_model(registry,programs,evidence,ask):
    nodes=[];catalog={}
    def add(name,states,parents,descriptions,kind):
        node=dict(id=name,states=states,parents=parents);rows=parent_rows(nodes,parents)
        payload=dict(variable=name,states=descriptions,declared_parent_domains={p:catalog[p] for p in parents},
            required_given_rows=rows,mechanism_kind=kind)
        if len(states)==1:table={key([r[p] for p in parents]):[1.] for r in rows}
        else:table=ask(name,payload,lambda value:table_answer(value,node,rows))
        node['cpt']=table;nodes.append(node);catalog[name]=descriptions
    add('C',[r['id'] for r in registry['contexts']],[],registry['contexts'],'subjective prior over intended room use/preferences')
    configs=[]
    for c in registry['configurations']:
        p=programs[c['id']]
        configs.append(dict(**c,joint_hidden_groups=[dict(name=g['name'],objects=g['modes'][1]['objects'])
            for g in p['geometry_domain']['groups']],declared_relations=p['relations']))
    add('Z',[r['id'] for r in configs],['C'],configs,'joint whole-world organization and inventory given intended use; no independent object absence')
    add('E',['other','signature'],['Z'],[dict(id='other',description='Joint positive local semantic signature is not recorded'),
        dict(id='signature',description=evidence)],'likelihood of the joint positive local semantic signature under the WHOLE configuration and fixed camera; '
        'unseen objects are marginalized, not counted as negative detections. Not an RGB likelihood or posterior confidence.')
    zs=[r['id'] for r in configs]
    nodes.append(dict(id='H',states=zs,parents=['Z'],cpt={z:[float(z==v) for v in zs] for z in zs}))
    return validate(dict(protocol=PROTOCOL,nodes=nodes,mechanism_source='LM_elicited_subjective_CPT_plus_deterministic_joint_expansion',
        domain_selected_with_evidence=True,RGB_likelihood_defined=False,real_causal_structure_identified=False,
        group_absence_sampling=False,geometry_density_defined=False))
