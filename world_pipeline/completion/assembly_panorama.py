"""v9 functional-spatial realizations, with protected camera-region state R.

No rear quota, no weights training. A service link is not a metric near edge.
Spatial programs are hypotheses sampled with their inventory variants, not facts.
"""
from copy import deepcopy
from html import escape
import json
import math

import numpy as np
from scipy.optimize import linprog

from ..common import read,write
from . import assembly_mechanism as v8,assembly_realization as v7,assembly_population as population
from .world_scm import validate,forward

PROTOCOL=v8.PROTOCOL
INTERFACE='camera_registered_functional_spatial_program_v9'
MODEL_VARIANT='functional_spatial_program_scm_v9'
KIND='functional_spatial_program'
PROPOSAL_REVISION='v9.1_single_schema_lossless_binding'
MODES=v8.MODES
FUNCTIONS=v8.FUNCTIONS
REGIONS=('left','right','rear','unspecified')
profile=v8.profile
hypotheses_answer=v8.hypotheses_answer
recognition_prompt=v8.recognition_prompt
activation_answer=v8.activation_answer
raw_activation=v8.raw_activation
raw_direct=v8.raw_direct
direct_answer=v8.direct_answer
load_domain=v8.load_domain
semantic_diagnostics=v8.semantic_diagnostics
_positive=v8._positive


def hypotheses_prompt(case,grounding):
    return v8.hypotheses_prompt(case,grounding)+(
        '\nThe photograph is one ORIGINAL CAMERA sector of a 360-degree inhabited environment. '
        'Reason about plausible whole-room organizations, including functional counterparts that may lie opposite the camera. '
        'Do not equate the visible corner with the entire room; no additional zone, category or rear object is mandatory. '
        'Keep room extent and unseen walls uncertain. No resident biography.')


def furniture_prompt(case,hypothesis,channel='surrounding'):
    # A single complete wire schema replaces the inherited v2/v5/v6/v7/v9
    # append-only instructions. The real v9 run omitted the old group.reason
    # in every first reply; it is redundant with spatial_reason, not inventory.
    schema=dict(alternatives=[dict(description='concrete uncertain hidden-world hypothesis',groups=[dict(
        function='reading',realization_mode='new_group',layout_region='unspecified',
        spatial_reason='one concise functional and spatial rationale',observed_refs=[],
        new_objects=[dict(key='unique_key',category='bookcase',variant='freestanding wooden bookcase',
            reason='why this distinct hidden instance is plausible',placement='qualitative hidden position',
            attachment=dict(kind='floor',host=None),affordances=['shelf'])],functional_links=[])],relations=[])])
    observed={o['track_id']:o['center_local_m']['value'] for o in read(case['state'])['objects']} if case.get('state') else {}
    focus=('Focus on surrounding functional arrangements, including their ordinary belongings.' if channel=='surrounding' else
        'Focus on ordinary belongings and their supporting furniture. Include a predicted host in the same group when needed.')
    return ('Infer a CAMERA-REGISTERED WHOLE 360-DEGREE WORLD compatible with this partial photograph and fixed observed inventory. '
        'This is an already inhabited world, not a minimum furniture design, a requirement checklist or everything imaginable. '
        'Use only the attached observation and the fixed facts below; no previous chat memory or resident biography. '
        +focus+' Return 1..2 complete alternatives, each with at most 4 groups and 24 new instances; these are interface capacities, '
        'not quantity targets. An empty groups list is legal. Include furniture, independent belongings and supported contents jointly. '
        'The photographed anchors already exist: never regenerate them under new IDs, change them, or count their integral parts as new instances. '
        'Another same-category object requires a separate instance hypothesis; do not pad inventories with repeated small objects. '
        'Every group has function equal to ONE exact tag from '+json.dumps(FUNCTIONS)+'. '
        'realization_mode is observed_extension or new_group. observed_extension is unseen furniture/belongings serving visible anchors, '
        'not an additional independent activity zone; give nonempty exact observed_refs and an attachment, typed spatial relation OR '
        'functional link to one of those anchors. new_group is an additional arrangement. The two modes can coexist. '
        'Groups with the same function and mode are alternative realizations; put objects that must coexist in one group. '
        'All new hosts and descendants must stay in the SAME group. New-to-new spatial/service links cannot cross groups. '
        'category is a concrete noun; the executor separately maps it to the dataset vocabulary and retains outside-table names. '
        'Object keys are unique within an alternative. top/seat/shelf/hang attachments require an exact same-group host key or observed ID; '
        'floor/wall/ceiling/free use host=null. A bookcase standing near a wall uses floor, not wall. '
        'Side/coffee/dining tables, chairs, sofas, desks and floor lamps cannot be installed on a sofa seat, shelf or tabletop. '
        'affordances is a subset of top,seat,shelf,hanging. For observed attachments consult OBSERVED SUPPORT DOMAINS: '
        'an empty candidate list for that kind means its possible contact envelope is inside the input frustum; a nonempty list is only necessary, not a physical certificate. '
        'layout_region is left/right/rear/unspecified in ORIGINAL CAMERA coordinates. rear requires COMPLETE boxes at Z<0. '
        'left/right are hidden side domains, not a furniture-relative direction; unspecified allows any hidden direction or a group spanning directions. '
        'These are uncertain world hypotheses, not measured walls. No direction, category or additional zone is mandatory. Unknown does not mean empty. '
        'near is adjacency; left_of/right_of order camera X; above/below order camera Y; farther_than means larger camera Z and nearer_than smaller Z. '
        'A rear object cannot be farther_than a positive-Z observed anchor. Sofa-relative behind, facing a sofa, or a far wall does not establish camera-depth order. '
        'When order is unsupported, omit that typed relation rather than asserting contradictory depth. All spatial relations go in alternative.relations '
        'as {"source":"exact key or obs ID","target":"exact other key or obs ID","kind":"near","reason":"why"}. '
        'Support is only in attachment; do not add on/on_top_of as spatial relations. '
        'functional_links is an explicit list, possibly empty, of {"source":"new key in THIS group","target":"same-group key or exact obs ID",'
        '"kind":"serves","purpose":"concrete connection"}. It expresses service, not near, verified facing or physical support. '
        'A media counterpart can serve visible seating across the room without being beside it. Do not add near merely to express functional association. '
        'spatial_reason is the single group rationale; no second group.reason is required. '
        'No dimensions, coordinates, probabilities or explanation outside the JSON. The bookcase below is only a syntax example, not a required category. '
        '\nSINGLE RESPONSE SCHEMA: '+json.dumps(schema,ensure_ascii=False)+
        '\nSCENE HYPOTHESIS: '+json.dumps(hypothesis,ensure_ascii=False)+
        '\nOBSERVED SUPPORT DOMAINS: '+json.dumps(case.get('observed_support_domains',{}),ensure_ascii=False)+
        '\nFIXED OBSERVED ANCHORS: '+json.dumps(case['context'],ensure_ascii=False)+
        '\nFIXED_OBSERVED_CAMERA_CENTERS_XYZ_M: '+json.dumps(observed))


def _bind_wire(value):
    """Lossless wire aliases only; never infer mode/region or repair geometry."""
    value=deepcopy(value);alt=value['alternatives'][0];bindings=[];errors=[]
    if not isinstance(alt,dict) or not isinstance(alt.get('groups'),list):raise ValueError('groups must be a list')
    for i,g in enumerate(alt['groups']):
        path='groups['+str(i)+']'
        if not isinstance(g,dict):errors.append(path+' must be an object');continue
        if g.get('realization_mode') not in MODES:errors.append(path+'.realization_mode: Explicit realization_mode required')
        if g.get('layout_region') not in REGIONS:errors.append(path+'.layout_region: Explicit layout_region left/right/rear/unspecified required')
        if not isinstance(g.get('spatial_reason'),str) or not g['spatial_reason'].strip():errors.append(path+'.spatial_reason: Explicit spatial_reason required')
        if not isinstance(g.get('functional_links'),list):errors.append(path+'.functional_links must be an explicit list, possibly empty')
        if not isinstance(g.get('new_objects'),list):errors.append(path+'.new_objects must be a list')
        if not isinstance(g.get('reason'),str) or not g['reason'].strip():
            if isinstance(g.get('spatial_reason'),str) and g['spatial_reason'].strip():
                g['reason']=g['spatial_reason'];bindings.append(dict(path=path+'.reason',source=path+'.spatial_reason',action='copy_existing_rationale_to_legacy_internal_field'))
        if 'relations' in g:
            nested=g.pop('relations');top=alt.setdefault('relations',[])
            if not isinstance(nested,list) or not isinstance(top,list):errors.append(path+'.relations and alternative.relations must be lists')
            else:
                top.extend(nested)
                bindings.append(dict(path=path+'.relations',target='alternative.relations',relations_preserved=len(nested),action='relocate_declared_relations_without_changing_endpoints_or_kind'))
    if errors:raise ValueError('Required group fields: '+json.dumps(errors,ensure_ascii=False))
    return value,bindings


def _single(value,case,hypothesis):
    # Reuse all pre-v7 typed identity/attachment/spatial checks, then apply the
    # v9 extension-binding rule. No fabricated near edge is injected to pass v7.
    value,bindings=_bind_wire(value);raw=value['alternatives'][0];declared=raw['groups']
    program=v7.previous.furniture_answer(value,case,hypothesis)[0]
    obs={o['id'] for o in case['context']}
    for i,g in enumerate(declared):
        keys={o['key'] for row in program['groups'] if row['assembly_key']=='g'+str(i) for o in row['objects']}
        links=[]
        for link in g['functional_links']:
            if not isinstance(link,dict) or set(link)!={'source','target','kind','purpose'} or link.get('kind')!='serves':raise ValueError('Functional links require source,target,kind=serves,purpose')
            if link['source'] not in keys or link['target'] not in keys|obs or link['source']==link['target']:raise ValueError('Functional links must stay in their group or bind exact observed IDs')
            if not isinstance(link['purpose'],str) or not link['purpose'].strip():raise ValueError('Functional purpose must be explicit')
            links.append(deepcopy(link))
        anchors=set(g.get('observed_refs',[]))
        if g['realization_mode']=='observed_extension':
            if not anchors:raise ValueError('observed_extension requires exact observed_refs')
            attached=any(o['attachment']['host'] in anchors for row in program['groups'] if row['assembly_key']=='g'+str(i) for o in row['objects'])
            related=any((r['source'] in keys and r['target'] in anchors) or (r['target'] in keys and r['source'] in anchors) for r in program['relations'])
            served=any(l['target'] in anchors for l in links)
            if not (attached or related or served):raise ValueError('Extension must attach, spatially relate or explicitly serve declared observed anchors')
        for row in program['groups']:
            if row['assembly_key']=='g'+str(i):row.update(realization_mode=g['realization_mode'],layout_region=g['layout_region'],
                spatial_reason=g['spatial_reason'].strip(),functional_links=deepcopy(links))
    check_region_order(program,case)
    if bindings:program['interface_normalizations']=bindings
    return program


def check_region_order(program,case):
    """Necessary center-only LP; reject contradictions, never certify full fit."""
    objects=[o for g in program['groups'] for o in g['objects']];ids={o['key']:i for i,o in enumerate(objects)};n=len(ids)
    if not n:return
    observed={o['track_id']:o['center_local_m']['value'] for o in read(case['state'])['objects']} if case.get('state') else {}
    if any(r[k].startswith('obs_') for r in program['relations'] for k in ('source','target')) and not observed:
        raise ValueError('Frozen observed coordinates required to validate explicit camera-region hypotheses')
    rows=[];rhs=[];tx=math.tan(math.radians(case['hfov'])/2)
    for group in program['groups']:
        region=group['layout_region']
        if region=='unspecified':continue
        for o in group['objects']:
            i=ids[o['key']]
            if region=='rear':
                row=np.zeros(2*n);row[2*i+1]=1;rows.append(row);rhs.append(-1e-5)
            else:
                sign=1 if region=='right' else -1
                row=np.zeros(2*n);row[2*i]=-sign;rows.append(row);rhs.append(-1e-5)
                row=np.zeros(2*n);row[2*i]=-sign;row[2*i+1]=tx;rows.append(row);rhs.append(-1e-5)
    for r in program['relations']:
        k=r['kind']
        if k not in ('left_of','right_of','in_front_of','behind'):continue
        dim=0 if k in ('left_of','right_of') else 2;axis=0 if dim==0 else 1;sign=1 if k in ('right_of','in_front_of') else -1
        row=np.zeros(2*n);constant=0.
        for field,mult in [('source',sign),('target',-sign)]:
            oid=r[field]
            if oid in ids:row[2*ids[oid]+axis]+=mult
            else:constant+=mult*observed[oid][dim]
        rows.append(-row);rhs.append(constant-1e-5)
    result=linprog(np.zeros(2*n),A_ub=np.array(rows) if rows else None,b_ub=np.array(rhs) if rows else None,bounds=[(None,None)]*(2*n),method='highs')
    if result.status==2:
        orders=[dict(source=r['source'],target=r['target'],kind={'in_front_of':'farther_than','behind':'nearer_than'}.get(r['kind'],r['kind']))
            for r in program['relations'] if r['kind'] in ('left_of','right_of','in_front_of','behind')]
        details=dict(region_by_new_key={o['key']:g['layout_region'] for g in program['groups'] for o in g['objects']},
            declared_XZ_orders=orders,fixed_observed_centers={k:observed[k] for r in orders for k in (r['source'],r['target']) if k in observed})
        raise ValueError('Camera region and typed XZ ordering are inconsistent; revise the COMPLETE hypothesis, not the observed coordinates. '
            'rear requires Z<0; farther_than requires larger camera Z. Check these exact declared constraints: '+json.dumps(details,ensure_ascii=False))
    if not result.success:raise ValueError('Region-order feasibility could not be verified: '+result.message)


def furniture_answer(value,case,hypothesis):
    rows=value.get('alternatives') if isinstance(value,dict) else None
    if not isinstance(rows,list) or not 1<=len(rows)<=2:raise ValueError('alternatives must contain 1..2 complete hypotheses')
    accepted=[];rejected=[]
    for i,alt in enumerate(rows):
        try:program=_single(dict(alternatives=[deepcopy(alt)]),case,hypothesis)
        except (ValueError,KeyError,TypeError,AttributeError) as exc:rejected.append(dict(alternative=i,error=str(exc)));continue
        program['proposal_alternative_index']=i;accepted.append(program)
    if not accepted:raise ValueError('No legal complete spatial alternative: '+json.dumps(rejected,ensure_ascii=False))
    audit=dict(policy='complete_functional_spatial_alternatives',accepted_indices=[p['proposal_alternative_index'] for p in accepted],rejected=rejected,objects_salvaged_from_invalid_alternatives=False)
    for program in accepted:program['proposal_validation']=deepcopy(audit)
    return accepted


def build_model(hypotheses,programs,unused_contents=None):
    return population.build_model(hypotheses,programs,unused_contents,signature_fields=('realization_mode','layout_region','functional_links'))


def _spatial_menu(domain,order):
    return [dict(id=c,possible_regions=sorted({g['layout_region'] for s in domain['slots'][c] if s!='none' for g in domain['programs'][s]['groups']})) for c in order]


def reference_prompt(domain,order):return v8.reference_prompt(domain,order)+'\nSPATIAL_SUPPORT: '+json.dumps(_spatial_menu(domain,order))
def activation_prompt(domain,h,order):return v8.activation_prompt(domain,h,order)+'\nSPATIAL_SUPPORT: '+json.dumps(_spatial_menu(domain,order))


def variant_prompt(domain,h,role,order):
    return v8.variant_prompt(domain,h,role,order)+('\nV9: Options are functional-SPATIAL realizations. Region is part of the selected state, not a later guess. '
        'Service links express function without requiring metric adjacency. No angular coverage target. '
        '\nSPATIAL_VARIANTS: '+json.dumps([dict(id=s,groups=[dict(layout_region=g['layout_region'],spatial_reason=g['spatial_reason'],functional_links=g['functional_links']) for g in domain['programs'][s]['groups']]) for s in order],ensure_ascii=False))


def direct_prompt(domain,order):
    return v8.direct_prompt(domain,order)+'\nSPATIAL_VARIANTS: '+json.dumps([dict(id=s,groups=[dict(layout_region=g['layout_region'],spatial_reason=g['spatial_reason'],functional_links=g['functional_links']) for g in domain['programs'][s]['groups']]) for c in order for s in domain['slots'][c] if s!='none'],ensure_ascii=False)


def prepare_model(base,case,config,out):
    m=v8.prepare_model(base,case,dict(config,mechanism_table_kind=KIND),None)
    for channel,states in m['slots'].items():
        rows={}
        for s in states:
            regions={g['layout_region'] for g in m['furniture_states'][s]['groups']} if s!='none' else {'unspecified'}
            if len(regions)!=1:raise ValueError('One channel variant must have one camera-region program')
            r=next(iter(regions));rows[s]=[float(v==r) for v in REGIONS]
        m['nodes'].append(dict(id='R_'+channel,parents=['M_'+channel],states=list(REGIONS),cpt=rows))
    m.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        generation_factorization=m['generation_factorization']+'; V includes camera-region realization; V->R->X; service links do not imply near',
        spatial_program_scope='Sampled camera-relative hypotheses, not recovered room walls. No rear quota or uniform angular target.')
    validate(m)
    if out is not None:
        write(out/'empirical_mechanisms.json',m['empirical_tables']);write(out/'probability_audit.json',marginals(m))
        write(out/'functional_probability_interventions.json',{h['id']:marginals(m,h['id']) for h in m['profiles']})
        write(out/'mechanism_diagnostics.json',dict(training_performed=False,mechanism_uncertainty=v8.uncertainty(m),spatial_program_scope=m['spatial_program_scope'],causal_gain_validated=False))
        if m['mechanism_policy']=='functional':write(out/'uniform_reference_ablation.json',dict(reference_shrunk=marginals(m),uniform_base=v8.uniform_reference_ablation(m),scope='Inventory probability-only base ablation'))
        (out/'probability_audit.html').write_text('<!doctype html><meta charset="utf-8"><h1>v9 功能—空间实现概率</h1><p>空间状态进入实际采样。无后方配额；未提出后方方案不等于后方为空。</p><pre>'+escape(json.dumps(marginals(m),ensure_ascii=False,indent=2))+'</pre>')
    return m


def marginals(m,do_profile=None):
    law=v8.marginals(m,do_profile);per_channel={c:dict.fromkeys(REGIONS,0.) for c in m['slots']};presence=dict.fromkeys(REGIONS,0.)
    region={s:m['furniture_states'][s]['groups'][0]['layout_region'] for s in m['state_slot']}
    for s,p in law['assembly_probabilities'].items():per_channel[m['state_slot'][s]][region[s]]+=p
    for h,qh in law['profile_probabilities'].items():
        if not qh:continue
        for j,pj in m['activation_rows'][h].items():
            for r in REGIONS:
                absent=1.
                for i,c in enumerate(m['slots']):
                    p=m['joint_components'][j][i]*sum(v for s,v in v8._variants(m,h,j,c).items() if region[s]==r)
                    absent*=1-p
                presence[r]+=qh*pj*(1-absent)
    law.update(channel_region_joint_presence=per_channel,any_declared_region_presence=presence,
        region_scope='Active assembly region events, not measured physical angular coverage; left/right domains can extend into rear half-space')
    return law


def active_world(m,state):
    world=v8.active_world(m,state);links=[];unresolved=[];program={};owned={o['id'] for o in world['objects']}
    for channel in m['slots']:
        s=state['M_'+channel]
        if s=='none':continue
        r=state['R_'+channel];program[channel]=dict(module=s,region=r,region_node='R_'+channel)
        objs=[o for o in m['component_registry'].values() if o['furniture_state']==s];mapping={o['key']:o['id'] for o in objs};seen=set()
        for group in world['groups']:
            if not group['key'].startswith(s+'_'):continue
            group['layout_region']=r
            for link in group.get('functional_links',[]):
                row=dict(link,source=mapping.get(link['source'],link['source']),target=mapping.get(link['target'],link['target']))
                signature=(row['source'],row['target'],row['kind'])
                if signature in seen:continue
                seen.add(signature)
                if row['source'] not in owned or (link['target'] in mapping and row['target'] not in owned):
                    unresolved.append(dict(row,reason='Functional endpoint absent after inventory intervention'))
                else:links.append(row)
    world.update(functional_links=links,unresolved_functional_links=unresolved,spatial_program=program)
    return world


def sample_worlds(m,seed=3407,count=4):
    if type(count) is not int or count<1:raise ValueError('Positive world count required')
    rng=np.random.default_rng(seed);law=marginals(m);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in m['nodes']};world=active_world(m,forward(m,noise))
        for o in world['objects']:o['presence_probability']=law['object_presence'][o['id']]
        world.update(noise=noise,seed=seed,trial=trial,mechanism_source=m['mechanism_source'],probability_scope=m['distribution_scope']);worlds.append(world)
    return worlds


def intervention_diagnostics(m,world):
    result=v8.intervention_diagnostics(m,world);rows=[]
    for c in world['spatial_program']:
        for r in ('left','right','rear'):
            after=active_world(m,forward(m,world['noise'],{'R_'+c:r}))
            rows.append(dict(do={'R_'+c:r},inventory_preserved=[o['id'] for o in after['objects']]==[o['id'] for o in world['objects']],
                spatial_program=after['spatial_program'],geometry_not_resolved=True,same_exogenous_noise=True))
    result['spatial_program_queries']=rows
    return result
