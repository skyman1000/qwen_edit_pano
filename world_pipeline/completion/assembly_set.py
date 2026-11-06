"""Observation-conditioned joint assembly SCM, without one-per-function slots.

H and the fixed observation condition ONE joint set of self-contained candidate
assemblies. Function tags annotate assemblies; they are not extra-zone gates.
No learned weights, rear quotas, mandatory inventory, or post-sample deletion.
"""
from copy import deepcopy
from html import escape
import json

from ..common import read,write
from . import assembly_panorama as previous,assembly_mechanism as mechanisms
from . import assembly_population as population
from .world_scm import validate

PROTOCOL=previous.PROTOCOL
INTERFACE='observation_conditioned_joint_assembly_set_v10'
MODEL_VARIANT='context_joint_assembly_scm_v10'
KIND='context_joint_assembly'
PROPOSAL_REVISION='independent_dependency_closed_candidate_units_v10'
FUNCTIONS=previous.FUNCTIONS
REGIONS=previous.REGIONS
profile=previous.profile
hypotheses_prompt=previous.hypotheses_prompt
hypotheses_answer=previous.hypotheses_answer
recognition_prompt=previous.recognition_prompt
raw_activation=previous.raw_activation
raw_direct=previous.raw_direct
load_domain=previous.load_domain
semantic_diagnostics=previous.semantic_diagnostics
_positive=previous._positive
marginals=previous.marginals
active_world=previous.active_world
sample_worlds=previous.sample_worlds
intervention_diagnostics=previous.intervention_diagnostics


def furniture_prompt(case,hypothesis,channel='surrounding'):
    schema=dict(alternatives=[dict(description='uncertain candidate collection',groups=[dict(
        function='reading',spatial_reason='why this optional assembly is plausible',observed_refs=[],layout_region='unspecified',
        new_objects=[dict(key='unique_key',category='bookcase',variant='freestanding wooden bookcase',
            reason='reason for a distinct hidden instance',placement='qualitative hidden placement',
            attachment=dict(kind='floor',host=None),affordances=['shelf'])],functional_links=[])],relations=[])])
    centers={o['track_id']:o['center_local_m']['value'] for o in read(case['state'])['objects']} if case.get('state') else {}
    focus=('Consider surrounding furniture together with its ordinary belongings.' if channel=='surrounding' else
        'Consider ordinary possessions, surface contents and their hosts, including independent small assemblies.')
    return ('Infer candidate ASSEMBLIES for an inhabited 360-degree world compatible with this partial photograph. '
        'These are optional building blocks for a later JOINT world sampler, not a minimal furnishing plan and not all mandatory. '+focus+
        ' Return 1..2 candidate collections in the single schema below. Use concise records within the token budget. '
        'A group can be one independent object, one item on an observed host, or furniture with its dependent contents. '
        'Include ordinary contents when plausible; a room is more than large functional furniture. No count or angular coverage goal. '
        'Several groups serving the SAME function or visible anchor can coexist: a media console and coffee table are not mutually exclusive, '
        'and unseen tables at two different sofas are not alternative readings of ONE required table. '
        'Do not gate a coffee table or personal belongings on the existence of another independent social area. '
        'Keep visible anchors immutable. Never regenerate visible instances or their integral parts with new IDs. '
        'New instances of the same category need a distinct hidden-instance hypothesis. Do not pad with duplicate remotes/books. '
        'function is one primary tag from '+json.dumps(FUNCTIONS)+'. It is an annotation, not a requirement or exclusive slot. '
        'No realization_mode field or mandatory anchor service proof. observed_refs may be empty; if given, use exact observed IDs. '
        'Put all interdependent NEW hosts, children and typed new-to-new relations in one self-contained group. '
        'Groups should be independently usable candidate units, not arbitrary pieces of one dependent arrangement. '
        'Spatial relations go in alternative.relations, with source,target,kind,reason; use exact object keys or observed IDs. '
        'near means actual proximity, not functional association. left_of/right_of order original camera X; above/below order Y. '
        'farther_than means GREATER original camera Z and nearer_than SMALLER Z. '
        'Facing a sofa does not establish camera-depth order. Omit unsupported depth ordering. '
        'Optional layout_region is left/right/rear/unspecified; omission means unspecified, not an inferred wall. '
        'rear means full boxes at camera Z<0, and cannot coexist with farther_than a positive-Z anchor. '
        'Optional functional_links use source,target,kind="serves",purpose. Source is a new key; target is same-unit key or observed ID. '
        'Service is not near, verified facing or support. Empty/omitted service links do not invalidate ordinary possessions. '
        'attachment.kind is floor/top/seat/shelf/hang/wall/ceiling/free. top/seat/shelf/hang require an exact host key in the unit or observed ID; '
        'other kinds require host=null. Affordances are top/seat/shelf/hanging. A freestanding bookcase uses floor, not wall mounting. '
        'Tables, chairs, sofas, desks and floor lamps cannot be installed on a sofa seat or tabletop. '
        'Physical support belongs only in attachment, not invented on/support relation types. '
        'An empty observed support domain cannot support a wholly outside-frustum addition. Nonempty is not a physical certificate. '
        'Use concrete category nouns, retain outside-vocabulary names, and do not invent residents or life histories. '
        'The schema bookcase is syntax only, not required. No metric coordinates, probabilities or prose outside JSON. '
        '\nSINGLE RESPONSE SCHEMA: '+json.dumps(schema,ensure_ascii=False)+
        '\nSCENE HYPOTHESIS: '+json.dumps(hypothesis,ensure_ascii=False)+
        '\nOBSERVED SUPPORT DOMAINS: '+json.dumps(case.get('observed_support_domains',{}),ensure_ascii=False)+
        '\nFIXED OBSERVED ANCHORS: '+json.dumps(case['context'],ensure_ascii=False)+
        '\nFIXED_OBSERVED_CAMERA_CENTERS_XYZ_M: '+json.dumps(centers))


def _units(alt):
    """Typed dependencies define atomic units; a bad unit does not erase others."""
    if not isinstance(alt,dict) or not isinstance(alt.get('groups'),list):raise ValueError('groups must be a list')
    groups=deepcopy(alt['groups']);parent=list(range(len(groups)));owners={};relations=list(alt.get('relations',[]))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(indices):
        for i in indices[1:]:parent[find(i)]=find(indices[0])
    for i,g in enumerate(groups):
        if not isinstance(g,dict) or not isinstance(g.get('new_objects'),list):raise ValueError('Each group needs new_objects list')
        for o in g['new_objects']:
            if not isinstance(o,dict) or not isinstance(o.get('key'),str):raise ValueError('Each object needs an exact key')
            owners.setdefault(o['key'],[]).append(i)
        if 'relations' in g:relations.extend(g.pop('relations'))
    for values in owners.values():join(values)
    for i,g in enumerate(groups):
        for o in g['new_objects']:
            host=(o.get('attachment') or {}).get('host')
            if host in owners:join([i]+owners[host])
        for l in g.get('functional_links',[]):
            for endpoint in ('source','target'):
                if l.get(endpoint) in owners:join([i]+owners[l[endpoint]])
    for r in relations:
        indices=sorted({i for k in ('source','target') for i in owners.get(r.get(k),[])})
        if not indices:raise ValueError('A declared relation must involve a new candidate key; cannot assign an unknown relation to an independent unit')
        join(indices)
    components={}
    for i in range(len(groups)):components.setdefault(find(i),[]).append(i)
    for indices in components.values():
        keys={o['key'] for i in indices for o in groups[i]['new_objects']}
        yield indices,[groups[i] for i in indices],[r for r in relations if r.get('source') in keys or r.get('target') in keys]


def _unit(groups,relations,description,case,hypothesis):
    functions=sorted({g.get('function') for g in groups})
    if any(f not in FUNCTIONS for f in functions):raise ValueError('Each group needs an exact primary function tag')
    regions={g.get('layout_region','unspecified') for g in groups}
    if len(regions)!=1:raise ValueError('A dependent unit spanning camera regions must declare a coherent group-level region or unspecified; no declared region is silently overwritten')
    g=deepcopy(groups[0])
    if len(groups)>1:
        # Grouping is transport, not a reason to reject valid host descendants.
        # Every object/link/order is retained in the whole dependency component.
        g.update(new_objects=[deepcopy(o) for row in groups for o in row['new_objects']],
            observed_refs=sorted({v for row in groups for v in row.get('observed_refs',[])}),
            functional_links=[deepcopy(v) for row in groups for v in row.get('functional_links',[])],
            spatial_reason='; '.join(row.get('spatial_reason') or row.get('reason') or description for row in groups))
    g.setdefault('layout_region','unspecified');g.setdefault('functional_links',[])
    g.setdefault('spatial_reason',g.get('reason') or description)
    g['realization_mode']='new_group'  # Legacy parser transport only; removed from the mechanism below.
    # Pre-v7 typed validation keeps identity, contact feasibility, mounting and
    # same-unit dependencies; no observed-extension gate is applied.
    value,_=previous._bind_wire(dict(alternatives=[dict(description=description,groups=[g],relations=relations)]))
    program=previous.v7.previous.furniture_answer(value,case,hypothesis)[0]
    keys={o['key'] for row in program['groups'] for o in row['objects']};obs={o['id'] for o in case['context']}
    for link in g['functional_links']:
        if not isinstance(link,dict) or set(link)!={'source','target','kind','purpose'} or link['kind']!='serves':raise ValueError('Service links require source,target,kind=serves,purpose')
        if link['source'] not in keys or link['target'] not in keys|obs or link['source']==link['target']:raise ValueError('Invalid exact functional endpoints')
        if not isinstance(link['purpose'],str) or not link['purpose'].strip():raise ValueError('Functional purpose must be explicit')
    for row in program['groups']:
        row.update(layout_region=g['layout_region'],spatial_reason=g['spatial_reason'],functional_links=deepcopy(g['functional_links']),realization_mode='candidate_assembly',function_tags=functions)
    previous.check_region_order(program,case)
    return program


def furniture_answer(value,case,hypothesis):
    rows=value.get('alternatives') if isinstance(value,dict) else None
    if not isinstance(rows,list) or not 1<=len(rows)<=2:raise ValueError('alternatives must contain 1..2 candidate collections')
    accepted=[];rejected=[];empty=[]
    for ai,alt in enumerate(rows):
        try:units=list(_units(alt))
        except (ValueError,KeyError,TypeError,AttributeError) as exc:rejected.append(dict(alternative=ai,error=str(exc)));continue
        if not units:empty.append(ai)
        for indices,groups,relations in units:
            if not any(g['new_objects'] for g in groups):empty.append(ai);continue
            try:program=_unit(groups,relations,alt.get('description') or 'Uncertain candidate assembly',case,hypothesis)
            except (ValueError,KeyError,TypeError,AttributeError) as exc:rejected.append(dict(alternative=ai,groups=indices,error=str(exc)));continue
            program.update(proposal_alternative_index=ai,proposal_component_indices=indices);accepted.append(program)
    audit=dict(policy='dependency_component_atomic_candidate_validation',accepted_indices=[p['proposal_alternative_index'] for p in accepted],
        accepted_units=[dict(alternative=p['proposal_alternative_index'],groups=p['proposal_component_indices']) for p in accepted],
        rejected=rejected,explicit_empty_collections=empty,objects_salvaged_from_invalid_components=False,
        scope='Independent candidate units, not accepted complete worlds. All sampled inventory remains atomic and is never pruned.')
    if not accepted:
        if rejected:raise ValueError('No legal candidate unit: '+json.dumps(audit,ensure_ascii=False))
        accepted=[dict(description='Explicit empty candidate collection',hypothesis=deepcopy(hypothesis),groups=[],relations=[],relation_bindings=[],proposal_alternative_index=empty[0] if empty else 0)]
    for p in accepted:p['proposal_validation']=deepcopy(audit)
    return accepted


def build_model(hypotheses,programs,unused_contents=None):
    return population.build_model(hypotheses,programs,unused_contents,signature_fields=('layout_region','functional_links','function_tags'))


def _menu(domain,order):
    result=[]
    for channel in order:
        state=next(s for s in domain['slots'][channel] if s!='none');p=domain['programs'][state]
        result.append(dict(id=state,functions=sorted({f for g in p['groups'] for f in g.get('function_tags',[g['function']])}),
            groups=[dict(observed_refs=g.get('observed_refs',[]),layout_region=g['layout_region'],functional_links=g.get('functional_links',[]),
                objects=[dict(key=o['key'],category=o['category'],variant=o['variant'],attachment=o['attachment']) for o in g['objects']]) for g in p['groups']],relations=p['relations']))
    return result


def _joint_prompt(domain,order):
    return ('Sample ONE JOINT SET of existing hidden candidate assemblies. Any subset, including empty, is legal. '
        'Function tags are annotations, not exclusive slots or additional-activity prerequisites. '
        'Several assemblies of the same function and assemblies at different observed anchors may coexist. '
        'Account jointly for functional use, ordinary possessions, support, observed anchors and redundancy. '
        'Do not select everything conceivable, and do not minimize furnishing. Lack of direct visibility is expected, not evidence of absence. '
        'Avoid interpreting alternative descriptions of the same intended object as two required copies. '
        'Use no count, room-area or rear-coverage target. Return exactly {"active_assemblies":[zero or more exact assembly IDs]}. '
        '\nCANDIDATE_ASSEMBLIES: '+json.dumps(_menu(domain,order),ensure_ascii=False)+
        '\nFIXED_COMMON_OBSERVED_INVENTORY: '+json.dumps(domain['common_inventory'],ensure_ascii=False))


def reference_prompt(domain,order):return _joint_prompt(domain,order)+'\nREFERENCE: D-only; no particular H or photograph is supplied.'
def activation_prompt(domain,h,order):return _joint_prompt(domain,order)+'\nCURRENT_PARENT_H: '+json.dumps(profile(h['id'],h['functions']))+'\nUse the attached partial observation when supplied. H is uncertain context, not an exhaustive list of hidden uses.'
def direct_prompt(domain,order):return _joint_prompt(domain,order)+'\nUse the attached partial observation. No latent H is supplied.'
variant_prompt=previous.variant_prompt


def activation_answer(value,channels):
    if not isinstance(value,dict) or set(value)!={'active_assemblies'} or not isinstance(value['active_assemblies'],list):raise ValueError('Return exactly active_assemblies: list of assembly IDs')
    allowed={c.removeprefix('unit_') for c in channels};active=value['active_assemblies']
    if any(not isinstance(s,str) or s not in allowed for s in active) or len(set(active))!=len(active):raise ValueError('Use distinct exact assembly IDs from '+str(sorted(allowed)))
    return ''.join('1' if c.removeprefix('unit_') in active else '0' for c in channels)


def direct_answer(value,domain):
    mask=activation_answer(value,list(domain['slots']))
    return json.dumps([next(s for s in states if s!='none') if bit=='1' else 'none' for bit,states in zip(mask,domain['slots'].values())],separators=(',',':'))


def prepare_model(base,case,config,out):
    config=deepcopy(config)
    if config['mechanism_source']=='per_backend_tag_only':config['mechanism_source']=config['execution_backend']+'_tag_only'
    m=mechanisms.prepare_model(base,case,dict(config,mechanism_table_kind=KIND),None)
    for channel,states in m['slots'].items():
        if len(states)!=2:raise ValueError('A candidate unit has one immutable assembly and none, not a function slot')
        rows={}
        for s in states:
            regions={g['layout_region'] for g in m['furniture_states'][s]['groups']} if s!='none' else {'unspecified'}
            if len(regions)!=1:raise ValueError('One candidate unit must declare a coherent camera-region hypothesis')
            r=next(iter(regions));rows[s]=[float(v==r) for v in REGIONS]
        m['nodes'].append(dict(id='R_'+channel,parents=['M_'+channel],states=list(REGIONS),cpt=rows))
    tag_only=config.get('assembly_rgb_conditioned',True) is False
    if m['empirical_tables'].get('conditional_RGB')!=(not tag_only):raise ValueError('Joint-selection RGB protocol/table mismatch')
    parents=(['H','D_text'] if tag_only else ['O_RGB','H','D_text']) if m['mechanism_policy']=='functional' else ['O_RGB','D_text']
    m.update(model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        generation_factorization=('q(H|O,D) p(B_joint|H,'+('D_text' if tag_only else 'O,D')+'); B->E,S,R; (O,H,B,E,S,R)->X' if m['mechanism_policy']=='functional' else m['generation_factorization'])+'; one optional candidate unit per B bit; no function-slot exclusivity',
        fixed_context_parents=dict(joint_selection=parents),
        evidence_delivery=('RGB at proposal, recognition, layout; joint selection H,D_text only' if tag_only else
            'RGB at proposal, recognition, joint selection and layout; inventory H edges removed' if m['mechanism_policy']=='function_free' else
            'RGB at direct joint selection and layout; no H in inventory; H metric context retained' if m['mechanism_policy']=='direct' else
            'RGB at proposal, recognition, joint selection and layout; (O_RGB,H,D_text)->B_joint'),
        assembly_rgb_conditioned=not tag_only,assembly_selection_scope='Joint set of dependency-closed units; function tags are not exclusive existence gates',
        spatial_program_scope='Explicit camera hypotheses remain protected; unspecified is unknown, not empty')
    validate(m)
    if out is not None:
        write(out/'empirical_mechanisms.json',m['empirical_tables']);write(out/'probability_audit.json',marginals(m))
        write(out/'functional_probability_interventions.json',{h['id']:marginals(m,h['id']) for h in m['profiles']})
        write(out/'mechanism_diagnostics.json',dict(training_performed=False,probabilities_calibrated=False,causal_gain_validated=False,
            fixed_context_parents=m['fixed_context_parents'],assembly_selection_scope=m['assembly_selection_scope'],mechanism_uncertainty=mechanisms.uncertainty(m)))
        if m['mechanism_policy']=='functional':write(out/'uniform_reference_ablation.json',dict(reference_shrunk=marginals(m),uniform_base=mechanisms.uniform_reference_ablation(m),scope='Probability-only reference-base ablation, not a quality test'))
        (out/'probability_audit.html').write_text('<!doctype html><meta charset="utf-8"><h1>联合功能装配集合概率</h1><p>图像条件与用途假设进入实际采样；无每功能一个装配、数量或后方配额。</p><pre>'+escape(json.dumps(marginals(m),ensure_ascii=False,indent=2))+'</pre>')
    return m
