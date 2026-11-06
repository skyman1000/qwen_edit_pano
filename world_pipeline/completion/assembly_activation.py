"""Joint role activation -> conditional assembly variants -> instance inventory.

This is an assumed conditional SCM, with empirical LM generation mechanisms.
It is not an identified image-formation SCM or a calibrated visual posterior.
"""
from collections import Counter
from copy import deepcopy
from html import escape
import itertools
import json
from pathlib import Path

import numpy as np

from ..common import read, write, sha
from . import assembly_modular as old, assembly_sdscene as sd
from .world_scm import validate, forward, key

PROTOCOL = old.PROTOCOL
INTERFACE = 'joint_activation_conditional_variants_v6'
MODEL_VARIANT = 'joint_activation_scene_scm_v6'
FUNCTIONS = old.FUNCTIONS
build_model = old.build_model
load_domain = old.load_domain
semantic_diagnostics = old.semantic_diagnostics
hypotheses_answer = old.hypotheses_answer
active_world = old.active_world


def profile(identifier, functions):
    value = old.profile(identifier, functions)
    value.update(description='Possible spatial organization for: '+(', '.join(value['functions']) or 'unspecified use'),
        assumptions=['Uncertain organization/use program; not biography, observed activity, or measured affordance.'])
    return value


def explicit_depth(text):
    return text.replace('in_front_of', 'farther_than').replace('behind', 'nearer_than')


def hypotheses_prompt(case, grounding):
    return old.hypotheses_prompt(case, grounding).replace('functional affordances', 'possible organization/use programs')


def furniture_prompt(case, hypothesis, channel='surrounding'):
    from .assembly_taxonomy import vocabulary
    # One vocabulary throughout the request, validator errors and repair.
    prompt = old.furniture_prompt(case, hypothesis, channel).replace(
        'rather than ambiguous in_front_of/behind.', 'Use only these two names for depth ordering.')
    prompt = explicit_depth(prompt).replace('Functional affordances:', 'Possible organization/use program:')
    return prompt + ('\nV6: H denotes a possible organization/use program. Actual usability is checked downstream. '
        'Keep concrete category nouns; the executor also assigns a dataset label. The dataset label table is '+
        json.dumps(vocabulary()['categories'], ensure_ascii=False)+'. '
        'Never rename an unknown object to a false known class. Outside-table names remain explicit and colored separately. '
        'farther_than means larger original camera Z; nearer_than means smaller Z. '
        'Do not describe an unseen object as centred inside the photographed sector. '
        'Provide genuinely different inventories, including smaller assemblies and versions without optional belongings when plausible; '
        'no minimum or maximum target object count. Objects within one assembly coexist; alternatives do not.')


def furniture_answer(value, case, hypothesis):
    try:
        return old.furniture_answer(value, case, hypothesis)
    except ValueError as exc:
        raise ValueError(explicit_depth(str(exc))) from exc


def activation_states(roles):
    return [''.join(bits) for bits in itertools.product('01', repeat=len(roles))]


def activation_answer(value, roles):
    if not isinstance(value, dict) or set(value) != {'active_roles'}:
        raise ValueError('Return exactly {"active_roles": [zero or more exact role names]}')
    active = value['active_roles']
    if not isinstance(active, list) or any(not isinstance(r, str) or r not in roles for r in active) or len(set(active)) != len(active):
        raise ValueError('active_roles must be a duplicate-free subset of '+str(roles))
    return ''.join('1' if r in active else '0' for r in roles)


def activation_prompt(domain, h, roles):
    return ('Sample ONE joint binary configuration of ADDITIONAL UNSEEN assemblies of the available roles. '
        'A role means one additional assembly, not whether the visible room supports that activity at all. '
        'Decide all roles TOGETHER: they may coexist, share a limited surrounding space, or be absent together. '
        'Do not independently assume that every plausible role is present. No room boundary or area has been measured. '
        'Absence of a role in H is not proof of its absence; inclusion in H does not require extra hidden equipment. '
        'Existing visible equipment and ordinary possessions are common context. No minimum furnishings, no design brief, '
        'no biography. This is a possible existing world, not everything that could fit. '
        'An empty list is one allowed world; any subset, including all roles, is allowed. '
        'Return only {"active_roles":[exact role names]}; no objects, scores or explanation. '
        '\nCURRENT_PARENT_H: '+json.dumps(profile(h['id'], h['functions']))+
        '\nAVAILABLE_ROLES: '+json.dumps(roles)+
        '\nFIXED_COMMON_OBSERVED_INVENTORY: '+json.dumps(domain['common_inventory'], ensure_ascii=False))


def variant_prompt(domain, h, role, order):
    # Activation is already 1. The variant menu must not decide existence.
    menu = []
    for fid in order:
        p = domain['programs'][fid]
        menu.append(dict(id=fid, objects=[dict(key=o['key'], category=o['category'], variant=o['variant'],
            attachment=o['attachment']) for g in p['groups'] for o in g['objects']], relations=p['relations']))
    return ('Conditioned on this role having ONE additional unseen assembly, sample its concrete variant. '
        'Existence was decided by another mechanism; do not decide it again. Choose one exact allowed ID. '
        'Variants include their declared hosts and belongings. H is uncertain use organization, not an actual activity. '
        'Return exactly {"choice":"allowed ID"}; no explanation or scores. '
        '\nCURRENT_PARENT_H: '+json.dumps(profile(h['id'], h['functions']))+
        '\nCURRENT_PARENT_ROLE_ACTIVATION: '+json.dumps({role: 1})+
        '\nFIXED_COMMON_OBSERVED_INVENTORY: '+json.dumps(domain['common_inventory'], ensure_ascii=False)+
        '\nALLOWED_VARIANTS: '+json.dumps(menu, ensure_ascii=False))


def recognition_prompt(domain, case):
    return sd.recognition_prompt(dict(domain, profiles=[profile(h['id'], h['functions']) for h in domain['profiles']]), case)


def prepare_model(base, case, config, out):
    bundle = Path(config['sdscene_bundle']); domain = read(bundle/'domain.json')
    source = config['mechanism_source']
    if source == 'per_backend': source = config['execution_backend']
    table = read(bundle/'mechanisms'/(source+'.json'))
    if table['domain_sha256'] != sha(bundle/'domain.json') or table['kind'] != 'joint_activation':
        raise ValueError('Frozen v6 mechanism/domain mismatch')
    model = deepcopy(base); profiles = domain['profiles']; hs = [h['id'] for h in profiles]
    roles = list(domain['slots']); patterns = activation_states(roles); slots = domain['slots']
    hp = table['recognition']['probabilities']
    if set(hp) != set(hs): raise ValueError('Recognition profile domain mismatch')
    ar = {h:table['activation'][h]['probabilities'] for h in hs}
    vr = {r:{h:table['variants'][h+'|'+r]['probabilities'] for h in hs} for r in roles}
    if any(set(row) != set(patterns) for row in ar.values()): raise ValueError('Joint activation states differ')
    if any(set(vr[r][h]) != set(slots[r])-{'none'} for r in roles for h in hs): raise ValueError('Variant states differ')
    policy = config.get('activation_policy', 'functional')
    if policy not in ('functional', 'function_free'): raise ValueError('Unknown activation policy')
    if policy == 'function_free':
        # Keep the whole activation-pattern law AND every module marginal.
        marginal_a = {a:sum(hp[h]*ar[h][a] for h in hs) for a in patterns}
        for i, r in enumerate(roles):
            weights = {h:hp[h]*sum(p for a,p in ar[h].items() if a[i]=='1') for h in hs}
            total = sum(weights.values())
            if total <= 0: raise ValueError('No positive activation support for '+r)
            row = {v:sum(weights[h]*vr[r][h][v] for h in hs)/total for v in slots[r] if v!='none'}
            vr[r] = {h:deepcopy(row) for h in hs}
        ar = {h:deepcopy(marginal_a) for h in hs}
    nodes = [dict(id='H', parents=[], states=hs, cpt={'root':[hp[h] for h in hs]})]
    nodes.append(dict(id='A', parents=[] if policy=='function_free' else ['H'], states=patterns,
        cpt={'root':[ar[hs[0]][a] for a in patterns]} if policy=='function_free' else
            {h:[ar[h][a] for a in patterns] for h in hs}))
    state_slot = {}
    for i, r in enumerate(roles):
        states = slots[r]; cpt = {}
        nodes.append(dict(id='A_'+r,parents=['A'],states=['0','1'],
            cpt={a:[float(a[i]=='0'),float(a[i]=='1')] for a in patterns}))
        for h in (hs if policy=='functional' else [hs[0]]):
            for bit in ('0','1'):
                row = [float(s=='none') if bit=='0' else (0. if s=='none' else vr[r][h][s]) for s in states]
                cpt[key([h,bit] if policy=='functional' else [bit])] = row
        nodes.append(dict(id='M_'+r, parents=['H','A_'+r] if policy=='functional' else ['A_'+r], states=states, cpt=cpt))
        state_slot.update({s:r for s in states if s!='none'})
    for node in base['nodes']:
        if node['id'].startswith('E_'):
            obj = next(o for o in model['component_registry'].values() if o['presence_node']==node['id'])
            fid = obj['furniture_state']; r = state_slot[fid]
            nodes.append(dict(id=node['id'], parents=['M_'+r], states=['0','1'],
                cpt={s:[float(s!=fid),float(s==fid)] for s in slots[r]}))
        elif node['id'].startswith('S_'): nodes.append(deepcopy(node))
    maximum = sum(max(sum(len(g['objects']) for g in domain['programs'][s]['groups']) for s in states if s!='none') for states in slots.values())
    model.update(nodes=nodes, profiles=profiles, slots=slots, state_slot=state_slot, activation_rows=ar,
        variant_rows=vr, activation_patterns=patterns, activation_policy=policy, mechanism_source=source,
        model_variant=MODEL_VARIANT, response_interface=INTERFACE, empirical_tables=table,
        maximum_hidden_instances=maximum,
        distribution_scope='conditional_joint_activation_and_variants_given_fixed_observation_and_domain',
        sampling_policy='IID_all_samples_retained_including_empty_duplicates_and_geometry_failures',
        generation_factorization='q(H|O,D) p(A_joint|H,D) product_r p(V_r|H,A_r,D); V->E->S<-S_host; (O,H,V)->X',
        conditional_independence_assumed='Variants independent given H and activation; role activations are joint. Assumed, not identified.',
        RGB_likelihood_defined=False, count_likelihood_defined=False, calibrated_posterior_defined=False,
        real_causal_structure_identified=False,
        candidate_support=dict(candidate_instances=len(model['component_registry']), nonempty_programs=len(state_slot),
            all_nonempty_proposals_missing=not model['component_registry']))
    if policy=='function_free': model['generation_factorization']='q(H|O,D) p_m(A_joint) product_r p_m(V_r|A_r); V->E->S<-S_host; (O,H,V)->X'
    validate(model)
    if out is not None:
        write(out/'empirical_mechanisms.json', table); write(out/'probability_audit.json', marginals(model))
        write(out/'functional_probability_interventions.json', {h:marginals(model,h) for h in hs})
        write(out/'mechanism_diagnostics.json', diagnostics(model,domain.get('matched_profile_pairs',[])))
        probability_page(model, out/'probability_audit.html')
    return model


def marginals(model, do_profile=None):
    hs = [h['id'] for h in model['profiles']]
    if do_profile is not None and do_profile not in hs: raise ValueError('Unknown profile')
    q = dict(zip(hs, model['nodes'][0]['cpt']['root']))
    if do_profile is not None: q = {h:float(h==do_profile) for h in hs}
    roles = list(model['slots']); categories = set(); by_state = {}
    for fid in model['state_slot']:
        counts = Counter(o['category'].lower().replace('_',' ').strip() for g in model['furniture_states'][fid]['groups'] for o in g['objects'])
        by_state[fid] = counts; categories.update(counts)
    cp = dict.fromkeys(categories,0.); ce = dict.fromkeys(categories,0.)
    assembly = {s:0. for s in model['state_slot']}; activation = dict.fromkeys(roles,0.)
    pattern_law = dict.fromkeys(model['activation_patterns'],0.); pmf = {}
    for h in hs:
        for a, pa in model['activation_rows'][h].items():
            weight = q[h]*pa; pattern_law[a] += weight; local = {0:1.}; absent = dict.fromkeys(categories,1.)
            for i,r in enumerate(roles):
                if a[i]=='0': continue
                activation[r] += weight; rp = model['variant_rows'][r][h]; nlaw = {}; present = dict.fromkeys(categories,0.)
                for s,p in rp.items():
                    assembly[s] += weight*p; counts = by_state[s]; n = sum(counts.values()); nlaw[n] = nlaw.get(n,0.)+p
                    for c,nc in counts.items(): present[c] += p; ce[c] += weight*p*nc
                for c in categories: absent[c] *= 1-present[c]
                nxt = {}
                for n,p in local.items():
                    for m,t in nlaw.items(): nxt[n+m] = nxt.get(n+m,0.)+p*t
                local = nxt
            for c in categories: cp[c] += weight*(1-absent[c])
            for n,p in local.items(): pmf[str(n)] = pmf.get(str(n),0.)+weight*p
    objects = {oid:assembly[o['furniture_state']] for oid,o in model['component_registry'].items()}
    return dict(profile_probabilities=q, activation_probabilities=activation, activation_pattern_probabilities=pattern_law,
        assembly_probabilities=assembly, object_presence=objects, category_presence=cp, expected_category_counts=ce,
        hidden_count_probabilities=pmf, empty_probability=pmf.get('0',0.), expected_hidden_instances=sum(objects.values()),
        do_profile=do_profile, probabilities_calibrated=False, probability_scope=model['distribution_scope'],
        conditional_independence_assumed=model['conditional_independence_assumed'])


def sample_worlds(model, seed=3407, count=4):
    if type(count) is not int or count<1: raise ValueError('count must be positive')
    rng = np.random.default_rng(seed); law = marginals(model); worlds = []
    for trial in range(count):
        noise = {n['id']:float(rng.random()) for n in model['nodes']}; world = active_world(model,forward(model,noise))
        for obj in world['objects']: obj['presence_probability'] = law['object_presence'][obj['id']]
        world.update(noise=noise, seed=seed, trial=trial, mechanism_source=model['mechanism_source'], probability_scope=model['distribution_scope'])
        worlds.append(world)
    return worlds


def intervention_diagnostics(model, world):
    result = old.intervention_diagnostics(model,world); before = {o['id'] for o in world['objects']}; rows = []
    for role in model['slots']:
        do = {'A_'+role:'0'}
        after = active_world(model,forward(model,world['noise'],do))
        rows.append(dict(do=do,removed=sorted(before-{o['id'] for o in after['objects']}),
            after_inventory=[o['id'] for o in after['objects']],same_exogenous_noise=True,model_called=False,geometry_not_resolved=True))
    result['role_activation_queries'] = rows
    return result


def diagnostics(model, pairs=()):
    """Measure numeric H effects; flat mechanisms remain flat, never forced."""
    laws = {h['id']:marginals(model,h['id']) for h in model['profiles']}; rows = []
    for pair in pairs:
        a,b = laws[pair['without']],laws[pair['with_role']]
        rows.append(dict(pair,activation_delta={r:b['activation_probabilities'][r]-a['activation_probabilities'][r] for r in model['slots']},
            max_category_presence_delta=max((abs(p-a['category_presence'][c]) for c,p in b['category_presence'].items()),default=0.),
            expected_hidden_delta=b['expected_hidden_instances']-a['expected_hidden_instances']))
    maximum = max((max(abs(v) for v in row['activation_delta'].values()) for row in rows),default=0.)
    return dict(matched_profile_effects=rows,max_matched_activation_delta=maximum,
        activation_function_effect_observed_in_this_finite_model=maximum>1e-10,
        flat_effects_not_repaired_with_hand_weights=True,causal_gain_validated=False,
        effect_scope='assumed_model_relative_CPT_effects_not_identified_real_world_effects',
        sampling_uncertainty='See Dirichlet predictive standard deviations and valid/invalid counts in empirical_mechanisms.json')


def probability_page(model, path):
    law = marginals(model)
    path.write_text('<!doctype html><meta charset="utf-8"><h1>v6 联合功能区域存在 → 条件装配 → 实例</h1>'
        '<p>模型内概率直接进入采样；未校准。没有最低新增数。功能用途假设与实际可用性分别建模。</p>'
        '<p>策略：'+escape(model['activation_policy'])+'；P(零新增)='+str(law['empty_probability'])+
        '；期望新增记录='+str(law['expected_hidden_instances'])+'</p>'
        '<p><a href="probability_audit.json">全部类别、实例、数量与联合激活概率</a> · '
        '<a href="functional_probability_interventions.json">逐 H 数值干预</a> · <a href="empirical_mechanisms.json">原始机制</a></p>'
        '<pre>'+escape(json.dumps(dict(activation=model['activation_rows'],variants=model['variant_rows']),ensure_ascii=False,indent=2))+'</pre>')
