"""Experimental, assumed need/realization SCM over an image-proposed domain.

Z -> R_k -> B_k -> E_i -> S_i <- S_host. A realization is a JOINT bundle;
the same instance may satisfy several needs. Uniform categorical laws are a
reference policy, never a calibrated photo posterior or an identified cause.
Metric inference deliberately uses the existing backend for a controlled pilot.
"""
from copy import deepcopy
import itertools
import json
import re

import numpy as np

from . import assembly_delta as delta, assembly_model as typed
from .world_scm import validate, forward, key
from .scene_program import text

PROTOCOL = delta.PROTOCOL
MODEL_VARIANT = 'need_realization_scm_pilot_v1'
INTERFACE = 'shared_instance_catalog_need_realizations_v1'
hypotheses_prompt = typed.hypotheses_prompt
hypotheses_answer = typed.hypotheses_answer
semantic_diagnostics = delta.semantic_diagnostics


def furniture_prompt(case, hypothesis):
    # One wire format only. The previous prompt appended the complete legacy
    # alternatives prompt; the real Qwen run copied that conflicting schema.
    schema=dict(description='<specific surrounding-world hypothesis>',groups=[dict(
        function='<specific activity/storage/background group>',reason='<evidence-compatible assumption>',
        observed_refs=[],new_objects=[dict(key='object_1',category='<concrete noun>',variant='<specific variant>',
            reason='<reason this separate unseen instance may exist>',placement='<qualitative surrounding position>',
            attachment=dict(kind='floor',host=None),affordances=[])])],requirements=[dict(
        key='need_1',purpose='<specific demand or background source>',kind='activity',evidence_refs=[],
        assumptions=['<uncertainty>'],strategies=[dict(key='realization_1',description='<one joint realization>',
            object_keys=['object_1'],observed_refs=[],relations=[])])])
    return (
        'Build an executable HYPOTHESIS about how this existing inhabited room was organized. '
        'Infer possible intended activities, storage demands, architecture and incidental belongings. '
        'Intended use is uncertain; actual present activity is not the historical cause of ownership. '
        'Return ONE mechanism, not a list of completed scene alternatives. '
        'Use a SHARED CATALOG of distinct new hidden instances. Reuse the SAME object key if '
        'one instance serves several needs. Never duplicate observed objects to meet a need. '
        'Return exactly the three top-level fields description, groups and requirements. '
        'Use the SINGLE RESPONSE SCHEMA below; replace all angle-bracket placeholders with concrete content. '
        'groups contains at most 4 records; new_objects defines ONLY new hidden instances, while observed_refs '
        'references immutable existing observed IDs. Never redefine observed inventory or its integral parts. '
        'requirements: 0..4 records; kind is activity/storage/background/architecture. '
        'Each has 1..3 MUTUALLY EXCLUSIVE realization strategies, sampled uniformly by the program. '
        'Different requirements can coexist. Each combination of their strategies must be coherent; '
        'encode strongly coupled choices as ONE joint requirement instead of independent requirements. '
        'A strategy may use only observed_refs and object_keys=[], when observed inventory suffices '
        'or no extra hidden instance is warranted. This option is allowed, not mandatory. '
        'Object keys are an exact JOINT bundle, including every NEW support ancestor needed by its items. '
        'Every catalog instance must appear in at least one strategy. Total catalog <=24 instances; '
        'no inventory target, minimum count, required furniture category, or minimal-room objective. '
        'Retain plausible lived-in possessions, collections, decor or unused items through explicit '
        'background/storage hypotheses; do not pretend every belonging is needed by an activity. '
        'Evidence_refs identify supporting observations, not proof that the requirement exists. '
        'No probabilities, numeric sizes or coordinates. Do not claim causal identification. '
        'Relations belong to a strategy, and both endpoints must exist in that strategy or its observed_refs. '
        'There is no other top-level response wrapper. '
        'Each new object key is unique, short and reused exactly by object_keys and attachment.host. '
        'attachment.kind is floor/top/seat/shelf/hang/wall/ceiling/free. top/seat/shelf/hang require an '
        'exact new host key or observed ID; other kinds use host=null. Wall means actually mounted, '
        'not merely standing beside a wall. Affordances offered to OTHER objects are a subset of '
        'top,seat,shelf,hanging, often []; they are distinct from how the object itself is attached. '
        'Category must be a concrete noun, not furniture, decor or lighting. Multiple same-category '
        'instances are allowed when distinct identity/collection/placement is explained. '
        'Complete new AABBs must lie outside the ORIGINAL input frustum. Observed support domains '
        'are necessary contact conditions only; an empty contact-domain list excludes that observed '
        'support hypothesis, a nonempty list does not certify support. '
        + typed.RELATION_CONTRACT +
        'Relations use camera X/Y/Z. in_front_of is GREATER camera Z, not facing another object. '
        'Use placement prose for facing intent. No Adapter category restriction or mandatory object type. '
        '\nSINGLE RESPONSE SCHEMA: '+json.dumps(schema,ensure_ascii=False)+
        '\nSCENE HYPOTHESIS: '+json.dumps(hypothesis,ensure_ascii=False)+
        '\nOBSERVED SUPPORT DOMAINS: '+json.dumps(case.get('observed_support_domains',{}),ensure_ascii=False)+
        '\nFIXED OBSERVED ANCHORS: '+json.dumps(case['context'],ensure_ascii=False)
    )


def _identifier(value, path):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,47}', value):
        raise ValueError(path + ' needs a short identifier')
    return value


def furniture_answer(value, case, hypothesis):
    # The existing parser owns identity, attachment, support-domain and noun
    # contracts. A shared catalog is not itself an active world.
    if not isinstance(value, dict):
        raise ValueError('Return one mechanism object')
    if value.get('relations') or 'alternatives' in value:
        raise ValueError('Use strategy-local relations and one shared catalog')
    catalog = delta.furniture_answer(dict(alternatives=[dict(
        description=value.get('description'), groups=value.get('groups'), relations=[])]), case, hypothesis)[0]
    objects = {o['key']: o for g in catalog['groups'] for o in g['objects']}
    observed = {o['id'] for o in case['context']}
    rows = value.get('requirements')
    if not isinstance(rows, list) or len(rows) > 4:
        raise ValueError('requirements must be an array of at most four joint demand/source mechanisms')
    requirements = []; used = set(); need_keys = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('Each requirement must be an object')
        rid = _identifier(row.get('key'), f'requirements[{index}].key')
        if rid in need_keys:
            raise ValueError('Duplicate requirement key: ' + rid)
        need_keys.add(rid)
        kind = row.get('kind')
        if kind not in ('activity', 'storage', 'background', 'architecture'):
            raise ValueError('requirement.kind must be activity/storage/background/architecture')
        refs = row.get('evidence_refs'); assumptions = row.get('assumptions')
        if not isinstance(refs, list) or any(not isinstance(v, str) or v not in observed for v in refs):
            raise ValueError(rid + '.evidence_refs must reference exact observed IDs')
        if not isinstance(assumptions, list):
            raise ValueError(rid + '.assumptions must be an array')
        strategies = row.get('strategies')
        if not isinstance(strategies, list) or not 1 <= len(strategies) <= 3:
            raise ValueError(rid + ' needs 1..3 joint realization strategies')
        clean = []; strategy_keys = set()
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ValueError('Each strategy must be an object')
            sid = _identifier(strategy.get('key'), rid + '.strategy.key')
            if sid in strategy_keys or sid == 'off':
                raise ValueError('Duplicate or reserved strategy key: ' + sid)
            strategy_keys.add(sid)
            ids = strategy.get('object_keys'); old = strategy.get('observed_refs')
            if (not isinstance(ids, list) or any(not isinstance(v, str) or v not in objects for v in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError(rid + '/' + sid + ': object_keys must be distinct catalog IDs')
            if (not isinstance(old, list) or any(not isinstance(v, str) or v not in observed for v in old)
                    or len(old) != len(set(old))):
                raise ValueError(rid + '/' + sid + ': observed_refs must be distinct observed IDs')
            # No silent repair of inventory: every realization explicitly owns
            # the support ancestors it relies on, including shared ancestors.
            for oid in ids:
                host = objects[oid]['attachment']['host']
                if host is not None and host not in set(ids) | set(old):
                    raise ValueError(rid + '/' + sid + ': missing support ancestor/reference ' + host)
            relations, bindings = typed.relations_answer(strategy.get('relations', []),
                set(ids) | set(old), [objects[i] for i in ids], path=rid + '/' + sid + '.relations')
            clean.append(dict(key=sid, description=text(strategy.get('description'), 'strategy.description'),
                object_keys=list(ids), observed_refs=list(old), relations=relations, relation_bindings=bindings))
            used.update(ids)
        requirements.append(dict(key=rid, kind=kind, purpose=text(row.get('purpose'), 'requirement.purpose'),
            evidence_refs=list(refs), assumptions=[text(v, 'assumption') for v in assumptions], strategies=clean))
    if used != set(objects):
        raise ValueError('Unassigned catalog objects: ' + str(sorted(set(objects) - used)))
    catalog['requirements'] = requirements
    return [catalog]


def build_model(hypotheses, programs, unused_contents=None):
    nodes = []; registry = {}; mechanisms = {}; groups = []
    def add(name, states, parents, function):
        domains = {n['id']: n['states'] for n in nodes}
        table = {key(values): function(dict(zip(parents, values)), states)
                 for values in itertools.product(*(domains[p] for p in parents))}
        nodes.append(dict(id=name, states=states, parents=parents, cpt=table))
    def one(states, wanted):
        return [float(s == wanted) for s in states]
    add('Z', [z['id'] for z in hypotheses], [], lambda g, s: [1 / len(s)] * len(s))
    if sorted(p['hypothesis']['id'] for p in programs.values()) != sorted(z['id'] for z in hypotheses):
        raise ValueError('Exactly one shared catalog mechanism per scene hypothesis is required')
    for program in programs.values():
        z = program['hypothesis']['id']
        objects = [o for g in program['groups'] for o in g['objects']]
        mapping = {o['key']: z + '_' + o['key'] for o in objects}
        memberships = {o['key']: [] for o in objects}
        for need in program['requirements']:
            rid = 'R_' + z + '_' + need['key']; bid = 'B_' + z + '_' + need['key']
            add(rid, ['0', '1'], ['Z'], lambda g, s, z=z: one(s, '1' if g['Z'] == z else '0'))
            states = ['off'] + [s['key'] for s in need['strategies']]
            add(bid, states, [rid], lambda g, s, rid=rid: one(s, 'off') if g[rid] == '0'
                else [0.] + [1 / (len(s) - 1)] * (len(s) - 1))
            mechanisms[bid] = dict(deepcopy(need), requirement_node=rid, hypothesis_id=z)
            for strategy in need['strategies']:
                for oid in strategy['object_keys']:
                    memberships[oid].append((bid, strategy['key']))
        external = {o['attachment']['host'] for o in objects if o['attachment']['host'] not in mapping}
        for obj in typed.order_objects(objects, external):
            oid = mapping[obj['key']]; eid = 'E_' + oid; sid = 'S_' + oid
            members = memberships[obj['key']]; parents = list(dict.fromkeys(b for b, _ in members))
            add(eid, ['0', '1'], parents, lambda g, s, members=members:
                one(s, '1' if any(g[b] == choice for b, choice in members) else '0'))
            host = mapping.get(obj['attachment']['host'], obj['attachment']['host'])
            hn = 'S_' + host if host in mapping.values() else None
            add(sid, ['0', '1'], [eid] + ([hn] if hn else []), lambda g, s, eid=eid, hn=hn:
                one(s, '1' if g[eid] == '1' and (hn is None or g[hn] == '1') else '0'))
            rec = deepcopy(obj)
            rec.update(id=oid, group=z + '_' + obj['group'], presence_node=eid, installation_node=sid,
                requirement_memberships=[dict(strategy_node=b, choice=c) for b, c in members])
            rec['attachment']['host'] = host; registry[oid] = rec
        for group in program['groups']:
            groups.append(dict(deepcopy(group), key=z + '_' + group['key'], hypothesis_id=z))
    return validate(dict(protocol=PROTOCOL, model_variant=MODEL_VARIANT, response_interface=INTERFACE,
        nodes=nodes, component_registry=registry, hypotheses=deepcopy(hypotheses), groups=groups,
        need_mechanisms=mechanisms, scene_program_states=deepcopy(programs),
        generation_factorization='Z -> R_k -> B_k -> shared E_i -> S_i <- S_host',
        distribution_scope='uniform_reference_over_image_proposed_hypotheses_and_realizations',
        sampling_policy='iid_ancestral_samples_duplicates_retained',
        RGB_likelihood_defined=False, calibrated_posterior_defined=False, real_causal_structure_identified=False,
        installation_node_scope='declared_host_dependency_only',
        metric_mechanisms_learned=False, metric_backend='unchanged_assembly_parametric'))


def active_world(model, state):
    z = state['Z']; objects = [deepcopy(o) for o in model['component_registry'].values()
                              if state[o['presence_node']] == '1']
    mapping = {o['key']: o['id'] for o in model['component_registry'].values() if o['id'].startswith(z + '_')}
    relations = []; realized = []
    for bid, need in model['need_mechanisms'].items():
        if state[bid] == 'off':
            continue
        choice = next(s for s in need['strategies'] if s['key'] == state[bid])
        realized.append(dict(requirement=need['purpose'], kind=need['kind'], strategy=choice['description'],
            requirement_node=need['requirement_node'], strategy_node=bid, evidence_refs=need['evidence_refs']))
        for r in choice['relations']:
            relation = dict(r, source=mapping.get(r['source'], r['source']), target=mapping.get(r['target'], r['target']))
            if relation not in relations:
                relations.append(relation)
    hypothesis = next(h for h in model['hypotheses'] if h['id'] == z)
    for obj in objects:
        obj['installation_ready'] = state[obj['installation_node']] == '1'
    return dict(configuration=hypothesis['description'], hypothesis=deepcopy(hypothesis), objects=objects,
        groups=[deepcopy(g) for g in model['groups'] if g['hypothesis_id'] == z], relations=relations,
        realized_requirements=realized, structural_state=deepcopy(state),
        unresolved_installations=[o['id'] for o in objects if not o['installation_ready']])


def sample_worlds(model, seed=3407, count=3):
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    rng = np.random.default_rng(seed); worlds = []
    for trial in range(count):
        noise = {n['id']: float(rng.random()) for n in model['nodes']}
        world = active_world(model, forward(model, noise))
        world.update(noise=noise, seed=seed, trial=trial)
        worlds.append(world)
    return worlds


def semantic_posterior(model, evidence):
    """Exact finite SEMANTIC conditioning; no RGB likelihood is supplied.

    Enumerate only stochastic Z/B variables, then derive E/S, avoiding the
    legacy generic enumerator's 2**(2*N) deterministic state explosion.
    """
    validate(model); byid = {n['id']: n for n in model['nodes']}; rows = []
    if any(k not in byid or v not in byid[k]['states'] for k, v in evidence.items()):
        raise ValueError('Invalid semantic evidence')
    for zi, z in enumerate(byid['Z']['states']):
        bids = [bid for bid, need in model['need_mechanisms'].items() if need['hypothesis_id'] == z]
        for choices in itertools.product(*(byid[b]['states'][1:] for b in bids)):
            noise = {n['id']: .5 for n in model['nodes']}
            ps = byid['Z']['cpt']['root']; noise['Z'] = sum(ps[:zi]) + ps[zi] / 2
            mass = ps[zi]
            for bid, choice in zip(bids, choices):
                node = byid[bid]; ps = node['cpt']['1']; j = node['states'].index(choice)
                noise[bid] = sum(ps[:j]) + ps[j] / 2; mass *= ps[j]
            state = forward(model, noise)
            if mass > 0 and all(state[k] == v for k, v in evidence.items()):
                rows.append(dict(state=state, joint_mass=mass))
    total = sum(r['joint_mass'] for r in rows)
    if total <= 0:
        raise ValueError('Semantic evidence has zero probability in the specified domain')
    for row in rows:
        row['probability'] = row['joint_mass'] / total
    return dict(rows=rows, evidence_probability=total,
        inference_scope='exact_semantic_conditioning_in_assumed_finite_model_not_RGB_Bayes')


def abduct_semantics(model, evidence, seed=3407):
    posterior = semantic_posterior(model, evidence); rng = np.random.default_rng(seed)
    state = posterior['rows'][rng.choice(len(posterior['rows']), p=[r['probability'] for r in posterior['rows']])]['state']
    noise = {}
    for node in model['nodes']:
        ps = node['cpt'][key([state[p] for p in node['parents']])]; i = node['states'].index(state[node['id']])
        noise[node['id']] = min(float(sum(ps[:i]) + ps[i] * rng.random()), float(np.nextafter(1., 0.)))
    if forward(model, noise) != state:
        raise RuntimeError('Semantic noise abduction failed to replay')
    return dict(factual_state=state, noise=noise, evidence=evidence, evidence_probability=posterior['evidence_probability'],
        scope='model_relative_semantic_abduction_inverse_CDF_coupling_assumed_not_identified')


def intervention_diagnostics(model, world):
    before = {o['id'] for o in world['objects']}; queries = []
    interventions = [{need['requirement_node']: '0'} for bid, need in model['need_mechanisms'].items()
                     if world['structural_state'][need['requirement_node']] == '1']
    for obj in world['objects']:
        if any(o['attachment']['host'] == obj['id'] for o in world['objects']):
            interventions.append({obj['presence_node']: '0'})
    for do in interventions:
        after = active_world(model, forward(model, world['noise'], do)); ids = {o['id'] for o in after['objects']}
        queries.append(dict(do=do, removed=sorted(before - ids), added=sorted(ids - before),
            after_inventory=sorted(ids), unresolved_installations=after['unresolved_installations'],
            same_exogenous_noise=True, model_called=False, geometry_not_resolved=True))
    return dict(queries=queries, causal_gain_validated=False,
        scope='assumed_historical_organization_mechanism_not_immediate_removal_physics_or_photo_counterfactual')
