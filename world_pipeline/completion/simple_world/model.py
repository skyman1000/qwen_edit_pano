"""Z -> whole program F -> E_i -> S_i <- S_host; F -> joint camera plan R.

LM proposals approximate abduction from O. Uniform finite reference weights
are declared assumptions, not estimated probabilities or a calibrated posterior.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from ...common import read, write, sha
from .. import assembly_delta as joint, assembly_inventory as inventory
from .. import assembly_compositional as spatial
from ..world_scm import validate, forward

PROTOCOL = joint.PROTOCOL
INTERFACE = 'simple_joint_hypothesis_host_scm_v1'
MODEL_VARIANT = 'simple_joint_hypothesis_host_scm_v1'
semantic_diagnostics = joint.semantic_diagnostics


def build_model(hypotheses, programs, unused=None):
    return joint.build_model(hypotheses, programs)


def load_domain(case, config):
    folder = Path(config['sdscene_bundle'])
    d = read(folder / 'joint_domain.json')
    # Bound to the exact positive observation and support-domain inference.
    if any(d['case'].get(k) != case.get(k) for k in ('context', 'hfov', 'image_size', 'observed_support_domains')):
        raise ValueError('Joint domain does not match frozen observation/camera/support domains')
    return d['hypotheses'], d['programs'], dict(source=str(folder),
        domain_sha256=sha(folder / 'joint_domain.json'), whole_alternatives_preserved=True,
        API_candidates_used=False, probability_tables_reused=False)


def prepare_model(base, case, config, out):
    m = deepcopy(base)
    plans, rejections = spatial.spatial_choices(m['furniture_states'], case)
    # Empty delta still has exactly one empty camera plan.
    for fid, p in m['furniture_states'].items():
        if not p['groups'] and not any(v['module'] == fid for v in plans.values()):
            plans[fid + '_empty'] = dict(module=fid, regions={})
    ids = list(plans)
    m['nodes'].append(dict(id='R', parents=['F'], states=ids, cpt={fid:[
        float(plans[r]['module'] == fid) / sum(v['module'] == fid for v in plans.values())
        for r in ids] for fid in m['furniture_states']}))
    m.update(model_variant=MODEL_VARIANT, response_interface=INTERFACE, spatial_choices=plans,
        rejected_camera_plans=rejections, mechanism_source='local_Qwen_joint_proposals_uniform_reference',
        maximum_hidden_instances=max(sum(len(g['objects']) for g in p['groups']) for p in m['furniture_states'].values()),
        distribution_scope='explicit_uniform_H_then_whole_program_then_legal_camera_plan_reference_not_photo_posterior',
        generation_factorization='q_reference(Z|O,D) p_reference(F|Z,D) p_reference(R|F,D); F->E_i->S_i<-S_host; (O,F,R)->dimensions_and_layout',
        sampling_policy='fixed_count_IID_empty_duplicates_and_failures_retained',
        conditional_independence_assumed='No independent role selection; every group inside F coexists',
        candidate_support=dict(candidate_instances=len(m['component_registry']),
            nonempty_programs=sum(any(g['objects'] for g in p['groups']) for p in m['furniture_states'].values()),
            explicit_empty_programs=sum(not any(g['objects'] for g in p['groups']) for p in m['furniture_states'].values())),
        causal_gain_validated=False, calibrated_posterior_defined=False, real_causal_structure_identified=False)
    validate(m)
    if out:
        write(out / 'probability_audit.json', dict(reference_law=m['distribution_scope'],
            hypothesis_probabilities=dict(zip(m['nodes'][0]['states'],m['nodes'][0]['cpt']['root'])),
            program_probabilities=m['nodes'][1]['cpt'], LM_probability_estimation_calls=0))
    return m


def active_world(m, state):
    world = inventory.active_world(m, state)
    regions = m['spatial_choices'][state['R']]['regions']
    world['spatial_program'] = {g['key']:dict(region=regions[g['key']], region_node='R') for g in world['groups']}
    for g in world['groups']: g['layout_region'] = regions[g['key']]
    mapping = {o['key']:o['id'] for o in m['component_registry'].values() if o['furniture_state'] == state['F']}
    owned = {o['id'] for o in world['objects']}; links = []; unresolved = []
    for g in world['groups']:
        for link in g.get('functional_links', []):
            row = dict(link, source=mapping.get(link['source'],link['source']), target=mapping.get(link['target'],link['target']))
            (links if row['source'] in owned and (row['target'] in owned or row['target'].startswith('obs_')) else unresolved).append(row)
    world.update(functional_links=links, unresolved_functional_links=unresolved)
    return world


def sample_worlds(m, seed=3407, count=4):
    rng = np.random.default_rng(seed); worlds = []
    for trial in range(count):
        noise = {n['id']:float(rng.random()) for n in m['nodes']}
        w = active_world(m, forward(m, noise)); w.update(noise=noise, seed=seed, trial=trial)
        worlds.append(w)
    return worlds


def intervention_diagnostics(m, w):
    queries = []
    for o in w['objects']:
        if any(v['attachment']['host'] == o['id'] for v in w['objects']):
            change = {o['presence_node']:'0'}; after = active_world(m, forward(m,w['noise'],change))
            queries.append(dict(do=change, after_inventory=[v['id'] for v in after['objects']],
                unresolved_installations=after['unresolved_installations']))
    configurations = []
    for z in m['hypotheses']:
        if z['id'] == w['structural_state']['Z']: continue
        after = active_world(m, forward(m,w['noise'],{'Z':z['id']}))
        configurations.append(dict(do={'Z':z['id']}, after_program=after['structural_state']['F'],
            after_inventory=[v['id'] for v in after['objects']]))
    return dict(queries=queries, configuration_queries=configurations, same_exogenous_noise=True,
        model_called=False, geometry_not_resolved=True,
        scope='Assumed causal program and declared host installation; not identified physical counterfactuals')
