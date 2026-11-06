"""v11: jointly sample optional inventory AND camera-relative organization.

Sparse joint categorical draws preserve correlations between assemblies/regions.
One canonical instance registry is shared by all region choices. No training,
rear quotas, hidden-room capacity, biography, or deletion of sampled objects.
"""
from collections import Counter
from copy import deepcopy
from html import escape
import json
from pathlib import Path

import numpy as np

from ..common import read, write, sha
from . import assembly_set as previous, assembly_panorama as panorama
from . import assembly_population as population
from .world_scm import validate, forward

PROTOCOL = previous.PROTOCOL
INTERFACE = 'joint_inventory_camera_organization_v11'
MODEL_VARIANT = 'holistic_functional_spatial_scm_v11'
KIND = 'joint_inventory_camera_organization'
PROPOSAL_REVISION = 'canonical_units_joint_spatial_choices_v11'
FUNCTIONS = previous.FUNCTIONS
REGIONS = panorama.REGIONS
profile = previous.profile
hypotheses_prompt = previous.hypotheses_prompt
hypotheses_answer = previous.hypotheses_answer
recognition_prompt = previous.recognition_prompt
semantic_diagnostics = previous.semantic_diagnostics
load_domain = previous.load_domain
_positive = previous._positive


def furniture_prompt(case, hypothesis, channel='surrounding'):
    return previous.furniture_prompt(case, hypothesis, channel) + (
        '\nV11 SPATIAL UNCERTAINTY: candidates are assemblies in ONE surrounding 360-degree world, '
        'not just furnishings next to the photographed anchors. Functional counterparts may be across the original camera. '
        'For a freestanding assembly without a known camera sector, omit layout_region or use unspecified; '
        'the joint mechanism will sample its sector together with all other assemblies. Optional region_options '
        'is a nonempty subset of left/right/rear when there is evidence restricting alternatives. '
        'Do not infer a camera-depth order from facing, opposite wall, or service. Leave unsupported orders absent. '
        'No sector, class or object count is mandatory. Distinguish a new hidden instance from a visible object described twice. '
        'For attachment to a fixed obs ID add observed_host_name equal to its EXACT fixed description, '
        'e.g. a picture frame must never be called a side table. Consider its actual support affordance; '
        'frames and lamps are not ordinary tables. For top/seat/shelf items optional support_model is '
        'rigid_box (default: box centre projection is the support proxy) or contact_only '
        '(explicitly uncertain flexible/unsupported mass model); neither certifies physical stability. '
        'Hosts and ordinary contents are one dependency-closed candidate, not separate mandatory needs.')


def furniture_answer(value, case, hypothesis):
    programs = previous.furniture_answer(value, case, hypothesis)
    accepted = []; rejected = []
    names = {o['id']: o['description'] for o in case['context']}
    for p in programs:
        if not p['groups']: accepted.append(p); continue
        ai = p['proposal_alternative_index']; indices = p['proposal_component_indices']
        raw_groups = [value['alternatives'][ai]['groups'][i] for i in indices]
        raw = {o['key']: o for g in raw_groups for o in g['new_objects']}
        try:
            options = None
            for g in raw_groups:
                domain = g.get('region_options')
                if domain is not None:
                    if not isinstance(domain, list) or not domain or len(set(domain)) != len(domain) or any(r not in REGIONS[:3] for r in domain):
                        raise ValueError('region_options must be distinct left/right/rear values')
                    options = set(domain) if options is None else options & set(domain)
            for g in p['groups']:
                for o in g['objects']:
                    source = raw[o['key']]; host = o['attachment']['host']
                    if host in names and source.get('observed_host_name') != names[host]:
                        raise ValueError(o['key']+': observed_host_name must be '+repr(names[host])+' for '+host)
                    support = source.get('support_model', 'rigid_box')
                    if support not in ('rigid_box', 'contact_only'): raise ValueError('Unknown support_model')
                    o['support_model'] = support
                    if host in names: o['observed_host_name'] = names[host]
            declared = {g['layout_region'] for g in p['groups']} - {'unspecified'}
            anchored = any(o['attachment']['host'] in names for g in p['groups'] for o in g['objects'])
            candidates = ['unspecified'] if anchored and not declared else sorted(declared or options or set(REGIONS[:3]))
            if options is not None:
                if not options: raise ValueError('Dependent groups have disjoint region_options')
                if declared and not declared <= options: raise ValueError('Declared region contradicts region_options')
                if anchored: raise ValueError('Fixed observed-host assemblies use layout_region, not free region_options')
            legal = []; conflicts = []
            for region in candidates:
                trial = deepcopy(p)
                for g in trial['groups']: g['layout_region'] = region
                try: panorama.check_region_order(trial, case); legal.append(region)
                except ValueError as exc: conflicts.append(dict(region=region, error=str(exc)))
            if not legal: raise ValueError('No region compatible with declared camera ordering')
            for g in p['groups']: g['region_options'] = legal
            p['region_domain_diagnostics'] = dict(legal=legal, rejected=conflicts,
                scope='Necessary centre/order test only; no measured room boundary or full physical fit')
            accepted.append(p)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(alternative=ai, groups=indices, error=str(exc)))
    if not accepted: raise ValueError('No valid v11 unit: '+json.dumps(rejected, ensure_ascii=False))
    for p in accepted:
        p['proposal_validation'].setdefault('rejected', []).extend(rejected)
        p['proposal_validation']['policy'] = 'dependency_component_atomic_with_host_identity_and_region_domain'
    return accepted


def build_model(hypotheses, programs, unused_contents=None):
    return population.build_model(hypotheses, programs, signature_fields=(
        'layout_region', 'region_options', 'functional_links', 'function_tags'))


def choice_answer(value, domain):
    if not isinstance(value, dict) or set(value) != {'assemblies'} or not isinstance(value['assemblies'], list):
        raise ValueError('Return exactly {"assemblies":[{"id":...,"region":...}]}')
    choices = dict.fromkeys(domain['slots'], 'none')
    for row in value['assemblies']:
        if not isinstance(row, dict) or set(row) != {'id', 'region'}: raise ValueError('Each selection needs exactly id and region')
        c = 'unit_'+str(row['id'])
        if c not in choices or choices[c] != 'none': raise ValueError('Use each exact assembly ID at most once')
        if row['region'] not in domain['region_domains'][c]: raise ValueError(c+' region must be in '+str(domain['region_domains'][c]))
        choices[c] = row['region']
    return json.dumps(list(choices.values()), separators=(',', ':'))


def joint_prompt(domain, order, h=None, reference=False):
    menu = previous._menu(domain, order)
    for row in menu: row['allowed_camera_regions'] = domain['region_domains']['unit_'+row['id']]
    text = ('Sample ONE coherent complete surrounding-world hypothesis: choose a JOINT SET of assemblies '
        'and ONE allowed ORIGINAL-CAMERA region for each selected assembly. Decide all choices together. '
        'The image covers only one camera sector; behind the ORIGINAL CAMERA is Z<0, not behind a sofa. '
        'Use functional organization, observed anchors, support dependencies, distinct possessions and redundancy. '
        'Do not require a second activity zone to select ordinary belongings. Same-function assemblies can coexist. '
        'A region choice transports that whole assembly; do not move a fixed observed host. '
        'Regions are uncertain layout hypotheses, not measured walls. Service/facing does not imply near or greater Z. '
        'Unphotographed is not absent. Sample an existing world, not just objects proved visible, a minimal design, '
        'or everything conceivable. Empty and repeated sampled worlds are legal. No count, rear quota or uniform coverage. '
        'Output only {"assemblies":[{"id":"exact assembly ID","region":"one allowed camera region"}]}. '
        '\nCANDIDATE_ASSEMBLIES: '+json.dumps(menu, ensure_ascii=False)+
        '\nFIXED_POSITIVE_OBSERVATIONS: '+json.dumps(domain['common_inventory'], ensure_ascii=False))
    if reference: return text+'\nREFERENCE: no particular H or RGB; the fixed candidate domain is the only context.'
    if h is not None: text += '\nCURRENT_PARENT_H: '+json.dumps(h, ensure_ascii=False)
    return text+'\nAttached RGB, when present, is the same partial observation. Unseen sectors are unknown.'


def raw_choices(votes):
    counts = Counter(v for v in votes if v is not None)
    if not counts: raise ValueError('All fixed-budget joint choices invalid; no empty fallback')
    return dict(counts=dict(counts), valid_draws=sum(counts.values()), invalid_draws=sum(v is None for v in votes))


def compile_model(base, domain, table, policy='functional'):
    if policy not in ('functional', 'function_free', 'direct'): raise ValueError('Unknown policy')
    m = deepcopy(base); hs = [h['id'] for h in domain['profiles']]; channels = list(domain['slots'])
    alpha = table['concentration']; eta = table['reference_concentration']
    _positive(alpha, 'concentration'); _positive(eta, 'reference_concentration')
    hp = table['recognition']['probabilities']
    if set(hp) != set(hs): raise ValueError('Recognition domain mismatch')
    ref = table['reference']; conditional = {h: table['direct'] if policy=='direct' else table['joint'][h] for h in hs}
    for row in [ref]+list(conditional.values()):
        if type(row['valid_draws']) is not int or row['valid_draws']<1 or sum(row['counts'].values())!=row['valid_draws'] or any(type(n) is not int or n<0 for n in row['counts'].values()):
            raise ValueError('Joint raw counts must be nonnegative integers with the declared nonzero total')
    patterns = sorted(set(ref['counts']) | {v for row in conditional.values() for v in row['counts']})
    vectors = {}
    for i, pattern in enumerate(patterns):
        values = json.loads(pattern)
        if len(values) != len(channels) or any(v not in ['none']+domain['region_domains'][c] for c,v in zip(channels,values)):
            raise ValueError('Invalid frozen joint vector')
        vectors['j'+str(i)] = values
    components = ['base']+list(vectors); rows = {}; ref_rows = {}
    for h, row in conditional.items():
        if row['valid_draws'] < 1 or sum(row['counts'].values()) != row['valid_draws']: raise ValueError('Invalid draw counts')
        q0 = {'base': eta/(ref['valid_draws']+eta)}
        q0.update({j: ref['counts'].get(pattern,0)/(ref['valid_draws']+eta) for j,pattern in zip(vectors,patterns)})
        ref_rows[h] = {j: alpha*q0[j]/(row['valid_draws']+alpha) for j in components}
        rows[h] = {j: ref_rows[h][j]+(row['counts'].get(patterns[components.index(j)-1],0)/(row['valid_draws']+alpha) if j!='base' else 0.) for j in components}
    if policy=='function_free':
        for collection in (rows, ref_rows):
            mean = {j: sum(hp[h]*collection[h][j] for h in hs) for j in components}
            for h in hs: collection[h] = deepcopy(mean)
    nodes = [dict(id='H', parents=[], states=hs, cpt={'root':[hp[h] for h in hs]})]
    functional = policy=='functional'
    nodes.append(dict(id='J', parents=['H'] if functional else [], states=components,
        cpt={h:[rows[h][j] for j in components] for h in hs} if functional else {'root':[rows[hs[0]][j] for j in components]}))
    probabilities = {}
    for i,c in enumerate(channels):
        states = ['none']+domain['region_domains'][c]; probabilities[c] = {}
        for j in components:
            # Inventory base mass is 1/2 regardless of the number of region choices.
            probabilities[c][j] = ([.5]+[.5/(len(states)-1)]*(len(states)-1) if j=='base'
                else [float(v==vectors[j][i]) for v in states])
        nodes.extend([
            dict(id='M_'+c, parents=['J'], states=states, cpt=probabilities[c]),
            dict(id='A_'+c, parents=['M_'+c], states=['0','1'], cpt={v:[float(v=='none'),float(v!='none')] for v in states}),
            dict(id='R_'+c, parents=['M_'+c], states=list(REGIONS), cpt={v:[float(r==(v if v!='none' else 'unspecified')) for r in REGIONS] for v in states})])
    for n in base['nodes']:
        if n['id'].startswith('E_'):
            o = next(o for o in base['component_registry'].values() if o['presence_node']==n['id'])
            c = 'unit_'+o['furniture_state']
            nodes.append(dict(id=n['id'], parents=['A_'+c], states=['0','1'], cpt={'0':[1.,0.],'1':[0.,1.]}))
        elif n['id'].startswith('S_'): nodes.append(deepcopy(n))
    m.update(nodes=nodes, profiles=domain['profiles'], slots=domain['slots'], region_domains=domain['region_domains'],
        joint_vectors=vectors, joint_rows=rows, reference_rows=ref_rows, component_probabilities=probabilities,
        mechanism_policy=policy, mechanism_source=table['source'], empirical_tables=table,
        model_variant=MODEL_VARIANT, response_interface=INTERFACE, training_performed=False,
        generation_factorization='q(H|O,D) p(B_joint,R_joint|H,O_positive,D); B->E->S<-S_host; (O,H,B,R,E,S)->X',
        distribution_scope='finite_domain_joint_inventory_spatial_reference_predictive_not_calibrated_world_posterior',
        evidence_delivery='RGB at proposal/recognition/layout; joint choice uses verified-positive inventory; explicit RGB control',
        conditional_independence_assumed='Whole vectors jointly sampled; only symmetric base factorizes over candidate units',
        sampling_policy='IID_all_empty_duplicate_and_failed_worlds_retained',
        joint_mixture_scope='J is a computational representation, not a physical cause',
        maximum_hidden_instances=len(base['component_registry']),
        candidate_support=dict(candidate_instances=len(base['component_registry']),nonempty_programs=len(channels),all_nonempty_proposals_missing=False),
        RGB_likelihood_defined=False, calibrated_posterior_defined=False, real_causal_structure_identified=False)
    return validate(m)


def prepare_model(base, case, config, out):
    bundle = Path(config['sdscene_bundle']); domain = read(bundle/'domain.json'); source = config['mechanism_source']
    if source=='per_backend': source=config['execution_backend']
    if source=='per_backend_rgb': source=config['execution_backend']+'_rgb'
    table = read(bundle/'mechanisms'/(source+'.json'))
    if table['kind'] != KIND or table['domain_sha256'] != sha(bundle/'domain.json'): raise ValueError('Joint spatial mechanism/domain mismatch')
    if table['conditional_RGB'] != (config['mechanism_source']=='per_backend_rgb'):raise ValueError('Joint RGB protocol mismatch')
    m = compile_model(base, domain, table, config.get('mechanism_policy','functional'))
    if out is not None:
        write(out/'empirical_mechanisms.json',table); write(out/'probability_audit.json',marginals(m))
        write(out/'functional_probability_interventions.json',{h['id']:marginals(m,h['id']) for h in m['profiles']})
        (out/'probability_audit.html').write_text('<meta charset="utf-8"><h1>联合库存与全景方位概率</h1><p>概率真实进入采样；模型未校准，无数量或后方配额。</p><pre>'+escape(json.dumps(marginals(m),ensure_ascii=False,indent=2))+'</pre>')
    return m


def marginals(m, do_profile=None):
    q = dict(zip([h['id'] for h in m['profiles']],m['nodes'][0]['cpt']['root']))
    if do_profile is not None:
        if do_profile not in q: raise ValueError('Unknown profile')
        q = {h:float(h==do_profile) for h in q}
    weights = {j:sum(q[h]*m['joint_rows'][h][j] for h in q) for j in next(iter(m['joint_rows'].values()))}
    counts = {c:Counter(o['category'].lower().replace('_',' ').strip() for g in m['furniture_states'][states[1]]['groups'] for o in g['objects']) for c,states in m['slots'].items()}
    cats = {k for v in counts.values() for k in v}; cp=dict.fromkeys(cats,0.); ce=dict(cp)
    activation=dict.fromkeys(m['slots'],0.); spatial={c:dict.fromkeys(REGIONS,0.) for c in m['slots']}; pmf={}; any_r=dict.fromkeys(REGIONS,0.)
    for j,w in weights.items():
        local={0:1.}; absent=dict.fromkeys(cats,1.); missing_r=dict.fromkeys(REGIONS,1.)
        for c in m['slots']:
            p = m['component_probabilities'][c][j]; on=1-p[0]; n=sum(counts[c].values()); activation[c]+=w*on
            for region,pr in zip(m['region_domains'][c],p[1:]): spatial[c][region]+=w*pr; missing_r[region]*=1-pr
            for cat,nc in counts[c].items(): absent[cat]*=1-on; ce[cat]+=w*on*nc
            nxt={}
            for k,v in local.items():
                for z,t in ((0,1-on),(n,on)): nxt[k+z]=nxt.get(k+z,0.)+v*t
            local=nxt
        for cat in cats: cp[cat]+=w*(1-absent[cat])
        for region in REGIONS: any_r[region]+=w*(1-missing_r[region])
        for n,p in local.items(): pmf[str(n)]=pmf.get(str(n),0.)+w*p
    assembly={states[1]:activation[c] for c,states in m['slots'].items()}
    objects={oid:assembly[o['furniture_state']] for oid,o in m['component_registry'].items()}
    return dict(profile_probabilities=q,assembly_probabilities=assembly,activation_probabilities=activation,
        object_presence=objects,category_presence=cp,expected_category_counts=ce,hidden_count_probabilities=pmf,
        empty_probability=pmf.get('0',0.),expected_hidden_instances=sum(objects.values()),
        channel_region_joint_presence=spatial,any_declared_region_presence=any_r,joint_component_probabilities=weights,
        probability_scope=m['distribution_scope'],probabilities_calibrated=False,do_profile=do_profile)


def active_world(m, state):
    mapped=deepcopy(state)
    for c,states in m['slots'].items(): mapped['M_'+c]=states[1] if state['M_'+c]!='none' else 'none'
    world=panorama.active_world(m,mapped);world['structural_state']=deepcopy(state)
    return world


def sample_worlds(m, seed=3407, count=4):
    if type(count) is not int or count<1: raise ValueError('Positive world count required')
    rng=np.random.default_rng(seed);law=marginals(m);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in m['nodes']};world=active_world(m,forward(m,noise))
        for o in world['objects']:o['presence_probability']=law['object_presence'][o['id']]
        world.update(noise=noise,seed=seed,trial=trial,probability_scope=m['distribution_scope'],mechanism_source=m['mechanism_source']);worlds.append(world)
    return worlds


def intervention_diagnostics(m, world):
    rows=[]
    for o in world['objects']:
        if any(child['attachment']['host']==o['id'] for child in world['objects']):
            intervention={o['presence_node']:'0'};after=active_world(m,forward(m,world['noise'],intervention))
            rows.append(dict(do=intervention,after_inventory=[v['id'] for v in after['objects']],unresolved_installations=after['unresolved_installations']))
    spatial=[]
    for c in world['spatial_program']:
        for region in m['region_domains'][c]:
            after=active_world(m,forward(m,world['noise'],{'R_'+c:region}))
            spatial.append(dict(do={'R_'+c:region},inventory_preserved=[o['id'] for o in after['objects']]==[o['id'] for o in world['objects']],geometry_not_resolved=True))
    functional=[]
    for h in m['profiles']:
        after=active_world(m,forward(m,world['noise'],{'H':h['id']}))
        functional.append(dict(do={'H':h['id']},probabilities=marginals(m,h['id']),
            after_inventory=[o['id'] for o in after['objects']],after_spatial_program=after['spatial_program'],geometry_not_resolved=True))
    return dict(queries=rows,spatial_program_queries=spatial,same_exogenous_noise=True,model_called=False,
        functional_queries=functional,
        scope='Assumed-model sensitivity only; no identified physical counterfactual or resolved intervention geometry')
