"""Evidence-conditioned functional SCM, conditional on a fixed candidate grammar.

No resident biography. No LM confidence scores. Exact enumeration of the small
functional bottleneck; all sampled inventories consume its numerical posterior.
Defaults are EXPLICIT ASSUMPTIONS, not fitted or calibrated scene frequencies.
"""
from copy import deepcopy
from html import escape
import itertools
import json
import re

import numpy as np

from . import assembly_needs as needs
from .world_scm import validate, forward, key
from ..common import read, write

PROTOCOL = needs.PROTOCOL
MODEL_VARIANT = 'evidence_functional_scm_v2'
INTERFACE = 'shared_catalog_functional_role_realizations_v2'
ROLES = ('social', 'reading', 'work', 'dining', 'storage', 'background')
# Every number below is elicited engineering prior knowledge, NOT a paper's
# fitted parameter or the model's self-reported confidence.
DEFAULTS = dict(version=1, provenance='elicited_reference_v1_NOT_FITTED',
    role_priors=dict(zip(ROLES, [.40, .30, .25, .25, .55, .50])),
    need_activation=dict(inactive=.12, active=.78),
    cue_leak=.08, detection=dict(coverage=.45, recall=.85, false_positive=.01),
    cue_strengths=dict(
        seating=dict(social=.80, reading=.30),
        task_light=dict(reading=.65, work=.55),
        books=dict(reading=.75, storage=.35),
        work_tools=dict(work=.90), dining_tools=dict(dining=.85),
        storage_units=dict(storage=.80), display=dict(background=.65),
        media=dict(social=.65), work_surface=dict(work=.65, dining=.25)),
    strategy_weights={})
# Conservative, auditable cue recognition. Unknown nouns produce no evidence.
# Category and description are aliases of ONE detection, never two factors.
PATTERNS = dict(seating=r'\b(sofa|couch|armchair|ottoman)\b|沙发|扶手椅|脚凳',
    task_light=r'\b((floor|desk|table|reading) lamp|task light)\b|落地灯|台灯|阅读灯',
    books=r'\b(books?|book stack|bookshelf|bookcase)\b|书籍|书架|书柜',
    work_tools=r'\b(laptop|computer|keyboard|monitor|printer)\b|电脑|键盘|显示器|打印机',
    dining_tools=r'\b(dining table|dinner plate|cutlery|dining chair)\b|餐桌|餐椅|餐具',
    storage_units=r'\b(wardrobe|bookcase|bookshelf|cabinet|dresser)\b|衣柜|书柜|书架|储物柜',
    display=r'\b(picture frame|painting|sculpture|vase|plant)\b|相框|绘画|雕塑|花瓶|绿植',
    media=r'\b(television|tv|projector|remote control)\b|电视|投影仪|遥控器',
    work_surface=r'\b(desk|workbench)\b|书桌|工作台')


def parameters(value=None):
    p = deepcopy(DEFAULTS if value is None else value)
    if set(p) != set(DEFAULTS) or p['version'] != 1:
        raise ValueError('Functional parameters require the complete version-1 schema')
    if set(p['role_priors']) != set(ROLES) or set(p['cue_strengths']) != set(PATTERNS):
        raise ValueError('Role/cue vocabulary differs from the model')
    def probability(v):
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or not 0 < v < 1:
            raise ValueError('Probabilities must be finite and strictly between zero and one')
    for v in [*p['role_priors'].values(), *p['need_activation'].values(),
              p['cue_leak'], *p['detection'].values()]:
        probability(v)
    if set(p['need_activation']) != {'inactive', 'active'} or set(p['detection']) != {'coverage','recall','false_positive'}:
        raise ValueError('Invalid activation/detection schema')
    for strengths in p['cue_strengths'].values():
        if not strengths or not set(strengths) <= set(ROLES):
            raise ValueError('Invalid noisy-OR parents')
        for v in strengths.values(): probability(v)
    if not isinstance(p['strategy_weights'], dict) or not isinstance(p['provenance'], str):
        raise ValueError('Parameters need provenance and a strategy_weights object')
    return p


def hypotheses_prompt(case, grounding):
    return ('Infer uncertain FUNCTIONAL ORGANIZATION configurations of a scene-generating program. '
        'Do not infer residents, occupation, wealth, habits, biography or ownership history. '
        'A reading organization is a latent generative hypothesis, not an observed intention or past activity. '
        + needs.hypotheses_prompt(case, grounding))


hypotheses_answer = needs.hypotheses_answer
semantic_diagnostics = needs.semantic_diagnostics
active_world = needs.active_world
intervention_diagnostics = needs.intervention_diagnostics


def furniture_prompt(case, hypothesis):
    prompt = needs.furniture_prompt(case, hypothesis)
    prefix, rest = prompt.split('\nSINGLE RESPONSE SCHEMA: ', 1)
    raw, suffix = rest.split('\nSCENE HYPOTHESIS:', 1)
    schema = json.loads(raw)
    schema['requirements'][0]['role'] = 'social|reading|work|dining|storage|background'
    prefix = prefix.replace('sampled uniformly by the program', 'sampled by a separate numerical mechanism')
    return ('Use functional organization in the generative model as the latent explanation. '
        'Each requirement MUST have role equal to exactly one of '+','.join(ROLES)+'. '
        'kind and role are DIFFERENT fields: kind is activity/storage/background/architecture. '
        'For role social/reading/work/dining, kind is activity, never social/reading/work/dining. '
        'Example: {"kind":"activity","role":"social"}. '
        'role selects an explicit CPT; it is not a confidence score. Use background for architecture, '
        'incidental objects or a function outside this limited vocabulary. Strongly coupled activities '
        'belong in a joint realization; choose its primary role and state the limitation in assumptions. '
        'Keep an observed-only strategy when the visible equipment can satisfy this requirement. '
        'No invented lifestyle history or probabilities. '
        + prefix + '\nSINGLE RESPONSE SCHEMA: ' + json.dumps(schema, ensure_ascii=False)
        + '\nSCENE HYPOTHESIS:' + suffix)


def furniture_answer(value, case, hypothesis):
    # A saved real Qwen answer used kind=social, role=social. This is an
    # unambiguous wire alias, not a reason to change inventory or probabilities.
    normalized=deepcopy(value);bindings=[]
    rows=normalized.get('requirements') if isinstance(normalized,dict) else None
    if isinstance(rows,list):
        for index,row in enumerate(rows):
            if isinstance(row,dict) and row.get('kind') in ('social','reading','work','dining'):
                if row.get('role')==row['kind']:
                    bindings.append(dict(path=f'requirements[{index}].kind',original=row['kind'],
                        normalized='activity',reason='matching_activity_role_used_as_kind'))
                    row['kind']='activity'
    parsed = needs.furniture_answer(normalized, case, hypothesis)
    for raw, clean in zip(value['requirements'], parsed[0]['requirements']):
        if raw.get('role') not in ROLES:
            raise ValueError('Every requirement needs one role from '+str(ROLES))
        clean['role'] = raw['role']
    if bindings:parsed[0]['interface_normalizations']=bindings
    return parsed


def observed_cues(case):
    """Positive anchor-zone reports only; missing detections stay unobserved.

    A cue is a coarse property of the observed anchor stratum, not an assertion
    about ALL inventory. Hidden E instances are separate conditional draws.
    """
    result = {}; unmatched = []
    for obj in case['context']:
        description = obj.get('description', '') + ' ' + obj.get('category', '')
        matched = [c for c, pattern in PATTERNS.items() if re.search(pattern, description, re.I)]
        # One anchor -> at most ONE cue factor; bookshelves must not be two
        # independent books+storage observations from the same detection.
        if matched:
            cue = matched[0]
            result.setdefault(cue, dict(value='1', source_ids=[]))['source_ids'].append(obj['id'])
        else: unmatched.append(obj['id'])
    return dict(positive_cues=result, unmatched_observed_ids=unmatched,
        missing_cues='unobserved_not_absent', repetitions='one_binary_factor_per_cue_family',
        cue_assignment='first_matching_rule_one_family_per_anchor')


def _cue_presence(p, cue, roles):
    return 1 - (1-p['cue_leak']) * np.prod([
        1-strength if roles[role] == '1' else 1.
        for role, strength in p['cue_strengths'][cue].items()])


def _report_probability(p, presence):
    d = p['detection']
    return d['false_positive'] + (1-d['false_positive']) * d['coverage'] * d['recall'] * presence


def build_model(hypotheses, programs, unused_contents=None):
    # Context and numeric parameters are bound by prepare_model, before ANY draw.
    return needs.build_model(hypotheses, programs)


def prepare_model(base, case, config, out):
    p = parameters(config.get('functional_parameters'))
    model = deepcopy(base); nodes = []
    def add(nid, states, parents, fn):
        domains = {n['id']: n['states'] for n in nodes}
        nodes.append(dict(id=nid,states=states,parents=parents,cpt={
            key(values): fn(dict(zip(parents, values)))
            for values in itertools.product(*(domains[parent] for parent in parents))}))
    for role in ROLES:
        v=p['role_priors'][role]
        add('H_'+role, ['0','1'], [], lambda g,v=v:[1-v,v])
    for cue, strengths in p['cue_strengths'].items():
        parents=['H_'+r for r in strengths]
        def law(g,cue=cue):
            v=float(_cue_presence(p,cue,{r:g.get('H_'+r,'0') for r in ROLES}))
            return [1-v,v]
        add('A_'+cue,['0','1'],parents,law)
        def detection(g,cue=cue):
            v=_report_probability(p,int(g['A_'+cue]))
            return [1-v,v]
        add('O_'+cue,['0','1'],['A_'+cue],detection)
    for node in model['nodes']:
        node=deepcopy(node)
        matching=[n for n in model['need_mechanisms'].values() if n['requirement_node']==node['id']]
        if matching:
            need=matching[0];role=need['role'];node['parents']=['Z','H_'+role];node['cpt']={}
            for z,h in itertools.product(model['nodes'][0]['states'],['0','1']):
                v=p['need_activation']['active' if h=='1' else 'inactive'] if z==need['hypothesis_id'] else 0.
                node['cpt'][key([z,h])]=[1-v,v]
        if node['id'] in model['need_mechanisms']:
            need=model['need_mechanisms'][node['id']]
            weights=p['strategy_weights'].get(node['id'])
            if weights is not None:
                if set(weights)!=set(node['states'][1:]) or any(
                        isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in weights.values()):
                    raise ValueError('strategy_weights must contain every strategy with positive finite weight')
                total=sum(weights.values());node['cpt']['1']=[0.]+[weights[s]/total for s in node['states'][1:]]
        nodes.append(node)
    unknown=set(p['strategy_weights'])-set(model['need_mechanisms'])
    if unknown: raise ValueError('Unknown strategy weight node IDs: '+str(sorted(unknown)))
    evidence=observed_cues(case)
    model.update(nodes=nodes,model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        functional_parameters=p, cue_evidence=evidence,
        generation_factorization='H -> A_f -> O_f; (H,Z) -> R_k -> B_k -> shared E_i -> S_i <- S_host',
        distribution_scope='exact_cue_posterior_in_assumed_model_conditional_on_fixed_image_proposed_catalog',
        sampling_policy='iid_functional_posterior_then_conditional_ancestral_duplicates_retained',
        cue_likelihood_defined=True, RGB_likelihood_defined=False,
        calibrated_posterior_defined=False, real_causal_structure_identified=False,
        resident_history_inferred=False, metric_mechanisms_learned=False,
        functional_sampling=config.get('functional_sampling','posterior'))
    if model['functional_sampling'] not in ('prior','posterior'):raise ValueError('Unknown functional_sampling')
    if model['functional_sampling']=='prior':
        model['sampling_policy']='iid_same_catalog_prior_ablation_duplicates_retained'
        model['distribution_scope']='same_fixed_catalog_prior_ablation_in_assumed_functional_model'
    validate(model)
    if out is not None:
        write(out/'functional_parameters.json',p);write(out/'cue_evidence.json',evidence)
        write(out/'probability_audit.json',probability_audit(model))
        probability_page(model,out/'probability_audit.html')
    return model


def role_posterior(model, use_evidence=True):
    """Exact 2**6 enumeration, including noisy-OR explaining-away effects."""
    p=model['functional_parameters'];rows=[]
    observed=model['cue_evidence']['positive_cues'] if use_evidence else {}
    for values in itertools.product(['0','1'],repeat=len(ROLES)):
        h=dict(zip(ROLES,values));prior=float(np.prod([
            p['role_priors'][r] if h[r]=='1' else 1-p['role_priors'][r] for r in ROLES]))
        likelihood=float(np.prod([_report_probability(p,_cue_presence(p,c,h)) for c in observed]))
        rows.append(dict(roles=h,prior=prior,likelihood=likelihood,joint_mass=prior*likelihood))
    mass=sum(r['joint_mass'] for r in rows)
    if mass<=0: raise ValueError('Cue evidence has zero probability')
    for row in rows: row['probability']=row['joint_mass']/mass
    return dict(rows=rows,evidence_probability=mass)


def marginals(model, use_evidence=True, do=None):
    """Exact inventory marginals; OR of independent requirement bundles given H,Z.

    do(R=0/1) is surgical: upstream H remains conditioned on factual evidence.
    This implements model-relative provisioning interventions, not removal physics.
    """
    do=do or {};byid={n['id']:n for n in model['nodes']}
    rids={n['requirement_node'] for n in model['need_mechanisms'].values()}
    if any(k not in rids or v not in ('0','1') for k,v in do.items()):
        raise ValueError('Marginal interventions support requirement nodes only')
    post=role_posterior(model,use_evidence);out={oid:0. for oid in model['component_registry']}
    roles={r:sum(row['probability'] for row in post['rows'] if row['roles'][r]=='1') for r in ROLES}
    for row in post['rows']:
        h=row['roles']
        for zi,z in enumerate(byid['Z']['states']):
            mass=row['probability']*byid['Z']['cpt']['root'][zi]
            for oid,obj in model['component_registry'].items():
                absent=1.
                for bid in dict.fromkeys(m['strategy_node'] for m in obj['requirement_memberships']):
                    need=model['need_mechanisms'][bid];rid=need['requirement_node']
                    pr=float(do[rid]) if rid in do else byid[rid]['cpt'][key([z,h[need['role']]])][1]
                    wanted={m['choice'] for m in obj['requirement_memberships'] if m['strategy_node']==bid}
                    pb=sum(byid[bid]['cpt']['1'][i] for i,s in enumerate(byid[bid]['states']) if s in wanted)
                    absent*=1-pr*pb
                out[oid]+=mass*(1-absent)
    counts={}
    for oid,v in out.items():
        category=model['component_registry'][oid]['category']
        counts[category]=counts.get(category,0.)+v
    return dict(role_presence=roles,object_presence=out,category_expected_counts=counts,
        expected_hidden_instances=sum(out.values()),do=do,evidence_probability=post['evidence_probability'])


def sample_worlds(model, seed=3407, count=3, use_evidence=None):
    if type(count) is not int or count<1: raise ValueError('count must be positive')
    if use_evidence is None:use_evidence=model.get('functional_sampling','posterior')=='posterior'
    post=role_posterior(model,use_evidence);rng=np.random.default_rng(seed);worlds=[]
    presence=marginals(model,use_evidence)['object_presence']
    byid={n['id']:n for n in model['nodes']};p=model['functional_parameters']
    for trial in range(count):
        selected=int(rng.choice(len(post['rows']),p=[r['probability'] for r in post['rows']]))
        h=post['rows'][selected]['roles'];noise={n['id']:float(rng.random()) for n in model['nodes']}
        def bind(nid,state,parents):
            node=byid[nid];ps=node['cpt'][key(parents)];i=node['states'].index(state)
            noise[nid]=min(float(sum(ps[:i])+ps[i]*rng.random()),float(np.nextafter(1.,0.)))
        for r in ROLES: bind('H_'+r,h[r],[])
        if use_evidence:
            for cue in model['cue_evidence']['positive_cues']:
                pa=_cue_presence(p,cue,h)
                pa_given_o=pa*_report_probability(p,1)/_report_probability(p,pa)
                a='1' if rng.random()<pa_given_o else '0'
                bind('A_'+cue,a,[h[r] for r in p['cue_strengths'][cue]])
                bind('O_'+cue,'1',[a])
        state=forward(model,noise);world=active_world(model,state)
        for obj in world['objects']:obj['presence_probability']=presence[obj['id']]
        world.update(noise=noise,seed=seed,trial=trial,functional_state=h,
            sampling_distribution='cue_posterior' if use_evidence else 'same_catalog_prior',
            probability_scope=model['distribution_scope'])
        worlds.append(world)
    return worlds


def probability_audit(model):
    prior=marginals(model,False);posterior=marginals(model)
    interventions={n['requirement_node']:marginals(model,do={n['requirement_node']:'0'})
        for n in model['need_mechanisms'].values()}
    return dict(prior=prior,posterior=posterior,do_requirement_off=interventions,
        posterior_total=sum(r['probability'] for r in role_posterior(model)['rows']),
        same_catalog_for_all_comparisons=True,explanatory_text_used_as_likelihood=False,
        parameter_provenance=model['functional_parameters']['provenance'],probabilities_calibrated=False,
        probability_mass_defined_only_on_fixed_catalog=True)


def probability_page(model,path):
    audit=probability_audit(model);prior=audit['prior'];post=audit['posterior']
    rows=[]
    for oid,obj in model['component_registry'].items():
        a=prior['object_presence'][oid];b=post['object_presence'][oid]
        rows.append('<tr><td>'+escape(oid+' / '+obj['category'])+'</td><td>'+f'{a:.4f}'+'</td><td>'+f'{b:.4f}'
            +'</td><td>'+f'{b-a:+.4f}'+'</td><td><meter min="0" max="1" value="'+str(b)+'"></meter></td></tr>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>功能后验与真实采样概率</title>'
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}td,th{padding:9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
        '<h1>固定候选目录：先验 → 观测更新 → 隐藏实例生成概率</h1>'
        '<p>本分支实际采样：'+escape(model.get('functional_sampling','posterior'))+'。下表并列展示两种分布。</p>'
        '<p>这些概率进入 structural_samples.json 的实际采样。重复样本保留；布局失败不会重新采样或改写原概率。'
        '参数来源见记录，概率尚未完成校准验证。这里只使用正观测线索；没有住户生活史变量。</p>'
        '<p><a href="probability_audit.json">完整数值及逐需求 do(R=0) 消融</a> · '
        '<a href="functional_parameters.json">全部概率参数</a> · <a href="cue_evidence.json">证据来源</a></p>'
        '<table><tr><th>共享实例</th><th>先验</th><th>后验</th><th>变化</th><th>后验概率</th></tr>'+''.join(rows)
        +'</table><h2>多用途功能状态边缘概率</h2><pre>'+escape(json.dumps(dict(prior=prior['role_presence'],
            posterior=post['role_presence']),ensure_ascii=False,indent=2))+'</pre>')
