"""SD-SCM-inspired conditional scene generator with empirical LM mechanisms.

Recognition q(H|O,D) and parent-only p(F|H,D) are estimated from fixed-budget
categorical draws. They are NOT token likelihoods or calibrated visual posteriors.
No hand-elicited count-density ratio or duplicated image likelihood is applied.
"""
from copy import deepcopy
from html import escape
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from ..common import read,write,sha
from . import assembly_population as population,assembly_delta as delta,assembly_inventory as inventory
from .world_scm import validate,forward,key

PROTOCOL=delta.PROTOCOL
INTERFACE='empirical_parent_only_scene_mechanisms_v4'
MODEL_VARIANT='sdscene_conditional_mechanisms_v4'
FUNCTIONS=population.ROLES+('sleep','cooking','hygiene','entry','exercise','unknown')
BROAD={'furniture','lighting','decor','decoration','object','fixture','everyday belonging'}
semantic_diagnostics=delta.semantic_diagnostics
build_model=population.build_model


def hypotheses_prompt(case,grounding):
    return (population.hypotheses_prompt(case,grounding)+'\nFUNCTION TAG CONTRACT: functions is a list '
        'of exact tags from '+json.dumps(FUNCTIONS)+'. Use [] for genuinely uncommitted use. '
        'Tags describe uncertain scene functions, not residents or actual current activities. '
        'Do not copy placeholder text such as specific use or coexisting activities.')


def hypotheses_answer(value):
    rows=delta.hypotheses_answer(value)
    for row in rows:
        if any(tag not in FUNCTIONS for tag in row['functions']):raise ValueError('functions must use exact canonical tags '+str(FUNCTIONS))
        row['functions']=sorted(set(row['functions'])-{'unknown'})
    return rows


def furniture_prompt(case,hypothesis,channel='surrounding'):
    return (population.furniture_prompt(case,hypothesis,channel)+
        '\nEvery category must be a concrete object noun, never furniture/lighting/decor. '
        'A freestanding or floor-standing variant cannot have attachment.kind=wall/ceiling. '
        'A wall-mounted variant must actually declare wall attachment; proximity to a wall is not mounting. '
        'Retain uncertain ordinary possessions. Do not add objects just to reach a count.')


def furniture_answer(value,case,hypothesis):
    programs=delta.furniture_answer(value,case,hypothesis)
    for program in programs:
        for group in program['groups']:
            for obj in group['objects']:
                if obj['category'].strip().lower() in BROAD:
                    raise ValueError(obj['key']+': category needs a concrete noun, not '+obj['category'])
                kind=obj['attachment']['kind'];variant=obj['variant'].lower()
                if kind in ('wall','ceiling') and re.search(r'\b(freestanding|floor[- ]standing)\b',variant):
                    raise ValueError(obj['key']+': freestanding variant contradicts mounting attachment')
                if re.search(r'\bwall[- ]mounted\b',variant) and kind!='wall':
                    raise ValueError(obj['key']+': wall-mounted variant needs an explicit wall attachment')
    return programs


def load_domain(case,config):
    bundle=Path(config['sdscene_bundle']);path=bundle/'domain.json';domain=read(path)
    if sha(Path(case['image']))!=domain['RGB_sha256'] or sha(Path(case['state']))!=domain['G_obs_sha256']:
        raise ValueError('Shared scene domain RGB/G_obs mismatch')
    if case['hfov']!=domain['hfov'] or case['image_size']!=domain['image_size']:
        raise ValueError('Shared scene domain camera mismatch')
    return domain['hypotheses'],domain['programs'],dict(source=str(path),sha256=sha(path),
        scope='shared_union_of_revalidated_model_proposals',old_geometry_reused=False)


def _program_menu(base,order):
    result=[]
    for fid in order:
        program=base['furniture_states'][fid]
        result.append(dict(id=fid,objects=[dict(key=o['key'],category=o['category'],variant=o['variant'],
            attachment=o['attachment']) for g in program['groups'] for o in g['objects']],
            groups=[g['function'] for g in program['groups']],relations=program['relations']))
    return result


def choice_answer(value,allowed):
    if not isinstance(value,dict) or set(value)!={'choice'} or value['choice'] not in allowed:
        raise ValueError('Return exactly {"choice": one exact allowed state ID}; allowed='+str(allowed))
    return value['choice']


def mechanism_prompt(domain,profile,order):
    # D is a fixed value-domain definition. No actual image/report, earlier
    # votes, backend name, or other latent node value enters this mechanism.
    return ('Sample ONE complete-world inventory state from the ALLOWED VALUE DOMAIN, conditioned only on '
        'CURRENT_PARENT_H and the fixed domain definition. This is a stochastic structural mechanism, '
        'not a request for confidence scores or an explanation. Evaluate whole inventory compatibility; '
        'visible equipment sufficiency does not imply no other possessions. The null inventory is allowed. '
        'Do not change any state or invent a state ID. Return exactly {"choice":"allowed ID"}. '
        '\nCURRENT_PARENT_H: '+json.dumps(profile,ensure_ascii=False)+
        '\nCOMMON_INVENTORY_IN_EVERY_ALLOWED_WORLD: '+json.dumps(domain['common_inventory'],ensure_ascii=False)+
        '\nALLOWED_VALUE_DOMAIN: '+json.dumps(_program_menu(domain['base_model'],order),ensure_ascii=False))


def recognition_prompt(domain,case):
    return ('From the attached local observation, sample ONE plausible latent functional profile from this fixed '
        'domain. Several profiles may fit. Do not infer biography or actual current activity. '
        'Return exactly {"choice":"profile ID"}; no confidence scores. '
        '\nPROFILE_DOMAIN: '+json.dumps(domain['profiles'],ensure_ascii=False)+
        '\nOBSERVED_ANCHORS: '+json.dumps(case['context'],ensure_ascii=False))


def estimate_table(draws,states,concentration=1.):
    if not math.isfinite(concentration) or concentration<=0:raise ValueError('Dirichlet concentration must be positive')
    valid=[v for v in draws if v in states]
    if not valid:raise ValueError('No valid categorical draws; mechanism is unavailable, not a uniform fallback')
    counts={s:valid.count(s) for s in states};denominator=len(valid)+concentration
    probabilities={s:(counts[s]+concentration/len(states))/denominator for s in states}
    return dict(counts=counts,valid_draws=len(valid),invalid_draws=len(draws)-len(valid),
        probabilities=probabilities,concentration=concentration,
        posterior_std_assuming_multinomial={s:math.sqrt(p*(1-p)/(denominator+1)) for s,p in probabilities.items()},
        scope='Dirichlet_predictive_of_valid_constrained_generation_channel_not_token_probabilities')


def pool_tables(tables,domain_sha):
    if any(t['domain_sha256']!=domain_sha for t in tables):raise ValueError('Mechanism domain changed')
    concentration=tables[0]['concentration']
    if any(t['concentration']!=concentration for t in tables):raise ValueError('Mechanism smoothing differs')
    def pooled(rows):
        states=list(rows[0]['counts']);draws=[]
        for row in rows:
            if set(row['counts'])!=set(states):raise ValueError('Mechanism states differ')
            for s,n in row['counts'].items():draws.extend([s]*n)
            draws.extend([None]*row['invalid_draws'])
        return estimate_table(draws,states,concentration)
    return dict(domain_sha256=domain_sha,concentration=concentration,source='pooled_for_geometry_control',
        recognition=pooled([t['recognition'] for t in tables]),
        mechanisms={h:pooled([t['mechanisms'][h] for t in tables]) for h in tables[0]['mechanisms']})


def prepare_model(base,case,config,out):
    bundle=Path(config['sdscene_bundle']);domain=read(bundle/'domain.json')
    source=config.get('mechanism_source','per_backend')
    if source=='per_backend':source=config['execution_backend']
    path=bundle/'mechanisms'/(source+'.json');table=read(path)
    if table['domain_sha256']!=sha(bundle/'domain.json'):raise ValueError('Frozen mechanism/domain hash mismatch')
    model=deepcopy(base);profiles=domain['profiles'];hstates=[h['id'] for h in profiles];fstates=list(model['furniture_states'])
    if set(table['recognition']['probabilities'])!=set(hstates) or set(table['mechanisms'])!=set(hstates):
        raise ValueError('Empirical profile domain differs')
    nodes=[dict(id='H',states=hstates,parents=[],cpt={'root':[table['recognition']['probabilities'][h] for h in hstates]})]
    fcpt={}
    for h in hstates:
        probs=table['mechanisms'][h]['probabilities']
        if set(probs)!=set(fstates):raise ValueError('Empirical inventory domain differs')
        fcpt[h]=[probs[f] for f in fstates]
    nodes.append(dict(id='F',states=fstates,parents=['H'],cpt=fcpt))
    observed,bindings=population.counts(case['context'],True);program_counts={};count_states={};lookup={};mapping={}
    for fid,program in model['furniture_states'].items():
        hidden,_=population.counts([o for g in program['groups'] for o in g['objects']]);full={k:observed[k]+hidden[k] for k in population.FAMILIES}
        vector=tuple(full.values())
        if vector not in lookup:lookup[vector]='n'+str(len(lookup));count_states[lookup[vector]]=full
        mapping[fid]=lookup[vector];program_counts[fid]=dict(full=full,hidden=hidden)
    nodes.append(dict(id='N_population',states=list(count_states),parents=['F'],cpt={fid:[float(s==mapping[fid]) for s in count_states] for fid in fstates}))
    nodes.extend(n for n in model['nodes'] if n['id'] not in ('Z','F'))
    model.update(nodes=nodes,profiles=profiles,model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        empirical_tables=table,mechanism_source=source,observed_counts=observed,program_counts=program_counts,
        count_states=count_states,program_count_state=mapping,scene_program_states=deepcopy(model['furniture_states']),
        distribution_scope='empirical_mechanism_constrained_conditional_scene_generator_given_fixed_observation_and_domain',
        sampling_policy='iid_empirical_mechanism_samples_duplicates_and_null_retained',
        generation_factorization='recognition q(H|O,D); conditional generation H -> F -> E/S,N; (O,H,F) -> metric layout',
        recognition_scope='amortized_functional_inference_not_identified_generative_image_posterior',
        RGB_likelihood_defined=False,count_likelihood_defined=False,calibrated_posterior_defined=False,
        real_causal_structure_identified=False,metric_mechanisms_learned=False,
        candidate_support=dict(candidate_instances=len(model['component_registry']),
            nonempty_programs=sum(any(g['objects'] for g in p['groups']) for p in model['furniture_states'].values()),
            all_nonempty_proposals_missing=not bool(model['component_registry'])))
    validate(model)
    if out is not None:
        write(out/'empirical_mechanisms.json',table);write(out/'count_observations.json',dict(counts=observed,bindings=bindings,
            used_as_additional_likelihood=False))
        audit=marginals(model);write(out/'probability_audit.json',audit)
        probability_page(model,audit,out/'probability_audit.html')
    return model


def marginals(model,do_profile=None):
    nodes={n['id']:n for n in model['nodes']};h=nodes['H'];f=nodes['F']
    if do_profile is not None and do_profile not in h['states']:raise ValueError('Unknown functional profile')
    programs={fid:sum((float(profile==do_profile) if do_profile is not None else h['cpt']['root'][i])*f['cpt'][profile][j]
        for i,profile in enumerate(h['states'])) for j,fid in enumerate(f['states'])}
    objects={oid:programs[o['furniture_state']] for oid,o in model['component_registry'].items()};count_pmf={}
    category_presence={};category_counts={}
    for fid,p in programs.items():
        n=str(sum(model['program_counts'][fid]['hidden'].values()));count_pmf[n]=count_pmf.get(n,0.)+p
        counts={}
        for group in model['furniture_states'][fid]['groups']:
            for obj in group['objects']:
                category=obj['category'].strip().lower().replace('_',' ')
                counts[category]=counts.get(category,0)+1
        for category,n in counts.items():
            category_presence[category]=category_presence.get(category,0.)+p
            category_counts[category]=category_counts.get(category,0.)+p*n
    return dict(program_probabilities=programs,object_presence=objects,hidden_count_probabilities=count_pmf,
        category_presence=category_presence,expected_category_counts=category_counts,
        category_scope='exact_declared_noun_normalized_for_case_and_underscores_not_cross_category_synonym_merging',
        expected_hidden_instances=sum(objects.values()),empty_probability=count_pmf.get('0',0.),
        profile_probabilities={s:float(s==do_profile) if do_profile is not None else h['cpt']['root'][i]
            for i,s in enumerate(h['states'])},do_profile=do_profile,
        probability_scope=model['distribution_scope'],probabilities_calibrated=False)


def active_world(model,state):
    world=inventory.active_world(model,state);profile=next(h for h in model['profiles'] if h['id']==state['H'])
    world['inventory_hypothesis']=world['hypothesis'];world['hypothesis']=deepcopy(profile)
    return world


def sample_worlds(model,seed=3407,count=3):
    if type(count) is not int or count<1:raise ValueError('count must be positive')
    rng=np.random.default_rng(seed);law=marginals(model);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in model['nodes']};state=forward(model,noise);world=active_world(model,state)
        for obj in world['objects']:obj['presence_probability']=law['object_presence'][obj['id']]
        world.update(noise=noise,seed=seed,trial=trial,program_probability=law['program_probabilities'][state['F']],
            probability_scope=model['distribution_scope'],mechanism_source=model['mechanism_source'])
        worlds.append(world)
    return worlds


def intervention_diagnostics(model,world):
    report=inventory.intervention_diagnostics(model,world);queries=[]
    for profile in model['profiles']:
        after=active_world(model,forward(model,world['noise'],{'H':profile['id']}))
        queries.append(dict(do={'H':profile['id']},after_program=after['structural_state']['F'],
            after_inventory=[o['id'] for o in after['objects']],same_exogenous_noise=True,model_called=False,
            geometry_not_resolved=True,scope='model_relative_latent_organization_sensitivity_not_real_historical_counterfactual'))
    report['functional_queries']=queries;return report


def probability_page(model,audit,path):
    rows=[]
    for fid,program in model['furniture_states'].items():
        rows.append('<tr><td>'+escape(fid+' '+program['description'])+'</td><td>'+str(sum(model['program_counts'][fid]['hidden'].values()))+
            '</td><td>'+f'{audit["program_probabilities"][fid]:.5f}'+'</td></tr>')
    interventions={h['id']:marginals(model,h['id']) for h in model['profiles']}
    write(path.with_name('functional_probability_interventions.json'),interventions)
    profile_rows=[];fids=list(model['furniture_states'])
    for h in model['profiles']:
        row=model['empirical_tables']['mechanisms'][h['id']]
        profile_rows.append('<tr><td>'+escape(h['id']+' '+', '.join(h['functions']))+'</td><td>'+
            f'{audit["profile_probabilities"][h["id"]]:.4f}'+'</td><td>'+str(row['valid_draws'])+'/'+str(row['invalid_draws'])+
            '</td>'+''.join('<td>'+f'{row["probabilities"][f]:.4f}'+'</td>' for f in fids)+'</tr>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>经验语言机制概率</title>'
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}td,th{padding:9px;border-bottom:1px solid #ddd}</style>'
        '<h1>父变量约束的语言机制：实际库存概率</h1><p>机制来源：'+escape(model['mechanism_source'])+
        '。这些是固定候选域、固定采样协议下的经验概率，不是 token 概率或已校准的真实场景置信度。</p>'
        '<p><a href="empirical_mechanisms.json">选择计数、无效率与机制表</a> · '
        '<a href="functional_probability_interventions.json">逐功能组织干预概率</a> · <a href="probability_audit.json">完整概率</a></p>'
        '<p>以下每行直接用于 F 的条件采样；修改 H 会切换这一行。行一致时功能变量不影响库存概率，必须如实报告。'
        '小样本差异也可能来自采样误差，不能直接认定为真实因果效应。</p>'
        '<table><tr><th>H</th><th>q(H|观测)</th><th>有效/无效抽样</th>'+''.join('<th>'+escape(f)+'</th>' for f in fids)+
        '</tr>'+''.join(profile_rows)+'</table>'
        '<table><tr><th>联合库存</th><th>新增数</th><th>实际生成概率</th></tr>'+''.join(rows)+'</table>')
