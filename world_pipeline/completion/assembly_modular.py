"""Functional assembly grammar with empirical, parent-restricted mechanisms.

One optional joint assembly per proposed role. No global empty-vs-full-world
contest, biography variable, calibrated image likelihood, or count reward.
Conditional independence of assembly choices GIVEN H is an explicit assumption.
"""
from collections import Counter
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path

import numpy as np

from ..common import read,write,sha
from . import assembly_sdscene as sd,assembly_delta as delta,assembly_population as population
from .world_scm import validate,forward

PROTOCOL=sd.PROTOCOL
INTERFACE='functional_assembly_grammar_v5'
MODEL_VARIANT='empirical_functional_assembly_scm_v5'
FUNCTIONS=tuple(x for x in sd.FUNCTIONS if x!='unknown')
semantic_diagnostics=delta.semantic_diagnostics
build_model=population.build_model
hypotheses_answer=sd.hypotheses_answer
load_domain=sd.load_domain


def profile(identifier,functions):
    """An intervention value contains NO scene caption or object inventory."""
    tags=sorted(set(functions))
    if any(t not in FUNCTIONS for t in tags):raise ValueError('Unknown functional tag')
    return dict(id=identifier,functions=tags,description='Functional affordances: '+(', '.join(tags) or 'unspecified'),
        assumptions=['Possible uses, not actual activity, biography, object ownership, or room extent.'])


def hypotheses_prompt(case,grounding):
    return (sd.hypotheses_prompt(case,grounding)+'\nThe executor will discard captions and inventory guesses from H. '
        'Only the canonical function tags define its value. Functions describe possible activities/affordances '
        'of the surrounding space, not confirmed hidden objects or unmet requirements.')


def furniture_prompt(case,hypothesis,channel='surrounding'):
    return (sd.furniture_prompt(case,profile(hypothesis['id'],hypothesis['functions']),channel)+
        '\nFUNCTIONAL ASSEMBLY CONTRACT v5: Each group is ONE optional, self-contained functional assembly, '
        'including furniture AND its ordinary belongings together. group.function must be exactly ONE tag from '+json.dumps(FUNCTIONS)+'. '
        'Use background for incidental objects, lighting or display not specific to another use. '
        'All predicted hosts must be in the SAME group as their descendants. Relations between new objects '
        'must stay within that group; relations to fixed obs IDs are allowed. Multiple groups of the same function '
        'will be interpreted as alternatives, not automatically combined. Put a multi-object activity arrangement '
        'in one group if those instances must coexist. Do not use placeholder function sentences. '
        'Side/coffee/dining tables, chairs, sofas, desks and floor lamps cannot be installed on a sofa seat, shelf or tabletop. '
        'above/below are original camera Y. For camera-depth ordering use farther_than (greater Z) or nearer_than (smaller Z), '
        'rather than ambiguous in_front_of/behind. These are spatial constraints, not causal edges. '
        'An existing visible activity does not exhaust possible unseen belongings. There is still no minimum object count.')


def furniture_answer(value,case,hypothesis):
    value=deepcopy(value)
    for alternative in value.get('alternatives',[]):
        for group in alternative.get('groups',[]):
            if group.get('function') not in FUNCTIONS:raise ValueError('group.function must be one exact canonical function tag')
        for relation in alternative.get('relations',[]):
            kind=relation.get('kind')
            if kind in ('in_front_of','behind'):raise ValueError('Use explicit camera farther_than/nearer_than for depth ordering')
            relation['kind']={'farther_than':'in_front_of','nearer_than':'behind'}.get(kind,kind)
    programs=sd.furniture_answer(value,case,profile(hypothesis['id'],hypothesis['functions']))
    rooted={'table','side table','coffee table','dining table','chair','armchair','sofa','desk','floor lamp','ottoman'}
    for raw,program in zip(value['alternatives'],programs):
        declared={f'g{i}':g for i,g in enumerate(raw['groups'])}
        membership={o['key']:o['semantic_group'] for g in program['groups'] for o in g['objects']}
        rebuilt={}
        for group in program['groups']:
            for obj in group['objects']:
                host=obj['attachment']['host'];noun=obj['category'].lower().replace('_',' ').strip()
                if noun in rooted and obj['attachment']['kind'] in ('top','seat','shelf','hang'):
                    raise ValueError(obj['key']+': ordinary '+noun+' cannot be installed on '+str(host)+' with '+obj['attachment']['kind'])
                if host in membership and membership[host]!=membership[obj['key']]:raise ValueError('All host descendants must stay in one functional assembly')
                assembly=membership[obj['key']];gid=assembly+('_obs' if group['key']=='g_obs' else '')
                if gid not in rebuilt:
                    g=declared[assembly];rebuilt[gid]=dict(key=gid,function=g['function'],reason=g['reason'],
                        observed_refs=list(g.get('observed_refs',[])),objects=[],assembly_key=assembly)
                obj['group']=gid;rebuilt[gid]['objects'].append(obj)
        for r in program['relations']:
            if r['source'] not in membership and r['target'] not in membership:raise ValueError('New relations must involve a proposed instance')
            if r['source'] in membership and r['target'] in membership and membership[r['source']]!=membership[r['target']]:
                raise ValueError('Cross-group dependencies must be placed in ONE joint assembly')
        program['groups']=list(rebuilt.values())
    return programs


def mechanism_prompt(domain,h,role,order):
    menu=[]
    for state in order:
        if state=='none':menu.append(dict(id=state,meaning='No additional unseen assembly of this role in this world.'))
        else:
            p=domain['programs'][state]
            menu.append(dict(id=state,objects=[dict(key=o['key'],category=o['category'],variant=o['variant'],attachment=o['attachment'])
                for g in p['groups'] for o in g['objects']],relations=p['relations']))
    return ('Draw ONE possible existing unseen functional assembly, conditioned only on CURRENT_PARENT_H and fixed domain D. '
        'This is generative sampling of an inhabited world, not a test of what has been visually proved and not a minimum furniture design. '
        'The photograph is a partial view. Lack of visual evidence for an unseen item is not evidence of its absence. '
        'An already supported activity may still have additional belongings or another area; no new object is required. '
        'Choose none when sampling a world without an extra assembly of this role. Compare ONLY alternatives within this role. '
        'Do not invent objects, explanations, scores or a different ID. Return exactly {"choice":"allowed ID"}. '
        '\nCURRENT_PARENT_H: '+json.dumps(profile(h['id'],h['functions']))+
        '\nFIXED_ROLE: '+role+'\nCOMMON_INVENTORY_IN_EVERY_WORLD: '+json.dumps(domain['common_inventory'],ensure_ascii=False)+
        '\nALLOWED_VALUE_DOMAIN: '+json.dumps(menu,ensure_ascii=False))


def recognition_prompt(domain,case):
    return sd.recognition_prompt(dict(domain,profiles=[profile(h['id'],h['functions']) for h in domain['profiles']]),case)


def prepare_model(base,case,config,out):
    bundle=Path(config['sdscene_bundle']);domain=read(bundle/'domain.json');source=config['mechanism_source']
    if source=='per_backend':source=config['execution_backend']
    table=read(bundle/'mechanisms'/(source+'.json'))
    if table['domain_sha256']!=sha(bundle/'domain.json'):raise ValueError('Frozen assembly mechanism/domain hash mismatch')
    model=deepcopy(base);profiles=domain['profiles'];hs=[h['id'] for h in profiles];slots=domain['slots'];policy=config.get('modular_policy','functional')
    if policy not in ('functional','function_free'):raise ValueError('Unknown assembly policy')
    if set(table['recognition']['probabilities'])!=set(hs):raise ValueError('Profile domain differs')
    nodes=[dict(id='H',parents=[],states=hs,cpt={'root':[table['recognition']['probabilities'][h] for h in hs]})]
    state_slot={};rows={}
    for role,states in slots.items():
        cpt={}
        for h in hs:
            raw=table['mechanisms'][h+'|'+role]['probabilities']
            if set(raw)!=set(states):raise ValueError('Assembly state domain differs')
            cpt[h]=[raw[s] for s in states]
        if policy=='function_free':
            average=np.mean(list(cpt.values()),axis=0).tolist();cpt={h:average[:] for h in hs}
        nodes.append(dict(id='M_'+role,parents=[] if policy=='function_free' else ['H'],states=states,
            cpt={'root':average[:]} if policy=='function_free' else cpt));rows[role]=cpt
        state_slot.update({s:role for s in states if s!='none'})
    for node in base['nodes']:
        if node['id'].startswith('E_'):
            obj=next(o for o in model['component_registry'].values() if o['presence_node']==node['id'])
            fid=obj['furniture_state'];role=state_slot[fid];states=slots[role]
            nodes.append(dict(id=node['id'],parents=['M_'+role],states=['0','1'],
                cpt={s:[float(s!=fid),float(s==fid)] for s in states}))
        elif node['id'].startswith('S_'):nodes.append(deepcopy(node))
    registry=model['component_registry'];maximum=sum(max(sum(len(g['objects']) for g in domain['programs'][s]['groups'])
        for s in states if s!='none') for states in slots.values())
    model.update(nodes=nodes,profiles=profiles,slots=slots,state_slot=state_slot,modular_policy=policy,
        model_variant=MODEL_VARIANT,response_interface=INTERFACE,mechanism_source=source,empirical_tables=table,
        observed_counts=population.counts(case['context'],True)[0],assembly_rows=rows,maximum_hidden_instances=maximum,
        distribution_scope='conditional_functional_assembly_grammar_given_fixed_observation_and_proposal_domain',
        sampling_policy='IID_all_samples_retained_including_empty_and_duplicates',
        generation_factorization=('q(H|O,D) product_role mean_profile p(M_role|profile,D); M->E->S<-S_host; (O,H,M)->X' if policy=='function_free' else
            'q(H|O,D) product_role p(M_role|H,D); M->E->S<-S_host; (O,H,M)->X'),
        conditional_independence_assumed='Role assemblies independent given H and fixed D; not identified from data',
        count_likelihood_defined=False,RGB_likelihood_defined=False,calibrated_posterior_defined=False,
        candidate_support=dict(candidate_instances=len(registry),nonempty_programs=len(state_slot),all_nonempty_proposals_missing=not registry))
    validate(model)
    if out is not None:
        write(out/'empirical_mechanisms.json',table);write(out/'probability_audit.json',marginals(model))
        write(out/'functional_probability_interventions.json',{h:marginals(model,h) for h in hs})
        probability_page(model,out/'probability_audit.html')
    return model


def marginals(model,do_profile=None):
    hnode=model['nodes'][0];hs=hnode['states']
    if do_profile is not None and do_profile not in hs:raise ValueError('Unknown profile')
    hp={h:float(h==do_profile) if do_profile is not None else hnode['cpt']['root'][i] for i,h in enumerate(hs)}
    assembly={};objects={};categories=set();by_state={}
    for fid,role in model['state_slot'].items():
        counts=Counter(o['category'].lower().replace('_',' ').strip() for g in model['furniture_states'][fid]['groups'] for o in g['objects'])
        by_state[fid]=counts;categories.update(counts)
    count_pmf={};cp={c:0. for c in categories};ce={c:0. for c in categories}
    for h in hs:
        pmf={0:1.};absence={c:1. for c in categories}
        for role,states in model['slots'].items():
            row=model['assembly_rows'][role][h];local={};local_cp={c:0. for c in categories}
            for s,p in zip(states,row):
                counts=by_state.get(s,{});n=sum(counts.values());local[n]=local.get(n,0.)+p
                assembly[s if s!='none' else 'none_'+role]=assembly.get(s if s!='none' else 'none_'+role,0.)+hp[h]*p
                for c,nc in counts.items():local_cp[c]+=p;ce[c]+=hp[h]*p*nc
            for c in categories:absence[c]*=1-local_cp[c]
            next_pmf={}
            for a,p in pmf.items():
                for b,q in local.items():next_pmf[a+b]=next_pmf.get(a+b,0.)+p*q
            pmf=next_pmf
        for c in categories:cp[c]+=hp[h]*(1-absence[c])
        for n,p in pmf.items():count_pmf[str(n)]=count_pmf.get(str(n),0.)+hp[h]*p
    for oid,obj in model['component_registry'].items():objects[oid]=assembly[obj['furniture_state']]
    return dict(assembly_probabilities=assembly,object_presence=objects,category_presence=cp,expected_category_counts=ce,
        profile_probabilities=hp,hidden_count_probabilities=count_pmf,empty_probability=count_pmf.get('0',0.),
        expected_hidden_instances=sum(objects.values()),do_profile=do_profile,probabilities_calibrated=False,
        probability_scope=model['distribution_scope'],conditional_independence_assumed=model['conditional_independence_assumed'])


def active_world(model,state):
    objects=[];groups=[];relations=[];descriptions=[]
    for role in model['slots']:
        fid=state['M_'+role]
        if fid=='none':continue
        program=model['furniture_states'][fid];local=[deepcopy(o) for o in model['component_registry'].values()
            if o['furniture_state']==fid and state[o['presence_node']]=='1'];mapping={o['key']:o['id'] for o in local}
        for obj in local:
            obj['group']=fid+'_'+obj['group'];obj['installation_ready']=state[obj['installation_node']]=='1'
        for group in program['groups']:groups.append(dict(deepcopy(group),key=fid+'_'+group['key']))
        available=set(mapping)|set(mapping.values());keys={o['key'] for o in model['component_registry'].values() if o['furniture_state']==fid}
        for r in program['relations']:
            if all(r[k] not in keys or r[k] in available for k in ('source','target')):
                relations.append(dict(r,source=mapping.get(r['source'],r['source']),target=mapping.get(r['target'],r['target'])))
        objects+=local;descriptions.append(program['description'])
    # A child inheriting g_obs needs a group even if a parser version did not
    # declare one. No object is dropped or attached to a different host.
    gids={g['key'] for g in groups}
    for obj in objects:
        if obj['group'] not in gids:groups.append(dict(key=obj['group'],function='background',objects=[]));gids.add(obj['group'])
    return dict(configuration='; '.join(descriptions) or 'Observed inventory only in this sampled world.',
        hypothesis=next(deepcopy(h) for h in model['profiles'] if h['id']==state['H']),objects=objects,groups=groups,
        relations=relations,structural_state=deepcopy(state),unresolved_installations=[o['id'] for o in objects if not o['installation_ready']])


def sample_worlds(model,seed=3407,count=4):
    if type(count) is not int or count<1:raise ValueError('count must be positive')
    rng=np.random.default_rng(seed);law=marginals(model);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in model['nodes']};world=active_world(model,forward(model,noise))
        for obj in world['objects']:obj['presence_probability']=law['object_presence'][obj['id']]
        world.update(noise=noise,seed=seed,trial=trial,mechanism_source=model['mechanism_source'],probability_scope=model['distribution_scope'])
        worlds.append(world)
    return worlds


def intervention_diagnostics(model,world):
    rows=[];before={o['id'] for o in world['objects']}
    for obj in world['objects']:
        if not any(o['attachment']['host']==obj['id'] for o in world['objects']):continue
        do={obj['presence_node']:'0'};after=active_world(model,forward(model,world['noise'],do));ids={o['id'] for o in after['objects']}
        rows.append(dict(do=do,removed=sorted(before-ids),unresolved_installations=after['unresolved_installations'],
            same_exogenous_noise=True,model_called=False,geometry_not_resolved=True))
    functions=[]
    for h in model['profiles']:
        after=active_world(model,forward(model,world['noise'],{'H':h['id']}))
        functions.append(dict(do={'H':h['id']},after_inventory=[o['id'] for o in after['objects']],same_exogenous_noise=True,
            model_called=False,geometry_not_resolved=True))
    return dict(queries=rows,functional_queries=functions,scope='assumed_model_relative_interventions_not_identified_real_world_effects',causal_gain_validated=False)


def probability_page(model,path):
    law=marginals(model);rows=[]
    for role,states in model['slots'].items():
        for h in model['profiles']:
            probs=dict(zip(states,model['assembly_rows'][role][h['id']]))
            rows.append('<tr><td>'+escape(role)+'</td><td>'+escape(h['id'])+'</td><td>'+escape(json.dumps(probs))+'</td></tr>')
    path.write_text('<!doctype html><meta charset="utf-8"><h1>功能组合 → 可选联合装配 → 实例存在</h1>'
        '<p>固定候选域下的经验生成概率；未校准为真实场景概率。无最低新增数。角色间条件独立是假设。</p>'
        '<p>策略：'+escape(model['modular_policy'])+'；P(空世界)='+f'{law["empty_probability"]:.4f}'+
        '；期望新增数='+f'{law["expected_hidden_instances"]:.3f}'+'</p>'
        '<p><a href="empirical_mechanisms.json">原始机制</a> · <a href="probability_audit.json">实例/类别/数量概率</a> · '
        '<a href="functional_probability_interventions.json">功能干预</a></p>'
        '<table border="1"><tr><th>角色</th><th>H</th><th>实际条件采样行</th></tr>'+''.join(rows)+'</table>')
