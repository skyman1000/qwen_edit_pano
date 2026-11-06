"""Fresh v7 proposals/mechanisms and matched Qwen/SiliconFlow experiments."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,write,sha
from . import assembly_realization as scm,assembly_activation_experiment as v6,assembly_sdscene as sd
from . import assembly_sdscene_experiment as previous
from .assembly_experiment import initialize,check,stage_seed
from .fresh_pipeline import provider


def prepare(root,source,seed=3407,worlds=4,draws=4):
    root=previous.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='realization')
    c=read(root/'context/experiment.json')
    initialize(root/'function_free',root/'context/input.png',c['hfov'],Path(c['local_model']),c['api_model'],seed,
        observations_from=root/'context',shared_observer='qwen',completion_model='realization',worlds=worlds)
    for label in ('context','end_to_end','layout_control','function_free'):
        path=root/label/'experiment.json';config=read(path)
        config.update(sampling_law='joint_anchored_extension_and_new_group_channels',stage_seed_policy='hashed_stage_v1',
            sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend',
            realization_policy='function_free' if label=='function_free' else 'functional')
        write(path,config)
    run=read(root/'run.json');run.update(version=7,completion_model='realization',experiments=['end_to_end','layout_control','function_free'],
        proposal_domain='shared_union_not_autonomous_backend_completion',proposal_failure_policy='Continue remaining fixed-budget stages; no retry until nonempty')
    write(root/'run.json',run);return root


def propose(root,backend,p=None):return previous.propose(root,backend,p,scm=scm,continue_on_error=True)


def pool(root):
    run=read(root/'run.json');case=read(root/'shared/case.json');hypotheses=[];programs={};sources={};tagsets={()};provenance={}
    for backend in ('qwen','api'):
        folder=root/'proposals'/backend
        provenance[backend]=read(folder/'summary.json') if (folder/'summary.json').is_file() else dict(status='not_run')
        if not (folder/'programs.json').is_file():continue
        hm={h['id']:scm.profile(backend+'_'+h['id'],h['functions']) for h in read(folder/'hypotheses.json')}
        hypotheses+=list(hm.values());tagsets.update(tuple(h['functions']) for h in hm.values())
        for pid,program in read(folder/'programs.json').items():
            for assembly in sorted({g['assembly_key'] for g in program['groups'] if g['objects']}):
                groups=[deepcopy(g) for g in program['groups'] if g['assembly_key']==assembly and g['objects']]
                modes={g['realization_mode'] for g in groups}
                if len(modes)!=1 or not modes<=set(scm.MODES):raise ValueError('A module has inconsistent realization modes')
                mode=next(iter(modes));keys={o['key'] for g in groups for o in g['objects']};renamed={k:mode+'_'+k for k in keys}
                relations=[deepcopy(r) for r in program['relations'] if r['source'] in keys or r['target'] in keys]
                if any(r[k] not in keys and not r[k].startswith('obs_') for r in relations for k in ('source','target')):raise ValueError('Cross-module reference')
                for g in groups:
                    for o in g['objects']:
                        o['key']=renamed[o['key']];host=o['attachment']['host'];o['attachment']['host']=renamed.get(host,host)
                for r in relations:
                    for k in ('source','target'):r[k]=renamed.get(r[k],r[k])
                fid='a'+str(len(programs));role=groups[0]['function']
                programs[fid]=dict(description=role+' / '+mode+': '+', '.join(o['variant'] for g in groups for o in g['objects']),
                    hypothesis=hm[program['hypothesis']['id']],groups=groups,relations=relations,relation_bindings=[],inventory_scope='one_realization_channel_variant')
                sources[fid]=dict(backend=backend,program=pid,assembly=assembly,mode=mode)
    if not programs:raise ValueError('No validated nonempty proposals; missing support is not inferred absence')
    base=scm.build_model(hypotheses,programs);slots={};definitions={};roles=set()
    for fid,p in base['furniture_states'].items():
        if fid=='f_null':continue
        role=p['groups'][0]['function'];mode=p['groups'][0]['realization_mode'];channel=role+'__'+mode
        slots.setdefault(channel,['none']).append(fid);definitions[channel]=dict(function=role,realization_mode=mode);roles.add(role)
    slots={r:slots[r] for r in sorted(slots)};roles=sorted(roles)
    recognition_tags=set(tagsets)
    tagsets.update((r,) for r in roles);tagsets.add(tuple(roles));tagsets.update(tuple(sorted(set(roles)-{r})) for r in roles)
    profiles=[scm.profile('h'+str(i),tags) for i,tags in enumerate(sorted(tagsets))];pairs=[]
    for role in roles:
        for h in profiles:
            if role in h['functions']:continue
            other=next((v for v in profiles if set(v['functions'])==set(h['functions'])|{role}),None)
            if other:pairs.append(dict(role=role,without=h['id'],with_role=other['id']))
    domain=dict(version=7,hypotheses=hypotheses,profiles=profiles,matched_profile_pairs=pairs,programs=base['furniture_states'],base_model=base,
        recognition_profile_ids=[h['id'] for h in profiles if tuple(h['functions']) in recognition_tags],
        slots=slots,channel_definitions=definitions,common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        proposal_provenance=provenance,program_proposal_sources=sources,
        support_gaps=[dict(function=r,missing_mode=mode,meaning='Not proposed; does not prove absence') for r in roles for mode in scm.MODES if r+'__'+mode not in slots])
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError('Domain already frozen')
    write(path,domain);k=run['draws'];multi=sum(len(s)>2 for s in slots.values());initial=k*(1+len(profiles)*(1+multi))
    write(root/'call_budget.json',dict(profiles=len(profiles),channels=len(slots),draws=k,mechanism_initial_calls_per_backend=initial,
        maximum_with_one_repair_each=2*initial,singleton_variants_compiled_without_model_calls=len(slots)-multi))
    return dict(status='complete',profiles=len(profiles),channels=len(slots),initial_calls_per_backend=initial,support_gaps=domain['support_gaps'])


def singleton(state):
    return dict(counts={state:0},valid_draws=0,invalid_draws=0,probabilities={state:1.},deterministic_domain=True,
        posterior_std_assuming_multinomial={state:0.},scope='One legal conditional value; no empirical model draw needed')


def recognition_table(votes,domain,alpha):
    eligible=domain['recognition_profile_ids'];row=sd.estimate_table(votes,eligible,alpha)
    for field in ('counts','probabilities','posterior_std_assuming_multinomial'):
        row[field]={h['id']:row[field].get(h['id'],0) for h in domain['profiles']}
    row['scope']='Recognition on proposed/uncommitted profiles; synthetic diagnostic profiles have zero observational mass'
    return row


def mechanisms(root,backend,p=None):
    config=check(root/'context');run=read(root/'run.json');p=p or provider(config,backend)
    domain=read(root/'shared/domain.json');case=read(root/'shared/case.json');k=run['draws'];channels=list(domain['slots']);hs=[h['id'] for h in domain['profiles']]
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_with_one_repair_each']
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=False);board=Image.open(root/'shared/observation_board.png').convert('RGB');plans=[]
    def permute(values,name):
        rng=np.random.default_rng(stage_seed(run['seed'],name));return [values[i] for i in rng.permutation(len(values))]
    def ask(name,image,prompt,validator,parents):
        plans.append(dict(stage=name,parents=parents,image_delivered=image is not None,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        try:return previous._ask(p,run,name,image,prompt,out/(name+'.txt'),validator,256)
        except ValueError as exc:plans.append(dict(stage=name,invalid_error=str(exc)));return None
    votes=[]
    for j in range(k):
        name='recognition_'+str(j);d=dict(domain,profiles=permute([h for h in domain['profiles'] if h['id'] in domain['recognition_profile_ids']],name))
        votes.append(ask(name,board,scm.recognition_prompt(d,case),lambda v:sd.choice_answer(v,domain['recognition_profile_ids']),['O','D']))
    recognition=recognition_table(votes,domain,run['concentration']);activation={};variants={}
    for h in domain['profiles']:
        votes=[]
        for j in range(k):
            name='activation_'+h['id']+'_'+str(j)
            votes.append(ask(name,None,scm.activation_prompt(domain,h,permute(channels,name)),lambda v:scm.activation_answer(v,channels),['H','D']))
        activation[h['id']]=scm.estimate_activation(votes,channels,run['concentration'])
        for channel,states in domain['slots'].items():
            allowed=[s for s in states if s!='none']
            if len(allowed)==1:variants[h['id']+'|'+channel]=singleton(allowed[0]);continue
            votes=[]
            for j in range(k):
                name='variant_'+h['id']+'_'+channel+'_'+str(j)
                votes.append(ask(name,None,scm.variant_prompt(domain,h,channel,permute(allowed,name)),lambda v,allowed=allowed:sd.choice_answer(v,allowed),['H','A_'+channel+'=1','D']))
            variants[h['id']+'|'+channel]=sd.estimate_table(votes,allowed,run['concentration'])
    result=dict(kind='anchored_realization',source=backend,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],recognition=recognition,
        activation=activation,variants=variants,original_parent_prompt_plan=plans,API_seed_sent=False,probabilities_calibrated=False)
    path=root/'shared/mechanisms'/(backend+'.json')
    if path.exists():raise FileExistsError('Mechanism already frozen')
    write(path,result);return dict(status='complete',profiles=len(hs),channels=len(channels))


def pool_mechanisms(root):
    domain=read(root/'shared/domain.json');tables=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    plans=[{p['stage']:p['prompt_sha256'] for p in t['original_parent_prompt_plan'] if 'prompt_sha256' in p} for t in tables]
    if plans[0]!=plans[1] or any(t['domain_sha256']!=sha(root/'shared/domain.json') for t in tables):raise ValueError('Paired prompts or domain differ')
    alpha=tables[0]['concentration']
    if any(t['concentration']!=alpha for t in tables):raise ValueError('Smoothing differs')
    def draws(rows):
        result=[]
        for row in rows:
            for value,n in row['counts'].items():result.extend([value]*n)
            result.extend([None]*row['invalid_draws'])
        return result
    def combine(rows):
        states=list(rows[0]['counts'])
        if all(r.get('deterministic_domain') for r in rows):return singleton(states[0])
        return sd.estimate_table(draws(rows),states,alpha)
    result=dict(kind='anchored_realization',source='pooled_for_geometry_control',domain_sha256=sha(root/'shared/domain.json'),concentration=alpha,
        recognition=recognition_table(draws([t['recognition'] for t in tables]),domain,alpha),
        activation={h:scm.estimate_activation(draws([t['activation'][h] for t in tables]),list(domain['slots']),alpha) for h in tables[0]['activation']},
        variants={h:combine([t['variants'][h] for t in tables]) for h in tables[0]['variants']})
    target=root/'shared/mechanisms/pooled.json'
    if target.exists():raise FileExistsError('Pooled mechanism already frozen')
    write(target,result);return dict(status='complete',paired_parent_prompts=len(plans[0]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path);p.add_argument('--backend',choices=['qwen','api'])
    p.add_argument('--experiment',choices=['end_to_end','layout_control','function_free']);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--worlds',type=int,default=4);p.add_argument('--draws',type=int,default=4);a=p.parse_args();root=a.root.resolve()
    if a.action in ('propose','mechanisms','ready','block') and a.backend is None:p.error('--backend required')
    if a.action in ('ready','block') and a.experiment is None:p.error('--experiment required')
    try:
        if a.action=='prepare':
            if a.source_root is None:p.error('--source-root required')
            result=prepare(root,a.source_root,a.seed,a.worlds,a.draws)
        elif a.action=='propose':result=propose(root,a.backend)
        elif a.action=='pool':result=pool(root)
        elif a.action=='mechanisms':result=mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms':result=pool_mechanisms(root)
        elif a.action=='ready':
            result=previous.ready(root,a.experiment,a.backend)
            if not result['ready']:print(json.dumps(result));raise SystemExit(1)
        elif a.action=='block':result=previous.block(root,a.experiment,a.backend)
        else:
            from .assembly_activation_visual import report
            result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'):raise SystemExit(1)
    except Exception as exc:
        errors=root/'phase_errors';errors.mkdir(parents=True,exist_ok=True)
        error=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc));write(errors/(a.action+'_'+str(a.backend)+'.json'),error)
        p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
