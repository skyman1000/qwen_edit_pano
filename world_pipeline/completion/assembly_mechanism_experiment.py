"""Fresh no-training v8 experiment; observations shared, all new mechanisms."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,write,sha
from . import assembly_mechanism as scm,assembly_realization as v7
from . import assembly_realization_experiment as old,assembly_sdscene_experiment as previous
from .assembly_experiment import initialize,check,stage_seed
from .fresh_pipeline import provider

EXPERIMENTS=('end_to_end','layout_control','function_free','direct_inventory')


def prepare(root,source,seed=3407,worlds=4,draws=4,concentration=1.,reference_concentration=1.):
    scm._positive(concentration,'concentration');scm._positive(reference_concentration,'reference concentration')
    root=previous.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='mechanism')
    config=read(root/'context/experiment.json')
    for label in ('function_free','direct_inventory'):
        initialize(root/label,root/'context/input.png',config['hfov'],Path(config['local_model']),config['api_model'],seed,
            observations_from=root/'context',shared_observer='qwen',completion_model='mechanism',worlds=worlds)
    for label in ('context',)+EXPERIMENTS:
        path=root/label/'experiment.json';c=read(path)
        c.update(sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend',
            mechanism_policy='direct' if label=='direct_inventory' else 'function_free' if label=='function_free' else 'functional',
            sampling_law='reference_shrunk_functional_or_direct_joint_inventory',stage_seed_policy='hashed_stage_v1',
            training_performed=False)
        write(path,c)
    run=read(root/'run.json');run.update(version=8,completion_model='mechanism',experiments=list(EXPERIMENTS),
        concentration=concentration,reference_concentration=reference_concentration,training_performed=False,
        proposal_domain='shared_union_not_autonomous_backend_completion',
        proposal_validation_policy='complete_alternative_atomic_validation_v8',
        proposal_failure_policy='Accept legal complete alternatives in first usable reply; at most one schema repair if none legal; continue fixed stages',
        direct_control_scope='O,D -> joint inventory; no H in inventory calls; same q(H) retained for metric context; not a whole-pipeline H removal',
        budget_scope='Same per-stage settings across backends; direct and functional inference costs differ and are reported')
    run['stage_token_budgets']['direct']=1024
    write(root/'run.json',run);return root


def propose(root,backend,p=None,*,model=scm):
    scm=model
    result=previous.propose(root,backend,p,scm=scm,continue_on_error=True)
    folder=root/'proposals'/backend;records={}
    if (folder/'programs.json').is_file():
        for program in read(folder/'programs.json').values():
            source=program['proposal_source'];stage='furniture_'+program['hypothesis']['id']+'_'+source['channel']
            records[stage]=program['proposal_validation']
    write(folder/'alternative_validation.json',records)
    result['partial_alternative_replies_accepted']=sum(bool(r['rejected']) for r in records.values())
    result['proposal_validation_policy']='complete_alternative_atomic_validation_v8';write(folder/'summary.json',result)
    return result


def pool(root):
    # Reuse only v7's validated domain construction, before v8 freezes its hash.
    old.pool(root)
    path=root/'shared/domain.json';domain=read(path);domain['version']=8
    domain['proposal_validation_policy']='complete_alternative_atomic_validation_v8'
    represented={v['function'] for v in domain['channel_definitions'].values()}
    eligible=set(domain['recognition_profile_ids'])
    domain['unrepresented_recognition_functions']=sorted({f for h in domain['profiles'] if h['id'] in eligible for f in h['functions']}-represented)
    write(path,domain)
    run=read(root/'run.json');k=run['draws'];multi=sum(len(s)>2 for s in domain['slots'].values())
    stages=dict(recognition=k,reference=k,conditional_activation=k*len(domain['profiles']),
        conditional_variants=k*len(domain['profiles'])*multi,direct_inventory=k)
    budget=dict(profiles=len(domain['profiles']),channels=len(domain['slots']),draws=k,stages_per_backend=stages,
        mechanism_initial_calls_per_backend=sum(stages.values()),maximum_with_one_repair_each=2*sum(stages.values()),
        singleton_variants_compiled_without_model_calls=len(domain['slots'])-multi,
        total_layout_attempts_per_backend=run['worlds']*len(EXPERIMENTS),
        maximum_layout_calls_per_backend=2*run['worlds']*len(EXPERIMENTS),
        direct_and_functional_budgets_matched=False)
    write(root/'call_budget.json',budget)
    return dict(status='complete',domain_sha256=sha(path),**budget)


def mechanisms(root,backend,p=None,*,model=scm):
    scm=model
    config=check(root/'context');run=read(root/'run.json');p=p or provider(config,backend)
    domain=read(root/'shared/domain.json');case=read(root/'shared/case.json');k=run['draws'];channels=list(domain['slots'])
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_with_one_repair_each']
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=False)
    board=Image.open(root/'shared/observation_board.png').convert('RGB');plans=[]
    def permute(values,name):
        rng=np.random.default_rng(stage_seed(run['seed'],name));return [values[i] for i in rng.permutation(len(values))]
    def ask(name,image,prompt,validator,parents,budget=None):
        if budget is None:budget=run['stage_token_budgets']['recognition' if name.startswith('recognition_') else 'mechanism']
        plans.append(dict(stage=name,parents=parents,image_delivered=image is not None,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),max_tokens=budget))
        try:return previous._ask(p,run,name,image,prompt,out/(name+'.txt'),validator,budget)
        except ValueError as exc:plans.append(dict(stage=name,invalid_error=str(exc)));return None
    votes=[]
    for j in range(k):
        name='reference_'+str(j)
        votes.append(ask(name,None,scm.reference_prompt(domain,permute(channels,name)),lambda v:scm.activation_answer(v,channels),['D']))
    reference=v7.estimate_activation(votes,channels,run['reference_concentration'])
    votes=[]
    for j in range(k):
        name='recognition_'+str(j);d=dict(domain,profiles=permute([h for h in domain['profiles'] if h['id'] in domain['recognition_profile_ids']],name))
        votes.append(ask(name,board,scm.recognition_prompt(d,case),lambda v:sd_choice(v,domain['recognition_profile_ids']),['O','D']))
    recognition=old.recognition_table(votes,domain,run['concentration']);activation={};variants={}
    for h in domain['profiles']:
        votes=[]
        for j in range(k):
            name='activation_'+h['id']+'_'+str(j)
            votes.append(ask(name,None,scm.activation_prompt(domain,h,permute(channels,name)),lambda v:scm.activation_answer(v,channels),['H','D']))
        activation[h['id']]=scm.raw_activation(votes,channels)
        for c,states in domain['slots'].items():
            allowed=[s for s in states if s!='none']
            if len(allowed)==1:variants[h['id']+'|'+c]=old.singleton(allowed[0]);continue
            votes=[]
            for j in range(k):
                name='variant_'+h['id']+'_'+c+'_'+str(j)
                votes.append(ask(name,None,scm.variant_prompt(domain,h,c,permute(allowed,name)),lambda v,a=allowed:sd_choice(v,a),['H','A_'+c+'=1','D']))
            from .assembly_sdscene import estimate_table
            variants[h['id']+'|'+c]=estimate_table(votes,allowed,run['concentration'])
    votes=[]
    for j in range(k):
        name='direct_'+str(j)
        votes.append(ask(name,board,scm.direct_prompt(domain,permute(channels,name)),lambda v:scm.direct_answer(v,domain),['O','D'],run['stage_token_budgets']['direct']))
    direct=scm.raw_direct(votes,domain)
    result=dict(kind=scm.KIND,source=backend,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],
        recognition=recognition,reference=reference,activation=activation,variants=variants,direct=direct,
        original_parent_prompt_plan=plans,API_seed_sent=False,training_performed=False,probabilities_calibrated=False,
        api_calls=getattr(p,'calls',None),api_usage=getattr(p,'usage',None))
    path=root/'shared/mechanisms'/(backend+'.json')
    if path.exists():raise FileExistsError('Mechanism already frozen')
    write(path,result);return dict(status='complete',initial_requests=sum('prompt_sha256' in r for r in plans),
        repair_requests=len(list(out.glob('*_repair.request.json'))),api_calls=getattr(p,'calls',None))


def sd_choice(value,allowed):
    from .assembly_sdscene import choice_answer
    return choice_answer(value,allowed)


def pool_mechanisms(root,*,model=scm):
    scm=model
    from .assembly_sdscene import estimate_table
    domain=read(root/'shared/domain.json');tables=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    plans=[{p['stage']:(p['prompt_sha256'],p['image_delivered'],p['parents'],p['max_tokens']) for p in t['original_parent_prompt_plan'] if 'prompt_sha256' in p} for t in tables]
    if plans[0]!=plans[1] or any(t['kind']!=scm.KIND or t['domain_sha256']!=sha(root/'shared/domain.json') for t in tables):
        raise ValueError('Paired prompts or frozen domains differ')
    alpha=tables[0]['concentration'];eta=tables[0]['reference']['concentration']
    if any(t['concentration']!=alpha or t['reference']['concentration']!=eta for t in tables):raise ValueError('Concentrations differ')
    def draws(rows):
        values=[]
        for row in rows:
            for v,n in row['counts'].items():values.extend([v]*n)
            values.extend([None]*row['invalid_draws'])
        return values
    def combine(rows):
        if all(r.get('deterministic_domain') for r in rows):return old.singleton(next(iter(rows[0]['counts'])))
        return estimate_table(draws(rows),list(rows[0]['counts']),alpha)
    result=dict(kind=scm.KIND,source='pooled_for_geometry_control',domain_sha256=sha(root/'shared/domain.json'),concentration=alpha,
        recognition=old.recognition_table(draws([t['recognition'] for t in tables]),domain,alpha),
        reference=v7.estimate_activation(draws([t['reference'] for t in tables]),list(domain['slots']),eta),
        activation={h:scm.raw_activation(draws([t['activation'][h] for t in tables]),list(domain['slots'])) for h in tables[0]['activation']},
        variants={s:combine([t['variants'][s] for t in tables]) for s in tables[0]['variants']},
        direct=scm.raw_direct(draws([t['direct'] for t in tables]),domain),training_performed=False,probabilities_calibrated=False)
    path=root/'shared/mechanisms/pooled.json'
    if path.exists():raise FileExistsError('Pooled mechanism already frozen')
    write(path,result);return dict(status='complete',paired_parent_prompts=len(plans[0]))


def ready(root,experiment,backend,*,model=scm):
    scm=model
    source='pooled' if experiment=='layout_control' else backend;path=root/'shared/mechanisms'/(source+'.json');domain=root/'shared/domain.json'
    result=dict(ready=False,source=source,domain=str(domain),table=str(path))
    if not domain.is_file() or not path.is_file():return result
    table=read(path)
    result['ready']=table.get('kind')==scm.KIND and table.get('domain_sha256')==sha(domain)
    return result


def block(root,experiment,backend,*,model=scm):
    dependency=ready(root,experiment,backend,model=model)
    if dependency['ready']:raise ValueError('Cannot block a ready experiment')
    out=root/experiment/backend/'completion';out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='blocked_dependency',worlds=[],blocked_worlds=[],failures=[],world_state_ready=False,
        error='Missing or incompatible v8 domain/mechanism; inspect phase_errors',dependency=dependency)
    write(out/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path)
    p.add_argument('--backend',choices=['qwen','api']);p.add_argument('--experiment',choices=EXPERIMENTS)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--worlds',type=int,default=4);p.add_argument('--draws',type=int,default=4)
    p.add_argument('--concentration',type=float,default=1.);p.add_argument('--reference-concentration',type=float,default=1.)
    a=p.parse_args();root=a.root.resolve()
    if a.action in ('propose','mechanisms','ready','block') and a.backend is None:p.error('--backend required')
    if a.action in ('ready','block') and a.experiment is None:p.error('--experiment required')
    try:
        if a.action=='prepare':
            if a.source_root is None:p.error('--source-root required')
            result=prepare(root,a.source_root,a.seed,a.worlds,a.draws,a.concentration,a.reference_concentration)
        elif a.action=='propose':result=propose(root,a.backend)
        elif a.action=='pool':result=pool(root)
        elif a.action=='mechanisms':result=mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms':result=pool_mechanisms(root)
        elif a.action=='ready':
            result=ready(root,a.experiment,a.backend)
            if not result['ready']:print(json.dumps(result));raise SystemExit(1)
        elif a.action=='block':result=block(root,a.experiment,a.backend)
        else:
            from .assembly_mechanism_visual import report
            result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'):raise SystemExit(1)
    except Exception as exc:
        folder=root/'phase_errors';folder.mkdir(parents=True,exist_ok=True)
        error=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc));write(folder/(a.action+'_'+str(a.backend)+'.json'),error)
        p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
