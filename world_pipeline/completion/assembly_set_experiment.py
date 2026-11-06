"""Matched local/API joint assembly sets and an explicit RGB-input ablation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,write,sha
from . import assembly_set as scm,assembly_mechanism_experiment as v8
from . import assembly_sdscene_experiment as old,assembly_realization_experiment as v7
from .assembly_experiment import initialize,check,stage_seed
from .fresh_pipeline import provider

EXPERIMENTS=('end_to_end','layout_control','function_free','direct_inventory','tag_only')


def prepare(root,source,seed=3407,worlds=4,draws=4,concentration=1.,reference_concentration=1.):
    scm._positive(concentration,'concentration');scm._positive(reference_concentration,'reference concentration')
    root=old.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='assemblyset')
    config=read(root/'context/experiment.json')
    for label in EXPERIMENTS[2:]:
        initialize(root/label,root/'context/input.png',config['hfov'],Path(config['local_model']),config['api_model'],seed,
            observations_from=root/'context',shared_observer='qwen',completion_model='assemblyset',worlds=worlds)
    for label in ('context',)+EXPERIMENTS:
        path=root/label/'experiment.json';c=read(path)
        c.update(sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend_tag_only' if label=='tag_only' else 'per_backend',
            mechanism_policy='direct' if label=='direct_inventory' else 'function_free' if label=='function_free' else 'functional',
            assembly_rgb_conditioned=label!='tag_only',stage_seed_policy='hashed_stage_v1',training_performed=False,
            api_base='https://api.siliconflow.cn/v1',sampling_law='joint_candidate_assembly_set_given_fixed_observation')
        write(path,c)
    run=read(root/'run.json');run.update(version=10,completion_model='assemblyset',experiments=list(EXPERIMENTS),
        concentration=concentration,reference_concentration=reference_concentration,training_performed=False,
        proposal_interface_revision=scm.PROPOSAL_REVISION,proposal_domain='shared_dependency_closed_candidate_units',
        inventory_policy='Joint optional units, no one-per-function/mode cap or extra-zone existence gate',
        budget_scope='Local/API matched; functional/direct joint-selection initial requests and output limits matched; shared proposal/recognition/reference; tag_only is extra ablation')
    run['stage_token_budgets'].update(mechanism=512,direct=512)
    write(root/'run.json',run);return root


def propose(root,backend,p=None):return v8.propose(root,backend,p,model=scm)


def pool(root):
    run=read(root/'run.json');case=read(root/'shared/case.json');hypotheses=[];programs={};sources={};tags={()};provenance={}
    for backend in ('qwen','api'):
        folder=root/'proposals'/backend;summary=folder/'summary.json';provenance[backend]=read(summary) if summary.exists() else dict(status='not_run')
        if not (folder/'programs.json').is_file():continue
        hm={h['id']:scm.profile(backend+'_'+h['id'],h['functions']) for h in read(folder/'hypotheses.json')}
        hypotheses+=list(hm.values());tags.update(tuple(h['functions']) for h in hm.values())
        for pid,p in read(folder/'programs.json').items():
            if not any(g['objects'] for g in p['groups']):continue
            fid='a'+str(len(programs));programs[fid]=deepcopy(p);programs[fid]['hypothesis']=hm[p['hypothesis']['id']]
            sources[fid]=dict(backend=backend,program=pid,source_alternative=p['proposal_alternative_index'],source_groups=p.get('proposal_component_indices',[]))
    if not programs:raise ValueError('No nonempty candidate units; inspect proposals instead of treating unavailable support as known empty space')
    base=scm.build_model(hypotheses,programs);slots={};definitions={};roles=set()
    for fid,p in base['furniture_states'].items():
        if fid=='f_null':continue
        channel='unit_'+fid;functions=sorted({f for g in p['groups'] for f in g.get('function_tags',[g['function']])});roles.update(functions)
        slots[channel]=['none',fid];definitions[channel]=dict(function='/'.join(functions),realization_mode='candidate_assembly',
            assembly_id=fid,scope='One optional dependency-closed unit, not an exclusive function/zone slot')
    recognition_tags=set(tags);tags.update((r,) for r in roles);tags.add(tuple(sorted(roles)));tags.update(tuple(sorted(roles-{r})) for r in roles)
    profiles=[scm.profile('h'+str(i),t) for i,t in enumerate(sorted(tags))];pairs=[]
    for role in sorted(roles):
        for h in profiles:
            if role in h['functions']:continue
            other=next((x for x in profiles if set(x['functions'])==set(h['functions'])|{role}),None)
            if other:pairs.append(dict(role=role,without=h['id'],with_role=other['id']))
    support={c:sorted({g['layout_region'] for g in base['furniture_states'][states[1]]['groups']}) for c,states in slots.items()}
    domain=dict(version=10,hypotheses=hypotheses,profiles=profiles,matched_profile_pairs=pairs,programs=base['furniture_states'],base_model=base,
        slots=slots,channel_definitions=definitions,recognition_profile_ids=[h['id'] for h in profiles if tuple(h['functions']) in recognition_tags],
        common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        proposal_provenance=provenance,program_proposal_sources=sources,spatial_support=support,
        rear_program_proposed=any('rear' in v for v in support.values()),missing_rear_support_meaning='Unproposed is unknown, not observed empty',
        candidate_validation_policy='Reject an invalid dependency component atomically; independent units remain candidates, not complete worlds',
        function_slot_exclusivity=False)
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError('Domain already frozen')
    write(path,domain);k=run['draws'];stages=dict(recognition=k,reference=k,conditional_joint_selection=k*len(profiles),
        tag_only_joint_selection=k*len(profiles),direct_inventory=k*len(profiles))
    write(root/'call_budget.json',dict(profiles=len(profiles),candidate_units=len(slots),draws=k,stages_per_backend=stages,
        mechanism_initial_calls_per_backend=sum(stages.values()),maximum_with_one_repair_each=2*sum(stages.values()),
        total_layout_attempts_per_backend=run['worlds']*len(EXPERIMENTS),direct_and_functional_budgets_matched=True,
        budget_match_scope='Initial joint-selection requests and max output tokens; actual repair/usage may differ; shared upstream costs and diagnostic branches separately reported'))
    return dict(status='complete',domain_sha256=sha(path),candidate_units=len(slots),candidate_instances=len(base['component_registry']))


def mechanisms(root,backend,p=None):
    config=check(root/'context');run=read(root/'run.json');domain=read(root/'shared/domain.json');case=read(root/'shared/case.json')
    p=p or provider(config,backend);k=run['draws'];channels=list(domain['slots']);plans=[]
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_with_one_repair_each']
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=False);board=Image.open(root/'shared/observation_board.png').convert('RGB')
    def permute(values,name):
        rng=np.random.default_rng(stage_seed(run['seed'],name));return [values[i] for i in rng.permutation(len(values))]
    def ask(name,image,prompt,validator,parents,budget):
        plans.append(dict(stage=name,parents=parents,image_delivered=image is not None,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),max_tokens=budget))
        try:return old._ask(p,run,name,image,prompt,out/(name+'.txt'),validator,budget)
        except ValueError as exc:plans.append(dict(stage=name,invalid_error=str(exc)));return None
    reference=scm.previous.v7.estimate_activation([ask('reference_'+str(j),None,scm.reference_prompt(domain,permute(channels,'reference_'+str(j))),
        lambda v:scm.activation_answer(v,channels),['D'],run['stage_token_budgets']['mechanism']) for j in range(k)],channels,run['reference_concentration'])
    votes=[]
    for j in range(k):
        name='recognition_'+str(j);d=dict(domain,profiles=permute([h for h in domain['profiles'] if h['id'] in domain['recognition_profile_ids']],name))
        votes.append(ask(name,board,scm.recognition_prompt(d,case),lambda v:v8.sd_choice(v,domain['recognition_profile_ids']),['O','D'],run['stage_token_budgets']['recognition']))
    recognition=v8.old.recognition_table(votes,domain,run['concentration']);activations={True:{},False:{}};variants={}
    for h in domain['profiles']:
        for with_image in (True,False):
            votes=[]
            for j in range(k):
                canonical='activation_'+h['id']+'_'+str(j);name=canonical if with_image else 'tag_'+canonical
                votes.append(ask(name,board if with_image else None,scm.activation_prompt(domain,h,permute(channels,canonical)),
                    lambda v:scm.activation_answer(v,channels),(['O_RGB'] if with_image else [])+['H','D'],run['stage_token_budgets']['mechanism']))
            activations[with_image][h['id']]=scm.raw_activation(votes,channels)
        for c,states in domain['slots'].items():variants[h['id']+'|'+c]=v8.old.singleton(states[1])
    direct=scm.raw_direct([ask('direct_'+str(j),board,scm.direct_prompt(domain,permute(channels,'direct_'+str(j))),
        lambda v:scm.direct_answer(v,domain),['O','D'],run['stage_token_budgets']['direct']) for j in range(k*len(domain['profiles']))],domain)
    for with_image in (True,False):
        name=backend if with_image else backend+'_tag_only';path=root/'shared/mechanisms'/(name+'.json')
        if path.exists():raise FileExistsError(path)
        write(path,dict(kind=scm.KIND,source=name,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],
            recognition=recognition,reference=reference,activation=activations[with_image],variants=variants,direct=direct,
            conditional_RGB=with_image,original_parent_prompt_plan=plans,API_seed_sent=False,training_performed=False,probabilities_calibrated=False,
            api_calls=getattr(p,'calls',None),api_usage=getattr(p,'usage',None)))
    return dict(status='complete',initial_requests=sum('prompt_sha256' in v for v in plans),repair_requests=len(list(out.glob('*_repair.request.json'))))


def pool_mechanisms(root):
    result=v8.pool_mechanisms(root,model=scm)
    path=root/'shared/mechanisms/pooled.json';t=read(path);t['conditional_RGB']=True;write(path,t)
    return result


def ready(root,experiment,backend):
    source='pooled' if experiment=='layout_control' else backend+'_tag_only' if experiment=='tag_only' else backend
    path=root/'shared/mechanisms'/(source+'.json');domain=root/'shared/domain.json';ok=False
    if path.exists() and domain.exists():
        t=read(path);ok=t.get('kind')==scm.KIND and t.get('domain_sha256')==sha(domain) and t.get('conditional_RGB') is (experiment!='tag_only')
    return dict(ready=ok,source=source,domain=str(domain),table=str(path))


def block(root,experiment,backend):
    dependency=ready(root,experiment,backend)
    if dependency['ready']:raise ValueError('Cannot block a ready experiment')
    out=root/experiment/backend/'completion';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='blocked_dependency',worlds=[],blocked_worlds=[],failures=[],world_state_ready=False,dependency=dependency,error='Missing joint-assembly domain/mechanism')
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path);p.add_argument('--backend',choices=['qwen','api']);p.add_argument('--experiment',choices=EXPERIMENTS)
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
            from .assembly_set_visual import report
            result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'):raise SystemExit(1)
    except Exception as exc:
        folder=root/'phase_errors';folder.mkdir(parents=True,exist_ok=True);error=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc))
        write(folder/(a.action+'_'+str(a.backend)+'.json'),error);p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
