"""Matched v11 local/API execution with canonical-seed RGB mechanism control."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,write,sha
from . import assembly_holistic as scm,assembly_sdscene_experiment as old
from . import assembly_mechanism_experiment as proposal_runner
from . import assembly_realization_experiment as tables
from .assembly_experiment import initialize,check,stage_seed,SeededProvider
from .fresh_pipeline import provider
from .world_scm_runner import ask_valid

EXPERIMENTS=('end_to_end','layout_control','function_free','direct_inventory','rgb_inventory')


def prepare(root,source,seed=3407,worlds=4,draws=4,concentration=1.,reference_concentration=1.):
    scm._positive(concentration,'concentration');scm._positive(reference_concentration,'reference concentration')
    root=old.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='holistic')
    c=read(root/'context/experiment.json')
    for label in EXPERIMENTS[2:]:
        initialize(root/label,root/'context/input.png',c['hfov'],Path(c['local_model']),c['api_model'],seed,
            observations_from=root/'context',shared_observer='qwen',completion_model='holistic',worlds=worlds)
    for label in ('context',)+EXPERIMENTS:
        path=root/label/'experiment.json';c=read(path)
        c.update(sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend_rgb' if label=='rgb_inventory' else 'per_backend',
            mechanism_policy='direct' if label=='direct_inventory' else 'function_free' if label=='function_free' else 'functional',
            stage_seed_policy='hashed_stage_v1',training_performed=False,api_base='https://api.siliconflow.cn/v1',
            sampling_law='joint_inventory_camera_organization_given_positive_observations')
        write(path,c)
    run=read(root/'run.json');run.update(version=11,completion_model='holistic',experiments=list(EXPERIMENTS),
        concentration=concentration,reference_concentration=reference_concentration,training_performed=False,
        proposal_interface_revision=scm.PROPOSAL_REVISION,primary_inventory_context='positive observed facts; RGB retained upstream and in geometry',
        budget_scope='Local/API same prompts/domain/settings; direct and functional selection initial requests/output budgets matched; RGB control uses canonical paired local seeds')
    run['stage_token_budgets'].update(mechanism=1536,direct=1536)
    write(root/'run.json',run);return root


def propose(root,backend,p=None):return proposal_runner.propose(root,backend,p,model=scm)


def pool(root):
    run=read(root/'run.json');case=read(root/'shared/case.json');hypotheses=[];programs={};tags={()};provenance={}
    for backend in ('qwen','api'):
        folder=root/'proposals'/backend;provenance[backend]=read(folder/'summary.json') if (folder/'summary.json').exists() else dict(status='missing')
        if not (folder/'programs.json').exists():continue
        hm={h['id']:scm.profile(backend+'_'+h['id'],h['functions']) for h in read(folder/'hypotheses.json')}
        hypotheses+=list(hm.values());tags.update(tuple(h['functions']) for h in hm.values())
        for p in read(folder/'programs.json').values():
            if not any(g['objects'] for g in p['groups']):continue
            fid='a'+str(len(programs));programs[fid]=deepcopy(p);programs[fid]['hypothesis']=hm[p['hypothesis']['id']]
    if not programs:raise ValueError('No nonempty candidate support; inspect fixed-budget proposals')
    base=scm.build_model(hypotheses,programs);slots={};regions={};roles=set()
    for fid,p in base['furniture_states'].items():
        if fid=='f_null':continue
        c='unit_'+fid;slots[c]=['none',fid];options={tuple(g['region_options']) for g in p['groups']}
        if len(options)!=1:raise ValueError('Dependency unit must share one coherent region domain')
        regions[c]=list(next(iter(options)));roles.update(f for g in p['groups'] for f in g.get('function_tags',[g['function']]))
    recognition_tags=set(tags);tags.update((r,) for r in roles);tags.add(tuple(sorted(roles)));tags.update(tuple(sorted(roles-{r})) for r in roles)
    profiles=[scm.profile('h'+str(i),t) for i,t in enumerate(sorted(tags))]
    domain=dict(version=11,hypotheses=hypotheses,profiles=profiles,programs=base['furniture_states'],base_model=base,
        slots=slots,region_domains=regions,recognition_profile_ids=[h['id'] for h in profiles if tuple(h['functions']) in recognition_tags],
        common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        proposal_provenance=provenance,spatial_support=regions,rear_program_proposed=any('rear' in v for v in regions.values()),
        function_slot_exclusivity=False,spatial_choice_policy='Joint categorical inventory/region vectors; canonical object identities are independent of region',
        symmetric_base='P(unit absent)=1/2; remaining 1/2 split across allowed regions; not an occupancy target or fitted prior')
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError(path)
    write(path,domain);k=run['draws'];stages=dict(recognition=k,reference=k,conditional_joint=k*len(profiles),
        rgb_control_joint=k*len(profiles),direct_joint=k*len(profiles))
    write(root/'call_budget.json',dict(draws=k,profiles=len(profiles),stages_per_backend=stages,
        mechanism_initial_calls_per_backend=sum(stages.values()),maximum_with_one_repair_each=2*sum(stages.values()),
        direct_and_functional_budgets_matched=True,total_layout_attempts_per_backend=run['worlds']*len(EXPERIMENTS)))
    return dict(status='complete',candidate_units=len(slots),candidate_instances=len(base['component_registry']),spatial_support=regions)


def mechanisms(root,backend,p=None):
    config=check(root/'context');run=read(root/'run.json');domain=read(root/'shared/domain.json');case=read(root/'shared/case.json')
    p=p or provider(config,backend);k=run['draws'];channels=list(domain['slots']);plans=[]
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_with_one_repair_each']
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=False);board=Image.open(root/'shared/observation_board.png').convert('RGB')
    def permute(values,name):
        rng=np.random.default_rng(stage_seed(run['seed'],name));return [values[i] for i in rng.permutation(len(values))]
    def ask(name,image,prompt,validator,parents,budget,canonical=None):
        canonical=canonical or name;draw_seed=stage_seed(run['seed'],canonical)
        plans.append(dict(stage=name,canonical_draw=canonical,local_seed=draw_seed,parents=parents,image_delivered=image is not None,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),max_tokens=budget))
        try:return ask_valid(SeededProvider(p,draw_seed),image,prompt,out/(name+'.txt'),validator,budget)
        except ValueError as exc:plans.append(dict(stage=name,invalid_error=str(exc)));return None
    reference=scm.raw_choices([ask('reference_'+str(j),None,scm.joint_prompt(domain,permute(channels,'reference_'+str(j)),reference=True),
        lambda v:scm.choice_answer(v,domain),['D'],run['stage_token_budgets']['mechanism']) for j in range(k)])
    votes=[]
    for j in range(k):
        name='recognition_'+str(j);d=dict(domain,profiles=permute([h for h in domain['profiles'] if h['id'] in domain['recognition_profile_ids']],name))
        votes.append(ask(name,board,scm.recognition_prompt(d,case),lambda v:proposal_runner.sd_choice(v,domain['recognition_profile_ids']),['O_RGB','D'],run['stage_token_budgets']['recognition']))
    recognition=tables.recognition_table(votes,domain,run['concentration']);joint={False:{},True:{}}
    for h in domain['profiles']:
        for rgb in (False,True):
            votes=[]
            for j in range(k):
                canonical='joint_'+h['id']+'_'+str(j);name=('rgb_' if rgb else '')+canonical
                votes.append(ask(name,board if rgb else None,scm.joint_prompt(domain,permute(channels,canonical),h),
                    lambda v:scm.choice_answer(v,domain),(['O_RGB'] if rgb else [])+['H','O_positive','D'],run['stage_token_budgets']['mechanism'],canonical))
            joint[rgb][h['id']]=scm.raw_choices(votes)
    direct=scm.raw_choices([ask('direct_'+str(j),None,scm.joint_prompt(domain,permute(channels,'direct_'+str(j))),
        lambda v:scm.choice_answer(v,domain),['O_positive','D'],run['stage_token_budgets']['direct']) for j in range(k*len(domain['profiles']))])
    for rgb in (False,True):
        name=backend+('_rgb' if rgb else '');path=root/'shared/mechanisms'/(name+'.json')
        if path.exists():raise FileExistsError(path)
        write(path,dict(kind=scm.KIND,source=name,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],
            reference_concentration=run['reference_concentration'],recognition=recognition,reference=reference,joint=joint[rgb],direct=direct,
            conditional_RGB=rgb,original_parent_prompt_plan=plans,API_seed_sent=False,training_performed=False,probabilities_calibrated=False,
            api_calls=getattr(p,'calls',None),api_usage=getattr(p,'usage',None)))
    return dict(status='complete',initial_requests=sum('prompt_sha256' in v for v in plans))


def pool_mechanisms(root):
    ts=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    if ts[0]['original_parent_prompt_plan']!=ts[1]['original_parent_prompt_plan']:raise ValueError('Backend parent prompts differ')
    if any(t['domain_sha256']!=sha(root/'shared/domain.json') or t['kind']!=scm.KIND for t in ts):raise ValueError('Frozen domain mismatch')
    if any(ts[0][k]!=ts[1][k] for k in ('concentration','reference_concentration')):raise ValueError('Smoothing differs')
    def merge(rows):
        counts=Counter()
        for row in rows:counts.update(row['counts'])
        return dict(counts=dict(counts),valid_draws=sum(r['valid_draws'] for r in rows),invalid_draws=sum(r['invalid_draws'] for r in rows))
    t=deepcopy(ts[0]);t.update(source='pooled_for_layout_control',reference=merge([v['reference'] for v in ts]),
        direct=merge([v['direct'] for v in ts]),joint={h:merge([v['joint'][h] for v in ts]) for h in ts[0]['joint']})
    recognition=merge([v['recognition'] for v in ts]);votes=[s for s,n in recognition['counts'].items() for _ in range(n)]+[None]*recognition['invalid_draws']
    t['recognition']=tables.recognition_table(votes,read(root/'shared/domain.json'),t['concentration'])
    path=root/'shared/mechanisms/pooled.json'
    if path.exists():raise FileExistsError(path)
    write(path,t);return dict(status='complete',paired_prompt_plans=True)


def ready(root,experiment,backend):
    source='pooled' if experiment=='layout_control' else backend+('_rgb' if experiment=='rgb_inventory' else '')
    path=root/'shared/mechanisms'/(source+'.json');d=root/'shared/domain.json';ok=False
    if path.exists() and d.exists():
        t=read(path);ok=t['kind']==scm.KIND and t['domain_sha256']==sha(d) and t['conditional_RGB']==(experiment=='rgb_inventory')
    return dict(ready=ok,source=source,table=str(path))


def block(root,experiment,backend):
    status=ready(root,experiment,backend)
    if status['ready']:raise ValueError('Ready experiment cannot be blocked')
    out=root/experiment/backend/'completion';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='blocked_dependency',worlds=[],blocked_worlds=[],failures=[],world_state_ready=False,dependency=status)
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
            from .assembly_holistic_visual import report
            result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'):raise SystemExit(1)
    except Exception as exc:
        folder=root/'phase_errors';folder.mkdir(parents=True,exist_ok=True);error=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc))
        write(folder/(a.action+'_'+str(a.backend)+'.json'),error);p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
