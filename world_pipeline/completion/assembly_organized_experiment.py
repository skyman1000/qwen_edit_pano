"""Controlled v13 completion from the actual v12 rich program domain."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..common import read,write,sha
from . import assembly_organized as scm,assembly_compositional as v12
from . import assembly_sdscene_experiment as old,assembly_realization_experiment as recognition
from .assembly_experiment import initialize,check,stage_seed,SeededProvider
from .fresh_pipeline import provider
from .world_scm_runner import ask_valid

EXPERIMENTS=('end_to_end','layout_control','function_free','geometry_control')


def prepare(root,source,seed=3407,draws=4,concentration=1.,variant_concentration=1.):
    source=source.resolve();r=read(source/'run.json')
    if root.resolve()==source or source in root.resolve().parents:raise ValueError('New run must be outside the original source run')
    if r.get('completion_model')!='compositional' or r['version']!=12:raise ValueError('Use the actual v12 run root, not an experiment subfolder')
    if type(draws) is not int or draws<4 or draws>20 or draws%4:raise ValueError('draws must be 4,8,12,16,20 for balanced encoding blocks')
    if not np.isfinite([concentration,variant_concentration]).all() or min(concentration,variant_concentration)<=0:raise ValueError('Positive finite concentrations required')
    worlds=r['worlds'];root=old.prepare(root,source/'end_to_end',seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='compositional')
    d=read(source/'shared/domain.json')
    if d['RGB_sha256']!=sha(root/'context/input.png') or d['G_obs_sha256']!=sha(root/'context/qwen/observer3d/G_obs.json'):raise ValueError('Source candidate observation binding differs')
    write(root/'shared/domain_v12.json',d)
    c=read(root/'context/experiment.json')
    for label in ('function_free','geometry_control'):
        initialize(root/label,root/'context/input.png',c['hfov'],Path(c['local_model']),c['api_model'],seed,
            observations_from=root/'context',shared_observer='qwen',completion_model='compositional',worlds=worlds)
    replay=root/'shared/replay';manifest={str(source/'shared/domain.json'):sha(source/'shared/domain.json'),str(source/'run.json'):sha(source/'run.json')}
    (root/'shared/source_mechanisms').mkdir()
    for b in ('qwen','api'):
        target=replay/b;target.mkdir(parents=True);base=source/'layout_control'/b/'completion';model=read(base/'causal_model.json');frozen=[]
        if read(base/'G_obs.json')!=read(root/'context/qwen/observer3d/G_obs.json'):raise ValueError('Replay observation differs')
        write(target/'model.json',model)
        for i in range(1,worlds+1):
            p=base/f'world_{i:02d}/semantic_world.json';frozen.append(read(p));manifest[str(p)]=sha(p)
        write(target/'worlds.json',frozen);manifest[str(base/'causal_model.json')]=sha(base/'causal_model.json')
        write(target/'support_plane.json',read(base/'support_plane.json'))
        manifest[str(base/'support_plane.json')]=sha(base/'support_plane.json')
        mechanism=source/'shared/mechanisms'/(b+'.json');write(root/'shared/source_mechanisms'/(b+'.json'),read(mechanism));manifest[str(mechanism)]=sha(mechanism)
    profiles=scm.organization_profiles(d['slots']);d.update(version=13,profiles=profiles,recognition_profile_ids=[h['id'] for h in profiles],
        organization_semantics='Candidate-world functional emphasis; other functions may coexist; no biography or deterministic ownership',
        source_domain_sha256=sha(source/'shared/domain.json'),candidate_proposals_reused=True)
    write(root/'shared/domain.json',d)
    rgb=os.environ.get('CAUPANO_LAYOUT_RGB','0')=='1'
    for label in ('context',)+EXPERIMENTS:
        p=root/label/'experiment.json';c=read(p);c.update(completion_revision='organized_v13',model_variant=scm.MODEL_VARIANT,response_interface=scm.INTERFACE,
            sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend',
            mechanism_policy='function_free' if label=='function_free' else 'functional',layout_image_delivery=rgb,stage_seed_policy='hashed_stage_v1',
            completion_scope='fixed_v12_semantic_candidates_new_organization_mechanisms_two_stage_layout',training_performed=False)
        if label=='geometry_control':c.update(frozen_world_bundle=str(replay),frozen_world_source=str(source/'layout_control'))
        write(p,c)
    run=read(root/'run.json');run.update(version=13,completion_model='organized',experiments=list(EXPERIMENTS),source_run=str(source),
        catalog_source=str(source),catalog_domain_sha256=sha(source/'shared/domain.json'),concentration=concentration,variant_concentration=variant_concentration,
        reference_concentration=variant_concentration,layout_image_delivery=rgb,proposal_calls=0,interface_probe=True,
        comparison_scope='Fixed rich v12 candidate domain; controlled new organization/response and metric mechanisms',
        source_manifest=manifest,training_performed=False)
    write(root/'run.json',run);k=draws;main=k*(1+2*len(profiles)*len(d['slots']));probe=k*len(read(root/'shared/domain_v12.json')['profiles'])*len(d['slots'])
    budget=dict(profiles=len(profiles),roles=list(d['slots']),draws=k,initial_main_calls_per_backend=main,initial_interface_probe_calls_per_backend=probe,
        maximum_mechanism_calls_with_one_repair_each=2*(main+probe),layout_world_attempts_per_backend=worlds*len(EXPERIMENTS),
        initial_layout_calls_per_nonempty_world=2,conditional_range_calls_per_nonempty_world=1,
        maximum_layout_calls_per_nonempty_world_with_one_schema_repair_each=6,
        proposal_calls=0,no_automatic_transport_retries=True)
    write(root/'call_budget.json',budget);return root


def verify_sources(root):
    run=read(root/'run.json')
    if any(not Path(p).is_file() or sha(Path(p))!=h for p,h in run['source_manifest'].items()):raise ValueError('Frozen source run changed')


def choice(value,aliases):
    if not isinstance(value,dict) or set(value)!={'choice'} or not isinstance(value['choice'],str) or value['choice'] not in aliases:raise ValueError('Return exactly one allowed choice alias')
    return aliases[value['choice']]


def binary_aliases(j):
    # Four surface encodings per block; responses are never forced to balance.
    values=['0','1'] if j%2==0 else ['1','0'];items=[('state_'+str(i),v) for i,v in enumerate(values)]
    if (j//2)%2:items.reverse()
    return dict(items)


class Requests:
    def __init__(self,p,root,backend,seed):
        self.provider=p;self.folder=root/'mechanism_calls'/backend;self.folder.mkdir(parents=True,exist_ok=True);self.seed=seed;self.plans=[]
    def ask(self,name,prompt,validator,canonical):
        seed=stage_seed(self.seed,canonical);digest=hashlib.sha256(prompt.encode()).hexdigest();cache=self.folder/(name+'.accepted.json')
        self.plans.append(dict(stage=name,canonical_draw=canonical,local_seed=seed,prompt_sha256=digest,image_delivered=False,max_tokens=256))
        if cache.exists():
            c=read(cache)
            if c['prompt_sha256']!=digest or c['seed']!=seed or sha(Path(c['raw_source']))!=c['raw_sha256']:raise ValueError('Cached request binding changed')
            return validator(read(c['raw_source']))
        i=1
        while (self.folder/(name+'_attempt'+str(i)+'.txt')).exists() or (self.folder/(name+'_attempt'+str(i)+'.request.json')).exists():i+=1
        path=self.folder/(name+'_attempt'+str(i)+'.txt')
        try:value=ask_valid(SeededProvider(self.provider,seed),None,prompt,path,validator,256)
        except ValueError:return None
        raw=path.with_name(path.stem+'_repair.json') if path.with_name(path.stem+'_repair.json').exists() else path.with_suffix('.json')
        write(cache,dict(prompt_sha256=digest,seed=seed,raw_source=str(raw),raw_sha256=sha(raw)));return value


def aliases_for(states,seed,canonical):
    rng=np.random.default_rng(stage_seed(seed,canonical));order=[states[i] for i in rng.permutation(len(states))]
    return {'value_'+str(i):v for i,v in enumerate(order)}


def mechanisms(root,backend,p=None):
    verify_sources(root);config=check(root/'context');run=read(root/'run.json');d=read(root/'shared/domain.json');p=p or provider(config,backend)
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_mechanism_calls_with_one_repair_each']
    requests=Requests(p,root,backend,run['seed']);votes=[];k=run['draws'];presence={};variants={}
    for j in range(k):
        canonical='recognition_'+str(j);a=aliases_for([h['id'] for h in d['profiles']],run['seed'],canonical)
        votes.append(requests.ask(canonical,scm.recognition_prompt(d,a),lambda v,a=a:choice(v,a),canonical))
    rec=recognition.recognition_table(votes,d,run['concentration'])
    for h in d['profiles']:
        for role,fs in d['slots'].items():
            av=[];vv=[];states=[s for s,v in d['spatial_choices'].items() if v['module'] in fs[1:]]
            for j in range(k):
                canonical='presence_'+role+'_'+str(j);a=binary_aliases(j)
                av.append(requests.ask('presence_'+h['id']+'_'+role+'_'+str(j),scm.presence_prompt(d,h,role,a),lambda v,a=a:choice(v,a),canonical))
                canonical='variant_'+role+'_'+str(j);a=aliases_for(states,run['seed'],canonical)
                vv.append(requests.ask('variant_'+h['id']+'_'+role+'_'+str(j),scm.variant_prompt(d,h,role,a),lambda v,a=a:choice(v,a),canonical))
            presence[h['id']+'|'+role]=v12.raw(av);variants[h['id']+'|'+role]=v12.raw(vv)
    result=dict(kind=scm.KIND,source=backend,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],variant_concentration=run['variant_concentration'],
        recognition=rec,presence=presence,variants=variants,original_parent_prompt_plan=deepcopy(requests.plans),training_performed=False,API_seed_sent=False,
        estimator_scope='Empirical LM response distribution averaged over four counterbalanced encodings; not calibrated world probabilities')
    target=root/'shared/mechanisms'/(backend+'.json')
    if target.exists():raise FileExistsError(target)
    write(target,result)
    # Old H + new existence interface, without generating extra geometries.
    old_domain=read(root/'shared/domain_v12.json');probe={};errors=[]
    try:
        for h in old_domain['profiles']:
            for role in old_domain['slots']:
                values=[]
                for j in range(k):
                    a=binary_aliases(j);canonical='presence_'+role+'_'+str(j)
                    values.append(requests.ask('interface_oldH_'+h['id']+'_'+role+'_'+str(j),scm.presence_prompt(old_domain,h,role,a),lambda v,a=a:choice(v,a),canonical))
                probe[h['id']+'|'+role]=v12.raw(values)
    except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
    write(root/'shared'/('interface_probe_'+backend+'.json'),dict(rows=probe,errors=errors,complete=not errors,
        source_H='unchanged_v12',new_main_H='organization_emphasis_hypotheses',original_parent_prompt_plan=requests.plans[len(result['original_parent_prompt_plan']):],
        contrast_scope='Same old H/domain, changed binary response interface; no layout calls'))
    summary=dict(status='partial' if errors else 'complete',main_mechanisms_ready=True,main_initial_calls=len(result['original_parent_prompt_plan']),
        interface_probe_initial_calls=len(requests.plans)-len(result['original_parent_prompt_plan']),api_calls=getattr(p,'calls',None),errors=errors)
    write(root/('mechanism_summary_'+backend+'.json'),summary);return summary


def pool_mechanisms(root):
    ts=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    if ts[0]['original_parent_prompt_plan']!=ts[1]['original_parent_prompt_plan']:raise ValueError('Local/API mechanism prompts or seeds differ')
    if any(t['domain_sha256']!=sha(root/'shared/domain.json') for t in ts):raise ValueError('Frozen domain changed')
    def merge(rows):
        counts=Counter()
        for row in rows:counts.update(row['counts'])
        return dict(counts=dict(counts),valid_draws=sum(v['valid_draws'] for v in rows),invalid_draws=sum(v['invalid_draws'] for v in rows))
    t=deepcopy(ts[0]);t.update(source='pooled_for_layout',presence={s:merge([v['presence'][s] for v in ts]) for s in ts[0]['presence']},
        variants={s:merge([v['variants'][s] for v in ts]) for s in ts[0]['variants']})
    row=merge([v['recognition'] for v in ts]);votes=[h for h,n in row['counts'].items() for _ in range(n)]+[None]*row['invalid_draws']
    t['recognition']=recognition.recognition_table(votes,read(root/'shared/domain.json'),t['concentration']);path=root/'shared/mechanisms/pooled.json'
    if path.exists():raise FileExistsError(path)
    write(path,t);return dict(status='complete')


def ready(root,branch,backend):
    if branch=='geometry_control':return (root/'shared/replay'/backend/'worlds.json').exists()
    source='pooled' if branch=='layout_control' else backend;p=root/'shared/mechanisms'/(source+'.json')
    return p.exists() and read(p)['kind']==scm.KIND and read(p)['domain_sha256']==sha(root/'shared/domain.json')


def world(root,branch,backend,p=None):
    verify_sources(root);check(root/branch)
    if not ready(root,branch,backend):
        out=root/branch/backend/'completion';out.mkdir(parents=True,exist_ok=False)
        result=dict(status='blocked_dependency',worlds=[],blocked_worlds=[],failures=[],world_state_ready=False,dependency=branch+'/'+backend)
        write(out/'summary.json',result);return result
    from .assembly_world import run
    return run(p or provider(read(root/branch/'experiment.json'),backend),root/branch,backend)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','mechanisms','pool-mechanisms','world','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-run',type=Path);p.add_argument('--backend',choices=['qwen','api']);p.add_argument('--experiment',choices=EXPERIMENTS)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--draws',type=int,default=4);p.add_argument('--concentration',type=float,default=1.);p.add_argument('--variant-concentration',type=float,default=1.)
    a=p.parse_args();root=a.root.resolve();root_existed=root.exists()
    if a.action in ('mechanisms','world') and a.backend is None:p.error('--backend required')
    if a.action=='world' and a.experiment is None:p.error('--experiment required')
    if a.action=='prepare' and a.source_run is None:p.error('--source-run required')
    try:
        if a.action!='prepare':
            marker=read(root/'run.json')
            if marker.get('version')!=13 or marker.get('completion_model')!='organized':raise ValueError('Action requires a v13 organized run; original source runs are read-only')
        if a.action=='prepare':result=prepare(root,a.source_run,a.seed,a.draws,a.concentration,a.variant_concentration)
        elif a.action=='mechanisms':result=mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms':result=pool_mechanisms(root)
        elif a.action=='world':result=world(root,a.experiment,a.backend)
        else:
            from .assembly_organized_visual import report
            result=report(root)
        print(json.dumps(result,ensure_ascii=False,default=str))
        if isinstance(result,dict) and (result.get('status') in ('failed','partial','blocked_dependency','no_world_exported') or result.get('failures')):raise SystemExit(1)
    except Exception as exc:
        error=dict(action=a.action,branch=a.experiment,backend=a.backend,error=type(exc).__name__+': '+str(exc));ours=False
        try:
            marker=read(root/'run.json');ours=marker.get('version')==13 and marker.get('completion_model')=='organized'
        except (OSError,ValueError,TypeError):pass
        if ours or (a.action=='prepare' and not root_existed and root.exists()):
            folder=root/'phase_errors';folder.mkdir(parents=True,exist_ok=True);write(folder/(a.action+'_'+str(a.experiment)+'_'+str(a.backend)+'.json'),error)
        p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
