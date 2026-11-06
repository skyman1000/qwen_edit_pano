"""Actual completion from a bound rich semantic catalog; all mechanisms fresh."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..common import read,write,sha
from . import assembly_compositional as scm,assembly_sdscene_experiment as old
from . import assembly_realization_experiment as recognition
from .assembly_experiment import initialize,check,stage_seed,SeededProvider
from .fresh_pipeline import provider
from .world_scm_runner import ask_valid

EXPERIMENTS=('end_to_end','layout_control','function_free')


def prepare(root,source,seed=3407,worlds=4,draws=4,concentration=1.,reference_concentration=1.):
    if concentration<=0 or reference_concentration<=0:raise ValueError('Positive smoothing concentrations required')
    catalog=os.environ.get('CAUPANO_PROGRAM_CATALOG')
    catalog=Path(catalog).resolve() if catalog else None
    if catalog:
        oldrun=read(catalog/'run.json')
        if sha(Path(source)/'input.png')!=oldrun['RGB_sha256'] or sha(Path(source)/'qwen/observer3d/G_obs.json')!=oldrun['G_obs_sha256']:
            raise ValueError('Catalog must be bound to the same local image AND immutable observed state')
    root=old.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='compositional')
    c=read(root/'context/experiment.json');initialize(root/'function_free',root/'context/input.png',c['hfov'],Path(c['local_model']),c['api_model'],seed,
        observations_from=root/'context',shared_observer='qwen',completion_model='compositional',worlds=worlds)
    rgb=os.environ.get('CAUPANO_LAYOUT_RGB','0')=='1'
    for label in ('context',)+EXPERIMENTS:
        path=root/label/'experiment.json';c=read(path);c.update(sdscene_bundle=str(root/'shared'),
            mechanism_source='pooled' if label=='layout_control' else 'per_backend',mechanism_policy='function_free' if label=='function_free' else 'functional',
            stage_seed_policy='hashed_stage_v1',layout_image_delivery=rgb,training_performed=False,
            api_base='https://api.siliconflow.cn/v1',sampling_law='optional_complete_functional_program_composition',
            completion_scope='fixed_observation_revalidated_rich_catalog_not_new_autonomous_proposal' if catalog else 'fresh_complete_program_proposals_given_fixed_observation')
        write(path,c)
    run=read(root/'run.json');run.update(version=12,completion_model='compositional',experiments=list(EXPERIMENTS),
        concentration=concentration,variant_concentration=reference_concentration,reference_concentration=reference_concentration,
        catalog_source=str(catalog) if catalog else None,catalog_domain_sha256=sha(catalog/'shared/domain.json') if catalog else None,layout_image_delivery=rgb,
        proposal_calls=0 if catalog else 'fresh_fixed_budget',training_performed=False,comparison_scope='Fresh conditional mechanisms/geometry on a bound rich catalog' if catalog else 'Fresh complete-program proposals, mechanisms and geometry; fixed observers',
        prior_scope='Binary presence symmetric 1/2; conditional prior uniform over programs, then over their legal camera plans')
    run['stage_token_budgets'].update(mechanism=256,variant=256);write(root/'run.json',run);return root


def propose(root,backend,p=None):
    run=read(root/'run.json')
    if not run['catalog_source']:
        result=old.propose(root,backend,p,scm=scm,continue_on_error=True)
        folder=root/'proposals'/backend;case=read(root/'shared/case.json')
        if (folder/'programs.json').exists():
            programs={fid:scm.revalidate_program(program,case) for fid,program in read(folder/'programs.json').items() if any(g['objects'] for g in program['groups'])}
            write(folder/'programs.json',programs)
        result['scope']='Fresh complete functional-program proposals, not catalog reuse';write(folder/'summary.json',result)
        return result
    # Catalog mode imports semantic candidates only. No provider is constructed.
    catalog=Path(run['catalog_source']);path=catalog/'shared/domain.json'
    if sha(path)!=run['catalog_domain_sha256']:raise ValueError('Saved rich catalog changed')
    d=read(path);case=read(root/'shared/case.json');out=root/'proposals'/backend;out.mkdir(parents=True,exist_ok=False)
    programs={};hypotheses={};sources=d.get('program_proposal_sources',{})
    for fid,p in d['programs'].items():
        if fid=='f_null':continue
        origin=sources.get(fid,p.get('proposal_source',{})).get('backend')
        if origin not in ('qwen','api'):origin='qwen'
        if origin!=backend:continue
        value=deepcopy(p);value['catalog_provenance']=dict(source=str(path),source_sha256=sha(path),program=fid,original_proposer=origin,
            old_CPT_reused=False,old_metric_layout_reused=False)
        clean=scm.revalidate_program(value,case);h=clean['hypothesis'];hypotheses[h['id']]=h;programs[fid]=clean
    write(out/'programs.json',programs);write(out/'hypotheses.json',list(hypotheses.values()))
    result=dict(status='complete',candidate_instances=sum(len(g['objects']) for p in programs.values() for g in p['groups']),programs=len(programs),
        backend=backend,api_calls=0,model_called=False,scope='Revalidated saved semantic proposals; not new proposal generation')
    write(out/'summary.json',result);return result


def pool(root,backends=('qwen','api')):
    backends=tuple(backends)
    if not backends or len(set(backends))!=len(backends) or any(b not in ('qwen','api') for b in backends):
        raise ValueError('Distinct nonempty proposal backends required')
    run=read(root/'run.json');case=read(root/'shared/case.json');programs={};hypotheses=[]
    for backend in backends:
        folder=root/'proposals'/backend;hs=read(folder/'hypotheses.json');hm={h['id']:scm.profile(backend+'_'+h['id'],h['functions']) for h in hs};hypotheses+=list(hm.values())
        for p in read(folder/'programs.json').values():
            # Collect ALL groups of one function in a complete program. Same
            # function groups in different observed-anchor areas can coexist.
            for role in sorted({g['function'] for g in p['groups']}):
                groups=[deepcopy(g) for g in p['groups'] if g['function']==role];keys={o['key'] for g in groups for o in g['objects']}
                rel=[deepcopy(r) for r in p['relations'] if r['source'] in keys or r['target'] in keys]
                if any(r[k] not in keys and not r[k].startswith('obs_') for r in rel for k in ('source','target')):
                    raise ValueError('A shared cross-role object dependency needs one program, not a silent split')
                fid='a'+str(len(programs));programs[fid]=dict(p,groups=groups,relations=rel,hypothesis=hm[p['hypothesis']['id']])
    if not programs:raise ValueError('No nonempty validated rich programs')
    base=scm.build_model(hypotheses,programs);slots={}
    for fid,p in base['furniture_states'].items():
        if fid!='f_null':slots.setdefault(p['groups'][0]['function'],['none']).append(fid)
    slots={r:slots[r] for r in sorted(slots)};tags={()}|{tuple(h['functions']) for h in hypotheses}
    profiles=[scm.profile('h'+str(i),t) for i,t in enumerate(sorted(tags))]
    choices,rejected=scm.spatial_choices(base['furniture_states'],case)
    domain=dict(version=12,hypotheses=hypotheses,profiles=profiles,recognition_profile_ids=[h['id'] for h in profiles],
        programs=base['furniture_states'],base_model=base,slots=slots,spatial_choices=choices,spatial_choice_rejections=rejected,
        positive_context=scm.positive_context(case),common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        catalog_source=run['catalog_source'],catalog_domain_sha256=run['catalog_domain_sha256'],
        scope='Complete functional-program domain; one alternative per role can contain several same-function groups and many instances')
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError(path)
    write(path,domain);k=run['draws'];n=k*(1+2*len(profiles)*len(slots))
    write(root/'call_budget.json',dict(profiles=len(profiles),roles=list(slots),draws=k,mechanism_initial_calls_per_backend=n,
        maximum_with_one_repair_each=2*n,proposal_calls=run['proposal_calls'],total_layout_attempts_per_backend=run['worlds']*len(EXPERIMENTS)))
    return dict(status='complete',programs=len(base['furniture_states'])-1,candidate_instances=len(base['component_registry']),spatial_plans=len(choices),roles=list(slots))


def mechanisms(root,backend,p=None):
    config=check(root/'context');run=read(root/'run.json');d=read(root/'shared/domain.json');p=p or provider(config,backend);k=run['draws'];plans=[]
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_with_one_repair_each']
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=True)
    def ask(name,prompt,validator,canonical):
        seed=stage_seed(run['seed'],canonical);digest=hashlib.sha256(prompt.encode()).hexdigest();cache=out/(name+'.accepted.json')
        plans.append(dict(stage=name,canonical_draw=canonical,local_seed=seed,prompt_sha256=digest,image_delivered=False,max_tokens=256))
        if cache.exists():
            record=read(cache)
            if record['prompt_sha256']!=digest or record['seed']!=seed or sha(Path(record['raw_source']))!=record['raw_sha256']:raise ValueError('Cached mechanism binding changed')
            return validator(read(record['raw_source']))
        attempt=1
        while (out/(name+'_attempt'+str(attempt)+'.txt')).exists() or (out/(name+'_attempt'+str(attempt)+'.request.json')).exists():attempt+=1
        path=out/(name+'_attempt'+str(attempt)+'.txt')
        try:result=ask_valid(SeededProvider(p,seed),None,prompt,path,validator,256)
        except ValueError:return None
        rawpath=path.with_name(path.stem+'_repair.json') if path.with_name(path.stem+'_repair.json').exists() else path.with_suffix('.json')
        write(cache,dict(prompt_sha256=digest,seed=seed,raw_source=str(rawpath),raw_sha256=sha(rawpath)))
        return result
    hs=[h['id'] for h in d['profiles']];votes=[]
    for j in range(k):
        rng=np.random.default_rng(stage_seed(run['seed'],'recognition_'+str(j)));order=[hs[i] for i in rng.permutation(len(hs))];aliases={'value_'+str(i):h for i,h in enumerate(order)}
        prompt=('Sample one plausible functional-use context for this partial observation. Functions are uncertain possible uses, not owned objects, biography or requirements. '
            'Return exactly {"choice":"one allowed alias"}.\nFIXED_POSITIVE_CONTEXT: '+json.dumps(d['positive_context'],ensure_ascii=False)+
            '\nALLOWED_CHOICES: '+json.dumps([dict(choice=a,profile=next(h for h in d['profiles'] if h['id']==v)) for a,v in aliases.items()],ensure_ascii=False))
        votes.append(ask('recognition_'+str(j),prompt,lambda v,a=aliases:a[old.sd.choice_answer(v,list(a))],'recognition_'+str(j)))
    rec=recognition.recognition_table(votes,d,run['concentration']);presence={};variants={}
    for h in d['profiles']:
        for role in d['slots']:
            av=[];vv=[];allowed=[cid for cid,v in d['spatial_choices'].items() if v['module'] in d['slots'][role][1:]]
            for j in range(k):
                canonical='presence_'+role+'_'+str(j)
                av.append(ask('presence_'+h['id']+'_'+role+'_'+str(j),scm.presence_prompt(d,h,role),scm.presence_answer,canonical))
                canonical='variant_'+role+'_'+str(j);rng=np.random.default_rng(stage_seed(run['seed'],canonical));order=[allowed[i] for i in rng.permutation(len(allowed))]
                aliases={'value_'+str(i):cid for i,cid in enumerate(order)}
                vv.append(ask('variant_'+h['id']+'_'+role+'_'+str(j),scm.variant_prompt(d,h,role,aliases),
                    lambda v,a=aliases:a[old.sd.choice_answer(v,list(a))],canonical))
            presence[h['id']+'|'+role]=scm.raw(av);variants[h['id']+'|'+role]=scm.raw(vv)
    result=dict(kind=scm.KIND,source=backend,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],
        variant_concentration=run['variant_concentration'],recognition=rec,presence=presence,variants=variants,
        original_parent_prompt_plan=plans,API_seed_sent=False,api_calls=getattr(p,'calls',None),api_usage=getattr(p,'usage',None),training_performed=False)
    path=root/'shared/mechanisms'/(backend+'.json')
    if path.exists():raise FileExistsError(path)
    write(path,result);return dict(status='complete',initial_plan_calls=len(plans),api_calls=getattr(p,'calls',None))


def pool_mechanisms(root):
    ts=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    if ts[0]['original_parent_prompt_plan']!=ts[1]['original_parent_prompt_plan']:raise ValueError('Paired prompts differ')
    if any(t['domain_sha256']!=sha(root/'shared/domain.json') for t in ts):raise ValueError('Domain mismatch')
    def merge(rows):
        c=Counter()
        for r in rows:c.update(r['counts'])
        return dict(counts=dict(c),valid_draws=sum(r['valid_draws'] for r in rows),invalid_draws=sum(r['invalid_draws'] for r in rows))
    t=deepcopy(ts[0]);t.update(source='pooled_for_layout',presence={s:merge([v['presence'][s] for v in ts]) for s in ts[0]['presence']},
        variants={s:merge([v['variants'][s] for v in ts]) for s in ts[0]['variants']})
    row=merge([v['recognition'] for v in ts]);votes=[v for v,n in row['counts'].items() for _ in range(n)]+[None]*row['invalid_draws']
    t['recognition']=recognition.recognition_table(votes,read(root/'shared/domain.json'),t['concentration'])
    path=root/'shared/mechanisms/pooled.json'
    if path.exists():raise FileExistsError(path)
    write(path,t);return dict(status='complete')


def ready(root,experiment,backend):
    source='pooled' if experiment=='layout_control' else backend;p=root/'shared/mechanisms'/(source+'.json');d=root/'shared/domain.json'
    return dict(ready=p.exists() and d.exists() and read(p)['kind']==scm.KIND and read(p)['domain_sha256']==sha(d),source=source)


def block(root,experiment,backend):
    out=root/experiment/backend/'completion';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='blocked_dependency',worlds=[],blocked_worlds=[],failures=[],world_state_ready=False,dependency=ready(root,experiment,backend))
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
            from .assembly_compositional_visual import report
            result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'):raise SystemExit(1)
    except Exception as exc:
        folder=root/'phase_errors';folder.mkdir(parents=True,exist_ok=True);error=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc))
        write(folder/(a.action+'_'+str(a.backend)+'.json'),error);p.exit(2,json.dumps(error,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
