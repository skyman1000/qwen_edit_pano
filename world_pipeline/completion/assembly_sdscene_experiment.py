"""Fixed-observation Qwen/API proposal, mechanism, and layout capability study."""
import argparse
from copy import deepcopy
import hashlib
from html import escape
import itertools
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,write,sha,new_output
from . import assembly_sdscene as sd
from . import assembly_population as population
from .assembly_experiment import initialize,check,SeededProvider,stage_seed
from .fresh_pipeline import provider
from .fresh_observer import load_case
from .assembly_observation import host_domains
from .scene_program_world import observation_board
from .world_scm_runner import ask_valid
from .world_generation import image_bytes


def prepare(output,source,seed=3407,worlds=4,draws=3,api_model=None,completion_model='sdscene'):
    if type(draws) is not int or not 1<=draws<=20:raise ValueError('draws must be 1..20')
    source=Path(source).resolve();config=check(source,strict_code=False);root=new_output(output)
    bundle=root/'shared';bundle.mkdir();(bundle/'mechanisms').mkdir()
    api_model=api_model or os.environ.get('SILICONFLOW_MODEL','deepseek-ai/DeepSeek-V4.1-Flash')
    def init(path,observations,image):
        target=initialize(path,image,config['hfov'],Path(config['local_model']),api_model,seed,
            observations_from=observations,shared_observer='qwen',completion_model=completion_model,worlds=worlds)
        c=read(target/'experiment.json');c['sampling_law']='empirical_recognition_qH_times_parent_only_inventory_pF_given_H_fixed_domain'
        write(target/'experiment.json',c);return target
    context=init(root/'context',source,source/'input.png')
    case=load_case(context,'qwen');observed=read(case['state']);case['observed_support_domains']=host_domains(case,observed)
    observation_board(case,observed,bundle/'observation_board.png');write(bundle/'case.json',case)
    for label,mechanism in [('end_to_end','per_backend'),('layout_control','pooled')]:
        target=init(root/label,context,context/'input.png');c=read(target/'experiment.json')
        c.update(sdscene_bundle=str(bundle),mechanism_source=mechanism,stage_seed_policy='hashed_stage_v1')
        write(target/'experiment.json',c)
    write(root/'run.json',dict(version=4,seed=seed,worlds=worlds,draws=draws,concentration=1.,
        stage_token_budgets=dict(proposal=8192,recognition=256,mechanism=256,layout=8192),
        backends=['qwen','api'],api_model=api_model,local_model=config['local_model'],
        RGB_sha256=sha(context/'input.png'),G_obs_sha256=sha(context/'qwen/observer3d/G_obs.json'),
        observation_source_backend='qwen',same_observation_for_API=True,
        common_image_sha256=hashlib.sha256(image_bytes(Image.open(bundle/'observation_board.png'))).hexdigest(),
        requested_generation=read(context/'experiment.json')['generation']))
    return root


def _ask(p,run,name,image,prompt,path,validator,budget):
    # Independent local draw seeds; API seed is deliberately not invented.
    seeded=SeededProvider(p,stage_seed(run['seed'],name))
    return ask_valid(seeded,image,prompt,path,validator,budget)


def propose(root,backend,p=None,scm=sd,*,continue_on_error=False):
    run=read(root/'run.json');config=check(root/'context');p=p or provider(config,backend)
    case=read(root/'shared/case.json');grounding=read(root/'context/qwen/observer2d/grounding.json')
    out=root/'proposals'/backend;out.mkdir(parents=True,exist_ok=False);(out/'calls').mkdir()
    board=Image.open(root/'shared/observation_board.png').convert('RGB')
    summary=dict(status='failed',backend=backend,programs=0,model_id=run['api_model'] if backend=='api' else run['local_model'])
    programs={};failed_queries=[]
    try:
        hypotheses=_ask(p,run,'hypotheses',board,scm.hypotheses_prompt(case,grounding),out/'calls/hypotheses.txt',
            scm.hypotheses_answer,run['stage_token_budgets']['proposal'])
        write(out/'hypotheses.json',hypotheses);programs={}
        for h in hypotheses:
            for channel in ('surrounding','belongings'):
                name='furniture_'+h['id']+'_'+channel
                try:
                    variants=_ask(p,run,name,board,scm.furniture_prompt(case,h,channel),out/'calls'/(name+'.txt'),
                        lambda value,h=h:scm.furniture_answer(value,case,h),run['stage_token_budgets']['proposal'])
                except Exception as exc:
                    if not continue_on_error:raise
                    failed_queries.append(dict(stage=name,error=type(exc).__name__+': '+str(exc)));continue
                for v in variants:
                    v['proposal_source']=dict(backend=backend,channel=channel)
                    programs['f'+str(len(programs))]=v
                write(out/'programs.json',programs)
        summary.update(status='complete',programs=len(programs),candidate_instances=sum(len(g['objects']) for v in programs.values() for g in v['groups']))
        if failed_queries:summary.update(status='partial' if programs else 'failed',failed_queries=failed_queries,
            error='One or more fixed-budget proposal stages failed; remaining planned stages were still attempted')
    except Exception as exc:
        summary['error']=type(exc).__name__+': '+str(exc)
    # Valid partial proposals remain part of D even if a later request fails.
    summary.update(programs=len(programs),candidate_instances=sum(len(g['objects']) for v in programs.values() for g in v['groups']))
    if summary['status'] in ('failed','partial') and programs:summary['partial_success']=True
    summary.update(api_calls=getattr(p,'calls',None),budget_repair_calls=len(list((out/'calls').glob('*_repair.request.json'))))
    write(out/'summary.json',summary);return summary


def pool(root):
    run=read(root/'run.json');case=read(root/'shared/case.json');hypotheses=[];programs={};profiles={};provenance={}
    for backend in run['backends']:
        folder=root/'proposals'/backend
        provenance[backend]=read(folder/'summary.json') if (folder/'summary.json').is_file() else dict(status='not_run')
        if not (folder/'hypotheses.json').is_file() or not (folder/'programs.json').is_file():continue
        source_h=read(folder/'hypotheses.json')
        for h in source_h:
            h=deepcopy(h);h['id']=backend+'_'+h['id'];hypotheses.append(h)
            tags=tuple(sorted(h['functions']))
            if tags:profiles.setdefault(tags,h)
        for fid,program in read(folder/'programs.json').items():
            program=deepcopy(program);program['hypothesis']['id']=backend+'_'+program['hypothesis']['id']
            programs[backend+'_'+fid]=program
    if not programs:raise ValueError('Neither backend produced a validated candidate program; inspect proposal summaries')
    # Neutral state IDs avoid disclosing which model proposed an option to
    # either backend's categorical mechanism. Provenance stays in domain.json.
    program_sources={}
    neutral={}
    for i,(source,program) in enumerate(programs.items()):
        fid='f'+str(i);neutral[fid]=program;program_sources[fid]=source
    base=population.build_model(hypotheses,neutral)
    if not base['component_registry']:raise ValueError('All proposals are empty; cannot evaluate hidden-world generation')
    states=[dict(id='h0',functions=[],description='Uncommitted ordinary mixed-use organization.',assumptions=['The specific functional organization is unknown.'])]
    for tags in sorted(profiles):
        h=profiles[tags];states.append(dict(id='h'+str(len(states)),functions=list(tags),description=h['description'],assumptions=h['assumptions']))
    domain=dict(version=4,hypotheses=hypotheses,programs=base['furniture_states'],base_model=base,profiles=states,
        common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        proposal_provenance=provenance,program_proposal_sources=program_sources)
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError('Domain already frozen')
    write(path,domain);return dict(profiles=len(states),programs=len(base['furniture_states']),domain_sha256=sha(path))


def mechanisms(root,backend,p=None):
    run=read(root/'run.json');config=check(root/'context');p=p or provider(config,backend)
    domain_path=root/'shared/domain.json';domain=read(domain_path);case=read(root/'shared/case.json')
    if hasattr(p,'max_calls'):p.max_calls=2*run['draws']*(1+len(domain['profiles']))
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=False)
    board=Image.open(root/'shared/observation_board.png').convert('RGB');draws=run['draws'];counts={};plans=[]
    def sample(name,image,prompt,states,budget):
        try:return _ask(p,run,name,image,prompt,out/(name+'.txt'),lambda value:sd.choice_answer(value,states),budget)
        except ValueError as exc:
            plans.append(dict(stage=name,invalid_error=str(exc)));return None
    hs=[h['id'] for h in domain['profiles']];root_draws=[]
    for j in range(draws):
        name='recognition_'+str(j);rng=np.random.default_rng(stage_seed(run['seed'],name));d=deepcopy(domain)
        d['profiles']=[domain['profiles'][i] for i in rng.permutation(len(hs))]
        prompt=sd.recognition_prompt(d,case)
        plans.append(dict(stage=name,parents=['observation_context'],image_delivered=True,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        root_draws.append(sample(name,board,prompt,hs,run['stage_token_budgets']['recognition']))
    recognition=sd.estimate_table(root_draws,hs,run['concentration'])
    fs=list(domain['base_model']['furniture_states'])
    for h in domain['profiles']:
        choices=[]
        for j in range(draws):
            name='mechanism_'+h['id']+'_'+str(j);rng=np.random.default_rng(stage_seed(run['seed'],name))
            order=[fs[i] for i in rng.permutation(len(fs))];prompt=sd.mechanism_prompt(domain,h,order)
            plans.append(dict(stage=name,parents=['H'],image_delivered=False,domain_order=order,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
            choices.append(sample(name,None,prompt,fs,run['stage_token_budgets']['mechanism']))
        counts[h['id']]=sd.estimate_table(choices,fs,run['concentration'])
    result=dict(source=backend,domain_sha256=sha(domain_path),concentration=run['concentration'],
        recognition=recognition,mechanisms=counts,protocol='empirical_parent_only_valid_JSON_channel_v1',
        API_seed_sent=False,token_logprobs_used=False,original_parent_prompt_plan=plans)
    target=root/'shared/mechanisms'/(backend+'.json')
    if target.exists():raise FileExistsError('Mechanism table already exists')
    write(target,result);write(out/'summary.json',dict(status='complete',source=backend,api_calls=getattr(p,'calls',None)))
    return dict(status='complete',backend=backend,profiles=len(counts),draws_per_profile=draws)


def pool_mechanisms(root):
    path=root/'shared/domain.json';tables=[read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    a={p['stage']:p['prompt_sha256'] for p in tables[0]['original_parent_prompt_plan'] if 'prompt_sha256' in p}
    b={p['stage']:p['prompt_sha256'] for p in tables[1]['original_parent_prompt_plan'] if 'prompt_sha256' in p}
    if a!=b:raise ValueError('Original parent-only prompts were not identical across backends')
    target=root/'shared/mechanisms/pooled.json'
    if target.exists():raise FileExistsError('Pooled mechanism already exists')
    write(target,sd.pool_tables(tables,sha(path)));return dict(status='complete',paired_parent_prompts=len(a))


def ready(root,experiment,backend):
    domain=root/'shared/domain.json';source='pooled' if experiment=='layout_control' else backend
    table=root/'shared/mechanisms'/(source+'.json')
    return dict(ready=domain.is_file() and table.is_file(),source=source,domain=str(domain),table=str(table))


def block(root,experiment,backend):
    status=ready(root,experiment,backend)
    if status['ready']:raise ValueError('Cannot block a ready experiment')
    out=root/experiment/backend/'completion';out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='blocked_dependency',worlds=[],blocked_worlds=[],failures=[],world_state_ready=False,
        error='Shared candidate domain or empirical mechanism unavailable; inspect phase_errors and proposal summaries',dependency=status)
    write(out/'summary.json',summary);return summary


def report(root):
    from .assembly_pilot_visual import build
    # Reuse the existing equal-axis/synchronized geometry gallery by making
    # links to the two backend experiments; all original artifacts stay intact.
    pages={};comparisons={}
    number=1
    while any((root/('visual_'+e+'_'+str(number).zfill(3))).exists() for e in ('end_to_end','layout_control')):number+=1
    for experiment in ('end_to_end','layout_control'):
        target=root/('visual_'+experiment+'_'+str(number).zfill(3));target.mkdir(exist_ok=False)
        for backend in ('qwen','api'):
            branch=target/backend;branch.mkdir()
            # Gallery expects METHOD/BACKEND. Both labels resolve to the exact
            # real backend folders without copying or fabricating any state.
            (branch/'qwen').symlink_to((root/experiment/backend).resolve(),target_is_directory=True)
            (branch/'input.png').symlink_to((root/experiment/'input.png').resolve())
            (branch/'experiment.json').symlink_to((root/experiment/'experiment.json').resolve())
        write(target/'pilot.json',dict(methods=['qwen','api'],comparison='sdscene_backend_'+experiment))
        page=build(target,'qwen');pages[experiment]=str(page.relative_to(root))
        paths=[root/experiment/b/'completion/structural_samples.json' for b in ('qwen','api')]
        equal=all(p.is_file() for p in paths) and sha(paths[0])==sha(paths[1])
        inventories_equal=all(p.is_file() for p in paths) and [sorted(o['id'] for o in w['objects']) for w in read(paths[0])]==[sorted(o['id'] for o in w['objects']) for w in read(paths[1])]
        comparisons[experiment]=dict(structural_samples_identical=equal,inventory_samples_identical=inventories_equal,
            purpose='isolates_metric_backend' if experiment=='layout_control' else 'includes_recognition_and_mechanism_differences')
        if experiment=='layout_control' and all(p.is_file() for p in paths) and not equal:
            raise ValueError('Layout-only control did not use identical structural samples')
    write(root/'comparison.json',comparisons)
    diagnostics={};mechanism_links=[]
    for backend in ('qwen','api'):
        path=root/'shared/mechanisms'/(backend+'.json')
        if path.is_file():
            table=read(path);rows=[list(r['probabilities'].values()) for r in table['mechanisms'].values()]
            diagnostics[backend]=dict(max_profile_mechanism_TV=max((sum(abs(a-b) for a,b in zip(x,y))/2
                for x,y in itertools.combinations(rows,2)),default=0.),
                recognition_valid=table['recognition']['valid_draws'],recognition_invalid=table['recognition']['invalid_draws'],
                mechanism_valid=sum(r['valid_draws'] for r in table['mechanisms'].values()),
                mechanism_invalid=sum(r['invalid_draws'] for r in table['mechanisms'].values()))
            if table.get('kind')=='modular':
                roles=sorted({k.split('|')[1] for k in table['mechanisms']});per_role={}
                for role in roles:
                    rr=[v['probabilities'] for k,v in table['mechanisms'].items() if k.split('|')[1]==role]
                    per_role[role]=max((sum(abs(a[s]-b[s]) for s in a)/2 for a,b in itertools.combinations(rr,2)),default=0.)
                diagnostics[backend]['per_role_max_profile_TV']=per_role
                diagnostics[backend]['max_profile_mechanism_TV']=max(per_role.values(),default=0.)
            mechanism_links.append('<a href="shared/mechanisms/'+backend+'.json">'+backend+' 经验机制</a>')
    write(root/'mechanism_diagnostics.json',diagnostics)
    errors=[]
    for path in sorted((root/'phase_errors').glob('*.json')):
        errors.append('<p>'+escape(path.name+': '+read(path)['error'])+'</p>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>本地 Qwen / API：同配置能力对照</h1>'
        '<p>两路复用同一 Qwen 观测，不重新识别图片。API 内部随机种子与视觉编码不可验证相同。</p>'+
        ''.join('<p><a href="'+page+'">'+escape(label)+' 实际三维对比</a></p>' for label,page in pages.items())+
        '<p><a href="comparison.json">固定库存一致性</a> · <a href="mechanism_diagnostics.json">机制差异与无效率</a></p>'+
        '<p>'+ ' · '.join(mechanism_links)+'</p>'+''.join(errors))
    return root/'index.html'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path)
    p.add_argument('--backend',choices=['qwen','api']);p.add_argument('--experiment',choices=['end_to_end','layout_control'])
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--worlds',type=int,default=4);p.add_argument('--draws',type=int,default=3)
    p.add_argument('--api-model');a=p.parse_args();root=a.root.resolve()
    if a.action in ('propose','mechanisms','ready','block') and a.backend is None:p.error('--backend required')
    if a.action in ('ready','block') and a.experiment is None:p.error('--experiment required')
    try:
        if a.action=='prepare':
            if a.source_root is None:p.error('--source-root required')
            result=prepare(root,a.source_root,a.seed,a.worlds,a.draws,a.api_model)
        elif a.action=='propose':result=propose(root,a.backend)
        elif a.action=='pool':result=pool(root)
        elif a.action=='mechanisms':result=mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms':result=pool_mechanisms(root)
        elif a.action=='ready':
            result=ready(root,a.experiment,a.backend)
            if not result['ready']:print(json.dumps(result));raise SystemExit(1)
        elif a.action=='block':result=block(root,a.experiment,a.backend)
        else:result=report(root)
        print(json.dumps(result,ensure_ascii=False,default=str))
        if isinstance(result,dict) and result.get('status')=='failed':raise SystemExit(1)
    except Exception as exc:
        errors=root/'phase_errors';errors.mkdir(parents=True,exist_ok=True)
        result=dict(action=a.action,backend=a.backend,error=type(exc).__name__+': '+str(exc))
        write(errors/(a.action+'_'+str(a.backend)+'.json'),result)
        p.exit(2,json.dumps(result,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
