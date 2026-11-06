"""Shared-domain local Qwen / SiliconFlow DeepSeek functional-assembly study."""
import argparse
from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,write,sha
from . import assembly_modular as modular,assembly_sdscene as sd,assembly_sdscene_experiment as previous
from .assembly_experiment import initialize,check,stage_seed
from .fresh_pipeline import provider


def prepare(root,source,seed=3407,worlds=4,draws=4):
    root=previous.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='modular')
    config=read(root/'context/experiment.json')
    extra=initialize(root/'function_free',root/'context/input.png',config['hfov'],Path(config['local_model']),
        config['api_model'],seed,observations_from=root/'context',shared_observer='qwen',completion_model='modular',worlds=worlds)
    for label in ('context','end_to_end','layout_control','function_free'):
        path=root/label/'experiment.json';c=read(path)
        c.update(sampling_law='qH_product_role_pM_given_H_finite_grammar',stage_seed_policy='hashed_stage_v1',
            sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend',
            modular_policy='function_free' if label=='function_free' else 'functional')
        write(path,c)
    run=read(root/'run.json');run.update(version=5,completion_model='modular',experiments=['end_to_end','layout_control','function_free'])
    write(root/'run.json',run);return root


def propose(root,backend,p=None):return previous.propose(root,backend,p,scm=modular)


def pool(root):
    case=read(root/'shared/case.json');run=read(root/'run.json');hypotheses=[];programs={};sources={};tagsets=set();provenance={}
    for backend in ('qwen','api'):
        folder=root/'proposals'/backend
        provenance[backend]=read(folder/'summary.json') if (folder/'summary.json').is_file() else dict(status='not_run')
        if not (folder/'programs.json').is_file():continue
        hm={h['id']:modular.profile(backend+'_'+h['id'],h['functions']) for h in read(folder/'hypotheses.json')}
        hypotheses+=list(hm.values());tagsets.update(tuple(h['functions']) for h in hm.values() if h['functions'])
        for pid,program in read(folder/'programs.json').items():
            keys=sorted({g['assembly_key'] for g in program['groups'] if g['objects']})
            for groupkey in keys:
                groups=[deepcopy(g) for g in program['groups'] if g['assembly_key']==groupkey and g['objects']]
                role=groups[0]['function'];objkeys={o['key'] for g in groups for o in g['objects']}
                relations=[deepcopy(r) for r in program['relations'] if r['source'] in objkeys or r['target'] in objkeys]
                if any(r[k] not in objkeys and not r[k].startswith('obs_') for r in relations for k in ('source','target')):
                    raise ValueError('A module has a dependency on an unselected assembly')
                fid='a'+str(len(programs));h=hm[program['hypothesis']['id']]
                programs[fid]=dict(description='Optional '+role+' assembly: '+', '.join(o['variant'] for g in groups for o in g['objects']),
                    hypothesis=h,groups=groups,relations=relations,relation_bindings=[],inventory_scope='one_optional_functional_assembly')
                sources[fid]=dict(backend=backend,program=pid,assembly=groupkey)
    if not programs:raise ValueError('No nonempty validated functional assembly was proposed')
    base=modular.build_model(hypotheses,programs);slots={}
    for fid,program in base['furniture_states'].items():
        if fid=='f_null':continue
        role=program['groups'][0]['function'];slots.setdefault(role,['none']).append(fid)
    slots={r:slots[r] for r in sorted(slots)}
    profiles=[modular.profile('h0',[])]+[modular.profile('h'+str(i+1),tags) for i,tags in enumerate(sorted(tagsets))]
    domain=dict(version=5,hypotheses=hypotheses,profiles=profiles,programs=base['furniture_states'],base_model=base,slots=slots,
        common_inventory=[dict(id=o['id'],name=o['description']) for o in case['context']],
        RGB_sha256=run['RGB_sha256'],G_obs_sha256=run['G_obs_sha256'],hfov=case['hfov'],image_size=case['image_size'],
        proposal_provenance=provenance,program_proposal_sources=sources,
        conditional_independence_assumed='One optional assembly per role, independent given canonical H and fixed domain')
    path=root/'shared/domain.json'
    if path.exists():raise FileExistsError('Domain already frozen')
    write(path,domain);return dict(profiles=len(profiles),roles=list(slots),assemblies=len(base['component_registry']))


def mechanisms(root,backend,p=None):
    config=check(root/'context');run=read(root/'run.json');p=p or provider(config,backend)
    domain=read(root/'shared/domain.json');case=read(root/'shared/case.json');draws=run['draws'];hs=[h['id'] for h in domain['profiles']]
    if hasattr(p,'max_calls'):p.max_calls=2*draws*(1+len(hs)*len(domain['slots']))
    out=root/'mechanism_calls'/backend;out.mkdir(parents=True,exist_ok=False)
    board=Image.open(root/'shared/observation_board.png').convert('RGB');plans=[];counts={}
    def sample(name,image,prompt,states):
        plans.append(dict(stage=name,parents=['observation_context'] if image is not None else ['H'],image_delivered=image is not None,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        try:return previous._ask(p,run,name,image,prompt,out/(name+'.txt'),lambda v:sd.choice_answer(v,states),256)
        except ValueError as exc:plans.append(dict(stage=name,invalid_error=str(exc)));return None
    def permuted(values,name):
        rng=np.random.default_rng(stage_seed(run['seed'],name));return [values[i] for i in rng.permutation(len(values))]
    votes=[]
    for j in range(draws):
        name='recognition_'+str(j);d=dict(domain,profiles=permuted(domain['profiles'],name))
        votes.append(sample(name,board,modular.recognition_prompt(d,case),hs))
    recognition=sd.estimate_table(votes,hs,run['concentration'])
    for h in domain['profiles']:
        for role,states in domain['slots'].items():
            votes=[]
            for j in range(draws):
                name='assembly_'+h['id']+'_'+role+'_'+str(j)
                votes.append(sample(name,None,modular.mechanism_prompt(domain,h,role,permuted(states,name)),states))
            counts[h['id']+'|'+role]=sd.estimate_table(votes,states,run['concentration'])
    result=dict(kind='modular',source=backend,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],
        recognition=recognition,mechanisms=counts,original_parent_prompt_plan=plans,API_seed_sent=False,token_logprobs_used=False)
    path=root/'shared/mechanisms'/(backend+'.json')
    if path.exists():raise FileExistsError('Mechanism already frozen')
    write(path,result);return dict(status='complete',profiles=len(hs),roles=len(domain['slots']),draws=draws)


def report(root):
    previous.report(root)
    from .assembly_pilot_visual import build
    number=1
    while (root/('visual_function_free_'+str(number).zfill(3))).exists():number+=1
    target=root/('visual_function_free_'+str(number).zfill(3));target.mkdir()
    for backend in ('qwen','api'):
        branch=target/backend;branch.mkdir();(branch/'qwen').symlink_to((root/'function_free'/backend).resolve(),target_is_directory=True)
        for name in ('input.png','experiment.json'):(branch/name).symlink_to((root/'function_free'/name).resolve())
    write(target/'pilot.json',dict(methods=['qwen','api'],comparison='sdscene_backend_function_free'))
    page=build(target,'qwen');effects={}
    for backend in ('qwen','api'):
        effects[backend]={}
        for label in ('end_to_end','function_free'):
            path=root/label/backend/'completion/probability_audit.json'
            if path.is_file():effects[backend][label]=read(path)
    write(root/'functional_edge_ablation.json',effects)
    links=[]
    for backend in ('qwen','api'):
        folder=root/('visual_edge_ablation_'+backend+'_'+str(number).zfill(3));folder.mkdir()
        for method,label in (('functional','end_to_end'),('function_free','function_free')):
            branch=folder/method;branch.mkdir();(branch/'qwen').symlink_to((root/label/backend).resolve(),target_is_directory=True)
            for name in ('input.png','experiment.json'):(branch/name).symlink_to((root/label/name).resolve())
        write(folder/'pilot.json',dict(methods=['functional','function_free'],comparison='modular_edge_ablation_'+backend))
        comparison=build(folder,'qwen');links.append('<a href="'+str(comparison.relative_to(root))+'">'+backend+' 功能边消融三维对比</a>')
    index=root/'index.html';index.write_text(index.read_text().replace('<h1>','<h1>v5 功能装配语法：',1)+
        '<p><a href="'+str(page.relative_to(root))+'">H→装配边消融的三维对比</a> · '
        '<a href="functional_edge_ablation.json">同域功能边消融的实际类别与数量概率</a></p><p>'+' · '.join(links)+'</p>')
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path);p.add_argument('--backend',choices=['qwen','api'])
    p.add_argument('--experiment',choices=['end_to_end','layout_control','function_free']);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--worlds',type=int,default=4);p.add_argument('--draws',type=int,default=4);a=p.parse_args();root=a.root.resolve()
    if a.action in ('propose','mechanisms','ready','block') and a.backend is None:p.error('--backend is required')
    if a.action in ('ready','block') and a.experiment is None:p.error('--experiment is required')
    try:
        if a.action=='prepare':
            if a.source_root is None:p.error('--source-root is required')
            result=prepare(root,a.source_root,a.seed,a.worlds,a.draws)
        elif a.action=='propose':result=propose(root,a.backend)
        elif a.action=='pool':result=pool(root)
        elif a.action=='mechanisms':result=mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms':result=previous.pool_mechanisms(root)
        elif a.action=='ready':
            result=previous.ready(root,a.experiment,a.backend)
            if not result['ready']:print(json.dumps(result));raise SystemExit(1)
        elif a.action=='block':result=previous.block(root,a.experiment,a.backend)
        else:result=report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status')=='failed':raise SystemExit(1)
    except Exception as exc:
        errors=root/'phase_errors';errors.mkdir(parents=True,exist_ok=True)
        result=dict(error=type(exc).__name__+': '+str(exc),action=a.action,backend=a.backend)
        write(errors/(a.action+'_'+str(a.backend)+'.json'),result);p.exit(2,json.dumps(result,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
