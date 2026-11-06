"""v6 matched local/API experiments with joint activation and conditional variants."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read, write, sha
from . import assembly_activation as scm, assembly_sdscene as sd
from . import assembly_sdscene_experiment as previous, assembly_modular_experiment as modular
from .assembly_experiment import initialize, check, stage_seed
from .fresh_pipeline import provider


def prepare(root, source, seed=3407, worlds=4, draws=4):
    root = previous.prepare(root,source,seed,worlds,draws,'deepseek-ai/DeepSeek-V4.1-Flash',completion_model='activation')
    c = read(root/'context/experiment.json')
    initialize(root/'function_free',root/'context/input.png',c['hfov'],Path(c['local_model']),c['api_model'],seed,
        observations_from=root/'context',shared_observer='qwen',completion_model='activation',worlds=worlds)
    for label in ('context','end_to_end','layout_control','function_free'):
        path = root/label/'experiment.json'; config = read(path)
        config.update(sampling_law='qH_p_joint_activation_product_conditional_variants',stage_seed_policy='hashed_stage_v1',
            sdscene_bundle=str(root/'shared'),mechanism_source='pooled' if label=='layout_control' else 'per_backend',
            activation_policy='function_free' if label=='function_free' else 'functional')
        write(path,config)
    run = read(root/'run.json'); run.update(version=6,completion_model='activation',
        experiments=['end_to_end','layout_control','function_free'],proposal_domain='shared_union_not_autonomous_backend_completion')
    write(root/'run.json',run); return root


def propose(root, backend, p=None):
    return previous.propose(root,backend,p,scm=scm)


def pool(root):
    # Reuse validated host-local module extraction; no old mechanism or geometry.
    modular.pool(root); path = root/'shared/domain.json'; domain = read(path)
    roles = list(domain['slots']); tags = {tuple(h['functions']) for h in domain['profiles']}
    tags.add(tuple()); tags.update((r,) for r in roles)
    # Matched presence/absence of every role at a common full-role context.
    tags.add(tuple(sorted(roles))); tags.update(tuple(sorted(set(roles)-{r})) for r in roles)
    profiles = [scm.profile('h'+str(i),t) for i,t in enumerate(sorted(tags))]
    pairs = []
    for role in roles:
        for h in profiles:
            if role in h['functions']: continue
            other = next((t for t in profiles if set(t['functions'])==set(h['functions'])|{role}),None)
            if other is not None: pairs.append(dict(role=role,without=h['id'],with_role=other['id']))
    domain.update(version=6, profiles=profiles, matched_profile_pairs=pairs,
        activation_patterns=scm.activation_states(roles), role_order=roles,
        conditional_independence_assumed='Joint role activation; variants independent given H and their activation',
        activation_prior='Uniform over all binary patterns: each role prior presence 1/2; independent of variant counts',
        profile_support_note='Proposed tags plus matched diagnostic support; added H states are not image evidence')
    write(path,domain)
    run = read(root/'run.json'); k = run['draws']; calls = k*(1+len(profiles)*(1+len(roles)))
    write(root/'call_budget.json',dict(profiles=len(profiles),roles=len(roles),draws=k,
        mechanism_initial_calls_per_backend=calls,maximum_with_one_repair_each=2*calls,
        proposal_and_layout_calls_extra=True,API_seed_supported=False))
    return dict(status='complete',profiles=len(profiles),roles=roles,matched_pairs=len(pairs),initial_calls_per_backend=calls)


def mechanisms(root, backend, p=None):
    config = check(root/'context'); run = read(root/'run.json'); p = p or provider(config,backend)
    domain = read(root/'shared/domain.json'); case = read(root/'shared/case.json'); k = run['draws']
    roles = domain['role_order']; hs = [h['id'] for h in domain['profiles']]
    if hasattr(p,'max_calls'): p.max_calls = 2*k*(1+len(hs)*(1+len(roles)))
    out = root/'mechanism_calls'/backend; out.mkdir(parents=True,exist_ok=False)
    board = Image.open(root/'shared/observation_board.png').convert('RGB'); plans = []
    def sample(name,image,prompt,validator,parents):
        plans.append(dict(stage=name,parents=parents,image_delivered=image is not None,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        try: return previous._ask(p,run,name,image,prompt,out/(name+'.txt'),validator,256)
        except ValueError as exc: plans.append(dict(stage=name,invalid_error=str(exc))); return None
    def permute(values,name):
        rng = np.random.default_rng(stage_seed(run['seed'],name)); return [values[i] for i in rng.permutation(len(values))]
    votes = []
    for j in range(k):
        name = 'recognition_'+str(j); d = dict(domain,profiles=permute(domain['profiles'],name))
        votes.append(sample(name,board,scm.recognition_prompt(d,case),lambda v:sd.choice_answer(v,hs),['O','D']))
    recognition = sd.estimate_table(votes,hs,run['concentration']); activation = {}; variants = {}
    for h in domain['profiles']:
        votes = []
        for j in range(k):
            name = 'activation_'+h['id']+'_'+str(j)
            votes.append(sample(name,None,scm.activation_prompt(domain,h,permute(roles,name)),
                lambda v:scm.activation_answer(v,roles),['H','D']))
        activation[h['id']] = sd.estimate_table(votes,domain['activation_patterns'],run['concentration'])
        for role, states in domain['slots'].items():
            allowed = [s for s in states if s!='none']; votes = []
            for j in range(k):
                name = 'variant_'+h['id']+'_'+role+'_'+str(j)
                votes.append(sample(name,None,scm.variant_prompt(domain,h,role,permute(allowed,name)),
                    lambda v,allowed=allowed:sd.choice_answer(v,allowed),['H','A_'+role+'=1','D']))
            variants[h['id']+'|'+role] = sd.estimate_table(votes,allowed,run['concentration'])
    result = dict(kind='joint_activation',source=backend,domain_sha256=sha(root/'shared/domain.json'),
        concentration=run['concentration'],recognition=recognition,activation=activation,variants=variants,
        original_parent_prompt_plan=plans,API_seed_sent=False,token_logprobs_used=False,
        probabilities_calibrated=False,invalid_policy='One repair; invalid draws recorded, not interpreted as empty worlds')
    path = root/'shared/mechanisms'/(backend+'.json')
    if path.exists(): raise FileExistsError('Mechanism already frozen')
    write(path,result); return dict(status='complete',profiles=len(hs),roles=len(roles),draws=k)


def pool_mechanisms(root):
    path = root/'shared/domain.json'; tables = [read(root/'shared/mechanisms'/(b+'.json')) for b in ('qwen','api')]
    plans = [{p['stage']:p['prompt_sha256'] for p in t['original_parent_prompt_plan'] if 'prompt_sha256' in p} for t in tables]
    if plans[0]!=plans[1]: raise ValueError('Parent-only prompts differ across backends')
    if any(t['domain_sha256']!=sha(path) for t in tables): raise ValueError('Mechanism domain differs')
    if tables[0]['concentration']!=tables[1]['concentration']: raise ValueError('Smoothing differs')
    def combine(rows):
        states = list(rows[0]['counts']); draws = []
        for row in rows:
            if set(row['counts'])!=set(states): raise ValueError('States differ')
            for s,n in row['counts'].items(): draws.extend([s]*n)
            draws.extend([None]*row['invalid_draws'])
        return sd.estimate_table(draws,states,tables[0]['concentration'])
    result = dict(kind='joint_activation',source='pooled_for_geometry_control',domain_sha256=sha(path),
        concentration=tables[0]['concentration'],recognition=combine([t['recognition'] for t in tables]))
    for field in ('activation','variants'):
        if set(tables[0][field])!=set(tables[1][field]): raise ValueError('Mechanism keys differ')
        result[field] = {k:combine([t[field][k] for t in tables]) for k in tables[0][field]}
    target = root/'shared/mechanisms/pooled.json'
    if target.exists(): raise FileExistsError('Pooled mechanism already frozen')
    write(target,result); return dict(status='complete',paired_parent_prompts=len(plans[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','propose','pool','mechanisms','pool-mechanisms','ready','block','report'])
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--source-root',type=Path)
    parser.add_argument('--backend',choices=['qwen','api']); parser.add_argument('--experiment',choices=['end_to_end','layout_control','function_free'])
    parser.add_argument('--seed',type=int,default=3407); parser.add_argument('--worlds',type=int,default=4); parser.add_argument('--draws',type=int,default=4)
    a = parser.parse_args(); root = a.root.resolve()
    if a.action in ('propose','mechanisms','ready','block') and a.backend is None: parser.error('--backend required')
    if a.action in ('ready','block') and a.experiment is None: parser.error('--experiment required')
    try:
        if a.action=='prepare':
            if a.source_root is None: parser.error('--source-root required')
            result = prepare(root,a.source_root,a.seed,a.worlds,a.draws)
        elif a.action=='propose': result = propose(root,a.backend)
        elif a.action=='pool': result = pool(root)
        elif a.action=='mechanisms': result = mechanisms(root,a.backend)
        elif a.action=='pool-mechanisms': result = pool_mechanisms(root)
        elif a.action=='ready':
            result = previous.ready(root,a.experiment,a.backend)
            if not result['ready']: print(json.dumps(result)); raise SystemExit(1)
        elif a.action=='block': result = previous.block(root,a.experiment,a.backend)
        else:
            from .assembly_activation_visual import report
            result = report(root)
        print(json.dumps(result,default=str,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status') in ('failed','partial'): raise SystemExit(1)
    except Exception as exc:
        errors = root/'phase_errors'; errors.mkdir(parents=True,exist_ok=True)
        result = dict(error=type(exc).__name__+': '+str(exc),action=a.action,backend=a.backend)
        write(errors/(a.action+'_'+str(a.backend)+'.json'),result); parser.exit(2,json.dumps(result,ensure_ascii=False)+'\n')


if __name__=='__main__': main()
