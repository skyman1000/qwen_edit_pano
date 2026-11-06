"""Fresh per-image v13 domains, matched backends, fixed-budget multi-scene tests."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

from PIL import Image

from ..common import ROOT,new_output,read,rows,sha,write
from . import assembly_compositional_experiment as fresh,assembly_organized_experiment as organized
from . import assembly_organized as scm,assembly_compositional as compositional
from .assembly_experiment import initialize,check,code_hashes,SeededProvider,stage_seed
from .fresh_pipeline import provider
from .fresh_observer import observe,load_case
from .qwen_upstream import DEFAULT_MODEL

DEFAULT_INPUTS=ROOT/'qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl'
EXPERIMENTS=('end_to_end','layout_control','function_free')
PHASES=('observer2d','observer3d','prepare-scenes','propose','pool','mechanisms','pool-mechanisms','worlds')


def hashes():
    p=ROOT/'qwen_edit_pano/scripts/organized_multiscene.sh'
    return {**code_hashes(),str(p):sha(p)}


def output_plan(mode,worlds=None):
    if mode not in ('single','evaluation'):raise ValueError('mode must be single or evaluation')
    if worlds is None:worlds=1 if mode=='single' else 4
    if type(worlds) is not int or not 1<=worlds<=8:raise ValueError('worlds must be 1..8')
    if mode=='single' and worlds!=1:raise ValueError('single mode generates exactly one complete world per input/backend')
    return worlds,('end_to_end',) if mode=='single' else EXPERIMENTS


def initialize_suite(root,inputs=DEFAULT_INPUTS,seed=3407,worlds=None,draws=4,local_model=None,mode='evaluation'):
    worlds,experiments=output_plan(mode,worlds)
    if type(worlds) is not int or not 1<=worlds<=8:raise ValueError('worlds must be 1..8')
    if draws not in (4,8,12,16,20):raise ValueError('Use complete four-encoding blocks: draws=4,8,12,16,20')
    if os.environ.get('CAUPANO_PROGRAM_CATALOG'):raise ValueError('Fresh images require fresh candidates; unset CAUPANO_PROGRAM_CATALOG')
    records=rows(inputs)
    if not records:raise ValueError('Empty input manifest')
    clean=[]
    for r in records:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']):raise ValueError('Unsafe scene id')
        image=Path(r['local_rgb']).resolve();hfov=r['hfov_degrees']
        if not image.is_file() or type(hfov) not in (int,float) or not math.isfinite(hfov) or not 0<hfov<180:raise ValueError('Existing image and finite HFOV required')
        if r.get('local_sha256') and sha(image)!=r['local_sha256']:raise ValueError('Input hash changed: '+r['id'])
        clean.append(dict(id=r['id'],image=str(image),image_sha256=sha(image),hfov=hfov,split=r.get('split','unspecified')))
    root=new_output(root);(root/'scenes').mkdir();(root/'phase_records').mkdir()
    suite=dict(kind='organized_v13_fresh_multiscene',seed=seed,worlds=worlds,draws=draws,concentration=1.,variant_concentration=1.,
        scenes=clean,mode=mode,experiments=list(experiments),local_model=str(Path(local_model or DEFAULT_MODEL).resolve()),
        api_model='deepseek-ai/DeepSeek-V4.1-Flash',api_base='https://api.siliconflow.cn/v1',
        layout_image_delivery=False,shared_observer='qwen_2d_official_WildDet3D',same_observation_and_candidate_domain_for_backends=True,
        observer_interface='chunked_with_whole_image_reconciliation_v2',numeric_request_interface='typed_coordinates_lossless_envelopes_v3',
        api_transport_policy='shared_start_spacing_bounded_HTTP429_only_v1',
        old_candidate_catalog_reused=False,training_performed=False,GT_world_used_for_generation=False,
        code_sha256=hashes(),input_manifest_sha256=sha(inputs),geometry_control_omitted='No old per-image world to replay',
        interface_probe_omitted='Historical old-H diagnostic is not part of the generative model')
    write(root/'suite.json',suite)
    for r in clean:
        s=root/'scenes'/r['id'];s.mkdir()
        initialize(s/'observation',Path(r['image']),r['hfov'],Path(suite['local_model']),suite['api_model'],seed,completion_model='compositional',worlds=worlds)
        config=read(s/'observation/experiment.json');config['id']=r['id'];write(s/'observation/experiment.json',config)
        write(s/'input_source.json',r)
    return suite


def verify_suite(root):
    suite=read(root/'suite.json')
    if suite.get('kind')!='organized_v13_fresh_multiscene':raise ValueError('Use a newly initialized multi-scene suite, not an old run')
    changed=[p for p,h in suite['code_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    if changed:raise ValueError('Code changed since suite initialization: '+str(changed))
    for r in suite['scenes']:
        if sha(Path(r['image']))!=r['image_sha256']:raise ValueError('Original input changed: '+r['id'])
        check(root/'scenes'/r['id']/'observation')
    if (root/'recovery_manifest.json').exists():
        from .assembly_multiscene_fork import verify_recovery
        verify_recovery(root)
    return suite


def prepare_scene(scene,suite):
    if os.environ.get('CAUPANO_PROGRAM_CATALOG'):raise ValueError('Unset CAUPANO_PROGRAM_CATALOG; no cross-image catalog reuse')
    root=scene/'completion'
    fresh.prepare(root,scene/'observation',suite['seed'],suite['worlds'],suite['draws'],suite['concentration'],suite['variant_concentration'])
    for label in ('context',)+EXPERIMENTS:
        p=root/label/'experiment.json';c=read(p);c.update(completion_revision='organized_v13',model_variant=scm.MODEL_VARIANT,response_interface=scm.INTERFACE,
            layout_image_delivery=False,numeric_request_interface='typed_coordinates_lossless_envelopes_v3',completion_scope='fresh_same_image_domain_organized_v13_no_training')
        write(p,c)
    active=suite.get('experiments',list(EXPERIMENTS))
    run=read(root/'run.json');run.update(version=13,completion_model='organized',experiments=list(active),interface_probe=False,
        output_mode=suite.get('mode','evaluation'),
        source_manifest={},comparison_scope='Fresh per-image shared proposals; v13 mechanisms and frozen-dimension geometry',
        geometry_control_available=False,stage_token_budgets=dict(proposal=8192,recognition=256,mechanism=256,layout=8192))
    if 'backends' in suite:run['backends']=list(suite['backends'])
    write(root/'run.json',run);return dict(status='complete',root=str(root))


def pool_domain(root,backends=('qwen','api')):
    # v12 pooling defines the unchanged complete-program grain; no v12 model
    # inference is run. Organization profiles are installed before v13 queries.
    backends=tuple(backends)
    result=fresh.pool(root,backends);d=read(root/'shared/domain.json');write(root/'shared/proposal_domain.json',deepcopy(d))
    profiles=scm.organization_profiles(d['slots']);d.update(version=13,profiles=profiles,recognition_profile_ids=[h['id'] for h in profiles],
        candidate_proposals_reused=False,organization_semantics='Uncertain functional organization emphasis, no resident biography',
        proposal_provenance={b:read(root/'proposals'/b/'summary.json') for b in backends})
    write(root/'shared/domain.json',d);run=read(root/'run.json');k=run['draws'];n=k*(1+2*len(profiles)*len(d['slots']))
    max_groups=sum(max(len(d['programs'][p]['groups']) for p in programs[1:]) for programs in d['slots'].values())
    max_batches=sum(max(sum(math.ceil(len(g['objects'])/6) for g in d['programs'][p]['groups']) for p in programs[1:]) for programs in d['slots'].values())
    write(root/'call_budget.json',dict(profiles=len(profiles),roles=list(d['slots']),draws=k,initial_main_calls_per_backend=n,
        maximum_mechanism_calls_with_one_repair_each=2*n,initial_interface_probe_calls_per_backend=0,
        layout_worlds_per_backend=run['worlds']*len(run['experiments']),dimension_initial_calls_per_nonempty_world=1,
        placement_initial_calls_per_group=1,maximum_groups_in_one_world=max_groups,
        maximum_layout_calls_per_nonempty_world_with_one_schema_repair_each=2+4*max_batches,
        numeric_request_interface='typed_coordinates_lossless_envelopes_v3',maximum_partial_coordinate_passes_per_batch=2,
        maximum_requested_coordinate_objects_per_batch=6,maximum_coordinate_batches_in_one_world=max_batches,layout_budget_is_conservative=True,
        automatic_transport_retries='HTTP429_only_at_most_3_retries_per_logical_request',scene_gt_used=False))
    return dict(result,status='complete')


def estimate(root,backend,p):
    """Exactly v13 main choices/smoothing, without the historical old-H probe."""
    check(root/'context');run=read(root/'run.json');d=read(root/'shared/domain.json')
    target=root/'shared/mechanisms'/(backend+'.json')
    if target.exists():raise FileExistsError(target)
    if hasattr(p,'max_calls'):p.max_calls=read(root/'call_budget.json')['maximum_mechanism_calls_with_one_repair_each']
    request=organized.Requests(p,root,backend,run['seed']);votes=[];presence={};variants={};k=run['draws']
    for j in range(k):
        name='recognition_'+str(j);aliases=organized.aliases_for([h['id'] for h in d['profiles']],run['seed'],name)
        votes.append(request.ask(name,scm.recognition_prompt(d,aliases),lambda v,a=aliases:organized.choice(v,a),name))
    recognition=fresh.recognition.recognition_table(votes,d,run['concentration'])
    for h in d['profiles']:
        for role,programs in d['slots'].items():
            av=[];vv=[];states=[s for s,v in d['spatial_choices'].items() if v['module'] in programs[1:]]
            for j in range(k):
                name='presence_'+role+'_'+str(j);aliases=organized.binary_aliases(j)
                av.append(request.ask('presence_'+h['id']+'_'+role+'_'+str(j),scm.presence_prompt(d,h,role,aliases),lambda v,a=aliases:organized.choice(v,a),name))
                name='variant_'+role+'_'+str(j);aliases=organized.aliases_for(states,run['seed'],name)
                vv.append(request.ask('variant_'+h['id']+'_'+role+'_'+str(j),scm.variant_prompt(d,h,role,aliases),lambda v,a=aliases:organized.choice(v,a),name))
            presence[h['id']+'|'+role]=compositional.raw(av);variants[h['id']+'|'+role]=compositional.raw(vv)
    table=dict(kind=scm.KIND,source=backend,domain_sha256=sha(root/'shared/domain.json'),concentration=run['concentration'],variant_concentration=run['variant_concentration'],
        recognition=recognition,presence=presence,variants=variants,original_parent_prompt_plan=request.plans,
        training_performed=False,API_seed_sent=False,estimator_scope='Same counterbalanced empirical v13 choices; no calibrated world probabilities')
    write(target,table);return dict(status='complete',main_initial_calls=len(request.plans))


def missing_inputs(scene,phase,backend,proposal_backends=('qwen','api')):
    obs=scene/'observation';comp=scene/'completion'
    if phase=='observer2d':required=[obs/'input.png',obs/'experiment.json']
    elif phase=='observer3d':required=[obs/'qwen/observer2d/G_2d.json',obs/'qwen/observer2d/summary.json']
    elif phase=='prepare-scenes':required=[obs/'qwen/observer3d/G_obs.json',obs/'qwen/observer2d/grounding.json']
    elif phase=='propose':required=[comp/'context/experiment.json',comp/'shared/case.json']
    elif phase=='pool':required=[comp/'run.json']+[comp/'proposals'/b/n for b in proposal_backends for n in ('programs.json','hypotheses.json')]
    elif phase=='mechanisms':required=[comp/'shared/domain.json',comp/'context/experiment.json']
    elif phase=='pool-mechanisms':
        required=[comp/'run.json']
        if required[0].exists() and 'layout_control' in read(required[0])['experiments']:required += [comp/'shared/mechanisms'/(b+'.json') for b in ('qwen','api')]
    elif phase=='worlds':
        required=[comp/'run.json',comp/'shared/domain.json',comp/'shared/mechanisms'/(backend+'.json')]
    else:raise ValueError('Unknown phase')
    return [str(p) for p in required if not p.is_file()]


def run_phase(root,phase,backend='qwen'):
    root=root.resolve();suite=verify_suite(root);results=[];local=None
    for r in suite['scenes']:
        scene=root/'scenes'/r['id'];marker=root/'phase_records'/(r['id']+'_'+phase+'_'+backend+'.json')
        if marker.exists():results.append(read(marker));continue  # no stochastic retry until success
        row=dict(scene=r['id'],phase=phase,backend=backend,status='failed')
        absent=missing_inputs(scene,phase,backend,suite.get('backends',('qwen','api')))
        if absent:
            row.update(status='blocked_dependency',missing_inputs=absent,model_called=False)
            write(marker,row);results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True);continue
        if phase=='observer3d':
            upstream=read(scene/'observation/qwen/observer2d/summary.json')
            if upstream.get('status')!='completed':
                row.update(status='blocked_dependency',upstream_phase='observer2d',
                    upstream_status=upstream.get('status'),error=upstream.get('error','2D observation not completed'),model_called=False)
                write(marker,row);results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True);continue
        try:
            obs=scene/'observation';comp=scene/'completion'
            def client(config):
                nonlocal local
                if backend=='api':
                    from .assembly_api_pacing import wrap
                    return wrap(provider(config,'api'))
                if local is None:local=provider(config,'qwen')
                return local
            if phase=='observer2d':
                if backend!='qwen':raise ValueError('One shared Qwen observer is required')
                from .assembly_io_recovery import observe_chunked
                config=check(obs);answer=observe_chunked(SeededProvider(client(config),stage_seed(suite['seed'],'shared_observer')),obs,'qwen',config['max_tokens'])
            elif phase=='observer3d':
                from .fresh_wilddet import run,DEFAULT_REPO,DEFAULT_CHECKPOINT
                answer=run(obs,'qwen',DEFAULT_REPO,DEFAULT_CHECKPOINT,check_experiment=check)
            elif phase=='prepare-scenes':load_case(obs,'qwen');answer=prepare_scene(scene,suite)
            elif phase=='propose':
                answer=fresh.propose(comp,backend,client(check(comp/'context')))
                # Empty proposal lists record a failed proposer, never an empty
                # world or a default probability row. Valid other proposals survive.
                for filename,empty in (('hypotheses.json',[]),('programs.json',{})):
                    path=comp/'proposals'/backend/filename
                    if not path.exists():write(path,empty)
            elif phase=='pool':answer=pool_domain(comp,suite.get('backends',('qwen','api')))
            elif phase=='mechanisms':
                read(comp/'shared/domain.json')
                answer=estimate(comp,backend,client(check(comp/'context')))
            elif phase=='pool-mechanisms':
                answer=organized.pool_mechanisms(comp) if 'layout_control' in read(comp/'run.json')['experiments'] else dict(status='complete',skipped=True,reason='Single output mode has no layout-control branch')
            elif phase=='worlds':
                answer={}
                for branch in read(comp/'run.json')['experiments']:
                    try:
                        config=check(comp/branch);saved=comp/branch/backend/'completion'
                        if saved.exists():
                            from .assembly_coordinate_transport import recover_layouts
                            previous=read(saved/'summary.json')
                            answer[branch]=recover_layouts(comp,branch,backend,client(config)) if previous.get('failures') else previous
                        elif organized.ready(comp,branch,backend):
                            from .assembly_world import _run
                            from . import assembly_coordinate_transport as numeric
                            selected=client(config)
                            if hasattr(selected,'max_calls'):
                                selected.max_calls=max(selected.max_calls,read(comp/'call_budget.json')['maximum_layout_calls_per_nonempty_world_with_one_schema_repair_each']*config['worlds'])
                            answer[branch]=_run(selected,comp/branch,backend,config,scm,numeric)
                        else:answer[branch]=organized.world(comp,branch,backend,None)
                    except Exception as exc:answer[branch]=dict(status='failed',error=type(exc).__name__+': '+str(exc))
                failed=any(v.get('failures') or v.get('status') in ('failed','no_world_exported','blocked_dependency') for v in answer.values())
                row.update(status='partial' if failed else 'complete',result=answer)
            else:raise ValueError('Unknown phase')
            if phase!='worlds':row.update(status='partial' if answer.get('status') in ('failed','partial') else 'complete',result=answer)
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        write(marker,row);results.append(row);print(json.dumps(dict(scene=r['id'],phase=phase,backend=backend,status=row['status']),ensure_ascii=False),flush=True)
    bad=any(r['status'] not in ('complete','partial') or (phase=='worlds' and r['status']=='partial') for r in results)
    return dict(status='partial_or_failed' if bad else 'complete_with_warnings' if any(r['status']=='partial' for r in results) else 'complete',scenes=results)


def coordinate_plan(root):
    """Fail before model initialization if a Qwen-only recovery is unsafe."""
    root=root.resolve();suite=verify_suite(root);pending=[]
    if suite.get('mode')!='single' or suite['worlds']!=1:raise ValueError('Coordinate-only entry requires single-world mode')
    for r in suite['scenes']:
        scene=root/'scenes'/r['id'];comp=scene/'completion'
        api=comp/'end_to_end/api/completion'
        if not (api/'world_01/G_full.json').is_file() or read(api/'summary.json').get('failures'):raise ValueError('API world incomplete; use full repair-generate rather than coordinate-only entry: '+r['id'])
        if missing_inputs(scene,'worlds','qwen'):raise ValueError('Frozen Qwen mechanisms unavailable: '+r['id'])
        c=comp/'end_to_end/qwen/completion';s=read(c/'summary.json');d=c/'world_01'
        if (d/'G_full.json').is_file():continue
        if not s.get('failures') or not all(str(f['stage']).startswith('layout_') for f in s['failures']) or not (d/'dimension_stage.json').is_file() or (d/'execution.json').exists():raise ValueError('Not an eligible pre-compiler coordinate failure: '+r['id'])
        if (c/'summary_before_coordinate_transport_v3.json').exists():raise ValueError('Bounded v3 recovery already attempted: '+r['id'])
        pending.append(r['id'])
    return dict(status='ready',pending_Qwen_coordinate_worlds=pending,api_calls_required=0,SCM_resampling=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','fork','coordinate-plan')+PHASES+('report',))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS);p.add_argument('--backend',choices=('qwen','api'),default='qwen')
    p.add_argument('--source-root',type=Path)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--worlds',type=int);p.add_argument('--draws',type=int,default=4);p.add_argument('--mode',choices=('single','evaluation'),default='evaluation');p.add_argument('--out',type=Path);p.add_argument('--all-branches',action='store_true')
    a=p.parse_args()
    try:
        if a.action=='prepare':r=initialize_suite(a.root,a.inputs,a.seed,a.worlds,a.draws,mode=a.mode);print(json.dumps(dict(root=str(a.root),mode=r['mode'],worlds_per_input_backend=r['worlds'],experiments=r['experiments'],scenes=[s['id'] for s in r['scenes']]),ensure_ascii=False))
        elif a.action=='fork':
            if a.source_root is None:p.error('--source-root required for fork')
            from .assembly_multiscene_fork import fork_suite
            print(json.dumps(fork_suite(a.root,a.source_root,a.inputs),ensure_ascii=False))
        elif a.action=='coordinate-plan':print(json.dumps(coordinate_plan(a.root),ensure_ascii=False))
        elif a.action=='report':
            from .assembly_multiscene_report import report_suite
            print(report_suite(a.root,a.out,all_branches=a.all_branches))
        else:
            r=run_phase(a.root,a.action,a.backend);print(json.dumps(dict(status=r['status']),ensure_ascii=False));sys.exit(0 if r['status'] in ('complete','complete_with_warnings') else 1)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
