"""Original v13 execution, retained local catalog and bounded proposal patches."""
import argparse
from collections import Counter
from copy import deepcopy
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback

from ...common import ROOT,read,sha
from ..candidate_repair import workflow as cached,contract as typed
from ..candidate_repair.requests import Requests,save
from .. import assembly_compositional as original
from ..assembly_experiment import check as context_check,stage_seed
from ..fresh_pipeline import provider
from . import contract
from .requests import PatchRequests

BASELINE=ROOT/'qwen_edit_pano/outputs/v13_candidates_fixed_001'
KIND='original_v13_retained_local_catalog_field_patch_v1'


def sources():
    return {str(p.resolve()):sha(p) for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'qwen_edit_pano/scripts/stable_world.sh']}


def prepare(root,source,worlds=4,draws=4,baseline=BASELINE):
    source=Path(source).resolve();baseline=Path(baseline).resolve() if baseline else None
    if baseline:cached.check(baseline)
    bindings={};retained_by_scene={}
    scenes={s['id']:s for s in read(baseline/'suite.json')['scenes']} if baseline else {}
    for s in read(source/'suite.json')['scenes']:
        observed,_=cached.base.observed_source(source,s)
        retained_by_scene[s['id']]=False
        if s['id'] not in scenes:continue
        old,_=cached.base.observed_source(baseline,scenes[s['id']])
        if s['hfov']!=scenes[s['id']]['hfov'] or any(sha(old/name)!=sha(observed/name)
            for name in ('input.png','qwen/observer3d/G_obs.json')):continue
        retained_by_scene[s['id']]=True
        for name in ('input.png','qwen/observer3d/G_obs.json'):
            bindings[str(old/name)]=sha(old/name)
        proposals=baseline/'scenes'/s['id']/'completion/proposals/qwen'
        summary=read(proposals/'summary.json')
        if summary.get('API_used') is not False:raise ValueError('Only explicitly local Qwen proposals may be retained')
        for name in ('hypotheses.json','programs.json','summary.json'):
            bindings[str(proposals/name)]=sha(proposals/name)
        for p in read(proposals/'programs.json').values():
            if p.get('proposal_source',{}).get('backend')!='qwen':raise ValueError('Non-Qwen proposal in local catalog')
    result=cached.prepare(root,source,rounds=1,worlds=worlds,draws=draws,baseline=baseline)
    save(root/'stable_world.json',dict(kind=KIND,source_sha256=sources(),baseline=str(baseline) if baseline else None,
        retained_by_scene=retained_by_scene,
        baseline_sha256=bindings,original_v13_model_and_geometry=True,API_used=False,
        retained='accepted local semantic proposals only; no cached CPT, noise, dimensions or layouts',
        new_proposal_rounds='one with matching catalog; two otherwise',max_initial_new_proposal_calls_per_scene=10,
        repair='one bounded field patch; no identity/category/group/alternative deletion'))
    reused=any(retained_by_scene.values())
    meta=read(root/'local_world_run.json');meta['old_candidates_reused']=reused;save(root/'local_world_run.json',meta)
    suite=read(root/'suite.json');suite['old_candidate_catalog_reused']=reused;save(root/'suite.json',suite)
    meta=read(root/'candidate_repair.json');meta['baseline_candidates_used_in_model']=reused;save(root/'candidate_repair.json',meta)
    return dict(result,model=KIND,retained_catalog_scenes=sum(retained_by_scene.values()))


def check(root):
    suite,_=cached.check(root);meta=read(root/'stable_world.json')
    if meta['kind']!=KIND or meta['source_sha256']!=sources():raise ValueError('Bound stable implementation changed; use a new run')
    for p,h in meta['baseline_sha256'].items():
        if sha(p)!=h:raise ValueError('Bound retained catalog changed: '+p)
    return suite,meta


def propose_scene(comp,baseline,client):
    case=read(comp/'shared/case.json');run=read(comp/'run.json');dest=comp/'proposals/qwen'
    seed_dir=baseline/'scenes'/comp.parent.name/'completion/proposals/qwen' if baseline else None
    hs=read(seed_dir/'hypotheses.json') if seed_dir else []
    programs=read(seed_dir/'programs.json') if seed_dir else {}
    retained=deepcopy(programs)
    for p in programs.values():original.revalidate_program(p,case)
    cov=typed.coverage(programs);cov.pop('complete_program_summaries')
    requests=PatchRequests(dest/'calls',stage_seed(run['seed'],comp.parent.name+':stable'),comp/'shared/observation_board.png',client)
    rejections=[];failures=[];new_h=[];added=0
    def failed(name,exc):
        receipt=requests.folder/(name+'.receipt.json');r=read(receipt) if receipt.exists() else {};err=r.get('error','')
        schema=r.get('status')=='failed' and err.startswith('ValueError: ') and str(exc) in (err,err[12:])
        row=dict(stage=name,error=err if schema else type(exc).__name__+': '+str(exc))
        (rejections if schema else failures).append(row)
    for round_index in range(1 if retained else 2):
        # Hypotheses contain no object inventory, so their own schema may be
        # redrawn once. Object-bearing requests use the field patch protocol.
        hname='hypotheses_round_'+str(round_index)
        try:
            hreq=Requests(requests.folder,requests.seed,requests.board,client)
            proposed=hreq.ask(hname,typed.hypotheses_prompt(case,
                read(comp/'context/qwen/observer2d/grounding.json'),cov),contract.hypotheses_answer,2048)
            for h in proposed:
                h['id']='fresh'+str(round_index)+'_'+h['id'];new_h.append(h)
                for channel in ('surrounding','belongings'):
                    name='furniture_'+h['id']+'_'+channel
                    try:
                        values=requests.ask(name,contract.furniture_prompt(case,h,channel,cov),
                            lambda v,h=h:contract.furniture_answer(v,case,h),8192)
                    except Exception as exc:failed(name,exc);continue
                    for p in values:
                        fid='fresh_f'+str(added);added+=1
                        p['proposal_source']=dict(backend='qwen',channel=channel,new_proposal=True,field_patch_protocol=True)
                        programs[fid]=p
        except Exception as exc:failed(hname,exc)
        cov=typed.coverage(programs);cov.pop('complete_program_summaries')
    # Retain exactly, including original provenance. This is an explicit
    # supplied catalog, never presented as new visual evidence or inference.
    if any(programs[fid]!=p for fid,p in retained.items()):raise ValueError('Retained proposal changed')
    hs+=new_h
    save(dest/'hypotheses.json',hs);save(dest/'programs.json',programs)
    summary=dict(status='partial' if failures else 'complete_with_warnings' if rejections else 'complete',backend='qwen',
        programs=len(programs),retained_programs=len(retained),fresh_programs=added,
        fresh_nonempty_programs=sum(any(g['objects'] for g in p['groups']) for fid,p in programs.items() if fid not in retained),
        candidate_instances=sum(len(g['objects']) for p in programs.values() for g in p['groups']),
        failed_queries=rejections+failures,rejections=rejections,execution_failures=failures,
        retained_local_proposals=bool(retained),API_used=False,observer_rerun=False,original_v13_executor=True,
        candidate_reachability_not_a_semantic_quality_score=True)
    save(dest/'summary.json',summary);save(dest/'coverage.json',typed.coverage(programs));return summary


def phase(root,name):
    suite,meta=check(root)
    if name!='propose':return cached.base.phase(root,name)
    local=None;rows=[]
    def client(comp):
        nonlocal local
        if local is None:local=provider(context_check(comp/'context'),'qwen')
        return local
    for s in suite['scenes']:
        path=root/'phase_records'/(s['id']+'_propose_qwen.json');comp=root/'scenes'/s['id']/'completion'
        if path.exists():rows.append(read(path));continue
        row=dict(scene=s['id'],phase='propose',backend='qwen',status='failed')
        try:
            baseline=Path(meta['baseline']) if meta['retained_by_scene'][s['id']] else None
            result=propose_scene(comp,baseline,lambda:client(comp))
            files=list((comp/'proposals/qwen').rglob('*'))
            save(root/'candidate_bindings'/(s['id']+'.json'),{str(p):sha(p) for p in files if p.is_file()})
            row.update(status=result['status'],result=result)
            if not any(g['objects'] for p in read(comp/'proposals/qwen/programs.json').values() for g in p['groups']):
                for next_phase in ('pool','mechanisms','worlds'):
                    save(root/'phase_records'/(s['id']+'_'+next_phase+'_qwen.json'),dict(scene=s['id'],
                        phase=next_phase,backend='qwen',status='blocked_dependency',reason='no_nonempty_candidate_domain',model_called=False))
                save(comp/'end_to_end/qwen/completion/summary.json',dict(status='empty_candidate_domain_no_completion',worlds=[],failures=[]))
        except Exception as exc:row.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        save(path,row);rows.append(row);print(json.dumps(dict(scene=s['id'],phase=name,status=row['status'])),flush=True)
    check(root);return dict(status='recorded',records=rows)


def preview(root):
    check(root);result=cached.candidate_report(root);rows=[]
    for s in check(root)[0]['scenes']:
        p=root/'scenes'/s['id']/'completion/proposals/qwen/summary.json'
        if p.exists():rows.append(dict(scene=s['id'],**read(p)))
    save(root/'proposal_audit.json',dict(records=rows,API_used=False,
        note='Retained local proposals and fresh proposals separately counted. More candidates do not prove better worlds.'))
    previous_status=result['status'];errors=[];rejections=[]
    for row in rows:
        errors.extend(dict(scene=row['scene'],**f) for f in row['execution_failures'])
        rejections.extend(dict(scene=row['scene'],**f) for f in row['rejections'])
    for path in (root/'phase_records').glob('*.json'):
        row=read(path)
        if row['phase'] in ('prepare-scenes','propose','pool') and row['status']=='failed':errors.append(row)
    result.update(execution_errors=errors,proposal_rejections=rejections,
        status='candidate_execution_partial' if errors else 'candidate_domain_incomplete' if result['scenes_without_domain'] else
            'candidate_domains_ready_with_warnings' if rejections else 'candidate_domains_ready_needs_review')
    save(root/'candidate_report.json',result)
    page=root/'candidates.html'
    page.write_text(page.read_text().replace(previous_status,result['status']).replace('<h1>',
        '<p><a href="proposal_audit.json">保留的本地候选、新提议及字段修正审计</a></p><h1>',1))
    return result


def report(root):
    suite,meta=check(root);preview(root);result=cached.reusable_report(root);errors=[];warnings=[]
    for path in (root/'phase_records').glob('*.json'):
        row=read(path);r=row.get('result',{})
        warnings.extend(dict(scene=row['scene'],**f) for f in r.get('rejections',[]))
        errors.extend(dict(scene=row['scene'],**f) for f in r.get('execution_failures',[]))
        if row['status']=='failed':errors.append(row)
        if row['phase']=='worlds':
            for outcome in r.values():
                if not isinstance(outcome,dict):continue
                if outcome.get('error'):errors.append(dict(scene=row['scene'],**outcome))
                for f in outcome.get('failures',[]):
                    folder=root/'scenes'/row['scene']/'completion/end_to_end/qwen/completion'/('world_%02d'%f['world'])
                    numeric=folder/'numeric_review.json'
                    rejected=(numeric.exists() and (folder/'G_full_candidates.json').exists() and
                        read(numeric).get('geometry_contract_passed') is False and f.get('stage')=='compile_'+str(f['world']))
                    (warnings if rejected else errors).append(dict(scene=row['scene'],**f))
    errors += [r for r in result['records'] if r['status']=='invalid_world']
    missing=[r for r in result['records'] if not r['exported'] or not r['placed_new']]
    result.update(model=KIND,counts=dict(Counter(r['status'] for r in result['records'])),execution_errors=errors,
        warnings=warnings,uncompleted_worlds=len(missing),retained_local_catalog=meta['baseline'],
        new_Qwen_quality_verified=False,semantic_quality_verified=False,
        status='execution_partial' if errors else 'incomplete_worlds' if missing else 'completed_needs_review')
    save(root/'results.json',result);save(root/'report/results.json',result)
    baseline_link='<a href="'+escape(os.path.relpath(Path(meta['baseline'])/'index.html',root))+'">原本地基线结果</a> · ' if meta['baseline'] else ''
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px/1.7 system-ui;margin:28px}pre{white-space:pre-wrap}</style><h1>原v13、本地候选保留与字段修正</h1><p>'+escape(result['status'])+' '+escape(str(result['counts']))+'</p><p>'+baseline_link+'<a href="report/index.html">原图、投影、平面图、六方向、ERP及交互三维</a> · <a href="candidates.html">候选及宿主</a> · <a href="proposal_audit.json">保留与新提议审计</a> · <a href="results.json">完整状态</a></p><p>仅图像、HFOV和观测完全匹配时保留纯本地Qwen候选，再新增一轮提议；新图无匹配目录时执行两轮新提议。概率与布局重新计算。数值导出不证明真实语义完整。</p>')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','check','candidate-report','report')+cached.base.PHASES)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--observations-from',type=Path)
    p.add_argument('--baseline',type=Path,default=BASELINE);p.add_argument('--worlds',type=int,default=4);p.add_argument('--draws',type=int,default=4)
    a=p.parse_args();root=a.root.resolve()
    try:
        if a.action=='prepare':
            if a.observations_from is None:p.error('--observations-from required')
            result=prepare(root,a.observations_from,a.worlds,a.draws,None if str(a.baseline)=='none' else a.baseline)
        elif a.action=='check':check(root);result=dict(status='verified_no_model_calls')
        elif a.action=='report':result=report(root)
        elif a.action=='candidate-report':result=preview(root)
        else:result=phase(root,a.action)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','execution_errors','warnings')},ensure_ascii=False),flush=True)
        if a.action=='report':return 2 if result['execution_errors'] else 1 if result['uncompleted_worlds'] else 0
        if a.action=='candidate-report':return 2 if result['execution_errors'] else 1 if result['scenes_without_domain'] else 0
        return 0
    except Exception as exc:traceback.print_exc();p.exit(2,type(exc).__name__+': '+str(exc)+'\n')


if __name__=='__main__':sys.exit(main())
