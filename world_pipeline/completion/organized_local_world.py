"""Local Qwen adapter for the original v13 world completion, using frozen G_obs.

No observer inference, API provider, replacement SCM, or success-conditioned
resampling. Each phase uses the existing v13 implementation unchanged apart
from selecting a single proposal backend.
"""
import argparse
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys

from PIL import Image

from ..common import ROOT,read,write,sha,validate_new_output,validate_graph
from . import assembly_multiscene as pipeline
from .local_observation import visible_state,bind_observations
from .assembly_presentation_visual import render_world

KIND='organized_v13_local_frozen_observation_v1'
PHASES=('prepare-scenes','propose','pool','mechanisms','worlds')


def entry_hashes():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'qwen_edit_pano/scripts/organized_local_world.sh')}


def observed_source(source,scene):
    obs=source/'scenes'/scene['id']/'observation'
    config=read(obs/'experiment.json')
    if config['hfov']!=scene['hfov'] or sha(obs/'input.png')!=config['image_sha256']:
        raise ValueError('Observation image/HFOV binding changed: '+scene['id'])
    for stage in ('observer2d','observer3d'):
        if read(obs/'qwen'/stage/'summary.json').get('status')!='completed':
            raise ValueError('Require completed '+stage+': '+scene['id'])
    two=visible_state(read(obs/'qwen/observer2d/G_2d.json'))
    three=visible_state(read(obs/'qwen/observer3d/G_obs.json'),metric=True)
    bind_observations(two,three)
    wild=read(obs/'qwen/observer3d/config.json')
    if wild['source_sha256']!=sha(obs/'qwen/observer2d/G_2d.json') or wild['image_sha256']!=sha(obs/'input.png') or wild['hfov']!=scene['hfov']:
        raise ValueError('WildDet3D was not bound to this RGB/G_2d/HFOV')
    return obs,three


def prepare(root,source,worlds=4,draws=4,seed=3407):
    root=validate_new_output(root);source=Path(source).resolve()
    original=read(source/'suite.json');records=[];bindings=[]
    # Validate the entire input batch before creating any completion experiment.
    for s in original['scenes']:
        obs,state=observed_source(source,s)
        records.append(dict(id=s['id'],local_rgb=str(obs/'input.png'),local_sha256=sha(obs/'input.png'),hfov_degrees=s['hfov'],split=s.get('split','unspecified')))
        files={str(p.relative_to(obs)):sha(p) for stage in ('observer2d','observer3d') for p in (obs/'qwen'/stage).rglob('*') if p.is_file()}
        files['input.png']=sha(obs/'input.png')
        bindings.append(dict(scene=s['id'],source=str(obs),observed=len(state['objects']),files=files))
    # The manifest is an input to initialize_suite, not a proposal/hidden catalog.
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w',suffix='.jsonl') as f:
        f.write(''.join(json.dumps(r)+'\n' for r in records));f.flush()
        suite=pipeline.initialize_suite(root,Path(f.name),seed,worlds,draws,original['local_model'],mode='evaluation')
    (root/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    suite.update(backends=['qwen'],experiments=['end_to_end'],observation_source=str(source),
        comparison_scope='Local Qwen v13 completion from immutable successful observations; no backend comparison')
    write(root/'suite.json',suite)
    for binding in bindings:
        src=Path(binding['source']);dst=root/'scenes'/binding['scene']/'observation'
        shutil.copyfile(src/'input.png',dst/'input.png')
        for stage in ('observer2d','observer3d'):shutil.copytree(src/'qwen'/stage,dst/'qwen'/stage)
        config=read(dst/'experiment.json');config['image_sha256']=sha(dst/'input.png');write(dst/'experiment.json',config)
    meta=dict(kind=KIND,scheduled_phases=list(PHASES),backends=['qwen'],API_used=False,
        hidden_state_generation=True,completion_model='organized_functional_program_scm_v13',
        observations_reestimated=False,old_candidates_reused=False,SCM_resampled_until_success=False,
        source_suite=str(source/'suite.json'),source_suite_sha256=sha(source/'suite.json'),
        observation_bindings=bindings,entry_sha256=entry_hashes())
    write(root/'local_world_run.json',meta)
    check_run(root)
    return dict(status='prepared_no_model_calls',scenes=len(bindings),observed=sum(b['observed'] for b in bindings),
        worlds_per_scene=worlds,scheduled_phases=list(PHASES))


def check_run(root):
    root=Path(root).resolve();meta=read(root/'local_world_run.json')
    if meta.get('kind')!=KIND or meta.get('scheduled_phases')!=list(PHASES) or meta.get('backends')!=['qwen']:
        raise ValueError('Not a local v13 full-world run')
    for p,h in {**meta['entry_sha256'],meta['source_suite']:meta['source_suite_sha256']}.items():
        if sha(p)!=h:raise ValueError('Bound source/entry changed: '+p)
    suite=pipeline.verify_suite(root)
    if suite.get('backends')!=['qwen'] or suite.get('experiments')!=['end_to_end']:
        raise ValueError('Only local Qwen end_to_end is scheduled')
    for binding in meta['observation_bindings']:
        for base in (Path(binding['source']),root/'scenes'/binding['scene']/'observation'):
            for name,digest in binding['files'].items():
                if sha(base/name)!=digest:raise ValueError('Frozen observation changed: '+str(base/name))
    return suite


def phase(root,name):
    if name not in PHASES:raise ValueError('Not a completion phase: '+name)
    check_run(root)
    result=pipeline.run_phase(Path(root),name,'qwen')
    check_run(root)
    return result


def report(root):
    root=Path(root).resolve();suite=check_run(root);out=root/'report';out.mkdir(exist_ok=True)
    records=[];cards=[]
    for s in suite['scenes']:
        sid=s['id'];scene=root/'scenes'/sid;comp=scene/'completion/end_to_end/qwen/completion'
        observed=read(scene/'observation/qwen/observer3d/G_obs.json')
        summary=read(comp/'summary.json') if (comp/'summary.json').is_file() else {}
        phases=[read(root/'phase_records'/f'{sid}_{p}_qwen.json') for p in PHASES if (root/'phase_records'/f'{sid}_{p}_qwen.json').is_file()]
        for index in range(1,suite['worlds']+1):
            folder=comp/f'world_{index:02d}';full=folder/'G_full.json';candidate=folder/'G_full_candidates.json'
            sem=read(folder/'semantic_world.json') if (folder/'semantic_world.json').is_file() else None
            status=('blocked_dependency' if any(p['status']=='blocked_dependency' for p in phases) else
                'phase_failed' if any(p['status']=='failed' for p in phases) else 'not_run') if not summary else 'failed'
            state=None;error=None
            row=dict(scene=sid,world=index,observed=len(observed['objects']),proposed=len(sem['objects']) if sem else None,
                placed_new=None,exported=False,observed_preserved=None,phase_statuses={p['phase']:p['status'] for p in phases})
            try:
                if full.is_file() or candidate.is_file():
                    state=validate_graph(read(full if full.is_file() else candidate))
                    if state['objects'][:len(observed['objects'])]!=observed['objects']:
                        raise ValueError('Completion changed the frozen visible instances')
                    generated=state['objects'][len(observed['objects']):]
                    if sem is None or {o['track_id'] for o in generated}!={o['id'] for o in sem['objects']}:
                        raise ValueError('Completion inventory differs from semantic-world inventory')
                    row.update(placed_new=len(generated),observed_preserved=True)
                    if full.is_file():
                        review=read(folder/'numeric_review.json')
                        if not review['geometry_contract_passed'] or state.get('world_state_ready') is not True:
                            raise ValueError('G_full did not pass its declared geometry/readiness contract')
                        row['exported']=True
                        status='exported_with_new_objects' if generated else 'observed_only_sample'
                    else:status='rejected_layout_candidate'
                elif sem is not None:status='layout_failed'
                elif summary.get('status')=='empty_candidate_domain_no_completion':status='empty_candidate_domain'
            except (ValueError,KeyError) as exc:
                error=str(exc);status='invalid_world';state=None;row['exported']=False
            row.update(status=status,error=error,failures=summary.get('failures',[]),completion_error=summary.get('error'),
                failed_phases=[p for p in phases if p['status']!='complete'])
            dest=out/sid/f'world_{index:02d}';dest.mkdir(parents=True,exist_ok=True)
            write(dest/'status.json',row)
            title=escape(sid)+f' / world_{index:02d}: '+status
            card='<article><h2>'+title+'</h2><p>可见：'+str(row['observed'])+'；计划新增：'+str(row['proposed'])+'；已放置新增：'+str(row['placed_new'])+'</p>'
            card+='<p><a href="'+str(dest.relative_to(out))+'/status.json">阶段状态与失败记录</a>'
            for name,path in [('完整世界 G_full',full),('布局候选（未通过）',candidate),('假设与对象库存',folder/'semantic_world.json'),('依赖关系',folder/'object_relations.json'),('因果模型',comp/'causal_model.json')]:
                if path.is_file():card+=' · <a href="'+escape(os.path.relpath(path,out))+'">'+name+'</a>'
            card+='</p>'
            if state is not None:
                with Image.open(scene/'observation/input.png') as im:
                    checks=render_world(state,im,s['hfov'],dest/'views')
                if checks['labels']!=len(state['objects']):raise ValueError('Visualization omitted objects')
                view=str(dest.relative_to(out))+'/views'
                card+='<p>'+('合法数值候选，语义与真实遮挡仍需核验。' if row['exported'] else '布局未通过，仅供诊断，不能当作完成世界。')+'</p>'
                card+='<a href="'+view+'/index.html"><img loading="lazy" src="'+view+'/overview.png">查看原图、原视角、平面图、六方向、ERP与交互3D</a>'
            elif error:card+='<p>'+escape(error)+'</p>'
            cards.append(card+'</article>');records.append(row)
    completed=sum(r['status']=='exported_with_new_objects' for r in records)
    bad=any(r['status']!='exported_with_new_objects' or any(p['status'] not in ('complete','partial') for p in r['failed_phases']) for r in records)
    result=dict(status='partial_or_failed' if bad else 'complete_needs_review',worlds_with_new_objects=completed,
        worlds_scheduled=len(records),total_exported_new_objects=sum(r['placed_new'] for r in records if r['exported']),
        API_used=False,observer_rerun=False,records=records,
        caveat='Hypothetical v13 SCM; causal identification, calibrated probabilities and RGB occlusion consistency are not established.')
    write(out/'results.json',result)
    title='本地 Qwen：固定观测 → v13 完整世界补全'
    page='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px/1.65 system-ui;margin:24px}article{border:1px solid #ccd3df;margin:18px 0;padding:14px}img{max-width:100%}h2{font-size:17px;overflow-wrap:anywhere}</style><h1>'+title+'</h1>'
    page+='<p>含新增且通过数值检查：'+str(completed)+' / '+str(len(records))+'。蓝色为冻结观测；橙色为映射到词汇表的新增；红色为未映射新增。灰色 ERP 区域未生成 RGB。</p>'
    page+='<p>各世界独立展示；空增量不会重抽到非空，失败不会靠删物体伪装成功。框级数值通过不等于语义或像素级观测相容性已经验证。</p><p><a href="results.json">完整状态记录</a></p>'+''.join(cards)
    (out/'index.html').write_text(page)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=report/index.html"><a href="report/index.html">打开完整世界及新增对象可视化</a>')
    check_run(root)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','check','report')+PHASES)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--observations-from',type=Path)
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--draws',type=int,default=4)
    parser.add_argument('--seed',type=int,default=3407)
    args=parser.parse_args()
    try:
        if args.action=='prepare':
            if args.observations_from is None:parser.error('prepare requires --observations-from')
            result=prepare(args.root,args.observations_from,args.worlds,args.draws,args.seed)
        elif args.action=='check':
            check_run(args.root);result=dict(status='verified_no_model_calls')
        elif args.action=='report':result=report(args.root)
        else:result=phase(args.root,args.action)
        shown={k:result[k] for k in ('status','worlds_with_new_objects','worlds_scheduled','total_exported_new_objects')} if args.action=='report' else result
        print(json.dumps(shown,ensure_ascii=False),flush=True)
        return 1 if result['status']=='partial_or_failed' else 0
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':sys.exit(main())
