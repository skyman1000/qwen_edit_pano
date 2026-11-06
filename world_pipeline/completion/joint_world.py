"""Local Qwen, frozen observations, whole-hypothesis host-conditioned completion.

Run proposal -> empirical finite mechanisms -> IID worlds -> continuous layout
-> approximate plausibility checks -> visualization. No remote provider or observer calls.
"""
import argparse
from copy import deepcopy
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback

from PIL import Image

from ..common import ROOT,read,write,sha
from . import organized_local_world as adapter, joint_world_model as semantic
from . import joint_world_geometry as geometry
from .fresh_observer import load_case
from .fresh_pipeline import provider
from .assembly_observation import host_domains
from .assembly_experiment import stage_seed
from .joint_world_requests import asker
from .scene_program_world import observation_board
from .assembly_organized_geometry import dimensions_prompt,dimensions_answer
from .scene_completion import export_states
from .off_fov_completion import validate_world
from .assembly_delta import intervention_diagnostics
from .assembly_presentation_visual import render_world
from .assembly_taxonomy import annotate

PHASES=('propose','mechanisms','worlds')
COMPLETED={'complete','complete_with_warnings'}


class LazyLocal:
    """Never load a checkpoint for blocked domains or observed-only samples."""
    def __init__(self,config):self.config=config;self.client=None
    def ask(self,*args,**kwargs):
        if self.client is None:self.client=provider(self.config,'qwen')
        return self.client.ask(*args,**kwargs)


def entry_hashes():
    paths=sorted(Path(__file__).parent.glob('joint_world*.py'))+[ROOT/'qwen_edit_pano/scripts/joint_world.sh']
    return {str(p.resolve()):sha(p) for p in paths}


def prepare(root,source,worlds=4,draws=4,seed=3407):
    result=adapter.prepare(root,source,worlds,draws,seed)
    write(Path(root)/'joint_world_run.json',dict(kind=semantic.VARIANT,entry_sha256=entry_hashes(),
        scheduled_phases=list(PHASES),import_adapter_only=True,API_used=False,observer_rerun=False,
        role_pooling=False,genuine_empty_draws_retained=True,proposal_failure_mass_audited=True,
        semantic_protocol='v13_complete_delta_new_objects',metric_fields_in_H=False,
        roots_only_generation=False,all_failed_domain_blocks_downstream=True,
        automatic_empty_injection=False,program_law='explicit_finite_reference_conditioned_on_available_semantic_proposals',
        layout_acceptance='approximate_plausibility_v1'))
    return dict(result,scheduled_phases=list(PHASES),model=semantic.VARIANT)


def check(root):
    suite=adapter.check_run(root);meta=read(Path(root)/'joint_world_run.json')
    if meta['kind']!=semantic.VARIANT:raise ValueError('Wrong joint-world entry')
    if meta['entry_sha256']!=entry_hashes():raise ValueError('Joint-world source changed; initialize a fresh output')
    return suite


def load(root,scene):
    obsroot=root/'scenes'/scene['id']/'observation';case=load_case(obsroot,'qwen');obs=read(case['state'])
    boxes={o['track_id']:o for o in obs['objects']}
    for row in case['context']:
        box=boxes[row['id']];row.update(center=box['center_local_m']['value'],size=box['size_aabb_local_m']['value'])
    case['observed_support_domains']=host_domains(case,obs)
    out=root/'scenes'/scene['id']/'joint_completion';out.mkdir(exist_ok=True)
    return case,obs,out,read(obsroot/'experiment.json')


def object_world(program):
    return dict(objects=[dict(deepcopy(o),id=o['key']) for g in program['groups'] for o in g['objects']])


def sizes(objects,ask,prefix):
    result={}
    for i in range(0,len(objects),8):
        w=dict(objects=objects[i:i+8])
        result.update(ask(prefix+'_'+str(i//8),None,dimensions_prompt(w),lambda v:dimensions_answer(v,w)))
    return result


def propose(case,observed,out,config,ask):
    _,board=observation_board(case,observed,out/'observation_board.png')
    hypotheses=ask('hypotheses',board,semantic.hypotheses_prompt(case),lambda v:semantic.hypotheses_answer(v,case,require_concrete=True))
    write(out/'hypotheses.json',hypotheses);programs={};failures=[]
    for h in hypotheses:
        fid='f_'+h['id']
        try:
            result=ask('programs_'+h['id'],board,semantic.programs_prompt(case,h),lambda v:semantic.programs_answer(v,case,h,require_concrete=True))
            for j,p in enumerate(result['valid']):programs[fid+'_'+str(j)]=p
            for failure in result['failed']:
                failures.append(dict(hypothesis=h['id'],**failure))
                programs[fid+'_failed_'+str(failure['alternative'])]=semantic.failed_program(h,failure['error'])
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc);failures.append(dict(hypothesis=h['id'],error=error))
            programs[fid]=semantic.failed_program(h,error)
            (out/(fid+'_error.txt')).write_text(traceback.format_exc())
    value=dict(hypotheses=hypotheses,programs=programs,proposal_failures=failures,
        metric_inference_deferred_until_inventory_frozen=True,protocol='v13_new_objects_complete_delta_no_root_only_gate',
        support_scope='finite_joint_program_per_global_hypothesis_with_explicit_absence_and_failure')
    write(out/'domain.json',value)
    status='complete_with_warnings' if failures and semantic.has_valid_program(value) else 'partial' if failures else 'complete'
    return dict(status=status,proposal_failures=failures)


def choice_answer(value,states):
    if value.get('choice') not in states:raise ValueError('Choose one exact ID from '+json.dumps(states))
    semantic.text(value.get('reason'));return value['choice']


def empirical(ask,name,board,case,choices,draws,question):
    states=list(choices);votes=[];failures=[]
    for i in range(draws):
        # Rotate list order; call seeds differ. Frequencies remain uncalibrated.
        ordered=states[i%len(states):]+states[:i%len(states)]
        prompt=(question+' Return {"choice":"exact ID","reason":"brief evidence and assumptions"}. '
            'Judge plausibility given the fixed RGB, not how many objects are generated. '
            'These are finite alternatives, not a known truth.\nALTERNATIVES: '+
            json.dumps({s:choices[s] for s in ordered},ensure_ascii=False)+'\nOBSERVED: '+json.dumps(case['context']))
        try:votes.append(ask(name+'_'+str(i),board,prompt,lambda v:choice_answer(v,states),512))
        except Exception as exc:failures.append(dict(draw=i,error=type(exc).__name__+': '+str(exc)))
    row=semantic.predictive(votes,states);row.update(invalid_draws=failures,requested_draws=draws)
    return row


def mechanisms(case,observed,out,config,ask,draws):
    d=read(out/'domain.json')
    if not semantic.has_valid_program(d):return dict(status='blocked_proposal_domain',model_called=False,error='All semantic proposals failed; no SCM estimation or world sampling is possible.')
    _,board=observation_board(case,observed,out/'observation_board.png')
    recognition=empirical(ask,'recognition',board,case,{h['id']:h for h in d['hypotheses']},draws,
        'Which WHOLE surrounding-world hypothesis is compatible with the photographed sector and indoor use? '
        'Unobserved regions are unknown, not observed empty; judge the stated assumptions without requiring unseen items to appear in RGB.')
    tables=dict(recognition=recognition,programs={})
    for h in d['hypotheses']:
        choices={fid:p for fid,p in d['programs'].items() if p['hypothesis']['id']==h['id']}
        tables['programs'][h['id']]=semantic.reference_program_law(choices)
    raw=semantic.compile_model(d['hypotheses'],d['programs'],tables)
    model,coverage=semantic.available_proposal_model(raw)
    write(out/'causal_model_unconditioned.json',raw);write(out/'proposal_coverage.json',coverage)
    write(out/'empirical_tables.json',tables);write(out/'causal_model.json',model)
    return dict(status='complete_with_warnings' if coverage['excluded_reference_mass']>1e-9 else 'complete',calibrated_probabilities=False,proposal_coverage=coverage)


def materialize(case,observed,out,world,dimensions,ask):
    h=world['hypothesis']
    _,board=observation_board(case,observed,out/'observation_board.png')
    plane=geometry.floor_hypothesis(case,observed,world,board,ask) if world['objects'] else dict(floor_y_m=None,kind='empty_delta_no_floor_required')
    write(out/'support_plane.json',plane)
    if world['objects']:
        layout,raw=geometry.infer(case,observed,world,dimensions,plane,board,ask)
        write(out/'continuous_placement.json',raw)
        solved=geometry.compile_layout(case,observed,world,layout,plane)
    else:
        solved=dict(plan=dict(scene_type='observed-only hypothesis',scene_hypothesis=world['configuration'],additions=[]),
            items=[],mappings=[],review=dict(geometry_contract_passed=True,joint_contract_errors=[],
            observed_state_preserved=True,physical_geometry_verified=False))
    return finish_materialization(case,observed,out,world,dimensions,plane,layout if world['objects'] else {},solved)


def finish_materialization(case,observed,out,world,dimensions,plane,layout,solved):
    """Common export path for fresh model poses and hash-verified saved poses."""
    h=world['hypothesis']
    write(out/'support_plane.json',plane)
    write(out/'execution.json',solved);write(out/'numeric_review.json',solved['review'])
    full,_,unmapped=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
    passed=solved['review']['geometry_contract_passed'];metadata={o['id']:o for o in world['objects']}
    for obj in full['objects'][len(observed['objects']):]:
        src=metadata[obj['track_id']];obj.update(variant=src['variant'],attachment=src['attachment'],
            raw_category=src['category'],generative_presence_node=src['presence_node'],installation_node=src['installation_node'])
        obj.update(intrinsic_size_m=dimensions[src['id']]['full_size_m'],
            yaw_deg=next(r['yaw_deg'] for g in layout.values() for r in g['objects'] if r['id']==src['id']))
    full.update(world_state_ready=passed,world_state_ready_scope='approximate_world_hypothesis_needs_semantic_review_not_precision_reconstruction',
        completion_model_variant=semantic.VARIANT,global_hypothesis=h,typed_relations=world['relations'],
        declared_support_plane=plane,RGB_likelihood_defined=False,hidden_inventory_not_uniquely_identified=True,
        layout_policy=solved['review'].get('layout_policy'),
        layout_warnings=solved['review'].get('soft_layout_warnings',[]),
        environment_support_assumptions=solved['review'].get('environment_support_assumptions',[]))
    full=annotate(full,world)
    if full['objects'][:len(observed['objects'])]!=observed['objects']:raise ValueError('Frozen observation modified')
    if {o['track_id'] for o in full['objects'][len(observed['objects']):]}!=set(metadata):raise ValueError('Inventory changed during layout/export')
    if passed:validate_world(observed,full,solved['items'],len(observed['objects'])+len(metadata))
    # Both valid and failed box candidates get complete existing-project views.
    with Image.open(case['image']) as im:checks=render_world(full,im,case['hfov'],out/'views')
    if checks['labels']!=len(full['objects']):raise ValueError('Visualization omitted identities')
    write(out/('G_full.json' if passed else 'G_full_candidates.json'),full)
    if passed:
        hidden=deepcopy(full);hidden.update(scope='hidden',objects=deepcopy(full['objects'][len(observed['objects']):]));write(out/'G_out.json',hidden)
    return dict(status=('exported_with_new_objects' if metadata else 'observed_only_sample') if passed else 'rejected_layout_candidate',
        proposed=len(metadata),roots=sum(o['attachment']['host'] is None for o in world['objects']),
        zones=len({o['group'] for o in world['objects']}),unmapped=len(unmapped),exported=passed,
        errors=solved['review']['joint_contract_errors'],warnings=solved['review'].get('soft_layout_warnings',[]),
        strict_geometry_contract_passed=solved['review'].get('strict_geometry_contract_passed',passed))


def worlds(case,observed,out,config,ask,count,seed,cached_only=False):
    d=read(out/'domain.json');model=read(out/'causal_model.json');samplefile=out/'structural_samples.json'
    if not samplefile.exists():write(samplefile,semantic.sample_worlds(model,seed,count))
    samples=read(samplefile)
    if len(samples)!=count:raise ValueError('Frozen sample count changed')
    records=[]
    for i,w in enumerate(samples,1):
        dest=out/f'world_{i:02d}';dest.mkdir(exist_ok=True)
        if (dest/'status.json').exists() and read(dest/'status.json')['status']!='pending_local_inference':records.append(read(dest/'status.json'));continue
        write(dest/'semantic_world.json',w)
        write(dest/'object_relations.json',dict(relations=w['relations'],attachments=[dict(id=o['id'],**o['attachment']) for o in w['objects']]))
        write(dest/'inventory_interventions.json',intervention_diagnostics(model,w))
        try:
            if w['proposal_failed']:raise ValueError('Selected hypothesis proposal failed; unresolved inventory is not empty absence')
            def call(name,image,prompt,validator,max_tokens=8192):return ask(f'world_{i:02d}_'+name,image,prompt,validator,max_tokens)
            from .joint_world_repair import cached_geometry
            cached=cached_geometry(out,w,i)
            if cached is not None:
                dims,plane,raw,origin=cached
                layout=geometry.convert(w,observed,dims,raw['origins'],raw['poses'],case['hfov'])
                solved=geometry.compile_layout(case,observed,w,layout,plane)
                write(dest/'continuous_placement.json',dict(raw,origin_initialization={g:v['origin_initialization'] for g,v in layout.items()},cache_origin=origin))
                write(dest/'dimension_stage.json',dict(dimensions=dims,all_scale_inventory_frozen=True,dimensions_frozen_before_positions=True,cache_origin=origin))
                record=finish_materialization(case,observed,dest,w,dims,plane,layout,solved);record['geometry_source']='verified_saved_model_predictions'
            elif cached_only:
                record=dict(status='pending_local_inference',exported=False,proposed=len(w['objects']),error='Valid semantic world has no matching saved metric predictions; GPU inference not executed in CPU preview.')
            else:
                dims=sizes(w['objects'],call,'sizes')
                write(dest/'dimension_stage.json',dict(dimensions=dims,all_scale_inventory_frozen=True,dimensions_frozen_before_positions=True))
                record=materialize(case,observed,dest,w,dims,call);record['geometry_source']='local_Qwen_predictions'
        except Exception as exc:
            record=dict(status='blocked_proposal' if w['proposal_failed'] else 'layout_failed',exported=False,
                proposed=None if w['proposal_failed'] else len(w['objects']),error=type(exc).__name__+': '+str(exc))
            (dest/'error.txt').write_text(traceback.format_exc())
        record.update(world=i,hypothesis=w['hypothesis']['id'],configuration=w['structural_state']['F'],observed=len(observed['objects']))
        write(dest/'status.json',record);records.append(record)
    return dict(status='partial' if any(not r['exported'] for r in records) else 'complete_with_warnings' if any(r.get('warnings') for r in records) else 'complete',records=records)


def phase(root,name,cached_only=False):
    suite=check(root);p=None;records=[]
    for scene in suite['scenes']:
        marker=root/'phase_records'/(scene['id']+'_joint_'+name+('_preview' if cached_only else '')+'.json')
        if marker.exists():records.append(read(marker));continue
        try:
            case,obs,out,config=load(root,scene)
            required=[] if name=='propose' else ['domain.json']
            if any(not (out/r).is_file() for r in required):
                record=dict(scene=scene['id'],phase=name,status='blocked_dependency',model_called=False,error='Missing prerequisite: '+str(required))
                write(marker,record);records.append(record);continue
            if name in ('mechanisms','worlds') and not semantic.has_valid_program(read(out/'domain.json')):
                record=dict(scene=scene['id'],phase=name,status='blocked_proposal_domain',model_called=False,error='No valid proposal; no downstream model initialization.')
                write(marker,record);records.append(record);continue
            if name=='worlds' and not (out/'causal_model.json').is_file():
                record=dict(scene=scene['id'],phase=name,status='blocked_dependency',model_called=False,error='No estimated causal_model.json; worlds not sampled.')
                write(marker,record);records.append(record);continue
            if p is None:p=LazyLocal(config)
            ask=asker(p,config,out/'calls')
            if name=='propose':result=propose(case,obs,out,config,ask)
            elif name=='mechanisms':result=mechanisms(case,obs,out,config,ask,suite['draws'])
            else:result=worlds(case,obs,out,config,ask,suite['worlds'],stage_seed(suite['seed'],scene['id']),cached_only)
            record=dict(scene=scene['id'],phase=name,**result)
        except Exception as exc:
            record=dict(scene=scene['id'],phase=name,status='failed',error=type(exc).__name__+': '+str(exc))
            marker.with_suffix('.txt').write_text(traceback.format_exc())
        write(marker,record);records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    check(root)
    return dict(status='partial' if any(r['status'] not in COMPLETED for r in records) else 'complete_with_warnings' if any(r['status']=='complete_with_warnings' for r in records) else 'complete',records=records)


def report(root):
    from .joint_world_review import scene_review,candidate_page
    suite=check(root);rows=[];cards=[];comparison=[];all_phases=[]
    for scene in suite['scenes']:
        case,obs,out,_=load(root,scene)
        observed_views=out/'observed_views'
        if not (observed_views/'index.html').exists():
            with Image.open(case['image']) as im:render_world(obs,im,case['hfov'],observed_views)
        observed_rel=os.path.relpath(observed_views,root)
        cards.append('<article><h2>'+escape(scene['id'])+'：固定观测（仅蓝色实例，不是隐藏补全）</h2><a href="'+escape(observed_rel+'/index.html')+'"><img loading="lazy" src="'+escape(observed_rel+'/overview.png')+'">查看原图及冻结观测</a></article>')
        if (out/'domain.json').exists():
            candidate_page(read(out/'domain.json'),out/'domain_review.html')
            cards.append('<p><a href="'+escape(os.path.relpath(out/'domain_review.html',root))+'">'+escape(scene['id'])+'：全部全局假设、候选新增与宿主依赖（未必通过几何）</a></p>')
        comparison.append(dict(scene=scene['id'],**scene_review(root/'scenes'/scene['id'],ROOT/'qwen_edit_pano/outputs/v13_local_world_001')))
        phases=[read(p) for p in (root/'phase_records').glob(scene['id']+'_joint_*.json')]
        all_phases+=phases
        for i in range(1,suite['worlds']+1):
            dest=out/f'world_{i:02d}';status=dest/'status.json'
            missing_status='blocked_proposal_domain' if any(p['status']=='blocked_proposal_domain' for p in phases) else 'blocked_dependency' if any(p['status']=='blocked_dependency' for p in phases) else 'phase_failed_or_not_run'
            row=read(status) if status.exists() else dict(world=i,status=missing_status,exported=False,phase_records=phases)
            row=dict(row,scene=scene['id']);rows.append(row);rel=os.path.relpath(dest,root)
            card='<article><h2>'+escape(scene['id'])+' / '+str(i)+': '+escape(row['status'])+'</h2>'
            card+='<p>假设 '+escape(row.get('hypothesis','?'))+'；新增 '+str(row.get('proposed','?'))+'；根对象 '+str(row.get('roots','?'))+'；区域实例 '+str(row.get('zones','?'))+'</p>'
            for label,artifact in [('联合候选域','domain.json'),('SCM及条件表','causal_model.json')]:
                if (out/artifact).exists():card+='<a href="'+escape(os.path.relpath(out/artifact,root))+'">'+label+'</a> · '
            for name,path in [('对象图与假设','semantic_world.json'),('依赖','object_relations.json'),('数值检查','numeric_review.json'),('状态','status.json'),('G_full','G_full.json'),('失败候选','G_full_candidates.json')]:
                if (dest/path).exists():card+='<a href="'+escape(rel+'/'+path)+'">'+name+'</a> · '
            if (dest/'views/index.html').exists():card+='<p><a href="'+escape(rel+'/views/index.html')+'"><img loading="lazy" src="'+escape(rel+'/views/overview.png')+'">原图、原视角、平面、六方向、ERP、交互3D</a></p>'
            if not status.exists():card+='<pre>'+escape(json.dumps(phases,ensure_ascii=False,indent=2))+'</pre>'
            elif row.get('error'):card+='<pre>'+escape(row['error'])+'</pre>'
            if row.get('warnings'):card+='<details><summary>近似布局假设 / 警告 '+str(len(row['warnings']))+' 项</summary><pre>'+escape(json.dumps(row['warnings'],ensure_ascii=False,indent=2))+'</pre></details>'
            if (out/'proposal_coverage.json').exists():card+='<p><a href="'+escape(os.path.relpath(out/'proposal_coverage.json',root))+'">提案失败与可用候选覆盖率</a></p>'
            cards.append(card+'</article>')
    from collections import Counter
    counts=dict(Counter(r['status'] for r in rows));result=dict(status='partial' if any(not r['exported'] for r in rows) or any(p['status'] not in COMPLETED for p in all_phases) else 'complete_with_warnings' if any(r.get('warnings') for r in rows) or any(p['status']=='complete_with_warnings' for p in all_phases) else 'complete_needs_review',
        counts=counts,completion_summary=dict(exported_nonempty_worlds=counts.get('exported_with_new_objects',0),
            observed_only_worlds=counts.get('observed_only_sample',0),unexported_worlds=sum(not r['exported'] for r in rows),
            exported_worlds_with_warnings=sum(r['exported'] and bool(r.get('warnings')) for r in rows)),
        records=rows,phase_records=all_phases,API_used=False,observer_rerun=False,model=semantic.VARIANT)
    write(root/'results.json',result)
    write(root/'comparison.json',dict(scenes=comparison,scope='same_observation_descriptive_architecture_comparison_not_causal_ablation'))
    page='<!doctype html><meta charset="utf-8"><title>全局假设联合补全</title><style>body{font:16px/1.6 system-ui;margin:24px}article{border:1px solid #ccc;padding:15px;margin:20px 0}img{max-width:100%}h2,pre{overflow-wrap:anywhere;white-space:pre-wrap}</style><h1>全局假设 → 共存区域 → 家具与用品 → 连续布局</h1>'
    if counts.get('pending_local_inference'):
        page+='<p>CPU缓存复核预览：'+str(counts['pending_local_inference'])+'份合法语义库存尚无对应尺寸/位置，等待本地模型推断；本预览没有执行模型推理。</p>'
    page+='<p>'+escape(json.dumps(counts,ensure_ascii=False))+'</p><p>蓝：冻结观测；橙：映射类别的新增；红：未映射新增。按近似合理的世界假设导出；墙面未测量、定性关系偏差等保留警告，具体见数值检查和G_full的layout_warnings。可见实例、宿主身份、视锥相容和显著穿插仍检查。每份世界独立，灰色ERP未生成RGB。提案失败保留为覆盖率记录，抽样条件为已有合法语义提案，不代表真实世界后验。</p><p><a href="results.json">完整结果</a> · <a href="comparison.json">基线对照与评审问题（数量不是质量评分）</a></p>'
    (root/'index.html').write_text(page+''.join(cards));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','prepare-repair','preview','check','report')+PHASES)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--observations-from',type=Path)
    parser.add_argument('--worlds',type=int,default=4);parser.add_argument('--draws',type=int,default=4);parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--source-run',type=Path)
    a=parser.parse_args();root=a.root.resolve()
    if a.action in ('prepare-repair','preview'):
        if a.source_run is None:parser.error('--source-run required')
        from .joint_world_repair import prepare_repair
        result=prepare_repair(root,a.source_run)
        if a.action=='preview':
            phase(root,'worlds',cached_only=True);summary=report(root)
            result=dict(status='preview_ready',counts=summary['counts'],API_used=False,GPU_inference_executed=False)
    elif a.action=='prepare':
        if a.observations_from is None:parser.error('--observations-from required')
        result=prepare(root,a.observations_from,a.worlds,a.draws,a.seed)
    elif a.action=='check':check(root);result=dict(status='verified_no_model_calls')
    elif a.action=='report':result=report(root)
    else:result=phase(root,a.action)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','phase_records')},ensure_ascii=False),flush=True)
    return 1 if result['status'] in ('failed','partial') else 0


if __name__=='__main__':sys.exit(main())
