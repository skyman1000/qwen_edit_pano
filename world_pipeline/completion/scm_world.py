"""Manual-run parent-only semantic SCM posterior -> fixed instances -> layout.

No training, GT, Adapter, or GPU submission from preflight. Count posterior is
explicit; metric layout is a conditional approximation, not a joint posterior.
"""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil
import traceback

from PIL import Image

from ..common import new_output,read,sha,validate_graph,write
from .batch_world_review import inventory_image
from .causal_world_completion import check_hashes,visible_inventory
from .conditional_world_geometry import support_plane
from .frozen_causal_world import ask_validated,load_case
from .off_fov_completion import validate_world
from .qwen_upstream import DEFAULT_MODEL,Qwen
from .scm_count_model import draw,infer,trace
from .scm_mechanisms import TextMechanismScorer,domain_answer,parameterize,registration_answer
from .scm_world_layout import instantiate,solve_batches,support_answer
from .scm_typed_layout import solve_typed
from .scm_typed_queries import ContinuationMechanismScorer
from .scm_numeric_mechanisms import NumericCountScorer
from .scm_relational_layout import solve_relational
from .scm_checkpoint import copy_checkpoint,reasoning_report,validate_checkpoint
from .scene_completion import export_states,map_plan
from .semantic_layout_visual import render

PROTOCOL='numeric_count_role_conditioned_joint_world_v4'


def protocol_for(layout_backend,numeric=False):
    if layout_backend=='joint' or numeric:return PROTOCOL
    return 'parent_only_censored_count_scm_world_v3' if layout_backend=='typed' else 'parent_only_censored_count_scm_world_v1'


def code_hashes():
    from .surrounding_world import code_hashes as previous
    names=('scm_world.py','scm_count_model.py','scm_mechanisms.py','scm_world_layout.py','scm_world_audit.py',
           'semantic_categories.py','semantic_layout.py','semantic_layout_visual.py','json_generation.py',
           'scm_typed_queries.py','scm_typed_layout.py','scm_checkpoint.py',
           'scm_numeric_mechanisms.py','scm_relational_layout.py','scm_joint_geometry.py')
    files=[Path(__file__).with_name(n) for n in names]+[Path(__file__).parent.parent/'common.py']
    return {**previous(),**{str(p.resolve()):sha(p) for p in files}}


def diagram(folder):
    (folder/'causal_graph.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="300">'
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        '<path d="M0 0 L7 3 L0 6" fill="none" stroke="black"/></marker></defs><rect width="100%" height="100%" fill="white"/>'
        '<g font-family="sans-serif" font-size="17"><text x="30" y="35">Assumed generative semantic model; probabilities are uncalibrated</text>'
        '<text x="30" y="100">B: background</text><text x="280" y="100">K: configuration</text>'
        '<text x="540" y="100">N_t: full counts</text><text x="815" y="100">O_t: recorded counts</text>'
        '<path d="M180 95 L265 95 M445 95 L525 95 M190 78 Q365 0 550 78 M690 95 L800 95" fill="none" stroke="black" marker-end="url(#a)"/>'
        '<text x="580" y="170">F: unknown view fraction; D: recording rate</text>'
        '<path d="M865 150 L890 110" fill="none" stroke="black" marker-end="url(#a)"/>'
        '<text x="30" y="230">External posterior sampling determines outside counts; inside/undetected instances are not exported.</text>'
        '<text x="30" y="270">Metric layout: conditional approximation. Camera-pose likelihood and true gravity are not modeled.</text></g></svg>')


def page(folder,record):
    body=['<!doctype html><meta charset="utf-8"><title>CauPano semantic SCM</title>',
        '<h1>生成机制、观测后验与世界状态</h1><p>2:1 图是状态方向图；未生成全景照片。语义后验为声明模型下的结果；三维布局是条件近似。</p>',
        '<pre>'+escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre>']
    for name,label in (('G_obs.json','冻结观测'),('semantic_model.json','生成机制'),
                       ('inference_trace.json','观测后验和数量诊断'),('mechanism_interventions.json','机制干预诊断'),
                       ('causal_reasoning.json','当前模型实际推断内容及先验敏感性')):
        if (folder/name).exists():body.append('<p><a href="'+name+'">'+label+'</a></p>')
    if (folder/'causal_graph.svg').exists():body.append('<img src="causal_graph.svg" width="1000">')
    if (folder/'qwen_2d.png').exists():body.append('<img src="qwen_2d.png" width="700">')
    for h in record.get('hypotheses',[]):
        dest=folder/h['folder'];body.append('<h2>'+escape(h['folder'])+'</h2>')
        if (dest/'semantic_plan.json').exists():
            plan=read(dest/'semantic_plan.json');body.append('<p>'+escape(', '.join(o['name'] for o in plan['additions']) or '本次采样没有视野外新增对象')+'</p>')
        if (dest/'numeric_review.json').exists():
            body.append('<p><a href="'+h['folder']+'/numeric_review.json">布局检查与未解决项</a></p>')
        if (dest/'joint_layout.json').exists():
            body.append('<p><a href="'+h['folder']+'/joint_layout.json">安装角色、宿主与联合求解记录</a></p>')
        if (dest/'world/index.html').exists():
            body.append('<a href="'+h['folder']+'/world/index.html">打开3D、2:1方向图和局部投影</a>')
    (folder/'index.html').write_text(''.join(body))


def run(model,scorer,case,output,max_count=6,generic_types=8,worlds=3,seed=3407,max_tokens=8192,
        batch_size=6,capacity=256,layout_backend='legacy',semantic_source=None):
    if layout_backend not in ('legacy','typed','joint'):raise ValueError('Unknown layout backend')
    protocol=protocol_for(layout_backend,getattr(scorer,'count_event_contract',None)=='json_integer_v1')
    folder=output/case['id'];folder.mkdir();hypotheses=[]
    record=dict(id=case['id'],protocol=protocol,layout_backend=layout_backend,
        semantic_checkpoint_reused=semantic_source is not None,status='failed',world_state_ready=False,
        observed_instances=len(case['context']),scene_gt_used=False,training_performed=False,
        adapter_connected=False,panorama_generated=False,semantics_verified=False,geometry_verified=False,
        causal_gain_validated=False,semantic_count_posterior_defined=False,joint_metric_posterior_defined=False,
        camera_pose_likelihood_defined=False)
    options=dict(seed=seed,sample=True,profile='official',max_tokens=max_tokens,guard_repetition=True)
    stage='input_validation'
    try:
        check_hashes(case['source_sha256']);observed=validate_graph(read(case['state']))
        write(folder/'input.json',case);write(folder/'G_obs.json',observed)
        shutil.copyfile(case['image'],folder/'local.png')
        inventory_image(case['image'],read(case['inventory']),folder/'qwen_2d.png')
        with Image.open(case['image']) as f:original=f.convert('RGB')
        with Image.open(folder/'qwen_2d.png') as f:marked=f.convert('RGB')
        marked.thumbnail((1024,1024))
        if semantic_source is not None:
            stage='semantic_checkpoint'
            state=copy_checkpoint(semantic_source,case,folder,max_count,generic_types)
            domain,registration,plane,spec=(state[k] for k in ('domain','registration','plane','spec'))
            diagram(folder)
        else:
            # This catalog call is text-only and receives no case information.
            stage='semantic_domain'
            domain=ask_validated(model,None,
                'Define a small finite hypothesis domain for GENERIC existing indoor household rooms, without '
                'any photograph or observed inventory. Return {"backgrounds":array,"configurations":array,'
                '"object_types":array}. backgrounds: 2..3 distinct short strings covering different possible '
                'room uses, space conditions and furnishing preferences. configurations: 2..3 short strings '
                'describing different furnishing patterns applicable across backgrounds. '
                f'object_types: up to {generic_types} distinct ordinary object nouns spanning furniture and '
                'secondary objects. Comparable category granularity, no synonyms duplicated. No room envelope '
                '(walls/floor/ceiling) and no superclass furniture/object. These are finite hypotheses, not facts '
                'about a pictured room. Return no probabilities or required object counts.',
                folder/'domain.txt',lambda v:domain_answer(v,generic_types),options)
            write(folder/'domain.normalized.json',domain)
            stage='observed_registration'
            registration=ask_validated(model,marked,
                'Register ONLY the fixed observed instances below; do not propose hidden objects or infer room use. '
                'Return {"instances":[{"id":string,"object_type":string}],"floor_anchor_ids":array}. '
                'Every observed ID exactly once. Use a specific ordinary canonical noun without colors/materials; '
                'prefer an existing catalog noun when semantically equivalent. All pillows of the same category '
                'use the same noun. New specific observed nouns may be added; do not merge distinct categories. '
                'floor_anchor_ids: observed furniture plausibly resting directly on the same floor; exclude '
                'pillows, trays and supported table tops. [] is allowed. No geometry or category IDs are changed.'+
                '\nNOUN CATALOG: '+json.dumps(domain['object_types'])+'\nOBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False),
                folder/'registration.txt',lambda v:registration_answer(v,case,domain),{**options,'seed':seed+2})
            write(folder/'registration.normalized.json',registration)
            if max(registration['observed_counts'].values())>max_count:
                raise ValueError('Observed count exceeds --max-count; counts are never clipped')
            if max_count*len(registration['object_types'])>capacity:
                raise ValueError('Finite count domain exceeds --max-objects capacity; no hidden truncation')
            plane=support_plane(observed,registration['floor_anchor_ids']);write(folder/'support_plane.json',plane)
            # Parameters receive no image, observed count, case ID or observed reason.
            stage='generative_mechanisms'
            spec,diagnostics=parameterize(domain,registration['object_types'],max_count,scorer,folder/'mechanisms')
            write(folder/'semantic_model.json',spec);write(folder/'score_diagnostics.json',diagnostics);diagram(folder)
        if max_count*len(registration['object_types'])>capacity:
            raise ValueError('Finite count domain exceeds --max-objects capacity; no hidden truncation')
        stage='count_posterior'
        record['count_event_contract']=spec.get('count_event_contract','sentence_v1')
        result=infer(spec,registration['observed_counts']);inference=trace(result)
        write(folder/'inference_trace.json',inference)
        write(folder/'causal_reasoning.json',reasoning_report(spec,result))
        interventions={k['id']:dict(prior=trace(infer(spec,{}, {'K':k['id']})),
                                   conditioned=trace(infer(spec,registration['observed_counts'],{'K':k['id']}))) for k in domain['configurations']}
        write(folder/'mechanism_interventions.json',dict(regimes=interventions,
            interpretation='mechanism_replacement_under_assumed_model_not_identified_world_effect'))
        record.update(semantic_count_posterior_defined=True,background_posterior=inference['background_posterior'],
            background_update_total_variation=inference['background_total_variation'],
            expected_outside_instances=inference['completion_diagnostics']['expected_outside_instances'],
            probability_zero_outside=inference['completion_diagnostics']['probability_zero_outside'])
        stage='world_samples'
        for rank in range(worlds):
            dest=folder/f'hypothesis_{rank+1:02d}';dest.mkdir();sample=draw(result,seed+1000+rank)
            write(dest/'posterior_sample.json',sample)
            plan=instantiate(sample,case,domain);write(dest/'sampled_instances.json',plan)
            h=dict(folder=dest.name,id=plan['id'],sample_seed=sample['seed'],status='failed',
                world_state_ready=False,proposed_instances=sample['new_instances'])
            world_stage='supports'
            try:
                if layout_backend=='joint':
                    world_stage='role_conditioned_joint_layout'
                    plan,mappings,items,review=solve_relational(model,marked,case,observed,plan,plane,dest,
                        {**options,'seed':seed+3000+rank*100})
                elif layout_backend=='typed':
                    world_stage='typed_layout'
                    plan,mappings,items,review=solve_typed(model,marked,case,observed,plan,plane,dest,
                        {**options,'seed':seed+3000+rank*100})
                elif plan['additions']:
                    prompt=('Assign off-camera qualitative locations and supports for these FIXED externally '
                        'sampled instances. Do not select, withdraw, merge, add or rename objects. '
                        'Return {"objects":[{"id":string,"placement":string,"support":{"kind":string,"target":null}}]}. '
                        'Support kinds floor (target null), object (a distinct valid observed/new ID), '
                        'unspecified/wall/ceiling (target null). Object means actual supporting contact, '
                        'not merely near a piece of furniture. Exact ID references; no support cycles. '
                        'A wall adjacent to a cabinet is not proof of wall support. Every placement must '
                        'be outside the ORIGINAL camera cone, not just behind a visible sofa. '
                        'This is a metric layout approximation, not another hidden-inventory proposal. '+
                        '\nFIXED SAMPLED PLAN: '+json.dumps(plan,ensure_ascii=False)+
                        '\nOBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False)+
                        '\nSUPPORT PRIOR: '+json.dumps(plane,ensure_ascii=False))
                    if plane['floor_y_m'] is None:
                        prompt+='\nNo floor anchor exists. Use unspecified support with later signed camera-Y estimation; do not assert measured floor contact.'
                    plan=ask_validated(model,marked,prompt,dest/'supports.txt',
                        lambda v:support_answer(v,plan,case),{**options,'seed':seed+2000+rank*10})
                write(dest/'semantic_plan.json',plan)
                if layout_backend=='legacy':mappings=map_plan(plan,'semantic_lexicon_v1')
                write(dest/'category_mapping.json',mappings)
                if layout_backend=='legacy':
                    world_stage='metric_layout'
                    items,review=solve_batches(model,marked,case,observed,plan,mappings,plane,dest,
                        {**options,'seed':seed+3000+rank*100},batch_size)
                world_stage='export_and_visualization'
                full,_,unmapped=export_states(observed,plan,items,mappings,review)
                full.update(completion_protocol=protocol,target='outside_input_frustum',adapter_connected=False,
                    semantic_count_posterior_defined=True,joint_metric_posterior_defined=False,
                    semantics_verified=False,geometry_verified=False,causal_gain_validated=False,
                    posterior_sample=sample,world_relations=plan['relations'],declared_support_plane=plane,
                    metric_layout_method=('role_conditioned_joint_XZ_soft_AABB_repulsion_no_posterior_density' if layout_backend=='joint' else
                        'typed_support_fixed_Y_off_frustum_solver_no_posterior_density' if layout_backend=='typed'
                        else 'conditional_VLM_bearing_support_approximation_no_posterior_density'),
                    semantic_plan_fully_exported=True)
                validate_world(observed,full,items,capacity)
                h.update(status='candidate_needs_review' if review['geometry_contract_passed'] else 'geometry_needs_review',
                    world_state_ready=review['geometry_contract_passed'],new_instances=len(items),unmapped_new=len(unmapped),
                    off_fov_passed=review['numeric_constraints_passed'],observed_state_preserved=True)
                if layout_backend=='joint':
                    full['layout_installation_roles']=read(dest/'joint_layout.json')['profiles']
                    h.update(layout_proxy_passed=review['layout_proxy_passed'],
                        remaining_AABB_overlap_pairs=len(review['joint_layout']['final_overlaps']),
                        unresolved_installations=review['unresolved_installations'])
                    if not review['layout_proxy_passed']:h['status']='layout_needs_review'
                if not items:h['status']='empty_outside_sample_needs_review'
                full.update(world_state_ready=h['world_state_ready'],quality_status=h['status'])
                write(dest/'G_full.json',full);hidden=deepcopy(full);hidden['scope']='hidden'
                hidden['objects']=full['objects'][len(observed['objects']):];write(dest/'G_out.json',validate_graph(hidden))
                write(dest/'layout_solved.json',dict(objects=items));write(dest/'numeric_review.json',review)
                visual=dest/'world';visual.mkdir()
                render(full,original,case['hfov'],visual,h,position_note='程序采样语义数量；三维为条件近似。观测保持不变，未估计三维联合后验。')
                vp=visual/'index.html';vp.write_text(vp.read_text().replace(
                    '红色候选保留在G_full_candidates中，不进入Adapter子集G_full。','未映射对象全部保留；本阶段不连接Adapter。'))
            except Exception as exc:
                h.update(error=f'{type(exc).__name__}: {exc}',failed_stage=world_stage);(dest/'error_traceback.txt').write_text(traceback.format_exc())
            write(dest/'summary.json',h);hypotheses.append(h)
        failed=sum(h['status']=='failed' for h in hypotheses)
        record.update(status='partial_failure' if failed else 'posterior_samples_exported_needs_review',hypotheses=hypotheses,
            hypotheses_exported=len(hypotheses)-failed,world_state_ready=any(h['world_state_ready'] for h in hypotheses),
            observed_state_preserved=True,world_state_ready_scope='numeric_export_contract_not_semantic_certification')
        # All samples are samples, not ranked probabilities. Root alias is first draw only.
        if hypotheses and hypotheses[0]['world_state_ready']:
            for name in ('G_full.json','G_out.json'):shutil.copyfile(folder/hypotheses[0]['folder']/name,folder/name)
            record['root_world_sample']=hypotheses[0]['folder']
        check_hashes(case['source_sha256'])
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',failed_stage=stage);(folder/'error_traceback.txt').write_text(traceback.format_exc())
    record['hypotheses']=hypotheses;write(folder/'summary.json',record);page(folder,record)
    write(output/'summary.json',dict(protocol=protocol,records=[record],scene_gt_used=False))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="'+case['id']+'/index.html">打开生成机制、后验和世界可视化</a>')
    print(json.dumps(record,ensure_ascii=False),flush=True);return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',type=Path)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--max-count',type=int,default=6)
    p.add_argument('--generic-types',type=int,default=8);p.add_argument('--worlds',type=int,default=3)
    p.add_argument('--max-tokens',type=int,default=8192);p.add_argument('--layout-batch',type=int,default=6)
    p.add_argument('--max-objects',type=int,default=256);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--layout-backend',choices=('joint','typed','legacy'),default='joint')
    p.add_argument('--mechanism-scoring',choices=('numeric','continuation','legacy_codes'),default='numeric')
    p.add_argument('--reuse-semantic-from',type=Path)
    p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if (a.output.exists() or not 1<=a.max_count<=16 or not 1<=a.generic_types<=12 or not 1<=a.worlds<=8
        or not 1<=a.layout_batch<=8 or not 512<=a.max_tokens<=16384 or not 1<=a.max_objects<=256
        or not 0<=a.seed<2**31):p.error('Need NEW output and valid finite computational budgets')
    case=load_case(a.case)
    protocol=protocol_for(a.layout_backend,a.mechanism_scoring=='numeric')
    if len(case['context'])>a.max_objects or not a.model.is_dir():p.error('Observed state exceeds capacity or local model missing')
    if a.reuse_semantic_from is not None:
        if a.mechanism_scoring not in ('continuation','numeric'):p.error('Semantic reuse requires continuation/numeric mechanisms')
        try:validate_checkpoint(a.reuse_semantic_from,case,a.max_count,a.generic_types,a.mechanism_scoring)
        except (ValueError,KeyError,OSError) as exc:p.error('Semantic checkpoint rejected: '+str(exc))
    if a.preflight:
        print(json.dumps(dict(status='preflight_passed',protocol=protocol,layout_backend=a.layout_backend,
            mechanism_scoring=a.mechanism_scoring,id=case['id'],observed=len(case['context']),
            semantic_checkpoint_reused=a.reuse_semantic_from is not None,
            gpu_started=False,model_loaded=False,semantic_posterior='parent_only_counts_with_censored_observations',
            joint_metric_posterior_defined=False,adapter_connected=False,scene_gt_used=False)));return
    output=new_output(a.output);hashes=code_hashes()
    write(output/'config.json',dict(protocol=protocol,case=case,code_sha256=hashes,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    try:model=Qwen(a.model)
    except Exception as exc:
        write(output/'summary.json',dict(status='model_load_failed',error=f'{type(exc).__name__}: {exc}'))
        raise
    if code_hashes()!=hashes:raise ValueError('Code changed during model loading')
    scorer=(NumericCountScorer(model) if a.mechanism_scoring=='numeric' else
        ContinuationMechanismScorer(model) if a.mechanism_scoring=='continuation' else TextMechanismScorer(model))
    record=run(model,scorer,case,output,a.max_count,a.generic_types,a.worlds,a.seed,
        a.max_tokens,a.layout_batch,a.max_objects,a.layout_backend,a.reuse_semantic_from)
    if record['status'] in ('failed','partial_failure'):raise SystemExit('Inspect preserved raw replies and summary; samples are never silently removed.')


if __name__=='__main__':main()
