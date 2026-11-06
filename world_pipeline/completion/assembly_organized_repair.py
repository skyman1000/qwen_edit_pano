"""Fork v13 and recover range-only schema failures without resampling its SCM."""
import argparse
from copy import deepcopy
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from PIL import Image

from ..common import read,write,sha,validate_new_output
from . import assembly_organized_geometry as geometry
from .assembly_experiment import check,code_hashes,SeededProvider,stage_seed
from .fresh_pipeline import provider
from .world_scm_runner import ask_valid

MARKER='range_repair_manifest.json'


def json_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*.json')) if p.is_file()}


def assert_hashes(root,expected):
    changed=[k for k,h in expected.items() if not (root/k).is_file() or sha(root/k)!=h]
    if changed:raise ValueError('Preserved artifacts changed: '+json.dumps(changed[:12]))


def inputs(root,entry):
    c=root/entry['branch']/entry['backend']/'completion';w=c/entry['world']
    return (c,w,read(c/'case.json'),read(c/'G_obs.json'),read(w/'semantic_world.json'),
        read(c/'support_plane.json'),read(w/'dimension_stage.json')['dimensions'],read(c/entry['initial_reply']))


def prepare(root,source):
    root=validate_new_output(root);source=Path(source).resolve();run=read(source/'run.json')
    if root==source or source in root.parents:raise ValueError('Repair output must be outside the source run')
    if run.get('version')!=13 or run.get('completion_model')!='organized':raise ValueError('Expected the v13 organized run root')
    if (source/MARKER).exists():raise ValueError('Use the original v13 run, not a previous repair copy')
    # Only the decoder/recovery orchestration may differ from the original run.
    allowed={'assembly_organized_geometry.py','assembly_organized_experiment.py','assembly_experiment.py'}
    pending=[];unsupported=[];source_changes={}
    for branch in run['experiments']:
        config=check(source/branch,strict_code=False)
        changes=config['source_changes_since_init'];source_changes[branch]=changes
        illegal=[p for p in changes if Path(p).name not in allowed]
        if illegal:raise ValueError('SCM/compiler or provider sources changed; range-only repair cannot certify reuse: '+str(illegal))
        for backend in run['backends']:
            c=source/branch/backend/'completion';s=read(c/'summary.json')
            observed=read(c/'G_obs.json')
            if observed.get('stage')!='prediction' or observed.get('scope')!='observed':raise ValueError('Predicted observations required; no GT replacement')
            if read(source/branch/backend/'observer3d/G_obs.json')!=observed:raise ValueError('Completion/observer binding changed')
            for failure in s['failures']:
                index=failure['world'];w=c/f'world_{index:02d}'
                if (w/'G_full.json').exists():continue
                entry=dict(branch=branch,backend=backend,world=w.name,index=index,
                    initial_reply=f'calls/layout_{index:02d}_placement.json',original_failure=failure)
                try:
                    _,_,case,obs,world,plane,dims,raw=inputs(source,entry)
                    captured=geometry.placement_capture(raw,world,obs,dims,case['hfov'])
                    if 'pending_ranges' not in captured:raise ValueError('Initial reply has no range-only failure')
                    entry.update(missing_group_ids=captured['missing_group_ids'],proposed_instances=len(world['objects']))
                    pending.append(entry)
                except (ValueError,KeyError,FileNotFoundError) as exc:unsupported.append(dict(**entry,reason=str(exc)))
    if not pending:raise ValueError('No range-only failures found; no copy or model calls needed')
    source_hashes=json_hashes(source)
    # Real copies: no hard links or write-through symlinks to original outputs.
    if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('Source contains symlinks; copy provenance requires ordinary files')
    root.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,root)
    repair=root/'range_repair';repair.mkdir()
    mutable={f"{e['branch']}/{e['backend']}/completion/summary.json" for e in pending}
    frozen={k:h for k,h in source_hashes.items() if k not in mutable}
    for relative in sorted(mutable):
        p=repair/'original_summaries'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,p)
    manifest=dict(revision='v13_range_only_recovery_1',source_root=str(source),created=datetime.now().isoformat(),
        pending=pending,unsupported_failures=unsupported,source_json_sha256=source_hashes,protected_copy_json_sha256=frozen,
        source_code_changes=source_changes,repair_code_sha256=code_hashes(),
        completed_states_reused=sum(k.endswith('/G_full.json') for k in source_hashes),
        initial_calls_by_backend={b:sum(e['backend']==b for e in pending) for b in run['backends']},
        maximum_calls_by_backend={b:2*sum(e['backend']==b for e in pending) for b in run['backends']},
        probabilities_changed=False,semantic_worlds_resampled=False,dimensions_changed=False,
        successful_worlds_recompiled=False,scene_gt_used=False,training_performed=False,
        comparison_scope='Recovery of the same sampled worlds; not fresh paired model generation')
    write(root/MARKER,manifest);assert_hashes(source,source_hashes);assert_hashes(root,frozen)
    return manifest


def verify(root,strict_code=True):
    root=Path(root).resolve()
    if not (root/MARKER).is_file():raise ValueError('Not a prepared range repair copy; original runs cannot be modified')
    m=read(root/MARKER);source=Path(m['source_root']).resolve()
    if root==source or source in root.parents:raise ValueError('Repair/source directory overlap')
    assert_hashes(source,m['source_json_sha256']);assert_hashes(root,m['protected_copy_json_sha256'])
    if strict_code:
        changed=[p for p,h in m['repair_code_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
        if changed:raise ValueError('Repair code changed after prepare; use a new repair output: '+str(changed))
    return m


def materialize(root,entry,layout,position_note=None):
    """Use the same compiler/export contract as assembly_world, no re-sampling."""
    from . import assembly_organized as scm
    from .assembly_pose import fit,diagnostics
    from .assembly_taxonomy import annotate
    from .scene_completion import export_states
    from .off_fov_completion import validate_world
    from .semantic_layout_visual import render
    c,dest,case,observed,world,plane,_,_=inputs(root,entry);model=read(c/'causal_model.json')
    solved=geometry.compile_layout(case,observed,world,layout,plane)
    write(dest/'execution.json',dict(layout=layout,solved=solved));write(dest/'numeric_review.json',solved['review'])
    pose=fit(world,solved['items'],observed);write(dest/'pose_model.json',pose)
    write(dest/'pose_interventions.json',diagnostics(pose,solved['items']))
    passed=solved['review']['geometry_contract_passed']
    full,_,unmapped=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
    full.update(completion_protocol=scm.PROTOCOL,target='outside_input_frustum',adapter_connected=False,
        completion_model_variant=scm.MODEL_VARIANT,geometry_parameterization=geometry.PARAMETERIZATION,
        world_state_ready=passed,world_state_ready_scope='numeric_candidate_not_semantic_certification',
        typed_relations=world['relations'],declared_support_plane=plane,
        spatial_program=deepcopy(world.get('spatial_program',{})),functional_links=deepcopy(world.get('functional_links',[])),
        unresolved_functional_links=deepcopy(world.get('unresolved_functional_links',[])))
    metadata={o['id']:o for o in world['objects']}
    for obj in full['objects'][len(observed['objects']):]:
        src=metadata[obj['track_id']];obj.update(variant=src['variant'],attachment=src['attachment'],
            generative_presence_node=src['presence_node'],installation_node=src['installation_node'],raw_category=src['category'])
        if 'presence_probability' in src:obj.update(generative_presence_probability=src['presence_probability'],
            presence_probability_scope=model['distribution_scope'],presence_probability_stage='before_geometry_no_failure_resampling')
        if not passed:obj['unseen_reason']='placement_not_verified_outside_input_view'
    full=geometry.annotate_state(annotate(full,world),layout)
    record=dict(folder=dest.name,new_instances=len(world['objects']),counts_kind='placement_roots_and_hosted',
        root_instances=sum(o['attachment']['host'] is None for o in world['objects']),
        hosted_instances=sum(o['attachment']['host'] is not None for o in world['objects']),unmapped_new=len(unmapped),
        observed_state_preserved=True,remaining_AABB_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']),
        status=('candidate_needs_semantic_review' if solved['review']['layout_proxy_passed'] else 'layout_needs_review') if passed else 'blocked_numeric_diagnostic')
    if passed:
        validate_world(observed,full,solved['items'],len(observed['objects'])+model.get('maximum_hidden_instances',24))
        hidden=deepcopy(full);hidden.update(scope='hidden',objects=deepcopy(full['objects'][len(observed['objects']):]))
        relations=dict(declared_relations=world['relations'],attachments=[dict(id=o['id'],**o['attachment']) for o in world['objects']],
            presence_parents={n['id']:n['parents'] for n in model['nodes'] if n['id'] in {o['presence_node'] for o in world['objects']}},
            installation_parents={n['id']:n['parents'] for n in model['nodes'] if n['id'] in {o['installation_node'] for o in world['objects']}})
    else:
        full.update(quality_status='blocked_numeric_diagnostic_not_completed_world',geometry_contract_passed=False)
        write(dest/'G_full_candidates.json',full);record.update(state_file='G_full_candidates.json',visual_folder='candidate_world')
    visual=dest/('world' if passed else 'candidate_world');visual.mkdir(exist_ok=False)
    with Image.open(root/entry['branch']/'input.png') as im:original=im.convert('RGB')
    render(full,original,case['hfov'],visual,record,position_note=position_note or '仅恢复缺失组距离；原库存、尺寸、有效朝向与相对偏移固定。数值通过不等于语义认证。')
    if passed:
        # A rendering error must not leave a falsely completed state behind.
        write(dest/'G_out.json',hidden);write(dest/'object_relations.json',relations);write(dest/'G_full.json',full)
    return passed,record


def execute(root,backend,p=None):
    root=Path(root).resolve();m=verify(root)
    if backend not in ('qwen','api'):raise ValueError('Unknown backend')
    target=root/'range_repair'/(backend+'.json')
    if target.exists():raise FileExistsError('This fixed-budget repair backend was already attempted: '+str(target))
    entries=[e for e in m['pending'] if e['backend']==backend]
    # No provider initialization (or API key) when there is no repair for it.
    if entries and p is None:p=provider(read(root/entries[0]['branch']/'experiment.json'),backend)
    if hasattr(p,'max_calls'):p.max_calls=2*len(entries)
    result=dict(backend=backend,status='running',initial_budget=len(entries),maximum_calls=2*len(entries),cases=[])
    write(target,result)
    for e in entries:
        c,dest,case,observed,world,plane,dims,raw=inputs(root,e);config=read(root/e['branch']/'experiment.json')
        attempt=dict(branch=e['branch'],world=e['world'],proposed_instances=len(world['objects']),missing_group_ids=e['missing_group_ids'],original_failure=e['original_failure'])
        try:
            missing=geometry.placement_capture(raw,world,observed,dims,case['hfov'])['missing_group_ids']
            stage=f"layout_{e['index']:02d}_ranges";calls=dest/'range_repair_calls';calls.mkdir()
            selected=SeededProvider(p,stage_seed(config['seed'],stage));image=None
            if config.get('layout_image_delivery'):
                from .scene_program_world import observation_board
                board=c/'observation_board.png'
                if not board.exists():raise ValueError('Saved matched observation board missing')
                with Image.open(board) as im:image=im.convert('RGB')
            print(json.dumps(dict(backend=backend,branch=e['branch'],world=e['world'],stage=stage,missing=missing)),flush=True)
            ranges=ask_valid(selected,image,geometry.ranges_prompt(case,observed,world,plane,dims,raw,missing),calls/(stage+'.txt'),lambda v:geometry.ranges_answer(v,missing),config['max_tokens'])
            layout,completed=geometry.apply_ranges(raw,ranges,world,observed,dims,case['hfov'])
            write(dest/'range_patch.json',dict(ranges=ranges,completed_placement=completed,only_requested_ranges_changed=True,initial_reply=e['initial_reply']))
            passed,record=materialize(root,e,layout)
            summary=read(c/'summary.json')
            if passed:
                summary['worlds'].append(record);summary['worlds'].sort(key=lambda r:r['folder'])
                summary['failures']=[f for f in summary['failures'] if f['world']!=e['index']]
                attempt.update(status='world_exported_needs_semantic_review',record=record)
            else:
                summary['blocked_worlds'].append(record)
                attempt.update(status='blocked_numeric_diagnostic',record=record)
                # Preserve the original schema failure, but make current reason explicit.
                for f in summary['failures']:
                    if f['world']==e['index']:f['range_recovery_status']='range_completed_numeric_contract_failed'
            summary.update(status='worlds_exported_needs_review' if summary['worlds'] and not summary['failures'] else 'partial_layouts_needs_review' if summary['worlds'] else 'no_world_exported',
                world_state_ready=bool(summary['worlds']),range_only_recovery=True)
            write(c/'summary.json',summary)
            from .assembly_world import page
            page(c,summary)
        except Exception as exc:
            attempt.update(status='repair_failed',error=type(exc).__name__+': '+str(exc))
            (dest/'range_repair_error.txt').write_text(traceback.format_exc())
        result['cases'].append(attempt);write(dest/'range_repair_result.json',attempt);write(target,result)
    result.update(status='complete' if all(c['status']=='world_exported_needs_semantic_review' for c in result['cases']) else 'partial_or_failed',
        provider_calls=getattr(p,'calls',None),api_usage=getattr(p,'usage',None))
    write(target,result);verify(root);return result


def report(root):
    from .assembly_holistic_visual import report as visual,table
    root=Path(root).resolve();m=verify(root);run=read(root/'run.json')
    # A new numbered gallery allows a prepared-only inspection followed by execution.
    n=1
    while (root/f'visual_range_repair_{n:03d}').exists():n+=1
    page=visual(root,Path(m['source_root']),root/f'visual_range_repair_{n:03d}',current_label='v13.range_repair',
        gallery_experiments=run['experiments'],baseline_experiments=run['experiments'])
    outcomes={}
    for b in run['backends']:
        p=root/'range_repair'/(b+'.json')
        if p.exists():outcomes[b]=read(p)
    rows=[]
    for e in m['pending']:
        status=next((c['status'] for c in outcomes.get(e['backend'],{}).get('cases',[]) if c['branch']==e['branch'] and c['world']==e['world']),'not_attempted')
        rows.append([e['branch'],e['backend'],e['world'],e['proposed_instances'],', '.join(e['missing_group_ids']),status])
    intro='<h2>缺失距离恢复：同一批世界，因果概率没有重估</h2><p>原版成功状态逐字节复用。初次有效 yaw、偏移、尺寸和库存固定，仅补缺失组距离；失败、重复、空集继续保留。数值成功不认证服务关系、真实房间范围或照明。此恢复不能当作新一轮配对模型能力对比。</p>'
    intro+=table(['分支','后端','世界','新增库存','补距组','本次状态'],rows)
    original_link=os.path.relpath(Path(m['source_root'])/'index.html',page.parent)
    intro+='<p>蓝：观测；橙：原名在类别表；紫：原名表外但可映射；红：未映射，仍保留。关系与尺寸问题继续参见 <a href="'+escape(original_link)+'">原版结果</a>；本次没有自动改服务目标或缩小物体。</p>'
    # Remove generic cross-version warnings which do not describe this same-sample repair.
    content=page.read_text().replace('跨版本候选域不同，差异不是单个箭头的因果效应。','此对照候选域、概率表和语义世界相同。')
    content=content.replace('RGB 对照仅改变选择阶段图像输入，本地初始 draw seed 相同；服务端 API 未传 seed。','')
    if run.get('synthetic_fixture'):intro='<h1>CPU 合成 fixture，非真实模型效果</h1>'+intro
    page.write_text(content+intro);verify(root);return page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','report','pending'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-run',type=Path)
    parser.add_argument('--backend',choices=('qwen','api'))
    a=parser.parse_args()
    try:
        if a.action=='prepare':
            if not a.source_run:parser.error('--source-run required')
            m=prepare(a.root,a.source_run);print(json.dumps(dict(root=str(a.root),calls=m['initial_calls_by_backend']),ensure_ascii=False))
        elif a.action=='pending':
            if not a.backend:parser.error('--backend required')
            print(verify(a.root)['initial_calls_by_backend'][a.backend])
        elif a.action=='run':
            if not a.backend:parser.error('--backend required')
            r=execute(a.root,a.backend);print(json.dumps(r,ensure_ascii=False));sys.exit(0 if r['status']=='complete' else 1)
        else:print(report(a.root))
    except (ValueError,FileNotFoundError,FileExistsError) as exc:parser.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
