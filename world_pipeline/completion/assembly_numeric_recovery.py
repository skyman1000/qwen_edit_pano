"""Recompile saved numeric candidates; never initialize a model/provider.

Successful states remain byte-identical. Failed world folders are archived
inside the new copy, retaining their JSON hashes, before a clean recompile.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import traceback

from ..common import read,write,sha
from .assembly_multiscene import DEFAULT_INPUTS,verify_suite
from .assembly_multiscene_fork import fork_suite,verify_recovery


def plan(source):
    source=Path(source).resolve();suite=read(source/'suite.json');pending=[];complete=[]
    if suite.get('mode')!='single' or suite.get('worlds')!=1:raise ValueError('Numeric recovery requires one world per input/backend')
    for scene in suite['scenes']:
        for backend in ('qwen','api'):
            c=source/'scenes'/scene['id']/'completion/end_to_end'/backend/'completion';d=c/'world_01'
            if (d/'G_full.json').exists():complete.append(str((d/'G_full.json').relative_to(source)));continue
            required=[d/'execution.json',d/'numeric_review.json',d/'semantic_world.json',d/'dimension_stage.json',c/'causal_model.json',c/'summary.json']
            if any(not p.exists() for p in required):raise ValueError('Not a fully generated numeric candidate: '+str(d))
            review=read(d/'numeric_review.json')
            if review['geometry_contract_passed']:raise ValueError('Numeric passed but export absent; inspect export error separately: '+str(d))
            layout=read(d/'execution.json')['layout'];world=read(d/'semantic_world.json');dims=read(d/'dimension_stage.json')['dimensions']
            rows={o['id']:o for g in layout.values() for o in g['objects']}
            if set(rows)!={o['id'] for o in world['objects']} or any(rows[k]['intrinsic_size_m']!=dims[k]['full_size_m'] for k in rows):raise ValueError('Frozen inventory/dimension binding changed')
            pending.append(dict(scene=scene['id'],branch='end_to_end',backend=backend,world='world_01',index=1,
                proposed_instances=len(world['objects']),previous_overlaps=len(review['program_solver']['final_overlaps'])))
    return dict(source_root=str(source),pending=pending,complete_worlds=complete,model_calls_required=0)


def prepare(root,source,inputs=DEFAULT_INPUTS):
    source=Path(source).resolve();root=Path(root).resolve();p=plan(source)
    fork_suite(root,source,inputs,numeric_recovery=True)
    manifest=read(root/'recovery_manifest.json');protected=manifest['protected_copy_json_sha256'];relocations=[]
    for case in p['pending']:
        relative=Path('scenes')/case['scene']/'completion'/case['branch']/case['backend']/'completion'/case['world']
        history=Path('numeric_solver_history')/case['scene']/case['branch']/case['backend']/case['world']
        current=root/relative;archive=root/history;archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(current),str(archive));current.mkdir()
        prefix=str(relative)+'/'
        for k in list(protected):
            if k.startswith(prefix):protected[str(history)+k[len(str(relative)):]]=protected.pop(k)
        for filename in ('semantic_world.json','dimension_stage.json'):
            shutil.copyfile(archive/filename,current/filename);protected[str(relative/filename)]=sha(current/filename)
        write(current/'numeric_recovery_input.json',dict(layout=read(archive/'execution.json')['layout'],source_execution=str(source/relative/'execution.json')))
        protected[str(relative/'numeric_recovery_input.json')]=sha(current/'numeric_recovery_input.json')
        relocations.append(dict(from_folder=str(relative),archive_folder=str(history),semantic_world_changed=False,dimensions_changed=False))
    manifest.update(protected_copy_json_sha256=protected,numeric_history_relocations=relocations)
    write(root/'recovery_manifest.json',manifest)
    suite=read(root/'suite.json');suite.update(recovery_scope='Recompile only saved numeric candidates with a bounded adaptive collision search; '+str(len(p['complete_worlds']))+' complete states unchanged; no model calls',numeric_recovery_only=True)
    write(root/'suite.json',suite);p.update(root=str(root),history_relocations=relocations,SCM_changed=False,probabilities_resampled=False,worlds_resampled=False)
    write(root/'numeric_recovery_manifest.json',p);verify_suite(root);return p


def execute(root):
    from .assembly_organized_repair import materialize
    from .assembly_world import page
    root=Path(root).resolve();verify_suite(root);p=read(root/'numeric_recovery_manifest.json');target=root/'numeric_recovery_result.json'
    if target.exists():return read(target)
    results=[]
    for case in p['pending']:
        comp=root/'scenes'/case['scene']/'completion';c=comp/case['branch']/case['backend']/'completion';d=c/case['world'];summary=read(c/'summary.json')
        write(c/'summary_before_numeric_search_recovery.json',summary)
        entry=dict(branch=case['branch'],backend=case['backend'],world=case['world'],index=case['index'],initial_reply=case['world']+'/numeric_recovery_input.json')
        try:
            print('Recompile saved world: '+case['scene']+' '+case['backend'],flush=True)
            passed,record=materialize(comp,entry,read(d/'numeric_recovery_input.json')['layout'],
                position_note='固定原对象、尺寸、朝向、宿主、相机区域和因果概率，仅增加碰撞分离搜索预算；未调用Qwen/API。')
            review=read(d/'numeric_review.json');solver=review['program_solver']
            for key in ('coordinate_transport_recovery','coordinate_recovery','named_coordinate_recovery'):
                if key in summary:summary.setdefault('prior_recovery_results',{})[key]=summary.pop(key)
            summary['blocked_worlds']=[r for r in summary.get('blocked_worlds',[]) if r['folder']!=case['world']]
            summary['failures']=[f for f in summary.get('failures',[]) if f['world']!=case['index']]
            if passed:summary['worlds'].append(record)
            else:
                summary['blocked_worlds'].append(record);summary['failures'].append(dict(world=case['index'],stage='compile_'+str(case['index']),
                    proposed_instances=case['proposed_instances'],error='Numeric geometry contract failed; residual overlap pairs='+str(len(solver['final_overlaps']))))
            result=dict(**case,status='exported' if passed else 'blocked_numeric',after_overlaps=len(solver['final_overlaps']),
                search_rounds=solver.get('collision_search_rounds_used'),search_budget=solver.get('collision_search_round_budget'),
                search_termination=solver.get('collision_search_termination'),record=record,model_calls=0)
            summary.update(status='worlds_exported_needs_review' if passed else 'no_world_exported',world_state_ready=passed,numeric_solver_recovery=result)
            write(c/'summary.json',summary);page(c,summary)
            write(root/'phase_records'/(case['scene']+'_worlds_'+case['backend']+'.json'),dict(scene=case['scene'],phase='worlds',backend=case['backend'],
                status='complete' if passed else 'partial',numeric_recovery_only=True,result={case['branch']:summary}))
        except Exception as exc:
            result=dict(**case,status='failed',error=type(exc).__name__+': '+str(exc),model_calls=0);(d/'numeric_recovery_error.txt').write_text(traceback.format_exc())
        results.append(result);print(str({k:result.get(k) for k in ('scene','status','previous_overlaps','after_overlaps','search_rounds')}),flush=True)
    verify_recovery(root)
    answer=dict(status='complete' if all(r['status']=='exported' for r in results) else 'partial',cases=results,
        complete_worlds_reused=len(p['complete_worlds']),model_calls=0,source_modified=False)
    write(target,answer);return answer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','prepare','execute','run'));p.add_argument('--source-root',type=Path);p.add_argument('--root',type=Path);p.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS);a=p.parse_args()
    try:
        if a.action=='plan':print(plan(a.source_root));return
        if a.root is None:p.error('--root required')
        if a.action in ('prepare','run'):
            if a.source_root is None:p.error('--source-root required')
            print(prepare(a.root,a.source_root,a.inputs),flush=True)
        if a.action in ('execute','run'):
            result=execute(a.root)
            from .assembly_multiscene_report import report_suite
            print(report_suite(a.root),flush=True);sys.exit(0 if result['status']=='complete' else 1)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
