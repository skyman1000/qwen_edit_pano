"""Scope-correct proposal extension; reuses frozen simple_world v1 utilities."""
import argparse
from html import escape
import json
from pathlib import Path
import sys
import traceback

from ...common import ROOT,read,sha
from ..simple_world import workflow as old
from ..candidate_repair.requests import Requests,save
from ..assembly_experiment import stage_seed,check as context_check
from ..fresh_pipeline import provider
from ..assembly_world import _run
from .. import assembly_coordinate_transport as geometry
from . import contract,model


def sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'qwen_edit_pano/scripts/scope_world.sh']
    return {str(p.resolve()):sha(p) for p in paths}


def prepare(root,source,worlds=2):
    answer=old.prepare(root,source,worlds)
    save(root/'scope_world.json',dict(model=model.MODEL_VARIANT,source_sha256=sources(),
        max_hypotheses=2,max_areas_per_hypothesis=4,max_world_objects=24,
        max_initial_semantic_calls_per_scene=9,LM_probability_table_calls=0,
        pipeline='scope_then_area_instantiation_then_original_geometry',API_used=False))
    return dict(answer,model=model.MODEL_VARIANT,scheduled_phases=['prepare-scenes','propose','worlds'])


def check(root):
    suite=old.check(root); meta=read(root/'scope_world.json')
    if meta['source_sha256']!=sources(): raise ValueError('Bound scope implementation changed; start a new output directory')
    return suite


def propose_scene(comp,client):
    case=read(comp/'shared/case.json');run=read(comp/'run.json');dest=comp/'proposals/qwen'
    req=Requests(dest/'calls',stage_seed(run['seed'],comp.parent.name+':scope'),comp/'shared/observation_board.png',client)
    hs=[];programs={};rejected=[];failures=[]
    def failure(name,exc,**context):
        path=req.folder/(name+'.receipt.json'); receipt=read(path) if path.exists() else {}; error=receipt.get('error','')
        schema=receipt.get('status')=='failed' and error.startswith('ValueError: ') and str(exc) in (error,error[12:])
        row=dict(context,error=error if schema else type(exc).__name__+': '+str(exc))
        if schema: row['type']='proposal_contract_rejection'
        (rejected if schema else failures).append(row)
    try:
        hs=req.ask('surrounding_scope',contract.hypotheses_prompt(case),lambda v:contract.hypotheses_answer(v,case),3072)
        for h in hs:
            answers={};used=0;prior=[]
            for area in h['hidden_areas']:
                name=h['id']+'_'+area['id']
                validator=contract.AreaValidator(case,h,area,24-used)
                try:
                    if used>=24:
                        a=dict(status='unresolved',reason='Shared interface capacity reached; this area is unresolved, not absent',group=None,relations=[])
                    else:
                        a=req.ask(name,contract.area_prompt(case,h,area,24-used,prior),
                            validator,4096)
                except Exception as exc:
                    failure(name,exc,hypothesis=h['id'],area=area['id'])
                    # A replayed failed receipt raises before the validator is
                    # called. Recover display-only commitments from its bound
                    # raw first reply, never invent or instantiate them.
                    first=req.folder/(name+'_attempt0.json')
                    if validator.commitment is None and first.exists():
                        rows=read(first).get('new_objects')
                        if isinstance(rows,list) and all(isinstance(o,dict) and isinstance(o.get('key'),str)
                            and isinstance(o.get('category'),str) for o in rows):
                            validator.commitment={o['key']:o['category'] for o in rows}
                    a=dict(status='unresolved',reason='Fixed-budget area request failed; see immutable request journal',group=None,relations=[],
                        uninstantiated_inventory=validator.commitment or {})
                answers[area['id']]=a
                if a['group'] is not None: used+=len(a['group']['new_objects'])
                prior.append(dict(area=area['id'],status=a['status'],
                    objects=[dict(key=o['key'],category=o['category'],attachment=o['attachment']) for o in a['group']['new_objects']] if a['group'] else []))
            p=contract.assemble(case,h,answers)
            p['proposal_source']=dict(backend='qwen',global_scope=True,all_planned_areas_coexist=True)
            programs['f'+str(len(programs))]=p
    except Exception as exc:
        failure('surrounding_scope',exc,stage='scope_or_assembly')
    active={p['hypothesis']['id'] for p in programs.values()}
    save(comp/'shared/joint_domain.json',dict(case=case,hypotheses=[h for h in hs if h['id'] in active],
        proposed_hypotheses=hs,programs=programs,unsupported_hypotheses=[h['id'] for h in hs if h['id'] not in active],
        rejections=rejected,execution_failures=failures,conditional_scope='Designed hypotheses, not exhaustive invisible-world posterior'))
    save(dest/'hypotheses.json',hs);save(dest/'programs.json',programs)
    result=dict(status='failed' if failures else 'empty_candidate_domain' if not programs else 'complete_with_warnings' if rejected else 'complete',
        programs=len(programs),rejections=rejected,execution_failures=failures,LM_probability_table_calls=0,
        scope_complete_programs=sum(p['scope_complete_within_hypothesis'] for p in programs.values()))
    save(dest/'summary.json',result); return result


def scope_page(root):
    cards=[]
    for scene in check(root)['scenes']:
        path=root/'scenes'/scene['id']/'completion/shared/joint_domain.json'
        if not path.exists(): continue
        d=read(path); cards.append('<article><h2>'+escape(scene['id'])+'</h2>')
        for fid,p in d['programs'].items():
            h=p['hypothesis'];cards.append('<h3>'+escape(fid+': '+h['description'])+'</h3>')
            cards.append('<p>房间假设：'+escape(h['room_kind'])+'；仍未知：'+escape(h['unmodelled_space'])+'</p>')
            cards.append('<p>可见漏检提示（未重建）：'+escape(', '.join(h['visible_unmodelled']) or '未提出；不代表无漏检')+'</p>')
            cards.append('<table><tr><th>共存区域</th><th>空间假设</th><th>实例化状态</th><th>原因</th></tr>')
            for a in p['scope_ledger']:
                cards.append('<tr>'+''.join('<td>'+escape(str(a[k]))+'</td>' for k in ('configuration','region','status','reason'))+'</tr>')
                if a['uninstantiated_inventory']:
                    cards.append('<tr><td colspan="4">未能实例化的原库存（不解释为空）：'+escape(json.dumps(a['uninstantiated_inventory'],ensure_ascii=False))+'</td></tr>')
            if not p['scope_ledger']: cards.append('<tr><td colspan="4">未提出隐藏区域；周围组织仍未知，不等于空房间。</td></tr>')
            cards.append('</table>')
        cards.append('</article>')
    (root/'scope.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:24px}h2{overflow-wrap:anywhere}article{border:1px solid #ccc;padding:16px;margin:18px 0}td,th{border:1px solid #ccc;padding:8px}</style><h1>整室假设 → 共存区域 → 对象实例化</h1><p><a href="candidates.html">完整库存和宿主</a> · <a href="index.html">三维结果</a></p><p>区域来自模型假设；没有必出类别、方向配额或完整性真值。</p>'+''.join(cards))
    candidate=root/'candidates.html'
    if candidate.exists():
        candidate.write_text(candidate.read_text().replace('显式空增量假设。',
            '当前没有可实例化的对象；显式空区域与未解决区域见 <a href="scope.html">区域记录</a>。'))


def phase(root,name):
    suite=check(root)
    if name=='prepare-scenes': return old.phase(root,name)
    local=None;records=[]
    def client(comp):
        nonlocal local
        if local is None: local=provider(context_check(comp/'context'),'qwen')
        return local
    for scene in suite['scenes']:
        sid=scene['id'];path=old.marker(root,sid,name);comp=root/'scenes'/sid/'completion'
        if path.exists(): records.append(read(path));continue
        row=dict(scene=sid,phase=name,backend='qwen',status='failed')
        try:
            if name=='propose':
                result=propose_scene(comp,lambda:client(comp))
                files=list((comp/'proposals/qwen').rglob('*'))+[comp/'shared/joint_domain.json']
                save(root/'simple_bindings'/(sid+'.json'),{str(p):sha(p) for p in files if p.is_file()})
                row.update(status=result['status'],result=result)
            elif name=='worlds':
                d=read(comp/'shared/joint_domain.json')
                if not d['programs']: row.update(status='blocked_dependency',reason='no_accepted_scope_program')
                else:
                    result=_run(client(comp),comp/'end_to_end','qwen',context_check(comp/'end_to_end'),model,geometry)
                    row.update(status='failed' if result['status']=='failed' else 'partial' if result['failures'] else 'complete',result=result)
            else: raise ValueError('Unknown phase')
        except Exception as exc: row.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        save(path,row);records.append(row);print(json.dumps(dict(scene=sid,phase=name,status=row['status'])),flush=True)
    check(root)
    if name=='propose': old.candidate_report(root);scope_page(root)
    return dict(status='recorded',records=records)


def report(root):
    check(root);result=old.report(root);scopes=[]
    for row in result['records']:
        comp=root/'scenes'/row['scene']/'completion/end_to_end/qwen/completion'
        path=comp/('world_%02d'%row['world'])/'semantic_world.json'
        sem=read(path) if path.exists() else {}
        row.update(scope_complete_within_hypothesis=sem.get('scope_complete_within_hypothesis',False),
            unresolved_areas=sem.get('unresolved_areas',[]),program=sem.get('structural_state',{}).get('F'),
            geometry_variant=sem.get('geometry_variant',False),task_semantic_quality_verified=False)
        if not row['scope_complete_within_hypothesis']: scopes.append(dict(scene=row['scene'],world=row['world']))
    result.update(model=model.MODEL_VARIANT,scope_incomplete_worlds=scopes,
        export_policy='whole_program_coverage_then_geometry_variants_not_IID',
        distinct_exported_programs_by_scene={s['id']:len({r['program'] for r in result['records'] if r['scene']==s['id'] and r['exported']}) for s in check(root)['scenes']})
    if scopes and not result['execution_errors']: result['status']='semantic_scope_incomplete'
    save(root/'results.json',result);save(root/'report/results.json',result);scope_page(root)
    page=root/'report/index.html';text=page.read_text().replace('<h1>','<p><a href="../scope.html">整室假设、区域兑现与未知空间</a></p><h1>',1)
    text=text.replace('各世界独立展示；','优先覆盖每份完整库存方案；额外输出可能只是几何变体。各世界独立展示；')
    page.write_text(text);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','check','prepare-scenes','propose','worlds','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--observations-from',type=Path);p.add_argument('--worlds',type=int,default=2)
    a=p.parse_args();root=a.root.resolve()
    try:
        if a.action=='prepare':
            if a.observations_from is None:p.error('--observations-from required')
            answer=prepare(root,a.observations_from,a.worlds)
        elif a.action=='check':check(root);answer=dict(status='verified_no_model_calls')
        elif a.action=='report':answer=report(root)
        else:answer=phase(root,a.action)
        print(json.dumps({k:v for k,v in answer.items() if k not in ('records','execution_errors','warnings')},ensure_ascii=False),flush=True)
        if a.action=='report':return 2 if answer['execution_errors'] else 1 if answer['uncompleted_worlds'] or answer['scope_incomplete_worlds'] else 0
        return 0
    except Exception as exc:traceback.print_exc();p.exit(2,type(exc).__name__+': '+str(exc)+'\n')


if __name__=='__main__':sys.exit(main())
