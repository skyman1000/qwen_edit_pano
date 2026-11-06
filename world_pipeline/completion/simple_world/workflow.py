"""Local-only completion with three inference stages and the original renderer."""
import argparse
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback

from ...common import ROOT, read, sha
from .. import organized_local_world as base, v13_workflow as pinned
from ..assembly_experiment import check as context_check, stage_seed
from ..assembly_world import _run
from .. import assembly_coordinate_transport as geometry
from ..candidate_repair.requests import Requests, save
from ..candidate_repair.workflow import reusable_report
from ..fresh_pipeline import provider
from . import contract, model

KIND = model.MODEL_VARIANT


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((Path(__file__).parent.parent / 'candidate_repair').glob('*.py'))
    paths += [Path(pinned.__file__), ROOT / 'qwen_edit_pano/scripts/simple_world.sh']
    return {str(p.resolve()):sha(p) for p in paths}


def prepare(root, observations, worlds=4):
    pinned.verify()
    if os.environ.get('CAUPANO_PROGRAM_CATALOG') or os.environ.get('CAUPANO_LAYOUT_RGB','0') != '0':
        raise ValueError('Unset external catalog and use CAUPANO_LAYOUT_RGB=0')
    answer = base.prepare(root, observations, worlds=worlds)
    save(root / 'simple_world.json', dict(kind=KIND, source_sha256=sources(),
        phases=['prepare-scenes','propose','worlds'], API_used=False, observer_rerun=False,
        hypothesis_initial_calls_per_scene=1, joint_program_initial_calls_per_scene='one per proposed hypothesis, at most two',
        max_schema_attempts=2, LM_probability_table_calls=0, original_geometry=True))
    return dict(answer, model=KIND, actual_phases=['prepare-scenes','propose','worlds'])


def check(root):
    pinned.verify(); suite = base.check_run(root); meta = read(root / 'simple_world.json')
    if meta['kind'] != KIND or meta['source_sha256'] != sources():
        raise ValueError('Bound implementation changed; preserve the run and start a new output directory')
    for binding in (root / 'simple_bindings').glob('*.json'):
        for name, digest in read(binding).items():
            if sha(name) != digest: raise ValueError('Consumed proposal changed: ' + name)
    return suite


def marker(root, sid, phase):
    return root / 'phase_records' / (sid + '_' + phase + '_qwen.json')


def candidate_report(root):
    cards = []
    for scene in check(root)['scenes']:
        sid = scene['id']; path = root / 'scenes' / sid / 'completion/shared/joint_domain.json'
        if not path.exists(): continue
        d = read(path); cards.append('<article><h2>'+escape(sid)+'</h2>')
        for fid, p in d['programs'].items():
            cards.append('<h3>'+escape(fid+': '+p['description'])+'</h3><p>全局假设：'+escape(p['hypothesis']['description'])+'</p>')
            cards.append('<p>以下各组在这一份完整方案中共同存在，不跨方案拼接。</p>')
            for g in p['groups']:
                cards.append('<h4>'+escape(g['function']+' / '+g.get('layout_region','unspecified')+': '+g['reason'])+'</h4>')
                cards.append('<table><tr><th>实例</th><th>类别</th><th>支撑 / 宿主</th><th>位置假设</th></tr>')
                for o in g['objects']:
                    cards.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in
                        (o['key'],o['category'],o['attachment'],o['placement_hint']))+'</tr>')
                cards.append('</table>')
            if not p['groups']: cards.append('<p>显式空增量假设。</p>')
        cards.append('<pre>'+escape(json.dumps(dict(rejections=d['rejections'],execution_failures=d['execution_failures']),ensure_ascii=False,indent=2))+'</pre></article>')
    (root / 'candidates.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:24px}h2{overflow-wrap:anywhere}td,th{padding:6px;border:1px solid #ccc}article{border:1px solid #ddd;margin:20px 0;padding:16px}pre{white-space:pre-wrap}</style><h1>全部联合世界方案及宿主依赖</h1><p>这是已提出的库存假设，尚不表示三维布局已通过。<a href="index.html">完整运行结果</a></p>'+''.join(cards))
    if not (root / 'index.html').exists():
        (root / 'index.html').write_text('<meta charset="utf-8"><a href="candidates.html">查看联合对象方案与依赖；布局阶段尚未结束</a>')


def propose_scene(comp, client):
    case = read(comp / 'shared/case.json'); run = read(comp / 'run.json')
    dest = comp / 'proposals/qwen'
    req = Requests(dest / 'calls', stage_seed(run['seed'], comp.parent.name + ':simple'),
                   comp / 'shared/observation_board.png', client)
    hypotheses = []; programs = {}; rejected = []; failures = []
    def failed_request(name, exc, **context):
        receipt = req.folder / (name + '.receipt.json')
        saved = read(receipt) if receipt.exists() else {}
        # A source/cache binding error is also a ValueError. Only a journaled
        # failed LM/schema request is a proposal rejection, never corruption or
        # an interrupted request without a receipt.
        error = saved.get('error','')
        schema = (saved.get('status') == 'failed' and error.startswith('ValueError: ') and
                  str(exc) in (error, error[len('ValueError: '):]))
        row = dict(context, error=type(exc).__name__+': '+str(exc))
        if schema: row['type'] = 'proposal_contract_rejection'
        (rejected if schema else failures).append(row)
    try:
        hypotheses = req.ask('hypotheses', contract.hypotheses_prompt(case,
            read(comp / 'context/qwen/observer2d/grounding.json')), contract.hypotheses_answer, 2048)
        for h in hypotheses:
            try:
                variants = req.ask('joint_' + h['id'], contract.furniture_prompt(case,h),
                                   lambda v,h=h:contract.furniture_answer(v,case,h))
                for p in variants:
                    p['proposal_source'] = dict(backend='qwen', joint=True, hypothesis=h['id'])
                    programs['f' + str(len(programs))] = p
            except Exception as exc:
                failed_request('joint_'+h['id'],exc,hypothesis=h['id'])
    except Exception as exc:
        failed_request('hypotheses',exc,stage='hypotheses')
    active = {p['hypothesis']['id'] for p in programs.values()}
    kept = [h for h in hypotheses if h['id'] in active]
    # Failed branches remain visible in the journal; they are not interpreted
    # as observed-only worlds, nor retried until accepted.
    domain = dict(case=case, hypotheses=kept, proposed_hypotheses=hypotheses, programs=programs,
        unsupported_hypotheses=[h['id'] for h in hypotheses if h['id'] not in active],
        conditional_scope='finite accepted proposal domain; rejected branches carry unknown mass',
        rejections=rejected, execution_failures=failures)
    save(comp / 'shared/joint_domain.json', domain)
    save(dest / 'hypotheses.json', hypotheses); save(dest / 'programs.json',programs)
    result = dict(status='failed' if failures else 'empty_candidate_domain' if not programs else
        'complete_with_warnings' if rejected else 'complete', programs=len(programs),
        rejections=rejected, execution_failures=failures,
        nonempty_programs=sum(any(g['objects'] for g in p['groups']) for p in programs.values()),
        LM_probability_table_calls=0)
    save(dest / 'summary.json', result)
    return result


def phase(root, name):
    suite = check(root)
    if name == 'prepare-scenes': return base.phase(root,name)
    local = None; records = []
    def client(comp):
        nonlocal local
        if local is None: local = provider(context_check(comp / 'context'),'qwen')
        return local
    for scene in suite['scenes']:
        sid = scene['id']; path = marker(root,sid,name)
        if path.exists(): records.append(read(path)); continue
        comp = root / 'scenes' / sid / 'completion'
        row = dict(scene=sid, phase=name, backend='qwen', status='failed')
        try:
            if name == 'propose':
                result = propose_scene(comp, lambda:client(comp))
                files = list((comp / 'proposals/qwen').rglob('*')) + [comp / 'shared/joint_domain.json']
                save(root / 'simple_bindings' / (sid+'.json'), {str(p):sha(p) for p in files if p.is_file()})
                row.update(status=result['status'], result=result)
            elif name == 'worlds':
                d = read(comp / 'shared/joint_domain.json')
                if not d['programs']:
                    row.update(status='blocked_dependency', reason='no_accepted_joint_program')
                else:
                    config = context_check(comp / 'end_to_end')
                    # Pass our small SCM to the unchanged numeric execution.
                    # There is no call to the old pool/estimate/H-A-V phases.
                    result = _run(client(comp), comp / 'end_to_end', 'qwen', config, model, geometry)
                    row.update(status='failed' if result['status']=='failed' else
                        'partial' if result['failures'] else 'complete', result=result)
                    out = comp / 'end_to_end/qwen/completion'
                    if (out / 'causal_model.json').exists():
                        m = read(out / 'causal_model.json')
                        save(out / 'mechanism_graph.json',dict(nodes=[dict(id=n['id'],parents=n['parents']) for n in m['nodes']],
                            generation=m['generation_factorization'], LM_probability_table_calls=0,
                            physical_causation_identified=False))
                        (out / 'mechanism_graph.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="180"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="18"><text x="20" y="40">Fixed O: global hypothesis Z → whole program F → E_i → S_i ← S_host</text><text x="20" y="85">F → joint camera plan R; (O,F,R) → frozen dimensions → original host layout</text><text x="20" y="130">Uniform reference within accepted proposals; no calibrated posterior or identified physical causes.</text></g></svg>')
            else: raise ValueError('Unknown phase')
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
        save(path,row); records.append(row)
        print(json.dumps(dict(scene=sid,phase=name,status=row['status']),ensure_ascii=False),flush=True)
    check(root)
    if name == 'propose': candidate_report(root)
    return dict(status='recorded',records=records)


def report(root):
    check(root); result = reusable_report(root)
    warnings = []; errors = []
    for p in (root / 'phase_records').glob('*.json'):
        row = read(p); r = row.get('result',{})
        warnings.extend(dict(scene=row['scene'],**v) for v in r.get('rejections',[]))
        errors.extend(dict(scene=row['scene'],**v) for v in r.get('execution_failures',[]))
        if row['status']=='failed' and not r.get('execution_failures'): errors.append(row)
        if row['phase']=='worlds':
            for f in r.get('failures',[]):
                folder = root / 'scenes' / row['scene'] / 'completion/end_to_end/qwen/completion' / ('world_%02d' % f['world'])
                if (folder / 'G_full_candidates.json').exists(): warnings.append(dict(scene=row['scene'],type='layout_rejection',**f))
                else: errors.append(dict(scene=row['scene'],**f))
    missing = [r for r in result['records'] if not r['exported'] or r['placed_new']==0]
    invalid = [r for r in result['records'] if r['status']=='invalid_world']
    errors.extend(invalid)
    result.update(status='execution_failed_or_partial' if errors else 'incomplete_worlds' if missing else
        'complete_with_warnings_needs_review' if warnings else 'complete_needs_review',
        model=KIND, execution_errors=errors, warnings=warnings, uncompleted_worlds=len(missing),
        semantic_quality_verified=False, LM_probability_table_calls=0,
        caveat='Assumed joint-program SCM; uniform finite reference, not a calibrated posterior. '
               'Box geometry passed does not certify semantic plausibility, real enclosure, or pixel occlusion consistency.')
    save(root / 'results.json',result)
    save(root / 'report/results.json',result)
    # Keep the real original-camera / plan / six directions / ERP / interactive
    # 3D report. Correct its inherited v13 heading rather than inventing views.
    page = root / 'report/index.html'
    text = page.read_text().replace('本地 Qwen：固定观测 → v13 完整世界补全','本地 Qwen：全局假设 → 联合对象程序 → 宿主布局')
    text = text.replace('假设的 v13 SCM','简化的假设 SCM')
    text = text.replace('<h1>', '<p><a href="../results.json">简化模型总状态与警告</a> · <a href="../candidates.html">全部联合库存方案与依赖</a></p><h1>',1)
    page.write_text(text)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','check','prepare-scenes','propose','worlds','report'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--observations-from',type=Path)
    p.add_argument('--worlds',type=int,default=4)
    a = p.parse_args(); root = a.root.resolve()
    try:
        if a.action=='prepare':
            if a.observations_from is None: p.error('prepare requires --observations-from')
            answer = prepare(root,a.observations_from,a.worlds)
        elif a.action=='check': check(root); answer = dict(status='verified_no_model_calls')
        elif a.action=='report': answer = report(root)
        else: answer = phase(root,a.action)
        shown = {k:v for k,v in answer.items() if k not in ('records','warnings','execution_errors')}
        print(json.dumps(shown,ensure_ascii=False),flush=True)
        if a.action=='report': return 2 if answer['execution_errors'] else 1 if answer['uncompleted_worlds'] else 0
        return 0
    except Exception as exc:
        traceback.print_exc(); p.exit(2,type(exc).__name__+': '+str(exc)+'\n')


if __name__=='__main__': sys.exit(main())
