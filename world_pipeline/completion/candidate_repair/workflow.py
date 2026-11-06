"""Repair proposal contracts/supply; original v13 mechanisms and geometry."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback

from ...common import ROOT, read, sha
from .. import organized_local_world as base, v13_workflow as pinned
from ..assembly_experiment import check as context_check, stage_seed
from ..fresh_pipeline import provider
from . import contract
from .requests import Requests, save

KIND = 'typed_candidates_fixed_local_budget_original_v13_v1'
DEFAULT_BASELINE = ROOT / 'qwen_edit_pano/outputs/v13_original_recovery_001'


def sources():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(pinned.__file__).resolve(),
        ROOT / 'qwen_edit_pano/scripts/candidate_world.sh']
    return {str(p.resolve()): sha(p) for p in paths}


def prepare(root, source, rounds=2, worlds=4, draws=4, seed=3407, baseline=DEFAULT_BASELINE):
    pinned.verify()
    if rounds not in (1, 2): raise ValueError('Fixed proposal rounds must be 1 or 2')
    if os.environ.get('CAUPANO_PROGRAM_CATALOG') or os.environ.get('CAUPANO_LAYOUT_RGB', '0') != '0':
        raise ValueError('Use fresh local proposals and original text-only numeric protocol')
    source = Path(source).resolve(); prior = {}; bindings = {}
    # Baseline contributes comparison statistics ONLY, never model prompts or
    # candidates. Its observation must match the source exactly.
    for scene in read(source / 'suite.json')['scenes']:
        sid = scene['id']; obs, _ = base.observed_source(source, scene)
        old = Path(baseline) / 'scenes' / sid if baseline is not None else None
        paths = [old / 'observation' / rel for rel in ('input.png', 'qwen/observer3d/G_obs.json')] if old else []
        domain_path = old / 'completion/shared/domain.json' if old else None
        comparable = bool(paths and all(p.is_file() for p in paths) and domain_path.is_file() and
                          all(sha(obs / rel) == sha(p) for rel, p in zip(('input.png', 'qwen/observer3d/G_obs.json'), paths)))
        if not comparable:
            prior[sid] = dict(comparable=False, reason='No same-observation baseline available; generation does not depend on a baseline')
            continue
        for p in paths: bindings[str(p.resolve())] = sha(p)
        d = read(domain_path); bindings[str(domain_path.resolve())] = sha(domain_path)
        prior[sid] = dict(comparable=True, roles=list(d['slots']), programs=len(d['programs'])-1,
            candidate_instances=len(d['base_model']['component_registry']),
            maximum_reachable_delta=sum(max(sum(len(g['objects']) for g in d['programs'][f]['groups']) for f in ids[1:]) for ids in d['slots'].values()))
    answer = base.prepare(root, source, worlds, draws, seed)
    save(root / 'candidate_repair.json', dict(kind=KIND, rounds=rounds, entry_sha256=sources(),
        baseline_statistics=prior, baseline_bindings=bindings, API_used=False, observer_rerun=False,
        changed='proposal_wire_contract_and_fixed_local_candidate_supply_only',
        original_HAV_and_geometry=True, baseline_candidates_used_in_model=False))
    return dict(answer, proposal_rounds=rounds)


def check(root):
    pinned.verify(); suite = base.check_run(root); meta = read(root / 'candidate_repair.json')
    if meta.get('kind') != KIND or meta['entry_sha256'] != sources():
        raise ValueError('Entry source changed or wrong run; preserve it and use a new output directory')
    for p, h in meta['baseline_bindings'].items():
        if sha(p) != h: raise ValueError('Comparison baseline changed: ' + p)
    for p in (root / 'candidate_bindings').glob('*.json'):
        for filename, digest in read(p).items():
            if sha(filename) != digest: raise ValueError('Consumed proposals changed: ' + filename)
    return suite, meta


def round_proposals(comp, index, client, previous):
    out = comp / 'proposal_contract_rounds' / f'round_{index:02d}'
    if (out / 'summary.json').exists(): return read(out / 'hypotheses.json'), read(out / 'programs.json'), read(out / 'summary.json')
    case = read(comp / 'shared/case.json'); run = read(comp / 'run.json')
    requests = Requests(out / 'calls', stage_seed(run['seed'], comp.parent.name + ':round:' + str(index)),
                        comp / 'shared/observation_board.png', client)
    programs = {}; hypotheses = []; errors = []; empty = []
    try:
        grounding = read(comp / 'context/qwen/observer2d/grounding.json')
        hypotheses = requests.ask('hypotheses', contract.hypotheses_prompt(case, grounding, previous),
                                  contract.hypotheses_answer, 2048)
        for h in hypotheses:
            for channel in ('surrounding', 'belongings'):
                name = 'furniture_' + h['id'] + '_' + channel
                try:
                    variants = requests.ask(name, contract.furniture_prompt(case, h, channel, previous),
                                             lambda v, h=h: contract.furniture_answer(v, case, h))
                except Exception as exc:
                    errors.append(dict(stage=name, error=type(exc).__name__ + ': ' + str(exc))); continue
                for p in variants:
                    if not any(g['objects'] for g in p['groups']): empty.append(p); continue
                    p['proposal_source'] = dict(backend='qwen', channel=channel, round=index, typed_contract=True)
                    programs['f' + str(len(programs))] = p
    except Exception as exc:
        errors.append(dict(stage='hypotheses', error=type(exc).__name__ + ': ' + str(exc)))
    summary = dict(status='partial' if errors else 'complete', round=index, programs=len(programs), failed_queries=errors,
                   empty_alternatives=len(empty), candidate_instances=sum(len(g['objects']) for p in programs.values() for g in p['groups']))
    save(out / 'hypotheses.json', hypotheses); save(out / 'programs.json', programs)
    save(out / 'empty_programs.json', empty); save(out / 'summary.json', summary)
    save(out / 'coverage.json', contract.coverage(programs))
    return hypotheses, programs, summary


def propose(root):
    suite, meta = check(root); local = None; records = []
    def client(comp):
        nonlocal local
        if local is None: local = provider(context_check(comp / 'context'), 'qwen')
        return local
    for scene in suite['scenes']:
        sid = scene['id']; marker = root / 'phase_records' / (sid + '_propose_qwen.json')
        if marker.exists(): records.append(read(marker)); continue
        comp = root / 'scenes' / sid / 'completion'
        try:
            rounds = []; summaries = []; previous = None
            for index in range(meta['rounds']):
                hs, ps, summary = round_proposals(comp, index, lambda: client(comp), previous)
                rounds.append((hs, ps)); summaries.append(summary)
                if index == 0:
                    previous = contract.coverage(ps)
                    previous.pop('complete_program_summaries')
                    # The short aggregate carries coverage, not a second copy
                    # of every object/reason or a failed long model response.
            hs, ps = pinned.merge_rounds(rounds); dest = comp / 'proposals/qwen'
            errors = [dict(f, round=s['round']) for s in summaries for f in s['failed_queries']]
            result = dict(status='partial' if errors else 'complete', backend='qwen', programs=len(ps),
                candidate_instances=sum(len(g['objects']) for p in ps.values() for g in p['groups']),
                failed_queries=errors, round_summaries=summaries, model_id=suite['local_model'],
                API_used=False, original_v13_executor=True, observer_rerun=False)
            save(dest / 'hypotheses.json', hs); save(dest / 'programs.json', ps); save(dest / 'summary.json', result)
            save(dest / 'coverage.json', contract.coverage(ps))
            paths = list(dest.glob('*.json')) + list((comp / 'proposal_contract_rounds').glob('round_*/*.json'))
            save(root / 'candidate_bindings' / (sid + '.json'), {str(p): sha(p) for p in paths})
            row = dict(scene=sid, phase='propose', backend='qwen', status=result['status'], result=result)
            if not ps:
                for phase in ('pool', 'mechanisms', 'worlds'):
                    save(root / 'phase_records' / (sid + '_' + phase + '_qwen.json'), dict(scene=sid,
                        phase=phase, backend='qwen', status='blocked_dependency', reason='no_nonempty_candidate_domain', model_called=False))
                save(comp / 'end_to_end/qwen/completion/summary.json',
                     dict(status='empty_candidate_domain_no_completion', worlds=[], failures=[]))
        except Exception as exc:
            row = dict(scene=sid, phase='propose', backend='qwen', status='failed',
                       error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        save(marker, row); records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    check(root)
    return dict(status='phase_recorded', records=records)


def candidate_report(root):
    suite, meta = check(root); records = []; cards = []
    for s in suite['scenes']:
        sid = s['id']; comp = root / 'scenes' / sid / 'completion'; path = comp / 'proposals/qwen/programs.json'
        programs = read(path) if path.exists() else {}; cov = contract.coverage(programs)
        domain = comp / 'shared/domain.json'; budget = comp / 'call_budget.json'
        summary = read(comp / 'proposals/qwen/summary.json') if path.exists() else {}
        row = dict(scene=sid, current=cov, baseline=meta['baseline_statistics'][sid],
                   pooled=domain.exists(), failed_queries=summary.get('failed_queries', []),
                   call_budget=read(budget) if budget.exists() else None)
        records.append(row)
        cards.append('<article><h2>'+escape(sid)+'</h2><p>基线功能：'+escape(', '.join(row['baseline'].get('roles', [])) or '无同观测对照')+
            '；本轮功能：'+escape(', '.join(cov['roles']))+'；基线/本轮可达新增上限：'+str(row['baseline'].get('maximum_reachable_delta', '不适用'))+
            ' / '+str(cov['maximum_reachable_delta'])+'（容量诊断，不是生成目标）。</p>')
        cards.append('<p>提案异常：'+str(len(row['failed_queries']))+'；重复集合待审：'+str(len(cov['repeated_instances_for_review']))+'</p>')
        if row['failed_queries']: cards.append('<pre>'+escape(json.dumps(row['failed_queries'], ensure_ascii=False, indent=2))+'</pre>')
        for fid, program in programs.items():
            cards.append('<details><summary>'+escape(fid+': '+program['description'])+'</summary>')
            for g in program['groups']:
                cards.append('<h3>'+escape(g['function']+': '+g['reason'])+'</h3><table><tr><th>实例</th><th>类别</th><th>依赖宿主 / 安装</th><th>假设理由</th></tr>')
                for o in g['objects']:
                    cards.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in
                        (o['key'], o['category'], str(o['attachment']['host'])+' / '+o['attachment']['kind'], o['distinct_instance_reason']))+'</tr>')
                cards.append('</table><pre>'+escape(json.dumps(g.get('functional_links', []), ensure_ascii=False, indent=2))+'</pre>')
            cards.append('</details>')
        cards.append('</article>')
    errors = [dict(scene=r['scene'], **f) for r in records for f in r['failed_queries']]
    for sid in (s['id'] for s in suite['scenes']):
        for phase in ('prepare-scenes', 'propose', 'pool'):
            marker = root / 'phase_records' / (sid + '_' + phase + '_qwen.json')
            if marker.exists() and read(marker)['status'] == 'failed': errors.append(read(marker))
    missing = [r['scene'] for r in records if not r['pooled']]
    result = dict(status='candidate_execution_partial' if errors else 'candidate_domain_incomplete' if missing else 'candidate_domains_ready_needs_review',
        records=records, execution_errors=errors, scenes_without_domain=missing,
        domains_ready=sum(r['pooled'] for r in records), API_used=False, semantic_quality_verified=False)
    save(root / 'candidate_report.json', result)
    page = '<!doctype html><meta charset="utf-8"><style>body{font:16px/1.65 system-ui;margin:24px}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}article{padding:16px;border:1px solid #ccd;margin:16px 0}pre{white-space:pre-wrap}h2{overflow-wrap:anywhere}</style><h1>候选接口与覆盖对照</h1>'
    page += '<p>'+escape(result['status'])+'；候选域已建立：'+str(result['domains_ready'])+' / '+str(len(records))+'</p><p>本地Qwen固定预算提案。候选阶段展示对象与宿主依赖，不伪造未执行的三维布局。增加数量不等于质量提高。</p><p><a href="candidate_report.json">完整覆盖、重复集合与调用预算</a> · <a href="index.html">运行总览</a></p>'+''.join(cards)
    (root / 'candidates.html').write_text(page)
    if not (root / 'index.html').exists():
        (root / 'index.html').write_text('<meta charset="utf-8"><a href="candidates.html">查看候选与依赖</a>；完整三维阶段尚未运行。')
    return result


def reusable_report(root):
    render = base.render_world
    def cached(state, image, hfov, out):
        inputs = dict(state=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                      image=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(), hfov=hfov)
        receipt = out / 'candidate_render_receipt.json'
        if receipt.exists():
            old = read(receipt)
            if old['inputs'] != inputs or any(sha(out / p) != h for p, h in old['files'].items()):
                raise ValueError('Existing rendered artifacts differ from the bound state: ' + str(out))
            return read(out / 'visual_validation.json')
        result = render(state, image, hfov, out); save(out / 'visual_validation.json', result)
        save(receipt, dict(inputs=inputs, files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
        return result
    base.render_world = cached
    try: return base.report(root)
    finally: base.render_world = render


def report(root):
    suite, meta = check(root); candidates = candidate_report(root); result = reusable_report(root)
    errors = []; quality = []
    for marker in (root / 'phase_records').glob('*.json'):
        row = read(marker)
        if row['status'] == 'failed' or row.get('result', {}).get('failed_queries'):
            errors.append(row)
        if row['phase'] == 'worlds':
            for branch, outcome in row.get('result', {}).items():
                if outcome.get('error'): errors.append(outcome)
                for fail in outcome.get('failures', []):
                    folder = root / 'scenes' / row['scene'] / 'completion' / branch / 'qwen/completion' / f"world_{fail['world']:02d}"
                    num = folder / 'numeric_review.json'; candidate = folder / 'G_full_candidates.json'
                    rejected = (fail.get('error', '').startswith('ValueError: Hidden-world numeric contract failed (') and
                        fail.get('stage') == 'compile_' + str(fail['world']) and num.exists() and candidate.exists() and
                        read(num).get('geometry_contract_passed') is False and read(candidate).get('world_state_ready') is False)
                    (quality if rejected else errors).append(dict(scene=row['scene'], **fail))
    counts = dict(Counter(r['status'] for r in result['records']))
    for row in result['records']:
        if row['status'] == 'invalid_world': errors.append(row)
    missing = [s['id'] for s in suite['scenes'] if not any(r['scene']==s['id'] and r['status']=='exported_with_new_objects' for r in result['records'])]
    result.update(status='execution_partial' if errors else 'completion_incomplete' if missing else 'completed_needs_review',
        counts=counts, execution_errors=errors, layout_rejections=quality, scenes_without_nonempty_export=missing,
        model=KIND, semantic_quality_verified=False)
    save(root / 'results.json', result)
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px/1.7 system-ui;margin:28px}pre{white-space:pre-wrap}</style><h1>原v13：候选接口与供给修复</h1><p>'+
        escape(result['status'])+'；'+escape(json.dumps(counts, ensure_ascii=False))+'</p><p><a href="report/index.html">完整三维：原图投影、俯视图、六方向、ERP与交互3D</a></p><p><a href="candidates.html">候选及依赖、与原运行的覆盖对照</a> · <a href="results.json">全部执行异常和质量状态</a></p><p>数值导出仍需检查语义；空增量与候选拒绝分别保留。</p>'+
        ('<pre>'+escape(json.dumps(errors, ensure_ascii=False, indent=2))+'</pre>' if errors else ''))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'candidate-report', 'report') + base.PHASES)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--observations-from', type=Path)
    parser.add_argument('--rounds', type=int, default=2); parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--draws', type=int, default=4); args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'prepare':
        if args.observations_from is None: parser.error('--observations-from required')
        result = prepare(root, args.observations_from, args.rounds, args.worlds, args.draws)
    else:
        check(root)
        if args.action == 'check': result = dict(status='verified_no_model_calls')
        elif args.action == 'candidate-report': result = candidate_report(root)
        elif args.action == 'report': result = report(root)
        elif args.action == 'propose': result = propose(root)
        else: result = base.phase(root, args.action)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','scenes','execution_errors')}, ensure_ascii=False), flush=True)
    if args.action == 'report': return 2 if result['execution_errors'] else 1 if result['scenes_without_nonempty_export'] else 0
    if args.action == 'candidate-report': return 2 if result['execution_errors'] else 1 if result['scenes_without_domain'] else 0
    return 0


if __name__ == '__main__': sys.exit(main())
