"""Upstream qualitative plans + unchanged v13 execution, local Qwen only."""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
import traceback

from ...common import ROOT, read, write as json_write, sha, validate_new_output
from .. import organized_local_world as baseline
from .. import v13_workflow as pinned
from ..assembly_experiment import check as check_context, stage_seed
from ..fresh_pipeline import provider
from . import proposals

KIND = 'qualitative_plan_conditioned_original_v13_v1'
PHASES = ('prepare-scenes', 'plan', 'propose', 'pool', 'mechanisms', 'worlds')


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json_write(path, value)


def sources():
    files = list(Path(__file__).parent.glob('*.py'))
    files.append(ROOT / 'qwen_edit_pano/scripts/planned_v13.sh')
    return {str(p.resolve()): sha(p) for p in files}


def prepare(root, source, plans=2, worlds=2, draws=4, seed=3407):
    pinned.verify()
    if type(plans) is not int or not 1 <= plans <= 3:
        raise ValueError('plans must be 1..3 (alternative hypotheses, not required objects)')
    if type(worlds) is not int or worlds < 1 or type(draws) is not int or not 1 <= draws <= 20:
        raise ValueError('worlds must be positive; draws must be 1..20')
    if os.environ.get('CAUPANO_PROGRAM_CATALOG') or os.environ.get('CAUPANO_LAYOUT_RGB', '0') != '0':
        raise ValueError('Use fresh local candidates and original text-only numeric executor')
    source = Path(source).resolve()
    # Check all observations before creating any run.
    suite = read(source / 'suite.json')
    for scene in suite['scenes']:
        baseline.observed_source(source, scene)
    root = validate_new_output(root); root.mkdir(parents=True)
    branches = []
    for i in range(plans):
        name = f'plan_{i+1:02d}'; branch = root / 'branches' / name
        baseline.prepare(branch, source, worlds, draws, stage_seed(seed, name))
        branches.append(name)
    meta = dict(kind=KIND, branches=branches, seed=seed, plans=plans, worlds_per_plan=worlds,
        scenes=[s['id'] for s in suite['scenes']], observations_from=str(source),
        entry_sha256=sources(), API_used=False, observer_rerun=False,
        executor='pinned organized v13 H/A/V + assembly_coordinate_transport',
        scope='separate candidate domain per global hypothesis; no joint calibrated posterior over plans',
        layout_precision='original approximate box/support compiler; no new metric room bounds')
    write(root / 'planned_v13.json', meta)
    return dict(status='prepared_no_model_calls', scenes=len(meta['scenes']), plans=plans,
                worlds_per_scene=plans * worlds)


def check(root):
    pinned.verify(); meta = read(root / 'planned_v13.json')
    if meta.get('kind') != KIND or meta['entry_sha256'] != sources():
        raise ValueError('Repair entry changed or wrong run; preserve it and use a new output directory')
    for name in meta['branches']:
        baseline.check_run(root / 'branches' / name)
    # Derived plans and proposal domains are bound once consumed, too.
    for receipt in (root / 'bindings').glob('*.json'):
        for p, digest in read(receipt).items():
            if sha(p) != digest:
                raise ValueError('Consumed planning/proposal artifact changed: ' + p)
    return meta


def bind(root, name, paths):
    path = root / 'bindings' / (name + '.json')
    hashes = {str(p): sha(p) for p in paths}
    if path.exists() and read(path) != hashes:
        raise ValueError('Consumed input changed: ' + str(path))
    write(path, hashes)


def branch_report(branch):
    """Reuse the original report, with input-bound reusable view artifacts."""
    render = baseline.render_world
    def cached(state, image, hfov, output):
        binding = dict(state_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                       image_sha256=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(), hfov=hfov)
        receipt = output / 'render_binding.json'
        if receipt.exists():
            saved = read(receipt)
            if saved['inputs'] != binding or any(sha(output / p) != h for p, h in saved['assets'].items()):
                raise ValueError('Rendered view binding changed: ' + str(output))
            return read(output / 'visual_validation.json')
        result = render(state, image, hfov, output)
        write(output / 'visual_validation.json', result)
        assets = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
        write(receipt, dict(inputs=binding, assets=assets))
        return result
    baseline.render_world = cached
    try:
        return baseline.report(branch)
    finally:
        baseline.render_world = render


def execution_failures(outcome, completion):
    """The old executor also records geometry rejection as a ValueError.

    Only verified saved diagnostic candidates are quality rejections; transport,
    JSON/schema, compilation exceptions and missing exports stay execution errors.
    """
    errors = []
    for failure in outcome.get('failures', []):
        folder = completion / f"world_{failure['world']:02d}"
        candidate = folder / 'G_full_candidates.json'
        numeric = folder / 'numeric_review.json'
        classified_rejection = (failure.get('stage') == 'compile_' + str(failure['world']) and
            failure.get('error', '').startswith('ValueError: Hidden-world numeric contract failed (') and
            candidate.is_file() and numeric.is_file() and
            read(candidate).get('world_state_ready') is False and
            read(numeric).get('geometry_contract_passed') is False)
        if not classified_rejection:
            errors.append(failure)
    if outcome.get('error'):
        errors.append(dict(error=outcome['error'], failed_stage=outcome.get('failed_stage')))
    return errors


def phase(root, name):
    meta = check(root); records = []; local = None
    def client(comp):
        nonlocal local
        if local is None:
            local = provider(check_context(comp / 'context'), 'qwen')
        return local
    if name == 'plan':
        for sid in meta['scenes']:
            dest = root / 'plans' / sid; marker = dest / 'status.json'
            if marker.exists():
                records.append(read(marker)); continue
            comp = root / 'branches' / meta['branches'][0] / 'scenes' / sid / 'completion'
            try:
                case = read(comp / 'shared/case.json')
                request = proposals.Requests(dest / 'calls', stage_seed(meta['seed'], sid),
                                              comp / 'shared/observation_board.png', lambda: client(comp))
                plans = request.ask('global_plans', proposals.plan_prompt(case, meta['plans']),
                                    lambda v: proposals.plan_answer(v, case, meta['plans']))
                write(dest / 'plans.json', plans)
                bind(root, 'plans_' + sid, [dest / 'plans.json'])
                record = dict(scene=sid, phase=name, status='complete', plans=len(plans))
            except Exception as exc:
                record = dict(scene=sid, phase=name, status='failed', error=type(exc).__name__ + ': ' + str(exc),
                              traceback=traceback.format_exc())
            write(marker, record); records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    elif name == 'propose':
        for i, branch_name in enumerate(meta['branches']):
            branch = root / 'branches' / branch_name
            for sid in meta['scenes']:
                marker = branch / 'phase_records' / (sid + '_propose_qwen.json')
                if marker.exists():
                    records.append(read(marker)); continue
                comp = branch / 'scenes' / sid / 'completion'
                try:
                    plan_path = root / 'plans' / sid / 'plans.json'
                    if not plan_path.exists():
                        upstream = read(plan_path.parent / 'status.json')
                        raise proposals.RecordedFailure('Planning failed: ' + upstream.get('error', str(upstream)))
                    plan = read(plan_path)[i]
                    result = proposals.propose(comp, plan, lambda: client(comp))
                    bind(root, 'proposals_' + branch_name + '_' + sid,
                         [comp / 'proposals/qwen' / n for n in
                          ('programs.json', 'hypotheses.json', 'global_plan.json', 'summary.json',
                           'rejected_programs.json', 'empty_programs.json')])
                    record = dict(scene=sid, phase=name, backend='qwen', branch=branch_name,
                                  status=result['status'], result=result)
                    if not result['programs']:
                        # Empty candidate supply != a sampled empty hidden world.
                        # Do not run downstream mechanisms on a fabricated domain.
                        errors = result['failed_queries']
                        reason = 'proposal_execution_failure' if errors else 'no_admissible_candidate_program'
                        for downstream in ('pool', 'mechanisms', 'worlds'):
                            write(branch / 'phase_records' / (sid + '_' + downstream + '_qwen.json'),
                                  dict(scene=sid, phase=downstream, backend='qwen', status='blocked_dependency',
                                       reason=reason, model_called=False, upstream=result))
                        write(comp / 'end_to_end/qwen/completion/summary.json',
                              dict(status='empty_candidate_domain_no_completion', worlds=[], failures=[], reason=reason))
                except Exception as exc:
                    record = dict(scene=sid, phase=name, backend='qwen', branch=branch_name, status='failed',
                                  error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
                write(marker, record); records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    else:
        if name not in baseline.PHASES:
            raise ValueError('Unknown phase: ' + name)
        for branch_name in meta['branches']:
            branch = root / 'branches' / branch_name
            baseline.phase(branch, name)
            records.extend(dict(read(branch / 'phase_records' / (sid + '_' + name + '_qwen.json')),
                                branch=branch_name) for sid in meta['scenes'])
    check(root)
    return dict(status='phase_recorded', phase=name, records=records,
                note='Final report distinguishes execution errors, candidate rejection, and sampled emptiness')


def report(root):
    meta = check(root); records = []; errors = []; cards = []
    for branch_name in meta['branches']:
        branch = root / 'branches' / branch_name
        try:
            result = branch_report(branch)
            records.extend(dict(r, branch=branch_name) for r in result['records'])
        except Exception as exc:
            errors.append(dict(branch=branch_name, phase='report', error=type(exc).__name__ + ': ' + str(exc)))
        for marker in sorted((branch / 'phase_records').glob('*.json')):
            r = read(marker)
            if r['status'] == 'failed' or r.get('result', {}).get('failed_queries'):
                errors.append(dict(branch=branch_name, record=str(marker.relative_to(root)), detail=r))
            # World execution failures are distinct from G_full_candidates geometry rejections.
            if r['phase'] == 'worlds':
                for label, outcome in r.get('result', {}).items():
                    completion = branch / 'scenes' / r['scene'] / 'completion' / label / 'qwen/completion'
                    failures = execution_failures(outcome, completion)
                    if failures:
                        errors.append(dict(branch=branch_name, scene=r['scene'], phase='worlds', detail=failures))
        link = f'branches/{branch_name}/report/index.html'
        if (root / link).is_file():
            cards.append(f'<p><a href="{link}">{escape(branch_name)}：原视角、俯视图、六方向、ERP、交互 3D</a></p>')
    cards.append('<h2>各世界实际新增与可视化</h2><table><tr><th>场景 / 假设 / 世界</th><th>状态</th><th>观测 / 新增</th><th>查看</th></tr>')
    for r in records:
        branch = root / 'branches' / r['branch']
        comp = branch / 'scenes' / r['scene'] / 'completion'
        dest = branch / 'report' / r['scene'] / f"world_{r['world']:02d}"
        world = comp / 'end_to_end/qwen/completion' / f"world_{r['world']:02d}"
        plan_path = comp / 'proposals/qwen/global_plan.json'
        manifest = dict(branch=r['branch'], scene=r['scene'], world=r['world'], status=r['status'],
            global_plan=read(plan_path) if plan_path.exists() else None,
            observed_sha256=sha(branch / 'scenes' / r['scene'] / 'observation/qwen/observer3d/G_obs.json'),
            observation_preserved=r['observed_preserved'],
            artifacts={n: dict(path=os.path.relpath(world / n, dest), sha256=sha(world / n))
                       for n in ('G_full.json', 'G_full_candidates.json', 'semantic_world.json',
                                 'object_relations.json', 'host_interventions.json', 'numeric_review.json')
                       if (world / n).is_file()},
            failed_phases=r['failed_phases'])
        write(dest / 'world_manifest.json', manifest)
        label = r['scene'] + ' / ' + r['branch'] + f" / {r['world']:02d}"
        links = '<a href="' + escape(os.path.relpath(dest / 'world_manifest.json', root)) + '">假设/依赖/失败</a>'
        if (dest / 'views/index.html').is_file():
            links += ' · <a href="' + escape(os.path.relpath(dest / 'views/index.html', root)) + '">三维与原相机投影</a>'
        cards.append('<tr><td>' + escape(label) + '</td><td>' + r['status'] + '</td><td>' +
                     str(r['observed']) + ' / ' + str(r['placed_new']) + '</td><td>' + links + '</td></tr>')
    cards.append('</table>')
    for sid in meta['scenes']:
        folder = root / 'plans' / sid
        if (folder / 'status.json').exists() and read(folder / 'status.json')['status'] == 'failed':
            errors.append(read(folder / 'status.json'))
        if not (folder / 'plans.json').exists():
            continue
        cards.append('<h2>' + escape(sid) + '</h2>')
        for p in read(folder / 'plans.json'):
            comp = root / 'branches' / p['id'] / 'scenes' / sid / 'completion'
            cards.append('<article><h3>' + escape(p['id'] + ': ' + p['description']) + '</h3>')
            cards.append('<p>建筑假设（不计入新增物体）：' + escape('；'.join(p['architecture_assumptions'])) + '</p>')
            cards.append('<p>可见漏检（不冒充隐藏增量）：' + escape('；'.join(p['visible_unmodeled'])) + '</p>')
            cards.append('<table><tr><th>原相机方向</th><th>可能的隐藏区域与内容</th><th>功能</th></tr>')
            for a in p['areas']:
                cards.append('<tr><td>' + escape(a['camera_region']) + '</td><td>' + escape(a['hidden_intent']) +
                             '</td><td>' + escape(a['function']) + '</td></tr>')
            cards.append('</table><p>可能被推翻的条件：' + escape('；'.join(p['failure_conditions'])) + '</p>')
            for label, path in [('全景规划 JSON', folder / 'plans.json'),
                                ('候选及身份审查', comp / 'proposals/qwen/programs.json'),
                                ('拒绝的完整候选及证据', comp / 'proposals/qwen/rejected_programs.json'),
                                ('生成异常', comp / 'proposals/qwen/summary.json')]:
                if path.is_file():
                    cards.append('<a href="' + escape(os.path.relpath(path, root)) + '">' + label + '</a> · ')
            cards.append('</article>')
    counts = dict(Counter(r['status'] for r in records))
    for r in records:
        if r['status'] == 'invalid_world':
            errors.append(dict(branch=r['branch'], scene=r['scene'], world=r['world'], error=r['error']))
    completed = counts.get('exported_with_new_objects', 0)
    # A partial output is useful, but does not count as completing all scenes.
    per_scene = {sid: sum(r['scene'] == sid and r['status'] == 'exported_with_new_objects' for r in records)
                 for sid in meta['scenes']}
    incomplete = [sid for sid, n in per_scene.items() if n == 0]
    execution_status = 'failed_or_partial' if errors else 'complete'
    status = 'execution_partial' if errors else 'completion_incomplete' if incomplete else 'completed_needs_review'
    result = dict(status=status, execution_status=execution_status, execution_errors=errors, counts=counts,
        exported_nonempty_worlds=completed, scenes_without_nonempty_export=incomplete, records=records,
        API_used=False, observer_rerun=False, model=KIND,
        caution='Box-level compatibility only; VLM identity review is fallible; plans are hypotheses, not calibrated posterior or identified causes')
    write(root / 'results.json', result)
    page = '<!doctype html><meta charset="utf-8"><title>全景假设 + 原 v13 补全</title>'
    page += '<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:28px}article{border:1px solid #bcc;padding:16px;margin:16px 0}td,th{border:1px solid #ddd;padding:8px}table{border-collapse:collapse}h2,h3{overflow-wrap:anywhere}</style>'
    page += '<h1>全景假设 → 原 v13 对象图与布局</h1><p>' + escape(status) + '；各类世界状态：' + escape(json.dumps(counts, ensure_ascii=False)) + '</p>'
    page += '<p>每个全景假设独立建候选库；原 H/A/V 与坐标编译不变。下表是定性规划，不是假造的精确房间平面图。</p>'
    page += '<p>可见实例原样保留。可见漏检与建筑假设单列，不计为新增物体。身份审查和框级检查仍可能漏错，需要查看原相机投影。</p>'
    page += '<p><a href="results.json">完整结果、执行异常和失败链</a></p>'
    if errors:
        page += '<details open><summary>执行异常（原始原因）</summary><pre>' + escape(json.dumps(errors, ensure_ascii=False, indent=2)) + '</pre></details>'
    (root / 'index.html').write_text(page + ''.join(cards))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'check', 'report') + PHASES)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--observations-from', type=Path)
    p.add_argument('--plans', type=int, default=2); p.add_argument('--worlds', type=int, default=2)
    p.add_argument('--draws', type=int, default=4); p.add_argument('--seed', type=int, default=3407)
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare':
        if a.observations_from is None:
            p.error('--observations-from required')
        result = prepare(root, a.observations_from, a.plans, a.worlds, a.draws, a.seed)
    elif a.action == 'check':
        check(root); result = dict(status='verified_no_model_calls')
    elif a.action == 'report':
        result = report(root)
    else:
        result = phase(root, a.action)
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'execution_errors')}, ensure_ascii=False), flush=True)
    if a.action == 'report':
        return 2 if result['execution_errors'] else 1 if result['scenes_without_nonempty_export'] else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
