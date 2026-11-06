"""Prepare or review matched-front-end completion pilots; never load a model here."""
import argparse
from html import escape
import json
from pathlib import Path

from ..common import new_output, read, write, sha
from .assembly_experiment import restart


def prepare(output, source, seed=3407, worlds=3):
    out=new_output(output)
    for method in ('baseline','joint','needs'):
        restart(out/method,source,method,seed,worlds)
    write(out/'pilot.json',dict(source=str(source.resolve()),seed=seed,worlds_requested=worlds,
        methods=['baseline','joint','needs'],scope='matched_frontend_pipeline_comparison_not_causal_effect_estimation',
        models_loaded_during_prepare=False,api_called_during_prepare=False))
    return out


def report(root):
    pilot=read(root/'pilot.json');records=[];hashes={};image_hashes=set()
    for method in pilot['methods']:
        path=root/method;config=read(path/'experiment.json');image_hashes.add(sha(path/'input.png'))
        if config['seed']!=pilot['seed'] or config['worlds']!=pilot['worlds_requested']:
            raise ValueError('Mismatched seed/world capacity')
        for backend in ('api','qwen'):
            digest=sha(path/backend/'observer3d/G_obs.json')
            if backend in hashes and hashes[backend]!=digest:
                raise ValueError('G_obs differs across methods for '+backend)
            hashes[backend]=digest
            dest=path/backend/'completion';summary=read(dest/'summary.json') if (dest/'summary.json').is_file() else {}
            semantic=list(dest.glob('world_*/semantic_world.json'))
            signatures=[tuple(sorted(o['id'] for o in read(p)['objects'])) for p in semantic]
            reviews=[read(p) for p in dest.glob('world_*/numeric_review.json')]
            record=dict(method=method,backend=backend,status=summary.get('status','not_run_or_no_summary'),
                error=summary.get('error'),proposed_instances=summary.get('proposed_sample_counts'),
                semantic_samples=len(semantic),distinct_instance_sets=len(set(signatures)),
                exported=len(summary.get('worlds',[])),blocked=len(summary.get('blocked_worlds',[])),
                layout_failures=len(summary.get('failures',[])),
                exported_new_instances=[r['new_instances'] for r in summary.get('worlds',[])],
                remaining_overlap_pairs=[len(r['program_solver']['final_overlaps']) for r in reviews],
                declared_installation_unverified=[len(r['unresolved_installations']) for r in reviews],
                relative_view=str((dest/'index.html').relative_to(root)) if (dest/'index.html').is_file() else None,
                matched_G_obs_sha256=digest,semantic_quality_measured=False,causal_gain_validated=False)
            records.append(record)
    if len(image_hashes)!=1:raise ValueError('Pilot methods use different RGB images')
    data=dict(pilot=pilot,records=records,matched_observation_hashes=hashes,
        interpretation='Numeric diagnostics only. Failure/duplicate rates retained; no semantic winner inferred.')
    write(root/'comparison.json',data)
    html=['<!doctype html><meta charset="utf-8"><title>CauPano mechanism pilot</title>',
        '<style>body{font:16px sans-serif;max-width:1250px;margin:32px auto}td,th{padding:12px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
        '<h1>固定观测：普通补全 / 联合程序 / 需求实现机制</h1>',
        '<p>每个后端在三种方法中使用同一 RGB、G_obs、随机种子、新增实例上限和几何后端。候选目录结构与采样策略仍有差别。',
        '数字是执行诊断；没有测量语义优劣，也不构成因果效应估计。请打开各组的实际 3D/XZ/ERP 状态图。</p>',
        '<table><tr><th>方法</th><th>后端</th><th>状态</th><th>语义样本 / 不同实例集</th><th>导出 / 阻塞</th><th>新增实例数</th><th>可视化</th></tr>']
    for r in records:
        link='<a href="'+escape(r['relative_view'],quote=True)+'">打开物体世界</a>' if r['relative_view'] else '尚无结果'
        html.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in (r['method'],r['backend'],r['status'],
            f"{r['semantic_samples']} / {r['distinct_instance_sets']}",f"{r['exported']} / {r['blocked']}",r['exported_new_instances']))+'<td>'+link+'</td></tr>')
    html.extend(['</table><p>同种子并不保证不同模型/提示词具有同一随机输出。新增多不等于合理；重复样本保持原样。</p>',
        '<p><a href="comparison.json">完整数字与观测哈希</a></p><details><summary>原始汇总</summary><pre>',
        escape(json.dumps(data,ensure_ascii=False,indent=2)),'</pre></details>'])
    (root/'index.html').write_text(''.join(html))
    return root/'index.html'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','report']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path);parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--worlds',type=int,default=3);args=parser.parse_args()
    if args.action=='prepare':
        if args.source_root is None:parser.error('--source-root is required')
        print(prepare(args.root,args.source_root,args.seed,args.worlds))
    else:print(report(args.root))


if __name__=='__main__':main()
