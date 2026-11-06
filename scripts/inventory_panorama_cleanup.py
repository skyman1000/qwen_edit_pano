"""Read-only inventory and deletion candidates; never deletes files."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'qwen_edit_pano/audit/cleanup_2026-10-20'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    def add(category,path,reason):
        if not path.exists() or path.is_symlink():return
        size=path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
        records.append(dict(category=category,path=str(path.relative_to(ROOT)),bytes=size,reason=reason))
    for p in [ROOT/'__pycache__',ROOT/'qwen_edit_pano/__pycache__',ROOT/'qwen_edit_pano/scripts/__pycache__']:
        add('A_regenerable',p,'Python bytecode only; source is retained')
    for p in (ROOT/'qwen_edit_pano/outputs/paired_smoke_val_v1').glob('**/*.png'):
        add('A_regenerable',p,'Failed old inference image; successful smoke_val_v2 exists; retain JSON diagnostics')
    training=ROOT/'qwen_edit_pano/outputs/paired_full_63637'
    for cp in sorted(training.glob('checkpoint-epoch*')):
        if not (cp/'COMPLETE.json').is_file():continue
        if cp.name not in ('checkpoint-epoch015','checkpoint-epoch025'):
            add('B_lose_epoch_resume',cp/'optimizer.bin',
                'Conditional: loses exact resume from this epoch; retain LoRA weights/configs; preserve complete epoch015 and025')
    # These are derived panels only, not their generation source directories.
    for dirname,pattern in [('analysis_scale03_comparison','*.jpg'),
                            ('comparison_epoch15_16_25_scale03_05','*.png')]:
        for p in sorted((ROOT/'qwen_edit_pano/outputs'/dirname).glob(pattern)):
            add('C_archive_derived',p,'Old derived comparison panel; rebuild from retained source PNGs; deleting breaks old gallery until rebuilt')
    # Old text-only entry points have historical dependants: archive, do not
    # label shared helpers as unused or offer to remove their dependency tree.
    for name in ['train.py','inference.py','cache_text.py','scripts/train_pano.sh','scripts/inference_pano.sh',
                 'scripts/cache_text.sh','scripts/smoke_test.sh']:
        add('D_archive_code_only',ROOT/'qwen_edit_pano'/name,
            'Legacy text-only entry; paired benchmark does not use it; historical scripts/tests do. Archive as a coherent group only after validation')
    totals={}
    for row in records:totals[row['category']]=totals.get(row['category'],0)+row['bytes']
    protected=[
        'qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch015 (entire directory)',
        'qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch025 (entire directory)',
        'All other checkpoint LoRA files/configs, training_config.json, parameter_report.json, train_log.jsonl',
        'qwen_edit_pano/outputs/comparison_epoch15_25_scale05 (includes new 8 images, source runs and current review)',
        'qwen_edit_pano/outputs/checkpoint_selection_70218 (current comparison sources)',
        'qwen_edit_pano/outputs/paired_val_epoch015_scale03, paired_val_epoch016_scale03, paired_val_epoch025_scale03',
        'qwen_edit_pano/outputs/pano_epoch_scale_v1 (historical ablation evidence and source references)',
        'qwen_edit_pano/outputs/native_vs_pano_epoch015_scale05_val925_v1 (new benchmark)',
        'qwen_edit_pano/data, qwen_edit_pano/cache/paired_full_v1, benchmark_assets',
        'qwen_pano (entire old implementation/outputs/cache)',
        'qwen_edit_pano/world_pipeline, third_party, all World/Observer/Completion outputs and their datasets/checkpoints',
        'research (including compatibility symlink research/world_pipeline)',
        'HuggingFace model snapshots and polished Arrow sources outside project (not disposable cache)',
        'Active jobs 70260/id-A and70261/id-B on GPU2 and all their assets; recheck before any future deletion',
        'paired_pipeline.py, paired_circular.py, circular.py, losses.py, common.py and other shared imports']
    result=dict(date='2026-10-20',files_deleted=0,scope='Panorama implementation and derived panels only; World work protected',
                candidates=records,category_bytes=totals,protected=protected,
                note='Logical file sizes; candidates are not a deletion authorization. B/C/D have explicit costs. No broad rm script generated.')
    (OUT/'deletion_candidates.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'deletion_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['category','path','bytes','reason']);w.writeheader();w.writerows(records)
    lines=['# Panorama 文件整理与删除候选（2026-10-20）','',
           '本轮没有删除或移动任何文件。精确路径与体积见 deletion_candidates.csv/json。当前额外任务在GPU2运行，World/Observer/Completion相关文件一律保护。', '',
           '## 先做什么','',
           '先完成native与epoch015/0.5的val925正式对比。保留选定epoch015和最终epoch025完整检查点、全部推理权重、日志、清单、缓存、原始数据、当前14张比较及其源图片。', '',
           '| 类别 | 候选体积 GiB | 代价 |','|---|---:|---|']
    costs={'A_regenerable':'字节码/旧失败图片，可重建；空间很小',
           'B_lose_epoch_resume':'其他23轮optimizer.bin；删除后对应轮不能精确续训',
           'C_archive_derived':'旧拼图可重建；删除会让旧HTML暂时失效',
           'D_archive_code_only':'旧纯文本入口；仅归档候选，不立即删代码'}
    for key,size in totals.items():lines.append(f'| {key} | {size/2**30:.4f} | {costs[key]} |')
    lines+=['','## 为什么不把旧代码全部删掉','',
            'paired_pipeline.py仍导入circular.py；losses.py复用qwen_pano.losses；训练复用qwen_pano.profiles等。文件名旧或不直接从CLI调用，不代表无依赖。text-only的train.py/inference.py可以先标为历史入口，但应连同其脚本整体归档，不能混删共享模块。', '',
            '小规模试验、旧指标报告和JSON审计通常很小；有些输出被HTML、符号链接或研究脚本引用。不要按目录名smoke/pilot/tmp就批量删除。GT源数据和世界模块暂不加入本轮删除候选。', '',
            '## 当前推荐主入口','',
            '- 训练/恢复：scripts/train_paired_pano.sh -> paired_train.py',
            '- 单组推理：scripts/paired.sh infer -> paired_inference.py',
            '- 正式无World对比：benchmark.py + scripts/benchmark_val925_generate.sh / benchmark_val925_evaluate.sh',
            '- 图像指标：DiT360/evaluation/evaluate.py（旧11项）+ benchmark_reference_metrics.py（旧三项投影协议）',
            '- 当前比较页：outputs/comparison_epoch15_25_scale05/review.html',
            '- 模型数据支持：paired_data.py / paired_pipeline.py / paired_circular.py / cache_paired.py / losses.py / common.py及其实际导入依赖', '',
            '## 保护清单','']+['- '+s for s in protected]
    (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(candidate_count=len(records),category_gib={k:round(v/2**30,4) for k,v in totals.items()},files_deleted=0),indent=2))


if __name__=='__main__':main()
