"""Offline GT/Observer field audit. No inference, relabeling, or training certification."""
import argparse
import csv
from collections import Counter
from pathlib import Path

from .common import read, rows, sha, write, new_output, validate_graph
from .evaluate_observer import iou, match

FIELDS = ('category_id', 'bbox2d_visible_xyxy_norm', 'center_local_m',
          'size_aabb_local_m', 'in_primary_room')


def inspect_candidates(gt, pred, vocabulary):
    """Both directions; closest box is a review candidate, NEVER an identity claim."""
    result = []
    for direction, left, right in [('gt_to_pred', gt, pred), ('pred_to_gt', pred, gt)]:
        for obj in left:
            box = obj['bbox2d_visible_xyxy_norm']['value']
            candidates = [(iou(box, other['bbox2d_visible_xyxy_norm']['value']), other)
                          for other in right if box is not None and
                          other['bbox2d_visible_xyxy_norm']['value'] is not None]
            score, other = max(candidates, key=lambda item: item[0]) if candidates else (0., None)
            # Disjoint boxes must not be presented as a meaningful candidate.
            if score <= 0:
                other = None
            cat = obj['category_id']['value']
            othercat = other['category_id']['value'] if other else None
            reason = ('no_overlap_candidate' if other is None else
                      'low_overlap_or_instance_extent' if score < .5 else
                      'category_disagreement' if cat != othercat else 'overlap_and_category_agree')
            result.append(dict(direction=direction, object_id=obj['track_id'],
                category_id=cat, category=vocabulary.get(str(cat), 'unknown'),
                candidate_id=other['track_id'] if other else None,
                candidate_category=vocabulary.get(str(othercat), 'unknown') if other else None,
                best_iou=score, reason=reason, decision='pending'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    complete = read(args.gt/'EXPORT_COMPLETE.json')
    config = read(args.predictions/'run_config.json')
    read(args.predictions/'COMPLETE.json')
    assert sha(args.gt/'gt_manifest.jsonl') == complete['gt_manifest_sha256'], 'GT manifest changed'
    assert sha(args.gt/'observer_inputs.jsonl') == config['inputs_sha256'], 'Input mismatch'
    assert sha(args.gt/'vocabulary.json') == config['vocabulary_sha256'], 'Vocabulary mismatch'
    vocabulary = read(args.gt/'vocabulary.json')['categories']
    index = {r['id']: r for r in rows(args.gt/'gt_manifest.jsonl')}
    coverage = {stage: {key: Counter() for key in FIELDS} for stage in ('GT_obs', 'Observer')}
    cases, samples, sources = [], [], {}
    for sid in config['selected_ids']:
        record = index[sid]
        directory = args.gt/record['directory']
        for name, digest in record['gt_sha256'].items():
            assert sha(directory/(name+'.json')) == digest, f'GT changed: {sid}/{name}'
        gtgraph = validate_graph(read(directory/'G_obs.json'))
        assert gtgraph['id'] == sid and gtgraph['split'] == record['split']
        gt = gtgraph['objects']
        status_path = args.predictions/sid/'status.json'
        status = read(status_path)
        assert status['id'] == sid
        pred = []
        sources[str(status_path.resolve())] = sha(status_path)
        if status['status'] == 'parsed':
            predpath = args.predictions/sid/'G_obs.json'
            graph = validate_graph(read(predpath))
            assert graph['id'] == sid and graph['split'] == record['split']
            assert graph['vocabulary_sha256'] == gtgraph['vocabulary_sha256']
            pred = graph['objects']
            sources[str(predpath.resolve())] = sha(predpath)
        for stage, objects in [('GT_obs', gt), ('Observer', pred)]:
            for obj in objects:
                for key in FIELDS:
                    coverage[stage][key].update(total=1, valid=int(obj[key]['valid']))
        provenance = read(directory/'provenance.json')
        object_paths = [Path(p) for p in provenance['sources'] if Path(p).name == 'objects.json']
        assert len(object_paths) == 1, 'Expected one original object catalog'
        sourcepath = object_paths[0]
        assert sha(sourcepath) == provenance['sources'][str(sourcepath)], 'Source object catalog changed'
        sources[str(sourcepath)] = sha(sourcepath)
        original = {str(o['id']):o for o in read(sourcepath)['objects']}
        sample_cases = inspect_candidates(gt, pred, vocabulary) if status['status'] == 'parsed' else []
        for case in sample_cases:
            gt_id = case['object_id'] if case['direction'] == 'gt_to_pred' else case['candidate_id']
            source = original.get(gt_id, {})
            case.update(sample_id=sid, split=record['split'],
                source_house_object_id=source.get('house_object_id'),
                source_category=source.get('category'), source_catalog=str(sourcepath),
                gt_flags=';'.join(record['review_flags']))
        cases.extend(sample_cases)
        samples.append(dict(id=sid, split=record['split'], status=status['status'],
            error=status.get('error'), gt_objects=len(gt), predicted_objects=len(pred),
            strict_matches=len(match(gt,pred)), review_flags=record['review_flags'],
            gt_preview=str((directory/'review.jpg').resolve()),
            prediction_preview=str((args.predictions/sid/'boxes.jpg').resolve())))
    output = new_output(args.output)
    write(output/'field_coverage.json', coverage)
    write(output/'samples.json', samples)
    write(output/'review_cases.json', cases)
    if cases:
        with (output/'review_cases.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(cases[0]))
            writer.writeheader(); writer.writerows(cases)
    write(output/'audit_manifest.json', dict(status='REVIEW_REQUIRED_NOT_SCHEMA_FROZEN',
        gt_manifest_sha256=sha(args.gt/'gt_manifest.jsonl'),
        observer_config_sha256=sha(args.predictions/'run_config.json'), source_hashes=sources,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        notes=['Coverage is among exported objects, NOT accuracy or scene recall.',
               'Best-IoU candidates are not certified instance matches; directions may be many-to-one.',
               'All selected samples included for audit, even with unresolved GT flags; not approved scoring.',
               'Invalid predictions have no auditable objects; failures are counted separately.',
               'Source catalog is an existing derived asset, not final proof of raw annotation correctness.']))
    lines = ['# GT / Observer 字段审计', '',
        f'样本 {len(samples)}；成功解析 {sum(s["status"]=="parsed" for s in samples)}。',
        '本报告不修改类别、框或GT，不冻结正式schema，不认证训练就绪。', '',
        '| 字段 | GT有效/导出对象 | Observer有效/已解析对象 |', '|---|---:|---:|']
    for key in FIELDS:
        a,b=coverage['GT_obs'][key],coverage['Observer'][key]
        lines.append(f'| {key} | {a["valid"]}/{a["total"]} | {b["valid"]}/{b["total"]} |')
    lines += ['', '有效只表示存在且符合格式，不表示正确。room字段暂未纳入此对象字段统计。',
        '', '下一步查看review_cases.csv的category_disagreement与low_overlap_or_instance_extent；',
        '结合samples.json的图片路径、源catalog与house_object_id追溯。最佳IoU只生成候选，不自动确认同一物体。',
        '人工结论应区分模型错误、类别映射问题、实例粒度、mesh覆盖、无法确定；无法确定保持pending。',
        '', '接口提案见项目research/world_pipeline/FIELD_CONTRACT_NEXT.md。',
        '本轮之后先做源语义/实例核查和GT可见性规则修订，再冻结tensorizer；不追加Observer调参。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'AUDIT COMPLETE: {output}/REPORT.md; schema remains DRAFT')


if __name__ == '__main__':
    main()
