"""Held-out structural diagnostics, reported per building and per seed."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..common import read, rows, write
from .hidden_data import objects


def matches(predicted, target, threshold):
    if not predicted or not target:
        return 0, 0.
    pc = np.array([o['center'] for o in predicted])[:, None]
    ps = np.array([o['size'] for o in predicted])[:, None]
    tc = np.array([o['center'] for o in target])[None]
    ts = np.array([o['size'] for o in target])[None]
    intersection = np.maximum(0, np.minimum(pc + ps / 2, tc + ts / 2) -
                              np.maximum(pc - ps / 2, tc - ts / 2)).prod(-1)
    iou = intersection / (ps.prod(-1) + ts.prod(-1) - intersection)
    same = np.array([o['category'] for o in predicted])[:, None] == np.array([o['category'] for o in target])[None]
    valid = same & (iou >= threshold)
    # First maximize number of valid matches, then IoU among those matches.
    r, c = linear_sum_assignment(-(valid * (min(len(predicted), len(target)) + 1) + iou * valid))
    keep = valid[r, c]
    distance = np.linalg.norm(pc - tc, axis=-1)
    return int(keep.sum()), float(distance[r[keep], c[keep]].sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('qwen_edit_pano/data/hidden_completion_v1'))
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--iou', type=float, default=.25)
    a = p.parse_args()
    scenes = {s['id']: s for s in rows(a.data / 'val.jsonl')}
    groups = defaultdict(list)
    for path in sorted(a.predictions.glob('*/seed*/G_full.json')):
        summary = read(path.parent / 'summary.json')
        if summary['input_kind'] != 'gt':
            raise ValueError('This evaluator isolates completion using GT observations')
        scene = scenes[summary['id']]
        observed = read(path.parent / 'G_obs.json')
        graph = read(path)
        n = len(observed['objects'])
        if graph['objects'][:n] != observed['objects']:
            raise ValueError('Observed objects changed during completion')
        predicted = objects(dict(graph, objects=graph['objects'][n:]))
        tp, distance = matches(predicted, scene['hidden'], a.iou)
        record = dict(id=scene['id'], predicted=len(predicted), target=len(scene['hidden']),
                      tp=tp, center_error_sum=distance, ended=int(summary['ended']))
        for building in (scene['building'], 'ALL'):
            groups[(building, summary['seed'])].append(record)
    if not groups:
        raise ValueError('No saved predictions found')
    report = []
    for (building, seed), values in sorted(groups.items()):
        sums = {k: sum(v[k] for v in values) for k in ('predicted', 'target', 'tp', 'center_error_sum', 'ended')}
        precision = sums['tp'] / sums['predicted'] if sums['predicted'] else None
        recall = sums['tp'] / sums['target'] if sums['target'] else None
        report.append(dict(building=building, seed=seed, samples=len(values),
            hidden_precision=precision, hidden_recall=recall,
            count_mae=float(np.mean([abs(v['predicted'] - v['target']) for v in values])),
            matched_center_mae_m=sums['center_error_sum'] / sums['tp'] if sums['tp'] else None,
            end_rate=sums['ended'] / len(values), **sums))
    write(a.predictions / 'evaluation.json', dict(iou_threshold=a.iou,
        scope='GT-observed completion; one-reference metrics, not plausibility certification', results=report))
    for r in report:
        print(r)


if __name__ == '__main__':
    main()
