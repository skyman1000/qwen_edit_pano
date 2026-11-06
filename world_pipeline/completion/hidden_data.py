"""CauPano task data: all observed objects + hidden[:k] -> hidden[k] / END."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..common import ELIGIBLE, FRAME, read, rows, sha, validate_graph, write, write_rows

VERSION = 'caupano_hidden_local_aabb_v1'
GEOMETRY_BINS = 4096
CATEGORIES = sorted(ELIGIBLE)
KEYS = ('category_id', 'center_local_m', 'size_aabb_local_m')


def objects(graph):
    validate_graph(graph)
    result = []
    for obj in graph['objects']:
        if not all(obj[k]['valid'] for k in KEYS):
            raise ValueError(f"{graph['id']}/{obj['track_id']}: completion needs category and 3D geometry")
        result.append(dict(id=obj['track_id'], category=obj[KEYS[0]]['value'],
                           center=obj[KEYS[1]]['value'], size=obj[KEYS[2]]['value']))
    return result


def split_scene(graph):
    packed = objects(graph)
    observed, hidden = [], []
    for obj, raw in zip(packed, graph['objects']):
        if raw['local_evidence'] not in ('visible', 'unobserved'):
            raise ValueError(f"{graph['id']}: unknown visibility is not a hidden target")
        (observed if raw['local_evidence'] == 'visible' else hidden).append(obj)
    # This defines the hidden target order, NOT an artificial observed prefix.
    hidden.sort(key=lambda o: (o['center'][0], o['center'][2], o['center'][1], o['id']))
    return observed, hidden


def prefix(scene, k):
    hidden = scene['hidden']
    if not 0 <= k <= len(hidden):
        raise ValueError('k must include the END position')
    return scene['observed'] + hidden[:k], hidden[k] if k < len(hidden) else None


def normalize(value, bounds):
    low, high = np.asarray(bounds, dtype=np.float32)
    return 2 * (np.asarray(value, dtype=np.float32) - low) / (high - low) - 1


def denormalize(value, bounds):
    low, high = np.asarray(bounds, dtype=np.float32)
    return (np.asarray(value) + 1) * (high - low) / 2 + low


def fit_bounds(scenes):
    all_objects = [o for s in scenes for o in s['observed'] + s['hidden']]
    centers = np.array([o['center'] for o in all_objects])
    sizes = np.array([o['size'] for o in all_objects])
    lo, hi = centers.min(0), centers.max(0)
    margin = np.maximum((hi - lo) * .1, .1)
    return dict(center=[(lo - margin).tolist(), (hi + margin).tolist()],
                size=[(sizes.min(0) * .5).tolist(), (sizes.max(0) * 1.1).tolist()])


def prepare(bundle, output):
    bundle, output = Path(bundle), Path(output)
    info = read(bundle / 'BUNDLE.json')
    scenes = {'train': [], 'val': []}
    for row in rows(bundle / 'samples.jsonl'):
        graph = read(row['graph'])
        if sha(row['graph']) != row['graph_sha256']:
            raise ValueError(f"Bundle graph changed: {row['id']}")
        observed, hidden = split_scene(graph)
        if len(observed) + len(hidden) > info['max_objects']:
            raise ValueError('Graph exceeds Adapter capacity')
        scenes[row['split']].append(dict(id=row['id'], building=row['paired']['scene_id'],
            graph=row['graph'], graph_sha256=row['graph_sha256'], observed=observed, hidden=hidden))
    buildings = {s: sorted({r['building'] for r in rs}) for s, rs in scenes.items()}
    if set(buildings['train']) & set(buildings['val']):
        raise ValueError('Train/val building overlap')
    bounds = fit_bounds(scenes['train'])
    outside = {}
    for name in ('center', 'size'):
        values = [normalize(o[name], bounds[name]) for s in scenes['val'] for o in s['observed'] + s['hidden']]
        outside[name] = int((np.abs(values) > 1).any(axis=1).sum())
    meta = dict(version=VERSION, frame=FRAME, categories=CATEGORIES, bounds=bounds,
                max_objects=info['max_objects'], text_condition=False, angle=0, geometry_bins=GEOMETRY_BINS,
                size_convention='full_local_axis_aabb_lengths_metres',
                hidden_order='local_x_z_y_then_track_id', source_bundle=str(bundle.resolve()),
                source_manifest_sha256=sha(bundle / 'samples.jsonl'), buildings=buildings,
                counts={s: len(v) for s, v in scenes.items()}, val_objects_outside_train_bounds=outside)
    output.mkdir(parents=True, exist_ok=False)
    for split, records in scenes.items():
        write_rows(output / f'{split}.jsonl', records)
    write(output / 'metadata.json', meta)
    print(json.dumps({k: meta[k] for k in ('counts', 'buildings', 'val_objects_outside_train_bounds')}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=Path('qwen_edit_pano/data/oracle_supplement_epoch006_v1'))
    p.add_argument('--output', type=Path, default=Path('qwen_edit_pano/data/hidden_completion_v1'))
    a = p.parse_args()
    prepare(a.bundle, a.output)


if __name__ == '__main__':
    main()
