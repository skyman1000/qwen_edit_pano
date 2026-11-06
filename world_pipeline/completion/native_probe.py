"""Official-layout diagnostic; no Qwen observations or camera registration."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np

from .geometry import official_stats, read, write

ROOT = Path('qwen_edit_pano')
REPO = ROOT / 'third_party/M3DLayout-code'
BUNDLE = ROOT / 'data/m3dlayout_native_probe_v1'


def description(scene):
    required, other = [], []
    for group in scene['description'].values():
        for key, sentences in group.items():
            (required if key in ('room_type', 'summary') else other).extend(sentences)
    # Fixed subset of released descriptions, shared by both arms; no rewriting.
    return ' '.join(required + other[:3])


def ordered_objects(scene):
    objects = scene['objects']
    if not objects:
        return []
    xyz = np.round(np.asarray([o['location'] for o in objects], dtype=float).reshape(-1, 3) / .5).astype(int)
    indices = np.lexsort((xyz[:, 1], xyz[:, 2], xyz[:, 0]))
    return [objects[i] for i in indices]


def transform(values, key, stats, inverse=False):
    lo, hi = np.asarray(stats['bounds_' + key], dtype=np.float32).reshape(2, -1)
    scale, offset = (hi - lo) / 2 + 1e-8, (hi + lo) / 2
    return values * scale + offset if inverse else (values - offset) / scale


def encode(objects, stats):
    classes = stats['class_labels']
    encoded = {'class_labels': np.eye(len(classes), dtype=np.float32)[
        [classes.index(o['category']) for o in objects]]}
    for key, field in [('translations', 'location'), ('sizes', 'size'), ('angles', 'rotation')]:
        values = np.asarray([o[field] for o in objects], dtype=np.float32).reshape(len(objects), -1)
        encoded[key] = transform(values, key, stats)
    return encoded


def plot(path, objects, prefix=0, reference=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(10, 9))
    for i, obj in enumerate(objects):
        x, _, z = obj['location']
        sx, _, sz = obj['size']  # Released native size = half extents.
        angle = obj['rotation']
        c, s = np.cos(angle), np.sin(angle)
        corners = np.array([[-sx, -sz], [sx, -sz], [sx, sz], [-sx, sz]])
        corners = corners @ np.array([[c, s], [-s, c]]).T + [x, z]
        color = 'tab:blue' if i < prefix else ('gray' if reference else 'tab:orange')
        ax.add_patch(Polygon(corners, fill=False, edgecolor=color))
        ax.text(x, z, obj['category'], fontsize=7, color=color)
    ax.relim(); ax.autoscale_view(); ax.margins(.15); ax.set_aspect('equal')
    ax.set_xlabel('scene x (m)'); ax.set_ylabel('scene z (m)')
    ax.set_title('Native OBB: blue = input prefix; gray = reference; orange = generated')
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def prepare(args):
    args.bundle.mkdir(parents=True, exist_ok=True)
    stats = official_stats(args.repo)
    classes = set(stats['class_labels']) - {'start', 'end'}
    audit, samples, train_categories = {}, [], set()
    train_min = {k: np.full(w, np.inf) for k, w in [('translations', 3), ('sizes', 3), ('angles', 1)]}
    train_max = {k: -v for k, v in train_min.items()}
    with zipfile.ZipFile(args.archive) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.json'):
                continue
            data = json.loads(archive.read(name))
            scenes = data['scenes']
            audit[name] = dict(scenes=len(scenes), unsupported_categories=sorted(set(data['categories']) - classes))
            if '_train' in name:
                train_categories.update(data['categories'])
                objects = [o for scene in scenes if len(scene['objects']) >= 2 for o in scene['objects']]
                for key, field in [('translations', 'location'), ('sizes', 'size'), ('angles', 'rotation')]:
                    values = np.asarray([o[field] for o in objects]).reshape(len(objects), -1)
                    train_min[key] = np.minimum(train_min[key], values.min(0))
                    train_max[key] = np.maximum(train_max[key], values.max(0))
            if '_val' not in name:
                continue
            eligible = [s for s in scenes if 4 <= len(s['objects']) <= 120
                        and all(o['category'] in classes for o in s['objects']) and description(s).strip()]
            eligible.sort(key=lambda s: s['scene_id'])
            random.Random(0).shuffle(eligible)
            audit[name]['eligible_scenes'] = len(eligible)
            source = Path(name).stem.replace('_val_merged', '').replace('_val', '')
            for scene in eligible[:2]:
                objects = ordered_objects(scene)
                sample = dict(source=source, scene_id=scene['scene_id'], text=description(scene),
                              objects=objects, prefix_count=len(objects)//2)
                samples.append(sample)
                directory = args.bundle / source / scene['scene_id']
                directory.mkdir(parents=True, exist_ok=True)
                write(directory/'reference.json', sample)
                plot(directory/'reference.png', objects, sample['prefix_count'], reference=True)
    bounds = {k: np.concatenate([train_min[k], train_max[k]]).tolist() for k in train_min}
    write(args.bundle/'audit.json', dict(
        archive=str(args.archive), archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        files=audit, training_category_count=len(train_categories), demo_object_category_count=len(classes),
        train_only_categories=sorted(train_categories-classes), demo_only_categories=sorted(classes-train_categories),
        released_train_bounds=bounds, demo_stats=stats,
        bounds_match={k: bool(np.allclose(v, stats['bounds_'+k], atol=1e-6, rtol=0)) for k, v in bounds.items()},
        limitation='Released data and demo vocabulary/statistics may differ. Keep demo mapping for existing weights; weight-training metadata not independently established.',
        native_size='half extents, following official _get_2d_polygon; native numbers passed unchanged to normalization',
        selection='two per validation source; 4..120 objects; supported categories; nonempty text; sorted IDs then shuffle seed 0'))
    write(args.bundle/'bundle.json', dict(stats=stats, samples=samples))
    for s in samples:
        print(f"{s['source']}/{s['scene_id']}: total={len(s['objects'])}, prefix={s['prefix_count']}")
    print(f"Prepared {len(samples)} scenes; audit: {args.bundle/'audit.json'}")


def run(args):
    import torch
    from .complete import load_pretrained
    bundle = read(args.bundle/'bundle.json')
    network, stats = load_pretrained(args.repo, args.weights, args.device)
    if stats != bundle['stats']:
        raise ValueError('Demo statistics changed since prepare; prepare the bundle again.')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'config.json', dict(bundle=str(args.bundle.resolve()), weights=str(args.weights.resolve()),
        seeds=args.seeds, stats=stats, text_condition=True, room_mask='ones_64x64_like_official_demo',
        sampler='unchanged_official', frame='native_scene_not_camera', max_decode_total=120,
        limitation='Native-condition compatibility diagnostic, not image-conditioned completion performance'))
    rows = []
    for sample in bundle['samples']:
        for mode in ('text', 'prefix'):
            n = sample['prefix_count'] if mode == 'prefix' else 0
            context = encode(sample['objects'][:n], stats) if n else None
            for seed in args.seeds:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                directory = args.output/sample['source']/sample['scene_id']/mode/f'seed{seed:03d}'
                directory.mkdir(parents=True, exist_ok=True)
                room = torch.ones(1, 1, 64, 64, device=args.device)
                with torch.inference_mode():
                    if n:
                        boxes = {k: torch.as_tensor(v[None], device=args.device) for k, v in context.items()}
                        result = network.complete_scene(boxes, room, max_boxes=120-n, device=args.device, text=sample['text'])
                    else:
                        result = network.generate_boxes(room, max_boxes=120, device=args.device, text=sample['text'])
                raw = {k: v.detach().cpu().numpy()[0] for k, v in result.items()}
                write(directory/'raw_normalized.json', {k: v.tolist() for k, v in raw.items()})
                preserved = all(np.array_equal(raw[k][1:1+n], v) for k, v in context.items()) if n else None
                decoded = {k: transform(raw[k], k, stats, inverse=True) for k in ('translations', 'sizes', 'angles')}
                objects, ended, unexpected_start = [], False, 0
                for i in range(1, len(raw['class_labels'])):
                    category = stats['class_labels'][int(raw['class_labels'][i].argmax())]
                    if category == 'end':
                        ended = True
                        break
                    if category == 'start':
                        unexpected_start += 1
                        continue
                    objects.append(dict(category=category, location=decoded['translations'][i].tolist(),
                        size=decoded['sizes'][i].tolist(), rotation=float(decoded['angles'][i, 0])))
                write(directory/'layout_scene.json', dict(frame='native_scene_y_up', size_convention='half_extents',
                    text=sample['text'], prefix_count=n, objects=objects))
                plot(directory/'layout_topdown.png', objects, n)
                row = dict(source=sample['source'], scene_id=sample['scene_id'], mode=mode, seed=seed,
                    reference_count=len(sample['objects']), prefix_count=n, generated=len(objects)-n,
                    ended=ended, prefix_preserved=preserved, unexpected_start=unexpected_start)
                write(directory/'summary.json', dict(**row, category_counts=dict(Counter(o['category'] for o in objects))))
                rows.append(row)
                with (args.output/'summary.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
                print(f"{sample['source']}/{sample['scene_id']} {mode} seed={seed}: new={len(objects)-n}, END={ended}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--archive', type=Path, default=ROOT/'data/m3dlayout_official/layout_dataset/M3DLayout_json.zip')
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--weights', type=Path, default=REPO/'weights/autoregressive_59000.pth')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/m3dlayout_native_probe_run001')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
