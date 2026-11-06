"""GT visibility transfer to official room layouts; no Completion training."""
import argparse
from collections import defaultdict
import csv
import html
import itertools
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import CATEGORY_IDS, SIGNS, official_stats, read, write, yaw_matrix
from .native_probe import ROOT, REPO, encode, ordered_objects, plot, transform

BUNDLE = ROOT / 'data/m3dlayout_visible_probe_v1'
INDOOR = set('abcdefghijklmnorstuvw') | {'B', 'C', 'D', 'S'}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def axis_maps():
    """World z-up -> native y-up, testing horizontal axis swaps/signs only.

    This is a restricted registration hypothesis, NOT an arbitrary pose solver.
    Unsupported room rotations fail preparation instead of being approximated.
    """
    for x, z in [(0, 1), (1, 0)]:
        for sx, sz in itertools.product([-1., 1.], repeat=2):
            matrix = np.zeros((3, 3))
            matrix[0, x], matrix[1, 2], matrix[2, z] = sx, 1., sz
            yield matrix


def register(native, source, tolerance=.02):
    """One room-wide axis map + translation, with unique object correspondence.

    Match category groups, sorted half extents and centers. No object rescaling,
    per-object translation, ICP deformation, or guessed visibility is used.
    All native objects must match; extra source objects are reported separately.
    """
    if len(native) < 3 or len(source) < len(native):
        return None, 'insufficient_source_objects'
    nc = np.asarray([o['location'] for o in native])
    sc = np.asarray([o['center_world'] for o in source])
    ns = np.sort(np.asarray([o['size'] for o in native]), axis=1)
    ss = np.sort(np.asarray([o['radii'] for o in source]), axis=1)
    cat = np.array([[CATEGORY_IDS.get(o['category'], -999) == s['category_id']
                     for s in source] for o in native])
    compatible = cat & (np.max(np.abs(ns[:, None] - ss[None]), axis=2) <= tolerance)
    if not compatible.any(1).all():
        return None, 'native_category_or_half_size_has_no_source_match'
    # Non-collinear anchors are needed to distinguish horizontal frame hypotheses.
    if np.linalg.matrix_rank(nc[:, [0, 2]] - nc[:, [0, 2]].mean(0), tol=tolerance) < 2:
        return None, 'horizontal_frame_underconstrained'
    solutions = []
    ni, si = np.where(compatible)
    for matrix in axis_maps():
        rotated = sc @ matrix.T
        offsets = np.unique(np.round(nc[ni] - rotated[si], 5), axis=0)
        found = {}
        for offset in offsets:
            dist = np.linalg.norm(nc[:, None] - (rotated + offset)[None], axis=2)
            valid = compatible & (dist <= tolerance)
            if not valid.any(1).all():
                continue
            ii, jj = linear_sum_assignment(np.where(valid, dist, 1e6))
            if not valid[ii, jj].all():
                continue
            offset = np.mean(nc - rotated[jj], axis=0)
            dist = np.linalg.norm(nc[:, None] - (rotated + offset)[None], axis=2)
            valid = compatible & (dist <= tolerance)
            # Multiple candidates for one native instance cannot transfer visibility safely.
            if not np.all(valid.sum(1) == 1) or not valid[ii, jj].all():
                continue
            errors = dist[ii, jj]
            found[tuple(jj)] = dict(world_to_native=matrix.tolist(), offset=offset.tolist(),
                source_indices=jj.tolist(), center_errors_m=errors.tolist(),
                max_center_error_m=float(errors.max()))
        solutions.extend(found.values())
    if len(solutions) != 1:
        return None, 'no_unique_room_frame' if solutions else 'no_axis_translation_match'
    return solutions[0], None


def native_corners(obj):
    return np.asarray(obj['location']) + (SIGNS * np.asarray(obj['size'])) @ yaw_matrix(obj['rotation']).T


def to_local(obj, registration, camera):
    matrix, offset = np.asarray(registration['world_to_native']), np.asarray(registration['offset'])
    world = (native_corners(obj) - offset) @ matrix
    local = (world - np.asarray(camera['origin_world'])) @ np.asarray(camera['basis_local_to_world'])
    return dict(category=obj['category'], center_local_m=local.mean(0).tolist(),
                size_aabb_local_m=np.ptp(local, axis=0).tolist(), corners_local=local.tolist())


def prepare(args):
    if (args.bundle/'bundle.json').exists():
        raise FileExistsError(f'Preparation already finished: {args.bundle}. Use a new --bundle directory to rebuild.')
    # A failed preparation may leave directories behind. Rebuild its manifests
    # from scratch; only bundle.json marks a completed preparation.
    args.bundle.mkdir(parents=True, exist_ok=True)
    stats = official_stats(args.repo)
    official = {}
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if '/mp3d_' in name and name.endswith('.json'):
                for scene in json.loads(archive.read(name))['scenes']:
                    official[scene['scene_id']] = (scene, Path(name).stem)
    old_ids = {r['id'] for r in rows(ROOT/'data/observer3d_indoor_val3_inputs.jsonl')}
    grouped = defaultdict(list)
    for row in rows(args.oracle/'samples.jsonl'):
        if row['split'] == 'val':
            grouped[row['paired']['scene_id']].append(row)
    audit, samples, inputs, selected_counts = [], [], [], {}
    for building, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda r: r['id'])
        random.Random(args.selection_seed).shuffle(candidates)
        selected_regions = set()
        for row in candidates:
            record = dict(id=row['id'], building=building)
            directory = Path(row['graph']).parent
            provenance = read(directory/'provenance.json')
            world_path = next(Path(k) for k in provenance['sources'] if k.endswith('/world_state.json'))
            world = read(world_path)
            region = world['room']['region_id']
            scene_id = f'{building}_region{region}'
            record.update(region_id=region, room_type=world['room']['type'], official_scene_id=scene_id)
            reason = None
            if row['id'] in old_ids:
                reason = 'previous_three_image_debug_set'
            elif world['room']['type'] not in INDOOR:
                reason = 'outside_declared_indoor_room_codes'
            elif not world['room']['supervision_eligible']:
                reason = 'source_room_not_eligible'
            elif region in selected_regions:
                reason = 'room_already_selected'
            elif len(selected_regions) >= args.per_building:
                reason = 'building_quota_reached'
            elif scene_id not in official:
                reason = 'official_room_missing'
            if reason:
                audit.append(dict(**record, status='excluded', reason=reason))
                continue
            scene, official_split = official[scene_id]
            record['official_split'] = official_split
            if not 4 <= len(scene['objects']) <= 120 or any(o['category'] not in stats['class_labels'] for o in scene['objects']):
                audit.append(dict(**record, status='excluded', reason='native_capacity_or_vocabulary'))
                continue
            native = ordered_objects(scene)
            source = [o for o in read(world_path.with_name('objects.json'))['objects'] if o['region_id'] == region]
            registration, reason = register(native, source)
            if reason:
                audit.append(dict(**record, status='excluded', reason=reason))
                continue
            source_ids = [source[i]['house_object_id'] for i in registration['source_indices']]
            matrix = np.asarray(registration['world_to_native'])
            offset = np.asarray(registration['offset'])
            corner_errors = []
            for obj, source_index in zip(native, registration['source_indices']):
                src = source[source_index]
                corners = np.asarray(src['center_world']) + (SIGNS*np.asarray(src['radii'])) @ np.asarray(src['axes_world'])
                transformed = corners @ matrix.T + offset
                distances = np.linalg.norm(transformed[:, None] - native_corners(obj)[None], axis=2)
                corner_errors.append(float(max(distances.min(0).max(), distances.min(1).max())))
            record['max_corner_error_m'] = max(corner_errors)
            if max(corner_errors) > .02:
                audit.append(dict(**record, status='excluded', reason='native_and_source_OBB_corners_disagree'))
                continue
            registration['corner_errors_m'] = corner_errors
            observed = read(directory/'G_obs.json')
            obs_ids = {o['gt_house_object_id'] for o in observed['objects']}
            visible = [i for i, oid in enumerate(source_ids) if oid in obs_ids]
            if not 0 < len(visible) < len(native):
                audit.append(dict(**record, status='excluded', reason='no_visible_or_no_heldout_native_objects'))
                continue
            camera = read(directory/'bridge.json')
            matched_obs = [o for o in observed['objects'] if o['gt_house_object_id'] in source_ids]
            # Independent check against stored camera-local centers (not inverse-roundtrip only).
            local_errors = [float(np.linalg.norm(np.asarray(to_local(native[source_ids.index(o['gt_house_object_id'])],
                            registration, camera)['center_local_m']) - o['center_local_m']['value'])) for o in matched_obs]
            if max(local_errors) > .02:
                audit.append(dict(**record, status='excluded', reason='stored_local_center_disagrees'))
                continue
            n = len(visible)
            contexts = dict(prefix=list(range(n)), visible=visible)
            sample = dict(**record, objects=native, contexts=contexts, registration=registration,
                source_house_object_ids=source_ids, camera=camera, local_rgb=str((directory/'local_rgb.png').resolve()),
                observed_graph=str((directory/'G_obs.json').resolve()),
                max_local_center_error_m=max(local_errors),
                observed_total=len(obs_ids), visible_native=n,
                observed_outside_native_ids=sorted(obs_ids-set(source_ids)),
                observed_outside_primary_room_ids=[o['gt_house_object_id'] for o in observed['objects']
                    if o['in_primary_room']['valid'] and not o['in_primary_room']['value']],
                common_heldout_indices=sorted(set(range(len(native)))-set(contexts['prefix'])-set(visible)))
            out = args.bundle/'samples'/row['id']
            out.mkdir(parents=True, exist_ok=True)
            write(out/'sample.json', sample)
            # Source OBBs are retained to inspect correspondence independently of native boxes.
            write(out/'source_matches.json', [dict(native_index=j, native=native[j], source=source[i])
                for j, i in enumerate(registration['source_indices'])])
            for mode, indices in contexts.items():
                remainder = [i for i in range(len(native)) if i not in indices]
                plot(out/f'{mode}_reference.png', [native[i] for i in indices+remainder], n, reference=True)
            write(out/'native_boxes_local.json', [to_local(o, registration, camera) for o in native])
            samples.append(sample)
            inputs.append(dict(id=row['id'], split='val', local_rgb=sample['local_rgb']))
            selected_regions.add(region)
            audit.append(dict(**record, status='selected', visible_native=n, native_total=len(native),
                              observed_total=len(obs_ids), max_local_center_error_m=max(local_errors)))
        selected_counts[building] = len(selected_regions)
    write(args.bundle/'audit.json', dict(selection_seed=args.selection_seed, per_building=args.per_building,
        selected_counts=selected_counts, records=audit, registration_tolerance_m=.02,
        selection='fixed shuffled order; first eligible distinct rooms; all exclusions logged; no generation-based selection',
        scope='GT visibility on official single-room inventory, not complete CauPano G_obs/G_full',
        registration='restricted horizontal signed-axis permutation + translation; all native objects uniquely matched',
        leakage_note='CauPano val rooms may occur in official M3D train split; official_split is recorded, this is not a held-out generalization benchmark'))
    (args.bundle/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    page = ['<!doctype html><meta charset="utf-8"><title>GT visibility transfer</title>',
            '<h1>GT visibility on native room inventory</h1><p>Blue = context; gray = held out. Inspect source_matches.json for source OBB correspondence.</p>']
    for sample in samples:
        sid = sample['id']
        page.append(f'<h2>{html.escape(sid)}</h2><p>Official split: {sample["official_split"]}; '
                    f'visible native {sample["visible_native"]}; all G_obs {sample["observed_total"]}; '
                    f'center error {sample["max_local_center_error_m"]:.6f} m</p>')
        page.append(f'<img width="380" src="{html.escape(sample["local_rgb"])}">')
        for mode in ['prefix', 'visible']:
            page.append(f'<img width="440" src="samples/{html.escape(sid)}/{mode}_reference.png">')
    (args.bundle/'review.html').write_text('\n'.join(page))
    write(args.bundle/'bundle.json', dict(stats=stats, samples=samples))
    print(f'Prepared {len(samples)} samples: {selected_counts}', flush=True)
    print(f'Review: {args.bundle}/review.html; exclusions: {args.bundle}/audit.json', flush=True)
    if not samples:
        raise SystemExit('No unambiguous registrations. Inspect audit.json; do not run describe/run or loosen thresholds blindly.')


def run(args):
    import torch
    from .complete import load_pretrained
    bundle = read(args.bundle/'bundle.json')
    if not bundle['samples']:
        raise ValueError('No prepared samples; inspect audit.json.')
    captions = {}
    for s in bundle['samples']:
        folder = args.captions/s['id']
        caption = (folder/'description.txt').read_text().strip()
        status = read(folder/'status.json')
        if not caption or status['reached_token_limit'] or Path(status['local_rgb']).resolve() != Path(s['local_rgb']).resolve():
            raise ValueError(f"{s['id']}: caption is missing, truncated or from a different image")
        captions[s['id']] = caption
    args.output.mkdir(parents=True, exist_ok=False)
    network, stats = load_pretrained(args.repo, args.weights, args.device)
    if stats != bundle['stats']:
        raise ValueError('Official demo stats changed after preparation.')
    write(args.output/'config.json', dict(bundle=str(args.bundle.resolve()), captions=str(args.captions.resolve()),
        weights=str(args.weights.resolve()), seeds=args.seeds, stats=stats, text_condition=True,
        room_mask='ones_64x64_like_native_probe', sampler='unchanged_official', max_total_decode=120,
        scope='native_single_room_with_GT_visibility_transfer; no ERP G_full export'))
    summaries = []
    for sample in bundle['samples']:
        text = captions[sample['id']]
        for mode, indices in sample['contexts'].items():
            context = [sample['objects'][i] for i in indices]
            encoded = encode(context, stats)
            n = len(context)
            for seed in args.seeds:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                out = args.output/sample['id']/mode/f'seed{seed:03d}'
                out.mkdir(parents=True)
                boxes = {k: torch.as_tensor(v[None], device=args.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    result = network.complete_scene(boxes, torch.ones(1, 1, 64, 64, device=args.device),
                        max_boxes=120-n, device=args.device, text=text)
                raw = {k: v.detach().cpu().numpy()[0] for k, v in result.items()}
                write(out/'raw_normalized.json', {k: v.tolist() for k, v in raw.items()})
                preserved = all(np.array_equal(raw[k][1:1+n], v) for k, v in encoded.items())
                decoded = {k: transform(raw[k], k, stats, inverse=True) for k in ['translations', 'sizes', 'angles']}
                generated, ended, starts = [], False, 0
                for i in range(n+1, len(raw['class_labels'])):
                    category = stats['class_labels'][int(raw['class_labels'][i].argmax())]
                    if category == 'end':
                        ended = True
                        break
                    if category == 'start':
                        starts += 1
                        continue
                    generated.append(dict(category=category, location=decoded['translations'][i].tolist(),
                        size=decoded['sizes'][i].tolist(), rotation=float(decoded['angles'][i, 0])))
                write(out/'layout_scene.json', dict(frame='official_native_scene', size_convention='half_extents',
                    text=text, context_indices=indices, prefix_count=n, objects=context+generated))
                write(out/'generated_local_boxes.json', dict(frame='local_view_x_right_y_image_up_z_forward_metres',
                    scope='generated_candidates_only_not_G_full', objects=[to_local(o, sample['registration'], sample['camera']) for o in generated]))
                plot(out/'layout_topdown.png', context+generated, n)
                row = dict(id=sample['id'], building=sample['building'], official_split=sample['official_split'],
                    mode=mode, seed=seed, context=n, reference_remaining=len(sample['objects'])-n,
                    common_heldout=len(sample['common_heldout_indices']), generated=len(generated),
                    ended=ended, context_preserved=preserved, unexpected_start=starts)
                write(out/'summary.json', row)
                summaries.append(row)
                with (args.output/'summary.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(summaries)
                print(f"{sample['id']} {mode} seed={seed}: {len(generated)} new, END={ended}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--oracle', type=Path, default=ROOT/'data/oracle_supplement_epoch006_v1')
    p.add_argument('--archive', type=Path, default=ROOT/'data/m3dlayout_official/layout_dataset/M3DLayout_json.zip')
    p.add_argument('--bundle', type=Path, default=BUNDLE)
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--weights', type=Path, default=REPO/'weights/autoregressive_59000.pth')
    p.add_argument('--captions', type=Path, default=ROOT/'outputs/m3dlayout_visible_probe_run001/captions')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/m3dlayout_visible_probe_run001/comparison')
    p.add_argument('--per-building', type=int, default=3)
    p.add_argument('--selection-seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    (prepare if args.action == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
