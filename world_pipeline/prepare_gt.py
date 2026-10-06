"""Prepare review-only face2 GT graphs and direct CPU mesh projections.

Run from the project root with python -m research.world_pipeline.prepare_gt.
Only existing assets are read. No training, model loading, or downloads.
"""
import argparse
import copy
import csv
import html
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from qwen_edit_pano.paired_data import local_image, target_image, paired_rows
from qwen_edit_pano.pair_alignment import estimate_alignment
from .common import ROOT, SCHEMA, FRAME, ELIGIBLE, field, read, rows, sha, write, write_rows, new_output, validate_graph
from .geometry import (alignment_basis, box_geometry, visible_box, ncc, view_scores,
                       MeshScene, erp_rays, local_rays, sample_erp)

from .checkpoint_io import (atomic_json, atomic_rows, output_lock, archive_partial,
                            commit_sample, load_sample, verify_files)

OLD = ROOT / 'qwen_pano/outputs/caupano'
MAPPING = ROOT / 'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'


def candidates(pair_root, train_limit, val_limit):
    official = {}
    house_split = {}
    for split in ('train', 'val', 'test'):
        for row in rows(ROOT / f'qwen_edit_pano/data/official_building_split/{split}.full.jsonl'):
            if row['id'] in official or house_split.setdefault(row['scene_id'], split) != split:
                raise ValueError('Official split contains identity/building leakage')
            official[row['id']] = (split, row['scene_id'], row['source_view_id'])
    selected, summary = [], {}
    for split, limit in [('train', train_limit), ('val', val_limit)]:
        entries = paired_rows(pair_root/f'{split}.jsonl', split)
        groups = {}
        for row in entries:
            if official.get(row['id']) != (split, row['scene_id'], row['source_view_id']):
                raise ValueError('Paired row differs from official building split')
            path = OLD/'world_state_pilot'/row['scene_id']/row['source_view_id']/'world_state.json'
            if path.is_file():
                groups.setdefault(row['scene_id'], []).append(row)
        ordered = [groups[h][i] for i in range(max([len(g) for g in groups.values()] or [0]))
                   for h in sorted(groups) if i < len(groups[h])]
        chosen = ordered[:limit] if limit else ordered
        summary[split] = dict(paired_rows=len(entries), world_identity_matches=len(ordered), selected=len(chosen))
        selected.extend(chosen)
    if not selected or not summary['train']['selected'] or not summary['val']['selected']:
        raise ValueError('Need existing world candidates in both train and val')
    return selected, summary


def overlay(rgb, instance, objects=None, local=False):
    value = np.asarray(rgb).copy()
    edge = np.zeros(instance.shape, dtype=bool)
    edge[1:] |= instance[1:] != instance[:-1]
    edge[:, 1:] |= instance[:, 1:] != instance[:, :-1]
    edge &= instance > 0
    value[edge] = [0, 255, 100]
    image = Image.fromarray(value)
    draw = ImageDraw.Draw(image)
    h, w = instance.shape
    for obj in objects or []:
        iid = int(obj['track_id'])
        y, x = np.where(instance == iid)
        if len(x):
            draw.text((int(np.median(x)), int(np.median(y))), str(iid), fill='yellow')
        if local:
            box = obj['bbox2d_visible_xyxy_norm']['value']
            if box is not None:
                draw.rectangle((box[0]*w, box[1]*h, box[2]*w-1, box[3]*h-1), outline='yellow', width=2)
    return image


def room_fields(state, origin, basis):
    valid = state['room']['supervision_eligible']
    layout = state['layout']
    up = np.array([0., 0., 1.]) @ basis
    def plane(height, ceiling=False):
        if not valid or height is None:
            return field(None, reason='source_room_invalid_or_height_missing')
        sign = -1 if ceiling else 1
        return field([*(sign*up).tolist(), float(sign*(origin[2]-height))], method='world_horizontal_plane_to_local')
    # extent is the source coarse REGION bbox; not claimed to be exact walls.
    extent = state['room']['extent']
    lo, hi = np.array(extent[:3]), np.array(extent[3:])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = (corners-origin) @ basis
    return dict(type=field(state['room']['type'] if valid else None, source='gt_raw',
                           method='source_Matterport_region_code', reason='conflicting_room_assignment'),
                extent_aabb_local=field(dict(center=((corners.min(0)+corners.max(0))/2).tolist(),
                                             size=np.ptp(corners, axis=0).tolist()) if valid else None,
                                        method='source_region_bbox_corners', reason='conflicting_room_assignment'),
                floor_plane_local=plane(layout['floor_z']), ceiling_plane_local=plane(layout['ceiling_z'], True))


def export_one(row, output, scene, args, mapping_hash):
    sid, house, uuid = row['id'], row['scene_id'], row['source_view_id']
    world_path = OLD/'world_state_pilot'/house/uuid/'world_state.json'
    state = read(world_path)
    geo_dir = OLD/'geometry_pilot'/house/uuid
    report = read(geo_dir/'geometry_report.json')
    if (state['sample_id'] != sid or report['sample_id'] != sid or
            not all(report['checks'].values()) or report['review_flags'] or report['erp_sampling'] != 'pixel_centers'):
        raise ValueError('Source identity/geometry checks failed')
    basis_old = np.asarray(report['erp_to_world_basis_candidate'], dtype=float)
    origin = np.asarray(report['origin_world_candidate'], dtype=float)
    pose = np.asarray(state['camera']['camera_to_world'], dtype=float)
    if (not np.allclose(basis_old.T@basis_old, np.eye(3), atol=1e-6) or
            not np.allclose(pose[:3, :3], basis_old, atol=1e-6) or
            not np.allclose(pose[:3, 3], origin, atol=1e-6) or
            not np.allclose(state['camera']['position_world'], origin, atol=1e-6)):
        raise ValueError('Inconsistent source camera bases/origins')
    local = local_image(row)
    target = target_image(row)
    with Image.open(geo_dir/'rgb_erp.png') as image:
        legacy = image.convert('RGB').copy()
    # Independent registration against LEGACY RGB, never applying polished roll to its basis.
    registration = estimate_alignment(local, legacy)
    basis_change = alignment_basis(registration)
    basis = basis_old @ basis_change
    source_aligned = sample_erp(np.asarray(legacy), erp_rays(512, 1024) @ basis_change.T)
    target_small = np.asarray(target.resize((1024, 512), Image.Resampling.BICUBIC))
    residual = estimate_alignment(local, target)
    legacy_front = sample_erp(source_aligned, local_rays(512))
    target_front = sample_erp(target_small, local_rays(512))
    local_small = np.asarray(local.resize((512, 512), Image.Resampling.BICUBIC))
    scores = view_scores(source_aligned, target_small)
    flags = []
    if registration['score'] < args.min_score or registration['margin'] < args.min_margin:
        flags.append('legacy_to_face2_weak_registration')
    residual_deg = min(residual['roll_fraction'], 1-residual['roll_fraction'])*360
    if residual['mirror'] or residual_deg > 1.5 or residual['score'] < args.min_score or residual['margin'] < args.min_margin:
        flags.append('paired_target_not_in_face2_frame')
    if any(s['ncc'] is None or s['ncc'] < args.min_score for s in scores):
        flags.append('legacy_vs_polished_multiview_disagreement')
    directory = output/'samples'/sid
    directory.mkdir(parents=True)
    local.save(directory/'local_rgb.png')
    # Full-resolution target is retained only for this small selected pilot.
    target.save(directory/'gt_erp.png')
    Image.fromarray(target_front).save(directory/'gt_front.png')
    Image.fromarray(legacy_front).save(directory/'legacy_front.png')
    bridge = dict(sample_id=sid, basis_local_to_world=basis.tolist(), origin_world=origin.tolist(),
                  basis_determinant=float(np.linalg.det(basis)),
                  up_local=(np.array([0., 0., 1.])@basis).tolist(), levelled=False,
                  legacy_to_face2=registration, paired_target_alignment=row['target_alignment'],
                  paired_target_residual=residual, six_view_scores=scores,
                  local_legacy_ncc=ncc(local_small, legacy_front), local_target_ncc=ncc(local_small, target_front),
                  method='legacy_camera_basis_composed_with_independent_rgb_yaw_mirror_registration',
                  limitations='Yaw/mirror bridge and inherited candidate origin; not a full SE3 calibration or parallax correction')
    local_hit = scene.cast(origin, basis, local_rays(512))
    erp_hit = scene.cast(origin, basis, erp_rays(512, 1024))
    for name, hit in [('local', local_hit), ('erp', erp_hit)]:
        np.savez_compressed(directory/f'{name}_visibility.npz', **hit)
    old_instance = np.load(geo_dir/'instance_erp.npy', allow_pickle=False)
    old_valid = np.load(geo_dir/'instance_valid.npy', allow_pickle=False)
    old_instance = np.where(old_valid, old_instance, -1)
    reproj_local = sample_erp(old_instance, local_rays(512)@basis_change.T, nearest=True)
    common = (reproj_local > 0) & local_hit['instance_valid']
    bridge['old_mask_vs_direct_local'] = dict(common_pixels=int(common.sum()),
        agreement=float((reproj_local[common] == local_hit['instance'][common]).mean()) if common.any() else None,
        note='Diagnostic only: nearest resampling and direct rays differ at edges/small objects')
    catalog_path = world_path.parent/'objects.json'
    catalog = read(catalog_path)['objects']
    if len({o['id'] for o in catalog}) != len(catalog):
        raise ValueError('Duplicate source instance IDs')
    known_ids = {o['id'] for o in catalog}
    hit_ids = set(np.unique(local_hit['instance'])) | set(np.unique(erp_hit['instance']))
    if hit_ids - known_ids - {-1}:
        raise ValueError('Mesh instance IDs absent from source object catalog')
    lc = dict(zip(*np.unique(local_hit['instance'], return_counts=True)))
    ec = dict(zip(*np.unique(erp_hit['instance'], return_counts=True)))
    objects, excluded = [], []
    for obj in catalog:
        iid = obj['id']
        if obj['category_id'] not in ELIGIBLE or not obj['supervision_mask']['category']:
            continue
        visible_local, visible_erp = lc.get(iid, 0) >= 16, ec.get(iid, 0) >= 16
        if not visible_local and not visible_erp:
            continue
        if visible_local and not visible_erp:
            # Keep observed object for review; block certification instead of silent loss.
            flags.append(f'local_object_below_erp_threshold:{iid}')
        try:
            center, size = box_geometry(obj, origin, basis)
            if not all(obj['supervision_mask'][k] for k in ('position', 'size', 'existence')):
                raise ValueError('Invalid source supervision mask')
        except ValueError as exc:
            excluded.append(dict(instance_id=iid, reason=str(exc)))
            flags.append(f'invalid_visible_object_geometry:{iid}')
            continue
        member = obj['room_supervision_eligible'] and state['room']['supervision_eligible']
        objects.append(dict(track_id=str(iid), gt_house_object_id=obj['house_object_id'],
            category_id=field(obj['category_id'], method='pinned_mpcat40_mapping'),
            center_local_m=field(center, method='source_OBB_center_to_local'),
            size_aabb_local_m=field(size, method='source_OBB_corners_camera_AABB_full_lengths'),
            bbox2d_visible_xyxy_norm=field(visible_box(local_hit['instance']==iid) if visible_local else None,
                                           method='tight_visible_instance_mask', reason='not_locally_observed'),
            local_evidence='visible' if visible_local else 'unobserved',
            in_primary_room=field(obj['region_id']==state['room']['region_id'] if member else None,
                                  method='validated_source_room_membership', reason='room_assignment_invalid'),
            visibility=dict(local_pixels=int(lc.get(iid, 0)), erp_pixels=int(ec.get(iid, 0)))))
    objects.sort(key=lambda x: int(x['track_id']))
    if not objects:
        flags.append('no_eligible_objects')
    full = dict(schema_version=SCHEMA, id=sid, split=row['split'], stage='gt', scope='full', frame=FRAME,
                vocabulary_sha256=mapping_hash, room=room_fields(state, origin, basis), objects=objects,
                relations=[], ready_for_training=False)
    validate_graph(full)
    obs = copy.deepcopy(full)
    obs['scope'] = 'observed'
    obs['objects'] = [o for o in obs['objects'] if o['local_evidence']=='visible']
    # Complete room geometry/type are NOT visual observations, even in GT G_obs.
    obs['room'] = {k: field(None, reason='not_provided_by_local_object_visibility') for k in full['room']}
    obs['observation_strength'] = 'oracle_full_geometry_of_visible_objects_not_observer_prediction'
    hidden = copy.deepcopy(full)
    hidden['scope'] = 'hidden'
    hidden['objects'] = [o for o in hidden['objects'] if o['local_evidence']=='unobserved']
    for name, graph in [('G_full', full), ('G_obs', obs), ('G_hidden', hidden)]:
        write(directory/f'{name}.json', validate_graph(graph))
    local_overlay = overlay(local_small, local_hit['instance'], objects, True)
    erp_overlay = overlay(target_small, erp_hit['instance'], objects)
    local_overlay.save(directory/'local_overlay.png')
    erp_overlay.save(directory/'erp_overlay.png')
    sheet = Image.new('RGB', (1536, 1100), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f'{sid} | REVIEW REQUIRED | Local / target front / legacy front', fill='black')
    for i, a in enumerate((local_small, target_front, legacy_front)):
        sheet.paste(Image.fromarray(a), (512*i, 32))
    sheet.paste(local_overlay, (0, 576))
    sheet.paste(erp_overlay, (512, 576))
    draw.text((8, 550), f'Green=mesh instance edges; yellow=IDs/visible boxes; objects={len(objects)} obs={len(obs["objects"])}', fill='black')
    sheet.save(directory/'review.jpg', quality=92)
    bridge.update(review_flags=flags, excluded_objects=excluded,
                  local_instance_coverage=float(local_hit['instance_valid'].mean()),
                  erp_instance_coverage=float(erp_hit['instance_valid'].mean()),
                  status='NUMERICAL_REVIEW_FLAGS' if flags else 'READY_FOR_VISUAL_REVIEW', ready_for_training=False)
    write(directory/'bridge.json', bridge)
    source_paths = [world_path, catalog_path, geo_dir/'geometry_report.json', geo_dir/'rgb_erp.png',
                    geo_dir/'instance_erp.npy', geo_dir/'instance_valid.npy',
                    OLD/'object_link_pilot'/house/'object_links.json']
    write(directory/'provenance.json', dict(sources={str(p): sha(p) for p in source_paths},
          scene_source_manifest=f'../../scene_sources/{house}.json', paired_row=row,
          local_png_sha256=sha(directory/'local_rgb.png'), inherited_geometry_certified=False))
    manifest = dict(id=sid, split=row['split'], scene_id=house, directory=str(directory.relative_to(output)),
                    status=bridge['status'], review_flags=flags, ready_for_training=False,
                    gt_sha256={name:sha(directory/f'{name}.json') for name in ('G_obs','G_full','G_hidden')})
    observer = dict(id=sid, split=row['split'], local_rgb=str((directory/'local_rgb.png').resolve()),
                    local_sha256=sha(directory/'local_rgb.png'), width=1024, height=1024, hfov_degrees=90)
    return manifest, observer



def validate_resume(output, contract, selected):
    old = read(output/'contract.json')
    for key in contract.keys() - {'source_sha256'}:
        if old.get(key) != contract[key]:
            raise ValueError(f'Resume configuration changed: {key}; use a new output')
    # Only orchestration changed in the supported legacy migration. Numerical
    # geometry/registration/representation and source data must remain identical.
    allowed_old = 'c077b8b6eb13233b72357960521685c926fce8da9c2ec6bab6a9b74c936c3a74'
    core = {str(Path(__file__).resolve()), str(Path(__file__).with_name('geometry.py').resolve()),
            str(Path(__file__).with_name('common.py').resolve())}
    for name, expected in old['source_sha256'].items():
        path = Path(name)
        if name in core or not path.is_relative_to(Path(__file__).parent):
            if sha(path) != expected and not (path.resolve() == Path(__file__).resolve() and expected == allowed_old):
                raise ValueError(f'Resume source changed: {path}')
    signature = dict(world_root=str(OLD.resolve()), ids=[r['id'] for r in selected],
                     implementation={str(Path(__file__).resolve()): sha(Path(__file__)),
                                     str(Path(__file__).with_name('checkpoint_io.py').resolve()): sha(Path(__file__).with_name('checkpoint_io.py'))})
    marker = output/'resume_contract.json'
    if marker.exists() and read(marker) != signature:
        raise ValueError('Resume root/selection/code changed; use a new export')
    # The legacy root is recoverable from provenance even before resume metadata.
    selected_ids = {r['id'] for r in selected}
    for prov in (output/'samples').glob('*/provenance.json'):
        if prov.parent.name not in selected_ids:
            continue
        try:
            data = read(prov)
        except json.JSONDecodeError:
            if (prov.parent/'DATA_COMPLETE.json').exists():
                raise
            continue  # interrupted, uncommitted legacy sample will be archived
        expected_world = OLD/'world_state_pilot'/data['paired_row']['scene_id']/data['paired_row']['source_view_id']/'world_state.json'
        if str(expected_world) not in data['sources']:
            raise ValueError('Cannot change world-root while resuming an export')
        verify_files(Path('/'), data['sources'])
    for scene in (output/'scene_sources').glob('*.json'):
        verify_files(Path('/'), read(scene)['source_sha256'])
    # Migrate only legacy rows whose graphs, images, visibility and provenance
    # can be read and whose previously recorded hashes still match.
    manifests = output/'gt_manifest.jsonl'
    if manifests.exists():
        lines = manifests.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise ValueError('Corrupt legacy manifest before last line')
                break  # interrupted legacy write; uncommitted sample is rebuilt
            directory = output/record['directory']
            if (directory/'DATA_COMPLETE.json').exists():
                continue
            try:
                for name, expected in record['gt_sha256'].items():
                    if sha(directory/f'{name}.json') != expected:
                        raise ValueError('Legacy graph hash mismatch')
                    validate_graph(read(directory/f'{name}.json'))
                prov = read(directory/'provenance.json')
                if sha(directory/'local_rgb.png') != prov['local_png_sha256']:
                    raise ValueError('Legacy local image changed')
                bridge = read(directory/'bridge.json')
                if bridge['sample_id'] != record['id'] or bridge['review_flags'] != record['review_flags']:
                    raise ValueError('Legacy bridge mismatch')
                for name in ('local_rgb.png','gt_erp.png','gt_front.png','legacy_front.png','local_overlay.png','erp_overlay.png','review.jpg'):
                    with Image.open(directory/name) as im:
                        im.verify()
                for name in ('local_visibility.npz','erp_visibility.npz'):
                    with np.load(directory/name, allow_pickle=False) as data:
                        for key in data.files:
                            data[key]
            except (OSError, KeyError, ValueError):
                # Keep the original directory until the per-sample retry archives it.
                continue
            observer = dict(id=record['id'], split=record['split'], local_rgb=str((directory/'local_rgb.png').resolve()),
                            local_sha256=sha(directory/'local_rgb.png'), width=1024, height=1024, hfov_degrees=90)
            commit_sample(directory, record, observer)
    atomic_json(marker, signature)
    # Preserve any older completion record; the current run writes its own at end.
    if (output/'EXPORT_COMPLETE.json').exists():
        archive_partial(output/'EXPORT_COMPLETE.json')


def main():
    global OLD
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=ROOT/'qwen_edit_pano/data/paired_pilot_v1')
    p.add_argument('--output', type=Path)
    p.add_argument('--world-root', type=Path, default=OLD, help='Root containing legacy-shaped GT stages')
    p.add_argument('--resume', action='store_true', help='Validate and reuse completed samples; preserve partial directories')
    p.add_argument('--require-all-pairs', action='store_true', help='Fail if any paired identity lacks a world state')
    p.add_argument('--train-limit', type=int, default=9, help='0 uses all identity matches')
    p.add_argument('--val-limit', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--min-score', type=float, default=.65)
    p.add_argument('--min-margin', type=float, default=.03)
    p.add_argument('--preflight', action='store_true', help='Read-only dependency and selection check; no mesh rendering')
    args = p.parse_args()
    OLD = args.world_root.resolve()
    if min(args.train_limit, args.val_limit) < 0 or args.threads < 1:
        p.error('Invalid limit/threads')
    if not -1 <= args.min_score <= 1 or not 0 <= args.min_margin <= 2:
        p.error('Invalid NCC thresholds')
    if not args.preflight and args.output is None:
        p.error('--output required')
    for module in ('numpy', 'PIL', 'cv2', 'open3d', 'pyarrow'):
        importlib.import_module(module)
    import cv2
    cv2.setNumThreads(args.threads)
    selected, counts = candidates(args.pairs, args.train_limit, args.val_limit)
    print(counts, flush=True)
    if args.require_all_pairs and any(c['selected'] != c['paired_rows'] for c in counts.values()):
        raise ValueError('Incomplete world coverage: finish expand_gt_sources first; no partial full-data export allowed')
    mapping_hash = sha(MAPPING)
    vocabulary = {}
    with MAPPING.open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            cid = int(row['mpcat40index'])
            if cid in ELIGIBLE:
                if str(cid) in vocabulary and vocabulary[str(cid)] != row['mpcat40']:
                    raise ValueError('Conflicting category names in pinned TSV')
                vocabulary[str(cid)] = row['mpcat40']
    for house in {r['scene_id'] for r in selected}:
        if read(OLD/'object_link_pilot'/house/'object_links.json')['category_mapping_sha256'] != mapping_hash:
            raise ValueError('Pinned vocabulary differs from old object links')
    if args.preflight:
        print('PREFLIGHT PASSED; no pixels rendered, no files written, no GPU/model loaded.', flush=True)
        return
    if args.resume:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT/'qwen_edit_pano/data') or not (output/'contract.json').is_file():
            raise ValueError('Resume needs an existing GT export under qwen_edit_pano/data')
    else:
        output = new_output(args.output)
    with output_lock(output):
        run_export(args, output, selected, counts, mapping_hash, vocabulary)


def run_export(args, output, selected, counts, mapping_hash, vocabulary):
    (output/'scene_sources').mkdir(exist_ok=True)
    sources = {str(f):sha(f) for f in Path(__file__).parent.glob('*.py')}
    for f in [args.pairs/'train.jsonl', args.pairs/'val.jsonl', MAPPING,
              ROOT/'qwen_edit_pano/paired_data.py', ROOT/'qwen_edit_pano/pair_alignment.py',
              ROOT/'qwen_pano/caupano/world_view.py']:
        sources[str(f)] = sha(f)
    for split in ('train','val','test'):
        f = ROOT/f'qwen_edit_pano/data/official_building_split/{split}.full.jsonl'
        sources[str(f)] = sha(f)
    contract = dict(schema_version=SCHEMA, frame=FRAME, units='metres',
        visibility_policy='direct_mesh_local512_erp512x1024_ge16_v1',
        local_without_erp_policy='retain_for_review_and_flag_not_certified',
        size='camera_axis_AABB_full_lengths_from_source_OBB', category_mapping_sha256=mapping_hash,
        vocabulary=vocabulary, source_sha256=sources,
        selection=counts, thresholds=dict(ncc=args.min_score, margin=args.min_margin), ready_for_training=False)
    if args.resume:
        validate_resume(output, contract, selected)
    else:
        atomic_json(output/'contract.json', contract)
    atomic_json(output/'vocabulary.json', dict(category_mapping_sha256=mapping_hash, categories=vocabulary))
    scene, current_house = None, None
    manifests, observer_rows, failures, rebuilt = [], [], [], set()
    for row in sorted(selected, key=lambda r:(r['scene_id'], r['id'])):
        directory = output/'samples'/row['id']
        if args.resume and (directory/'DATA_COMPLETE.json').is_file():
            manifest, observer = load_sample(directory)
            if manifest['id'] != row['id'] or manifest['split'] != row['split']:
                raise ValueError('Resume sample identity mismatch')
            manifests.append(manifest)
            observer_rows.append(observer)
            print(f'Reusing {row["id"]}', flush=True)
            continue
        archive_partial(directory)
        rebuilt.add(row['id'])
        print(f'Building {row["split"]} {row["id"]}', flush=True)
        try:
            if current_house != row['scene_id']:
                del scene
                scene = None
                links_path = OLD/'object_link_pilot'/row['scene_id']/'object_links.json'
                scene = MeshScene(links_path, args.threads)
                current_house = row['scene_id']
                atomic_json(output/'scene_sources'/f'{current_house}.json', dict(source_sha256=scene.source_sha256))
            manifest, observer = export_one(row, output, scene, args, mapping_hash)
            commit_sample(directory, manifest, observer)
            manifests.append(manifest)
            observer_rows.append(observer)
        except (ValueError, OSError, KeyError) as exc:
            failures.append(dict(id=row['id'], split=row['split'], error=f'{type(exc).__name__}: {exc}'))
            print(f'FAILED {row["id"]}: {exc}', flush=True)
        atomic_rows(output/'gt_manifest.jsonl', manifests)
        atomic_rows(output/'observer_inputs.jsonl', observer_rows)
        atomic_json(output/'failures.json', failures)
    atomic_rows(output/'gt_manifest.jsonl', manifests)
    atomic_rows(output/'observer_inputs.jsonl', observer_rows)
    atomic_json(output/'failures.json', failures)
    reviews = read(output/'review_decisions.json') if (output/'review_decisions.json').exists() else {}
    if any(rid in reviews for rid in rebuilt):
        atomic_json(output/'review_decisions.before_rebuild.json', reviews)
    for r in manifests:
        if r['id'] in rebuilt:
            reviews[r['id']] = dict(decision='pending', notes='Rebuilt sample; inspect new output')
        reviews.setdefault(r['id'], dict(decision='pending', notes=''))
    atomic_json(output/'review_decisions.json', reviews)
    links = ['<meta charset="utf-8"><h1>GT pilot: visual review required</h1>',
             '<p>Compare local / target front / legacy front, then mesh boundaries and boxes. Edit review_decisions.json only after checking each sample.</p>']
    for row in manifests:
        name, path = html.escape(row['id']), html.escape(row['directory'])
        links.append(f'<h2>{name} ({row["split"]})</h2><p>{html.escape(str(row["review_flags"]))}</p>'
                     f'<a href="{path}/bridge.json">bridge diagnostics</a><br><img width="1200" src="{path}/review.jpg">')
    (output/'review.html').write_text('\n'.join(links))
    atomic_json(output/'EXPORT_COMPLETE.json', dict(status='REVIEW_REQUIRED', selected=len(selected),
         exported=len(manifests), failures=len(failures), numerical_flags=sum(bool(r['review_flags']) for r in manifests),
         ready_for_training=False, observer_inputs_sha256=sha(output/'observer_inputs.jsonl'),
         gt_manifest_sha256=sha(output/'gt_manifest.jsonl')))
    print(f'Inspect {output}/review.html and bridge.json. No training approval inferred.', flush=True)
    if failures or not manifests:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
