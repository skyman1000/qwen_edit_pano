"""Build minimal polished/face2 object data and visible-instance supervision.

CPU only; user-run full construction. No model/cache generation, download,
content hashing, full-state export, sensor depth/normal export, or spatial bias.
"""
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import json
import multiprocessing
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from qwen_pano.caupano.world_view import erp_rays, local_rays, sample_erp
from qwen_edit_pano.pair_alignment import apply_alignment
from .minimal_world_sources import MinimalHouse, image_from_zip


ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'polished_face2_minimal_objects_instance_regions_v1'
FRAME = 'local_view_x_right_y_image_up_z_forward_metres'


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def atomic_mask(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def pair_identity(row):
    return {k:row[k] for k in ('id', 'split', 'scene_id', 'source_view_id',
                               'local_condition', 'image', 'target_alignment')}


def normalized_ncc(first, second):
    first, second = first.astype(float), second.astype(float)
    first -= first.mean(axis=(0, 1), keepdims=True)
    second -= second.mean(axis=(0, 1), keepdims=True)
    denominator = np.sqrt(np.square(first).sum()*np.square(second).sum())
    return float((first*second).sum()/denominator) if denominator > 1e-8 else -1.


def aligned_images(row, height):
    from qwen_pano.cached_data import ArrowImages
    with ZipFile(row['local_condition']['archive']) as archive:
        local = Image.fromarray(image_from_zip(archive, row['local_condition']['member']))
    with ArrowImages().open(row['image']) as image:
        if image.width != 2*image.height:
            raise ValueError('Polished target is not 2:1')
        target = apply_alignment(image.convert('RGB'), row['target_alignment'])
    rgb = np.asarray(target.resize((2*height, height), Image.Resampling.BICUBIC)).copy()
    return np.asarray(local.resize((512, 512), Image.Resampling.BICUBIC)).copy(), rgb


def validate_objects(objects, names):
    ids = set()
    for obj in objects:
        iid = int(obj['instance_id'])
        if iid <= 0 or iid in ids or str(obj['category_id']) not in names:
            raise ValueError('Invalid/duplicate instance or excluded class')
        ids.add(iid)
        for key, valid in (('center_local_m', 'center_valid'), ('size_aabb_local_m', 'size_valid')):
            if not obj[valid]:
                if obj[key] is not None:
                    raise ValueError('Unknown geometry must be null')
                continue
            value = np.asarray(obj[key], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Invalid object geometry')
            if key == 'size_aabb_local_m' and np.any(value <= 0):
                raise ValueError('AABB full lengths must be positive')
    return ids


def save_preview(path, rgb, instance, objects):
    overlay = rgb.copy()
    edge = np.zeros(instance.shape, dtype=bool)
    edge[1:] |= instance[1:] != instance[:-1]
    edge[:, 1:] |= instance[:, 1:] != instance[:, :-1]
    edge &= instance > 0
    overlay[edge] = [30, 255, 70]
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    largest = sorted(objects, key=lambda o:np.count_nonzero(instance==int(o['instance_id'])), reverse=True)[:20]
    for obj in largest:
        y, x = np.where(instance==int(obj['instance_id']))
        if len(x):
            draw.text((int(np.median(x)), int(np.median(y))),
                      f'{obj["instance_id"]}:{obj["category_id"]}', fill='yellow')
    image.save(path, quality=88)


def build_sample(row, house, args, names, preview=False):
    directory = args.output/'samples'/row['id']
    directory.mkdir(parents=True, exist_ok=True)
    reuse = row['objects'] is not None and not args.rebuild_existing
    warnings = []
    local, rgb = aligned_images(row, args.height)
    front = sample_erp(rgb, local_rays(512))
    score = normalized_ncc(local[32:-32, 32:-32], front[32:-32, 32:-32])
    if score < args.min_front_ncc:
        if reuse:
            warnings.append(f'inherited_selected_state_front_NCC:{score:.4f}')
        else:
            Image.fromarray(np.concatenate([local, front], axis=1)).save(directory/'alignment_failure.jpg')
            raise ValueError(f'Polished target/front mismatch: NCC={score:.4f}')
    details = dict(local_target_front_ncc=score)
    if reuse:
        supervision = row['object_region_supervision']
        if supervision is None:
            raise ValueError('Existing object state has no region supervision; use --rebuild-existing')
        with np.load(supervision['path'], allow_pickle=False) as data:
            instance, valid = data['instance'].copy(), data['instance_valid'].astype(bool)
        if instance.shape != (args.height, 2*args.height):
            raise ValueError('Existing mask resolution differs; use --height 512 or --rebuild-existing')
        objects = row['objects']
        bridge_path = Path(supervision['path']).parent/'bridge.json'
        if not bridge_path.is_file():
            raise ValueError('Existing mask lacks its frame metadata')
        bridge = read(bridge_path)
        if bridge['sample_id'] != row['id']:
            raise ValueError('Existing frame identity mismatch')
        details.update(origin_world=bridge['origin_world'], basis_local_to_world=bridge['basis_local_to_world'],
                       method='reuse_current_selected_state_and_instance_projection',
                       inherited_review_flags=bridge.get('review_flags', []))
    else:
        origin, basis, registration = house.register(row, args.sift_features, args.ransac_trials,
                                                      args.registration_fit_parities)
        details.update(registration)
        instance, hit_valid = house.cast(origin, basis, erp_rays(args.height, 2*args.height))
        local_instance, _ = house.cast(origin, basis, local_rays(512))
        valid = instance > 0
        if not hit_valid.any():
            raise ValueError('No ERP ray hits the source mesh')
        details['mesh_hit_fraction'] = float(hit_valid.mean())
        details['source_link_issues'] = house.issues
        details['source_label_repairs'] = house.repairs
        local_counts = dict(zip(*np.unique(local_instance, return_counts=True)))
        erp_counts = dict(zip(*np.unique(instance, return_counts=True)))
        objects = []
        for iid, obj in sorted(house.objects.items()):
            if str(obj['category_id']) not in names or not obj['geometry_valid']:
                continue
            visible_local = local_counts.get(iid, 0) >= args.min_instance_pixels
            visible_erp = erp_counts.get(iid, 0) >= args.min_instance_pixels
            if not (visible_local or visible_erp):
                continue
            center = (obj['center_world']-origin) @ basis
            size = 2*obj['radii'] @ np.abs(obj['axes_world'] @ basis)
            objects.append(dict(instance_id=str(iid), category_id=obj['category_id'],
                                center_local_m=center.tolist(), size_aabb_local_m=size.tolist(),
                                center_valid=True, size_valid=True))
            if visible_local and not visible_erp:
                warnings.append(f'local_object_below_ERP_threshold:{iid}')
    ids = validate_objects(objects, names)
    if instance.shape != valid.shape or not np.issubdtype(instance.dtype, np.integer):
        raise ValueError('Invalid instance mask shape/dtype')
    if np.any(valid & (instance <= 0)):
        raise ValueError('Valid instance pixels have no positive ID')
    # Only represented task objects receive instance supervision. Other surfaces
    # remain in the RGB target and the scene as occluders, never as fake void.
    represented = np.isin(instance, np.array(sorted(ids), dtype=np.int32))
    latitude = 90-(np.arange(args.height)+.5)/args.height*180
    # Conservative training-only support avoids generatively polished pole cores.
    # This is not an official capture-validity annotation; global RGB loss is unchanged.
    reliable_latitude = np.abs(latitude)[:, None] <= args.object_loss_max_latitude
    loss_valid = valid & represented & reliable_latitude
    mask_path = directory/'object_regions.npz'
    atomic_mask(mask_path, instance=instance.astype(np.int32), instance_valid=valid,
                object_loss_valid=loss_valid)
    details.update(profile=PROFILE, frame=FRAME, sample_id=row['id'], object_count=len(objects),
                   supervised_pixel_fraction=float(loss_valid.mean()), warnings=warnings,
                   object_loss_max_latitude=args.object_loss_max_latitude,
                   pole_support_method='conservative_latitude_cut_for_object_loss_only',
                   state_scope='current task classes visible in ERP or face2; not a full building state')
    atomic_json(directory/'calibration.json', details)
    if preview:
        save_preview(directory/'instance_preview.jpg', rgb, instance, objects)
    result = pair_identity(row)
    result.update(frame=FRAME, objects=objects,
                  object_region_supervision=dict(path=str(mask_path.resolve()),
                      arrays=['instance', 'instance_valid', 'object_loss_valid'],
                      instance_id_is_condition=False),
                  state_status='REUSED_EXISTING_STATE' if reuse else 'BUILT_MINIMAL_STATE',
                  calibration=str((directory/'calibration.json').resolve()))
    atomic_json(directory/'record.json', result)
    failure_path = directory/'failure.json'
    if failure_path.exists():
        failure_path.unlink()
    return result


def build_house(payload):
    rows, args, names = payload
    import cv2
    cv2.setNumThreads(args.threads)
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=args.threads)
    except ImportError:
        context = nullcontext()
    house = None
    initialization_error = None
    done, failed = [], []
    with context:
        for row in rows:
            start = time.monotonic()
            directory = args.output/'samples'/row['id']
            record_path = directory/'record.json'
            try:
                if record_path.is_file() and not (directory/'failure.json').exists():
                    record = read(record_path)
                    if pair_identity(record) != pair_identity(row):
                        raise ValueError('Resume sample identity/source pointers differ')
                    validate_objects(record['objects'], names)
                    with np.load(record['object_region_supervision']['path'], allow_pickle=False) as data:
                        if data['instance'].shape != (args.height, 2*args.height):
                            raise ValueError('Resume mask shape mismatch')
                        if data['object_loss_valid'].shape != data['instance'].shape:
                            raise ValueError('Resume loss support shape mismatch')
                    done.append(row['id'])
                    continue
                if row['objects'] is None or args.rebuild_existing:
                    if initialization_error:
                        raise ValueError(initialization_error)
                    if house is None:
                        try:
                            house = MinimalHouse(args.raw_root/row['scene_id'], args.category_mapping, args.threads)
                        except Exception as exc:
                            initialization_error = f'House source initialization: {type(exc).__name__}: {exc}'
                            raise ValueError(initialization_error) from exc
                result = build_sample(row, house, args, names, preview=not done)
                done.append(row['id'])
                print(json.dumps(dict(id=row['id'], status=result['state_status'], objects=len(result['objects']),
                                      seconds=round(time.monotonic()-start, 2))), flush=True)
            except Exception as exc:
                directory.mkdir(parents=True, exist_ok=True)
                error = dict(id=row['id'], split=row['split'], error=f'{type(exc).__name__}: {exc}')
                atomic_json(directory/'failure.json', error)
                failed.append(error)
                print(json.dumps(dict(status='FAILED', **error)), flush=True)
    return dict(house=rows[0]['scene_id'], completed=len(done), failures=failed)


def summarize(output, candidates):
    manifests, failures = {'train':[], 'val':[]}, []
    sample_root = output/'samples'
    present = {p.name for p in sample_root.iterdir() if p.is_dir()} if sample_root.is_dir() else set()
    for row in candidates:
        if row['id'] not in present:
            continue
        directory = output/'samples'/row['id']
        if (directory/'failure.json').is_file():
            failures.append(read(directory/'failure.json'))
        elif (directory/'record.json').is_file():
            record = read(directory/'record.json')
            if pair_identity(record) != pair_identity(row):
                raise ValueError('Summary record has different source pointers')
            if not Path(record['object_region_supervision']['path']).is_file():
                failures.append(dict(id=row['id'], split=row['split'], error='Missing completed mask file'))
                continue
            manifests[row['split']].append(record)
    for split, records in manifests.items():
        path = output/f'{split}.jsonl'
        tmp = path.with_suffix('.tmp.jsonl')
        with tmp.open('w') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        tmp.replace(path)
    objects = [len(row['objects']) for records in manifests.values() for row in records]
    completed = sum(map(len, manifests.values()))
    result = dict(profile=PROFILE, requested=len(candidates), completed=completed,
                  all_candidates_built=completed==len(candidates) and not failures,
                  counts={split:len(records) for split, records in manifests.items()},
                  reused=sum(row['state_status']=='REUSED_EXISTING_STATE' for rs in manifests.values() for row in rs),
                  new=sum(row['state_status']=='BUILT_MINIMAL_STATE' for rs in manifests.values() for row in rs),
                  maximum_objects=max(objects, default=0), samples_over_old_capacity160=sum(n>160 for n in objects),
                  empty_object_samples=sum(n==0 for n in objects), failures=failures,
                  missing=len(candidates)-completed-len(failures),
                  hashes_computed=False, spatial_bias=False, legacy_loader_compatible=False,
                  note='CPU object data only. No text embeddings, image/VAE caches, model training or '
                       'claim of individual visual certification. Do not silently treat partial manifests as full data.')
    selection_path=output/'SELECTION.json'
    if selection_path.is_file():
        selection=read(selection_path)
        if len(candidates)!=selection['selected_requested']:
            raise ValueError('Selection denominator differs from the current candidates')
        result.update(original_requested=selection['original_requested'],
                      excluded_from_original=selection['excluded_count'],
                      all_original_candidates_built=result['all_candidates_built'] and selection['excluded_count']==0,
                      seeded_successes=selection['seeded_successes'],
                      selection_note='Completeness refers to the explicitly selected subset; original exclusions remain recorded in SELECTION.json.')
    atomic_json(output/'BUILD_RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=ROOT/'qwen_edit_pano/data/world_polished_full_candidates_v2')
    parser.add_argument('--raw-root', type=Path, default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    parser.add_argument('--category-mapping', type=Path,
                        default=ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=2, help='Independent CPU house processes')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads per worker')
    parser.add_argument('--height', type=int, default=512, help='Instance supervision ERP height; RGB target unchanged')
    parser.add_argument('--min-instance-pixels', type=int, default=16)
    parser.add_argument('--min-front-ncc', type=float, default=.65)
    parser.add_argument('--object-loss-max-latitude', type=float, default=54.)
    parser.add_argument('--sift-features', type=int, default=3600)
    parser.add_argument('--ransac-trials', type=int, default=3000)
    parser.add_argument('--registration-fit-parities', nargs='+', type=int, choices=[0,1], default=[0],
                        help='Allowed disjoint fit/validation yaw splits; default preserves the original even-fit protocol')
    parser.add_argument('--rebuild-existing', action='store_true', help='Diagnostic fresh build instead of reusing selected states')
    parser.add_argument('--max-samples', type=int, default=0, help='Execution limit only; zero runs all pending candidates')
    parser.add_argument('--sample-id', nargs='+', help='Diagnostic execution filter, not a change to the full candidate set')
    args = parser.parse_args()
    if (min(args.workers, args.threads, args.height, args.min_instance_pixels, args.sift_features, args.ransac_trials)<1
            or args.max_samples<0 or not -1<=args.min_front_ncc<=1 or not 0<args.object_loss_max_latitude<=90):
        parser.error('Invalid CPU dimensions or thresholds')
    if len(set(args.registration_fit_parities))!=len(args.registration_fit_parities):
        parser.error('Registration partitions must be distinct')
    for key in ('candidates', 'raw_root', 'category_mapping', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output==args.candidates or args.output in args.candidates.parents or args.candidates in args.output.parents:
        parser.error('Output must be separate from the candidates directory')
    candidates = [row for split in ('train', 'val') for row in read_rows(args.candidates/f'{split}.jsonl')]
    ids, houses = set(), {}
    for row in candidates:
        if row['id'] in ids or houses.setdefault(row['scene_id'], row['split'])!=row['split']:
            raise ValueError('Duplicate ID or building split overlap')
        if row['local_condition']['face_index']!=2:
            raise ValueError('Only single face2 inputs are supported')
        ids.add(row['id'])
    names = read(args.candidates/'category_names.json')
    config = dict(profile=PROFILE, frame=FRAME, target='polished_ERP', Local='single_original_face2',
                  size='camera_AABB_full_side_lengths_metres', height=args.height,
                  min_instance_pixels=args.min_instance_pixels, min_front_ncc=args.min_front_ncc,
                  object_loss_max_latitude=args.object_loss_max_latitude,
                  sift_features=args.sift_features, ransac_trials=args.ransac_trials,
                  rebuild_existing=args.rebuild_existing, names=names,
                  sources=[pair_identity(row) for row in candidates], raw_root=str(args.raw_root),
                  category_mapping=str(args.category_mapping))
    if args.registration_fit_parities != [0]:
        config['registration_fit_parities'] = args.registration_fit_parities
    existed = args.output.exists()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output/'BUILD.lock').open('a')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another writer holds this output directory')
    if existed:
        if not args.resume or not (args.output/'CONFIG.json').is_file() or read(args.output/'CONFIG.json')!=config:
            parser.error('Existing output requires --resume and identical data/config; otherwise use a new output')
    else:
        atomic_json(args.output/'CONFIG.json', config)
        atomic_json(args.output/'category_names.json', names)
    sample_root = args.output/'samples'
    present = {p.name for p in sample_root.iterdir() if p.is_dir()} if sample_root.is_dir() else set()
    selected = []
    for row in candidates:
        directory = sample_root/row['id']
        if row['id'] not in present or not (directory/'record.json').is_file() or (directory/'failure.json').is_file():
            selected.append(row)
        elif not Path(read(directory/'record.json')['object_region_supervision']['path']).is_file():
            parser.error(f'Completed record lacks mask: {row["id"]}; inspect that sample before resuming')
    if args.sample_id:
        if set(args.sample_id)-ids:
            parser.error('Unknown sample ID')
        selected = [row for row in selected if row['id'] in args.sample_id]
    if args.max_samples:
        selected = selected[:args.max_samples]
    groups = {}
    for row in selected:
        groups.setdefault(row['scene_id'], []).append(row)
    print(json.dumps(dict(profile=PROFILE, candidates=len(candidates), selected=len(selected),
                          houses=len(groups), workers=args.workers, threads=args.threads,
                          output=str(args.output), hashes=False, GPU=False)), flush=True)
    try:
        progress = {}
        if args.workers==1:
            for group in groups.values():
                house_result = build_house((group, args, names))
                progress[house_result['house']] = house_result
                atomic_json(args.output/'progress.json', progress)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(build_house, (group, args, names)) for group in groups.values()]
                for future in as_completed(futures):
                    house_result = future.result()
                    progress[house_result['house']] = house_result
                    atomic_json(args.output/'progress.json', progress)
                    print(json.dumps(house_result), flush=True)
    finally:
        result = summarize(args.output, candidates)
        print(json.dumps({k:v for k,v in result.items() if k!='failures'}, indent=2), flush=True)
    if result['failures']:
        raise SystemExit('Some samples failed. Inspect BUILD_RESULT.json and per-sample failure.json; do not label this a complete dataset.')


if __name__ == '__main__':
    main()
