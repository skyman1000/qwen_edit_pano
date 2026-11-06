"""CPU visual review of existing and recoverable polished object supervision.

No downloads, training, hashes, automatic exclusion, or source-dataset writes.
"""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import html
import io
import json
import multiprocessing
from pathlib import Path
import random

import numpy as np
from PIL import Image

from qwen_pano.caupano.world_view import local_rays, sample_erp
from .build_polished_world import (ROOT, FRAME, aligned_images, atomic_json, build_house,
                                   pair_identity, read, read_rows, validate_objects)


DEFAULT_AUDIT = ROOT/'research/audits/polished_build_failures_2026-10-18/registration_checks.jsonl'


def choose_existing(records, calibrations, count):
    """Purposeful coverage sample, not an accuracy estimate for the full dataset."""
    chosen, ids, houses = [], set(), set()
    random_order = sorted(records, key=lambda row: row['id'])
    random.Random(20261018).shuffle(random_order)
    def take(pool, amount, reason):
        for _ in range(amount):
            remaining = [row for row in pool if row['id'] not in ids]
            if not remaining or len(chosen) >= count:
                break
            row = next((row for row in remaining if row['scene_id'] not in houses), remaining[0])
            ids.add(row['id']); houses.add(row['scene_id'])
            chosen.append(dict(id=row['id'], reason=reason))
    take([row for row in random_order if row['state_status']=='REUSED_EXISTING_STATE'], 2, 'reused_state')
    take([row for row in random_order if row['state_status']=='BUILT_MINIMAL_STATE'], 2, 'new_state')
    take(sorted(records, key=lambda row:(-len(row['objects']),row['id'])), 2, 'many_objects')
    take(sorted(records, key=lambda row:(calibrations[row['id']].get('local_target_front_ncc',1.),row['id'])), 2, 'lower_front_NCC')
    take(sorted(records, key=lambda row:(calibrations[row['id']].get('supervised_pixel_fraction',1.),row['id'])), 2, 'lower_region_loss_coverage')
    take(random_order, max(0,count-len(chosen)), 'different_views')
    return chosen


def encode_ids(instance, valid, object_ids):
    represented = valid & np.isin(instance, list(object_ids))
    ids = np.where(represented,instance,0).astype(np.int64)
    if ids.min() < 0 or ids.max() >= 2**24:
        raise ValueError('Instance IDs exceed lossless RGB encoding capacity')
    return np.stack([(ids>>16)&255,(ids>>8)&255,ids&255],axis=-1).astype(np.uint8), ids


def contours(ids, periodic=True):
    # Longitude wraps; latitude does not.
    if periodic:
        edge = (ids!=np.roll(ids,1,axis=1)) | (ids!=np.roll(ids,-1,axis=1))
    else:
        edge=np.zeros(ids.shape,dtype=bool)
        edge[:,1:] |= ids[:,1:]!=ids[:,:-1]
        edge[:,:-1] |= ids[:,:-1]!=ids[:,1:]
    edge[1:] |= ids[1:]!=ids[:-1]
    edge[:-1] |= ids[:-1]!=ids[1:]
    return edge & (ids>0)


def overlay(rgb, ids, loss, periodic=True):
    result=rgb.copy(); edge=contours(ids,periodic)
    result[edge & loss]=[30,255,70]
    result[edge & ~loss]=[255,60,70]
    return result


def image_url(array, lossless=False):
    buffer=io.BytesIO()
    Image.fromarray(array).save(buffer,format='PNG' if lossless else 'JPEG', **({} if lossless else dict(quality=88)))
    mime='image/png' if lossless else 'image/jpeg'
    return 'data:'+mime+';base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def render_case(record, calibration, names, expected, directory, reason):
    if record['frame']!=FRAME:
        raise ValueError('Unexpected coordinate frame')
    object_ids=validate_objects(record['objects'],names)
    if record['id']!=calibration['sample_id']:
        raise ValueError('Calibration/sample identity differs')
    with np.load(record['object_region_supervision']['path'],allow_pickle=False) as arrays:
        instance=arrays['instance'].copy()
        valid=arrays['instance_valid'].copy()
        loss=arrays['object_loss_valid'].copy()
    shape=(expected['height'],2*expected['height'])
    if instance.shape!=shape or valid.shape!=shape or loss.shape!=shape:
        raise ValueError('Unexpected instance/mask shape')
    if not np.issubdtype(instance.dtype,np.integer) or valid.dtype!=bool or loss.dtype!=bool:
        raise ValueError('Instance/mask dtype differs from contract')
    if np.any(valid & (instance<=0)):
        raise ValueError('Valid instance pixels require positive IDs')
    encoded,ids=encode_ids(instance,valid,object_ids)
    limit=calibration.get('object_loss_max_latitude',expected['object_loss_max_latitude'])
    lat=90-(np.arange(shape[0])+.5)/shape[0]*180
    permitted=np.abs(lat)[:,None]<=limit
    expected_loss=(ids>0) & permitted
    if not np.array_equal(loss,expected_loss):
        raise ValueError('Stored region-loss support differs from represented/valid/latitude policy')
    local,rgb=aligned_images(record,shape[0])
    rays=local_rays(512)
    front=sample_erp(rgb,rays)
    front_ids=sample_erp(ids,rays,nearest=True)
    front_loss=sample_erp(loss,rays,nearest=True)
    front_encoded,_=encode_ids(front_ids,np.ones(front_ids.shape,dtype=bool),object_ids)
    values=[]
    for obj in record['objects']:
        iid=int(obj['instance_id']); selected=ids==iid
        values.append(dict(id=iid,category_id=obj['category_id'],category=names[str(obj['category_id'])],
            center=obj['center_local_m'],size=obj['size_aabb_local_m'],
            center_valid=obj['center_valid'],size_valid=obj['size_valid'],
            ERP_pixels=int(selected.sum()),region_loss_pixels=int((selected & loss).sum()),
            front_sampled_pixels=int((front_ids==iid).sum())))
    metadata=dict(id=record['id'],split=record['split'],scene_id=record['scene_id'],reason=reason,
        state_status=record['state_status'],objects=values,latitude_limit=limit,
        front_NCC=calibration.get('local_target_front_ncc'),
        fit_inliers=calibration.get('fit_inliers'),heldout_inliers=calibration.get('heldout_inliers'),
        heldout_observations=calibration.get('heldout_observations'),
        heldout_median_degrees=calibration.get('heldout_median_degrees'),
        fit_yaw_parity=calibration.get('fit_yaw_parity'),
        region_loss_fraction=float(loss.mean()),represented_object_fraction=float((ids>0).mean()),
        objects_without_region_loss=sum(obj['region_loss_pixels']==0 for obj in values),
        small_ERP_objects=sum(0<obj['ERP_pixels']<=64 for obj in values),
        source_mask=record['object_region_supervision']['path'],source_calibration=record['calibration'],
        geometry_quality='pending_visual_review',ready_for_training=False,
        note='Front mask is sampled from ERP supervision, not independent Local GT. AABB includes occluded/empty volume; it is not a visible silhouette.')
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'statistics.json',metadata)
    top=np.concatenate([local,overlay(front,front_ids,front_loss,False)],axis=1)
    bottom=overlay(rgb,ids,loss)
    preview=np.concatenate([top,np.asarray(Image.fromarray(bottom).resize((1024,512),Image.Resampling.NEAREST))],axis=0)
    Image.fromarray(preview).save(directory/'preview.jpg',quality=90)
    return dict(**metadata,images=dict(local=image_url(local),ERP=image_url(rgb),
        ERP_ids=image_url(encoded,True),ERP_loss=image_url(loss.astype(np.uint8)*255,True),
        front=image_url(front),front_ids=image_url(front_encoded,True),
        front_loss=image_url(front_loss.astype(np.uint8)*255,True)))


def make_page(cases, failures):
    template=(Path(__file__).with_name('polished_review.html')).read_text()
    # Embedded images allow offline file:// use without fetch/CORS dependencies.
    payload=json.dumps(cases,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    errors=html.escape(json.dumps(failures,ensure_ascii=False,indent=2))
    return template.replace('__CASES__',payload).replace('__FAILURES__',errors)


def run(args):
    source_config=read(args.source/'CONFIG.json')
    source_result=read(args.source/'BUILD_RESULT.json')
    records=[row for split in ('train','val') for row in read_rows(args.source/f'{split}.jsonl')]
    index={row['id']:row for row in records}
    if len(index)!=len(records) or len(records)!=source_result['completed']:
        raise ValueError('Source successful manifests are duplicated or inconsistent')
    candidates=[row for split in ('train','val') for row in read_rows(args.candidates/f'{split}.jsonl')]
    if source_config['sources']!=[pair_identity(row) for row in candidates]:
        raise ValueError('Candidate sources differ from the original build')
    candidates_by_id={row['id']:row for row in candidates}
    checked=read_rows(args.registration_audit)
    audited_ids={row['id'] for row in checked}
    if len(audited_ids)!=len(checked):
        raise ValueError('Duplicate calibration audit IDs')
    original_failures={row['id'] for row in source_result['failures']}
    recovered=[]
    for check in checked:
        if check['id'] not in original_failures:
            raise ValueError('Calibration audit references a non-failed source view')
        if check['status']=='registration_passed':
            report=check['report']
            if (report['fit_inliers']<60 or report['heldout_inliers']<30 or report['heldout_observations']<3
                    or report['requested_fit_parities']!=[0,1]
                    or report['sift_features']!=source_config['sift_features']
                    or report['ransac_trials']!=source_config['ransac_trials']):
                raise ValueError('Calibration audit settings/support differ from this review')
            recovered.append(check['id'])
    recovered=sorted(recovered)
    if args.recover_count is not None:
        recovered=recovered[:args.recover_count]
    calibrations={row['id']:read(row['calibration']) for row in records}
    selected=choose_existing(records,calibrations,args.existing_count)
    config=dict(protocol='polished_object_alignment_review_v1',source=str(args.source),
        candidates=str(args.candidates),registration_audit=str(args.registration_audit),
        existing_selection=selected,recovered_selection=recovered,source_requested=source_result['requested'],
        source_completed=source_result['completed'],source_build_config=source_config,
        automatic_exclusions=False,hashes_computed=False,GPU=False)
    if args.preflight:
        print(json.dumps(dict(status='PREFLIGHT_PASSED',existing_reviews=len(selected),
            recovered_reviews=len(recovered),source_completed=len(records),dataset_modified=False,GPU=False)),flush=True)
        return
    if args.output.exists():
        if not args.resume or read(args.output/'CONFIG.json')!=config:
            raise ValueError('Review output exists; --resume requires the same source and selection')
    else:
        args.output.mkdir(parents=True)
        atomic_json(args.output/'CONFIG.json',config)
    review_lock=(args.output/'REVIEW.lock').open('a')
    try:
        fcntl.flock(review_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError('Another process is writing this review directory') from exc
    names=source_config['names']
    build_root=args.output/'recovered'
    build_root.mkdir(exist_ok=True)
    build_args=argparse.Namespace(**{key:source_config[key] for key in
        ('height','min_instance_pixels','min_front_ncc','object_loss_max_latitude','sift_features','ransac_trials')},
        output=build_root,raw_root=Path(source_config['raw_root']),
        category_mapping=Path(source_config['category_mapping']),threads=args.threads,
        rebuild_existing=False,registration_fit_parities=[0,1])
    groups={}
    for sid in recovered:
        row=candidates_by_id[sid]
        if row['objects'] is not None:
            raise ValueError('Expected a fresh failed sample, not inherited state: '+sid)
        groups.setdefault(row['scene_id'],[]).append(row)
    print(json.dumps(dict(existing=len(selected),recovered_to_build=len(recovered),
        houses=len(groups),workers=args.workers,GPU=False,output=str(args.output))),flush=True)
    build_results=[]
    if args.workers==1:
        for group in groups.values():
            result=build_house((group,build_args,names));build_results.append(result)
            atomic_json(build_root/'progress.json',build_results)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(build_house,(group,build_args,names)) for group in groups.values()]
            for future in as_completed(futures):
                build_results.append(future.result());atomic_json(build_root/'progress.json',build_results)
    failures=[failure for result in build_results for failure in result['failures']]
    review_rows=[(index[item['id']],item['reason']) for item in selected]
    for sid in recovered:
        directory=build_root/'samples'/sid
        if not (directory/'failure.json').exists() and (directory/'record.json').is_file():
            review_rows.append((read(directory/'record.json'),'recovered_candidate'))
    cases=[]
    for row,reason in review_rows:
        try:
            case=render_case(row,read(row['calibration']),names,source_config,args.output/'cases'/row['id'],reason)
            cases.append(case)
            print(json.dumps(dict(id=row['id'],status='REVIEW_RENDERED',objects=len(case['objects']))),flush=True)
        except Exception as exc:
            failures.append(dict(id=row['id'],stage='review_render',error=f'{type(exc).__name__}: {exc}'))
        atomic_json(args.output/'progress.json',dict(rendered=len(cases),failures=failures))
    (args.output/'review.html').write_text(make_page(cases,failures))
    summary=dict(protocol=config['protocol'],original_requested=source_result['requested'],
        original_completed=source_result['completed'],existing_selected=len(selected),recovered_selected=len(recovered),
        recovered_built_for_review=sum(result['completed'] for result in build_results),
        rendered=len(cases),failures=failures,source_dataset_modified=False,automatic_exclusions=False,
        ready_for_training=False,visual_quality_status='pending_manual_review',
        note='A diagnostic sample, not a full-data accuracy estimate. Native source records/masks were not changed. No download, GPU, model or hash computation.')
    atomic_json(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('OPEN: '+str(args.output/'review.html'),flush=True)
    if failures:
        raise SystemExit('Review has failures; inspect summary.json and the available review.html. No samples were excluded.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'qwen_edit_pano/data/world_polished_full_v2')
    parser.add_argument('--candidates',type=Path,default=ROOT/'qwen_edit_pano/data/world_polished_full_candidates_v2')
    parser.add_argument('--registration-audit',type=Path,default=DEFAULT_AUDIT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing-count',type=int,default=12)
    parser.add_argument('--recover-count',type=int,default=None,help='Default: review all candidates that passed the saved calibration audit')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.existing_count<0 or (args.recover_count is not None and args.recover_count<0) or min(args.workers,args.threads)<1:
        parser.error('Invalid sample counts or CPU settings')
    for key in ('source','candidates','registration_audit','output'):
        setattr(args,key,getattr(args,key).resolve())
    allowed=(ROOT/'qwen_edit_pano/outputs',ROOT/'research/audits')
    if not any(args.output.is_relative_to(root) and args.output!=root for root in allowed):
        parser.error('Review output must be a separate directory under qwen_edit_pano/outputs or research/audits')
    if any(args.output==path or args.output in path.parents or path in args.output.parents for path in (args.source,args.candidates)):
        parser.error('Review output must be separate from source data/candidates')
    with (args.source/'BUILD.lock').open('r') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Source build is still writing; run review when it finishes')
        run(args)


if __name__=='__main__':
    main()
