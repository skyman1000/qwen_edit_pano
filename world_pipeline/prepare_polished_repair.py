"""Prepare a separate, explicitly selected repair build; never run it or alter v2.

Excluding views without source poses requires the explicit CLI flag. Existing
successful masks/calibrations are referenced in place, not rebuilt or overwritten.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from .build_polished_world import ROOT, FRAME, PROFILE, atomic_json, pair_identity, read, read_rows, validate_objects


MISSING_POSE = 'ValueError: Panorama missing from house position/camera table'


def recovered_records(review, config, source, candidates, retry, names):
    """Reuse completed review builds, without equating rendering with visual approval."""
    review_config = read(review/'CONFIG.json')
    if (review_config['protocol'] != 'polished_object_alignment_review_v1'
            or Path(review_config['source']).resolve() != source
            or Path(review_config['candidates']).resolve() != candidates
            or review_config['source_build_config'] != config):
        raise ValueError('Recovered review uses different source data/configuration')
    by_id = {row['id']: row for split in ('train', 'val')
             for row in read_rows(candidates/f'{split}.jsonl')}
    records = []
    for sid in sorted(retry):
        directory = review/'recovered/samples'/sid
        if not (directory/'record.json').is_file() or (directory/'failure.json').exists():
            continue
        row = read(directory/'record.json')
        if (pair_identity(row) != pair_identity(by_id[sid]) or row['frame'] != FRAME
                or row['state_status'] != 'BUILT_MINIMAL_STATE'):
            raise ValueError('Recovered record has a different identity/frame/state: '+sid)
        object_ids = validate_objects(row['objects'], names)
        calibration = read(row['calibration'])
        if (calibration['sample_id'] != sid or calibration['profile'] != PROFILE
                or calibration['frame'] != FRAME or calibration['object_count'] != len(row['objects'])
                or calibration['requested_fit_parities'] != [0, 1]
                or calibration['fit_inliers'] < 60 or calibration['heldout_inliers'] < 30
                or calibration['heldout_observations'] < 3
                or calibration['local_target_front_ncc'] < config['min_front_ncc']
                or any(calibration[key] != config[key] for key in
                       ('sift_features', 'ransac_trials', 'object_loss_max_latitude'))):
            raise ValueError('Recovered calibration differs from the repair contract: '+sid)
        with np.load(row['object_region_supervision']['path'], allow_pickle=False) as arrays:
            instance, valid, loss = (arrays[k] for k in ('instance', 'instance_valid', 'object_loss_valid'))
            shape = (config['height'], 2*config['height'])
            if (any(a.shape != shape for a in (instance, valid, loss))
                    or not np.issubdtype(instance.dtype, np.integer) or valid.dtype != bool or loss.dtype != bool):
                raise ValueError('Recovered masks have invalid shapes/types: '+sid)
            latitude = 90-(np.arange(shape[0])+.5)/shape[0]*180
            expected = valid & np.isin(instance, list(object_ids)) & (
                np.abs(latitude)[:, None] <= config['object_loss_max_latitude'])
            if not np.array_equal(loss, expected):
                raise ValueError('Recovered object-loss support differs from the contract: '+sid)
        records.append(row)
    return records


def prepare(source, candidates, output, candidate_output, exclude_missing_poses=False,
            registration_audit=None, exclude_unregistered=False, preflight=False, recovered_review=None,
            resume=False):
    source,candidates,output,candidate_output = [Path(p).resolve() for p in (source,candidates,output,candidate_output)]
    for target in (output,candidate_output):
        if any(target==p or target in p.parents or p in target.parents for p in (source,candidates)):
            raise ValueError('Use separate NEW output and candidate directories')
        if target.exists() and not resume:
            raise ValueError('Output already exists; use --resume to validate and finish preparation, or use separate NEW directories: '+str(target))
        if target.exists() and not target.is_dir():
            raise ValueError('Output path is not a directory: '+str(target))
    if output==candidate_output or output in candidate_output.parents or candidate_output in output.parents:
        raise ValueError('Build output and candidates must be separate')
    with ExitStack() as stack:
        lock = stack.enter_context((source/'BUILD.lock').open('r'))
        try:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Source build is still writing; wait for it to finish') from exc
        if recovered_review is not None:
            recovered_review = Path(recovered_review).resolve()
            review_lock = stack.enter_context((recovered_review/'REVIEW.lock').open('r'))
            try:
                fcntl.flock(review_lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Recovered review is still writing; wait for it to finish') from exc
        config,result = read(source/'CONFIG.json'),read(source/'BUILD_RESULT.json')
        rows = [row for split in ('train','val') for row in read_rows(candidates/f'{split}.jsonl')]
        names = read(candidates/'category_names.json')
        if config['sources'] != [pair_identity(row) for row in rows] or config['names'] != names:
            raise ValueError('Candidates differ from the source build contract')
        by_id = {row['id']:row for row in rows}
        if len(by_id)!=len(rows) or result['requested']!=len(rows):
            raise ValueError('Duplicate candidate IDs or wrong source build denominator')
        failures = {row['id']:row for row in result['failures']}
        if len(failures)!=len(result['failures']):
            raise ValueError('Duplicate source failure IDs')
        excluded,retry = {},{}
        for sid,failure in failures.items():
            if sid not in by_id:
                raise ValueError('Failure references an unknown candidate')
            error = failure['error']
            if error==MISSING_POSE:
                excluded[sid]=deepcopy(failure)
            elif error.startswith('ValueError: Weak held-out registration:') or error=='ValueError: No reliable registration consensus':
                retry[sid]=deepcopy(failure)
            else:
                raise ValueError('Unclassified failure requires inspection: '+str(failure))
        if excluded and not exclude_missing_poses:
            raise ValueError(f'{len(excluded)} views lack source poses. Exclusion changes coverage; explicitly pass --exclude-missing-poses to prepare this subset.')
        if registration_audit is not None:
            checked=read_rows(Path(registration_audit))
            checked_by_id={row['id']:row for row in checked}
            if len(checked_by_id)!=len(checked) or set(checked_by_id)!=set(retry):
                raise ValueError('Registration audit must cover each registration failure exactly once')
            failed_checks=[]
            for sid,check in checked_by_id.items():
                if check['status']=='registration_passed':
                    report=check['report']
                    if (report['fit_inliers']<60 or report['heldout_inliers']<30 or report['heldout_observations']<3
                            or report['requested_fit_parities']!=[0,1]
                            or report['sift_features']!=config['sift_features'] or report['ransac_trials']!=config['ransac_trials']):
                        raise ValueError('Audit used a different registration contract or insufficient support: '+sid)
                elif check['status']=='registration_failed':
                    failed_checks.append(sid)
                else:
                    raise ValueError('Unknown calibration audit status: '+sid)
            if failed_checks and not exclude_unregistered:
                raise ValueError(f'{len(failed_checks)} views still fail independent registration. Explicit --exclude-unregistered is required to prepare this filtered subset.')
            for sid in failed_checks:
                failure=retry.pop(sid)
                excluded[sid]=dict(id=sid,split=failure['split'],
                    error='Registration rejected by both disjoint partitions',
                    original_error=failure['error'],registration_error=checked_by_id[sid]['error'])
        elif exclude_unregistered:
            raise ValueError('--exclude-unregistered requires a complete registration audit')
        completed = [row for split in ('train','val') for row in read_rows(source/f'{split}.jsonl')]
        successful = set()
        for row in completed:
            sid=row['id']
            if sid in successful or sid not in by_id or sid in failures or pair_identity(row)!=pair_identity(by_id[sid]):
                raise ValueError('Invalid, duplicate or conflicting completed record: '+sid)
            validate_objects(row['objects'],names)
            if not Path(row['object_region_supervision']['path']).is_file() or not Path(row['calibration']).is_file():
                raise ValueError('Completed sample lacks its mask/calibration: '+sid)
            successful.add(sid)
        if len(successful)!=result['completed'] or successful|set(failures)!=set(by_id):
            raise ValueError('Source manifests/failures do not account for every original candidate')
        recovered = (recovered_records(recovered_review, config, source, candidates, retry, names)
                     if recovered_review is not None else [])
        for row in recovered:
            retry.pop(row['id'])
        selected=[row for row in rows if row['id'] not in excluded]
        new_config=deepcopy(config)
        new_config['sources']=[pair_identity(row) for row in selected]
        new_config['registration_fit_parities']=[0,1]
        selection=dict(protocol='explicit_source_pose_selection_and_registration_repair_v1',
            source_build=str(source),original_candidates=str(candidates),
            original_requested=len(rows),original_counts=dict(Counter(row['split'] for row in rows)),
            selected_requested=len(selected),selected_counts=dict(Counter(row['split'] for row in selected)),
            excluded_count=len(excluded),excluded=list(excluded.values()),
            seeded_successes=len(completed)+len(recovered),original_successes=len(completed),
            recovered_successes=len(recovered),recovered_review=str(recovered_review) if recovered_review else None,
            visual_quality_status='not_certified_by_packaging',registration_retries=len(retry),
            registration_retry_ids=sorted(retry),registration_fit_parities=[0,1],
            thresholds_unchanged=dict(fit_inliers=60,heldout_inliers=30,heldout_observations=3,angular_degrees=2.5),
            original_files_modified=False,hashes_computed=False,ready_for_training=False,
            registration_audit=str(Path(registration_audit).resolve()) if registration_audit else None,
            note='Subset completeness does not mean all original pairs have object GT. Keep the source build and any recovered review: successful mask/calibration files remain referenced there.')
        # Lock the output before checking existing preparation files or filling gaps.
        # A preflight never creates files; source and review locks are still held.
        if not preflight:
            output.mkdir(parents=True,exist_ok=resume)
            output_lock=stack.enter_context((output/'BUILD.lock').open('a'))
            try:
                fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Another process is preparing/building this output') from exc
        expected_json={output/'CONFIG.json':new_config}
        for directory in (candidate_output,output):
            expected_json[directory/'category_names.json']=names
            expected_json[directory/'SELECTION.json']=selection
        if resume:
            for path,expected in expected_json.items():
                if path.exists() and read(path)!=expected:
                    raise ValueError('Resume configuration/selection differs: '+str(path))
            for split in ('train','val'):
                path=candidate_output/f'{split}.jsonl'
                if path.exists() and read_rows(path)!=[row for row in selected if row['split']==split]:
                    raise ValueError('Resume candidate manifest differs: '+str(path))
            allowed={row['id'] for row in completed+recovered}|set(retry)
            sample_root=output/'samples'
            if sample_root.is_dir() and any(p.name not in allowed for p in sample_root.iterdir() if p.is_dir()):
                raise ValueError('Resume output contains an unexpected sample directory')
            for row in completed+recovered:
                path=sample_root/row['id']/'record.json'
                if path.exists() and read(path)!=row:
                    raise ValueError('Resume completed record differs: '+str(path))
                if (path.parent/'failure.json').exists():
                    raise ValueError('Resume seed has an unexpected failure: '+row['id'])
            for sid,failure in retry.items():
                directory=sample_root/sid
                if (directory/'record.json').exists():
                    raise ValueError('Preparation already advanced to building; resume run_build.sh instead')
                if (directory/'failure.json').exists() and read(directory/'failure.json')!=failure:
                    raise ValueError('Resume failure record differs: '+sid)
        if preflight:
            return selection
        candidate_output.mkdir(parents=True,exist_ok=resume)
        for split in ('train','val'):
            path=candidate_output/f'{split}.jsonl'
            if path.exists():
                continue
            temporary=path.with_suffix('.tmp.jsonl')
            with temporary.open('w') as handle:
                for row in selected:
                    if row['split']==split:
                        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            temporary.replace(path)
        for path,value in expected_json.items():
            if not path.exists():
                atomic_json(path,value)
        for row in completed+recovered:
            directory=output/'samples'/row['id']
            directory.mkdir(parents=True,exist_ok=resume)
            if not (directory/'record.json').exists():
                atomic_json(directory/'record.json',row)
        for sid,failure in retry.items():
            directory=output/'samples'/sid
            directory.mkdir(parents=True,exist_ok=resume)
            if not (directory/'failure.json').exists():
                atomic_json(directory/'failure.json',failure)
        command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.build_polished_world',
                 '--candidates',str(candidate_output),'--output',str(output),'--resume',
                 '--registration-fit-parities','0','1','--workers','2','--threads','4']
        for flag,key in [('raw-root','raw_root'),('category-mapping','category_mapping'),('height','height'),
                         ('min-instance-pixels','min_instance_pixels'),('min-front-ncc','min_front_ncc'),
                         ('object-loss-max-latitude','object_loss_max_latitude'),('sift-features','sift_features'),
                         ('ransac-trials','ransac_trials')]:
            command.extend(['--'+flag,str(config[key])])
        if config['rebuild_existing']:
            command.append('--rebuild-existing')
        (output/'run_build.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(command)+'\n')
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'qwen_edit_pano/data/world_polished_full_v2')
    parser.add_argument('--candidates',type=Path,default=ROOT/'qwen_edit_pano/data/world_polished_full_candidates_v2')
    parser.add_argument('--output',type=Path,default=ROOT/'qwen_edit_pano/data/world_polished_repair_v1')
    parser.add_argument('--candidate-output',type=Path,default=ROOT/'qwen_edit_pano/data/world_polished_repair_candidates_v1')
    parser.add_argument('--exclude-missing-poses',action='store_true')
    parser.add_argument('--registration-audit',type=Path,help='CPU audit JSONL covering all registration failures')
    parser.add_argument('--exclude-unregistered',action='store_true',help='Explicitly exclude views rejected by both partitions in that audit')
    parser.add_argument('--preflight',action='store_true',help='Validate the real selection and source records without creating directories')
    parser.add_argument('--recovered-review',type=Path,help='Reuse completed object/mask builds from this compatible review directory')
    parser.add_argument('--resume',action='store_true',help='Validate existing preparation and fill missing records without replacing differing files')
    args=parser.parse_args()
    print('Checking source and existing records; this may take a few minutes on network storage.',flush=True)
    result=prepare(args.source,args.candidates,args.output,args.candidate_output,args.exclude_missing_poses,
                   args.registration_audit,args.exclude_unregistered,args.preflight,args.recovered_review,args.resume)
    print(json.dumps({k:v for k,v in result.items() if k not in ('excluded','registration_retry_ids')},ensure_ascii=False,indent=2))
    if args.preflight:
        print('PREFLIGHT PASSED. No new candidate/build directories created.')
    else:
        print('Prepared only. Run on CPU: bash '+str(args.output/'run_build.sh'))


if __name__=='__main__':
    main()
