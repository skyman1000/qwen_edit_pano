"""Plan/build all paired GT sources using existing Matterport CPU builders.

No network or GPU work. Outputs and checkpoints live in a NEW work root.
Existing validated legacy stages are linked read-only, never written through.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, read, rows, sha
from .checkpoint_io import atomic_json, output_lock, archive_partial, file_hashes, verify_files

ARCHIVES = ('matterport_skybox_images', 'house_segmentations', 'region_segmentations',
            'matterport_camera_intrinsics', 'matterport_camera_poses',
            'undistorted_camera_parameters', 'undistorted_color_images',
            'undistorted_depth_images', 'undistorted_normal_images')
GROUPS = ('camera', 'house', 'region', 'object_link', 'canonical')


def paired_selection(pair_root):
    result, houses, ids = [], {}, set()
    for split in ('train', 'val'):
        official = {r['id']: r for r in rows(ROOT/f'qwen_edit_pano/data/official_building_split/{split}.full.jsonl')}
        for row in rows(pair_root/f'{split}.jsonl'):
            if row['id'] in ids or houses.setdefault(row['scene_id'], split) != split:
                raise ValueError('Duplicate sample or building split leakage')
            expected = official.get(row['id'])
            if expected is None or any(row[k] != expected[k] for k in ('scene_id', 'source_view_id')):
                raise ValueError('Pair differs from official building split')
            ids.add(row['id'])
            result.append(dict(id=row['id'], scene_id=row['scene_id'], source_view_id=row['source_view_id'], split=split))
    return sorted(result, key=lambda r: (r['scene_id'], r['id']))


def source_stats(scan):
    return {name: dict(size=(scan/f'{name}.zip').stat().st_size,
                       mtime_ns=(scan/f'{name}.zip').stat().st_mtime_ns) for name in ARCHIVES}


def validate_report(path, status, identity=None):
    report = read(path)
    if report['status'] not in status or report.get('errors') or report.get('review_flags'):
        raise ValueError(f'Stage needs review: {path}')
    if identity and report.get('sample_id') != identity:
        raise ValueError(f'Stage identity mismatch: {path}')
    if path.name == 'geometry_report.json' and (report.get('height') != 512 or
                                               not all(report.get('checks', {}).values())):
        raise ValueError(f'Geometry dimensions/checks mismatch: {path}')


class Builder:
    def __init__(self, args):
        self.args = args
        self.work = args.work_root
        self.state = self.work/'checkpoints'
        self.state.mkdir(exist_ok=True)
        self.logs = self.work/'logs'
        self.logs.mkdir(exist_ok=True)

    def run(self, module, arguments, label):
        log = self.logs/f'{label}.log'
        print(f'Running {module}; log={log}', flush=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(self.args.threads))
        with log.open('a') as handle:
            proc = subprocess.run([sys.executable, '-m', 'qwen_pano.caupano.'+module,
                                   *map(str, arguments)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'{module} exited {proc.returncode}; see {log}')

    def canonical(self, house):
        paths = {k: self.work/f'{k}_pilot'/house for k in GROUPS}
        checkpoint = self.state/f'{house}_canonical.json'
        scan = self.args.raw_root/house
        stats = source_stats(scan)
        if checkpoint.exists():
            saved = read(checkpoint)
            if stats != saved['archives']:
                raise ValueError(f'Raw archives changed for {house}; use a new work-root')
            verify_files(self.work, saved['files'])
            return paths['canonical']
        # Legacy canonical indices have already consumed all five parser stages.
        old = self.args.reuse_root
        can_reuse = (old/'canonical_pilot'/house/'index_validation.json').is_file()
        if can_reuse:
            validate_report(old/'canonical_pilot'/house/'index_validation.json', {'PASS'})
            legacy_manifest = read(old/'canonical_pilot'/house/'index_manifest.json')
            if legacy_manifest['scan_id'] != house:
                raise ValueError('Legacy canonical identity mismatch')
            verify_files(Path('/'), legacy_manifest['source_sha256'])
            if Path(read(old/'camera_pilot'/house/'parse_summary.json')['scan_dir']).resolve() != scan.resolve():
                raise ValueError('Legacy canonical uses another raw directory')
            for k, path in paths.items():
                src = old/f'{k}_pilot'/house
                if not src.is_dir():
                    raise ValueError(f'Incomplete legacy canonical dependency: {src}')
                if path.is_symlink() and path.resolve() == src.resolve():
                    continue
                archive_partial(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(src.resolve(), target_is_directory=True)
        else:
            for path in paths.values():
                archive_partial(path)
            def run(module, args):
                self.run(module, args, house+'_'+module.rsplit('.', 1)[-1])
            mapping = ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv'
            for k in ('camera', 'house'):
                run(f'data.matterport.{k}_parser', ['--scan-dir', scan, '--output', paths[k]])
            counts = read(paths['house']/'parse_summary.json')['declared_counts']
            run('tools.validate_cameras', ['--input', paths['camera'], '--expected-panoramas', counts['panoramas'], '--expected-observations', counts['images']])
            run('tools.validate_house', ['--input', paths['house'], '--cameras', paths['camera'], '--category-mapping', mapping])
            run('data.matterport.region_parser', ['--house', paths['house'], '--category-mapping', mapping, '--output', paths['region']])
            run('tools.validate_regions', ['--input', paths['region'], '--category-mapping', mapping])
            run('data.matterport.object_linker', ['--regions', paths['region'], '--category-mapping', mapping, '--output', paths['object_link']])
            run('tools.validate_object_links', ['--input', paths['object_link']])
            run('data.canonical_index', ['--cameras', paths['camera'], '--objects', paths['object_link'], '--output', paths['canonical']])
            run('tools.validate_canonical_index', ['--input', paths['canonical'], '--expected-panoramas', counts['panoramas'], '--expected-observations', counts['images']])
        validate_report(paths['canonical']/'index_validation.json', {'PASS'})
        files = {str(path.relative_to(self.work)/name): digest for path in paths.values() for name, digest in file_hashes(path).items()}
        atomic_json(checkpoint, dict(archives=stats, files=files))
        return paths['canonical']

    def sample(self, row, index):
        house, uuid = row['scene_id'], row['source_view_id']
        available = {json.loads(s)['sample_id'] for s in (index/'panorama_index.jsonl').read_text().splitlines()}
        if row['id'] not in available:
            raise ValueError('Paired identity missing from canonical index')
        alignment = self.work/'erp_alignment_pilot'/house/uuid
        geometry = self.work/'geometry_pilot'/house/uuid
        world = self.work/'world_state_pilot'/house/uuid
        stages = [
            (alignment, 'alignment_report.json', {'CANDIDATE_REQUIRES_VISUAL_REVIEW'}, 'tools.erp_alignment_pilot',
             ['--index', index, '--panorama-uuid', uuid, '--output', alignment]),
            (geometry, 'geometry_report.json', {'BUILT_FOR_REVIEW'}, 'data.build_geometry_pilot',
             ['--index', index, '--alignment', alignment, '--output', geometry, '--height', 512, '--threads', self.args.threads]),
            (world, 'build_report.json', {'BUILT'}, 'data.world_state_builder',
             ['--index', index, '--geometry', geometry, '--output', world, '--threads', self.args.threads])]
        for directory, report, statuses, module, arguments in stages:
            marker = self.state/f'{row["id"]}_{directory.parts[-3]}.json'
            if marker.exists():
                verify_files(directory, read(marker)['files'])
                validate_report(directory/report, statuses, row['id'])
                continue
            old = self.args.reuse_root/directory.relative_to(self.work)
            if (old/report).is_file():
                validate_report(old/report, statuses, row['id'])
                if directory.is_symlink() and directory.resolve() == old.resolve():
                    pass
                else:
                    archive_partial(directory)
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    directory.symlink_to(old.resolve(), target_is_directory=True)
            else:
                archive_partial(directory)
                self.run(module, arguments, row['id']+'_'+module.rsplit('.', 1)[-1])
            validate_report(directory/report, statuses, row['id'])
            atomic_json(marker, dict(files=file_hashes(directory)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=ROOT/'qwen_edit_pano/data/paired_full_v1')
    p.add_argument('--raw-root', type=Path, default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    p.add_argument('--reuse-root', type=Path, default=ROOT/'qwen_pano/outputs/caupano')
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--build', action='store_true', help='Run CPU stages; otherwise only write inventory and scan lists')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    for k in ('pairs', 'raw_root', 'reuse_root', 'work_root'):
        setattr(args, k, getattr(args, k).resolve())
    if args.threads < 1 or not args.work_root.is_relative_to(ROOT/'qwen_edit_pano/data') or args.work_root == ROOT/'qwen_edit_pano/data':
        p.error('Use a dedicated new work-root under qwen_edit_pano/data; threads >= 1')
    if args.work_root.exists() and not args.resume:
        p.error('Work-root exists; use --resume')
    args.work_root.mkdir(parents=True, exist_ok=True)
    with output_lock(args.work_root):
        selected = paired_selection(args.pairs)
        houses = sorted({r['scene_id'] for r in selected})
        signature = dict(pairs=str(args.pairs), raw_root=str(args.raw_root), reuse_root=str(args.reuse_root),
                         rows=selected, code={str(f): sha(f) for f in
                             [Path(__file__), Path(__file__).with_name('checkpoint_io.py'),
                              ROOT/'benchmark_assets/Matterport3D_metadata/official/category_mapping.tsv',
                              ROOT/'qwen_pano/geometry.py',
                              *sorted((ROOT/'qwen_pano/caupano/data').rglob('*.py')),
                              *sorted((ROOT/'qwen_pano/caupano/tools').glob('*.py'))]})
        config = args.work_root/'SOURCE_RUN.json'
        if config.exists() and read(config) != signature:
            if (args.work_root/'checkpoints').exists() or (args.work_root/'build_progress.json').exists():
                raise ValueError('Selection/source/code changed after building began; use a new work-root')
            archive_partial(config)  # only an unexecuted inventory plan existed
        atomic_json(config, signature)
        missing = {h: [k for k in ARCHIVES if not (args.raw_root/h/f'{k}.zip').is_file()] for h in houses}
        report = dict(paired_counts=dict(Counter(r['split'] for r in selected)), houses=len(houses),
                      required_archive_types=list(ARCHIVES), missing_by_house=missing,
                      fully_present_houses=sum(not v for v in missing.values()), ready_for_training=False)
        atomic_json(args.work_root/'inventory.json', report)
        (args.work_root/'scans.txt').write_text('\n'.join(houses)+'\n')
        (args.work_root/'missing_scans.txt').write_text(''.join(h+'\n' for h in houses if missing[h]))
        print({k: v for k, v in report.items() if k != 'missing_by_house'}, flush=True)
        if not args.build:
            return
        builder = Builder(args)
        results = {}
        for house in houses:
            group = [r for r in selected if r['scene_id'] == house]
            if missing[house]:
                results[house] = dict(status='MISSING_ARCHIVES', missing=missing[house])
            else:
                try:
                    index = builder.canonical(house)
                except (ValueError, OSError, RuntimeError, KeyError) as exc:
                    results[house] = dict(status='FAILED_CANONICAL', error=str(exc))
                else:
                    done, failures = [], []
                    for row in group:
                        try:
                            builder.sample(row, index)
                            done.append(row['id'])
                            print(f'WORLD_READY {row["id"]}', flush=True)
                        except (ValueError, OSError, RuntimeError, KeyError) as exc:
                            failures.append(dict(id=row['id'], error=str(exc)))
                        results[house] = dict(status='REVIEW_REQUIRED' if not failures else 'PARTIAL', completed=done, failures=failures)
                        atomic_json(args.work_root/'build_progress.json', results)
            atomic_json(args.work_root/'build_progress.json', results)
        complete = all(r['status'] == 'REVIEW_REQUIRED' for r in results.values())
        atomic_json(args.work_root/'BUILD_RESULT.json', dict(all_requested_built=complete, ready_for_training=False, houses=results))
        if not complete:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
