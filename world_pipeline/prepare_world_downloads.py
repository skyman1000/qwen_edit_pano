"""Classify local assets for the polished, single-Local minimal World build.

CPU/local metadata only. Never downloads, hashes, reads full ZIP payloads,
checks CRCs, imports models, or launches the legacy full-state builder.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT/'qwen_edit_pano/data/world_polished_full_candidates_v2'
# The minimal path needs RGB camera registration and semantic mesh projection.
# Sensor-depth/normal export and the legacy canonical full-asset gate are omitted.
BUILD_TYPES = (
    'matterport_skybox_images', 'house_segmentations', 'region_segmentations',
    'matterport_camera_intrinsics', 'matterport_camera_poses',
    'undistorted_camera_parameters', 'undistorted_color_images',
)


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare(candidates, raw_root, output):
    groups, splits = {}, {}
    for split in ('train', 'val'):
        for row in read_rows(candidates/f'{split}.jsonl'):
            house = row['scene_id']
            if row['split'] != split or splits.setdefault(house, split) != split:
                raise ValueError(f'Building split mismatch: {house}')
            groups.setdefault(house, []).append(row)
    inventory, source_errors, missing_counts = [], [], Counter()
    for house, selected in sorted(groups.items()):
        scan = raw_root/house
        present = {kind: (scan/(kind+'.zip')).is_file() and
                   (scan/(kind+'.zip')).stat().st_size > 0 for kind in BUILD_TYPES}
        missing = [kind for kind, exists in present.items() if not exists]
        missing_counts.update(missing)
        item = dict(scene_id=house, split=splits[house], candidates=len(selected),
                    present_types=[kind for kind, exists in present.items() if exists],
                    missing_types=missing,
                    partial_files=[p.name for p in sorted(scan.glob('*.zip.part'))])
        inventory.append(item)
        members = set()
        if present['matterport_skybox_images']:
            try:
                with zipfile.ZipFile(scan/'matterport_skybox_images.zip') as archive:
                    members = set(archive.namelist())
            except (OSError, zipfile.BadZipFile) as exc:
                source_errors.append(dict(scene_id=house, reason='skybox_directory_unreadable', error=str(exc)))
        for row in selected:
            if row['local_condition']['member'] not in members:
                source_errors.append(dict(id=row['id'], reason='missing_Local_ZIP_member'))
    # Check each Arrow path once; do not decode/copy/export the images.
    targets = {row['image']['arrow_file'] for selected in groups.values() for row in selected}
    missing_targets = sorted(path for path in targets if not Path(path).is_file() or Path(path).stat().st_size==0)
    missing_houses = [item['scene_id'] for item in inventory if item['missing_types']]
    summary = dict(
        profile='polished_face2_objects_and_visible_instance_regions_v2',
        candidates={split:sum(len(selected) for h, selected in groups.items() if splits[h]==split)
                    for split in ('train', 'val')},
        buildings={split:dict(total=sum(s==split for s in splits.values()),
                              ready=sum(item['split']==split and not item['missing_types'] for item in inventory),
                              missing=sum(item['split']==split and bool(item['missing_types']) for item in inventory))
                   for split in ('train', 'val')},
        required_types=list(BUILD_TYPES), omitted_legacy_types=['undistorted_depth_images','undistorted_normal_images'],
        missing_zip_files=sum(missing_counts.values()), missing_by_type=dict(missing_counts),
        missing_house_count=len(missing_houses), polished_arrow_files=len(targets),
        missing_polished_files=missing_targets, local_source_errors=source_errors,
        ready_for_minimal_build=not missing_houses and not missing_targets and not source_errors,
        ready_for_full_training=False, hashes_computed=False, crc_checked=False,
        inspection='ZIP filenames/file presence and Local ZIP member directories; no full payload validation',
        note='This dependency profile is for the forthcoming minimal builder. Do not run the legacy '
             'full-state runner with these reduced assets. Object states and region masks still need '
             'construction even after all downloads are present.',
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'SUMMARY.json', summary)
    write_json(output/'assets_by_house.json', inventory)
    (output/'missing_houses.txt').write_text(''.join(h+'\n' for h in missing_houses))
    for split in ('train', 'val'):
        (output/f'missing_{split}_houses.txt').write_text(''.join(
            item['scene_id']+'\n' for item in inventory if item['split']==split and item['missing_types']))
    if missing_houses:
        types = [kind for kind in BUILD_TYPES if missing_counts[kind]]
        # The existing downloader owns interactive ToS confirmation, retries,
        # transfer resume and skipping of already present archives.
        cmd = ('"$world_download_python" -u download_mp_py3.py '
               '-o '+shlex.quote(str(raw_root.parent.parent))+' '
               '--scan-list '+shlex.quote(str((output/'missing_houses.txt').resolve()))+' '
               '--type '+' '.join(types)+' --retries 8 --timeout 60 --retry-delay 5')
        script = ('#!/usr/bin/env bash\nset -euo pipefail\n'
                  'cd '+shlex.quote(str(ROOT))+'\n'
                  'world_download_python="${WORLD_DATA_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"\n'
                  '# User-run only. No --verify-existing, hash computation, or GPU work.\n'+cmd+'\n')
    else:
        script = '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "No required ZIP downloads are missing; inspect SUMMARY.json for source errors."\n'
    (output/'download_missing.sh').write_text(script)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument('--raw-root', type=Path, default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    parser.add_argument('--output', type=Path, default=DEFAULT_CANDIDATES/'download_plan')
    args = parser.parse_args()
    print(json.dumps(prepare(args.candidates.resolve(), args.raw_root.resolve(), args.output.resolve()),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
