"""Prepare minimal full-pair candidates and classify local ZIP assets on CPU.

No content hashes, ZIP payload/CRC checks, image extraction, model loading,
downloads, or training. Missing object states remain explicitly unbuilt.
"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = (
    'matterport_skybox_images', 'house_segmentations', 'region_segmentations',
    'matterport_camera_intrinsics', 'matterport_camera_poses',
    'undistorted_camera_parameters', 'undistorted_color_images',
    'undistorted_depth_images', 'undistorted_normal_images',
)


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=ROOT/'qwen_edit_pano/data/paired_full_v1')
    parser.add_argument('--existing-bundle', type=Path,
                        default=ROOT/'qwen_edit_pano/data/oracle_supplement_epoch015_scale05_v1')
    parser.add_argument('--raw-root', type=Path, default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    parser.add_argument('--vocabulary', type=Path,
                        default=ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # This preparation path intentionally does not call legacy hash validators.
    existing = {row['id']: row for row in rows(args.existing_bundle/'samples.jsonl')}
    prepared, houses, ids = {}, {}, set()
    for split in ('train', 'val'):
        selected = []
        for row in rows(args.pairs/f'{split}.jsonl'):
            house = row['scene_id']
            if row['split'] != split or row['id'] in ids:
                raise ValueError(f'Duplicate ID or split mismatch: {row["id"]}')
            ids.add(row['id'])
            if houses.setdefault(house, split) != split:
                raise ValueError(f'Building split overlap: {house}')
            local = row['local_condition']
            if local['face_index'] != 2:
                raise ValueError(f'Expected single face2 Local: {row["id"]}')
            target = row['image']
            if not isinstance(target, dict) or 'matterport3_d_polished' not in target['arrow_file']:
                raise ValueError(f'Expected existing polished pointer: {row["id"]}')
            alignment = row['target_alignment']
            entry = dict(
                id=row['id'], split=split, scene_id=house, source_view_id=row['source_view_id'],
                local_condition={key: local[key] for key in ('archive', 'member', 'face_index')},
                image={key: target[key] for key in ('arrow_file', 'batch_index', 'row_index')},
                target_alignment={key: alignment[key] for key in ('mirror', 'roll_fraction')},
                objects=None, object_region_supervision=None, state_status='UNBUILT',
            )
            old = existing.get(row['id'])
            if old is not None:
                graph_path = Path(old['graph'])
                graph = json.loads(graph_path.read_text())
                if graph['id'] != row['id'] or graph['split'] != split:
                    raise ValueError(f'Existing state identity mismatch: {row["id"]}')
                # Only fields consumed by the current encoder, plus the instance
                # association needed for the requested region loss.
                entry['objects'] = [dict(
                    instance_id=obj['track_id'], category_id=obj['category_id']['value'],
                    center_local_m=obj['center_local_m']['value'],
                    size_aabb_local_m=obj['size_aabb_local_m']['value'],
                    center_valid=obj['center_local_m']['valid'],
                    size_valid=obj['size_aabb_local_m']['valid'],
                ) for obj in graph['objects']]
                mask_path = graph_path.parent/'erp_visibility.npz'
                if mask_path.is_file():
                    entry['object_region_supervision'] = dict(
                        path=str(mask_path), arrays=['instance', 'instance_valid'],
                        instance_id_is_condition=False,
                    )
                entry['state_status'] = 'REUSED_EXISTING_STATE'
            selected.append(entry)
        prepared[split] = selected

    inventory = []
    for house, split in sorted(houses.items()):
        directory = args.raw_root/house
        available = sorted(p.name for p in directory.glob('*.zip'))
        missing = [name+'.zip' for name in ASSETS if not (directory/(name+'.zip')).is_file()]
        inventory.append(dict(scene_id=house, split=split, available_zip=available,
                              missing_legacy_builder_zip=missing))
    summary = dict(
        stage='FULL_POLISHED_CANDIDATES_WITH_PARTIAL_EXISTING_STATES',
        ready_for_full_training=False, local='single_original_skybox_face2', target='polished_ERP',
        spatial_bias=False, content_hash_validation=False,
        counts={split: dict(
            candidates=len(selected), buildings=len({r['scene_id'] for r in selected}),
            existing_states=sum(r['objects'] is not None for r in selected),
            unbuilt_states=sum(r['objects'] is None for r in selected),
        ) for split, selected in prepared.items()},
        asset_buildings={split: dict(
            present=sum(h['split']==split and not h['missing_legacy_builder_zip'] for h in inventory),
            missing=sum(h['split']==split and bool(h['missing_legacy_builder_zip']) for h in inventory),
        ) for split in prepared},
        note='Candidates are not completed GT. ZIP classification is file presence only; '
             'no geometry or mask certification. The listed nine assets are dependencies '
             'of the existing builder, not all encoder input fields. The legacy training '
             'loader is not compatible with these minimal rows; a new loader is required.',
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for split, selected in prepared.items():
        with (args.output/f'{split}.jsonl').open('w') as handle:
            for entry in selected:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    dump(args.output/'zip_assets_by_house.json', inventory)
    dump(args.output/'category_names.json', json.loads(args.vocabulary.read_text())['categories'])
    dump(args.output/'SUMMARY.json', summary)
    for split in prepared:
        (args.output/f'missing_{split}_houses.txt').write_text(''.join(
            h['scene_id']+'\n' for h in inventory
            if h['split']==split and h['missing_legacy_builder_zip']))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
