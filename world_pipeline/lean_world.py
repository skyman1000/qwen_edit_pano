"""Minimal source catalog for the existing GT exporter, not a full world state.

Skips isolated-object raycasts, visibility ratios and heuristic relations.
Actual Local/ERP visibility is still computed by prepare_gt with full-scene rays.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import read, sha
from .checkpoint_io import atomic_json

VERSION = 'encoder_source_catalog_v1'


def catalog(linked, geo):
    excluded = set(geo['room_supervision_excluded_house_object_ids'])
    result = []
    for obj in linked['objects']:
        axes = np.asarray(obj['axes_world_source'], dtype=float)
        axes = np.vstack([axes, np.cross(axes[0], axes[1])])
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        result.append(dict(id=obj['instance_id'], house_object_id=obj['house_object_id'],
            category_id=obj['mpcat40_id'], center_world=obj['center_world'],
            axes_world=axes.tolist(), radii=obj['radii'], region_id=obj['region_id'],
            room_supervision_eligible=obj['house_object_id'] not in excluded and obj['region_id'] >= 0,
            supervision_mask=dict(category=obj['semantic_valid'], position=True, size=True, existence=True)))
    if len({o['id'] for o in result}) != len(result):
        raise ValueError('Duplicate linked instance ID')
    return result


def build(index, geometry, output):
    geo = read(geometry/'geometry_report.json')
    if (geo['status'] != 'BUILT_FOR_REVIEW' or not all(geo['checks'].values()) or
            geo['review_flags'] or geo['height'] != 512 or geo['erp_sampling'] != 'pixel_centers'):
        raise ValueError('Geometry checks/flags unresolved; lean mode does not relax quality gates')
    sample = next(r for r in map(json.loads, (index/'panorama_index.jsonl').read_text().splitlines())
                  if r['sample_id'] == geo['sample_id'])
    linked = read(sample['object_links_path'])
    house_path = Path(linked['house_directory'])/'house.json'
    house = read(house_path)
    room = next(r for r in house['regions'] if r['id'] == sample['region_id'])
    objects = catalog(linked, geo)
    basis = np.asarray(geo['erp_to_world_basis_candidate'])
    origin = np.asarray(geo['origin_world_candidate'])
    pose = np.eye(4); pose[:3,:3] = basis; pose[:3,3] = origin
    inputs = [index/'panorama_index.jsonl', geometry/'geometry_report.json',
              Path(sample['object_links_path']), house_path]
    metadata = dict(version=VERSION, source_sha256={str(p.resolve()): sha(p) for p in inputs},
        omitted=['isolated_projection', 'visibility_ratio', 'heuristic_relations', 'floor_and_ceiling_estimation'],
        visibility='Recomputed by existing prepare_gt full-scene raycasts; not inferred from boxes',
        note='Source catalog only. No claim of complete room state or training approval.')
    state = dict(sample_id=sample['sample_id'], source_profile=metadata,
        camera=dict(position_world=origin.tolist(), camera_to_world=pose.tolist()),
        room=dict(region_id=sample['region_id'], type=sample['region_type'], extent=room['bbox_world'],
                  supervision_eligible=sample['room_supervision_eligible']),
        layout=dict(floor_z=None, ceiling_z=None, unknown_reason='Not computed in encoder-only source profile'),
        relations=[], ready_for_training=False)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'world_state.json', state)
    atomic_json(output/'objects.json', dict(objects=objects, source_profile=metadata))
    atomic_json(output/'build_report.json', dict(status='BUILT', sample_id=sample['sample_id'],
        source_profile=VERSION, source_object_count=len(objects), ready_for_training=False,
        note='Minimal exporter source, NOT a complete legacy world state'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    build(args.index, args.geometry, args.output)


if __name__ == '__main__':
    main()
