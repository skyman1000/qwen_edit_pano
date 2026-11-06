"""CPU preparation of training-room demonstrations; never reads target-scene GT.

LayoutGPT-inspired ICL, adapted to this project's camera-AABB contract. No
training, meshes, room-coordinate conversion, target GT or layout correction.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from ..common import FRAME, ROOT, new_output, read, rows, sha, write
from .box_layout import frustum_margin
from .geometry import CANONICAL_CATEGORIES

PROTOCOL = 'training_room_camera_aabb_demonstrations_v1'
DEFAULT_MANIFEST = ROOT/'qwen_edit_pano/data/oracle_supplement_epoch015_scale05_v1/samples.jsonl'
# Furniture-level examples keep ICL short; this does not restrict predictions.
EXAMPLE_CATEGORIES = {3, 5, 6, 7, 9, 10, 11, 13, 14, 19, 22, 28, 31}
MAJOR_ADDITIONS = {3, 5, 7, 13, 19, 31}


def compact_object(obj, oid):
    return dict(id=oid, category=CANONICAL_CATEGORIES[obj['category_id']['value']],
                center=[round(v, 2) for v in obj['center_local_m']['value']],
                size=[round(v, 2) for v in obj['size_aabb_local_m']['value']])


def room_example(row, graph, hfov):
    """Use primary-room objects only, excluding neighbouring-room GT clutter."""
    if graph['split'] != 'train' or graph['frame'] != FRAME or graph['room']['type']['value'] != 'b':
        return None
    observed, additions, source_ids = [], [], dict(observed=[], additions=[])
    tangent = math.tan(math.radians(hfov)/2)
    for obj in graph['objects']:
        if (obj.get('in_primary_room', {}).get('value') is not True or
                obj['category_id']['value'] not in EXAMPLE_CATEGORIES or
                not obj['center_local_m']['valid'] or not obj['size_aabb_local_m']['valid']):
            continue
        item = compact_object(obj, 'unused')
        if min(item['size']) <= 0:
            continue
        if obj['local_evidence'] == 'visible':
            if not obj['bbox2d_visible_xyxy_norm']['valid']:
                continue
            item['id'] = f'observed_{len(observed):04d}'
            observed.append(item)
            source_ids['observed'].append(obj['track_id'])
        elif (obj.get('visibility', {}).get('erp_pixels', 0) > 0 and
              frustum_margin(np.array(item['center']), np.array(item['size']), tangent, tangent) <= 0):
            item.update(id=f'new_{len(additions):04d}', description=item['category'],
                reason='Additional furniture or fixture in the same bedroom outside this partial camera view.',
                unseen_reason='outside_fov')
            additions.append(item)
            source_ids['additions'].append(obj['track_id'])
    # Keep complete furniture subsets of manageable size, rather than truncate.
    if not (1 <= len(observed) <= 14 and 2 <= len(additions) <= 6 and
            any(o['category'] == 'bed' for o in observed)):
        return None
    return dict(sample_id=row['id'], building=row['paired']['scene_id'], split='train',
        graph=str(Path(row['graph']).resolve()), graph_sha256=row['graph_sha256'],
        image_sha256=row['paired']['local_condition']['sha256'], source_object_ids=source_ids,
        input=dict(hfov_deg=hfov, fixed_observed=observed),
        output=dict(room_type='bedroom', scene_hypothesis='A bedroom completed around its existing furniture.',
                    additions=additions))


def prepare(manifest, output, count=3):
    records = rows(manifest)
    heldout_buildings = {r['paired']['scene_id'] for r in records if r['split'] != 'train'}
    candidates = []
    for row in records:
        if row['split'] != 'train' or row['paired']['scene_id'] in heldout_buildings:
            continue
        graph = read(row['graph'])
        example = room_example(row, graph, 90.)
        if example is None:
            continue
        obs, added = example['input']['fixed_observed'], example['output']['additions']
        major_count = len({o['category'] for o in added} & {CANONICAL_CATEGORIES[k] for k in MAJOR_ADDITIONS})
        score = (sum(o['category'] == 'bed' for o in obs) != 1,
                 -major_count, abs(len(obs)-7)+abs(len(added)-3), row['id'])
        candidates.append((score, example))
    selected, buildings = [], set()
    for _, example in sorted(candidates, key=lambda pair: pair[0]):
        if example['building'] in buildings:
            continue
        if sha(example['graph']) != example['graph_sha256']:
            raise ValueError('Training graph differs from its manifest: '+example['sample_id'])
        selected.append(example)
        buildings.add(example['building'])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Insufficient independent training bedrooms; no validation fallback')
    directory = new_output(output)
    bank = dict(protocol=PROTOCOL, frame=FRAME, room_type='bedroom',
        vocabulary_sha256=read(selected[0]['graph'])['vocabulary_sha256'],
        manifest=str(Path(manifest).resolve()), manifest_sha256=sha(manifest), examples=selected,
        note='Training GT furniture examples, not target GT or exhaustive rooms; '
             'camera xyz/full AABB retained, no room bounds or floor supplied.')
    write(directory/'bedroom.json', bank)
    print(json.dumps(dict(status='prepared', output=str(directory/'bedroom.json'),
                         training_examples=[e['sample_id'] for e in selected], gpu_used=False)))


def load(path, vocabulary_hash, hfov, target_image_sha256):
    bank = read(path)
    if (bank.get('protocol') != PROTOCOL or bank.get('frame') != FRAME or
            bank.get('vocabulary_sha256') != vocabulary_hash or
            not isinstance(bank.get('examples'), list) or not 1 <= len(bank['examples']) <= 6):
        raise ValueError('Layout examples have a different protocol/frame/vocabulary or invalid count')
    if sha(bank['manifest']) != bank['manifest_sha256']:
        raise ValueError('Training example manifest changed')
    records = {r['id']: r for r in rows(bank['manifest'])}
    heldout = {r['paired']['scene_id'] for r in records.values() if r['split'] != 'train'}
    buildings = set()
    for example in bank['examples']:
        row = records[example['sample_id']]
        if (row['split'] != 'train' or example['split'] != 'train' or
                row['paired']['scene_id'] in heldout or example['building'] in buildings or
                example['building'] != row['paired']['scene_id']):
            raise ValueError('Examples must be independent training buildings, disjoint from validation')
        if example['image_sha256'] == target_image_sha256:
            raise ValueError('Target photograph cannot be its own GT demonstration')
        if (Path(example['graph']).resolve() != Path(row['graph']).resolve() or
                example['graph_sha256'] != row['graph_sha256'] or
                sha(example['graph']) != example['graph_sha256']):
            raise ValueError('Training example graph provenance differs from the manifest')
        if not math.isclose(example['input']['hfov_deg'], hfov, abs_tol=1e-9):
            raise ValueError('Example FOV differs; do not reinterpret camera coordinates')
        # Recompute the demonstrated coordinates, labels, and visibility from
        # source data rather than trust an AI-written bank or its metadata.
        if room_example(row, read(row['graph']), hfov) != example:
            raise ValueError('Demonstration differs from its source training room')
        buildings.add(example['building'])
    return bank


def messages(bank):
    result = []
    for i, example in enumerate(bank['examples'], 1):
        question = ('INDEPENDENT TRAINING ROOM EXAMPLE '+str(i)+'. '
            'Camera origin, x right, y image-up, z forward, metres; size is FULL camera-axis AABB. '
            'These are furniture-level subsets. Complete around the fixed observed objects. '
            +json.dumps(example['input'], ensure_ascii=False, separators=(',', ':')))
        result.extend([dict(role='user', content=question),
                       dict(role='assistant', content=json.dumps(example['output'], ensure_ascii=False, separators=(',', ':')))])
    return deepcopy(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)


if __name__ == '__main__':
    main()
