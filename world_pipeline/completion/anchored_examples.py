"""Training-only camera-frame demonstrations, retrieved by visible furniture.

No target GT is read. This adapts LayoutGPT's matched demonstrations to the
project's observation/offset contract, not its known-floor-plan protocol.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from ..common import FRAME, new_output, read, rows, sha, write
from .anchored_completion import PROTOCOL as OUTPUT_PROTOCOL
from .box_layout import frustum_margin
from .geometry import CANONICAL_CATEGORIES
from .layout_examples import DEFAULT_MANIFEST

PROTOCOL = 'training_camera_anchor_demonstrations_v1'
# Furniture/fixtures for short demonstrations. These sets do NOT restrict
# observation or completion categories. Retrieval avoids generic lights/art.
DEMO_CATEGORIES = set(CANONICAL_CATEGORIES)-{8,20,38,39,40}
MATCH_CATEGORIES = {3,5,7,10,11,13,14,15,18,19,23,25,26,31,33,34,36,37}


def example(row, graph, hfov):
    if row['split'] != 'train' or graph['split'] != 'train' or graph['frame'] != FRAME:
        return None
    visible, hidden = [], []
    tangent = math.tan(math.radians(hfov)/2)
    for obj in graph['objects']:
        if (obj['category_id']['value'] not in DEMO_CATEGORIES or
                obj.get('in_primary_room', {}).get('value') is not True or
                not obj['center_local_m']['valid'] or not obj['size_aabb_local_m']['valid']):
            continue
        if obj['local_evidence'] == 'visible' and obj['bbox2d_visible_xyxy_norm']['valid']:
            visible.append(obj)
        elif (obj['local_evidence'] == 'unobserved' and obj.get('visibility', {}).get('erp_pixels',0) >= 16 and
              frustum_margin(np.array(obj['center_local_m']['value']),np.array(obj['size_aabb_local_m']['value']),
                             tangent,tangent) <= 0):
            hidden.append(obj)
    # These are explicitly furniture-level subsets to fit the 8B context;
    # counts are demonstration budgets, not constraints on predicted scenes.
    order = lambda o: (-float(np.prod(o['size_aabb_local_m']['value'])),o['track_id'])
    visible = sorted(visible,key=order)[:12]
    hidden = sorted(hidden,key=order)[:6]
    if not visible or not hidden:
        return None
    observed = [dict(id=f'observed_{i:04d}', category=CANONICAL_CATEGORIES[o['category_id']['value']],
                     center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value'])
                for i,o in enumerate(visible)]
    added = []
    for i,obj in enumerate(hidden):
        centre = np.array(obj['center_local_m']['value'])
        anchor = min(observed,key=lambda o:float(np.linalg.norm(centre-o['center'])))
        cat = CANONICAL_CATEGORIES[obj['category_id']['value']]
        added.append(dict(id=f'new_{i:04d}',category=cat,size=obj['size_aabb_local_m']['value'],
                          anchor_id=anchor['id'], offset_m=(centre-anchor['center']).tolist(),
                          description=cat,unseen_reason='outside_fov',
                          reason='Additional object in the same training room outside this camera view.'))
    return dict(sample_id=row['id'], building=row['paired']['scene_id'], split='train',
        image_sha256=row['paired']['local_condition']['sha256'],graph=row['graph'],graph_sha256=row['graph_sha256'],
        source_object_ids=dict(observed=[o['track_id']for o in visible],additions=[o['track_id']for o in hidden]),
        match_categories=sorted({o['category_id']['value']for o in visible}&MATCH_CATEGORIES),
        input=dict(hfov_deg=hfov,fixed_observed=observed),
        output=dict(room_type='indoor room',scene_hypothesis='Furniture-level continuation of this independent training room.',
                    additions=added))


def prepare(manifest, output, hfov=90.):
    records = rows(manifest)
    heldout = {r['paired']['scene_id']for r in records if r['split']!='train'}
    pool = []
    vocab_hash = None
    for row in records:
        if row['split'] != 'train' or row['paired']['scene_id'] in heldout:
            continue
        if sha(row['graph']) != row['graph_sha256']:
            raise ValueError('Training graph changed: '+row['id'])
        graph = read(row['graph'])
        if vocab_hash is not None and graph['vocabulary_sha256'] != vocab_hash:
            raise ValueError('Mixed training vocabularies')
        vocab_hash = graph['vocabulary_sha256']
        item = example(row,graph,hfov)
        if item is not None and item['match_categories']:
            pool.append(item)
    if not pool:
        raise ValueError('No suitable training demonstrations; no validation fallback')
    directory = new_output(output)
    bank = dict(protocol=PROTOCOL, output_protocol=OUTPUT_PROTOCOL, frame=FRAME,
                vocabulary_sha256=vocab_hash,manifest=str(Path(manifest).resolve()),manifest_sha256=sha(manifest),
                hfov_deg=hfov, examples=pool, training_gt_examples=True, target_scene_gt_used=False,
                selection='visible_furniture_category_overlap; no measured target floor plan',
                note='Furniture subsets, not complete GT rooms or an evaluation target')
    write(directory/'pool.json',bank)
    print(dict(examples=len(pool),buildings=len({e['building']for e in pool}),
               output=str(directory/'pool.json'),gpu_executed=False))


def select(path, vocabulary_hash, hfov, image_sha256, observed, count=3):
    bank = read(path)
    if (bank.get('protocol') != PROTOCOL or bank.get('output_protocol') != OUTPUT_PROTOCOL or
            bank.get('frame') != FRAME or bank.get('vocabulary_sha256') != vocabulary_hash or
            not isinstance(bank.get('examples'),list) or not math.isclose(bank['hfov_deg'],hfov,abs_tol=1e-9)):
        raise ValueError('Anchored example bank protocol/frame/vocabulary/FOV mismatch')
    if sha(bank['manifest']) != bank['manifest_sha256']:
        raise ValueError('Training example manifest changed')
    records = {r['id']:r for r in rows(bank['manifest'])}
    heldout = {r['paired']['scene_id']for r in records.values() if r['split']!='train'}
    target = {cid for cid,name in CANONICAL_CATEGORIES.items() if any(o['category']==name for o in observed)}&MATCH_CATEGORIES
    ranked = []
    for item in bank['examples']:
        cats = set(item['match_categories'])
        overlap = len(cats&target)
        if not overlap or item['image_sha256']==image_sha256:
            continue
        score = overlap/len(cats|target)
        ranked.append((-score,-overlap,item['sample_id'],item))
    selected, buildings = [], set()
    for _,_,_,item in sorted(ranked):
        if item['building'] in buildings:
            continue
        row = records[item['sample_id']]
        if (row['split']!='train' or item['split']!='train' or row['paired']['scene_id'] in heldout or
                item['building']!=row['paired']['scene_id'] or item['graph']!=row['graph'] or
                item['graph_sha256']!=row['graph_sha256'] or sha(row['graph'])!=row['graph_sha256']):
            raise ValueError('Demonstration is not bound to an independent training source')
        if example(row,read(row['graph']),hfov) != item:
            raise ValueError('Demonstration differs from source training geometry')
        selected.append(item); buildings.add(item['building'])
        if len(selected)==count:
            break
    return dict(protocol=PROTOCOL, output_protocol=OUTPUT_PROTOCOL, frame=FRAME,
                pool_sha256=sha(path),examples=selected, target_scene_gt_used=False,
                selection=bank['selection'], no_match=not selected)


def messages(bank):
    result = []
    for item in bank['examples']:
        context = deepcopy(item['input'])
        answer = deepcopy(item['output'])
        original = {o['id']:o for o in item['input']['fixed_observed']}
        compact = {o['id']:o for o in context['fixed_observed']}
        for obj in compact.values():
            obj['center'] = [round(v,3) for v in obj['center']]
            obj['size'] = [float(f'{v:.3g}') for v in obj['size']]
        for obj in answer['additions']:
            ref = original[obj['anchor_id']]
            centre = np.array(ref['center'])+obj['offset_m']
            # Recompute from the displayed anchor, so shortened examples still
            # reconstruct the displayed target, rather than accumulating error.
            obj['offset_m'] = [round(round(v,3)-a,3) for v,a in zip(centre,compact[obj['anchor_id']]['center'])]
            obj['size'] = [float(f'{v:.3g}') for v in obj['size']]
        question = ('Independent TRAINING example, a furniture-level subset, not the current room. '
            'Fixed boxes and offsets are in camera METRES (x right, y image-up, z forward); '
            'sizes are FULL camera-axis AABBs. Complete around the fixed instances using anchor_id/offset_m. '
            'Demonstration coordinates are displayed to millimetres, not a claim of prediction precision. '
            +json.dumps(context,ensure_ascii=False,separators=(',',':')))
        result += [dict(role='user',content=question),
                   dict(role='assistant',content=json.dumps(answer,ensure_ascii=False,separators=(',',':')))]
    return deepcopy(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.manifest,args.output)


if __name__=='__main__':
    main()
