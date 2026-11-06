"""Read-only audit of saved numerical layouts. No model, GT, repair or download."""
import argparse
from itertools import product
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import FRAME, new_output, read, sha, validate_graph, write
from .geometry import CANONICAL_CATEGORIES


def project_aabb(center, size, hfov, image_size):
    """Convex-box containment in the pinhole frustum, independent of exporter checks.

    An entire box inside the frustum contradicts 'outside_fov'. It does NOT prove
    visibility: the object may still be occluded. A partially intersecting box is
    deliberately not classified as entirely outside by this test.
    """
    if not 0 < hfov < 180 or min(image_size) <= 0:
        raise ValueError('Invalid pinhole camera')
    corners = np.asarray(center) + np.array(list(product((-1., 1.), repeat=3))) * np.asarray(size)/2
    tx = math.tan(math.radians(hfov)/2)
    ty = tx * image_size[1]/image_size[0]
    x, y, z = corners.T
    contained = bool(np.all((z > 0) & (abs(x) <= tx*z) & (abs(y) <= ty*z)))
    bbox = None
    if np.all(z > 0):
        uv = np.column_stack((.5 + x/(2*tx*z), .5 - y/(2*ty*z)))
        bbox = [*uv.min(0).tolist(), *uv.max(0).tolist()]
    return dict(entire_aabb_inside_input_frustum=contained, projected_bbox_xyxy_norm=bbox,
                azimuth_deg=math.degrees(math.atan2(center[0], center[2])))


def audit(source):
    config = read(source/'run_config.json')
    records = []
    for case in config['cases']:
        folder = source/case['id']
        if not (folder/'G_full.json').exists():
            records.append(dict(id=case['id'], status='missing_export'))
            continue
        observed = validate_graph(read(folder/'G_obs.json'))
        full = validate_graph(read(folder/'G_full.json'))
        candidates = validate_graph(read(folder/'G_full_candidates.json'))
        raw = read(folder/'layout.json')['objects']
        planned = {p['id']: p for p in read(folder/'semantic_plan.json')['additions']}
        exported = {o['track_id']: o for o in candidates['objects'][len(observed['objects']):]}
        request = read(folder/'layout.request.json')
        with Image.open(folder/'local.png') as image:
            image_size = image.size
        rows = []
        for obj in raw:
            saved = exported[obj['id']]
            check = project_aabb(obj['center'], obj['size'], case['hfov'], image_size)
            check.update(id=obj['id'], name=planned[obj['plan_id']]['name'], category=obj['category'],
                         center=obj['center'], size=obj['size'], unseen_reason=obj['unseen_reason'],
                         outside_claim_contradicted=obj['unseen_reason']=='outside_fov' and check['entire_aabb_inside_input_frustum'],
                         raw_geometry_preserved=(obj['center']==saved['center_local_m']['value'] and
                                                 obj['size']==saved['size_aabb_local_m']['value']),
                         raw_category_preserved=saved['category_id']['value']=={v:k for k,v in CANONICAL_CATEGORIES.items()}.get(obj['category']))
            rows.append(check)
        count = len(observed['objects'])
        source_hashes = dict(case['source_sha256'])
        source_hashes.update({str(p.resolve()): sha(p) for p in
                             (source/'run_config.json', folder/'layout.json', folder/'G_full.json')})
        records.append(dict(id=case['id'], status='audited', frame=FRAME,
            source_hashes=source_hashes,
            source_files_unchanged=all(sha(Path(p))==h for p,h in case['source_sha256'].items()),
            g_obs_matches_frozen_input=observed==case['observed'],
            observed_prefix_preserved=full['objects'][:count]==observed['objects']==candidates['objects'][:count],
            prompt_preserved=request['prompt']==case['prompt'],
            entire_aabbs_inside=sum(r['entire_aabb_inside_input_frustum'] for r in rows),
            outside_claims_contradicted=sum(r['outside_claim_contradicted'] for r in rows),
            generation=read(folder/'layout.generation.json'), objects=rows,
            semantics_verified=False, occlusion_verified=False,
            note='Inside frustum does not prove visible. Camera image-up is not measured gravity.'))
    return dict(source=str(source.resolve()), model=config['model'], records=records,
                input_mutated=False, model_loaded=False, scene_gt_used=False)


def report(result):
    lines = ['# Numerical layout audit', '',
             'This report checks saved numbers and export provenance; it does not reconstruct a room.', '',
             '| Case | New boxes | Entire AABB inside input frustum | Contradictory outside claims | Raw geometry preserved |',
             '|---|---:|---:|---:|---|']
    for row in result['records']:
        if row['status'] != 'audited':
            lines.append(f'| {row["id"]} | missing export | | | |')
            continue
        lines.append(f'| {row["id"]} | {len(row["objects"])} | {row["entire_aabbs_inside"]} | '
                     f'{row["outside_claims_contradicted"]} | {all(o["raw_geometry_preserved"] for o in row["objects"])} |')
    lines += ['', 'An entire AABB inside the input frustum cannot be outside that frustum. '
              'Occlusion remains a separate, unverified possibility. Canonical category membership '
              'does not verify semantic classification. No coordinates have been changed.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source.resolve())
    output = new_output(args.output)
    write(output/'audit.json', result)
    (output/'REPORT.md').write_text(report(result))
    print(json.dumps(dict(output=str(output), cases=len(result['records']),
                          entire_aabbs_inside=sum(r.get('entire_aabbs_inside',0) for r in result['records']),
                          outside_claims_contradicted=sum(r.get('outside_claims_contradicted',0) for r in result['records']))))


if __name__ == '__main__':
    main()
