"""Bind the existing six validation IDs to the newly created bundle."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from ..common import read, rows, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args()
    original = Path('qwen_edit_pano/data/oracle_indoor6_selection_v1.json')
    selection = read(original)
    old_bundle = Path('qwen_edit_pano/data/oracle_supplement_epoch006_v1')
    if selection['bundle_sha256'] != sha(old_bundle / 'BUNDLE.json'):
        raise ValueError('Original selection binding changed')
    # Same data/GT/features; only frozen Panorama weights are being changed.
    old = rows(old_bundle / 'samples.jsonl')
    new = rows(args.bundle / 'samples.jsonl')
    if new != old:
        raise ValueError('New oracle bundle must preserve the entire old sample list exactly')
    old_config = read(old_bundle / 'BUNDLE.json')
    new_config = read(args.bundle / 'BUNDLE.json')
    for key in ('feature_version', 'base_snapshot', 'height', 'padding_columns', 'max_objects'):
        if old_config[key] != new_config[key]:
            raise ValueError('Changed oracle contract: ' + key)
    selection.update(bundle_sha256=sha(args.bundle / 'BUNDLE.json'),
                     source_selection_sha256=sha(original),
                     source_selection=str(original))
    write(args.bundle / 'indoor6_selection.json', selection)
    # Fixed pre-generation diagnostic sets, stratified by held-out building.
    # These broader sets are not claimed to be indoor-only.
    for split, limit, filename in [('train',8,'train8_selection.json'),('val',24,'val24_selection.json')]:
        houses=defaultdict(list)
        for entry in new:
            if entry['split']==split:
                houses[entry['paired']['scene_id']].append(entry['id'])
        for values in houses.values():
            values.sort(key=lambda sid: hashlib.sha256(('world-val-v2:'+sid).encode()).hexdigest())
        ids=[]
        for offset in range(max(map(len,houses.values()))):
            for house in sorted(houses):
                if offset<len(houses[house]) and len(ids)<limit:ids.append(houses[house][offset])
        lookup={e['id']:e for e in new}
        shuffled={sid:next(other for other in ids[i+1:]+ids[:i]
                          if lookup[other]['paired']['scene_id']!=lookup[sid]['paired']['scene_id'])
                  for i,sid in enumerate(ids)}
        write(args.bundle/filename,dict(version='building_stratified_diagnostic_v2',
              bundle_sha256=sha(args.bundle/'BUNDLE.json'),split=split,ids=ids,shuffled_sources=shuffled,
              selection_rule='Deterministic building-stratified selection before generation; not indoor-only'))
    print(dict(train=new_config['train_count'],val=new_config['val_count'],
               selection=str(args.bundle / 'indoor6_selection.json'),same_gt_and_pairs=True))


if __name__ == '__main__':
    main()
