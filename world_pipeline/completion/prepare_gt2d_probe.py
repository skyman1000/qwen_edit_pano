"""Prepare a controlled visible-geometry probe: GT category/2D only, no GT 3D."""
import argparse
from pathlib import Path

from ..common import read, rows, write
from .geometry import CANONICAL_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    refs = {r['id']: r for r in rows(args.bundle/'references.jsonl')}
    total = 0
    for entry in rows(args.bundle/'inputs.jsonl'):
        state = read(refs[entry['id']]['gt_obs'])
        instances = []
        for i, obj in enumerate(state['objects']):
            category = CANONICAL_CATEGORIES[obj['category_id']['value']]
            box = obj['bbox2d_visible_xyxy_norm']
            if not box['valid']:
                raise ValueError(f'{entry["id"]}: GT visible 2D box unavailable')
            instances.append(dict(id=f'obs_{i:04d}', category=category,
                bbox_2d=[float(v)*1000 for v in box['value']],
                description=f'{category} in the specified image region'))
        directory = args.output/entry['id']
        directory.mkdir(parents=True, exist_ok=True)
        write(directory/'instances.normalized.json', instances)
        write(directory/'grounding_audit.json', {})
        write(directory/'status.json', dict(status='completed', issues=[],
              source='GT category and visible 2D boxes only; geometry intentionally omitted'))
        (directory/'description.txt').write_text('')
        total += len(instances)
    print(f'Prepared {len(refs)} images, {total} GT 2D instances at {args.output}; no GT 3D exported')


if __name__ == '__main__':
    main()
