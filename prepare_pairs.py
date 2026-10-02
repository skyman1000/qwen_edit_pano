"""Freeze a local subset and produce input/GT-front previews; no GPU/model calls."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from .common import ROOT, atomic_json, check_ids, read_jsonl, sha256, write_jsonl, require_local_output
from .paired_data import local_image, target_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', type=Path, default=ROOT/'qwen_edit_pano/data/official_building_split')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-limit', type=int, default=32, help='0 uses all available training rows')
    p.add_argument('--val-limit', type=int, default=4)
    p.add_argument('--preview-limit', type=int, default=16)
    p.add_argument('--min-alignment-score', type=float, default=.65)
    p.add_argument('--min-alignment-margin', type=float, default=.03)
    args = p.parse_args()
    require_local_output(args.output, 'data')
    if min(args.train_limit, args.val_limit, args.preview_limit) < 0:
        p.error('Limits must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory: frozen manifests must not change during training')
    import numpy as np
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ROOT/'benchmark_assets/PanFusion/external'))
    import py360convert
    all_rows = {split: read_jsonl(args.splits/f'{split}.full.jsonl') for split in ('train', 'val', 'test')}
    houses = {s: {r['scene_id'] for r in rr} for s, rr in all_rows.items()}
    if any(houses[a] & houses[b] for a, b in [('train','val'), ('train','test'), ('val','test')]):
        raise ValueError('Building leakage in split manifests')
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output/'heldout.jsonl', all_rows['val'] + all_rows['test'])
    report = {'counts': {}, 'alignment': 'PENDING_USER_VISUAL_REVIEW', 'yaw_augmentation': False, 'rejected': [],
              'alignment_thresholds': {'score':args.min_alignment_score,'margin':args.min_alignment_margin},
              'source_hashes': {s: sha256(args.splits/f'{s}.full.jsonl') for s in all_rows}}
    for split, limit in [('train', args.train_limit), ('val', args.val_limit)]:
        available = read_jsonl(args.splits/f'{split}.available.jsonl')
        official = {r['id']: r for r in all_rows[split]}
        groups = {}
        for row in available:
            if row['id'] not in official or row['scene_id'] != official[row['id']]['scene_id']:
                raise ValueError('Available inventory differs from official split')
            groups.setdefault(row['scene_id'], []).append(row)
        # Round robin buildings: small smoke manifests are not all one room/house.
        ordered = [groups[h][i] for i in range(max(map(len, groups.values())))
                   for h in sorted(groups) if i < len(groups[h])]
        selected = ordered[:limit] if limit else ordered
        result = []
        from .pair_alignment import estimate_alignment, apply_alignment
        for i, source in enumerate(selected):
            row = {k: v for k, v in source.items() if k not in ('caption', 'original_split')}
            c = dict(row['local_condition'])
            with zipfile.ZipFile(c['archive']) as z:
                raw = z.read(c['member'])
            c['sha256'] = hashlib.sha256(raw).hexdigest()
            row['local_condition'] = c
            local = local_image(row)
            raw_gt = target_image(row)
            alignment = estimate_alignment(local,raw_gt)
            row['target_alignment'] = alignment
            gt = apply_alignment(raw_gt,alignment)
            reliable = alignment['score'] >= args.min_alignment_score and alignment['margin'] >= args.min_alignment_margin
            print(f'{split} {i+1}/{len(selected)} {row["id"]} score={alignment["score"]:.3f} margin={alignment["margin"]:.3f} accepted={reliable}',flush=True)
            if not reliable:
                report['rejected'].append(dict(id=row['id'],split=split,alignment=alignment))
            if i < args.preview_limit or not reliable:
                front = py360convert.e2p(np.array(gt), (90,90), 0, 0, (256,256), mode='bilinear')
                sheet = Image.new('RGB', (1024, 580), 'white')
                sheet.paste(local.resize((256,256)), (0, 40))
                sheet.paste(Image.fromarray(front.astype('uint8')), (256, 40))
                sheet.paste(gt.resize((512,256)), (512, 40))
                draw = ImageDraw.Draw(sheet)
                draw.text((5, 3), row['id'], fill='black')
                draw.text((5, 20), 'LOCAL FACE2                 GT FRONT 90deg               GT ERP', fill='black')
                before = py360convert.e2p(np.array(raw_gt),(90,90),0,0,(256,256),mode='bilinear')
                sheet.paste(Image.fromarray(before.astype('uint8')),(0,320))
                sheet.paste(raw_gt.resize((512,256)),(256,320))
                draw.text((5,300),f'BEFORE ALIGNMENT; score={alignment["score"]:.3f}, margin={alignment["margin"]:.3f}; accepted={reliable}',fill='black')
                sheet.save(args.output/f'{split}_preview_{i:03d}.jpg')
            if reliable:
                result.append(row)
        if not result:
            atomic_json(args.output/'pair_report.json',report)
            raise ValueError(f'No reliable {split} pairs; inspect rejected previews instead of training')
        check_ids(result)
        write_jsonl(args.output/f'{split}.jsonl', result)
        report['counts'][split] = len(result)
        report[f'{split}_sha256'] = sha256(args.output/f'{split}.jsonl')
    atomic_json(args.output/'pair_report.json', report)
    print(json.dumps(report, indent=2))
    print('Review previews: LOCAL and GT FRONT must show matching direction/content. No geometry approval was inferred.')


if __name__ == '__main__':
    main()
