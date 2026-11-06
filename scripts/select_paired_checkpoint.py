"""Six fixed validation images, three epochs, two scales; two independent GPU workers."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
JOBS = [(epoch, scale) for epoch in (15, 16, 25) for scale in (0.5, 0.75)]


def load(path):
    return json.loads(path.read_text())


def name(epoch, scale):
    return f'e{epoch:03d}_s{scale:g}_seed0'


def write_review(output, ids):
    if not all((output / name(e, s) / 'COMPLETE.json').is_file() for e, s in JOBS):
        print('Other worker is still running; review will be written when all six settings finish.', flush=True)
        return
    first = output / name(*JOBS[0])
    reference = load(first / 'run_config.json')
    for epoch, scale in JOBS:
        folder = output / name(epoch, scale)
        config = load(folder / 'run_config.json')
        for key in ('base_snapshot', 'prompt', 'steps', 'cfg', 'seed', 'manifest_sha256', 'width', 'height'):
            if config[key] != reference[key]:
                raise ValueError(f'Comparison settings differ: {folder.name}/{key}')
        if {p.parent.name for p in folder.glob('*/sample.json')} != set(ids):
            raise ValueError(f'Wrong sample set: {folder}')
        for sid in ids:
            if load(folder / sid / 'sample.json') != load(first / sid / 'sample.json'):
                raise ValueError(f'Sample seeds differ: {folder}/{sid}')
    parts = ['<!doctype html><meta charset="utf-8"><title>Panorama checkpoint selection</title>',
             '<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{width:400px}th{position:sticky;top:0;background:white}</style>',
             '<h1>6 validation images / epochs 15,16,25 / LoRA 0.5,0.75</h1>',
             '<p>One generation per setting and image. Seed 0 (deterministic per-image seed), 28 steps, CFG 4. '
             'Six images from three held-out houses; diagnostic comparison, not a full validation score. '
             'No World Adapter. Click an image for full resolution.</p><table><tr><th>Local input</th><th>GT ERP</th>']
    parts += [f'<th>{name(e,s)}</th>' for e,s in JOBS]
    parts.append('</tr>')
    for sid in ids:
        paths = [(first / sid / 'input.png'), (first / sid / 'gt_erp.png')]
        paths += [output / name(e,s) / sid / 'generated_erp.png' for e,s in JOBS]
        parts.append('<tr>')
        for i, path in enumerate(paths):
            rel = html.escape(path.relative_to(output).as_posix(), quote=True)
            label = html.escape(sid) + '<br>' if i == 0 else ''
            parts.append(f'<td>{label}<a href="{rel}"><img loading="lazy" src="{rel}"></a></td>')
        parts.append('</tr>')
    parts.append('</table>')
    for filename, content in [('review.html', '\n'.join(parts)),
                              ('COMPLETE.json', json.dumps(dict(status='completed', images=6, settings=6,
                                                               generated_erp=36, sample_ids=ids), indent=2)+'\n')]:
        temporary = output / f'.{filename}.{os.getpid()}.tmp'
        temporary.write_text(content)
        temporary.replace(output / filename)
    print(f'Review: {output / "review.html"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int, choices=(0, 1), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Print commands without loading models')
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'qwen_edit_pano/outputs'):
        p.error('Output must be under qwen_edit_pano/outputs')
    manifest = ROOT / 'qwen_edit_pano/data/pano_checkpoint_val6_v1.jsonl'
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    ids = [r['id'] for r in rows]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError('Expected exactly six unique images')
    run = ROOT / 'qwen_edit_pano/outputs/paired_full_63637'
    configs = []
    for epoch in (15, 16, 25):
        checkpoint = run / f'checkpoint-epoch{epoch:03d}'
        for filename in ('COMPLETE.json', 'pytorch_lora_weights.safetensors', 'adapter_resume.safetensors'):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)
        configs.append(load(checkpoint / 'pano_config.json'))
    for key in ('base_snapshot', 'conditioning', 'prompt', 'padding_columns'):
        if any(c[key] != configs[0][key] for c in configs):
            raise ValueError(f'Checkpoint setting differs: {key}')
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    for epoch, scale in JOBS[args.worker::2]:
        folder = output / name(epoch, scale)
        checkpoint = run / f'checkpoint-epoch{epoch:03d}'
        command = [sys.executable, '-m', 'qwen_edit_pano.paired_inference', '--local-files-only',
                   '--checkpoint', str(checkpoint), '--model', configs[0]['base_snapshot'],
                   '--manifest', str(manifest), '--output', str(folder), '--limit', '6',
                   '--steps', '28', '--cfg', '4', '--seed', '0', '--lora-scale', str(scale),
                   '--prompt', configs[0]['prompt']]
        print(f'Worker {args.worker}: epoch={epoch}, LoRA={scale}, images=6', flush=True)
        if args.dry_run:
            print(command)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'Use a fresh output directory; existing results: {folder}')
        subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        write_review(output, ids)


if __name__ == '__main__':
    main()
