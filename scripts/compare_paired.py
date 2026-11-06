"""Sequential base/finetuned inference with matched inputs and comparison panels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def read_json(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--cfg', type=float, default=4.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lora-scale', type=float, default=0.5)
    p.add_argument('--prompt', help='Same override for both runs; otherwise checkpoint prompt')
    p.add_argument('--offload', choices=['model', 'sequential', 'none'], default='model')
    args = p.parse_args()
    if args.limit < 1 or args.steps < 1 or not math.isfinite(args.lora_scale) or args.lora_scale < 0:
        p.error('Invalid limit, steps or lora-scale')
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root / 'qwen_edit_pano' / 'outputs'):
        p.error('Output must be inside qwen_edit_pano/outputs')
    if output.exists() and any(output.iterdir()):
        p.error('Use a new or empty comparison output directory')
    checkpoint = args.checkpoint.resolve()
    if not any((checkpoint / n).is_file() for n in ('COMPLETE.json', 'SMOKE_COMPLETE.json')):
        p.error('Checkpoint has no completion marker')
    config = read_json(checkpoint / 'pano_config.json')
    prompt = args.prompt if args.prompt is not None else config['prompt']
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        p.error('Manifest does not exist')
    from PIL import Image, ImageDraw, ImageOps
    shared = [sys.executable, '-m', 'qwen_edit_pano.paired_inference',
              '--local-files-only', '--model', config['base_snapshot'],
              '--manifest', str(manifest), '--prompt', prompt,
              '--limit', str(args.limit), '--steps', str(args.steps),
              '--cfg', str(args.cfg), '--seed', str(args.seed), '--offload', args.offload]
    for name, extra in [('base', []), ('finetuned', ['--checkpoint', str(checkpoint),
                                                   '--lora-scale', str(args.lora_scale)])]:
        print(f'Running {name} in a separate process', flush=True)
        subprocess.run(shared + ['--output', str(output / name)] + extra, cwd=root, check=True)

    base, tuned = output / 'base', output / 'finetuned'
    bcfg, tcfg = read_json(base / 'run_config.json'), read_json(tuned / 'run_config.json')
    for key in ('base_snapshot', 'prompt', 'steps', 'cfg', 'seed', 'width', 'height', 'manifest_sha256', 'versions'):
        if bcfg[key] != tcfg[key]:
            raise ValueError(f'Comparison settings differ: {key}')
    bids = {f.parent.name for f in base.glob('*/sample.json')}
    tids = {f.parent.name for f in tuned.glob('*/sample.json')}
    if not bids or bids != tids:
        raise ValueError('Comparison sample IDs differ or are empty')
    panels = output / 'comparisons'
    panels.mkdir()
    for sid in sorted(bids):
        b, t = base / sid, tuned / sid
        if read_json(b / 'sample.json') != read_json(t / 'sample.json'):
            raise ValueError(f'Sample seed/dimensions differ: {sid}')
        for name in ('input.png', 'gt_erp.png'):
            if hashlib.sha256((b / name).read_bytes()).digest() != hashlib.sha256((t / name).read_bytes()).digest():
                raise ValueError(f'Comparison image differs: {sid}/{name}')
        # Letterbox local RGB; preserve the 2:1 aspect ratio of each ERP preview.
        canvas = Image.new('RGB', (2048, 296), 'white')
        draw = ImageDraw.Draw(canvas)
        items = [('LOCAL RGB', b / 'input.png'), ('BASE (no LoRA / no pano padding)', b / 'generated_erp.png'),
                 (f'FINETUNED (LoRA {args.lora_scale:g} / pano padding)', t / 'generated_erp.png'),
                 ('GT ERP', t / 'gt_erp.png')]
        for i, (label, path) in enumerate(items):
            with Image.open(path) as im:
                preview = ImageOps.contain(im.convert('RGB'), (512, 256))
            canvas.paste(preview, (i * 512 + (512 - preview.width) // 2, 40 + (256 - preview.height) // 2))
            draw.text((i * 512 + 8, 12), label, fill='black')
        canvas.save(panels / f'{sid}.png')
    report = dict(status='comparison_completed', samples=len(bids), lora_scale=args.lora_scale,
                  matched_settings=True, matched_inputs_and_seeds=True,
                  base_padding=bcfg['padding'], finetuned_padding=tcfg['padding'],
                  note='Native base versus full panorama finetuning setup; not a LoRA-only ablation. GT is not an inference condition.')
    (output / 'COMPLETE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Comparison panels: {panels}', flush=True)


if __name__ == '__main__':
    main()
