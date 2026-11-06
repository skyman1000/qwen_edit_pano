"""Two single-GPU workers extend the existing comparison with eight validation images."""
import argparse
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'qwen_edit_pano/outputs/comparison_epoch15_25_scale05'
MANIFEST = ROOT / 'qwen_edit_pano/data/pano_epoch15_25_extra8_v1.jsonl'


def load(path):
    return json.loads(path.read_text())


def folder(epoch):
    return OUTPUT / 'extra8_v1' / f'epoch{epoch:03d}_scale05'


def update_review(rows):
    # Only one worker writes the shared page. Completion markers are written
    # after all images; neither worker waits while occupying the other's GPU.
    with (OUTPUT / '.review.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((folder(e) / 'COMPLETE.json').is_file() for e in (15, 25)):
            print('Other GPU is still running; it will update review.html when finished.', flush=True)
            return
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 26)
        except OSError:
            font = ImageFont.load_default()
        page = ['<!doctype html><meta charset="utf-8"><title>Epoch15 vs25 / LoRA0.5 / 14 images</title>',
                '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%;height:auto}section{margin-bottom:40px}</style>',
                '<h1>Epoch 015 vs 025 | LoRA 0.5</h1>',
                '<p>Top: Local RGB / GT. Bottom: epoch015 / epoch025. '
                'Previous 6 + new 8 validation images. One generation per image and checkpoint; '
                'seed 0, 28 steps, CFG 4. Click to enlarge.</p><h2>Previous 6 images</h2>']
        old = load(ROOT / 'qwen_edit_pano/data/oracle_indoor6_selection_v1.json')
        for i, (sid, label) in enumerate(zip(old['ids'], old['labels']), 1):
            name = f'{i:02d}_{label}.png'
            if not (OUTPUT / name).is_file():
                raise FileNotFoundError(OUTPUT / name)
            page.append(f'<section><h3>{i}. {html.escape(sid)}</h3><a href="{name}">'
                        f'<img loading="lazy" src="{name}"></a></section>')
        a, b = folder(15), folder(25)
        ca, cb = load(a / 'run_config.json'), load(b / 'run_config.json')
        for key in ('lora_scale', 'prompt', 'steps', 'cfg', 'seed', 'base_snapshot', 'manifest_sha256'):
            if ca[key] != cb[key]:
                raise ValueError('Run settings differ: ' + key)
        page.append('<h2>New 8 images</h2>')
        panels = OUTPUT / 'extra8_v1/panels'
        panels.mkdir(exist_ok=True)
        for i, row in enumerate(rows, 7):
            sid = row['id']
            if load(a / sid / 'sample.json') != load(b / sid / 'sample.json'):
                raise ValueError('Sample seed/dimensions differ: ' + sid)
            sheet = Image.new('RGB', (2048, 1104), 'white')
            draw = ImageDraw.Draw(sheet)
            files = [('LOCAL RGB', a / sid / 'input.png'), ('GT ERP', a / sid / 'gt_erp.png'),
                     ('Epoch 015 | LoRA 0.5', a / sid / 'generated_erp.png'),
                     ('Epoch 025 | LoRA 0.5', b / sid / 'generated_erp.png')]
            for j, (title, path) in enumerate(files):
                with Image.open(path) as im:
                    preview = ImageOps.contain(im.convert('RGB'), (1024, 512))
                x, y = j % 2 * 1024, j // 2 * 552
                draw.text((x + 12, y + 6), title, font=font, fill='black')
                sheet.paste(preview, (x + (1024 - preview.width)//2, y + 40 + (512 - preview.height)//2))
            name = f'{i:02d}_{sid}.png'
            sheet.save(panels / name)
            rel = 'extra8_v1/panels/' + name
            page.append(f'<section><h3>{i}. {html.escape(sid)}</h3><a href="{rel}">'
                        f'<img loading="lazy" src="{rel}"></a></section>')
        for name, content in [('review.html', '\n'.join(page)), ('COMPLETE.json', json.dumps(
                dict(images=14, previous_images=6, new_images=8, epochs=[15, 25], lora_scale=.5), indent=2)+'\n')]:
            temporary = OUTPUT / f'.{name}.{os.getpid()}.tmp'
            temporary.write_text(content)
            temporary.replace(OUTPUT / name)
        print(f'Updated 14-image comparison: {OUTPUT / "review.html"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int, choices=(0, 1), default=int(os.environ.get('SLURM_PROCID', '0')))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    rows = [json.loads(s) for s in MANIFEST.read_text().splitlines() if s.strip()]
    if len(rows) != 8 or len({r['id'] for r in rows}) != 8:
        raise ValueError('Expected eight unique validation images')
    epoch = (15, 25)[args.worker]
    checkpoint = ROOT / f'qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch{epoch:03d}'
    if not (checkpoint / 'COMPLETE.json').is_file():
        raise ValueError('Checkpoint incomplete')
    cfg = load(checkpoint / 'pano_config.json')
    command = [sys.executable, '-m', 'qwen_edit_pano.paired_inference', '--local-files-only',
               '--model', cfg['base_snapshot'], '--checkpoint', str(checkpoint),
               '--manifest', str(MANIFEST), '--output', str(folder(epoch)),
               '--limit', '8', '--steps', '28', '--cfg', '4', '--seed', '0',
               '--lora-scale', '0.5', '--prompt', cfg['prompt']]
    print(f'Worker {args.worker}: epoch{epoch}, LoRA0.5, 8 images; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
    if args.dry_run:
        print(command)
        return
    if not (folder(epoch) / 'COMPLETE.json').is_file():
        subprocess.run(command, cwd=ROOT, check=True)
    update_review(rows)


if __name__ == '__main__':
    main()
