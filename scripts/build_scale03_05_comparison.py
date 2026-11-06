"""Build CPU-only comparison panels from existing epoch15/16/25 inference results."""
import argparse
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load(path):
    return json.loads(path.read_text())


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path,
                   default=root / 'qwen_edit_pano/outputs/comparison_epoch15_16_25_scale03_05')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = load(root / 'qwen_edit_pano/data/oracle_indoor6_selection_v1.json')
    settings = [(e, s) for s in (0.3, 0.5) for e in (15, 16, 25)]

    def folder(epoch, scale):
        if scale == 0.3:
            return root / f'qwen_edit_pano/outputs/paired_val_epoch{epoch:03d}_scale03'
        return root / f'qwen_edit_pano/outputs/checkpoint_selection_70218/e{epoch:03d}_s0.5_seed0'

    first = folder(15, 0.3)
    reference = load(first / 'run_config.json')
    for epoch, scale in settings:
        cfg = load(folder(epoch, scale) / 'run_config.json')
        assert cfg['lora_scale'] == scale
        assert load(folder(epoch, scale) / 'COMPLETE.json')['samples'] == 6
        for key in ('prompt', 'steps', 'cfg', 'seed', 'base_snapshot', 'manifest_sha256'):
            assert cfg[key] == reference[key], (epoch, scale, key)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 24)
    except OSError:
        font = ImageFont.load_default()
    page = ['<!doctype html><meta charset="utf-8"><title>Epoch 15/16/25: LoRA 0.3 vs 0.5</title>',
            '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%;height:auto}section{margin-bottom:40px}</style>',
            '<h1>Epoch 15 / 16 / 25 — LoRA 0.3 / 0.5</h1>',
            '<p>Six fixed validation images. Seed 0, 28 steps, CFG 4. '
            'Existing results only; no new inference. Click a panel for full resolution.</p>']
    for index, (sid, label) in enumerate(zip(selection['ids'], selection['labels']), 1):
        sheet = Image.new('RGB', (3072, 1656), 'white')
        draw = ImageDraw.Draw(sheet)
        files = [first / sid / 'input.png', first / sid / 'gt_erp.png']
        files += [folder(e, s) / sid / 'generated_erp.png' for e, s in settings]
        titles = ['LOCAL RGB', 'GT ERP'] + [f'Epoch {e:03d} | LoRA {s:g}' for e, s in settings]
        cells = [(0, 0), (1, 0)] + [(i % 3, 1 + i // 3) for i in range(6)]
        for path, title, (column, row) in zip(files, titles, cells):
            with Image.open(path) as im:
                preview = ImageOps.contain(im.convert('RGB'), (1024, 512))
            x, y = column * 1024, row * 552
            draw.text((x + 12, y + 8), title, fill='black', font=font)
            sheet.paste(preview, (x + (1024 - preview.width) // 2,
                                 y + 40 + (512 - preview.height) // 2))
        draw.multiline_text((2060, 50), 'Columns: epoch 015 / 016 / 025\n\nMiddle row: LoRA 0.3\nBottom row: LoRA 0.5',
                            fill='black', font=font, spacing=12)
        for e, s in settings:
            assert load(folder(e, s) / sid / 'sample.json') == load(first / sid / 'sample.json')
        filename = f'{index:02d}_{label}.png'
        sheet.save(output / filename)
        page.append(f'<section><h2>{index}. {html.escape(label)} — {html.escape(sid)}</h2>'
                    f'<a href="{filename}"><img loading="lazy" src="{filename}"></a><p>Original ERP: ')
        for e, s in settings:
            rel = html.escape(os.path.relpath(folder(e, s) / sid / 'generated_erp.png', output), quote=True)
            page.append(f'<a href="{rel}">epoch{e}/{s}</a> &nbsp; ')
        page.append('</p></section>')
    (output / 'review.html').write_text('\n'.join(page))
    (output / 'COMPLETE.json').write_text(json.dumps(dict(images=6, settings=6, scales=[0.3, 0.5],
                                                        epochs=[15, 16, 25], new_inference=False), indent=2)+'\n')
    print(output / 'review.html')


if __name__ == '__main__':
    main()
