"""Native Edit Local RGB -> ERP baseline. No training, adapters or panorama hooks.

Benchmark mode reads existing skybox2 JPEG bytes from ZIP, without reprojection.
GT is copied for inspection only and is never passed to the model.
"""
import argparse
import hashlib
import io
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

from .common import (ROOT, DEFAULT_MODEL, atomic_json, check_ids, model_snapshot,
                     read_jsonl, require_local_output, require_versions, sample_seed,
                     sha256, versions, write_jsonl)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--revision')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--manifest', type=Path, default=ROOT/'benchmark_assets/mp3d_stitched1092/reference.jsonl')
    p.add_argument('--raw-root', type=Path, default=ROOT/'benchmark_assets/Matterport3D_raw/v1/scans')
    p.add_argument('--sample-id', help='Select exactly this benchmark sample')
    p.add_argument('--limit', type=int, default=4, help='Select round-robin across buildings; ignored for --sample-id/--image')
    p.add_argument('--image', type=Path, help='Use your own RGB image instead of benchmark skybox2')
    p.add_argument('--prompt-file', type=Path, default=ROOT/'qwen_edit_pano/prompts/native_erp.txt')
    p.add_argument('--negative-prompt', default=' ')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--true-cfg-scale', type=float, default=4.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--offload', choices=['model', 'sequential', 'none'], default='model')
    p.add_argument('--vae-tiling', action='store_true')
    args = p.parse_args()
    if args.image and args.sample_id:
        p.error('Choose --image OR --sample-id')
    if args.width != args.height*2 or args.height <= 0 or args.height % 16:
        p.error('Output must be a 2:1 ERP with dimensions divisible by 16')
    if args.limit <= 0 or args.steps <= 0 or not 0 <= args.seed < 2**63:
        p.error('Invalid limit, steps or seed')
    if not math.isfinite(args.true_cfg_scale) or args.true_cfg_scale <= 0:
        p.error('CFG must be finite and positive')
    require_local_output(args.output, 'outputs')
    return args


def select_rows(args):
    if args.image:
        return [dict(id='custom_image', local_file=str(args.image.resolve()))]
    rows = read_jsonl(args.manifest)
    check_ids(rows)
    if args.sample_id:
        selected = [r for r in rows if r['id'] == args.sample_id]
        if not selected:
            raise ValueError(f'Unknown sample ID: {args.sample_id}')
    else:
        # Spread a small pilot over buildings, deterministically.
        groups = {}
        for row in rows:
            groups.setdefault(row['scene_id'], []).append(row)
        selected = []
        for index in range(max(map(len, groups.values()))):
            for group in groups.values():
                if index < len(group):
                    selected.append(group[index])
                if len(selected) == min(args.limit, len(rows)):
                    break
            if len(selected) == min(args.limit, len(rows)):
                break
    # Deliberately discard dataset captions before any model operation.
    return [dict(id=r['id'], scene_id=r['scene_id'], view_id=r['view_id'],
                 gt_file=str((args.manifest.parent/r['image']).resolve())) for r in selected]


def read_local(row, raw_root):
    if 'local_file' in row:
        path = Path(row['local_file'])
        return path.read_bytes(), dict(local_file=str(path))
    # skybox2 is F in the existing PanFusion converter; no face rotation/flip.
    for key in ('scene_id', 'view_id'):
        if not row[key] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row[key]):
            raise ValueError(f'Invalid {key}')
    archive = raw_root/row['scene_id']/'matterport_skybox_images.zip'
    basename = row['view_id']+'_skybox2_sami.jpg'
    with zipfile.ZipFile(archive) as handle:
        # ZIP paths may contain a double slash; match member basename.
        matches = [n for n in handle.namelist() if Path(n).name == basename]
        if len(matches) != 1:
            raise ValueError(f'Expected one {basename} in {archive}, got {len(matches)}')
        member = matches[0]
        data = handle.read(member)  # zipfile verifies the member CRC on read.
    return data, dict(archive=str(archive.resolve()), member=member, face_index=2,
                      face_convention='PanFusion from_mp3d_skybox: skybox2=F',
                      source='existing skybox JPEG; no ERP crop or reprojection')


def main():
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('Empty prompt')
    rows = select_rows(args)
    from PIL import Image

    # Preflight every selected sample before allocating any GPU model.
    prepared = []
    for row in rows:
        data, source = read_local(row, args.raw_root)
        with Image.open(io.BytesIO(data)) as im:
            local = im.convert('RGB')
            local.load()
        if 'scene_id' in row and local.size != (1024, 1024):
            raise ValueError(f'Unexpected skybox face size: {row["id"]}: {local.size}')
        if row.get('gt_file'):
            with Image.open(row['gt_file']) as gt:
                if gt.width != 2*gt.height:
                    raise ValueError('Reference image is not 2:1')
                gt.verify()
        prepared.append((row, data, local, source))

    require_versions()
    import torch
    from diffusers import QwenImageEditPlusPipeline

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This entry requires a CUDA GPU')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('bfloat16 support required')
    snapshot = model_snapshot(args.model, args.revision, local_files_only=args.local_files_only)
    out = args.output.resolve()
    # Require a brand new directory: partial/old results are never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    config = dict(task='native_edit_local_rgb_to_erp', pipeline='QwenImageEditPlusPipeline',
                  model=args.model, base_snapshot=snapshot, revision=args.revision,
                  prompt=prompt, prompt_file_sha256=sha256(args.prompt_file),
                  negative_prompt=args.negative_prompt, steps=args.steps,
                  true_cfg_scale=args.true_cfg_scale, height=args.height, width=args.width,
                  base_seed=args.seed, seed_mode='per-id', device=args.device,
                  offload=args.offload, vae_tiling=args.vae_tiling, dtype='bfloat16',
                  conditioning='native Edit image+text; no GT caption or GT image',
                  lora=None, world_adapter=None, circular_padding=False,
                  selected_ids=[r['id'] for r in rows], versions=versions(),
                  manifest_sha256=sha256(args.manifest) if not args.image else None,
                  source_sha256=sha256(Path(__file__)),
                  common_sha256=sha256(Path(__file__).with_name('common.py')),
                  note='2:1 output dimensions do not certify valid spherical geometry')
    atomic_json(out/'generation_config.json', config)
    pipe = QwenImageEditPlusPipeline.from_pretrained(snapshot, torch_dtype=torch.bfloat16,
                                                     local_files_only=True)
    for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
        module.requires_grad_(False).eval()
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    if args.offload == 'model':
        pipe.enable_model_cpu_offload(device=args.device)
    elif args.offload == 'sequential':
        pipe.enable_sequential_cpu_offload(device=args.device)
    else:
        pipe.to(device)
    records = []
    for row, original_bytes, local, source in prepared:
        dest = out/row['id']
        dest.mkdir()
        (dest/'input_source.bin').write_bytes(original_bytes)
        local.save(dest/'input.png')
        seed = sample_seed(args.seed, row['id'], 'per-id')
        print(f'Generate {row["id"]}, input={local.size}, seed={seed}', flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = pipe(image=local, prompt=prompt, negative_prompt=args.negative_prompt,
                          width=args.width, height=args.height, num_inference_steps=args.steps,
                          true_cfg_scale=args.true_cfg_scale,
                          generator=torch.Generator(device=device).manual_seed(seed)).images[0]
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start
        if result.size != (args.width, args.height):
            raise ValueError(f'Pipeline changed dimensions: {result.size}')
        result.save(dest/'generated_erp.png')
        # Read/copy GT only for output inspection; never feed it into pipe().
        if row.get('gt_file'):
            shutil.copyfile(row['gt_file'], dest/'gt_erp.png')
        record = dict(id=row['id'], seed=seed, source=source, input_size=list(local.size),
                      input_source_sha256=hashlib.sha256(original_bytes).hexdigest(),
                      generated_erp=str((dest/'generated_erp.png').relative_to(out)),
                      generated_sha256=sha256(dest/'generated_erp.png'),
                      gt_source=row.get('gt_file'),
                      gt_sha256=sha256(dest/'gt_erp.png') if row.get('gt_file') else None,
                      elapsed_seconds=elapsed,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
        atomic_json(dest/'sample.json', record)
        records.append(record)
        write_jsonl(out/'generated.jsonl', records)
    atomic_json(out/'COMPLETE.json', dict(count=len(records), status='generation_finished',
                geometry_quality='not automatically assessed', local_preservation='not automatically assessed'))
    print(f'Finished: {out}', flush=True)


if __name__ == '__main__':
    main()
