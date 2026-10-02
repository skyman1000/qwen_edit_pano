"""Load paired LoRA into native EditPlus; Local RGB only, no GT model inputs."""
import argparse
import json
from pathlib import Path
from .common import (DEFAULT_MODEL, atomic_json, check_ids, model_snapshot, require_local_output,
                     require_versions, read_jsonl, sample_seed, sha256, versions)
from .paired_data import CONDITIONING, DEFAULT_PROMPT, local_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path)
    source.add_argument('--image',type=Path)
    p.add_argument('--checkpoint',type=Path,help='Omit for the native no-LoRA baseline (also disables panorama padding)')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--prompt',help='Optional override; otherwise use the training task instruction')
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--steps',type=int,default=28)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--offload',choices=['model','sequential','none'],default='model')
    args = p.parse_args()
    if args.limit < 1 or args.steps < 1:
        p.error('limit/steps must be positive')
    require_local_output(args.output,'outputs')
    require_versions()
    import torch
    from PIL import Image
    from .paired_pipeline import PairedPanoPipeline
    from diffusers import AutoencoderKLQwenImage
    from .inference_state import enforce_inference_eval
    snapshot = model_snapshot(args.model,local_files_only=args.local_files_only)
    config = None
    if args.checkpoint:
        lora_weight_name = 'pytorch_lora_weights.safetensors'
        if not (args.checkpoint / lora_weight_name).is_file():
            raise FileNotFoundError(args.checkpoint / lora_weight_name)
        if not any((args.checkpoint/name).is_file() for name in ['COMPLETE.json','SMOKE_COMPLETE.json']):
            raise ValueError('Checkpoint has no completion marker')
        config = json.loads((args.checkpoint/'pano_config.json').read_text())
        if config.get('conditioning') != CONDITIONING or config['base_snapshot'] != snapshot:
            raise ValueError('Wrong checkpoint conditioning/base snapshot')
    prompt = args.prompt if args.prompt is not None else (config['prompt'] if config else DEFAULT_PROMPT.read_text().strip())
    rows = read_jsonl(args.manifest)[:args.limit] if args.manifest else [dict(id='custom_image')]
    check_ids(rows)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty inference output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    vae = AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe = PairedPanoPipeline.from_pretrained(snapshot,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.vae.to(dtype=torch.float32)
    if config:
        pipe.load_lora_weights(str(args.checkpoint), weight_name=lora_weight_name,
                               local_files_only=True, use_safetensors=True)
        pipe.enable_pano(config['padding_columns'])
    for module in (pipe.transformer,pipe.vae,pipe.text_encoder):
        enforce_inference_eval(module)
    if args.offload == 'model':
        pipe.enable_model_cpu_offload()
    elif args.offload == 'sequential':
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to('cuda')
    run_config = dict(base_snapshot=snapshot,prompt=prompt,steps=args.steps,cfg=args.cfg,seed=args.seed,
                      width=2048,height=1024,versions=versions(),checkpoint=str(args.checkpoint),
                      checkpoint_sha256=sha256(args.checkpoint/'adapter_resume.safetensors') if config else None,
                      manifest_sha256=sha256(args.manifest) if args.manifest else None,
                      padding='persistent_target_only' if config else 'none')
    atomic_json(args.output/'run_config.json',run_config)
    for row in rows:
        image = Image.open(args.image).convert('RGB') if args.image else local_image(row)
        directory = args.output/row['id']
        directory.mkdir()
        image.save(directory/'input.png')
        seed = sample_seed(args.seed,row['id'],'per-id')
        result = pipe(image=image,prompt=prompt,negative_prompt=' ',height=1024,width=2048,
                      num_inference_steps=args.steps,true_cfg_scale=args.cfg,
                      generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
        if result.size != (2048,1024):
            raise AssertionError('Incorrect ERP output dimensions')
        result.save(directory/'generated_erp.png')
        # GT is read only AFTER generation, solely for side-by-side inspection.
        if args.manifest and 'image' in row:
            from .paired_data import target_image
            target_image(row).resize((2048,1024)).save(directory/'gt_erp.png')
        atomic_json(directory/'sample.json',dict(id=row['id'],seed=seed,output_size=result.size))
    atomic_json(args.output/'COMPLETE.json',dict(samples=len(rows),status='inference_completed',
                note='Dimensions checked; visual ERP geometry and image fidelity require evaluation.'))


if __name__ == '__main__':
    main()
