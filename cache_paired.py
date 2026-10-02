"""Cache native Edit image+text embeddings and clean reference latents, no target ERP."""
import argparse
import json
from pathlib import Path
from .common import DEFAULT_MODEL, atomic_json, model_snapshot, require_versions, require_local_output, sha256, versions
from .paired_data import CONDITIONING, DEFAULT_PROMPT, cache_key, local_image, paired_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    require_versions()
    require_local_output(args.output,'cache')
    import torch
    from diffusers import QwenImageEditPlusPipeline, AutoencoderKLQwenImage
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE
    from safetensors.torch import save_file, load_file
    rows = paired_rows(args.manifest,'train')
    snapshot = model_snapshot(args.model, local_files_only=args.local_files_only)
    prompt = args.prompt_file.read_text().strip()
    if not prompt:
        raise ValueError('Empty instruction')
    config = dict(conditioning=CONDITIONING,base_snapshot=snapshot,versions=versions(),
                  manifest_sha256=sha256(args.manifest),prompt=prompt,flip_variants=[False,True],
                  reference_vae_dtype='float32',reference_sampling='mode',max_sequence_length=1024,
                  cache_sources={n:sha256(Path(__file__).parent/n) for n in ('cache_paired.py','paired_data.py')})
    args.output.mkdir(parents=True,exist_ok=True)
    meta = args.output/'cache_config.json'
    if meta.exists():
        if json.loads(meta.read_text()) != config:
            raise ValueError('Cache configuration changed; use a new output')
    elif any(args.output.iterdir()):
        raise ValueError('Cache directory has no provenance')
    else:
        atomic_json(meta,config)
    pipe = None
    for i,row in enumerate(rows):
        for flip in (False,True):
            path = args.output/(cache_key(row,flip)+'.safetensors')
            if path.exists():
                tensors = load_file(str(path))
                if set(tensors) != {'embeddings','mask','reference','ref_shape'}:
                    raise ValueError('Invalid cache entry')
                continue
            if pipe is None:
                vae = AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
                pipe = QwenImageEditPlusPipeline.from_pretrained(snapshot,transformer=None,vae=vae,
                         torch_dtype=torch.bfloat16,local_files_only=True)
                pipe.text_encoder.requires_grad_(False).eval().to(args.device)
                pipe.vae.requires_grad_(False).eval().to(device=args.device,dtype=torch.float32)
            image = local_image(row,flip)
            cw,ch = calculate_dimensions(CONDITION_IMAGE_SIZE,image.width/image.height)
            vw,vh = calculate_dimensions(VAE_IMAGE_SIZE,image.width/image.height)
            condition = pipe.image_processor.resize(image,ch,cw)
            pixels = pipe.image_processor.preprocess(image,vh,vw).unsqueeze(2).to(args.device,torch.bfloat16).float()  # Match native prepare_latents input cast
            with torch.inference_mode():
                emb,mask = pipe.encode_prompt(prompt=prompt,image=[condition],device=torch.device(args.device))
                latent = pipe._encode_vae_image(pixels,generator=None).to(torch.bfloat16)
                ref = pipe._pack_latents(latent,1,latent.shape[1],latent.shape[3],latent.shape[4])
            if mask is None:
                mask = torch.ones(emb.shape[:2],device=emb.device,dtype=torch.bool)
            data = dict(embeddings=emb[0].cpu().contiguous(),mask=mask[0].bool().cpu().contiguous(),
                        reference=ref[0].cpu().contiguous(),ref_shape=torch.tensor([1,vh//16,vw//16]))
            if not all(torch.isfinite(v).all() for v in data.values()):
                raise FloatingPointError('Nonfinite conditioning cache')
            tmp = path.with_suffix('.tmp')
            save_file(data,str(tmp))
            tmp.replace(path)
        print(f'[{i+1}/{len(rows)}] {row["id"]}',flush=True)
    atomic_json(args.output/'complete.json',dict(count=len(rows)*2,config_sha256=sha256(meta)))


if __name__ == '__main__':
    main()
