"""User-run GPU caching, strictly local weights; never train the frozen models."""
import argparse
import gc
import fcntl
from pathlib import Path

from ..build_polished_world import atomic_json, read
from .data import DEFAULT_PACK, load_pack, target_image, atomic_tensors


def semantic_phrases(names):
    aliases={'tv_monitor':'television or computer monitor','objects':'object','misc':'miscellaneous object'}
    return {key:aliases.get(name,name.replace('_',' ')) for key,name in names.items()}


def extract_semantics(model, tokenizer, names, device):
    import torch
    phrases=semantic_phrases(names)
    table=torch.zeros(42,model.config.text_config.hidden_size)
    for key,phrase in phrases.items():
        prefix='a photo of a '
        text=prefix+phrase+'.'
        batch=tokenizer(text,return_offsets_mapping=True,return_tensors='pt')
        offsets=batch.pop('offset_mapping')[0]
        selected=(offsets[:,1]>len(prefix)) & (offsets[:,0]<len(prefix)+len(phrase))
        if not selected.any():
            raise ValueError('No category tokens found: '+phrase)
        inputs={k:v.to(device) for k,v in batch.items() if k in ('input_ids','attention_mask')}
        # model.model is Qwen2_5_VLModel, avoiding vocabulary logits and image-only
        # arguments accessed unconditionally by the Edit encode_prompt wrapper.
        outputs=model.model(**inputs,use_cache=False,return_dict=True)
        table[int(key)]=outputs.last_hidden_state[0,selected.to(device)].float().mean(0).cpu()
    return table


def check_condition(path):
    from safetensors import safe_open
    with safe_open(str(path),framework='pt',device='cpu') as tensors:
        if set(tensors.keys())!={'embeddings','mask','reference','ref_shape'}:
            raise ValueError('Wrong native condition cache')
        es=tensors.get_slice('embeddings').get_shape();ms=tensors.get_slice('mask').get_shape()
        rs=tensors.get_slice('reference').get_shape()
        if len(es)!=2 or es[1]!=3584 or ms!=[es[0]] or len(rs)!=2 or rs[1]!=64:
            raise ValueError('Native condition dimensions differ')


def check_posterior(path, height):
    from safetensors import safe_open
    with safe_open(str(path),framework='pt',device='cpu') as tensors:
        if set(tensors.keys())!={'mean','std'}:
            raise ValueError('Wrong posterior cache')
        expected=[1,16,1,height//8,height//4]
        if any(tensors.get_slice(key).get_shape()!=expected for key in ('mean','std')):
            raise ValueError('Target posterior resolution differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,default=DEFAULT_PACK)
    p.add_argument('--smoke',action='store_true',help='All class vectors, only the selected two target samples')
    p.add_argument('--check-only',action='store_true',help='CPU cache completeness/header checks')
    args=p.parse_args();args.pack=args.pack.resolve()
    plan,records=load_pack(args.pack)
    if not (args.pack/'PREPARED.json').is_file():
        raise ValueError('Run CPU preparation first')
    train=[row for row in records if row['split']=='train' and (not args.smoke or row['id'] in plan['smoke_ids'])]
    from safetensors.torch import load_file
    semantic_path=args.pack/'class_features.safetensors'
    semantic_config=dict(policy=plan['semantic_policy'],names=plan['names'],phrases=semantic_phrases(plan['names']),
                         base_snapshot=plan['base_snapshot'],pooling='mean of name tokens only; padding/prefix excluded')
    if semantic_path.exists():
        if read(args.pack/'class_features.json')!=semantic_config:
            raise ValueError('Class semantic cache configuration differs')
        import torch
        table=load_file(str(semantic_path))['features']
        if table.shape!=(42,3584) or not torch.isfinite(table).all():
            raise ValueError('Class feature dimensions/values differ')
        if any(table[int(key)].norm()==0 for key in plan['names']):
            raise ValueError('Empty semantic vector for a task category')
    missing_conditions=[];missing_posteriors=[]
    for row in train:
        cp,pp=Path(row['condition_cache']),Path(row['posterior_cache'])
        if cp.exists():check_condition(cp)
        else:missing_conditions.append(row)
        if pp.exists():check_posterior(pp,plan['height'])
        else:missing_posteriors.append(row)
        if not Path(row['region_cache']).is_file():
            raise ValueError('Missing CPU region cache')
    status=dict(samples=len(train),missing_conditions=len(missing_conditions),missing_posteriors=len(missing_posteriors),
                missing_class_features=not semantic_path.is_file(),smoke=args.smoke)
    print(status,flush=True)
    if args.check_only:
        if missing_conditions or missing_posteriors or not semantic_path.is_file():
            raise SystemExit('Caches incomplete; run cache on your GPU.')
        print('CACHE CHECK PASSED.',flush=True);return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Allocate one BF16 CUDA GPU; caching never downloads weights')
    with (args.pack/'PACK.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=torch.device('cuda:0');snapshot=plan['base_snapshot']
        if missing_conditions:
            from diffusers import QwenImageEditPlusPipeline,AutoencoderKLQwenImage
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions,CONDITION_IMAGE_SIZE,VAE_IMAGE_SIZE
            from qwen_edit_pano.paired_data import local_image
            vae=AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
            pipe=QwenImageEditPlusPipeline.from_pretrained(snapshot,transformer=None,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
            pipe.text_encoder.requires_grad_(False).eval().to(device)
            pipe.vae.requires_grad_(False).eval().to(device=device,dtype=torch.float32)
            with torch.inference_mode():
                for index,row in enumerate(missing_conditions):
                    image=local_image(row)
                    cw,ch=calculate_dimensions(CONDITION_IMAGE_SIZE,image.width/image.height)
                    vw,vh=calculate_dimensions(VAE_IMAGE_SIZE,image.width/image.height)
                    condition=pipe.image_processor.resize(image,ch,cw)
                    pixels=pipe.image_processor.preprocess(image,vh,vw).unsqueeze(2).to(device,torch.bfloat16).float()
                    emb,mask=pipe.encode_prompt(prompt=plan['prompt'],image=[condition],device=device)
                    latent=pipe._encode_vae_image(pixels,generator=None).to(torch.bfloat16)
                    reference=pipe._pack_latents(latent,1,latent.shape[1],latent.shape[3],latent.shape[4])
                    if mask is None:mask=torch.ones(emb.shape[:2],device=device,dtype=torch.bool)
                    atomic_tensors(row['condition_cache'],dict(embeddings=emb[0],mask=mask[0].bool(),reference=reference[0],ref_shape=torch.tensor([1,vh//16,vw//16])))
                    print(dict(stage='native_condition',completed=index+1,total=len(missing_conditions)),flush=True)
            del pipe,vae;gc.collect();torch.cuda.empty_cache()
        if not semantic_path.is_file():
            from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration
            tokenizer=AutoTokenizer.from_pretrained(snapshot,subfolder='tokenizer',local_files_only=True,use_fast=True)
            encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(snapshot,subfolder='text_encoder',torch_dtype=torch.bfloat16,local_files_only=True).requires_grad_(False).eval().to(device)
            with torch.inference_mode():
                features=extract_semantics(encoder,tokenizer,plan['names'],device)
            if not torch.isfinite(features).all():raise FloatingPointError('Nonfinite class vectors')
            atomic_json(args.pack/'class_features.json',semantic_config)
            atomic_tensors(semantic_path,dict(features=features))
            print('Frozen class-name features saved.',flush=True)
            del encoder,tokenizer;gc.collect();torch.cuda.empty_cache()
        if missing_posteriors:
            import numpy as np
            from PIL import Image
            from diffusers import AutoencoderKLQwenImage
            vae=AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).requires_grad_(False).eval().to(device)
            if 2**len(vae.temperal_downsample)!=8:raise ValueError('Expected VAE factor eight')
            with torch.inference_mode():
                for index,row in enumerate(missing_posteriors):
                    rgb=target_image(row).resize((2*plan['height'],plan['height']),Image.Resampling.BICUBIC)
                    pixels=torch.from_numpy(np.array(rgb).copy()).permute(2,0,1).float()/127.5-1
                    posterior=vae.encode(pixels[None,:,None].to(device)).latent_dist
                    atomic_tensors(row['posterior_cache'],dict(mean=posterior.mean,std=posterior.std))
                    del posterior,pixels
                    print(dict(stage='target_posterior',id=row['id'],completed=index+1,total=len(missing_posteriors)),flush=True)
            del vae;gc.collect();torch.cuda.empty_cache()
        for row in train:
            check_condition(row['condition_cache']);check_posterior(row['posterior_cache'],plan['height'])
        atomic_json(args.pack/('CACHE_SMOKE_COMPLETE.json' if args.smoke else 'CACHE_COMPLETE.json'),
                    dict(samples=len(train),ids=[row['id'] for row in train],base_snapshot=snapshot,
                         semantic_policy=plan['semantic_policy'],file_content_hashes_computed=False))
        print('CACHE COMPLETE. No training performed.',flush=True)


if __name__=='__main__':main()
