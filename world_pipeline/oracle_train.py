"""Frozen paired Panorama LoRA + trainable object World Encoder/Adapter.

Single GPU, no flips, GT only, explicit optimizer-step budget. Does not resume Edit training.
"""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

from .common import read,sha,write,new_output
from .oracle_data import load_bundle
from .oracle_model import WorldBranch,tensorize,FEATURE_VERSION
from .oracle_memory import training_autocast,ActivationStorage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--condition',choices=('full','observed','constant'),default='full')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--warmup-steps',type=int,default=20)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--blocks',type=int,nargs='+',default=[20,40,59])
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--purpose',choices=('pilot','oracle'),default='pilot')
    p.add_argument('--resume',type=Path,help='World checkpoint; writes a NEW output directory')
    p.add_argument('--smoke',action='store_true',help='Exactly 2 optimizer steps; zero-gate and gradient checks')
    p.add_argument('--preflight',action='store_true',help='CPU provenance/data validation only')
    p.add_argument('--activation-storage',choices=('cpu','gpu'),default='cpu',
                   help='CPU offloads saved differentiable activations; weights stay on GPU')
    args=p.parse_args()
    if args.smoke:
        args.steps=2;args.warmup_steps=0
    if (min(args.steps,args.accumulation,args.save_every,args.width)<1 or args.width%8 or
        not 0<=args.warmup_steps<args.steps or not 0<args.lr<1 or
        min(args.blocks)<0 or len(set(args.blocks))!=len(args.blocks)):
        p.error('Invalid dimensions, steps, blocks, warmup or learning rate')
    from qwen_edit_pano.common import require_versions,versions,source_hashes
    require_versions()
    bundle,entries=load_bundle(args.bundle)
    train=[r for r in entries if r['split']=='train']
    if args.purpose=='oracle' and not bundle['val_count']:
        raise ValueError('Formal oracle needs reviewed held-out validation; current bundle is pilot only')
    if args.preflight:
        print(f'PREFLIGHT PASSED: train={len(train)}, val={bundle["val_count"]}; GT only; no model loaded')
        return
    if args.output is None:
        p.error('--output required')
    import numpy as np
    import torch
    from diffusers import AutoencoderKLQwenImage,QwenImageTransformer2DModel,FlowMatchEulerDiscreteScheduler
    from diffusers import QwenImageEditPlusPipeline as Pipeline
    from safetensors.torch import save_file,load_file
    from peft import LoraConfig,set_peft_model_state_dict
    from PIL import Image
    from qwen_edit_pano.paired_data import target_image
    from qwen_edit_pano.paired_circular import install_circular_padding
    from qwen_pano.profiles import REPO_PANORAMA
    from qwen_edit_pano.losses import panorama_loss
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Allocate one BF16 CUDA GPU')
    device=torch.device('cuda:0'); torch.manual_seed(args.seed)
    base=Path(bundle['base_checkpoint']); pano=read(base/'pano_config.json'); old=read(base/'training_config.json')
    config=dict(feature_version=FEATURE_VERSION,bundle=str(args.bundle.resolve()),bundle_sha256=sha(args.bundle/'BUNDLE.json'),
        base_checkpoint=str(base),base_snapshot=bundle['base_snapshot'],condition=args.condition,
        blocks=args.blocks,width=args.width,max_objects=bundle['max_objects'],height=bundle['height'],
        padding_columns=bundle['padding_columns'],lr=args.lr,warmup_steps=args.warmup_steps,
        accumulation=args.accumulation,seed=args.seed,purpose=args.purpose,smoke=args.smoke,
        augmentation='none',geometry_loss='flow+0.5cube+0.5yaw_from_first_step',
        autocast_cache_enabled=False,activation_storage=args.activation_storage,
        versions=versions(),paired_source_hashes=source_hashes(),
        oracle_source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('oracle_*.py')})
    start=0
    if args.resume:
        previous=read(args.resume/'config.json')
        saved=read(args.resume/'COMPLETE.json')
        if (previous!=config or saved['feature_version']!=FEATURE_VERSION or
            sha(args.resume/'world.safetensors')!=saved['world_sha256'] or
            sha(args.resume/'optimizer.pt')!=saved['optimizer_sha256']):
            raise ValueError('Resume configuration/source changed; use identical world experiment settings')
        start=read(args.resume/'COMPLETE.json')['step']
        if start>=args.steps:
            raise ValueError('steps must exceed saved optimizer step')
    out=new_output(args.output); write(out/'config.json',config)
    def memory(stage,**extra):
        free,total=torch.cuda.mem_get_info(device)
        record=dict(stage=stage,total_gib=total/2**30,free_gib=free/2**30,
            allocated_gib=torch.cuda.memory_allocated(device)/2**30,
            reserved_gib=torch.cuda.memory_reserved(device)/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,**extra)
        with (out/'memory.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(record,flush=True)
    memory('start')
    write(out/'run_plan.json',dict(total_optimizer_steps=args.steps,resume=str(args.resume),
        warning='Pilot fit does not establish held-out benefit',train_count=len(train),val_count=bundle['val_count']))
    # Encode posterior parameters before placing the 20B transformer on GPU.
    # Sample a fresh target posterior each microbatch; reference stays original cache/mode.
    post=out/'target_posteriors';post.mkdir()
    print(f'Encoding {len(train)} target VAE posteriors; Local conditions reuse existing cache.',flush=True)
    vae=AutoencoderKLQwenImage.from_pretrained(bundle['base_snapshot'],subfolder='vae',
        torch_dtype=torch.float32,local_files_only=True).requires_grad_(False).eval().to(device)
    mean=torch.tensor(vae.config.latents_mean).reshape(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std).reshape(1,-1,1,1,1)
    vae_scale=2**len(vae.temperal_downsample)
    if vae_scale!=8:
        raise ValueError('Expected VAE factor 8')
    with torch.inference_mode():
        for entry in train:
            rgb=target_image(entry['paired']).resize((2*bundle['height'],bundle['height']),Image.Resampling.BICUBIC)
            pixels=torch.from_numpy(np.array(rgb).copy()).permute(2,0,1).float()/127.5-1
            posterior=vae.encode(pixels[None,:,None].to(device)).latent_dist
            save_file({'mean':posterior.mean.cpu().contiguous(),'std':posterior.std.cpu().contiguous()},str(post/(entry['id']+'.safetensors')))
            print(f'Target posterior saved: {entry["id"]}',flush=True)
            del posterior,pixels
    del vae;gc.collect();torch.cuda.empty_cache()
    print('VAE released; loading frozen Edit transformer and epoch Panorama LoRA.',flush=True)
    transformer=QwenImageTransformer2DModel.from_pretrained(bundle['base_snapshot'],subfolder='transformer',
        torch_dtype=torch.bfloat16,local_files_only=True)
    if not transformer.zero_cond_t or transformer.config.guidance_embeds:
        raise ValueError('Expected Edit-2511 zero_cond_t')
    transformer.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],
        lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    result=set_peft_model_state_dict(transformer,load_file(str(base/'adapter_resume.safetensors')))
    if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
        raise ValueError('Panorama LoRA load incomplete')
    transformer.requires_grad_(False).eval().to(device)
    install_circular_padding(transformer,bundle['padding_columns'])
    transformer.enable_gradient_checkpointing()
    hidden=transformer.config.num_attention_heads*transformer.config.attention_head_dim
    world=WorldBranch(hidden,args.blocks,args.width).to(device)
    world.attach(transformer)
    memory('model_loaded')
    print(f'Frozen transformer/LoRA; trainable world parameters={sum(p.numel() for p in world.parameters()):,}',flush=True)
    optimizer=torch.optim.AdamW(world.parameters(),lr=args.lr,betas=(.9,.999),weight_decay=1e-5,eps=1e-6)
    if args.resume:
        world.load_state_dict(load_file(str(args.resume/'world.safetensors')),strict=True)
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location=device,weights_only=True))
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(bundle['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device)
    mean=mean.to(device);std=std.to(device)
    lossargs=SimpleNamespace(**REPO_PANORAMA)
    graphs={r['id']:tensorize(read(r['graph']),bundle['max_objects'],args.condition) for r in train}
    write(out/'parameters.json',dict(frozen_transformer=sum(p.numel() for p in transformer.parameters()),
        trainable_world=sum(p.numel() for p in world.parameters()),
        base_trainable=sum(p.numel() for p in transformer.parameters() if p.requires_grad),
        target_posterior_sha256={p.name:sha(p) for p in post.glob('*.safetensors')}))
    first_before={k:v.detach().cpu().clone() for k,v in world.state_dict().items()} if args.smoke else None
    started=time.monotonic();encoder_has_gradient=False
    for step in range(start,args.steps):
        # Step-local RNG makes restart independent of VAE/cache traversal.
        torch.manual_seed(args.seed+step)
        lr=args.lr*min(1.,(step+1)/max(1,args.warmup_steps))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);total=0.;component_totals={}
        for micro in range(args.accumulation):
            entry=train[int(torch.randint(len(train),(1,)))]
            posterior=load_file(str(post/(entry['id']+'.safetensors')),device='cuda:0')
            z=posterior['mean']+posterior['std']*torch.randn_like(posterior['std'])
            z=((z-mean)/std).to(torch.bfloat16)
            condition=load_file(entry['condition_cache'],device='cuda:0')
            noise=torch.randn_like(z)
            idx=int(torch.randint(scheduler.config.num_train_timesteps,(1,)))
            sigma=sigmas[idx].to(z.dtype);noisy=(1-sigma)*z+sigma*noise
            packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat([packed,condition['reference'][None]],1),
                timestep=timesteps[idx:idx+1]/1000,encoder_hidden_states=condition['embeddings'][None],
                encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            features=tuple(x.to(device) for x in graphs[entry['id']])
            if args.smoke and step==0 and micro==0:
                with torch.no_grad(),training_autocast('cuda'):
                    world.set_world(None);baseline=transformer(**kwargs)[0]
                    world.set_world(features);zero=transformer(**kwargs)[0]
                    torch.testing.assert_close(zero,baseline,rtol=0,atol=0)
                    del baseline,zero
                world.set_world(None)
                memory('zero_gate_check_passed')
            storage=ActivationStorage(enabled=args.activation_storage=='cpu')
            with storage.context():
                with training_autocast('cuda'):
                    world.set_world(features)
                    prediction=transformer(**kwargs)[0][:,:packed.shape[1]]
                    prediction=Pipeline._unpack_latents(prediction,bundle['height'],2*bundle['height'],vae_scale)
                    mask=torch.ones(1,1,z.shape[3],z.shape[4],device=device,dtype=torch.bool)
                    loss,components=panorama_loss(prediction,noise-z,mask,torch.ones(1,device=device,dtype=torch.bool),
                        torch.ones(1,device=device),lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
                if not torch.isfinite(loss):raise FloatingPointError(entry['id'])
                if micro==0:
                    memory('before_backward',step=step+1,saved_activation_copies=storage.count,
                           saved_activation_copy_gib=storage.bytes/2**30)
                (loss/args.accumulation).backward()
            if micro==0:memory('after_backward',step=step+1)
            total+=float(loss.detach())/args.accumulation
            for k,v in components.items():component_totals[k]=component_totals.get(k,0.)+float(v.detach())/args.accumulation
            world.set_world(None)
            del prediction,loss,kwargs,packed,condition,posterior,z,noise,noisy
        if any(p.grad is not None for p in transformer.parameters()):
            raise AssertionError('Frozen base/LoRA received gradients')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters()):
            raise FloatingPointError('Nonfinite world gradient')
        encoder_has_gradient |= any(p.grad is not None and bool(torch.count_nonzero(p.grad)>0) for p in world.encoder.parameters())
        norm=torch.nn.utils.clip_grad_norm_(world.parameters(),1.)
        optimizer.step()
        record=dict(step=step+1,loss=total,components=component_totals,lr=lr,grad_norm=float(norm),
            encoder_nonzero_gradient_seen=bool(encoder_has_gradient),world=world.diagnostics(),
            elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(record,flush=True)
        if (step+1)%args.save_every==0 or step+1==args.steps:
            ckpt=out/f'checkpoint-step{step+1:06d}';ckpt.mkdir()
            state={k:v.detach().cpu().contiguous() for k,v in world.state_dict().items()}
            save_file(state,str(ckpt/'world.safetensors'));torch.save(optimizer.state_dict(),ckpt/'optimizer.pt')
            write(ckpt/'config.json',config)
            restored=load_file(str(ckpt/'world.safetensors'))
            if not all(torch.equal(v,restored[k]) for k,v in state.items()):raise AssertionError('Checkpoint mismatch')
            if args.smoke:
                if not encoder_has_gradient or not any(not torch.equal(v,first_before[k]) for k,v in state.items()):
                    raise AssertionError('Smoke: no encoder gradients or parameter updates')
                write(ckpt/'SMOKE_COMPLETE.json',dict(zero_gate_baseline_exact=True,base_and_lora_frozen=True,
                    encoder_gradient=True,parameters_changed=True,saved_state_roundtrip=True,
                    note='GPU reload inference still required; no quality claim'))
            write(ckpt/'COMPLETE.json',dict(step=step+1,feature_version=FEATURE_VERSION,
                world_sha256=sha(ckpt/'world.safetensors'),optimizer_sha256=sha(ckpt/'optimizer.pt')))
    write(out/'COMPLETE.json',dict(optimizer_steps=args.steps,checkpoint=str(ckpt),purpose=args.purpose,
        note='Training completed; generation benefit not established'))


if __name__=='__main__':
    main()
