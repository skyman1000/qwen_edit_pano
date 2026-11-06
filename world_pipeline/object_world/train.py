"""Frozen Panorama LoRA + semantic/Fourier World + instance-balanced velocity loss."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from ..build_polished_world import read, atomic_json
from ..common import new_output
from ..oracle_memory import training_autocast, ActivationStorage
from ..scaled_world.binding import apply_scale
from ..scaled_world.runtime import EpochSampler, seed_step, learning_rate
from ..inference_diagnostics import RelativeResidualRecorder
from .data import DEFAULT_PACK, load_pack, tensorize, atomic_tensors
from .runtime import VERSION, WorldBranch, object_region_loss


def gradient_norms(world):
    """Read-only pre-clip norms; disjoint groups, no extra backward passes."""
    import torch
    groups={key:[] for key in ('semantic','geometry','encoder_other','adapter_attention','gates')}
    for name,parameter in world.named_parameters():
        if parameter.grad is None:continue
        if name.startswith('encoder.semantic.'):key='semantic'
        elif name.startswith('encoder.geometry.'):key='geometry'
        elif name.startswith('encoder.'):key='encoder_other'
        elif name.endswith('.gate'):key='gates'
        else:key='adapter_attention'
        groups[key].append(parameter.grad.detach().float().square().sum())
    return {key:float(torch.stack(values).sum().sqrt()) if values else 0. for key,values in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,default=DEFAULT_PACK)
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--warmup-steps',type=int,default=500)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--blocks',type=int,nargs='+',default=[20,40,59])
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--pano-lora-scale',type=float,default=.5)
    p.add_argument('--region-weight',type=float,default=.25)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--activation-storage',choices=('cpu','gpu'),default='cpu')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--resume',type=Path,help='Checkpoint to resume into a NEW output directory')
    args=p.parse_args();args.pack=args.pack.resolve()
    if args.smoke:args.steps=2;args.warmup_steps=0
    if (min(args.steps,args.accumulation,args.width,args.save_every)<1 or args.width%8
            or not 0<=args.warmup_steps<args.steps or not 0<args.lr<1 or not 0<=args.region_weight<=10
            or not 0<args.pano_lora_scale<=2 or not args.blocks or min(args.blocks)<0
            or len(set(args.blocks))!=len(args.blocks)):
        p.error('Invalid training dimensions/coefficients')
    plan,records=load_pack(args.pack)
    train=[row for row in records if row['split']=='train' and (not args.smoke or row['id'] in plan['smoke_ids'])]
    base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    if any(plan[key]!=pano[key] for key in ('base_snapshot','height','padding_columns','prompt')):
        raise ValueError('Frozen Panorama checkpoint configuration differs from the training pack')
    architecture=read(Path(plan['base_snapshot'])/'transformer/config.json')
    if not architecture['zero_cond_t'] or max(args.blocks)>=architecture['num_layers']:
        raise ValueError('Invalid Edit architecture/injection blocks')
    missing={key:sum(not Path(row[key]).is_file() for row in train) for key in ('condition_cache','posterior_cache','region_cache')}
    missing['class_features']=int(not (args.pack/'class_features.safetensors').is_file())
    features=None;semantic_meta=None
    if not missing['class_features']:
        import torch
        from safetensors.torch import load_file
        from .cache import semantic_phrases
        semantic_meta=read(args.pack/'class_features.json')
        expected_semantic=dict(policy=plan['semantic_policy'],names=plan['names'],phrases=semantic_phrases(plan['names']),
            base_snapshot=plan['base_snapshot'],pooling='mean of name tokens only; padding/prefix excluded')
        if semantic_meta!=expected_semantic:
            raise ValueError('Class semantics do not match the dataset/model/pooling policy')
        features=load_file(str(args.pack/'class_features.safetensors'))['features']
        if (features.shape!=(42,architecture['joint_attention_dim']) or not torch.isfinite(features).all()
                or any(features[int(key)].norm()==0 for key in plan['names'])):
            raise ValueError('Class semantic table dimensions/support differ from the task categories')
    print(dict(version=VERSION,train=len(train),val=plan['val_count'],steps=args.steps,
        sample_exposures=args.steps*args.accumulation,passes=args.steps*args.accumulation/len(train),
        missing_caches=missing,spatial_bias=False),flush=True)
    if args.preflight:
        print('CPU CONFIG CHECK PASSED; missing caches must be built before GPU training.',flush=True);return
    if any(missing.values()):raise ValueError('Run cache-smoke/cache before training: '+str(missing))
    if args.output is None:p.error('--output required for training')
    import torch
    from safetensors.torch import load_file
    from diffusers import AutoencoderKLQwenImage,QwenImageTransformer2DModel,FlowMatchEulerDiscreteScheduler
    from diffusers import QwenImageEditPlusPipeline as Pipeline
    from peft import LoraConfig,set_peft_model_state_dict
    from qwen_edit_pano.paired_circular import install_circular_padding
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    from .cache import check_condition,check_posterior
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Allocate one BF16 CUDA GPU')
    for row in train:
        check_condition(row['condition_cache']);check_posterior(row['posterior_cache'],plan['height'])
    config=dict(version=VERSION,pack=str(args.pack),pack_plan=plan,condition='full',
        pano_lora_scale=args.pano_lora_scale,blocks=args.blocks,width=args.width,
        lr=args.lr,warmup_steps=args.warmup_steps,accumulation=args.accumulation,seed=args.seed,
        activation_storage=args.activation_storage,smoke=args.smoke,region_weight=args.region_weight,
        geometry_loss='flow+0.5cube+0.5yaw+region_weight*object_instance_mean',
        semantic_cache=semantic_meta,file_content_hashes_computed=False)
    start=0
    if args.resume:
        if read(args.resume/'config.json')!=config:
            raise ValueError('Resume config differs; use identical settings and a new output')
        start=read(args.resume/'COMPLETE.json')['step']
        if start>=args.steps:raise ValueError('Total steps must exceed checkpoint step')
    out=new_output(args.output);atomic_json(out/'config.json',config)
    atomic_json(out/'run_plan.json',dict(total_steps=args.steps,resume=str(args.resume) if args.resume else None,
        train_count=len(train),val_count=plan['val_count'],note='GT Oracle training; no benefit claim until held-out generation'))
    device=torch.device('cuda:0')
    vae_config=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae_config['latents_mean'],device=device).reshape(1,-1,1,1,1)
    std=torch.tensor(vae_config['latents_std'],device=device).reshape(1,-1,1,1,1)
    vae_scale=2**len(vae_config['temperal_downsample'])
    if vae_scale!=8:raise ValueError('Expected VAE scale eight')
    transformer=QwenImageTransformer2DModel.from_pretrained(plan['base_snapshot'],subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    transformer.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    result=set_peft_model_state_dict(transformer,load_file(str(base/'adapter_resume.safetensors')))
    if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
        raise ValueError('Incomplete frozen Panorama LoRA load')
    atomic_json(out/'pano_scale_check.json',apply_scale(transformer,args.pano_lora_scale,'default'))
    transformer.to(device);install_circular_padding(transformer,plan['padding_columns'])
    transformer.enable_gradient_checkpointing()
    seed_step(args.seed,0)
    hidden=transformer.config.num_attention_heads*transformer.config.attention_head_dim
    world=WorldBranch(hidden,features,args.blocks,args.width).to(device)
    world.attach(transformer)
    # Read-only metrics: absolute deltas have different activation scales across
    # layers; also record their ratio to the input target RMS. No output changes.
    relative_recorder=RelativeResidualRecorder(world)
    optimizer=torch.optim.AdamW(world.parameters(),lr=args.lr,betas=(.9,.999),weight_decay=1e-5,eps=1e-6)
    atomic_json(out/'optimizer_policy.json',dict(name='AdamW',lr=args.lr,betas=[.9,.999],
        weight_decay=1e-5,eps=1e-6,clip_norm=1.,microbatch_size=1,accumulation=args.accumulation,
        warmup_steps=args.warmup_steps,schedule='linear warmup then constant',
        noise_sampling='uniform native scheduler index',region_weight=args.region_weight))
    if args.resume:
        state=load_file(str(args.resume/'world.safetensors'))
        torch.testing.assert_close(state['encoder.class_features'],world.encoder.class_features.cpu(),rtol=0,atol=0)
        world.load_state_dict(state,strict=True)
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location=device,weights_only=True))
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device)
    lossargs=SimpleNamespace(**REPO_PANORAMA)
    tensors={row['id']:tensorize(row,plan['max_objects']) for row in train}
    atomic_json(out/'parameters.json',dict(trainable_world=sum(p.numel() for p in world.parameters() if p.requires_grad),
        frozen_base=sum(p.numel() for p in transformer.parameters()),base_trainable=sum(p.numel() for p in transformer.parameters() if p.requires_grad),
        semantic_features_frozen=not world.encoder.class_features.requires_grad))
    sampler=EpochSampler(len(train),args.seed);started=time.monotonic();encoder_gradient=False;semantic_gradient=False
    initial={key:v.cpu().clone() for key,v in world.state_dict().items()} if args.smoke else None
    for step in range(start,args.steps):
        relative_recorder.reset()
        seed_step(args.seed,step);lr=learning_rate(args.lr,args.warmup_steps,step)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);total=0.;components_total={};sample_ids=[]
        feature_samples=[];noise_samples=[]
        for micro in range(args.accumulation):
            row=train[sampler.index(step*args.accumulation+micro)];sample_ids.append(row['id'])
            posterior=load_file(row['posterior_cache'],device='cuda:0')
            z=posterior['mean']+posterior['std']*torch.randn_like(posterior['std'])
            z=((z-mean)/std).to(torch.bfloat16)
            condition=load_file(row['condition_cache'],device='cuda:0')
            weights=load_file(row['region_cache'],device='cuda:0')['weights']
            noise=torch.randn_like(z);idx=int(torch.randint(scheduler.config.num_train_timesteps,(1,)))
            sigma=sigmas[idx].to(z.dtype);noisy=(1-sigma)*z+sigma*noise
            packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat([packed,condition['reference'][None]],1),
                timestep=timesteps[idx:idx+1]/1000,encoder_hidden_states=condition['embeddings'][None],
                encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            world_input=tuple(t.to(device) for t in tensors[row['id']])
            if args.smoke and step==0 and micro==0:
                with torch.no_grad(),training_autocast('cuda'):
                    world.set_world(None);baseline=transformer(**kwargs)[0]
                    world.set_world(world_input);zero=transformer(**kwargs)[0]
                    torch.testing.assert_close(zero,baseline,rtol=0,atol=0)
                    del baseline,zero
                world.set_world(None)
            storage=ActivationStorage(enabled=args.activation_storage=='cpu')
            with storage.context():
                with training_autocast('cuda'):
                    world.set_world(world_input)
                    feature_samples.append(dict(world.encoder.feature_diagnostics))
                    noise_samples.append(dict(scheduler_index=idx,sigma=float(sigma),timestep=float(timesteps[idx])))
                    prediction=transformer(**kwargs)[0][:,:packed.shape[1]]
                    prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],vae_scale)
                    mask=torch.ones(1,1,z.shape[3],z.shape[4],device=device,dtype=torch.bool)
                    global_loss,components=panorama_loss(prediction,noise-z,mask,
                        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),
                        lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
                    region=object_region_loss(prediction,noise-z,weights)
                    loss=global_loss+args.region_weight*region
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/args.accumulation).backward()
            total+=float(loss.detach())/args.accumulation
            components={**components,'object_region':region.detach(),'object_region_weighted':args.region_weight*region.detach()}
            for key,value in components.items():components_total[key]=components_total.get(key,0.)+float(value)/args.accumulation
            world.set_world(None)
            del prediction,loss,global_loss,region,components,weights,kwargs,packed,condition,posterior,z,noise,noisy,world_input
        if any(p.grad is not None for p in transformer.parameters()):raise AssertionError('Frozen base received gradients')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters()):raise FloatingPointError('World gradients')
        encoder_gradient|=any(p.grad is not None and bool(torch.count_nonzero(p.grad)>0) for p in world.encoder.parameters())
        semantic_gradient|=any(p.grad is not None and bool(torch.count_nonzero(p.grad)>0) for p in world.encoder.semantic.parameters())
        branch_gradient_norms=gradient_norms(world)
        norm=torch.nn.utils.clip_grad_norm_(world.parameters(),1.)
        optimizer.step()
        log=dict(step=step+1,loss=total,components=components_total,lr=lr,grad_norm=float(norm),sample_ids=sample_ids,
            passes_seen=(step+1)*args.accumulation/len(train),encoder_nonzero_gradient_seen=encoder_gradient,
            semantic_projection_nonzero_gradient_seen=semantic_gradient,
            feature_scales=world.encoder.feature_diagnostics,
            feature_scale_microbatches=feature_samples,noise_samples=noise_samples,
            gradient_norms_before_clip=branch_gradient_norms,clip_factor=min(1.,1./(float(norm)+1e-6)),
            weighted_region_to_global=components_total['object_region_weighted']/max(
                total-components_total['object_region_weighted'],1e-12),
            gates_after_update={key:float(torch.tanh(branch.gate)) for key,branch in world.branches.items()},
            world=world.diagnostics(),world_relative=relative_recorder.summary(),
            elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'train_log.jsonl').open('a') as handle:handle.write(json.dumps(log)+'\n')
        print(log,flush=True)
        if (step+1)%args.save_every==0 or step+1==args.steps:
            ckpt=out/f'checkpoint-step{step+1:06d}';ckpt.mkdir()
            state={key:value.detach().cpu().contiguous() for key,value in world.state_dict().items()}
            atomic_tensors(ckpt/'world.safetensors',state)
            torch.save(optimizer.state_dict(),ckpt/'optimizer.pt')
            atomic_json(ckpt/'config.json',config)
            restored=load_file(str(ckpt/'world.safetensors'))
            if not all(torch.equal(value,restored[key]) for key,value in state.items()):raise AssertionError('Checkpoint roundtrip')
            if args.smoke:
                if not encoder_gradient or not semantic_gradient or not any(not torch.equal(value,initial[key]) for key,value in state.items()):
                    raise AssertionError('Smoke did not update World encoder')
                atomic_json(ckpt/'SMOKE_COMPLETE.json',dict(zero_gate_baseline_exact=True,base_frozen=True,
                    semantic_features_frozen=True,encoder_gradient=True,semantic_projection_gradient=True,parameters_changed=True,
                    object_region_loss_enabled=args.region_weight>0,note='GPU generation still needed; not a quality result'))
            atomic_json(ckpt/'COMPLETE.json',dict(step=step+1,version=VERSION))
    atomic_json(out/'COMPLETE.json',dict(steps=args.steps,checkpoint=str(ckpt),note='Training complete; held-out quality not established'))


if __name__=='__main__':main()
