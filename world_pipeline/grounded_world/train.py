"""User-run matched 128-image A/B pilot; frozen Panorama base, fresh World."""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from ..build_polished_world import read,atomic_json
from ..common import new_output
from ..object_world.data import DEFAULT_PACK,tensorize,atomic_tensors,identity
from ..object_world.runtime import object_region_loss
from ..oracle_memory import training_autocast,ActivationStorage
from ..scaled_world.runtime import EpochSampler,seed_step,learning_rate
from ..id_binding_world.parallel import seed_microbatch
from ..scaled_world.binding import apply_scale
from .data import inputs,contract,check_rows,source_meta,load_support,support_path,smoke_rows,DEFAULT_PANEL,DEFAULT_SUPPORT
from .runtime import VERSION,WorldBranch
from .supervision import POLICY,sample


def norms(world):
    groups={key:[] for key in ('semantic','geometry','fusion','qk','value_output','gates')}
    for name,p in world.named_parameters():
        if p.grad is None:continue
        if name.startswith('encoder.semantic.'):key='semantic'
        elif name.startswith('encoder.geometry.'):key='geometry'
        elif name.startswith('encoder.'):key='fusion'
        elif name.endswith('.gate'):key='gates'
        elif '.q.' in name or '.k.' in name:key='qk'
        else:key='value_output'
        groups[key].append(p.grad.detach().float().square().sum())
    return {k:float(torch.stack(v).sum().sqrt()) if v else 0. for k,v in groups.items()}


def draw_latents(posterior,mean,std,seed,index,num_timesteps):
    """Explicit CPU generator: A/B/continuations get identical actual inputs."""
    if any(t.device.type!='cpu' for t in (posterior['mean'],posterior['std'],mean,std)):
        raise ValueError('Paired latent sampling uses CPU tensors')
    generator=torch.Generator(device='cpu').manual_seed(seed*1000003+index)
    sampled=posterior['mean'].float()+posterior['std'].float()*torch.randn(posterior['std'].shape,generator=generator)
    clean=((sampled-mean)/std).to(torch.bfloat16)
    noise=torch.randn(clean.shape,generator=generator,dtype=torch.float32).to(torch.bfloat16)
    timestep=int(torch.randint(num_timesteps,(1,),generator=generator))
    return clean,noise,timestep


def configuration(plan,args,rows):
    from qwen_edit_pano.common import versions
    from ..spatial_world.projection import POLICY as PROJECTION_POLICY
    from ..build_polished_world import FRAME
    return dict(version=VERSION,pack=str(args.pack.resolve()),pack_contract=contract(plan),panel=str(args.panel.resolve()),
        train_ids=[r['id'] for r in rows],sample_sources=[identity(r) for r in rows],
        cache_paths={r['id']:{k:r[k] for k in ('condition_cache','posterior_cache','region_cache')} for r in rows},
        group=args.group,smoke=args.smoke,seed=args.seed,blocks=[20,40,59],width=256,
        binding_block=40,binding_weight=0. if args.group=='A' else .01,binding_ramp_steps=20 if not args.smoke else 0,
        region_weight=.25,pano_lora_scale=.5,lr=1e-4,adam_eps=1e-8,weight_decay=1e-5,betas=[.9,.999],clip_norm=1.,warmup_steps=0 if args.smoke else 20,
        accumulation=4,activation_storage='cpu',support_root=str(args.support.resolve()),binding_policy=POLICY,
        category_encoding='frozen_Qwen_name_features_separate_geometry_concat',old_world_loaded=False,
        frame=FRAME,versions=versions(),routing_policy=dict(projection=PROJECTION_POLICY,device='CPU in training and inference',candidate='positive input-derived coverage',null_key_value='exactly zero'),
        precision='frozen base BF16; World master/Adam FP32; scoped BF16 autocast without cast cache',
        random_policy='local CPU generator per global microbatch for posterior/noise/timestep; independent NumPy binding sampling; fixed Python yaw seed',
        initialization='fresh seeded World; zero gates match frozen Panorama-only base',file_content_hashes_computed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=('A','B'),required=True)
    p.add_argument('--pack',type=Path,default=DEFAULT_PACK);p.add_argument('--panel',type=Path,default=DEFAULT_PANEL)
    p.add_argument('--support',type=Path,default=DEFAULT_SUPPORT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--stop-file',type=Path);args=p.parse_args()
    if args.smoke:args.steps=2
    if args.steps<1 or (not args.smoke and args.steps<=20):p.error('Pilot target must exceed the 20-step warmup')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('One GPU per arm; run A and B as independent jobs')
    torch.set_num_threads(4);plan,_,selected,features=inputs(args.pack,args.panel)
    rows=smoke_rows(selected) if args.smoke else selected;config=configuration(plan,args,rows);check_rows(rows,plan)
    supports={}
    for row in rows:
        path=support_path(args.support,row['id']);meta=read(path.with_suffix('.json'))
        if meta['source']!=source_meta(row,plan):raise ValueError('Prepared support source changed')
        supports[row['id']]=load_support(args.support,row['id'])
    start=0
    if args.resume:
        if read(args.resume/'config.json')!=config or read(args.resume/'COMPLETE.json')['version']!=VERSION:
            raise ValueError('Resume requires the same new version/group/panel/configuration')
        start=read(args.resume/'COMPLETE.json')['step']
        if start>=args.steps:raise ValueError('Checkpoint already reached the requested target')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('User must allocate one BF16 CUDA GPU')
    device=torch.device('cuda:0');out=new_output(args.output)
    atomic_json(out/'config.json',config);atomic_json(out/'run_plan.json',dict(steps=args.steps,resume=str(args.resume) if args.resume else None,
        train_count=len(rows),exposures=args.steps*4,passes=args.steps*4/len(rows),no_automatic_evaluation=True))
    atomic_json(out/'runtime_environment.json',dict(gpu_name=torch.cuda.get_device_name(device),
        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,torch=torch.__version__,cuda=torch.version.cuda))
    from diffusers import AutoencoderKLQwenImage,QwenImageTransformer2DModel,FlowMatchEulerDiscreteScheduler,QwenImageEditPlusPipeline as Pipeline
    from peft import LoraConfig,set_peft_model_state_dict
    from qwen_edit_pano.paired_circular import install_circular_padding
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    if any(plan[k]!=pano[k] for k in ('base_snapshot','height','padding_columns','prompt')):raise ValueError('Panorama/pack contract mismatch')
    model=QwenImageTransformer2DModel.from_pretrained(plan['base_snapshot'],subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    if not model.config.zero_cond_t:raise ValueError('Expected the native Edit zero_cond_t architecture')
    model.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    loaded=set_peft_model_state_dict(model,load_file(str(base/'adapter_resume.safetensors')))
    if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):raise ValueError('Incomplete Panorama LoRA')
    atomic_json(out/'pano_scale_check.json',apply_scale(model,.5,'default'))
    model.to(device);install_circular_padding(model,plan['padding_columns']);model.enable_gradient_checkpointing()
    seed_step(args.seed,0);hidden=model.config.num_attention_heads*model.config.attention_head_dim
    world=WorldBranch(hidden,features,height=plan['height'],padding=plan['padding_columns']).to(device)
    optimizer=torch.optim.AdamW(world.parameters(),lr=config['lr'],betas=tuple(config['betas']),weight_decay=config['weight_decay'],eps=config['adam_eps'])
    if args.resume:
        world.load_state_dict(load_file(str(args.resume/'world.safetensors')),strict=True)
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location=device,weights_only=True))
    else:atomic_tensors(out/'initial_world.safetensors',world.state_dict())
    world.attach(model)
    if any(p.requires_grad for p in model.parameters()):raise AssertionError('Base must be frozen')
    atomic_json(out/'parameters.json',dict(trainable_world=sum(p.numel() for p in world.parameters()),base_trainable=0,
        old_world_loaded=False,master_dtypes=sorted({str(p.dtype) for p in world.parameters()})))
    if any(p.dtype!=torch.float32 for p in world.parameters()):raise AssertionError('World masters must be FP32')
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1)
    std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
    if 2**len(vae['temperal_downsample'])!=8:raise ValueError('Unexpected VAE scale')
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device);lossargs=SimpleNamespace(**REPO_PANORAMA)
    tensors={r['id']:tensorize(r,plan['max_objects']) for r in rows};sampler=EpochSampler(len(rows),args.seed)
    started=time.monotonic();step_times=[];semantic_gradient=False;body_gradient=False;zero_equivalent=False
    for step in range(start,args.steps):
        tick=time.monotonic();seed_step(args.seed,step);lr=learning_rate(config['lr'],config['warmup_steps'],step)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);total=0.;parts_total={};ids=[];noise_log=[];binding_log=[];routing_log=[]
        beta=config['binding_weight']*min(1.,(step+1)/max(1,config['binding_ramp_steps']))
        for micro in range(4):
            index=step*4+micro;seed_microbatch(args.seed,index);row=rows[sampler.index(index)];ids.append(row['id'])
            posterior=load_file(row['posterior_cache'],device='cpu');condition=load_file(row['condition_cache'],device=str(device))
            z,noise,idx=draw_latents(posterior,mean,std,args.seed,index,scheduler.config.num_train_timesteps)
            z=z.to(device);noise=noise.to(device);sigma=sigmas[idx].to(z.dtype)
            noisy=(1-sigma)*z+sigma*noise;packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=timesteps[idx:idx+1]/1000,
                encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            state=tuple(t.to(device) for t in tensors[row['id']]);weights=load_file(row['region_cache'],device=str(device))['weights']
            if args.smoke and step==0 and micro==0:
                with torch.no_grad(),training_autocast('cuda'):
                    world.set_world(None);baseline=model(**kwargs)[0]
                    world.set_world(state);zero=model(**kwargs)[0]
                    torch.testing.assert_close(zero,baseline,rtol=0,atol=0);zero_equivalent=True;del baseline,zero
                world.clear_context()
            labels=None
            if beta:
                labels,label_meta=sample(supports[row['id']],args.seed*1000003+index)
                labels=tuple(t.to(device) for t in labels);binding_log.append(label_meta)
            with ActivationStorage(enabled=True).context():
                with training_autocast('cuda'):
                    world.set_supervision(labels);world.set_world(state);routing_log.append(dict(world.routing_meta))
                    prediction=model(**kwargs)[0][:,:packed.shape[1]]
                    prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],8)
                    global_loss,parts=panorama_loss(prediction,noise-z,torch.ones(1,1,*z.shape[-2:],device=device,dtype=torch.bool),
                        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
                    region=object_region_loss(prediction,noise-z,weights)
                    binding=world.take_binding_loss() if labels is not None else region.new_zeros(())
                    loss=global_loss+.25*region+beta*binding
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/4).backward()
            if args.smoke and beta and step==0 and micro==0:
                if not len(labels[2]) or not all(p.grad is not None and bool(p.grad.count_nonzero()) for p in (
                    world.branches['40'].q.weight,world.branches['40'].k.weight,world.encoder.semantic[0].weight)):
                    raise AssertionError('Binding must train actual Q/K/semantic projection at zero gate')
            total+=float(loss.detach())/4
            for k,v in {**parts,'object_region':region,'object_region_weighted':.25*region,'binding':binding,'binding_weighted':beta*binding}.items():
                parts_total[k]=parts_total.get(k,0.)+float(v.detach())/4
            noise_log.append(dict(index=idx,sigma=float(sigma)));world.clear_context()
            del prediction,loss,global_loss,region,binding,packed,kwargs,posterior,condition,weights,z,noise,noisy,state,labels
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base got gradients')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters()):raise FloatingPointError('Nonfinite gradients')
        gn=norms(world);semantic_gradient|=gn['semantic']>0
        body_gradient|=any(p.grad is not None and bool(p.grad.count_nonzero()) for name,p in world.named_parameters() if '.v.weight' in name or '.out.weight' in name)
        norm=torch.nn.utils.clip_grad_norm_(world.parameters(),config['clip_norm']);optimizer.step();elapsed=time.monotonic()-started
        step_times.append(time.monotonic()-tick);seconds=sum(step_times[-10:])/len(step_times[-10:])
        log=dict(step=step+1,group=args.group,loss=total,components=parts_total,binding_weight=beta,lr=lr,grad_norm=float(norm),
            gradient_norms=gn,sample_ids=ids,noise_samples=noise_log,binding_samples=binding_log,routing_samples=routing_log,
            passes=(step+1)*4/len(rows),world=world.diagnostics(),features=world.encoder.feature_diagnostics,
            elapsed_seconds=elapsed,recent_seconds_per_step=seconds,eta_seconds=(args.steps-step-1)*seconds,
            peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,base_trainable=0)
        with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(log,allow_nan=False)+'\n')
        print(log,flush=True);stop=args.stop_file is not None and args.stop_file.exists()
        if (step+1)%(2 if args.smoke else 50)==0 or step+1==args.steps or stop:
            ckpt=out/f'checkpoint-step{step+1:06d}';ckpt.mkdir()
            saved={k:v.detach().cpu().contiguous() for k,v in world.state_dict().items()}
            atomic_tensors(ckpt/'world.safetensors',saved);torch.save(optimizer.state_dict(),ckpt/'optimizer.pt');atomic_json(ckpt/'config.json',config)
            restored=load_file(str(ckpt/'world.safetensors'))
            if not all(torch.equal(v,restored[k]) for k,v in saved.items()):raise AssertionError('Checkpoint roundtrip mismatch')
            if args.smoke:
                if not zero_equivalent or not semantic_gradient or not body_gradient:raise AssertionError('Two-step smoke did not train semantic/value paths')
                atomic_json(ckpt/'SMOKE_COMPLETE.json',dict(zero_gate_panorama_exact=True,semantic_gradient=True,value_output_gradient=True,
                    base_trainable=0,old_world_loaded=False,group=args.group,binding_weight=config['binding_weight'],peak_gpu_gib=log['peak_gpu_gib']))
            atomic_json(ckpt/'COMPLETE.json',dict(version=VERSION,step=step+1))
            if stop and step+1<args.steps:atomic_json(out/'STOPPED.json',dict(checkpoint=str(ckpt)));return
    atomic_json(out/'COMPLETE.json',dict(version=VERSION,steps=args.steps,checkpoint=str(ckpt),group=args.group,
        note='Training ended; semantic generation quality is not certified. No automatic evaluation.'))


if __name__=='__main__':main()
