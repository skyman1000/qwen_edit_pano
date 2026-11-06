"""Warm-start matched spatial/control arms; same A losses and native Edit path."""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from ..build_polished_world import read, atomic_json
from ..common import new_output
from ..oracle_memory import training_autocast, ActivationStorage
from ..scaled_world.binding import apply_scale
from ..scaled_world.runtime import EpochSampler, seed_step, learning_rate
from ..inference_diagnostics import RelativeResidualRecorder
from ..object_world.data import DEFAULT_PACK, load_pack, tensorize, atomic_tensors
from .runtime import VERSION, WorldBranch, object_region_loss
from ..id_binding_world.supervision import POLICY, sample_binding
from .projection import POLICY as ROUTING_POLICY
from .data import DEFAULT_INITIAL, select_train
from .runtime import load_world, LEGACY_VERSION
from ..id_binding_world.parallel import Parallel, micro_indices, seed_microbatch, average_gradients, check_replicas
from . import calibration


def gradient_norms(world):
    """Read-only pre-clip norms; disjoint groups, no extra backward passes."""
    import torch
    groups={key:[] for key in ('category','geometry','encoder_other','adapter_attention','gates','spatial','spatial_gates')}
    for name,parameter in world.named_parameters():
        if parameter.grad is None:continue
        if '.local.' in name:key='spatial_gates' if name.endswith('.gate') else 'spatial'
        elif name.startswith('encoder.category.'):key='category'
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
    p.add_argument('--group',choices=('A',),default='A')
    p.add_argument('--mode',choices=('spatial','control'),default='spatial')
    p.add_argument('--init-checkpoint',type=Path,default=DEFAULT_INITIAL)
    p.add_argument('--train-limit',type=int,default=32,help='0 uses all training data; val is never optimized')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--binding-ramp-steps',type=int,default=200)
    p.add_argument('--warmup-steps',type=int,default=20)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--blocks',type=int,nargs='+',default=[20,40,59])
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--pano-lora-scale',type=float,default=.5)
    p.add_argument('--region-weight',type=float,default=.25)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--activation-storage',choices=('cpu','gpu'),default='cpu')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--local-only-calibration',action='store_true',help='Fresh local-only optimizer from a trained spatial v1 checkpoint; unchanged forward/loss')
    p.add_argument('--resume',type=Path,help='Checkpoint to resume into a NEW output directory')
    p.add_argument('--stop-file',type=Path,help='Save at next complete optimizer step, then exit for restart')
    args=p.parse_args();args.pack=args.pack.resolve()
    if args.smoke:args.steps=2;args.warmup_steps=0;args.binding_ramp_steps=0
    binding_weight=0.
    if args.local_only_calibration and (args.mode!='spatial' or args.smoke or args.region_weight!=.25):
        p.error('Local calibration requires spatial mode, region 0.25, and a trained non-smoke initialization')
    adam_eps=calibration.POLICY['adam_eps'] if args.local_only_calibration else 1e-6
    args.init_checkpoint=args.init_checkpoint.resolve()
    replicas=int(os.environ.get('WORLD_SIZE','1'))
    if (min(args.steps,args.accumulation,args.width,args.save_every)<1 or args.width%8
            or not 0<=args.warmup_steps<args.steps or not 0<args.lr<1 or not 0<=args.region_weight<=10
            or not 0<args.pano_lora_scale<=2 or not args.blocks or min(args.blocks)<0
            or len(set(args.blocks))!=len(args.blocks) or 40 not in args.blocks or args.binding_ramp_steps<0):
        p.error('Invalid training dimensions/coefficients')
    if replicas not in (1,2) or args.accumulation%replicas:
        p.error('Use one/two replicas and a global accumulation divisible by the replica count')
    plan,records=load_pack(args.pack)
    train=select_train(records,args.train_limit,args.seed,plan['smoke_ids'] if args.smoke else None)
    if not train:raise ValueError('No training samples selected')
    initial_config=read(args.init_checkpoint/'config.json')
    initial_complete=read(args.init_checkpoint/'COMPLETE.json')
    initial_step,sampling_offset=calibration.initialization(initial_config,initial_complete,args.local_only_calibration)
    if initial_config['pack_plan']!=plan:raise ValueError('Initialization must be bound to this exact training pack')
    if any(initial_config[k]!=getattr(args,k) for k in ('blocks','width','pano_lora_scale','region_weight')):
        raise ValueError('Warm-start architecture/objective differs')
    if args.local_only_calibration and ([r['id'] for r in train]!=initial_config['train_ids']
            or any(initial_config[k]!=getattr(args,k) for k in ('seed','accumulation'))):
        raise ValueError('Local calibration reuses the same train-only panel, seed and global batch')
    base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    if any(plan[key]!=pano[key] for key in ('base_snapshot','height','padding_columns','prompt')):
        raise ValueError('Frozen Panorama checkpoint configuration differs from the training pack')
    architecture=read(Path(plan['base_snapshot'])/'transformer/config.json')
    if not architecture['zero_cond_t'] or max(args.blocks)>=architecture['num_layers']:
        raise ValueError('Invalid Edit architecture/injection blocks')
    missing={key:sum(not Path(row[key]).is_file() for row in train) for key in ('condition_cache','posterior_cache','region_cache')}
    missing['instance_supervision']=sum(not Path(row['object_region_supervision']['path']).is_file() for row in train) if binding_weight else 0
    if int(os.environ.get('RANK','0'))==0:print(dict(version=VERSION,train=len(train),val=plan['val_count'],steps=args.steps,
        sample_exposures=args.steps*args.accumulation,passes=args.steps*args.accumulation/len(train),
        missing_caches=missing,spatial_bias=False,group=args.group,category_encoding='learned_ID_42x256',binding_weight=binding_weight,
        replicas=replicas,global_accumulation=args.accumulation,local_accumulation=args.accumulation//replicas),flush=True)
    if args.preflight:
        if any(missing.values()):raise ValueError('Missing selected caches: '+str(missing))
        import torch
        from safetensors.torch import load_file
        torch.set_num_threads(4)
        cpu=WorldBranch(architecture['num_attention_heads']*architecture['attention_head_dim'],args.blocks,args.width,plan['height'],plan['padding_columns'],args.mode=='spatial')
        state=load_file(str(args.init_checkpoint/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite initial weights')
        load_world(cpu,state,initial_config)
        if args.local_only_calibration:
            print(dict(optimization_policy=calibration.POLICY,sampling_step_offset=sampling_offset,**calibration.configure(cpu)),flush=True)
        print(dict(mode=args.mode,initial_step=initial_step,selected_train_ids=[r['id'] for r in train],routing=ROUTING_POLICY,base_frozen=True),flush=True)
        print('CPU PREFLIGHT PASSED; no GPU or downloads performed.',flush=True);return
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
    from ..object_world.cache import check_condition,check_posterior
    if not torch.cuda.is_available():
        raise ValueError('Allocate BF16 CUDA GPUs')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    if not torch.cuda.is_bf16_supported():raise ValueError('Selected GPU must support BF16')
    parallel=Parallel();device=parallel.device
    if parallel.main:
        for row in train:
            check_condition(row['condition_cache']);check_posterior(row['posterior_cache'],plan['height'])
    parallel.barrier()
    config=dict(version=VERSION,pack=str(args.pack),pack_plan=plan,condition='full',
        pano_lora_scale=args.pano_lora_scale,blocks=args.blocks,width=args.width,
        lr=args.lr,warmup_steps=args.warmup_steps,accumulation=args.accumulation,seed=args.seed,
        activation_storage=args.activation_storage,smoke=args.smoke,region_weight=args.region_weight,
        geometry_loss='flow+0.5cube+0.5yaw+region_weight*object_instance_mean+binding_weight*coverage_ce',
        category_encoding='learned_ID_42_slots',group=args.group,binding_weight=binding_weight,
        binding_block=40,binding_ramp_steps=args.binding_ramp_steps,binding_policy=POLICY,
        file_content_hashes_computed=False,spatial_mode=args.mode,routing_policy=ROUTING_POLICY,
        initial_checkpoint=str(args.init_checkpoint),initial_step=initial_step,train_limit=args.train_limit,train_ids=[r['id'] for r in train])
    if args.local_only_calibration:
        config.update(optimization_policy=calibration.POLICY,sampling_step_offset=sampling_offset)
    start=0
    if args.resume:
        if read(args.resume/'config.json')!=config:
            raise ValueError('Resume config differs; use identical settings and a new output')
        start=read(args.resume/'COMPLETE.json')['step']
        if start>=args.steps:raise ValueError('Total steps must exceed checkpoint step')
    out=args.output.resolve()
    if parallel.main:
        new_output(out);atomic_json(out/'config.json',config)
        atomic_json(out/'run_plan.json',dict(total_steps=args.steps,resume=str(args.resume) if args.resume else None,
            initial_checkpoint=str(args.init_checkpoint),initial_step=initial_step,train_count=len(train),val_count=plan['val_count'],note=(
                'Local-only calibration from trained Spatial checkpoint; fresh optimizer on warm start; old encoder/attention frozen'
                if args.local_only_calibration else 'New spatial/control experiment initialized from legacy A; fresh optimizer on warm start; not full-base finetuning')))
        atomic_json(out/'runtime_environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(device),gpu_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
            group=args.group,seed=args.seed,distributed=parallel.size>1,world_size=parallel.size,
            global_accumulation=args.accumulation,local_accumulation=args.accumulation//parallel.size,
            gradient_reduction='mean after local accumulation; World only',
            random_policy='per_global_microbatch_v1',
            note='Same effective batch and objective; multi-GPU floating-point/RNG trajectory is not bitwise single-GPU continuation'))
    parallel.barrier()
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
    scale_check=apply_scale(transformer,args.pano_lora_scale,'default')
    if parallel.main:atomic_json(out/'pano_scale_check.json',scale_check)
    transformer.to(device);install_circular_padding(transformer,plan['padding_columns'])
    transformer.enable_gradient_checkpointing()
    seed_step(args.seed,0)
    hidden=transformer.config.num_attention_heads*transformer.config.attention_head_dim
    world=WorldBranch(hidden,args.blocks,args.width,plan['height'],plan['padding_columns'],args.mode=='spatial').to(device)
    load_world(world,load_file(str(args.init_checkpoint/'world.safetensors')),initial_config)
    frozen_reference=None
    if args.local_only_calibration:
        policy_check=calibration.configure(world);frozen_reference=calibration.frozen_snapshot(world)
        if parallel.main:atomic_json(out/'calibration_policy.json',dict(**policy_check,policy=calibration.POLICY,sampling_step_offset=sampling_offset))
    world.attach(transformer)
    # Read-only metrics: absolute deltas have different activation scales across
    # layers; also record their ratio to the input target RMS. No output changes.
    relative_recorder=RelativeResidualRecorder(world)
    optimizer=torch.optim.AdamW((p for p in world.parameters() if p.requires_grad),lr=args.lr,betas=(.9,.999),weight_decay=1e-5,eps=adam_eps)
    if parallel.main:atomic_json(out/'optimizer_policy.json',dict(name='AdamW',lr=args.lr,betas=[.9,.999],
        weight_decay=1e-5,eps=adam_eps,clip_norm=1.,microbatch_size=1,accumulation=args.accumulation,
        warmup_steps=args.warmup_steps,schedule='linear warmup then constant',
        noise_sampling='uniform native scheduler index',region_weight=args.region_weight))
    if args.resume:
        state=load_file(str(args.resume/'world.safetensors'))
        world.load_state_dict(state,strict=True)
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt',map_location=device,weights_only=True))
        if args.local_only_calibration:calibration.check_frozen(world,frozen_reference)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device)
    lossargs=SimpleNamespace(**REPO_PANORAMA)
    tensors={row['id']:tensorize(row,plan['max_objects']) for row in train}
    if parallel.main:atomic_json(out/'parameters.json',dict(trainable_world=sum(p.numel() for p in world.parameters() if p.requires_grad),
        frozen_base=sum(p.numel() for p in transformer.parameters()),base_trainable=sum(p.numel() for p in transformer.parameters() if p.requires_grad),
        category_embedding_parameters=world.encoder.category.weight.numel()))
    if not args.resume and parallel.main:
        atomic_tensors(out/'initial_world.safetensors',world.state_dict())
    check_replicas(world)
    sampler=EpochSampler(len(train),args.seed);started=time.monotonic();encoder_gradient=False;category_gradient=False
    binding_gradient_at_zero_gate=False;spatial_gradient=False
    initial={key:v.cpu().clone() for key,v in world.state_dict().items()} if args.smoke else None
    for step in range(start,args.steps):
        relative_recorder.reset()
        sample_step=step+sampling_offset
        seed_step(args.seed,sample_step);lr=learning_rate(args.lr,args.warmup_steps,step)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);total=0.;components_total={};sample_ids=[]
        feature_samples=[];noise_samples=[];binding_samples=[];routing_samples=[]
        effective_binding_weight=binding_weight*min(1.,(step+1)/max(1,args.binding_ramp_steps))
        local_accumulation=args.accumulation//parallel.size
        for micro_index in micro_indices(sample_step,args.accumulation,parallel.rank,parallel.size):
            micro=micro_index-sample_step*args.accumulation
            seed_microbatch(args.seed,micro_index)  # identical per-example RNG in both arms and GPU counts
            row=train[sampler.index(micro_index)];sample_ids.append(row['id'])
            posterior=load_file(row['posterior_cache'],device=str(device))
            z=posterior['mean']+posterior['std']*torch.randn_like(posterior['std'])
            z=((z-mean)/std).to(torch.bfloat16)
            condition=load_file(row['condition_cache'],device=str(device))
            weights=load_file(row['region_cache'],device=str(device))['weights']
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
                    enabled=world.enabled;world.enabled=False
                    world.set_world(world_input);baseline=transformer(**kwargs)[0]
                    world.enabled=enabled
                    world.set_world(world_input);zero=transformer(**kwargs)[0]
                    torch.testing.assert_close(zero,baseline,rtol=0,atol=0)
                    del baseline,zero
                world.set_world(None)
            labels=None
            if binding_weight:
                labels,meta=sample_binding(row,plan['height'],plan['padding_columns'],args.seed*1000003+step*args.accumulation+micro)
                labels=tuple(t.to(device) for t in labels)
                binding_samples.append(meta)
            world.set_supervision(labels)
            storage=ActivationStorage(enabled=args.activation_storage=='cpu')
            with storage.context():
                with training_autocast('cuda'):
                    world.set_world(world_input)
                    feature_samples.append(dict(world.encoder.feature_diagnostics));routing_samples.append(dict(world.routing_meta))
                    noise_samples.append(dict(scheduler_index=idx,sigma=float(sigma),timestep=float(timesteps[idx])))
                    prediction=transformer(**kwargs)[0][:,:packed.shape[1]]
                    prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],vae_scale)
                    mask=torch.ones(1,1,z.shape[3],z.shape[4],device=device,dtype=torch.bool)
                    global_loss,components=panorama_loss(prediction,noise-z,mask,
                        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),
                        lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
                    region=object_region_loss(prediction,noise-z,weights)
                    binding=world.take_binding_loss() if labels is not None else region.new_zeros(())
                    loss=global_loss+args.region_weight*region+effective_binding_weight*binding
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/local_accumulation).backward()
            if args.smoke and binding_weight and step==0 and micro==0:
                if not len(labels[2]) or float(binding.detach())<=0:
                    raise AssertionError('Binding smoke needs positive instance support')
                if not all(p.grad is not None and bool(torch.count_nonzero(p.grad)>0)
                    for p in (world.branches['40'].q.weight,world.branches['40'].k.weight,world.encoder.category.weight)):
                    raise AssertionError('Binding loss has no Q/K/category gradient at zero gate')
                binding_gradient_at_zero_gate=True
            total+=float(loss.detach())/local_accumulation
            components={**components,'object_region':region.detach(),'object_region_weighted':args.region_weight*region.detach(),
                'binding_coverage':binding.detach(),'binding_weighted':effective_binding_weight*binding.detach()}
            for key,value in components.items():components_total[key]=components_total.get(key,0.)+float(value)/local_accumulation
            world.clear_context()
            del prediction,loss,global_loss,region,components,weights,kwargs,packed,condition,posterior,z,noise,noisy,world_input,binding,labels
        if any(p.grad is not None for p in transformer.parameters()):raise AssertionError('Frozen base received gradients')
        if args.local_only_calibration and any(p.grad is not None for name,p in world.named_parameters() if '.local.' not in name):
            raise AssertionError('Frozen encoder/legacy attention received gradients')
        if parallel.any(any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters())):
            raise FloatingPointError('World gradients')
        average_gradients(world)
        encoder_gradient|=any(p.grad is not None and bool(torch.count_nonzero(p.grad)>0) for p in world.encoder.parameters())
        category_gradient|=any(p.grad is not None and bool(torch.count_nonzero(p.grad)>0) for p in world.encoder.category.parameters())
        spatial_gradient|=any(p.grad is not None and bool(torch.count_nonzero(p.grad)>0) for name,p in world.named_parameters() if '.local.' in name and not name.endswith('.gate'))
        if args.local_only_calibration and step==start and not spatial_gradient:
            raise AssertionError('First local calibration step has no local-body gradient')
        branch_gradient_norms=gradient_norms(world)
        norm=torch.nn.utils.clip_grad_norm_(world.parameters(),1.)
        optimizer.step()
        if step==start:check_replicas(world)
        rank_metrics=parallel.gather(dict(sample_ids=sample_ids,feature_scale_microbatches=feature_samples,
            noise_samples=noise_samples,binding_microbatches=binding_samples,world=world.diagnostics(),
            world_relative=relative_recorder.summary(),peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30))
        total=parallel.mean(total)
        components_total={k:parallel.mean(v) for k,v in components_total.items()}
        sample_ids=[sid for r in rank_metrics for sid in r['sample_ids']]
        feature_samples=[v for r in rank_metrics for v in r['feature_scale_microbatches']]
        noise_samples=[v for r in rank_metrics for v in r['noise_samples']]
        binding_samples=[v for r in rank_metrics for v in r['binding_microbatches']]
        log=dict(step=step+1,initial_step=initial_step,spatial_mode=args.mode,spatial_non_gate_gradient_seen=spatial_gradient,spatial=world.spatial_diagnostics(),routing_microbatches=routing_samples,loss=total,components=components_total,lr=lr,grad_norm=float(norm),sample_ids=sample_ids,
            passes_seen=(step+1)*args.accumulation/len(train),encoder_nonzero_gradient_seen=encoder_gradient,
            category_embedding_nonzero_gradient_seen=category_gradient,
            feature_scales=world.encoder.feature_diagnostics,
            feature_scale_microbatches=feature_samples,noise_samples=noise_samples,
            binding_weight=effective_binding_weight,binding_microbatches=binding_samples,
            gradient_norms_before_clip=branch_gradient_norms,clip_factor=min(1.,1./(float(norm)+1e-6)),
            weighted_region_to_global=components_total['object_region_weighted']/max(
                total-components_total['object_region_weighted']-components_total['binding_weighted'],1e-12),
            gates_after_update={key:float(torch.tanh(branch.gate)) for key,branch in world.branches.items()},
            world=world.diagnostics(),world_relative=relative_recorder.summary(),
            elapsed_seconds=time.monotonic()-started,peak_gpu_gib=max(r['peak_gpu_gib'] for r in rank_metrics),
            distributed_world_size=parallel.size,global_accumulation=args.accumulation,
            **({'rank_metrics':rank_metrics} if parallel.size>1 else {}))
        if args.local_only_calibration:
            log.update(optimization_policy=calibration.POLICY['version'],sampling_step=sample_step,
                frozen_encoder_and_legacy_no_gradient=True)
        if parallel.main:
            with (out/'train_log.jsonl').open('a') as handle:handle.write(json.dumps(log)+'\n')
            print(log,flush=True)
        stop=parallel.any(args.stop_file is not None and args.stop_file.exists())
        if ((step+1)%args.save_every==0 or step+1==args.steps or stop) and parallel.main:
            ckpt=out/f'checkpoint-step{step+1:06d}';ckpt.mkdir()
            if args.local_only_calibration:atomic_json(ckpt/'FROZEN_CHECK.json',calibration.check_frozen(world,frozen_reference))
            state={key:value.detach().cpu().contiguous() for key,value in world.state_dict().items()}
            atomic_tensors(ckpt/'world.safetensors',state)
            torch.save(optimizer.state_dict(),ckpt/'optimizer.pt')
            atomic_json(ckpt/'config.json',config)
            restored=load_file(str(ckpt/'world.safetensors'))
            if not all(torch.equal(value,restored[key]) for key,value in state.items()):raise AssertionError('Checkpoint roundtrip')
            if args.smoke:
                if (args.mode=='spatial' and not spatial_gradient) or not encoder_gradient or not category_gradient or not any(not torch.equal(value,initial[key]) for key,value in state.items()):
                    raise AssertionError('Smoke did not update ID World encoder')
                atomic_json(ckpt/'SMOKE_COMPLETE.json',dict(zero_added_gate_matches_warm_start_exact=True,base_frozen=True,spatial_non_gate_gradient_seen=spatial_gradient,
                    category_encoding='learned_ID',encoder_gradient=True,category_embedding_gradient=True,parameters_changed=True,
                    binding_loss_enabled=binding_weight>0,
                    binding_gradient_at_zero_gate=binding_gradient_at_zero_gate,
                    object_region_loss_enabled=args.region_weight>0,note='GPU generation still needed; not a quality result'))
            atomic_json(ckpt/'COMPLETE.json',dict(step=step+1,version=VERSION))
        if (step+1)%args.save_every==0 or step+1==args.steps or stop:parallel.barrier()
        if stop and step+1<args.steps:
            if parallel.main:atomic_json(out/'STOPPED.json',dict(step=step+1,reason='stop_file',checkpoint=str(ckpt)))
            parallel.close();return
    if parallel.main:atomic_json(out/'COMPLETE.json',dict(steps=args.steps,checkpoint=str(ckpt),group=args.group,category_encoding='learned_ID',note='Training complete; held-out quality not established'))
    parallel.barrier();parallel.close()


if __name__=='__main__':main()
