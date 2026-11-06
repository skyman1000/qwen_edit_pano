"""User-run Legacy teacher-forcing diagnosis; no training update or image generation.

Uses only existing train caches. Unseen samples are unexposed TRAIN views,
not held-out validation buildings. This is a mechanism test, not validation.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

from ..build_polished_world import ROOT,read,atomic_json
from ..common import new_output
from ..scaled_world.runtime import EpochSampler,seed_step
from .data import load_pack,tensorize
from .runtime import VERSION,WorldBranch,object_region_loss
from .diagnostic_features import MODES,set_condition,AttentionProbe,compare_gradients

DEFAULT_RUN=ROOT/'qwen_edit_pano/outputs/object_world_layer_pilot_legacy_v1'


def sample_cases(records,run,count):
    train=[r for r in records if r['split']=='train']
    config=read(run/'config.json');complete=read(run/'COMPLETE.json')
    exposed=complete['steps']*config['accumulation']
    if not 2<=count<=min(exposed,len(train)-exposed)*2:
        raise ValueError('Need cached seen/unexposed train cases; this diagnostic expects a partial first pass')
    sampler=EpochSampler(len(train),config['seed']);result=[]
    for i in range(count):
        position=i//2 if i%2==0 else exposed+i//2
        row=train[sampler.index(position)]
        result.append(dict(row=row,seen_by_step500=i%2==0,stream_position=position))
    return result


def query_plan(row,height,padding):
    import numpy as np
    th,tw=height//16,height//8
    queries=list(dict.fromkeys(int(y)*(tw+2*padding)+int(x)+padding
        for y in np.linspace(0,th-1,8).astype(int) for x in np.linspace(0,tw-1,8).astype(int)))
    pairs=[]
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as arrays:
        instance=arrays['instance'];valid=arrays['object_loss_valid']
        if instance.shape[0]%th or instance.shape[1]%tw:raise ValueError('Instance/token grid ratio not integer')
        fy,fx=instance.shape[0]//th,instance.shape[1]//tw
        for slot,obj in enumerate(row['objects'],1):
            support=(instance==int(obj['instance_id']))&valid
            if support.sum()<16:continue
            masses=support.reshape(th,fy,tw,fx).sum((1,3));y,x=np.unravel_index(masses.argmax(),masses.shape)
            index=int(y)*(tw+2*padding)+int(x)+padding
            if index not in queries:queries.append(index)
            pairs.append((queries.index(index),slot))
    return queries,pairs


def forward_prediction(model,world,kwargs,tensors,mode,class_ids,height):
    from diffusers import QwenImageEditPlusPipeline as Pipeline
    set_condition(world,tensors,mode,class_ids)
    packed_count=(height//16)*(height//8)
    out=model(**kwargs)[0][:,:packed_count]
    return Pipeline._unpack_latents(out,height,2*height,8)


def loss_values(prediction,target,weights,region_weight=.25):
    import torch
    flow=(prediction.float()-target.float()).square().mean()
    region=object_region_loss(prediction,target,weights)
    return dict(flow=float(flow),object_region=float(region),flow_plus_region=float(flow+region_weight*region))


def backward_check(model,world,kwargs,tensors,target,weights,config,case_seed,noise,clean,
                   mode='checkpointed',memory_callback=None):
    import torch
    from types import SimpleNamespace
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    from ..oracle_memory import training_autocast,ActivationStorage
    from .train import gradient_norms
    if mode not in ('checkpointed','parity'):raise ValueError('Unknown backward diagnostic mode')
    gradients=[];losses=[];groups=[]
    for enabled in ((True,) if mode=='checkpointed' else (False,True)):
        world.zero_grad(set_to_none=True)
        # enable_gradient_checkpointing() may replace the checkpoint function
        # installed by World.attach. Toggle the native flag only, keep wrapper.
        model.gradient_checkpointing=enabled
        seed_step(case_seed,0)
        if memory_callback:memory_callback('backward_start',checkpointed=enabled)
        with ActivationStorage(enabled=True).context():
            with training_autocast(kwargs['hidden_states'].device.type):
                pred=forward_prediction(model,world,kwargs,tensors,'correct',config['pack_plan']['names'],config['pack_plan']['height'])
                # Same full objective as train.py, including Python-RNG yaw.
                args=SimpleNamespace(**REPO_PANORAMA)
                loss,_=panorama_loss(pred,target,torch.ones(1,1,*target.shape[-2:],device=target.device,dtype=torch.bool),
                    torch.ones(1,device=target.device,dtype=torch.bool),torch.ones(1,device=target.device),
                    args.geometry_start_epoch,args,noise=noise,clean=clean)
                loss=loss+config['region_weight']*object_region_loss(pred,target,weights)
            world.set_world(None)  # Explicit state must survive checkpoint recompute.
            loss.backward()
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base received gradients')
        grads={k:p.grad.detach().cpu().clone() for k,p in world.named_parameters() if p.grad is not None}
        if not grads or any(not torch.isfinite(g).all() for g in grads.values()):raise ValueError('Invalid World gradients')
        gradients.append(grads);losses.append(float(loss));groups.append(gradient_norms(world))
        del pred,loss
        world.set_world(None);gc.collect()
        if target.device.type=='cuda':torch.cuda.empty_cache()
        if memory_callback:memory_callback('backward_done',checkpointed=enabled)
    world.zero_grad(set_to_none=True);model.gradient_checkpointing=False
    result=dict(mode=mode,base_frozen=True,optimizer_steps=0,gradient_groups=groups,
                gradient_parameter_count=len(gradients[-1]),finite_world_gradients=True)
    if mode=='parity':
        result.update(compare_gradients(*gradients),loss_without_checkpoint=losses[0],loss_with_checkpoint=losses[1])
        result['checkpoint_parity_pass']=(result['relative_l2_difference']<1e-3 and abs(losses[0]-losses[1])<1e-5)
    else:
        result.update(loss_with_checkpoint=losses[0],checkpoint_parity_tested=False,
            encoder_gradient_nonzero=any(float(g.abs().sum())>0 for k,g in gradients[0].items() if k.startswith('encoder.')),
            semantic_gradient_nonzero=any(float(g.abs().sum())>0 for k,g in gradients[0].items() if k.startswith('encoder.semantic.')))
        if not result['encoder_gradient_nonzero'] or not result['semantic_gradient_nonzero']:
            raise AssertionError('Trained Legacy checkpoint did not propagate encoder/semantic gradients')
    return result


def record_key(record):return record['step'],record['id'],record['scheduler_index'],record['mode']


def resume_records(output,budget):
    """Validate saved numeric records/protocol; never hash or overwrite them."""
    previous=read(output/'PLAN.json')
    for key in ('samples','sigmas','checkpoint_steps','modes','cases','scope'):
        if previous[key]!=budget[key]:raise ValueError('Resume diagnostic protocol differs: '+key)
    if (output/'COMPLETE.json').exists():raise ValueError('Diagnostic is already complete')
    cases={c['id']:c for c in budget['cases']};saved={}
    path=output/'results.jsonl'
    if not path.exists():return saved
    for line in path.read_text().splitlines():
        row=json.loads(line);key=record_key(row)
        if (row['step'] not in budget['checkpoint_steps'] or row['id'] not in cases or row['mode'] not in budget['modes']
                or row['seen_by_step500']!=cases[row['id']]['seen_by_step500']
                or not 0<row['sigma']<1 or key in saved
                or not all(math.isfinite(row[k]) for k in ('flow','object_region','flow_plus_region','relative_prediction_difference_from_correct'))):
            raise ValueError('Invalid/duplicate saved diagnostic record')
        saved[key]=row
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2)
    p.add_argument('--sigmas',type=float,nargs='+',default=[.2,.5,.8])
    p.add_argument('--steps',type=int,nargs='+',default=[250,500],help='Saved Legacy checkpoint steps')
    p.add_argument('--skip-backward',action='store_true')
    p.add_argument('--backward-mode',choices=('checkpointed','parity'),default='checkpointed',
        help='48GB default: checkpointed gradient smoke only; parity disables checkpoint and needs more VRAM')
    p.add_argument('--resume',action='store_true',help='Reuse matching completed records in this existing output')
    p.add_argument('--calibrate-order',action='store_true',
        help='Fixed small calibration: step500, 2 cached cases, sigma .5, repeats/probe/order controls; no backward')
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args();args.run=args.run.resolve()
    if args.calibrate_order:
        args.samples=2;args.steps=[500];args.sigmas=[.5];args.skip_backward=True
    modes=('off','correct','repeat_correct','repeat_probed','order_permute','yaw90','class_cycle') if args.calibrate_order else MODES
    ordered_modes=('correct',*([m for m in modes if m not in ('off','correct')]),'off')
    config=read(args.run/'config.json');plan,records=load_pack(config['pack'])
    if config['version']!=VERSION or config['blocks']!=[20,40,59] or config['pack_plan']!=plan:
        raise ValueError('Use the bound Legacy semantic/region checkpoint and unchanged training pack')
    if len(set(args.steps))!=len(args.steps) or not args.steps or not args.sigmas or len(set(args.sigmas))!=len(args.sigmas) or any(not 0<s<1 for s in args.sigmas):
        p.error('Invalid checkpoints/sigmas')
    if args.samples<2 or args.samples%2:p.error('--samples must be even, alternating seen/unexposed train')
    cases=sample_cases(records,args.run,args.samples)
    for step in args.steps:
        if read(args.run/f'checkpoint-step{step:06d}/COMPLETE.json')['step']!=step:
            raise ValueError('Incomplete checkpoint')
    for case in cases:
        for key in ('condition_cache','posterior_cache','region_cache'):
            if not Path(case['row'][key]).is_file():raise ValueError('Missing existing diagnostic cache')
    budget=dict(samples=args.samples,sigmas=args.sigmas,checkpoint_steps=args.steps,modes=list(modes),
        forward_calls=args.samples*len(args.sigmas)*(1+len(args.steps)*(len(modes)-1)),
        backward_mode='skip' if args.skip_backward else args.backward_mode,
        backward_passes=0 if args.skip_backward else 1 if args.backward_mode=='checkpointed' else 2,
        generated_images=0,training_updates=0,
        cases=[dict(id=c['row']['id'],seen_by_step500=c['seen_by_step500'],stream_position=c['stream_position']) for c in cases],
        scope='repeat/probe/order numerical calibration; NOT quality evaluation' if args.calibrate_order else
              'seen/unexposed train views; NOT held-out validation quality')
    existing=resume_records(args.output,budget) if args.resume else {}
    budget.update(cached_condition_records=len(existing),
                  total_condition_records=args.samples*len(args.sigmas)*len(args.steps)*len(modes))
    budget['remaining_condition_records']=budget['total_condition_records']-len(existing)
    print(budget,flush=True)
    if args.preflight:return
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from diffusers import QwenImageTransformer2DModel,FlowMatchEulerDiscreteScheduler,AutoencoderKLQwenImage
    from diffusers import QwenImageEditPlusPipeline as Pipeline
    from peft import LoraConfig,set_peft_model_state_dict
    from qwen_edit_pano.paired_circular import install_circular_padding
    from ..scaled_world.binding import apply_scale
    from ..oracle_memory import training_autocast
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Allocate one BF16 CUDA GPU')
    output=args.output.resolve() if args.resume else new_output(args.output)
    if not args.resume:atomic_json(output/'PLAN.json',budget)
    else:
        if read(output/'pano_binding.json')['pano_lora_scale']!=config['pano_lora_scale']:
            raise ValueError('Resume frozen Panorama scale differs')
        attempt=1
        while (output/f'ATTEMPT_{attempt:03d}.json').exists():attempt+=1
        atomic_json(output/f'ATTEMPT_{attempt:03d}.json',budget)
    device=torch.device('cuda:0');base=Path(plan['base_checkpoint'])
    pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    model=QwenImageTransformer2DModel.from_pretrained(plan['base_snapshot'],subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    model.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    bound=set_peft_model_state_dict(model,load_file(str(base/'adapter_resume.safetensors')))
    if bound.unexpected_keys or any('lora_' in k for k in bound.missing_keys):raise ValueError('Incomplete frozen Panorama load')
    binding=apply_scale(model,config['pano_lora_scale'],'default')
    if args.resume:
        if read(output/'pano_binding.json')!=binding:raise ValueError('Frozen Panorama binding changed')
    else:atomic_json(output/'pano_binding.json',binding)
    model.to(device);install_circular_padding(model,plan['padding_columns'])
    # Install checkpoint wrapper while the native checkpoint function exists.
    model.enable_gradient_checkpointing()
    state=load_file(str(args.run/f'checkpoint-step{args.steps[0]:06d}/world.safetensors'))
    world=WorldBranch(model.config.num_attention_heads*model.config.attention_head_dim,
        state['encoder.class_features'],config['blocks'],config['width']).to(device).eval()
    world.load_state_dict(state,strict=True);world.attach(model);model.gradient_checkpointing=False
    probe=AttentionProbe(world)
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean'],device=device).reshape(1,-1,1,1,1)
    std=torch.tensor(vae['latents_std'],device=device).reshape(1,-1,1,1,1)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigma_table=scheduler.sigmas[:scheduler.config.num_train_timesteps]
    indices=[int((sigma_table-s).abs().argmin()) for s in args.sigmas]
    if len(set(indices))!=len(indices):raise ValueError('Requested sigmas map to duplicate scheduler indices')
    expected={(step,c['row']['id'],index,mode) for step in args.steps for c in cases for index in indices for mode in modes}
    if not set(existing)<=expected:raise ValueError('Saved records contain an unexpected noise index')
    baselines={};started=time.monotonic();completed=0;backward_done=False
    stage='forward'
    def memory_event(event,**values):
        free,total=torch.cuda.mem_get_info(device)
        record=dict(stage=event,allocated_gib=torch.cuda.memory_allocated(device)/2**30,
            reserved_gib=torch.cuda.memory_reserved(device)/2**30,free_gib=free/2**30,
            total_gib=total/2**30,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,**values)
        with (output/'memory_log.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(record,flush=True)
    memory_event('model_loaded')
    for step in args.steps:
        world.load_state_dict(load_file(str(args.run/f'checkpoint-step{step:06d}/world.safetensors')),strict=True)
        for case_index,case in enumerate(cases):
            row=case['row'];posterior=load_file(row['posterior_cache'],device='cuda:0')
            cond=load_file(row['condition_cache'],device='cuda:0');weights=load_file(row['region_cache'],device='cuda:0')['weights']
            generator=torch.Generator(device='cuda').manual_seed(7821+case_index)
            with torch.no_grad():
                z=((posterior['mean']+posterior['std']*torch.randn(posterior['std'].shape,device=device,generator=generator)-mean)/std).to(torch.bfloat16)
                noise=torch.randn(z.shape,device=device,dtype=z.dtype,generator=generator);target=noise-z
            tensors=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
            queries,pairs=query_plan(row,plan['height'],plan['padding_columns'])
            for requested_sigma in args.sigmas:
                index=int((sigma_table-requested_sigma).abs().argmin());sigma=sigma_table[index].to(device,dtype=z.dtype)
                noisy=(1-sigma)*z+sigma*noise;packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
                kwargs=dict(hidden_states=torch.cat([packed,cond['reference'][None]],1),
                    timestep=scheduler.timesteps[index:index+1].to(device)/1000,
                    encoder_hidden_states=cond['embeddings'][None],encoder_hidden_states_mask=cond['mask'][None],
                    img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(cond['ref_shape'].tolist())]],return_dict=False)
                correct_prediction=None;group=[];key=(case_index,index);new_records=False
                group_keys={(step,row['id'],index,m) for m in modes}
                all_cached=group_keys<=set(existing)
                if all_cached:
                    group=[existing[(step,row['id'],index,m)] for m in ordered_modes]
                    print(dict(stage='reuse_group',step=step,id=row['id'],scheduler_index=index,records=len(group)),flush=True)
                for mode in ordered_modes:
                    if all_cached:break
                    rkey=(step,row['id'],index,mode)
                    if rkey in existing:
                        group.append(existing[rkey]);continue
                    # A partial group may already have the correct record but
                    # still need its tensor to compare remaining conditions.
                    if mode!='correct' and correct_prediction is None:
                        with torch.no_grad(),training_autocast('cuda'):
                            correct_prediction=forward_prediction(model,world,kwargs,tensors,'correct',plan['names'],plan['height']).detach().clone()
                        world.set_world(None);completed+=1
                    probe.reset(queries,pairs)
                    probe.active=(mode=='correct' and not args.calibrate_order) or mode=='repeat_probed'
                    condition='correct' if mode in ('repeat_correct','repeat_probed') else mode
                    with torch.no_grad(),training_autocast('cuda'):
                        if mode=='off' and key in baselines:prediction=baselines[key].to(device)
                        else:
                            prediction=forward_prediction(model,world,kwargs,tensors,condition,plan['names'],plan['height'])
                            if mode=='off':baselines[key]=prediction.detach().cpu()
                            completed+=1
                        if mode=='correct':correct_prediction=prediction.detach().clone()
                        record=dict(step=step,id=row['id'],seen_by_step500=case['seen_by_step500'],
                            sigma=float(sigma),scheduler_index=index,mode=mode,**loss_values(prediction,target,weights,config['region_weight']),
                            relative_prediction_difference_from_correct=float((prediction.float()-correct_prediction.float()).square().mean().sqrt()/correct_prediction.float().square().mean().sqrt().clamp_min(1e-12)),
                            feature_scales=world.encoder.feature_diagnostics if mode!='off' else None,
                            attention=dict(probe.records) if probe.active else None)
                    world.set_world(None);probe.active=False
                    with (output/'results.jsonl').open('a') as handle:
                        handle.write(json.dumps(record)+'\n')
                    group.append(record);existing[rkey]=record;new_records=True
                    print(dict(step=step,id=row['id'],sigma=float(sigma),mode=mode,
                        flow=record['flow'],region=record['object_region'],completed=completed,total=budget['forward_calls'],
                        elapsed_seconds=time.monotonic()-started),flush=True)
                off=next(r for r in group if r['mode']=='off');correct=group[0]
                if new_records:
                    atomic_json(output/f'step{step:06d}_case{case_index}_sigma{index}.json',dict(results=group,
                        off_minus_correct_region=off['object_region']-correct['object_region']))
                if not args.skip_backward and step==args.steps[-1] and not backward_done:
                    stage='backward_'+args.backward_mode
                    # Release evaluation outputs and allocator cache before a
                    # differentiable forward. Do not retain the no-grad graph.
                    prediction=None;correct_prediction=None;world.set_world(None);probe.active=False
                    gc.collect();torch.cuda.empty_cache();memory_event('before_'+stage)
                    try:
                        result=backward_check(model,world,kwargs,tensors,target,weights,config,921,noise,z,
                            mode=args.backward_mode,memory_callback=memory_event)
                    except torch.OutOfMemoryError:
                        memory_event('oom',operation=stage)
                        atomic_json(output/'OOM.json',dict(stage=stage,step=step,id=row['id'],scheduler_index=index,
                            note='Forward records preserved. Resume with --backward-mode checkpointed or --skip-backward.'))
                        raise
                    atomic_json(output/'backward_check.json',result);backward_done=True
                    if result.get('checkpoint_parity_pass') is False:
                        raise AssertionError('Production checkpoint gradient mismatch; inspect backward_check.json')
                    stage='forward'
            del posterior,cond,weights,z,noise,target,tensors,kwargs,packed,correct_prediction
            prediction=None;world.set_world(None);gc.collect();torch.cuda.empty_cache()
    saved=load_file(str(args.run/f'checkpoint-step{args.steps[-1]:06d}/world.safetensors'))
    if any(not torch.equal(v.detach().cpu(),saved[k]) for k,v in world.state_dict().items()):
        raise AssertionError('Diagnostic unexpectedly changed World parameters or buffers')
    if set(existing)!=expected:raise AssertionError('Diagnostic condition records incomplete')
    if not args.resume and completed!=budget['forward_calls']:raise AssertionError('Diagnostic forward budget differs')
    atomic_json(output/'COMPLETE.json',dict(forward_calls_this_attempt=completed,expected_fresh_forward_calls=budget['forward_calls'],
        condition_records=len(existing),reused_condition_records=budget['cached_condition_records'],
        backward_check_completed=backward_done,backward_mode=budget['backward_mode'],elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,training_updates=0,generated_images=0,
        world_parameters_and_buffers_unchanged=True))


if __name__=='__main__':main()
