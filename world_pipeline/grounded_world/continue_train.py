"""Matched C/D continuation from actual B200; identical architecture and base."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from ..build_polished_world import read,atomic_json
from ..common import new_output
from ..object_world.data import tensorize,atomic_tensors
from ..object_world.runtime import object_region_loss
from ..oracle_memory import training_autocast,ActivationStorage
from ..scaled_world.runtime import EpochSampler
from ..id_binding_world.parallel import seed_microbatch
from .data import inputs,load_support,DEFAULT_SUPPORT,smoke_rows
from .runtime import WorldBranch,VERSION
from .supervision import sample
from .train import norms,draw_latents
from .evaluate import transformer
from . import category_contrast as cc


def configuration(plan,rows,prepared,group,smoke=False):
    base=cc.source_contract(plan,rows)
    return dict(**{k:v for k,v in base.items() if k not in ('version','group','smoke','old_world_loaded','initialization')},version=cc.EXPERIMENT,
        architecture_version=VERSION,group=group,smoke=smoke,base_config=base,initial_checkpoint=str(cc.INITIAL),
        continuation_start=200,continuation_target=202 if smoke else 600,
        category_contrast_weight=cc.POLICY['weight'] if group=='D' else 0.,category_policy=cc.POLICY,
        target_sources=prepared['sources'],category_targets=prepared['targets'],
        old_world_loaded=True,initialization='same real B200 World and Adam state in both C/D; no reinitialization',
        pair_replay='negative no-grad scalar; correct backward; negative recompute/backward with exact detached chain coefficients; one graph at a time',
        frozen_mode='eval; all positive-probability Dropout modules must be inactive',
        smoke_policy='B200 optimizer/weights; two steps alternate supported-TV and max-object sample; sigma 0.8/0.5; full auxiliary weight' if smoke else None)


def predict(model,world,state,kwargs,height,labels=None):
    from diffusers import QwenImageEditPlusPipeline as Pipeline
    world.set_supervision(labels);world.set_world(state)
    # Training circular hooks crop target padding before returning the model
    # output; the reference suffix must never enter the unpacked prediction.
    count=(height//16)*(height//8)
    prediction=model(**kwargs)[0][:,:count]
    return Pipeline._unpack_latents(prediction,height,2*height,8)


def error_forward(model,world,state,kwargs,target,weights,height,gradient=False):
    device=target.device.type
    if gradient:
        with training_autocast(device):return cc.regional_error(predict(model,world,state,kwargs,height),target,weights)
    with torch.no_grad(),training_autocast(device):return cc.regional_error(predict(model,world,state,kwargs,height),target,weights).detach()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=('C','D'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-file',type=Path);args=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('One independent single-GPU arm per job')
    torch.set_num_threads(4);prepared=cc.prepare();plan,_,rows,features=inputs();byid={r['id']:r for r in rows}
    config=configuration(plan,rows,prepared,args.group,args.smoke);start=200;target_step=config['continuation_target']
    checkpoint=args.resume or cc.INITIAL
    if args.resume:
        if read(checkpoint/'config.json')!=config or read(checkpoint/'COMPLETE.json')['version']!=cc.EXPERIMENT:
            raise ValueError('Continuation resume data/group/configuration changed')
        start=read(checkpoint/'COMPLETE.json')['step']
    if start>=target_step:raise ValueError('Target already complete')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('User must allocate one BF16 CUDA GPU')
    out=new_output(args.output);atomic_json(out/'config.json',config)
    atomic_json(out/'run_plan.json',dict(resume=str(checkpoint),start=start,target=target_step,additional_steps=target_step-start,
        main_train_count=128,initial_source_step=200,automatic_evaluation=False,automatic_extension=False))
    device=torch.device('cuda:0');model=transformer(plan,config,device);model.enable_gradient_checkpointing()
    if any(isinstance(m,torch.nn.Dropout) and m.training and m.p>0 for m in model.modules()):raise AssertionError('Frozen base dropout must be inactive')
    hidden=model.config.num_attention_heads*model.config.attention_head_dim
    world=WorldBranch(hidden,features,height=plan['height'],padding=plan['padding_columns']).to(device)
    world.load_state_dict(load_file(str(checkpoint/'world.safetensors')),strict=True)
    opt=torch.optim.AdamW(world.parameters(),lr=config['lr'],betas=tuple(config['betas']),weight_decay=config['weight_decay'],eps=config['adam_eps'])
    opt.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location=device,weights_only=True))
    if any(p.dtype!=torch.float32 for p in world.parameters()):raise AssertionError('FP32 World masters required')
    for entry in opt.state.values():
        for key in ('exp_avg','exp_avg_sq'):
            if key in entry and entry[key].dtype!=torch.float32:raise AssertionError('FP32 optimizer moments required')
    if not args.resume:atomic_tensors(out/'initial_world.safetensors',world.state_dict())
    world.attach(model)
    if any(p.requires_grad for p in model.parameters()):raise AssertionError('Frozen base got trainable parameters')
    atomic_json(out/'runtime.json',dict(base_trainable=0,world_trainable=sum(p.numel() for p in world.parameters()),
        gpu=torch.cuda.get_device_name(device),gpu_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
        source_optimizer_loaded=True,source_optimizer_max_step=max(float(v['step']) for v in opt.state.values()),
        initial_is_B200=not bool(args.resume),pano_lora_scale=.5,dropout_active=False))
    from diffusers import AutoencoderKLQwenImage,FlowMatchEulerDiscreteScheduler,QwenImageEditPlusPipeline as Pipeline
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device);lossargs=SimpleNamespace(**REPO_PANORAMA)
    sampler=EpochSampler(len(rows),config['seed']);tensors={r['id']:tensorize(r,plan['max_objects']) for r in rows}
    supports={r['id']:load_support(DEFAULT_SUPPORT,r['id']) for r in rows}
    smoke_tv=next(r for r in rows if any(t['category_id']==22 for t in prepared['targets'][r['id']]))
    smoke_samples=[smoke_tv,max(rows,key=lambda r:len(r['objects']))]
    started=time.monotonic();times=[];pair_verified=False;sampled_pairs=0
    for step in range(start,target_step):
        tick=time.monotonic();opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=config['lr']
        alpha=cc.weight(step,args.smoke) if args.group=='D' else 0.;ids=[];noise_log=[];pairs=[];parts_total={};total=0.;used=False;correct_world={}
        for micro in range(4):
            index=step*4+micro;seed_microbatch(config['seed'],index)
            row=smoke_samples[(micro+step-200)%2] if args.smoke else rows[sampler.index(index)];ids.append(row['id'])
            posterior=load_file(row['posterior_cache'],device='cpu');condition=load_file(row['condition_cache'],device=str(device))
            z,noise,idx=draw_latents(posterior,mean,std,config['seed'],index,scheduler.config.num_train_timesteps)
            if args.smoke:idx=int((scheduler.sigmas[:scheduler.config.num_train_timesteps]-(.8 if micro==0 else .5)).abs().argmin())
            z=z.to(device);noise=noise.to(device);velocity=noise-z;sigma=sigmas[idx].to(z.dtype)
            noisy=(1-sigma)*z+sigma*noise;packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=timesteps[idx:idx+1]/1000,
                encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            state=tuple(t.to(device) for t in tensors[row['id']]);weights=load_file(row['region_cache'],device=str(device))['weights']
            labels,_=sample(supports[row['id']],config['seed']*1000003+index);labels=tuple(t.to(device) for t in labels)
            candidate=None
            if not used and .5<=float(sigma)<=.9:
                candidate=cc.select_target(prepared['targets'][row['id']],plan['names'],config['seed']*1000003+index)
            w=negative=negative_value=None
            if candidate is not None:
                used=True;sampled_pairs+=1;w=cc.instance_weights(row,candidate,plan['height']).to(device);negative=cc.negative_state(state,candidate)
                if alpha:
                    negative_value=error_forward(model,world,negative,kwargs,velocity,w,plan['height']);world.clear_context()
            with ActivationStorage(enabled=True).context():
                with training_autocast('cuda'):
                    prediction=predict(model,world,state,kwargs,plan['height'],labels);correct_world=dict(world.diagnostics())
                    base,parts=panorama_loss(prediction,velocity,torch.ones(1,1,*z.shape[-2:],device=device,dtype=torch.bool),
                        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
                    region=object_region_loss(prediction,velocity,weights);binding=world.take_binding_loss()
                    original=base+.25*region+.01*binding;backward=original/4
                    positive_error=cc.regional_error(prediction,velocity,w) if candidate is not None else None
                    if alpha and candidate is not None:
                        contrast,cp,cn=cc.coefficients(positive_error.detach(),negative_value)
                        backward=backward+alpha*cp*positive_error
                if not torch.isfinite(backward):raise FloatingPointError(row['id'])
                backward.backward()
            total+=float(original.detach())/4
            for key,value in {**parts,'object_region':region,'binding':binding}.items():parts_total[key]=parts_total.get(key,0.)+float(value.detach())/4
            record=None
            if candidate is not None:
                record=dict(**candidate,id=row['id'],micro=micro,index=index,sigma=float(sigma),positive_error=float(positive_error.detach()),enabled=bool(alpha))
                if alpha:record.update(negative_error=float(negative_value),contrast=float(contrast),weight=alpha,positive_coefficient=float(cp),negative_coefficient=float(cn))
                pairs.append(record)
            noise_log.append(dict(index=idx,sigma=float(sigma)));world.clear_context()
            del prediction,base,parts,region,binding,original,backward,positive_error
            if alpha and candidate is not None:
                before={n:p.grad.detach().clone() for n,p in world.named_parameters() if args.smoke and p.grad is not None and ('.v.weight' in n or '.out.weight' in n)}
                with ActivationStorage(enabled=True).context():
                    error=error_forward(model,world,negative,kwargs,velocity,w,plan['height'],gradient=True)
                    torch.testing.assert_close(error.detach(),negative_value,rtol=1e-5,atol=1e-7)
                    (alpha*cn*error).backward()
                if args.smoke and not any(bool((p.grad-before[n]).count_nonzero()) for n,p in world.named_parameters() if n in before):
                    raise AssertionError('Negative branch did not reach actual V/output gradients')
                pair_verified=True;total+=alpha*float(contrast);world.clear_context();del error,contrast,cp,cn,before
            del posterior,condition,z,noise,velocity,sigma,noisy,packed,kwargs,state,weights,labels,w,negative,negative_value
        if args.smoke and args.group=='D' and not used:raise AssertionError('Smoke failed to exercise category contrast')
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base got gradients')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters()):raise FloatingPointError('Nonfinite World gradients')
        gn=norms(world);norm=torch.nn.utils.clip_grad_norm_(world.parameters(),config['clip_norm']);opt.step()
        times.append(time.monotonic()-tick);seconds=sum(times[-10:])/len(times[-10:])
        log=dict(step=step+1,group=args.group,loss=total,components=parts_total,contrast_pairs=pairs,contrast_weight=alpha,
            sample_ids=ids,noise_samples=noise_log,gradient_norms=gn,grad_norm=float(norm),world=correct_world,
            sample_exposures=(step+1)*4,passes=(step+1)*4/128,elapsed_seconds=time.monotonic()-started,
            recent_seconds_per_step=seconds,eta_seconds=(target_step-step-1)*seconds,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
        with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(log,allow_nan=False)+'\n')
        print(log,flush=True);stop=args.stop_file is not None and args.stop_file.exists()
        if (step+1)%50==0 or step+1==target_step or stop:
            path=out/f'checkpoint-step{step+1:06d}';path.mkdir();atomic_tensors(path/'world.safetensors',world.state_dict())
            torch.save(opt.state_dict(),path/'optimizer.pt');atomic_json(path/'config.json',config)
            if args.smoke:
                if args.group=='D' and not pair_verified:raise AssertionError('Missing negative recompute/gradient verification')
                atomic_json(path/'SMOKE_COMPLETE.json',dict(group=args.group,source_optimizer_loaded=True,base_trainable=0,
                    negative_replay_verified=pair_verified,actual_value_output_gradient_verified=pair_verified,
                    maximum_objects=max(len(r['objects']) for r in smoke_samples),peak_gpu_gib=log['peak_gpu_gib']))
            atomic_json(path/'COMPLETE.json',dict(version=cc.EXPERIMENT,step=step+1))
            if stop and step+1<target_step:atomic_json(out/'STOPPED.json',dict(checkpoint=str(path)));return
    atomic_json(out/'COMPLETE.json',dict(version=cc.EXPERIMENT,step=target_step,checkpoint=str(path),group=args.group,
        sampled_pairs=sampled_pairs,automatic_evaluation=False,note='No claim of generated-object success. No automatic extension.'))


if __name__=='__main__':main()
