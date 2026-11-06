"""One allocated GPU per arm, same A10000 World/Adam, no target caption/routing."""
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
from ..oracle_memory import ActivationStorage,training_autocast
from ..scaled_world.runtime import EpochSampler
from ..id_binding_world.parallel import seed_microbatch
from ..id_binding_world.train import gradient_norms
from ..grounded_world.evaluate import transformer
from ..grounded_world.continue_train import predict
from . import data as d
from .runtime import WorldBranch
from .supervision import sample_binding


def resume_step(checkpoint,config):
    if checkpoint is None:return config['source_step']
    marker=read(checkpoint/'COMPLETE.json')
    if (marker['version']!=d.VERSION or marker['step']!=marker['optimizer_step']
            or type(marker['step']) is not int or marker['step']<config.get('source_step',0)
            or checkpoint.name!=f"checkpoint-step{marker['step']:06d}"
            or read(checkpoint/'config.json')!=config
            or any(not (checkpoint/n).is_file() for n in ('world.safetensors','optimizer.pt'))):
        raise ValueError('Only complete same-arm same-contract checkpoints can resume')
    return marker['step']


def objective(pred,velocity,weights,noise,z,lossargs,aux,coverage_weight,concentration_weight):
    from qwen_edit_pano.losses import panorama_loss
    device=pred.device
    base,parts=panorama_loss(pred,velocity,torch.ones(1,1,*z.shape[-2:],device=device,dtype=torch.bool),
        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),
        lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
    region=object_region_loss(pred,velocity,weights)
    loss=base+.25*region+coverage_weight*aux[0]+concentration_weight*aux[1]
    return loss,{**parts,'object_region':region,'object_region_weighted':.25*region,
        'binding_coverage':aux[0],'binding_concentration':aux[1],
        'binding_weighted':coverage_weight*aux[0]+concentration_weight*aux[1]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=('control','binding'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--additional-steps',type=int,default=1000);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-file',type=Path);a=p.parse_args();torch.set_num_threads(4)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Each arm uses one independent GPU, not DDP')
    if not 1<=a.additional_steps<=1000:raise ValueError('Finite pilot budget: 1..1000 additional steps')
    plan,rows,old=d.inputs();config=d.configuration(plan,old,a.group,a.smoke)
    start=resume_step(a.resume,config);target=10002 if a.smoke else 10000+a.additional_steps
    if start>=target:raise ValueError('Already reached target')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('User must allocate a BF16 CUDA GPU')
    torch.cuda.set_device(0);device=torch.device('cuda:0');out=new_output(a.output)
    atomic_json(out/'config.json',config);checkpoint=a.resume or d.SOURCE
    atomic_json(out/'run_plan.json',dict(start=start,target=target,source=str(checkpoint),
        warm_start_A10000=a.resume is None,smoke=a.smoke,global_batch=4,replicas=1,
        automatic_evaluation=False,automatic_extension=False))
    model=transformer(plan,config,device);model.enable_gradient_checkpointing()
    if any(isinstance(m,torch.nn.Dropout) and m.training and m.p>0 for m in model.modules()):raise AssertionError('Frozen dropout active')
    hidden=model.config.num_attention_heads*model.config.attention_head_dim
    world=WorldBranch(hidden,tuple(config['blocks']),config['width']).to(device)
    opt=torch.optim.AdamW(world.parameters(),lr=config['lr'],betas=tuple(config['betas']),
        eps=config['adam_eps'],weight_decay=config['weight_decay'])
    if d.restore(world,opt,checkpoint)!=start:raise ValueError('Unexpected restore step')
    if a.resume is None:atomic_tensors(out/'initial_world.safetensors',world.state_dict())
    world.attach(model)
    if any(p.requires_grad for p in model.parameters()):raise AssertionError('Base not frozen')
    atomic_json(out/'runtime.json',dict(gpu=torch.cuda.get_device_name(device),source_optimizer_step=start,
        world_parameters=sum(p.numel() for p in world.parameters()),base_trainable=0,replicas=1,
        global_batch=4,caption=False,spatial_routing=False,inherited_smoke=False))
    from diffusers import AutoencoderKLQwenImage,FlowMatchEulerDiscreteScheduler,QwenImageEditPlusPipeline as Pipeline
    from qwen_pano.profiles import REPO_PANORAMA
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean'],device=device).reshape(1,-1,1,1,1)
    std=torch.tensor(vae['latents_std'],device=device).reshape(1,-1,1,1,1)
    if 2**len(vae['temperal_downsample'])!=8:raise ValueError('Unexpected VAE scale')
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device);lossargs=SimpleNamespace(**REPO_PANORAMA)
    sampler=EpochSampler(len(rows),config['seed']);representatives=d.smoke_rows(rows)
    started=time.monotonic();encoder_seen=False;category_seen=False;aux_seen=False;negative_seen=False;equal_forward=False
    for step in range(start,target):
        tick=time.monotonic();opt.zero_grad(set_to_none=True);parts_total={};ids=[];noise_log=[];binding_log=[];total=0.
        factor=1. if a.smoke else min(1.,(step+1-10000)/config['binding_ramp_steps'])
        wc=config['coverage_weight']*factor;wf=config['concentration_weight']*factor
        for group in opt.param_groups:group['lr']=config['lr']
        for micro in range(4):
            index=step*4+micro;seed_microbatch(config['seed'],index)
            row=representatives[(micro+step-start)%len(representatives)] if a.smoke else rows[sampler.index(index)]
            ids.append(row['id']);posterior=load_file(row['posterior_cache'],device=str(device))
            z=((posterior['mean']+posterior['std']*torch.randn_like(posterior['std'])-mean)/std).to(torch.bfloat16)
            noise=torch.randn_like(z);idx=int(torch.randint(scheduler.config.num_train_timesteps,(1,)))
            sigma=sigmas[idx].to(z.dtype);noisy=(1-sigma)*z+sigma*noise;velocity=noise-z
            condition=load_file(row['condition_cache'],device=str(device))
            packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=timesteps[idx:idx+1]/1000,
                encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            state=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
            weights=load_file(row['region_cache'],device=str(device))['weights'];labels=None
            if a.group=='binding':
                labels,meta=sample_binding(row,plan['height'],plan['padding_columns'],config['seed']*1000003+index)
                labels=tuple(t.to(device) for t in labels);binding_log.append(meta)
            if a.smoke and step==start and micro==0:
                with torch.no_grad(),training_autocast('cuda'):
                    without=predict(model,world,state,kwargs,plan['height'])
                    with_labels=predict(model,world,state,kwargs,plan['height'],labels)
                    torch.testing.assert_close(without,with_labels,rtol=0,atol=0);equal_forward=True
                    if labels is not None:world.take_binding_loss()
                    del without,with_labels
                world.clear_context()
            with ActivationStorage(enabled=True).context():
                with training_autocast('cuda'):
                    prediction=predict(model,world,state,kwargs,plan['height'],labels)
                    aux=world.take_binding_loss() if labels is not None else prediction.new_zeros(2,dtype=torch.float32)
                    loss,parts=objective(prediction,velocity,weights,noise,z,lossargs,aux,wc,wf)
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/4).backward()
            aux_seen|=float(aux[0].detach())>0;negative_seen|=float(aux[1].detach())>0
            total+=float(loss.detach())/4
            for key,value in parts.items():parts_total[key]=parts_total.get(key,0.)+float(value.detach())/4
            noise_log.append(dict(index=index,timestep_index=idx,sigma=float(sigma)))
            world.clear_context()
            del posterior,z,noise,noisy,velocity,condition,packed,kwargs,state,weights,labels,prediction,aux,loss,parts
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base received gradients')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters()):raise FloatingPointError('World gradients')
        encoder_seen|=any(p.grad is not None and bool(p.grad.count_nonzero()) for p in world.encoder.parameters())
        category_seen|=world.encoder.category.weight.grad is not None and bool(world.encoder.category.weight.grad.count_nonzero())
        groups=gradient_norms(world);norm=torch.nn.utils.clip_grad_norm_(world.parameters(),config['clip_norm']);opt.step()
        if d.optimizer_step(opt)!=step+1:raise AssertionError('Adam step did not advance exactly once')
        log=dict(step=step+1,additional_step=step+1-10000,group=a.group,loss=total,components=parts_total,
            coverage_weight=wc,concentration_weight=wf,lr=config['lr'],sample_ids=ids,noise_samples=noise_log,
            binding_samples=binding_log,grad_norm=float(norm),gradient_norms=groups,
            encoder_gradient_seen=encoder_seen,category_gradient_seen=category_seen,world=world.diagnostics(),
            seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,
            peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
            remaining_seconds_estimate=(time.monotonic()-started)/(step+1-start)*(target-step-1))
        with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        print(log,flush=True)
        stop=a.stop_file is not None and a.stop_file.exists()
        if (step+1)%config['save_every']==0 or step+1==target or stop:
            ckpt=out/f'checkpoint-step{step+1:06d}';ckpt.mkdir()
            atomic_tensors(ckpt/'world.safetensors',world.state_dict())
            temporary=ckpt/'optimizer.tmp.pt';torch.save(opt.state_dict(),temporary);temporary.replace(ckpt/'optimizer.pt')
            atomic_json(ckpt/'config.json',config)
            if a.smoke:
                if not (equal_forward and encoder_seen and category_seen) or a.group=='binding' and not (aux_seen and negative_seen):
                    raise AssertionError('Smoke requires unchanged forward, gradients and both binding terms')
                atomic_json(ckpt/'SMOKE_COMPLETE.json',dict(version=d.VERSION,group=a.group,config=config,
                    source_optimizer_loaded=True,forward_unchanged_with_labels=True,base_frozen=True,
                    encoder_gradient=True,category_gradient=True,both_binding_terms=a.group=='binding',
                    peak_gpu_gib=log['peak_gpu_gib'],note='Numerical/functional check, not a quality claim'))
            atomic_json(ckpt/'COMPLETE.json',dict(version=d.VERSION,step=step+1,optimizer_step=d.optimizer_step(opt)))
        if stop:
            atomic_json(out/'STOPPED.json',dict(step=step+1,checkpoint=str(ckpt)));return
    atomic_json(out/'COMPLETE.json',dict(version=d.VERSION,step=target,checkpoint=str(ckpt),group=a.group,
        note='Finite pilot finished; no automatic generation or longer training'))


if __name__=='__main__':main()
