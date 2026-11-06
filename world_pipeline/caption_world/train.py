"""Fresh one/two-GPU World training, fixed global batch four, no binding."""
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
from ..scaled_world.runtime import seed_step,learning_rate,EpochSampler
from ..id_binding_world.parallel import Parallel,micro_indices,seed_microbatch,average_gradients,check_replicas
from ..grounded_world.runtime import WorldBranch
from ..grounded_world.evaluate import transformer
from ..grounded_world.continue_train import predict
from ..grounded_world.train import draw_latents,norms
from . import data as d


def fresh_world(hidden,features,plan,config,device):
    seed_step(config['seed'],0)
    world=WorldBranch(hidden,features,blocks=tuple(config['blocks']),width=config['width'],
        height=plan['height'],padding=plan['padding_columns'],binding_block=config['binding_block']).to(device)
    opt=torch.optim.AdamW(world.parameters(),lr=config['lr'],betas=tuple(config['betas']),
        eps=config['adam_eps'],weight_decay=config['weight_decay'])
    if opt.state or any(float(b.gate)!=0 for b in world.branches.values()):
        raise AssertionError('Fresh optimizer must be empty and residual gates zero')
    return world,opt


def objective(prediction,velocity,weights,lossargs,noise,clean):
    from qwen_edit_pano.losses import panorama_loss
    device=prediction.device
    base,parts=panorama_loss(prediction,velocity,torch.ones(1,1,*velocity.shape[-2:],device=device,dtype=torch.bool),
        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),
        lossargs.geometry_start_epoch,lossargs,noise=noise,clean=clean)
    region=object_region_loss(prediction,velocity,weights)
    return base+.25*region,{**parts,'object_region':region}


def resume_step(checkpoint,config):
    if checkpoint is None:return 0
    marker=read(checkpoint/'COMPLETE.json')
    if read(checkpoint/'config.json')!=config or marker['version']!=d.VERSION or marker['optimizer_step']!=marker['step']:
        raise ValueError('Resume must use this fresh-caption experiment/configuration')
    if not all((checkpoint/name).is_file() for name in ('world.safetensors','optimizer.pt')):
        raise ValueError('Incomplete checkpoint')
    return marker['step']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=d.OUTPUT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-initial',type=Path);p.add_argument('--stop-file',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    plan,allrows,features,prepared,_=d.inputs(a.root)
    config=d.configuration(a.root,plan,allrows,prepared,a.smoke);start=resume_step(a.resume,config)
    if not 0<=start<a.steps<=config['maximum_steps'] or a.smoke and a.steps!=2:raise ValueError('Invalid target step')
    if d.check(a.root,a.smoke):raise ValueError('Build missing caption caches on allocated GPU(s) first')
    replicas=int(os.environ.get('WORLD_SIZE','1'))
    if not torch.cuda.is_available() or torch.cuda.device_count()<replicas:raise ValueError('Allocate visible CUDA GPUs matching --gpus')
    parallel=Parallel();device=parallel.device
    if parallel.size not in (1,2) or not torch.cuda.is_bf16_supported():raise ValueError('Allocate one/two BF16 CUDA GPUs')
    if parallel.main:
        new_output(a.output);atomic_json(a.output/'config.json',config)
        atomic_json(a.output/'run_plan.json',dict(start=start,target=a.steps,resume=str(a.resume) if a.resume else None,
            replicas=parallel.size,fresh_initialization=a.resume is None,old_C_or_A_loaded=False,
            inherited_smoke_weights=False,global_batch=4,training_images=7390,automatic_evaluation=False))
    parallel.barrier();out=a.output
    model=transformer(plan,config,device);model.enable_gradient_checkpointing()
    if any(isinstance(m,torch.nn.Dropout) and m.training and m.p>0 for m in model.modules()):raise AssertionError('Frozen base dropout active')
    hidden=model.config.num_attention_heads*model.config.attention_head_dim
    world,opt=fresh_world(hidden,features,plan,config,device)
    if a.resume:
        world.load_state_dict(load_file(str(a.resume/'world.safetensors'),device='cpu'),strict=True)
        opt.load_state_dict(torch.load(a.resume/'optimizer.pt',map_location=device,weights_only=True))
        if max(float(v['step']) for v in opt.state.values())!=start:raise ValueError('Optimizer step mismatch')
    elif a.smoke_initial is not None:
        initial=load_file(str(a.smoke_initial),device='cpu')
        if world.state_dict().keys()!=initial.keys() or any(not torch.equal(t.cpu(),initial[k]) for k,t in world.state_dict().items()):
            raise AssertionError('Fresh full initialization differs from tested smoke initialization')
        del initial
    if any(p.dtype!=torch.float32 for p in world.parameters()):raise AssertionError('World masters must be FP32')
    for state in opt.state.values():
        if any(state[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq') if k in state):raise AssertionError('Adam moments must be FP32')
    check_replicas(world)
    if parallel.main:
        if not a.resume:atomic_tensors(out/'initial_world.safetensors',world.state_dict())
        atomic_json(out/'runtime.json',dict(gpu=torch.cuda.get_device_name(device),replicas=parallel.size,base_trainable=0,
            world_trainable=sum(p.numel() for p in world.parameters()),global_batch=4,local_accumulation=4//parallel.size,
            initial_optimizer_step=start,pano_lora_scale=.5,binding_labels_loaded=False,binding_weight=0.,
            caption_condition=True,old_C_or_A_loaded=False,resumed_same_experiment=a.resume is not None))
    world.attach(model)
    if any(p.requires_grad for p in model.parameters()):raise AssertionError('Base must remain frozen')
    from diffusers import AutoencoderKLQwenImage,FlowMatchEulerDiscreteScheduler,QwenImageEditPlusPipeline as Pipeline
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device);lossargs=SimpleNamespace(**config['panorama_loss'])
    rows=d.smoke_rows(allrows) if a.smoke else [r for r in allrows if r['split']=='train']
    sampler=EpochSampler(len(rows),config['seed']);local_accumulation=4//parallel.size
    started=time.monotonic();times=[];semantic_gradient=False;body_gradient=False;zero_equivalent=False
    for step in range(start,a.steps):
        tick=time.monotonic();opt.zero_grad(set_to_none=True);parts_total={};ids=[];noise_log=[];total=0.;diagnostics={}
        lr=config['lr'] if a.smoke else learning_rate(config['lr'],config['warmup_steps'],step)
        for group in opt.param_groups:group['lr']=lr
        for micro,index in enumerate(micro_indices(step,4,parallel.rank,parallel.size)):
            seed_microbatch(config['seed'],index)
            row=rows[index%len(rows)] if a.smoke else rows[sampler.index(index)];ids.append(row['id'])
            posterior=load_file(row['posterior_cache'],device='cpu');condition=d.load_condition(row,device)
            z,noise,idx=draw_latents(posterior,mean,std,config['seed'],index,scheduler.config.num_train_timesteps)
            z=z.to(device);noise=noise.to(device);velocity=noise-z;sigma=sigmas[idx].to(z.dtype)
            noisy=(1-sigma)*z+sigma*noise;packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=timesteps[idx:idx+1]/1000,
                encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            state=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
            weights=load_file(row['region_cache'],device=str(device))['weights']
            if a.smoke and step==0 and micro==0:
                with torch.no_grad(),training_autocast(device.type):
                    baseline=predict(model,world,None,kwargs,plan['height'])
                    zero=predict(model,world,state,kwargs,plan['height'])
                    torch.testing.assert_close(zero,baseline,rtol=0,atol=0);zero_equivalent=True
                    del baseline,zero
                world.clear_context()
            with ActivationStorage(enabled=True).context():
                with training_autocast(device.type):
                    prediction=predict(model,world,state,kwargs,plan['height'])
                    if world.supervision is not None or world._losses:raise AssertionError('Binding supervision must be absent')
                    diagnostics=dict(world.diagnostics());loss,parts=objective(prediction,velocity,weights,lossargs,noise,z)
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/local_accumulation).backward()
            total+=float(loss.detach())/local_accumulation
            for key,value in parts.items():parts_total[key]=parts_total.get(key,0.)+float(value.detach())/local_accumulation
            noise_log.append(dict(index=index,timestep_index=idx,sigma=float(sigma),caption_tokens=condition['embeddings'].shape[0]))
            world.clear_context()
            del prediction,loss,parts,posterior,condition,z,noise,velocity,sigma,noisy,packed,kwargs,state,weights
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base received gradients')
        if parallel.any(any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters())):
            raise FloatingPointError('Nonfinite gradients')
        average_gradients(world);gn=norms(world);semantic_gradient|=gn['semantic']>0;body_gradient|=gn['value_output']>0
        norm=torch.nn.utils.clip_grad_norm_(world.parameters(),config['clip_norm']);opt.step()
        loss_mean=parallel.mean(total);components={k:parallel.mean(v) for k,v in parts_total.items()}
        gathered=parallel.gather(dict(sample_ids=ids,noise_samples=noise_log));peak=max(parallel.gather(torch.cuda.max_memory_allocated(device)/2**30))
        times.append(time.monotonic()-tick);seconds=sum(times[-10:])/len(times[-10:]);stop=parallel.any(a.stop_file is not None and a.stop_file.exists())
        if parallel.main:
            record=dict(step=step+1,optimizer_step=step+1,loss=loss_mean,components=components,lr=lr,grad_norm=float(norm),
                gradient_norms=gn,sample_ids=[sid for r in gathered for sid in r['sample_ids']],
                noise_samples=[n for r in gathered for n in r['noise_samples']],sample_exposures=(step+1)*4,
                passes=(step+1)*4/len(rows),replicas=parallel.size,binding_weight=0.,world_rank0_last_micro=diagnostics,
                elapsed_seconds=time.monotonic()-started,recent_seconds_per_step=seconds,eta_seconds=(a.steps-step-1)*seconds,peak_gpu_gib=peak)
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(record,flush=True)
        if (step+1)%config['save_every']==0 or step+1==a.steps or stop:
            check_replicas(world)
            if a.smoke and parallel.any(not (semantic_gradient and body_gradient and zero_equivalent)):
                raise AssertionError('Smoke needs zero-gate equivalence plus semantic and V/output gradients')
            if parallel.main:
                path=out/f'checkpoint-step{step+1:06d}';path.mkdir();atomic_tensors(path/'world.safetensors',world.state_dict())
                torch.save(opt.state_dict(),path/'optimizer.pt');atomic_json(path/'config.json',config)
                if a.smoke:atomic_json(path/'SMOKE_COMPLETE.json',dict(version=d.VERSION,replicas=parallel.size,zero_gate_equivalent=True,
                    semantic_gradient=True,value_output_gradient=True,base_trainable=0,binding_weight=0.,caption_condition=True,
                    ids=[r['id'] for r in rows],maximum_objects=max(len(r['objects']) for r in rows),peak_gpu_gib=peak))
                atomic_json(path/'COMPLETE.json',dict(version=d.VERSION,step=step+1,optimizer_step=step+1))
            parallel.barrier()
        if stop:break
    if parallel.main:atomic_json(out/'COMPLETE.json',dict(version=d.VERSION,step=step+1,checkpoint=str(path),automatic_evaluation=False))
    parallel.barrier();parallel.close()


if __name__=='__main__':main()
