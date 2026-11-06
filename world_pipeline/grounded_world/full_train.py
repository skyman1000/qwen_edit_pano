"""One fixed Full-C model, one/two GPU replicas, exact global batch four."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from ..build_polished_world import read, atomic_json
from ..common import new_output
from ..object_world.data import tensorize, atomic_tensors
from ..object_world.runtime import object_region_loss
from ..oracle_memory import ActivationStorage, training_autocast
from ..scaled_world.runtime import EpochSampler
from ..id_binding_world.parallel import Parallel, micro_indices, seed_microbatch, average_gradients, check_replicas
from .data import load_support
from .runtime import WorldBranch
from .supervision import sample
from .evaluate import transformer
from .continue_train import predict
from .train import draw_latents, norms
from . import full_data as fd


def sample_indices(step, replicas=1, rank=0):
    """Separate NEW full-data order from the inherited stochastic-noise index."""
    return [(i, 600*4+i) for i in micro_indices(step,4,rank,replicas)]


def resume_step(checkpoint, config):
    if checkpoint is None:return 0,fd.SOURCE
    checkpoint=Path(checkpoint);marker=read(checkpoint/'COMPLETE.json')
    if read(checkpoint/'config.json')!=config or marker['version']!=fd.VERSION:
        raise ValueError('Full-C resume belongs to different data/model/settings')
    if not all((checkpoint/name).is_file() for name in ('world.safetensors','optimizer.pt')):
        raise ValueError('Incomplete Full-C checkpoint')
    return marker['step'],checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--steps',type=int,default=2000);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-file',type=Path);a=p.parse_args();torch.set_num_threads(4)
    plan,_,rows,features=fd.full_inputs();config=fd.configuration(plan,rows,a.smoke)
    fd.require_prepared(plan,rows);start,checkpoint=resume_step(a.resume,config)
    if not 0 <= start < a.steps <= config['maximum_new_steps'] or a.smoke and a.steps!=2:
        raise ValueError('Use a larger new-step target <=10000; smoke is exactly two steps')
    parallel=Parallel()
    if parallel.size not in (1,2):raise ValueError('Use one or two GPU replicas; global batch stays four')
    if not torch.cuda.is_bf16_supported():raise ValueError('BF16 CUDA required')
    if parallel.main:
        out=new_output(a.output);atomic_json(out/'config.json',config)
        atomic_json(out/'run_plan.json',dict(resume=str(a.resume) if a.resume else None,initialization_checkpoint=str(checkpoint),start=start,target=a.steps,replicas=parallel.size,
            source_C_step=600,new_full_steps=True,automatic_evaluation=False,automatic_extension=False))
    parallel.barrier();out=Path(a.output);device=parallel.device
    model=transformer(plan,config,device);model.enable_gradient_checkpointing()
    if any(isinstance(m,torch.nn.Dropout) and m.training and m.p>0 for m in model.modules()):
        raise AssertionError('Frozen base dropout must remain inactive')
    hidden=model.config.num_attention_heads*model.config.attention_head_dim
    world=WorldBranch(hidden,features,blocks=tuple(config['blocks']),width=config['width'],
        height=plan['height'],padding=plan['padding_columns'],binding_block=config['binding_block']).to(device)
    world.load_state_dict(load_file(str(checkpoint/'world.safetensors')),strict=True)
    opt=torch.optim.AdamW(world.parameters(),lr=config['lr'],betas=tuple(config['betas']),eps=config['adam_eps'],weight_decay=config['weight_decay'])
    opt.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location=device,weights_only=True))
    optimizer_max_step=max(float(v['step']) for v in opt.state.values())
    if optimizer_max_step!=600+start:raise ValueError('Optimizer step disagrees with the C600/full-checkpoint ancestry')
    if any(p.dtype!=torch.float32 for p in world.parameters()):raise AssertionError('FP32 World masters required')
    for state in opt.state.values():
        if any(state[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq') if k in state):
            raise AssertionError('FP32 optimizer moments required')
    check_replicas(world)
    if parallel.main:
        if not a.resume:atomic_tensors(out/'initial_world.safetensors',world.state_dict())
        atomic_json(out/'runtime.json',dict(base_trainable=0,world_trainable=sum(p.numel() for p in world.parameters()),
            gpu=torch.cuda.get_device_name(device),replicas=parallel.size,global_batch=4,local_accumulation=4//parallel.size,
            initial_optimizer_max_step=optimizer_max_step,pano_lora_scale=.5))
    world.attach(model)
    if any(p.requires_grad for p in model.parameters()):raise AssertionError('Base must stay frozen')
    from diffusers import AutoencoderKLQwenImage,FlowMatchEulerDiscreteScheduler,QwenImageEditPlusPipeline as Pipeline
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device);lossargs=SimpleNamespace(**REPO_PANORAMA)
    sampler=EpochSampler(len(rows),config['seed']);local_accumulation=4//parallel.size
    smoke_rows=[max(rows,key=lambda r:len(r['objects'])),rows[0]]
    started=time.monotonic();times=[];semantic_gradient=False;value_output_gradient=False
    for step in range(start,a.steps):
        tick=time.monotonic();opt.zero_grad(set_to_none=True);parts_total={};ids=[];noise_log=[];total=0.;diagnostics={}
        for group in opt.param_groups:group['lr']=config['lr']
        for data_index,noise_index in sample_indices(step,parallel.size,parallel.rank):
            seed_microbatch(config['seed'],noise_index)
            row=smoke_rows[data_index%2] if a.smoke else rows[sampler.index(data_index)]
            ids.append(row['id']);posterior=load_file(row['posterior_cache'],device='cpu')
            condition=load_file(row['condition_cache'],device=str(device))
            z,noise,idx=draw_latents(posterior,mean,std,config['seed'],noise_index,scheduler.config.num_train_timesteps)
            z=z.to(device);noise=noise.to(device);velocity=noise-z;sigma=sigmas[idx].to(z.dtype)
            noisy=(1-sigma)*z+sigma*noise;packed=Pipeline._pack_latents(noisy,1,z.shape[1],z.shape[3],z.shape[4])
            kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=timesteps[idx:idx+1]/1000,
                encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,z.shape[3]//2,z.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            state=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
            weights=load_file(row['region_cache'],device=str(device))['weights']
            labels,_=sample(load_support(fd.SUPPORT,row['id']),config['seed']*1000003+noise_index)
            labels=tuple(t.to(device) for t in labels)
            with ActivationStorage(enabled=True).context():
                with training_autocast(device.type):
                    prediction=predict(model,world,state,kwargs,plan['height'],labels);diagnostics=dict(world.diagnostics())
                    base,parts=panorama_loss(prediction,velocity,torch.ones(1,1,*z.shape[-2:],device=device,dtype=torch.bool),
                        torch.ones(1,device=device,dtype=torch.bool),torch.ones(1,device=device),lossargs.geometry_start_epoch,lossargs,noise=noise,clean=z)
                    region=object_region_loss(prediction,velocity,weights);binding=world.take_binding_loss()
                    loss=base+.25*region+.01*binding
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/local_accumulation).backward()
            total+=float(loss.detach())/local_accumulation
            for k,v in {**parts,'object_region':region,'binding':binding}.items():parts_total[k]=parts_total.get(k,0.)+float(v.detach())/local_accumulation
            noise_log.append(dict(data_index=data_index,noise_index=noise_index,index=idx,sigma=float(sigma)))
            world.clear_context()
            del prediction,base,parts,region,binding,loss,posterior,condition,z,noise,velocity,sigma,noisy,packed,kwargs,state,weights,labels
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base got gradients')
        if parallel.any(any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters())):
            raise FloatingPointError('Nonfinite World gradients')
        average_gradients(world);gn=norms(world)
        semantic_gradient|=gn['semantic']>0;value_output_gradient|=gn['value_output']>0
        norm=torch.nn.utils.clip_grad_norm_(world.parameters(),config['clip_norm']);opt.step()
        loss_mean=parallel.mean(total);components={k:parallel.mean(v) for k,v in parts_total.items()}
        gathered=parallel.gather(dict(sample_ids=ids,noise_samples=noise_log));peak=max(parallel.gather(torch.cuda.max_memory_allocated(device)/2**30))
        times.append(time.monotonic()-tick);seconds=sum(times[-10:])/len(times[-10:]);stop=parallel.any(a.stop_file is not None and a.stop_file.exists())
        if parallel.main:
            log=dict(step=step+1,optimizer_step=600+step+1,loss=loss_mean,components=components,gradient_norms=gn,grad_norm=float(norm),
                sample_ids=[s for r in gathered for s in r['sample_ids']],noise_samples=[s for r in gathered for s in r['noise_samples']],
                full_sample_exposures=(step+1)*4,full_passes=(step+1)*4/len(rows),world_rank0_last_micro=diagnostics,
                elapsed_seconds=time.monotonic()-started,recent_seconds_per_step=seconds,eta_seconds=(a.steps-step-1)*seconds,
                peak_gpu_gib=peak,replicas=parallel.size)
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(log,allow_nan=False)+'\n')
            print(log,flush=True)
        if (step+1)%config['save_every']==0 or step+1==a.steps or stop:
            check_replicas(world)
            if parallel.main:
                path=out/f'checkpoint-step{step+1:06d}';path.mkdir();atomic_tensors(path/'world.safetensors',world.state_dict())
                torch.save(opt.state_dict(),path/'optimizer.pt');atomic_json(path/'config.json',config)
                if a.smoke:
                    if not semantic_gradient or not value_output_gradient:raise AssertionError('Smoke did not update the actual semantic/V/output paths')
                    atomic_json(path/'SMOKE_COMPLETE.json',dict(version=fd.VERSION,replicas=parallel.size,
                        maximum_objects=max(len(r['objects']) for r in smoke_rows),base_trainable=0,peak_gpu_gib=peak,
                        semantic_gradient=True,value_output_gradient=True))
                atomic_json(path/'COMPLETE.json',dict(version=fd.VERSION,step=step+1,optimizer_step=601+step))
            parallel.barrier()
        if stop:break
    if parallel.main:atomic_json(out/'COMPLETE.json',dict(version=fd.VERSION,step=step+1,checkpoint=str(path),
        requested_new_steps=a.steps,automatic_evaluation=False,automatic_extension=False))
    parallel.barrier();parallel.close()


if __name__=='__main__':main()
