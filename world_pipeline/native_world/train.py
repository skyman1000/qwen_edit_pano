"""Two replicas train ONE fresh World; synchronized gradients after local mean."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from ..build_polished_world import atomic_json
from ..common import new_output
from ..object_world.data import tensorize, atomic_tensors
from ..oracle_memory import ActivationStorage, training_autocast
from ..scaled_world.runtime import EpochSampler, learning_rate
from ..scaled_world.binding import check_scale
from ..id_binding_world.parallel import Parallel, micro_indices, seed_microbatch, average_gradients, check_replicas
from ..grounded_world.evaluate import transformer
from ..grounded_world.continue_train import predict
from .runtime import WorldBranch
from ..binding_refine_world.supervision import sample_binding
from ..binding_refine_world.train import objective
from . import data as d


def gradient_norms(world):
    groups = {}
    for name, parameter in world.named_parameters():
        key = ('semantic_connector' if 'encoder.semantic.' in name else
               'geometry' if 'encoder.geometry.' in name else
               'global' if name == 'encoder.empty' else
               'gates' if name.endswith('.gate') else
               'key_delta' if '.key_delta.' in name else 'value_delta')
        if parameter.grad is not None:
            groups[key] = groups.get(key, 0.) + float(parameter.grad.float().square().sum())
    return {key:value**.5 for key,value in groups.items()}


def schedule(config, step):
    ramp = min(1., (step+1)/config['binding_ramp_steps'])
    return (learning_rate(config['lr'],config['warmup_steps'],step),
            config['coverage_weight']*ramp,config['concentration_weight']*ramp)


def save_checkpoint(out, world, opt, config, step):
    if d.optimizer_step(opt) != step:
        raise AssertionError('Adam/checkpoint step mismatch')
    if any(not torch.isfinite(value[key]).all() for value in opt.state.values()
           for key in ('exp_avg','exp_avg_sq')):
        raise FloatingPointError('Refusing nonfinite Adam checkpoint')
    path = out/f'checkpoint-step{step:06d}';path.mkdir()
    atomic_tensors(path/'world.safetensors', world.state_dict())
    temporary = path/'optimizer.tmp.pt'
    torch.save(opt.state_dict(), temporary);temporary.replace(path/'optimizer.pt')
    atomic_json(path/'config.json', config)
    # COMPLETE is written LAST; partial writes are never used for restart.
    atomic_json(path/'COMPLETE.json', dict(version=d.VERSION,step=step,optimizer_step=step))
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-file',type=Path);p.add_argument('--manual-stop-file',type=Path);a=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 2:
        raise ValueError('Use torchrun with exactly two allocated CUDA GPUs')
    if not 2 <= a.steps <= 10000:
        p.error('Use a finite total target between 2 and 10000 steps')
    if a.smoke and a.resume:
        p.error('Smoke always starts from untrained initialization')
    torch.set_num_threads(4)
    plan, rows = d.inputs();config=d.configuration(plan,a.smoke)
    start=d.resume_step(a.resume,config);target=2 if a.smoke else a.steps
    if start>=target:
        raise ValueError('Already at target')
    if not torch.cuda.is_available():
        raise ValueError('User must allocate two BF16 CUDA GPUs')
    parallel=Parallel();device=parallel.device
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 GPU required')
    out=a.output.resolve()
    if parallel.main:
        new_output(out);atomic_json(out/'config.json',config)
        atomic_json(out/'run_plan.json',dict(start=start,target=target,resume=str(a.resume) if a.resume else None,
            initial_world='native projection anchors + new zero residual connectors',fresh_optimizer=a.resume is None,replicas=2,global_batch=4,
            automatic_evaluation=False,inherited_smoke=False,caption=False))
    parallel.barrier()
    model=transformer(plan,config,device);model.enable_gradient_checkpointing()
    if any(isinstance(m,torch.nn.Dropout) and m.training and m.p>0 for m in model.modules()):
        raise AssertionError('Frozen dropout active')
    # Identical new trainable tensors on both replicas; never load old A/B/C.
    torch.manual_seed(config['seed'])
    world=WorldBranch.from_model(model,d.load_features(),blocks=tuple(config['blocks']),
        binding_block=config['binding_block'],rank=config['rank'],
        connector_rank=config['connector_rank'],geometry_width=config['geometry_width'])
    opt=torch.optim.AdamW(world.parameters(),lr=config['lr'],betas=tuple(config['betas']),
        eps=config['adam_eps'],weight_decay=config['weight_decay'])
    if a.resume:
        if d.restore(world,opt,a.resume) != start:
            raise ValueError('Restore step mismatch')
    else:
        if opt.state:
            raise AssertionError('Fresh Adam must have empty state')
        if parallel.main:
            atomic_tensors(out/'initial_world.safetensors',world.state_dict())
    world.attach(model);check_replicas(world)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError('Base not frozen')
    if any(p.dtype!=torch.float32 for p in world.parameters()):
        raise AssertionError('FP32 World masters required')
    scale_check=check_scale(model,config['pano_lora_scale'],'default')
    runtime=parallel.gather(dict(rank=parallel.rank,gpu=torch.cuda.get_device_name(device),
        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30))
    if parallel.main:
        atomic_json(out/'pano_scale_check.json',scale_check)
        atomic_json(out/'runtime.json',dict(devices=runtime,torch=torch.__version__,cuda=torch.version.cuda,
            world_parameters=sum(p.numel() for p in world.parameters()),base_trainable=0,
            replicas=2,global_batch=4,local_accumulation=2,optimizer_start_step=start,
            gradient_reduction='mean of rank-local two-microbatch means; clip after reduction',
            fresh_optimizer=a.resume is None,inherited_smoke=False))
    from diffusers import AutoencoderKLQwenImage,FlowMatchEulerDiscreteScheduler,QwenImageEditPlusPipeline as Pipeline
    from qwen_pano.profiles import REPO_PANORAMA
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean'],device=device).reshape(1,-1,1,1,1)
    std=torch.tensor(vae['latents_std'],device=device).reshape(1,-1,1,1,1)
    if 2**len(vae['temperal_downsample'])!=8:
        raise ValueError('Unexpected VAE scale')
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to(device);timesteps=scheduler.timesteps.to(device)
    lossargs=SimpleNamespace(**REPO_PANORAMA);sampler=EpochSampler(len(rows),config['seed'])
    representatives=d.smoke_rows(rows);started=time.monotonic()
    encoder_seen=False;category_seen=False;aux_seen=False;negative_seen=False;equal_forward=False;zero_gate_aux=False
    value_seen=False
    for step in range(start,target):
        tick=time.monotonic();opt.zero_grad(set_to_none=True)
        lr,wc,wf=schedule(config,step)
        for group in opt.param_groups:group['lr']=lr
        total=0.;parts_total={};ids=[];noise_log=[];binding_log=[]
        for index in micro_indices(step,4,parallel.rank,2):
            seed_microbatch(config['seed'],index)
            row=representatives[(index+step)%len(representatives)] if a.smoke else rows[sampler.index(index)]
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
            weights=load_file(row['region_cache'],device=str(device))['weights']
            labels,meta=sample_binding(row,plan['height'],plan['padding_columns'],config['seed']*1000003+index)
            labels=tuple(t.to(device) for t in labels);binding_log.append(meta)
            first_local=index==step*4+parallel.rank*2
            if a.smoke and step==0 and first_local:
                with torch.no_grad(),training_autocast('cuda'):
                    world.set_supervision(None);world.set_world(None)
                    baseline=model(**kwargs)[0][:,:(plan['height']//16)*(plan['height']//8)]
                    baseline=Pipeline._unpack_latents(baseline,plan['height'],2*plan['height'],8)
                    plain=predict(model,world,state,kwargs,plan['height'])
                    labelled=predict(model,world,state,kwargs,plan['height'],labels);world.take_binding_loss()
                    torch.testing.assert_close(plain,baseline,rtol=0,atol=0)
                    torch.testing.assert_close(plain,labelled,rtol=0,atol=0);equal_forward=True
                    del baseline,plain,labelled
                world.clear_context()
            with ActivationStorage(enabled=True).context():
                with training_autocast('cuda'):
                    prediction=predict(model,world,state,kwargs,plan['height'],labels)
                    aux=world.take_binding_loss()
                    loss,parts=objective(prediction,velocity,weights,noise,z,lossargs,aux,wc,wf)
                if not torch.isfinite(loss):raise FloatingPointError(row['id'])
                (loss/2).backward()  # Two local microbatches; then mean the two ranks.
            if a.smoke and step==0 and first_local:
                needed=(world.branches['40'].key_delta.up.weight,world.encoder.semantic.up.weight,
                        world.encoder.geometry[-1].weight)
                if not all(p.grad is not None and bool(p.grad.count_nonzero()) for p in needed):
                    raise AssertionError('Binding must reach key delta / semantic connector / geometry at zero gates')
                zero_gate_aux=True
            aux_seen |= float(aux[0].detach())>0;negative_seen |= float(aux[1].detach())>0
            total+=float(loss.detach())/2
            for key,value in parts.items():parts_total[key]=parts_total.get(key,0.)+float(value.detach())/2
            noise_log.append(dict(index=index,timestep_index=idx,sigma=float(sigma)))
            world.clear_context()
            del posterior,z,noise,noisy,velocity,condition,packed,kwargs,state,weights,labels,prediction,aux,loss,parts
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen base gradients')
        if parallel.any(any(p.grad is not None and not torch.isfinite(p.grad).all() for p in world.parameters())):
            raise FloatingPointError('Nonfinite World gradients')
        average_gradients(world)
        encoder_seen |= any(p.grad is not None and bool(p.grad.count_nonzero()) for p in world.encoder.parameters())
        category_seen |= world.encoder.semantic.up.weight.grad is not None and bool(world.encoder.semantic.up.weight.grad.count_nonzero())
        value_seen |= any(b.value_delta.up.weight.grad is not None and bool(b.value_delta.up.weight.grad.count_nonzero())
                          for b in world.branches.values())
        groups=gradient_norms(world);norm=torch.nn.utils.clip_grad_norm_(world.parameters(),config['clip_norm']);opt.step()
        if d.optimizer_step(opt)!=step+1:raise AssertionError('Adam step')
        if step==start:check_replicas(world)
        total=parallel.mean(total);parts_total={k:parallel.mean(v) for k,v in parts_total.items()}
        ranks=parallel.gather(dict(sample_ids=ids,noise_samples=noise_log,binding_samples=binding_log,
            peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
            encoder_gradient=encoder_seen,category_gradient=category_seen,
            value_generation_gradient=value_seen,
            smoke_ok=equal_forward and zero_gate_aux and encoder_seen and category_seen and value_seen and aux_seen and negative_seen,
            world=world.diagnostics()))
        log=dict(step=step+1,loss=total,components=parts_total,lr=lr,coverage_weight=wc,concentration_weight=wf,
            grad_norm=float(norm),gradient_norms=groups,sample_ids=[s for r in ranks for s in r['sample_ids']],
            noise_samples=[s for r in ranks for s in r['noise_samples']],rank_metrics=ranks,
            passes_seen=(step+1)*4/len(rows),seconds=time.monotonic()-tick,
            elapsed_seconds=time.monotonic()-started,peak_gpu_gib=max(r['peak_gpu_gib'] for r in ranks),
            remaining_seconds_estimate=(time.monotonic()-started)/(step+1-start)*(target-step-1))
        if parallel.main:
            with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(log,flush=True)
        stop=parallel.any(any(path is not None and path.exists() for path in (a.stop_file,a.manual_stop_file)))
        if (step+1)%config['save_every']==0 or step+1==target or stop:
            check_replicas(world)
            if parallel.main:
                ckpt=save_checkpoint(out,world,opt,config,step+1)
                if a.smoke and step+1==2:
                    if not all(r['smoke_ok'] for r in ranks):raise AssertionError('Dual smoke did not pass on both ranks')
                    atomic_json(ckpt/'SMOKE_COMPLETE.json',dict(version=d.VERSION,config=config,
                        both_ranks_passed=True,zero_gate_baseline_exact=True,labels_do_not_change_forward=True,
                        zero_gate_connector_key_gradients=True,encoder_gradient=True,base_frozen=True,
                        replica_parameters_equal=True,optimizer_step=2,fresh_initialization=True,
                        both_binding_terms=True,value_generation_gradient=True,peak_gpu_gib=log['peak_gpu_gib']))
            parallel.barrier()
        if stop:
            if parallel.main:atomic_json(out/'STOPPED.json',dict(step=step+1,checkpoint=str(ckpt)))
            parallel.close();return
    check_replicas(world)
    if parallel.main:
        atomic_json(out/'COMPLETE.json',dict(version=d.VERSION,step=target,checkpoint=str(ckpt),
            automatic_evaluation=False,note='Training finished; held-out generation determines quality'))
    parallel.barrier();parallel.close()


if __name__=='__main__':main()
