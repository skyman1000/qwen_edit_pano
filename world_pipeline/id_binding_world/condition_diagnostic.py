"""User-run fixed-noise A1000/A10000 diagnostics, with six NEW World val cases.

Forward only: no optimizer, backward, checkpoint update or panorama generation.
Missing selected-val caches are built inside this diagnostic output, never train.
"""
import argparse
import gc
import random
import time
from pathlib import Path
from types import SimpleNamespace

from ..build_polished_world import ROOT,read,atomic_json
from ..common import new_output
from ..object_world.data import DEFAULT_PACK,load_pack,tensorize,atomic_tensors,target_image
from .runtime import VERSION,WorldBranch,object_region_loss
from .condition_diagnostic_data import MODES,ensure_panel,condition_variant,case_support,checkpoint_exposures
from .condition_diagnostic_probe import ActualAttentionProbe
from .state_probe import DEFAULT_CHECKPOINT,DEFAULT_REFERENCE

DEFAULT_PANEL=ROOT/'research/audits/id_condition_diagnostic_2026-10-25/PANEL.json'
DEFAULT_OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_id_condition_diagnostic_v1'


def cache_paths(row,output):
    directory=output/'cache'/row['id']
    return {key:(Path(row[key]) if Path(row[key]).is_file() else directory/(key+'.safetensors'))
            for key in ('condition_cache','posterior_cache','region_cache')}


def build_missing_caches(cases,byid,plan,output):
    import torch
    import numpy as np
    from ..object_world.runtime import region_weights
    from ..object_world.cache import check_condition,check_posterior
    from qwen_edit_pano.paired_data import local_image
    from diffusers import AutoencoderKLQwenImage,QwenImageEditPlusPipeline
    source_file=output/'CACHE_SOURCES.json'
    if source_file.exists():
        saved=read(source_file)
        if set(saved)!={c['id'] for c in cases}:raise ValueError('Saved cache panel differs')
        paths={sid:{k:Path(v) for k,v in item.items()} for sid,item in saved.items()}
    else:
        paths={c['id']:cache_paths(byid[c['id']],output) for c in cases}
        atomic_json(source_file,{sid:{k:str(v.resolve()) for k,v in item.items()} for sid,item in paths.items()})
    missing_conditions=[c for c in cases if not paths[c['id']]['condition_cache'].is_file()]
    missing_posteriors=[c for c in cases if not paths[c['id']]['posterior_cache'].is_file()]
    device=torch.device('cuda:0')
    if missing_conditions:
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions,CONDITION_IMAGE_SIZE,VAE_IMAGE_SIZE
        vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
        pipe=QwenImageEditPlusPipeline.from_pretrained(plan['base_snapshot'],transformer=None,vae=vae,
            torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.text_encoder.requires_grad_(False).eval().to(device)
        pipe.vae.requires_grad_(False).eval().to(device=device,dtype=torch.float32)
        with torch.inference_mode():
            for index,c in enumerate(missing_conditions,1):
                row=byid[c['id']];image=local_image(row)
                cw,ch=calculate_dimensions(CONDITION_IMAGE_SIZE,image.width/image.height)
                vw,vh=calculate_dimensions(VAE_IMAGE_SIZE,image.width/image.height)
                condition=pipe.image_processor.resize(image,ch,cw)
                pixels=pipe.image_processor.preprocess(image,vh,vw).unsqueeze(2).to(device,torch.bfloat16).float()
                embeddings,mask=pipe.encode_prompt(prompt=plan['prompt'],image=[condition],device=device)
                latent=pipe._encode_vae_image(pixels,generator=None).to(torch.bfloat16)
                reference=pipe._pack_latents(latent,1,latent.shape[1],latent.shape[3],latent.shape[4])
                if mask is None:mask=torch.ones(embeddings.shape[:2],device=device,dtype=torch.bool)
                atomic_tensors(paths[row['id']]['condition_cache'],dict(embeddings=embeddings[0],mask=mask[0].bool(),
                    reference=reference[0],ref_shape=torch.tensor([1,vh//16,vw//16])))
                print(dict(stage='selected_local_cache',completed=index,total=len(missing_conditions),id=row['id']),flush=True)
        del pipe,vae;gc.collect();torch.cuda.empty_cache()
    if missing_posteriors:
        from PIL import Image
        vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
        vae.requires_grad_(False).eval().to(device)
        with torch.inference_mode():
            for index,c in enumerate(missing_posteriors,1):
                row=byid[c['id']];rgb=target_image(row).resize((2*plan['height'],plan['height']),Image.Resampling.BICUBIC)
                pixels=torch.from_numpy(np.array(rgb).copy()).permute(2,0,1).float()/127.5-1
                posterior=vae.encode(pixels[None,:,None].to(device)).latent_dist
                atomic_tensors(paths[row['id']]['posterior_cache'],dict(mean=posterior.mean,std=posterior.std))
                del pixels,posterior
                print(dict(stage='selected_target_cache',completed=index,total=len(missing_posteriors),id=row['id']),flush=True)
        del vae;gc.collect();torch.cuda.empty_cache()
    for case in cases:
        row=byid[case['id']];p=paths[row['id']]
        if not p['region_cache'].is_file():
            with np.load(row['object_region_supervision']['path'],allow_pickle=False) as f:
                weights=region_weights(f['instance'],f['object_loss_valid'],
                    [int(o['instance_id']) for o in row['objects']],(plan['height']//8,plan['height']//4))
            atomic_tensors(p['region_cache'],dict(weights=weights))
        check_condition(p['condition_cache']);check_posterior(p['posterior_cache'],plan['height'])
    return paths


def fixed_noise(posterior,mean,std,seed):
    """CPU RNG makes checkpoint/mode/resume comparisons independent of GPU RNG."""
    import torch
    generator=torch.Generator(device='cpu').manual_seed(seed)
    sample=posterior['mean'].float()+posterior['std'].float()*torch.randn(posterior['std'].shape,generator=generator)
    clean=((sample-mean.cpu())/std.cpu()).to(torch.bfloat16)
    noise=torch.randn(clean.shape,generator=generator,dtype=torch.float32).to(torch.bfloat16)
    return clean,noise


def metrics(prediction,target,weights,instance_weights,noise,clean,config,metric_seed):
    import torch
    from qwen_edit_pano.losses import panorama_loss
    from qwen_pano.profiles import REPO_PANORAMA
    args=SimpleNamespace(**REPO_PANORAMA)
    # Exactly the current objective, with the SAME yaw choice in every arm.
    previous=random.getstate();random.seed(metric_seed)
    try:
        loss,parts=panorama_loss(prediction,target,
            torch.ones(1,1,*target.shape[-2:],device=target.device,dtype=torch.bool),
            torch.ones(1,device=target.device,dtype=torch.bool),torch.ones(1,device=target.device),
            args.geometry_start_epoch,args,noise=noise,clean=clean)
    finally:random.setstate(previous)
    region=object_region_loss(prediction,target,weights)
    error=(prediction.float()-target.float()).square().mean((0,1,2))
    return dict(total=float(loss+config['region_weight']*region),
        components={k:float(v) for k,v in parts.items()},object_region=float(region),
        selected_object_errors=(instance_weights*error).sum((-2,-1)).cpu().tolist())


def record_path(output,step,sid,index,mode):return output/'records'/f'{step:06d}'/sid/f'{index:04d}'/(mode+'.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',type=Path,default=[DEFAULT_REFERENCE,DEFAULT_CHECKPOINT])
    p.add_argument('--pack',type=Path,default=DEFAULT_PACK)
    p.add_argument('--panel',type=Path,default=DEFAULT_PANEL)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--train-count',type=int,default=4)
    p.add_argument('--val-count',type=int,default=6)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--sigmas',nargs='+',type=float,default=[.2,.5,.8])
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    if min(args.train_count,args.val_count)<1 or not args.sigmas or any(not 0<s<1 for s in args.sigmas):p.error('Invalid counts/sigmas')
    if len(set(args.sigmas))!=len(args.sigmas):p.error('Duplicate sigma values')
    plan,records=load_pack(args.pack);byid={r['id']:r for r in records}
    configurations=[read(path/'config.json') for path in args.checkpoints]
    steps=[read(path/'COMPLETE.json')['step'] for path in args.checkpoints]
    if len(set(steps))!=len(steps):p.error('Checkpoints must have distinct steps')
    for path,c in zip(args.checkpoints,configurations):
        if c['version']!=VERSION or c['group']!='A' or c['pack_plan']!=plan:raise ValueError('Use current A checkpoints bound to this pack')
        if read(path/'COMPLETE.json')['version']!=VERSION:raise ValueError('Checkpoint incomplete')
        if any(c[k]!=configurations[0][k] for k in ('blocks','width','pano_lora_scale','category_encoding','region_weight')):
            raise ValueError('Checkpoint architectures/objectives differ')
    exposures={step:checkpoint_exposures(path) for step,path in zip(steps,args.checkpoints)}
    panel=ensure_panel(args.panel,plan,records,args.train_count,args.val_count,args.seed,set(exposures[min(steps)]))
    exposure_check={str(step):{c['id']:counter[c['id']] for c in panel['cases']} for step,counter in exposures.items()}
    if any(exposures[s][c['id']] for s in steps for c in panel['cases'] if c['split']=='val'):
        raise AssertionError('A selected validation image appears in optimization history')
    if any(not exposures[min(steps)][c['id']] for c in panel['cases'] if c['split']=='train'):
        raise AssertionError('Training panel must already be seen at the earliest checkpoint')
    atomic_json(args.panel.parent/'EXPOSURE_CHECK.json',exposure_check)
    import torch
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler,AutoencoderKLQwenImage
    from qwen_edit_pano.common import versions
    torch.set_num_threads(4)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas[:scheduler.config.num_train_timesteps]
    indices=[int((sigmas-value).abs().argmin()) for value in args.sigmas]
    if len(set(indices))!=len(indices):raise ValueError('Duplicate scheduler indices')
    schedule=[dict(requested=s,index=i,sigma=float(sigmas[i].bfloat16()),timestep=float(scheduler.timesteps[i])) for s,i in zip(args.sigmas,indices)]
    architecture=read(Path(plan['base_snapshot'])/'transformer/config.json')
    hidden=architecture['num_attention_heads']*architecture['attention_head_dim']
    cpu_world=WorldBranch(hidden,configurations[0]['blocks'],configurations[0]['width'])
    for path in args.checkpoints:
        state=load_file(str(path/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite World checkpoint')
        cpu_world.load_state_dict(state,strict=True)
    del cpu_world,state
    protocol=dict(version='id_world_fixed_noise_diagnostic_v1',panel=panel,checkpoints=[str(p.resolve()) for p in args.checkpoints],
        steps=steps,schedule=schedule,modes=list(MODES),base_checkpoint=plan['base_checkpoint'],base_snapshot=plan['base_snapshot'],
        pack=str(args.pack.resolve()),pano_lora_scale=configurations[0]['pano_lora_scale'],region_weight=configurations[0]['region_weight'],
        height=plan['height'],padding_columns=plan['padding_columns'],seed=args.seed,
        versions=versions(),
        training_exposures=exposure_check,
        note='Teacher-forced fixed-noise forward diagnostic; target latent contains GT, so not free-generation or semantic edit certification. No training updates, images generated, downloads or hashes.')
    missing={key:sum(not Path(byid[c['id']][key]).is_file() for c in panel['cases']) for key in ('condition_cache','posterior_cache','region_cache')}
    total=len(panel['cases'])*len(indices)*len(steps)*len(MODES)
    budget=dict(train=args.train_count,val=args.val_count,checkpoint_steps=steps,sigmas=schedule,condition_records=total,
        maximum_forward_calls=total,missing_selected_caches=missing,generated_panoramas=0,optimizer_steps=0,
        selected_ids=[c['id'] for c in panel['cases']],new_world_val_cases=sum(c['split']=='val' and not c['prior_world_diagnostic'] for c in panel['cases']))
    print(dict(stage='preflight',**budget),flush=True)
    if args.preflight:
        atomic_json(args.panel.parent/'PREFLIGHT.json',budget);return
    output=args.output.resolve()
    if output.exists():
        if read(output/'PROTOCOL.json')!=protocol:raise ValueError('Existing diagnostic settings differ; choose another --output')
    else:
        if args.report_only:raise ValueError('Missing diagnostic output')
        new_output(output);atomic_json(output/'PROTOCOL.json',protocol);atomic_json(output/'PANEL.json',panel)
    expected=[record_path(output,s,c['id'],i,m) for s in steps for c in panel['cases'] for i in indices for m in MODES]
    if args.report_only or all(p.is_file() for p in expected):
        from .condition_diagnostic_report import report
        report(output,protocol);return
    import fcntl
    run_lock=(output/'RUN.lock').open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Allocate one BF16 CUDA GPU')
    paths=build_missing_caches(panel['cases'],byid,plan,output)
    gc.collect();torch.cuda.empty_cache()
    # Text encoder and VAE have been released BEFORE loading the 20B denoiser.
    from diffusers import QwenImageTransformer2DModel,QwenImageEditPlusPipeline as Pipeline
    from peft import LoraConfig,set_peft_model_state_dict
    from qwen_edit_pano.paired_circular import install_circular_padding
    from ..scaled_world.binding import apply_scale
    from ..oracle_memory import training_autocast
    device=torch.device('cuda:0');base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    model=QwenImageTransformer2DModel.from_pretrained(plan['base_snapshot'],subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    model.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    bound=set_peft_model_state_dict(model,load_file(str(base/'adapter_resume.safetensors')))
    if bound.unexpected_keys or any('lora_' in k for k in bound.missing_keys):raise ValueError('Incomplete Panorama weights')
    atomic_json(output/'pano_scale_check.json',apply_scale(model,configurations[0]['pano_lora_scale'],'default'))
    model.to(device);install_circular_padding(model,plan['padding_columns'])
    world=WorldBranch(hidden,configurations[0]['blocks'],configurations[0]['width']).to(device).eval().requires_grad_(False)
    world.attach(model);probe=ActualAttentionProbe(world)
    if any(p.requires_grad for module in (model,world) for p in module.parameters()):
        raise AssertionError('Diagnostic must freeze every parameter')
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).view(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).view(1,-1,1,1,1)
    started=time.monotonic();completed=sum(p.is_file() for p in expected)
    for checkpoint,config,step in zip(args.checkpoints,configurations,steps):
        world.load_state_dict(load_file(str(checkpoint/'world.safetensors')),strict=True)
        for case_index,case in enumerate(panel['cases']):
            row=byid[case['id']];posterior=load_file(str(paths[row['id']]['posterior_cache']),device='cpu')
            clean,noise=fixed_noise(posterior,mean,std,case['latent_seed']);clean=clean.to(device);noise=noise.to(device);target=noise-clean
            condition=load_file(str(paths[row['id']]['condition_cache']),device='cuda:0')
            weights=load_file(str(paths[row['id']]['region_cache']),device='cuda:0')['weights']
            iw,spec=case_support(row,case,plan['height'],plan['padding_columns']);iw=iw.to(device)
            tensors=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
            for entry in schedule:
                index=entry['index'];group_paths=[record_path(output,step,row['id'],index,m) for m in MODES]
                if all(p.is_file() for p in group_paths):continue
                sigma=torch.tensor(entry['sigma'],device=device,dtype=clean.dtype)
                noisy=(1-sigma)*clean+sigma*noise;packed=Pipeline._pack_latents(noisy,1,clean.shape[1],clean.shape[3],clean.shape[4])
                kwargs=dict(hidden_states=torch.cat([packed,condition['reference'][None]],1),
                    timestep=scheduler.timesteps[index:index+1].to(device)/1000,
                    encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                    img_shapes=[[(1,clean.shape[3]//2,clean.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
                correct=None
                for mode,path in zip(MODES,group_paths):
                    if mode!='correct' and path.is_file():continue
                    world.clear_context();variant,perturbation=condition_variant(tensors,mode)
                    probe.reset(spec,active=mode!='world_off')
                    with torch.inference_mode(),training_autocast('cuda'):
                        world.set_world(variant)
                        prediction=model(**kwargs)[0][:,:packed.shape[1]]
                        prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],8)
                        if mode=='correct':correct=prediction.detach().clone()
                        values=metrics(prediction,target,weights,iw,noise,clean,config,case['latent_seed']+index)
                        difference=float((prediction.float()-correct.float()).square().mean().sqrt()/correct.float().square().mean().sqrt().clamp_min(1e-12))
                    if set(probe.records)!=(set(world.branches) if mode!='world_off' else set()):
                        raise AssertionError('World attention readout did not match the requested branch state')
                    record=dict(step=step,id=row['id'],split=row['split'],mode=mode,scheduler_index=index,sigma=entry['sigma'],
                        latent_seed=case['latent_seed'],relative_prediction_change=difference,**values,
                        perturbation=perturbation,attention=probe.records,selected_objects=spec['objects'],
                        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
                    if not path.exists():
                        path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,record);completed+=1
                    world.clear_context();del prediction
                    print(dict(stage='diagnostic_forward',step=step,id=row['id'],sigma=entry['sigma'],mode=mode,
                        completed=completed,total=total,elapsed_seconds=time.monotonic()-started),flush=True)
                del correct,kwargs,packed,noisy
            del clean,noise,target,posterior,condition,weights,iw,tensors
    probe.close()
    from .condition_diagnostic_report import report
    report(output,protocol)


if __name__=='__main__':main()
