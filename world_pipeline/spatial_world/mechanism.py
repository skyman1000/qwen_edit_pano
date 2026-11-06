"""User-run same-checkpoint local on/off + casting diagnostic. No training."""
import argparse
import fcntl
import math
import statistics as st
import time
from pathlib import Path

from ..build_polished_world import ROOT,read,atomic_json
from ..common import new_output
from ..object_world.data import load_pack,tensorize
from ..id_binding_world.condition_diagnostic_data import case_support
from .diagnose import cache_paths,fixed_noise,metrics,training_exposures
from .runtime import VERSION,load_world
from .mechanism_runtime import MechanismWorld,MODES

DEFAULT_CHECKPOINT=ROOT/'qwen_edit_pano/outputs/object_world_spatial_v1/pilot/spatial/run-1792923761168994486/checkpoint-step000200'
PANEL=ROOT/'research/audits/id_condition_diagnostic_2026-10-25/PANEL.json'
AUDIT=ROOT/'research/audits/spatial_mechanism_2026-10-25'
DEFAULT_OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_spatial_mechanism_v1'


def record_path(output,sid,index,mode):return output/'records'/sid/f'{index:04d}'/(mode+'.json')


def summarize(output,protocol):
    rows=[]
    for case in protocol['panel']['cases']:
        for entry in protocol['schedule']:
            for mode in MODES:
                path=record_path(output,case['id'],entry['index'],mode)
                if not path.is_file():raise ValueError('Incomplete mechanism diagnostic: '+str(path))
                row=read(path)
                expected=dict(id=case['id'],split=case['split'],mode=mode,scheduler_index=entry['index'],sigma=entry['sigma'],latent_seed=case['latent_seed'])
                if any(row[k]!=v for k,v in expected.items()):raise ValueError('Wrong mechanism record identity')
                if not all(math.isfinite(row[k]) for k in ('total','object_region','relative_prediction_change')):raise ValueError('Nonfinite mechanism record')
                if mode=='measured_v1' and not row.get('native_prediction_exact'):raise ValueError('Measured-v1 path did not reproduce native-v1')
                rows.append(row)
    lookup={(r['id'],r['scheduler_index'],r['mode']):r for r in rows}
    paired=[];layer=[]
    for split in ('train','val'):
        for entry in protocol['schedule']:
            for mode in MODES:
                selected=[r for r in rows if r['split']==split and r['scheduler_index']==entry['index'] and r['mode']==mode]
                paired.append(dict(split=split,sigma=entry['sigma'],mode=mode,images=len(selected),
                    full_gap_pct=st.mean(100*(r['total']/lookup[r['id'],entry['index'],'native_v1']['total']-1) for r in selected),
                    region_gap_pct=st.mean(100*(r['object_region']/lookup[r['id'],entry['index'],'native_v1']['object_region']-1) for r in selected),
                    prediction_change_pct=st.mean(100*r['relative_prediction_change'] for r in selected),
                    images_with_lower_full_loss=sum(r['total']<lookup[r['id'],entry['index'],'native_v1']['total'] for r in selected)))
                if mode not in ('measured_v1','single_cast'):continue
                for block in selected[0]['measurements']:
                    values=[r['measurements'][block] for r in selected]
                    layer.append(dict(split=split,sigma=entry['sigma'],mode=mode,block=block,
                        observed_hidden_dtypes=sorted({r['hidden_dtype'] for r in values}),
                        **{key:st.mean(r[key] for r in values) for key in ('nominal_relative_rms','actual_relative_rms',
                            'actual_to_nominal_rms','changed_element_fraction','changed_supported_element_fraction','supported_element_fraction')}))
    result=dict(checkpoint=protocol['checkpoint'],records=len(rows),paired_summary=paired,layer_summary=layer,
        all_measured_v1_predictions_exact=True,
        interpretation='Same Spatial checkpoint. Positive loss gap means native v1 has lower error. This is strength/numerics diagnosis, not semantic generation accuracy.')
    atomic_json(output/'SUMMARY.json',result);atomic_json(output/'RESULTS.json',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric,title in ((axes[0],'changed_supported_element_fraction','Changed elements within routed support (%)'),
                             (axes[1],'actual_relative_rms','Actual added residual / hidden RMS (%)')):
        for m,mode in enumerate(('measured_v1','single_cast')):
            values=[100*st.mean(r[metric] for r in layer if r['mode']==mode and r['block']==block) for block in ('20','40','59')]
            ax.bar(np.arange(3)+(m-.5)*.34,values,width=.32,label=mode)
        ax.set_xticks([0,1,2],['20','40','59']);ax.set_xlabel('Block');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Measured actual additions; same checkpoint, correct condition; not object accuracy')
    fig.tight_layout();fig.savefig(output/'LOCAL_EFFECT.png',dpi=150);plt.close(fig)
    import html
    def table(items,keys):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join(
            '<td>'+html.escape(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in items)+'</table>'
    body=['<!doctype html><meta charset="utf-8"><title>同权重local机制检查</title>',
        '<style>body{font-family:system-ui;margin:24px;max-width:1300px}p{line-height:1.7}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>',
        '<h1>同一Spatial checkpoint：local开关与回写精度</h1>',
        '<p>native_v1是原实现；measured_v1复现其运算并要求预测逐元素完全一致；local_off只关闭新增local，保留同一套Encoder和旧attention；single_cast把旧/新残差在FP32合并后统一回写。不训练、不修改权重。</p>',
        '<p>正Loss差值表示native_v1更低；actual_to_nominal_rms可能因量化跳变超过1，不能当成0～1的保留率。改变元素比例也不是对象识别或位置成功率。</p>',
        '<p>输入为GT加噪潜变量，只有正确状态。本诊断不证明完整生成控制改善，也不是新增训练方案的验收。</p>',
        '<img src="LOCAL_EFFECT.png">',table(paired,['split','sigma','mode','images','full_gap_pct','region_gap_pct','prediction_change_pct']),
        table(layer,['split','sigma','mode','block','observed_hidden_dtypes','nominal_relative_rms','actual_relative_rms','changed_supported_element_fraction'])]
    (output/'report.html').write_text('\n'.join(body))
    atomic_json(output/'COMPLETE.json',dict(version=protocol['version'],records=len(rows),optimizer_steps=0,generated_panoramas=0,
        checkpoint=protocol['checkpoint'],native_reproduction_exact=True))
    print(dict(stage='mechanism_complete',records=len(rows),output=str(output)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--output',type=Path)
    p.add_argument('--sigmas',type=float,nargs='+',default=[.8])
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args();args.checkpoint=args.checkpoint.resolve()
    if not args.sigmas or any(not 0<s<1 for s in args.sigmas):p.error('Use sigmas inside (0,1)')
    config=read(args.checkpoint/'config.json');complete=read(args.checkpoint/'COMPLETE.json')
    if config['version']!=VERSION or complete['version']!=VERSION or config['spatial_mode']!='spatial' or config['binding_weight']!=0:
        raise ValueError('Use the spatial v1 checkpoint without instance binding loss')
    plan,records=load_pack(config['pack']);byid={r['id']:r for r in records};panel=read(PANEL)
    if config['pack_plan']!=plan:raise ValueError('Training pack differs from checkpoint')
    exposures=training_exposures(args.checkpoint)
    if any(exposures[c['id']] for c in panel['cases'] if c['split']=='val'):raise ValueError('Validation image entered optimization history')
    output=(args.output or DEFAULT_OUTPUT/(args.checkpoint.parent.name+'-'+args.checkpoint.name)).resolve()
    paths={c['id']:cache_paths(byid[c['id']],output) for c in panel['cases']}
    if any(not path.is_file() for item in paths.values() for path in item.values()):raise ValueError('Missing fixed-panel caches; this short diagnostic never builds or downloads caches')
    import torch
    from safetensors.torch import load_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from qwen_edit_pano.common import versions
    torch.set_num_threads(4)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    indices=[int((scheduler.sigmas[:scheduler.config.num_train_timesteps]-value).abs().argmin()) for value in args.sigmas]
    if len(set(indices))!=len(indices):raise ValueError('Duplicate scheduler indices')
    schedule=[dict(requested=value,index=i,sigma=float(scheduler.sigmas[i].bfloat16()),timestep=float(scheduler.timesteps[i])) for value,i in zip(args.sigmas,indices)]
    architecture=read(Path(plan['base_snapshot'])/'transformer/config.json');hidden=architecture['num_attention_heads']*architecture['attention_head_dim']
    state=load_file(str(args.checkpoint/'world.safetensors'),device='cpu')
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
    world=MechanismWorld(hidden,config['blocks'],config['width'],plan['height'],plan['padding_columns']).eval().requires_grad_(False)
    load_world(world,state,config);del state
    protocol=dict(version='same_spatial_checkpoint_mechanism_v1',checkpoint=str(args.checkpoint),checkpoint_step=complete['step'],
        modes=list(MODES),panel=panel,schedule=schedule,pack=config['pack'],base_snapshot=plan['base_snapshot'],base_checkpoint=plan['base_checkpoint'],
        pano_lora_scale=config['pano_lora_scale'],region_weight=config['region_weight'],height=plan['height'],padding=plan['padding_columns'],
        versions=versions(),cache_sources={sid:{k:str(v) for k,v in item.items()} for sid,item in paths.items()},
        note='One unchanged checkpoint, forward only; same target/noise/timestep; no optimizer, generation, downloads, hashes or new training.')
    budget=dict(records=len(panel['cases'])*len(schedule)*len(MODES),train=4,val=6,schedule=schedule,missing_caches=0,
        checkpoint=str(args.checkpoint),optimizer_steps=0,generated_panoramas=0)
    print(dict(stage='mechanism_preflight',**budget),flush=True)
    if args.preflight:
        AUDIT.mkdir(parents=True,exist_ok=True);atomic_json(AUDIT/'PREFLIGHT.json',budget);return
    if output.exists():
        if read(output/'PROTOCOL.json')!=protocol:raise ValueError('Existing mechanism settings differ; choose a new --output')
    else:
        if args.report_only:raise ValueError('Missing mechanism output')
        new_output(output);atomic_json(output/'PROTOCOL.json',protocol)
    expected=[record_path(output,c['id'],entry['index'],mode) for c in panel['cases'] for entry in schedule for mode in MODES]
    if args.report_only or all(path.is_file() for path in expected):summarize(output,protocol);return
    lock=(output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Allocate one BF16 CUDA GPU')
    from diffusers import QwenImageTransformer2DModel,QwenImageEditPlusPipeline as Pipeline,AutoencoderKLQwenImage
    from peft import LoraConfig,set_peft_model_state_dict
    from qwen_edit_pano.paired_circular import install_circular_padding
    from ..scaled_world.binding import apply_scale
    from ..oracle_memory import training_autocast
    device=torch.device('cuda:0');base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    model=QwenImageTransformer2DModel.from_pretrained(plan['base_snapshot'],subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    model.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    loaded=set_peft_model_state_dict(model,load_file(str(base/'adapter_resume.safetensors')))
    if loaded.unexpected_keys or any('lora_' in key for key in loaded.missing_keys):raise ValueError('Incomplete Panorama adapter')
    atomic_json(output/'pano_scale_check.json',apply_scale(model,config['pano_lora_scale'],'default'))
    model.to(device);install_circular_padding(model,plan['padding_columns']);world.to(device);world.attach(model)
    if any(p.requires_grad for module in (model,world) for p in module.parameters()):raise AssertionError('Mechanism model must be entirely frozen')
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).view(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).view(1,-1,1,1,1)
    started=time.monotonic();completed=sum(path.is_file() for path in expected)
    for case in panel['cases']:
        row=byid[case['id']];posterior=load_file(str(paths[row['id']]['posterior_cache']),device='cpu')
        clean,noise=fixed_noise(posterior,mean,std,case['latent_seed']);clean=clean.to(device);noise=noise.to(device);target=noise-clean
        condition=load_file(str(paths[row['id']]['condition_cache']),device='cuda:0');weights=load_file(str(paths[row['id']]['region_cache']),device='cuda:0')['weights']
        iw,spec=case_support(row,case,plan['height'],plan['padding_columns']);iw=iw.to(device)
        tensors=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
        for entry in schedule:
            group=[record_path(output,row['id'],entry['index'],mode) for mode in MODES]
            if all(path.is_file() for path in group):continue
            sigma=torch.tensor(entry['sigma'],device=device,dtype=clean.dtype);noisy=(1-sigma)*clean+sigma*noise
            packed=Pipeline._pack_latents(noisy,1,clean.shape[1],clean.shape[3],clean.shape[4])
            kwargs=dict(hidden_states=torch.cat([packed,condition['reference'][None]],1),timestep=scheduler.timesteps[entry['index']:entry['index']+1].to(device)/1000,
                encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                img_shapes=[[(1,clean.shape[3]//2,clean.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
            reference=None
            for mode,path in zip(MODES,group):
                if mode!='native_v1' and path.is_file():continue
                world.clear_context();world.set_measure_mode(mode)
                with torch.inference_mode(),training_autocast('cuda'):
                    world.set_world(tensors);prediction=model(**kwargs)[0][:,:packed.shape[1]]
                    prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],8)
                    if mode=='native_v1':reference=prediction.detach().clone()
                    exact=bool(torch.equal(prediction,reference))
                    if mode=='measured_v1' and not exact:raise AssertionError('Instrumented v1 changed predictions; stop and inspect, do not interpret its measurements')
                    values=metrics(prediction,target,weights,iw,noise,clean,config,case['latent_seed']+entry['index'])
                    difference=float((prediction.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12))
                record=dict(id=row['id'],split=row['split'],mode=mode,scheduler_index=entry['index'],sigma=entry['sigma'],latent_seed=case['latent_seed'],
                    **values,native_prediction_exact=exact,relative_prediction_change=difference,measurements=world.measurements(),
                    peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
                if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,record);completed+=1
                world.clear_context();del prediction
                print(dict(stage='mechanism_forward',id=row['id'],sigma=entry['sigma'],mode=mode,completed=completed,total=len(expected),
                    elapsed_seconds=time.monotonic()-started),flush=True)
            del reference,noisy,packed,kwargs
        del posterior,clean,noise,target,condition,weights,iw,tensors
    summarize(output,protocol)


if __name__=='__main__':main()
