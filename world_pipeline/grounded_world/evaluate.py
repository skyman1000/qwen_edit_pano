"""Explicit user-run, finite pilot evaluation: fixed-noise records + 4 scenes."""
import argparse
import fcntl
import html
import json
import math
import random
import statistics as st
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ..build_polished_world import read,atomic_json
from ..object_world.data import load_pack,tensorize,target_image,identity
from ..spatial_world.diagnose import cache_paths,fixed_noise,metrics
from ..spatial_world.controls import condition_variant,fully_outside
from ..id_binding_world.condition_diagnostic_data import case_support
from ..oracle_memory import training_autocast
from .runtime import WorldBranch,VERSION
from .data import contract,EVAL_PANEL,DEFAULT_ROOT,inputs

MODES=('correct','repeat_correct','wrong_category','wrong_position','world_off')


def load_protocol(paths,root):
    configs={group:read(path/'config.json') for group,path in paths.items()};first=next(iter(configs.values()))
    plan,rows,_,features=inputs(first['pack'],first['panel']);byid={r['id']:r for r in rows}
    common=None;steps={}
    for group,path in paths.items():
        c=configs[group];meta=read(path/'COMPLETE.json')
        if c['version']!=VERSION or meta['version']!=VERSION or c['group']!=group or c['smoke'] or c['pack_contract']!=contract(plan):
            raise ValueError('Use complete new-version non-smoke checkpoints on the same pack')
        stripped={k:v for k,v in c.items() if k not in ('group','binding_weight')}
        if common is not None and stripped!=common:raise ValueError('A/B differ beyond the binding loss')
        common=stripped;steps[group]=meta['step']
    if len(set(steps.values()))!=1:raise ValueError('A/B must be compared at the same step')
    # Verify actual initial tensors, not merely a repeated seed in config.
    initial=None
    for path in paths.values():
        run=path.parent
        while not (run/'initial_world.safetensors').is_file():
            resume=read(run/'run_plan.json')['resume']
            if not resume:raise ValueError('Missing initialization ancestry')
            run=Path(resume).parent
        state=load_file(str(run/'initial_world.safetensors'),device='cpu')
        if initial is not None and (state.keys()!=initial.keys() or not all(torch.equal(v,initial[k]) for k,v in state.items())):
            raise ValueError('A/B were not initialized with identical weights/features')
        initial=state
    panel=read(EVAL_PANEL)
    for c in panel['cases']:
        if c['source']!=identity(byid[c['id']]):raise ValueError('Panel image source/alignment changed')
        if byid[c['id']]['split']!=c['split']:raise ValueError('Diagnostic split changed')
        if c['split']=='val' and any(c['id'] in cfg['train_ids'] for cfg in configs.values()):raise ValueError('Validation entered training')
        if c['split']=='train' and any(c['id'] not in cfg['train_ids'] for cfg in configs.values()):raise ValueError('Training anchor is not in the pilot')
    tag='-'.join(group+'-'+path.parent.name+'-'+path.name for group,path in paths.items())
    output=Path(root)/'evaluation'/tag
    protocol=dict(version=VERSION,checkpoints={g:str(p.resolve()) for g,p in paths.items()},steps=steps,
        configs=configs,panel=panel,sigmas=[.2,.5,.8],modes=list(MODES),same_actual_initial_tensors=True,
        input_objects={c['id']:byid[c['id']]['objects'] for c in panel['cases']},
        note='Development panel, fixed GT-noise velocity error is not generation success. GT image/instance masks never enter free generation.')
    return plan,byid,features,protocol,output


def transformer(plan,config,device):
    from diffusers import QwenImageTransformer2DModel
    from peft import LoraConfig,set_peft_model_state_dict
    from qwen_edit_pano.paired_circular import install_circular_padding
    from ..scaled_world.binding import apply_scale
    base=Path(plan['base_checkpoint']);pano=read(base/'pano_config.json');old=read(base/'training_config.json')
    model=QwenImageTransformer2DModel.from_pretrained(plan['base_snapshot'],subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
    model.add_adapter(LoraConfig(r=pano['rank'],lora_alpha=pano['lora_alpha'],lora_dropout=old['lora_dropout'],target_modules=pano['targets']))
    loaded=set_peft_model_state_dict(model,load_file(str(base/'adapter_resume.safetensors')))
    if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):raise ValueError('Incomplete Panorama load')
    apply_scale(model,config['pano_lora_scale'],'default');model.to(device);install_circular_padding(model,plan['padding_columns'])
    model.requires_grad_(False).eval();return model


def diagnostic_report(output,protocol):
    rows=[];paired=[]
    for group in protocol['checkpoints']:
        for case in protocol['panel']['cases']:
            for index in protocol['indices']:
                found={}
                for mode in MODES:
                    r=read(output/'records'/group/case['id']/str(index)/(mode+'.json'))
                    if (r['group'],r['id'],r['index'],r['mode'])!=(group,case['id'],index,mode):raise ValueError('Wrong record identity')
                    if r['latent_seed']!=case['latent_seed']:raise ValueError('Wrong fixed-noise seed')
                    if not all(math.isfinite(r[k]) for k in ('total','object_region','relative_prediction_change')):raise ValueError('Nonfinite record')
                    found[mode]=r;rows.append(r)
                a=found['correct']
                if found['repeat_correct']['relative_prediction_change']!=0:raise ValueError('Repeat-control prediction changed')
                for mode in ('wrong_category','wrong_position','world_off'):
                    b=found[mode];selected=[i for i,o in enumerate(b['selected_objects']) if o['slot'] in b['perturbation'].get('changed_slots',[])]
                    paired.append(dict(group=group,split=case['split'],id=case['id'],sigma=b['sigma'],mode=mode,
                        full_gap_pct=100*(b['total']/a['total']-1),region_gap_pct=100*(b['object_region']/a['object_region']-1),
                        changed_gap_pct=(100*(st.mean(b['selected_object_errors'][i] for i in selected)/st.mean(a['selected_object_errors'][i] for i in selected)-1) if selected else None)))
    if len(protocol['checkpoints'])==2:
        reference_group=next(iter(protocol['checkpoints']))
        off={(r['id'],r['index']):r for r in rows if r['group']==reference_group and r['mode']=='world_off'}
        for r in rows:
            if r['group']!=reference_group and r['mode']=='world_off':
                if r['total']!=off[r['id'],r['index']]['total']:raise ValueError('Frozen base differs across arms')
    summary=[]
    for group in protocol['checkpoints']:
        for split in ('train','val'):
            for sigma in sorted({r['sigma'] for r in rows}):
                for mode in ('wrong_category','wrong_position','world_off'):
                    values=[r for r in paired if (r['group'],r['split'],r['sigma'],r['mode'])==(group,split,sigma,mode)]
                    active=[r for r in values if r['changed_gap_pct'] is not None]
                    summary.append(dict(group=group,split=split,sigma=sigma,mode=mode,images=len(values),
                        full_gap_pct=st.mean(r['full_gap_pct'] for r in values),region_gap_pct=st.mean(r['region_gap_pct'] for r in values),
                        changed_gap_pct=st.mean(r['changed_gap_pct'] for r in active) if active else None,
                        changed_images=len(active),correct_better_changed=sum(r['changed_gap_pct']>0 for r in active)))
    atomic_json(output/'SUMMARY.json',dict(records=len(rows),repeat_floor=0,paired=summary,not_semantic_success=True))
    atomic_json(output/'RESULTS.json',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
    for ri,split in enumerate(('train','val')):
        for ci,sigma in enumerate(sorted({r['sigma'] for r in rows})):
            ax=axes[ri,ci]
            for j,group in enumerate(protocol['checkpoints']):
                values=[next(r['changed_gap_pct'] for r in summary if (r['group'],r['split'],r['sigma'],r['mode'])==(group,split,sigma,m)) for m in ('wrong_category','wrong_position')]
                ax.bar(np.arange(2)+(j-.5)*.32,values,width=.30,label=group)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks([0,1],['Category','Position']);ax.set_title(f'{split}, sigma={sigma:.3f}');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Changed selected instance regions: positive favors correct; not generation accuracy')
    fig.tight_layout();fig.savefig(output/'CONDITION_GAPS.png',dpi=150);plt.close(fig)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Matched A/B diagnostic</h1><p>Positive gaps favor correct. Fixed noisy GT latents, development images; not generation accuracy.</p><img style="max-width:100%" src="CONDITION_GAPS.png"><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>')
    atomic_json(output/'COMPLETE.json',dict(version=protocol['version'],records=len(rows),optimizer_steps=0))


def diagnose(plan,byid,features,protocol,output,paths,preflight):
    from diffusers import FlowMatchEulerDiscreteScheduler,AutoencoderKLQwenImage,QwenImageEditPlusPipeline as Pipeline
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    indices=[int((scheduler.sigmas[:scheduler.config.num_train_timesteps]-s).abs().argmin()) for s in protocol['sigmas']]
    protocol=dict(protocol,indices=indices)
    sources={c['id']:cache_paths(byid[c['id']],output) for c in protocol['panel']['cases']}
    if any(not p.is_file() for item in sources.values() for p in item.values()):raise ValueError('Missing existing diagnostic caches; no automatic download/build')
    expected=[output/'records'/g/c['id']/str(i)/(m+'.json') for g in paths for c in protocol['panel']['cases'] for i in indices for m in MODES]
    print(dict(stage='diagnostic_preflight',records=len(expected),missing_caches=0,train=4,val=6),flush=True)
    if preflight:return
    output.mkdir(parents=True,exist_ok=True)
    if (output/'PROTOCOL.json').is_file() and read(output/'PROTOCOL.json')!=protocol:raise ValueError('Diagnostic protocol changed')
    atomic_json(output/'PROTOCOL.json',protocol)
    if all(p.is_file() for p in expected):diagnostic_report(output,protocol);return
    from ..cuda_check import require_cuda
    require_cuda('fixed_noise_diagnose', output)
    device=torch.device('cuda:0');config=next(iter(protocol['configs'].values()));model=transformer(plan,config,device)
    hidden=model.config.num_attention_heads*model.config.attention_head_dim
    world=WorldBranch(hidden,features,height=plan['height'],padding=plan['padding_columns']).to(device).eval().requires_grad_(False);world.attach(model)
    vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
    mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
    completed=sum(p.is_file() for p in expected)
    for group,path in paths.items():
        world.load_state_dict(load_file(str(path/'world.safetensors')),strict=True)
        for case in protocol['panel']['cases']:
            row=byid[case['id']];posterior=load_file(str(sources[row['id']]['posterior_cache']),device='cpu')
            clean,noise=fixed_noise(posterior,mean,std,case['latent_seed']);clean=clean.to(device);noise=noise.to(device);target=noise-clean
            condition=load_file(str(sources[row['id']]['condition_cache']),device='cuda:0');weights=load_file(str(sources[row['id']]['region_cache']),device='cuda:0')['weights']
            iw,spec=case_support(row,case,plan['height'],plan['padding_columns']);iw=iw.to(device)
            state=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
            for index in indices:
                files=[output/'records'/group/row['id']/str(index)/(m+'.json') for m in MODES]
                if all(p.is_file() for p in files):continue
                sigma=scheduler.sigmas[index].to(device=device,dtype=clean.dtype);noisy=(1-sigma)*clean+sigma*noise
                packed=Pipeline._pack_latents(noisy,1,clean.shape[1],clean.shape[3],clean.shape[4])
                kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=scheduler.timesteps[index:index+1].to(device)/1000,
                    encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                    img_shapes=[[(1,clean.shape[3]//2,clean.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
                reference=None
                for mode,file in zip(MODES,files):
                    if mode!='correct' and file.is_file():continue
                    variant,perturbation=condition_variant(state,mode);world.clear_context()
                    with torch.inference_mode(),training_autocast('cuda'):
                        world.set_world(variant);prediction=model(**kwargs)[0][:,:packed.shape[1]]
                        prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],8)
                        if mode=='correct':reference=prediction.detach().clone()
                        values=metrics(prediction,target,weights,iw,noise,clean,config,case['latent_seed']+index)
                        difference=float((prediction.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12))
                    record=dict(group=group,id=row['id'],split=row['split'],index=index,sigma=float(sigma),mode=mode,latent_seed=case['latent_seed'],**values,
                        relative_prediction_change=difference,perturbation=perturbation,selected_objects=spec['objects'],world=world.diagnostics() if variant is not None else {})
                    if not file.exists():file.parent.mkdir(parents=True,exist_ok=True);atomic_json(file,record);completed+=1
                    world.clear_context();del prediction
                    print(dict(stage='diagnose',group=group,id=row['id'],mode=mode,completed=completed,total=len(expected)),flush=True)
                del reference,packed,kwargs,noisy
            del clean,noise,target,posterior,condition,weights,iw,state
    diagnostic_report(output,protocol)


def render_scenes(protocol,byid):
    train_ids=next(iter(protocol['configs'].values()))['train_ids']
    val_ids=[c['id'] for c in protocol['panel']['cases'] if c['split']=='val']
    def order(ids):
        def tv(sid):
            c,g,m=tensorize(byid[sid]);outside=fully_outside(g)[0]
            return bool(((c[0]==22)&outside).any())
        return sorted(ids,key=lambda sid:(not tv(sid),ids.index(sid)))[:2]
    selected=order(train_ids)+order(val_ids)
    if len(set(selected))!=4:raise ValueError('Need two training and two held-out rendering scenes')
    return selected


def render_variant(row,mode):
    """Prefer one fully-offscreen TV edit; never edit a visible Local object."""
    state=tensorize(row);category,geometry,mask=(t.clone() for t in state)
    outside=fully_outside(geometry)[0]
    slots=((category[0]==22)&outside).nonzero().flatten()
    if not len(slots):return condition_variant(state,mode)
    slot=int(slots[0]);original=geometry.clone()
    if mode=='wrong_category':category[0,slot]=21
    elif mode=='wrong_position':
        geometry[0,slot,0]+=1. if float(geometry[0,slot,0])>=0 else -1.
        if not bool(fully_outside(geometry)[0,slot]):
            geometry[0,slot,:3]=original[0,slot,:3];geometry[0,slot,2]-=1.
        if not bool(fully_outside(geometry)[0,slot]):raise ValueError('Cannot move target without conflicting with Local')
    else:raise ValueError('Unexpected rendering intervention')
    return (category,geometry,mask),dict(operation=mode,scope='one_input_AABB_fully_outside_Local_TV',
        changed_slots=[slot],instance_id=row['objects'][slot-1]['instance_id'],
        description='TV to mirror, geometry fixed' if mode=='wrong_category' else 'TV moved by 1m, category/size fixed',
        original_center=original[0,slot,:3].tolist(),new_center=geometry[0,slot,:3].tolist())


def render(plan,byid,features,protocol,output,paths,preflight):
    scenes=render_scenes(protocol,byid);edit_group='B' if 'B' in paths else next(iter(paths))
    cases={sid:[('baseline',None,None)]+[(f'correct_{g}',g,'correct') for g in paths]+[
        (f'{"wrong_category" if i%2==0 else "wrong_position"}_{edit_group}',edit_group,'wrong_category' if i%2==0 else 'wrong_position')]
        for i,sid in enumerate(scenes)}
    custom=protocol.get('single_object_render')
    if custom:
        scenes=custom['ids']
        cases={sid:[('baseline',None,None)]+[(f'correct_{g}',g,'correct') for g in paths]+
            [(custom['targets'][sid]['mode']+'_'+g,g,custom['targets'][sid]['mode']) for g in paths] for sid in scenes}
    settings=dict(protocol,render_ids=scenes,cases={sid:[list(x) for x in entries] for sid,entries in cases.items()},
        steps=28,cfg=4.,seed=20261026,render_intervention_policy='one fully-offscreen TV to mirror or moved 1m when present; otherwise offscreen-only permutation',
        selection='two train and two fixed-panel val; prioritize fully-offscreen TV by input metadata, never generation scores')
    if custom:
        settings['selection']=custom['selection'];settings['render_intervention_policy']='one explicitly recorded supported offscreen object in BOTH arms; no fallback permutation'
    print(dict(stage='render_preflight',scenes=scenes,train=2,val=2,images=sum(map(len,cases.values()))),flush=True)
    if preflight:return
    output.mkdir(parents=True,exist_ok=True)
    if (output/'PROTOCOL.json').is_file() and read(output/'PROTOCOL.json')!=settings:raise ValueError('Render protocol changed')
    atomic_json(output/'PROTOCOL.json',settings)
    pending=[(sid,name,group,mode) for sid,entries in cases.items() for name,group,mode in entries
        if not (output/sid/(name+'.png')).is_file() or not (output/sid/(name+'.json')).is_file()]
    if pending:
        from ..cuda_check import require_cuda
        require_cuda('panorama_render', output)
        from diffusers import AutoencoderKLQwenImage
        from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
        from qwen_edit_pano.paired_data import local_image
        from qwen_edit_pano.inference_state import enforce_inference_eval
        from ..scaled_world.binding import apply_scale
        vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True,low_cpu_mem_usage=True)
        pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
        pipe.vae.to(dtype=torch.float32);pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
        apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
        for module in (pipe.transformer,pipe.vae,pipe.text_encoder):enforce_inference_eval(module);module.requires_grad_(False)
        hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
        world=WorldBranch(hidden,features,height=plan['height'],padding=plan['padding_columns']).cuda().eval().requires_grad_(False)
        world.attach(pipe.transformer)
        from ..render_memory import configure_pipeline_memory,peak_process_cpu_gib
        memory=configure_pipeline_memory(pipe,output);loaded=None
        for i,(sid,name,group,mode) in enumerate(pending,1):
            row=byid[sid];directory=output/sid;directory.mkdir(parents=True,exist_ok=True)
            if mode in ('wrong_category','wrong_position'):
                if custom:
                    from .continue_evaluate import single_variant
                    variant,perturbation=single_variant(row,custom['targets'][sid])
                else:variant,perturbation=render_variant(row,mode)
            elif mode=='correct':variant,perturbation=tensorize(row,plan['max_objects']),dict(operation='correct')
            else:variant,perturbation=None,dict(operation='world_off')
            variant=None if variant is None else tuple(t.cuda() for t in variant)
            if group is not None and loaded!=group:
                world.load_state_dict(load_file(str(paths[group]/'world.safetensors')),strict=True);loaded=group
            seed=20261026+scenes.index(sid);world.clear_context();local=local_image(row);local.save(directory/'local.png')
            with torch.inference_mode():
                world.set_world(variant)
                image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                    num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
            tmp=directory/(name+'.tmp.png');image.save(tmp);tmp.replace(directory/(name+'.png'))
            atomic_json(directory/(name+'.json'),dict(id=sid,split=row['split'],group=group,mode=mode,seed=seed,perturbation=perturbation,
                checkpoint=str(paths[group]) if group else None,routing=world.routing_meta,world=world.diagnostics() if group else {},execution_memory=memory))
            world.clear_context();del variant
            print(dict(stage='render',completed=i,total_pending=len(pending),id=sid,case=name,peak_process_cpu_gib=peak_process_cpu_gib()),flush=True)
    # Read GT RGB only after generation, for evaluation gallery/crops.
    from PIL import Image
    from qwen_pano.geometry import perspective
    from qwen_pano.caupano.world_view import local_rays,sample_erp
    sections=[];local_metrics=[]
    for sid in scenes:
        row=byid[sid];directory=output/sid;target_image(row).save(directory/'gt.png')
        names=['gt',*[entry[0] for entry in cases[sid]]]
        _,geometry,_=tensorize(row,plan['max_objects']);outside=fully_outside(geometry)[0]
        objects=[o for slot,o in enumerate(row['objects'],1) if o['category_id']==22 and outside[slot]]
        if not objects:objects=[o for slot,o in enumerate(row['objects'],1) if outside[slot] and o['category_id'] not in (39,40)][:1]
        objects=objects[:1];cards=[]
        if custom:objects=[row['objects'][custom['targets'][sid]['slot']-1]]
        with Image.open(directory/'local.png') as im:local=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32)/255
        for name in names:
            with Image.open(directory/(name+'.png')) as img:array=np.asarray(img.convert('RGB'),dtype=np.float32)/255
            local_metrics.append(dict(id=sid,split=row['split'],case=name,local_rgb_mae=float(np.abs(sample_erp(array,local_rays(512))-local).mean())))
            if objects:
                x,y,z=objects[0]['center_local_m'];yaw=math.degrees(math.atan2(x,z));pitch=math.degrees(math.atan2(y,math.hypot(x,z)))
                with torch.inference_mode():crop=perspective(torch.from_numpy(array).permute(2,0,1)[None],yaw,pitch,fov=70.,size=384)
                Image.fromarray((crop[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)).save(directory/(name+'_object.jpg'))
            cards.append(f'<figure><figcaption>{html.escape(name)}</figcaption><a href="{sid}/{name}.png"><img width="350" src="{sid}/{name}.png"></a>'+
                (f'<img width="350" src="{sid}/{name}_object.jpg">' if objects else '')+'</figure>')
        sections.append(f'<h2>{html.escape(sid)} ({row["split"]})</h2><p>GT only matches unmodified correct state; edited states have no edited GT.</p><div style="display:flex;flex-wrap:wrap">'+''.join(cards)+'</div>')
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><h1>2 train + 2 val: semantic recovery and condition response</h1><p>Same Local, seed and frozen Panorama scale0.5. Object views use original input boxes, not detector results. This is development evaluation.</p>'+''.join(sections))
    atomic_json(output/'LOCAL_RGB_MAE.json',dict(rows=local_metrics,note='Fixed 90-degree RGB pixel MAE; auxiliary, not the formal LPIPS/SSIM/DISTS or object success.'))
    atomic_json(output/'COMPLETE.json',dict(version=protocol['version'],scenes=scenes,images=sum(map(len,cases.values())),optimizer_steps=0,
        note='Images generated; object success still requires inspection. No automatic longer training.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('diagnose','render'))
    p.add_argument('--checkpoint',nargs=2,action='append',required=True,metavar=('GROUP','PATH'))
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    paths={g:Path(path) for g,path in args.checkpoint}
    if len(paths)!=len(args.checkpoint) or not set(paths)<=set(('A','B')):p.error('Use distinct A/B checkpoints')
    torch.set_num_threads(4);plan,byid,features,protocol,output=load_protocol(paths,args.root)
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json')
    world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],features,height=plan['height'],padding=plan['padding_columns'])
    for path in paths.values():
        state=load_file(str(path/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite World checkpoint')
        world.load_state_dict(state,strict=True)
    del world,state
    fn=diagnose if args.stage=='diagnose' else render
    if args.preflight:fn(plan,byid,features,protocol,output/args.stage,paths,True)
    else:
        directory=output/args.stage;directory.mkdir(parents=True,exist_ok=True)
        with (directory/'RUN.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fn(plan,byid,features,protocol,directory,paths,False)


if __name__=='__main__':main()
