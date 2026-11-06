"""Finite user-run Value ablation of existing checkpoints, with no training."""
import argparse
from contextlib import contextmanager
import fcntl
import math
from pathlib import Path
import statistics as st
import time
import numpy as np
import torch
from safetensors.torch import load_file
from ..build_polished_world import read,atomic_json
from ..object_world.data import tensorize,identity
from ..object_world.evaluation import object_report
from ..id_binding_world.condition_diagnostic_data import case_support
from ..spatial_world.controls import fully_outside
from ..spatial_world.diagnose import cache_paths,fixed_noise,metrics
from ..oracle_memory import training_autocast
from .data import DEFAULT_ROOT
from .evaluate import load_protocol,render_scenes,transformer
from .run import checkpoints
from .runtime import VERSION,WorldBranch

POLICY='mean_over_declared_category_V_at_same_geometry_v1'
MODES=('correct','repeat_correct','mean_class_value','world_off')


@torch.no_grad()
def mean_class_values(world,state,category_ids):
    """Remove class variation from V only; keep normal Q/K computation/routing.

    Average V after the actual encoder/world_norm/value projections. Object
    geometry stays fixed for every category; scene-mean V would remove geometry
    as well and is a different, confounded intervention.
    """
    cat,geometry,mask=state
    if not category_ids or len(set(category_ids))!=len(category_ids):raise ValueError('Need distinct declared categories')
    result={}
    with training_autocast(cat.device.type):
        for cid in category_ids:
            changed=cat.clone();changed[:,1:]=cid
            tokens,_=world.encoder(changed,geometry,mask)
            for key,branch in world.branches.items():
                v=branch.v(branch.world_norm(tokens.float()))
                if key not in result:result[key]=v.float().clone();result[key][:,0]=0
                else:result[key][:,1:]+=v[:,1:].float()
    return {key:(v/len(category_ids)).to(tokens.dtype) for key,v in result.items()}


@contextmanager
def replace_values(world,values):
    handles=[]
    try:
        for key,branch in world.branches.items():
            replacement=values[key]
            def hook(module,args,output,_replacement=replacement):
                if output.shape!=_replacement.shape or output.device!=_replacement.device:
                    raise ValueError('Value replacement shape/device differs from actual forward')
                return _replacement.to(output.dtype)
            handles.append(branch.v.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:handle.remove()


def cases_for_probe(plan,byid,protocol):
    cases=[]
    for i,sid in enumerate(render_scenes(protocol,byid)):
        row=byid[sid]
        with np.load(row['object_region_supervision']['path'],allow_pickle=False) as f:
            report=object_report(row,plan['names'],f['instance'],f['instance_valid'],f['object_loss_valid'])
        _,geometry,_=tensorize(row,plan['max_objects']);outside=fully_outside(geometry)[0]
        slot={o['instance_id']:j for j,o in enumerate(row['objects'],1)}
        visible=[o for o in report['objects'] if o['supported_pixels']>=64 and o['category_id'] not in (39,40) and outside[slot[o['instance_id']]]]
        tv=[o for o in visible if o['category_id']==22]
        targets=tv[:1] if tv else sorted((o for o in visible if o['eligible']),key=lambda o:-o['sphere_fraction'])[:4]
        if not targets:raise ValueError('No sufficiently supported offscreen targets in '+sid)
        cases.append(dict(id=sid,split=row['split'],source=identity(row),latent_seed=20261027+i,objects=targets))
    return cases


def report(output,expected,protocol):
    rows=[read(p) for p in expected];paired=[]
    seeds={c['id']:c['latent_seed'] for c in protocol['cases']}
    for file,row in zip(expected,rows):
        if (row['group'],row['id'],row['index'],row['mode'])!=(file.parents[2].name,file.parents[1].name,int(file.parent.name),file.stem):
            raise ValueError('Value probe record identity changed')
        if row['latent_seed']!=seeds[row['id']]:raise ValueError('Value probe seed changed')
        if not all(math.isfinite(v) for v in (row['total'],row['object_region'],row['relative_prediction_change'],*row['selected_object_errors'])):
            raise FloatingPointError('Nonfinite Value probe result')
    for g in protocol['checkpoints']:
        for case in protocol['cases']:
            for index in protocol['indices']:
                found={r['mode']:r for r in rows if (r['group'],r['id'],r['index'])==(g,case['id'],index)}
                a=found['correct'];b=found['mean_class_value']
                if found['repeat_correct']['relative_prediction_change']!=0:raise ValueError('Nonzero repeat floor')
                paired.append(dict(group=g,id=case['id'],split=case['split'],sigma=a['sigma'],
                    value_removed_prediction_change_pct=100*b['relative_prediction_change'],
                    world_removed_prediction_change_pct=100*found['world_off']['relative_prediction_change'],
                    selected_error_gap_pct=100*(st.mean(b['selected_object_errors'])/st.mean(a['selected_object_errors'])-1),
                    total_error_gap_pct=100*(b['total']/a['total']-1)))
    off={(r['id'],r['index']):r['total'] for r in rows if r['group']=='A' and r['mode']=='world_off'}
    if any(r['total']!=off[r['id'],r['index']] for r in rows if r['group']=='B' and r['mode']=='world_off'):
        raise ValueError('A/B frozen baseline differs')
    summary=[]
    for g in protocol['checkpoints']:
        for split in ('train','val'):
            for sigma in sorted({r['sigma'] for r in rows}):
                rr=[r for r in paired if (r['group'],r['split'],r['sigma'])==(g,split,sigma)]
                summary.append(dict(group=g,split=split,sigma=sigma,images=len(rr),
                    **{key:st.mean(r[key] for r in rr) for key in ('value_removed_prediction_change_pct','world_removed_prediction_change_pct','selected_error_gap_pct','total_error_gap_pct')}))
    atomic_json(output/'SUMMARY.json',dict(rows=len(rows),repeat_floor=0,paired=paired,summary=summary,
        interpretation='Positive error gaps favor actual V; Q/K still contain categories. This isolates Value use, not total semantic use or generation success.'))
    atomic_json(output/'COMPLETE.json',dict(version=VERSION,policy=POLICY,forward_records=len(rows),optimizer_steps=0,generated_panoramas=0))
    print(dict(stage='value_probe_complete',summary=str(output/'SUMMARY.json')),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);paths=checkpoints(args.root);plan,byid,features,base,unused=load_protocol(paths,args.root)
    tag=unused.name;output=args.root/'value_probe'/tag
    from diffusers import FlowMatchEulerDiscreteScheduler,AutoencoderKLQwenImage,QwenImageEditPlusPipeline as Pipeline
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(plan['base_snapshot'],subfolder='scheduler',local_files_only=True)
    indices=[int((scheduler.sigmas[:scheduler.config.num_train_timesteps]-s).abs().argmin()) for s in (.5,.8)]
    cases=cases_for_probe(plan,byid,base)
    sources={c['id']:cache_paths(byid[c['id']],unused/'diagnose') for c in cases}
    if any(not path.is_file() for item in sources.values() for path in item.values()):raise ValueError('Missing existing cache; no automatic caching/download')
    protocol=dict(version=VERSION,policy=POLICY,checkpoints=base['checkpoints'],configs=base['configs'],cases=cases,
        category_ids=sorted(map(int,plan['names'])),indices=indices,sigmas=[float(scheduler.sigmas[i]) for i in indices],modes=list(MODES),
        source_paths={sid:{k:str(v) for k,v in item.items()} for sid,item in sources.items()},
        input_objects={c['id']:byid[c['id']]['objects'] for c in cases},
        preserves='same geometry, normal Q/K computation and candidates, LoRA0.5, seed and noisy latent; later image Q can respond to the intervened earlier residual; GT masks used only for metric weights',
        limitations='Mean-category V is a diagnostic intervention, not a valid scene, training objective or full semantic ablation. No optimizer or free generation.')
    expected=[output/'records'/g/c['id']/str(i)/(m+'.json') for g in paths for c in cases for i in indices for m in MODES]
    print(dict(stage='value_probe_preflight',records=len(expected),scenes=len(cases),missing_caches=0,output=str(output)),flush=True)
    if args.preflight:return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'PROTOCOL.json').is_file() and read(output/'PROTOCOL.json')!=protocol:raise ValueError('Value probe protocol/source changed')
        atomic_json(output/'PROTOCOL.json',protocol)
        if all(p.is_file() for p in expected):report(output,expected,protocol);return
        if not torch.cuda.is_available():raise ValueError('User must allocate one CUDA GPU')
        device=torch.device('cuda:0');config=next(iter(base['configs'].values()));model=transformer(plan,config,device)
        hidden=model.config.num_attention_heads*model.config.attention_head_dim
        world=WorldBranch(hidden,features,height=plan['height'],padding=plan['padding_columns']).to(device).eval().requires_grad_(False);world.attach(model)
        vae=AutoencoderKLQwenImage.load_config(plan['base_snapshot'],subfolder='vae',local_files_only=True)
        mean=torch.tensor(vae['latents_mean']).reshape(1,-1,1,1,1);std=torch.tensor(vae['latents_std']).reshape(1,-1,1,1,1)
        completed=sum(p.is_file() for p in expected);started=time.monotonic()
        for group,path in paths.items():
            world.load_state_dict(load_file(str(path/'world.safetensors')),strict=True)
            for case in cases:
                row=byid[case['id']];state=tuple(t.to(device) for t in tensorize(row,plan['max_objects']))
                with torch.inference_mode():values=mean_class_values(world,state,protocol['category_ids'])
                posterior=load_file(str(sources[row['id']]['posterior_cache']),device='cpu');clean,noise=fixed_noise(posterior,mean,std,case['latent_seed'])
                clean=clean.to(device);noise=noise.to(device);target=noise-clean
                condition=load_file(str(sources[row['id']]['condition_cache']),device='cuda:0');weights=load_file(str(sources[row['id']]['region_cache']),device='cuda:0')['weights']
                iw,spec=case_support(row,case,plan['height'],plan['padding_columns']);iw=iw.to(device)
                for index in indices:
                    files=[output/'records'/group/row['id']/str(index)/(m+'.json') for m in MODES]
                    if all(file.is_file() for file in files):continue
                    sigma=scheduler.sigmas[index].to(device=device,dtype=clean.dtype);noisy=(1-sigma)*clean+sigma*noise
                    packed=Pipeline._pack_latents(noisy,1,clean.shape[1],clean.shape[3],clean.shape[4])
                    kwargs=dict(hidden_states=torch.cat((packed,condition['reference'][None]),1),timestep=scheduler.timesteps[index:index+1].to(device)/1000,
                        encoder_hidden_states=condition['embeddings'][None],encoder_hidden_states_mask=condition['mask'][None],
                        img_shapes=[[(1,clean.shape[3]//2,clean.shape[4]//2),tuple(condition['ref_shape'].tolist())]],return_dict=False)
                    reference=None
                    for mode,file in zip(MODES,files):
                        if mode!='correct' and file.is_file():continue
                        world.clear_context()
                        with torch.inference_mode(),training_autocast('cuda'):
                            world.set_world(None if mode=='world_off' else state)
                            if mode=='mean_class_value':
                                with replace_values(world,values):prediction=model(**kwargs)[0][:,:packed.shape[1]]
                            else:prediction=model(**kwargs)[0][:,:packed.shape[1]]
                            prediction=Pipeline._unpack_latents(prediction,plan['height'],2*plan['height'],8)
                            if mode=='correct':reference=prediction.detach().clone()
                            error=metrics(prediction,target,weights,iw,noise,clean,config,case['latent_seed']+index)
                            difference=float((prediction.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12))
                        record=dict(group=group,id=row['id'],split=row['split'],mode=mode,index=index,sigma=float(sigma),latent_seed=case['latent_seed'],
                            relative_prediction_change=difference,selected_objects=spec['objects'],**error)
                        if not file.exists():file.parent.mkdir(parents=True,exist_ok=True);atomic_json(file,record);completed+=1
                        print(dict(stage='value_probe',completed=completed,total=len(expected),elapsed_seconds=time.monotonic()-started),flush=True)
                        world.clear_context();del prediction
                    del reference,kwargs,packed,noisy
                del values,condition,weights,iw,state,posterior,clean,noise,target
        report(output,expected,protocol)


if __name__=='__main__':main()
