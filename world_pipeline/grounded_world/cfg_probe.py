"""User-run 12-image, same-device checkpoint/CFG comparison; no training."""
import argparse
import fcntl
import html
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import identity,tensorize,target_image
from ..cuda_check import require_cuda
from ..render_memory import configure_pipeline_memory
from . import full_data as fd
from .runtime import WorldBranch
from .continue_evaluate import single_variant
from .cfg_policy import world_cfg_policy

DEFAULT_OUTPUT=ROOT/'qwen_edit_pano/outputs/object_world_cfg_policy_probe_v1'
OLD_EVAL=fd.OUTPUT/'evaluation/new-step2000-run-1793169576359904021'
NEW_EVAL=fd.OUTPUT/'evaluation/new-step4000-run-1793292615851906372'
ARMS={'step2000_both':(2000,'both'),'step4000_both':(4000,'both'),
      'step4000_positive_only':(4000,'positive_only')}


def prepare():
    plan,allrows,rows,features=fd.full_inputs();byid={r['id']:r for r in allrows}
    old=read(OLD_EVAL/'render/PROTOCOL.json');new=read(NEW_EVAL/'render/PROTOCOL.json')
    for key in ('render_ids','single_object_render','cfg','steps','seed'):
        if old[key]!=new[key]:raise ValueError('Prior probe cases/settings changed')
    checkpoints={2000:Path(old['checkpoints']['Full-C']),4000:Path(new['checkpoints']['Full-C'])}
    configs={step:read(path/'config.json') for step,path in checkpoints.items()}
    if configs[2000]!=configs[4000]:raise ValueError('Checkpoints differ beyond continuation budget')
    if configs[4000]!=fd.configuration(plan,rows):raise ValueError('Current data/base do not match the reviewed checkpoints')
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json')
    world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],features)
    for step,path in checkpoints.items():
        meta=read(path/'COMPLETE.json')
        if meta['version']!=fd.VERSION or meta['step']!=step:raise ValueError('Wrong completed Full-C checkpoint')
        state=load_file(str(path/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
        world.load_state_dict(state,strict=True)
    del world,state
    # Fixed historical TV train and larger gym-equipment val, chosen without
    # ranking the new generated results. This is a causal probe, not a benchmark.
    ids=[new['render_ids'][0],new['render_ids'][3]]
    cases=[]
    for sid in ids:
        row=byid[sid];target=new['single_object_render']['targets'][sid];single_variant(row,target)
        cases.append(dict(id=sid,split=row['split'],source=identity(row),target=target,
            seed=new['seed']+new['render_ids'].index(sid)))
    protocol=dict(version='grounded_cfg_policy_probe_v1',checkpoints={str(s):str(p) for s,p in checkpoints.items()},
        configs={str(s):c for s,c in configs.items()},cases=cases,arms={a:dict(step=s,policy=p) for a,(s,p) in ARMS.items()},
        generation_steps=28,text_cfg=4.,pano_lora_scale=.5,height=plan['height'],width=2*plan['height'],
        prompt=plan['prompt'],negative_prompt=' ',base_snapshot=plan['base_snapshot'],base_checkpoint=plan['base_checkpoint'],
        images=12,optimizer_steps=0,GT_masks_enter_forward=False,
        note='Same-device checkpoint+CFG causal probe, one train/one val, original single-object edits. No claim of final object success or fully learned layout CFG.')
    return plan,byid,features,protocol,checkpoints


def run(output,preflight=False):
    torch.set_num_threads(4);plan,byid,features,protocol,checkpoints=prepare()
    pending=[(c,a,mode) for c in protocol['cases'] for a in ARMS for mode in ('correct','edited')
        if not all((output/c['id']/(a+'_'+mode+suffix)).is_file() for suffix in ('.png','.json'))]
    print(dict(stage='cfg_probe_preflight',images=12,pending=len(pending),training=False),flush=True)
    if preflight:return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'PROTOCOL.json').is_file() and read(output/'PROTOCOL.json')!=protocol:raise ValueError('Probe protocol changed')
        atomic_json(output/'PROTOCOL.json',protocol)
        if pending:
            status=require_cuda('cfg_policy_probe',output)
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
            world.attach(pipe.transformer);memory=configure_pipeline_memory(pipe,output)
            signature=dict(device=status['device0'],torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
            if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=signature:
                raise ValueError('Resume on the same device type/runtime/placement; use a new --output for a different environment')
            atomic_json(output/'EXECUTION.json',signature);loaded=None
            for i,(case,arm,mode) in enumerate(pending,1):
                sid=case['id'];row=byid[sid];step,policy=ARMS[arm];directory=output/sid;directory.mkdir(exist_ok=True)
                if loaded!=step:world.load_state_dict(load_file(str(checkpoints[step]/'world.safetensors')),strict=True);loaded=step
                state,edit=(tensorize(row),dict(operation='correct')) if mode=='correct' else single_variant(row,case['target'])
                state=tuple(t.cuda() for t in state);local=local_image(row);local.save(directory/'local.png');world.clear_context()
                with torch.inference_mode():
                    world.set_world(state)
                    with world_cfg_policy(pipe.transformer,world,policy) as calls:
                        image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                            num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(case['seed'])).images[0]
                if calls['cond']!=28 or calls['uncond']!=28 or calls['disabled_uncond']!=(28 if policy=='positive_only' else 0):
                    raise ValueError('CFG intervention did not run exactly once per branch/step')
                name=arm+'_'+mode;tmp=directory/(name+'.tmp.png');image.save(tmp);tmp.replace(directory/(name+'.png'))
                atomic_json(directory/(name+'.json'),dict(id=sid,arm=arm,mode=mode,checkpoint=str(checkpoints[step]),
                    seed=case['seed'],perturbation=edit,branch_calls=calls,execution=signature))
                world.clear_context();print(dict(stage='cfg_probe_render',completed=i,total_pending=len(pending),id=sid,arm=arm,mode=mode),flush=True)
        # GT RGB is read after generation solely to build the gallery.
        from qwen_pano.geometry import perspective
        from PIL import Image
        sections=[]
        for case in protocol['cases']:
            sid=case['id'];directory=output/sid;row=byid[sid];target_image(row).save(directory/'gt.png')
            obj=row['objects'][case['target']['slot']-1];x,y,z=obj['center_local_m']
            yaw=np.degrees(np.arctan2(x,z));pitch=np.degrees(np.arctan2(y,np.hypot(x,z)))
            names=['gt']+[a+'_'+m for a in ARMS for m in ('correct','edited')];cards=[]
            for name in names:
                with Image.open(directory/(name+'.png')) as im:array=np.asarray(im.convert('RGB'),dtype=np.float32)/255
                with torch.inference_mode():crop=perspective(torch.from_numpy(array).permute(2,0,1)[None],float(yaw),float(pitch),fov=70.,size=384)
                Image.fromarray((crop[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)).save(directory/(name+'_object.jpg'))
                cards.append('<figure><figcaption>'+html.escape(name)+'</figcaption><img width="400" src="'+sid+'/'+name+'.png"><img width="300" src="'+sid+'/'+name+'_object.jpg"></figure>')
            sections.append('<h2>'+html.escape(sid)+'</h2><div style="display:flex;flex-wrap:wrap">'+''.join(cards)+'</div>')
        (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><h1>Same-device checkpoint and World CFG probe</h1><p>Same positive prompt, text CFG4, seed and Local. Positive-only removes World from the negative branch; it is NOT an independently calibrated World guidance scale. No edited GT exists.</p>'+''.join(sections))
        atomic_json(output/'COMPLETE.json',dict(version=protocol['version'],images=12,optimizer_steps=0,object_control_certified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();run(a.output.resolve(),a.preflight)
