"""User-run comparison on six historical A10000 val images; no training.

Each arm uses a fresh process on the same allocated GPU, preserving the original
A and C implementations without stacking or replacing their attention hooks.
"""
import argparse
import fcntl
import html
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import tensorize,identity
from ..id_binding_world.runtime import WorldBranch as OldWorld,VERSION as OLD_VERSION
from .runtime import WorldBranch as CurrentWorld
from . import full_data as fd
from .state_evaluate import VERSION,checked_pending,expected_record,summarize
from .cfg_policy import world_cfg_policy

BENCH=ROOT/'qwen_edit_pano/outputs/native_pano_gt_A10000_val923_v1'
OLD=ROOT/'qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-1792820440973994739/checkpoint-step010000'
CURRENT=fd.OUTPUT/'train/run-1793411437364499805/checkpoint-step006000'
OUTPUT=ROOT/'qwen_edit_pano/outputs/old_A10000_vs_C6000_val6_v1'
HISTORICAL=ROOT/'research/audits/old_A10000_bridge_2026-10-31/historical'
CASES=(
    ('stair_hall','X7HyMhZNoso_87b4508bfdbf497299bd26eb4b23282a'),
    ('living_room','X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e'),
    ('spa_room','x8F5xyUWy9e_3b3649d2a7284321a0b9afdaa7b86df6'),
    ('indoor_bar','x8F5xyUWy9e_96a8a59e077f405c84ff77a07f6b108f'),
    ('bedroom','zsNo4HB9uLZ_aabf649f606a48a1b16d5bdcd5970602'),
    ('living_room','zsNo4HB9uLZ_f237319a500640d8ac172db225a3ce9c'),
)


def make_world(arm,hidden,features,plan,old_config):
    if arm=='pano_only':return None
    if arm=='A10000':
        return OldWorld(hidden,old_config['blocks'],old_config['width'],old_config['binding_block'])
    if arm=='C6000':return CurrentWorld(hidden,features,height=plan['height'],padding=plan['padding_columns'])
    raise ValueError('Unknown comparison arm')


def prepare():
    plan,allrows,train,features=fd.full_inputs();byid={r['id']:r for r in allrows}
    old_config=read(OLD/'config.json');current_config=read(CURRENT/'config.json');benchmark=read(BENCH/'plan.json')
    if (old_config['version']!=OLD_VERSION or old_config['group']!='A' or old_config['condition']!='full'
            or old_config['pack_plan']!=plan or old_config['pano_lora_scale']!=.5
            or old_config['binding_weight']!=0 or old_config['region_weight']!=.25):
        raise ValueError('Legacy A configuration differs from the reviewed A10000')
    if current_config!=fd.configuration(plan,train):raise ValueError('Current C6000 data/base/configuration changed')
    if read(OLD/'COMPLETE.json')['step']!=10000 or read(CURRENT/'COMPLETE.json')['step']!=6000:
        raise ValueError('Use completed A10000/C6000 checkpoints')
    if (benchmark['checkpoint']!=str(OLD) or benchmark['steps']!=28 or benchmark['cfg']!=4.
            or benchmark['lora_scale']!=.5 or benchmark['model']!=plan['base_snapshot']
            or benchmark['base_checkpoint']!=plan['base_checkpoint'] or benchmark['prompt']!=plan['prompt']):
        raise ValueError('Historical benchmark settings no longer match')
    sources=read(BENCH/'GENERATION_COMPLETE.json')['sources'];cases=[]
    for label,sid in CASES:
        row=byid[sid]
        if row['split']!='val' or sid in current_config['train_ids'] or sid not in benchmark['ids']:
            raise ValueError('Historical case is not in both validation sets')
        for arm in ('native','finetuned','gt_state'):
            source=sources[arm][sid]
            if not all(Path(source[k]).is_file() for k in ('image','local','gt')):
                raise ValueError('Historical image missing')
        original=read(Path(sources['gt_state'][sid]['image']).with_suffix('.json'))
        baseline=read(Path(sources['finetuned'][sid]['image']).parent/'sample.json')
        if (not original['oracle'] or original['world_source']!=sid or original['seed']!=baseline['seed']):
            raise ValueError('Historical GT state and baseline seeds/source disagree')
        cases.append(dict(id=sid,label=label,split='val',source=identity(row),seed=original['seed'],
            historical_sources={a:sources[a][sid] for a in ('native','finetuned','gt_state')}))
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json')
    hidden=arch['num_attention_heads']*arch['attention_head_dim']
    for arm,path in (('A10000',OLD),('C6000',CURRENT)):
        world=make_world(arm,hidden,features,plan,old_config)
        state=load_file(str(path/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(t).all() for t in state.values()):raise ValueError('Nonfinite checkpoint')
        world.load_state_dict(state,strict=True)
        del world,state
    protocol=dict(version=VERSION,experiment='historical_A10000_panel_vs_C6000_v1',cases=cases,
        arms=dict(pano_only=None,A10000=str(OLD),C6000=str(CURRENT)),height=plan['height'],width=2*plan['height'],
        padding_columns=plan['padding_columns'],prompt=plan['prompt'],negative_prompt=' ',generation_steps=28,text_cfg=4.,
        pano_lora_scale=.5,world_cfg_policy='both',base_snapshot=plan['base_snapshot'],base_checkpoint=plan['base_checkpoint'],
        checkpoint_configs=dict(A10000=fd.file_stat(OLD/'config.json'),C6000=fd.file_stat(CURRENT/'config.json')),
        images=18,optimizer_steps=0,GT_RGB_and_masks_enter_forward=False,
        selection='Six previously displayed historical validation images, fixed before new C generations; not a random/blind benchmark.',
        comparison='Practical legacy vs current models, different architecture and training budgets; not a causal step-count ablation.')
    return plan,byid,features,old_config,protocol


def quality_views(sources,output,note):
    """CPU, one ray reprojection from original PNGs into fixed 90-degree views."""
    from PIL import Image,ImageDraw
    from qwen_pano.geometry import cubemap
    output.mkdir(parents=True,exist_ok=True);sections=[]
    faces=('front','right','back_seam','left','up','down')
    for sid,inputs in sources.items():
        directory=output/sid;directory.mkdir(exist_ok=True)
        names=list(inputs);sheet=Image.new('RGB',(1536,284*len(names)),'#eeeeee');draw=ImageDraw.Draw(sheet)
        erp=Image.new('RGB',(1024,540*len(names)),'#eeeeee');ed=ImageDraw.Draw(erp)
        larger={index:Image.new('RGB',(512*len(names),540),'#eeeeee') for index in (0,2)}
        for ri,(name,path) in enumerate(inputs.items()):
            with Image.open(path) as im:
                image=im.convert('RGB');array=np.asarray(image,dtype=np.float32)/255
                erp.paste(image.resize((1024,512)),(0,ri*540+28));ed.text((5,ri*540+5),name,fill='black')
            with torch.inference_mode():cube=cubemap(torch.from_numpy(array).permute(2,0,1)[None],face_size=512)[0]
            for fi,face in enumerate(faces):
                pixels=(cube[fi].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
                view=Image.fromarray(pixels)
                sheet.paste(view.resize((256,256)),(fi*256,ri*284+28));draw.text((fi*256+4,ri*284+5),name+' / '+face,fill='black')
                if fi in larger:
                    larger[fi].paste(view,(ri*512,28));ImageDraw.Draw(larger[fi]).text((ri*512+4,5),name+' / '+face,fill='black')
        sheet.save(directory/'cube90.jpg',quality=97);erp.save(directory/'erp.jpg',quality=96)
        for index,image in larger.items():image.save(directory/(faces[index]+'512.png'))
        links=' | '.join('<a href="'+html.escape(os.path.relpath(path,directory),quote=True)+'">'+html.escape(name)+' original PNG</a>' for name,path in inputs.items())
        sections.append('<h2>'+html.escape(sid)+'</h2><p>'+links+'</p>'+
            '<p><a href="'+sid+'/erp.jpg">ERP</a> | <a href="'+sid+'/front512.png">front512</a> | <a href="'+sid+'/back_seam512.png">back512</a></p>'+
            '<img style="max-width:100%" src="'+sid+'/cube90.jpg">')
    (output/'quality.html').write_text('<!doctype html><meta charset="utf-8"><h1>Geometry and texture quality review</h1><p>'+html.escape(note)+
        '</p><p>Fixed 90-degree rectilinear cubemap faces, same rays for every model and GT. Face-edge stretching exists; compare centered structures. '+
        'Reprojection does not repair malformed furniture or smeared textures. PNG links retain source resolution.</p>'+''.join(sections))


def run_arm(arm,output):
    plan,byid,features,old_config,protocol=prepare()
    pending=[entry for entry in checked_pending(output,protocol) if entry[1]==arm]
    if not pending:print(dict(arm=arm,already_complete=True),flush=True);return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic_json(output/'PROTOCOL.json',protocol)
        pending=[entry for entry in checked_pending(output,protocol) if entry[1]==arm]
        from ..cuda_check import require_cuda
        status=require_cuda('old_panel_'+arm,output)
        from diffusers import AutoencoderKLQwenImage
        from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
        from qwen_edit_pano.paired_data import local_image
        from qwen_edit_pano.inference_state import enforce_inference_eval
        from ..scaled_world.binding import apply_scale
        from ..render_memory import configure_pipeline_memory
        vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True,low_cpu_mem_usage=True)
        pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
        pipe.vae.to(dtype=torch.float32);pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
        apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
        for module in (pipe.transformer,pipe.vae,pipe.text_encoder):enforce_inference_eval(module);module.requires_grad_(False)
        hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
        world=make_world(arm,hidden,features,plan,old_config)
        if world is not None:
            world=world.cuda().eval().requires_grad_(False)
            world.load_state_dict(load_file(str(Path(protocol['arms'][arm])/'world.safetensors'),device='cpu'),strict=True)
            world.attach(pipe.transformer)
        memory=configure_pipeline_memory(pipe,output)
        props=torch.cuda.get_device_properties(0)
        signature=dict(device=status['device0'],device_uuid=str(getattr(props,'uuid','unavailable')),hostname=socket.gethostname(),
            torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
            allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
        if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=signature:
            raise ValueError('Use the same allocated GPU/runtime/placement for all three arms, or a new output directory')
        atomic_json(output/'EXECUTION.json',signature)
        for i,(case,_) in enumerate(pending,1):
            row=byid[case['id']];directory=output/case['id'];directory.mkdir(exist_ok=True)
            local=local_image(row);local.save(directory/'local.png')
            state=None if world is None else tuple(t.cuda() for t in tensorize(row,plan['max_objects']))
            with torch.inference_mode():
                if world is not None:world.clear_context();world.set_world(state)
                with world_cfg_policy(pipe.transformer,world if world is not None else SimpleNamespace(current=None),'both') as calls:
                    image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                        num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(case['seed'])).images[0]
            if calls!=dict(policy='both',cond=28,uncond=28,disabled_uncond=0):raise ValueError('Unexpected CFG branches')
            tmp=directory/(arm+'.tmp.png');image.save(tmp);tmp.replace(directory/(arm+'.png'))
            atomic_json(directory/(arm+'.json'),dict(**expected_record(protocol,case,arm),branch_calls=calls,execution=signature))
            if world is not None:world.clear_context()
            print(dict(stage='old_panel_generate',arm=arm,completed=i,total_pending=len(pending),id=case['id']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--arm',choices=('pano_only','A10000','C6000'));p.add_argument('--preflight',action='store_true')
    p.add_argument('--historical-only',action='store_true',help='CPU export of existing old outputs; no model inference')
    a=p.parse_args();torch.set_num_threads(4)
    if a.arm and (a.preflight or a.historical_only):p.error('Arm selection is only for the GPU child process')
    if a.arm:run_arm(a.arm,a.output.resolve());return
    plan,byid,features,_,protocol=prepare();output=a.output.resolve()
    pending=checked_pending(output,protocol)
    print(dict(stage='old_panel_preflight',val_images=6,new_images=18,pending=len(pending),training=False,
        cases=[c['id'] for c in protocol['cases']],hashes_computed=False),flush=True)
    if a.preflight:return
    if a.historical_only:
        historical={case['id']:dict(gt=Path(case['historical_sources']['finetuned']['gt']),
            native=Path(case['historical_sources']['native']['image']),
            pano_only=Path(case['historical_sources']['finetuned']['image']),
            A10000=Path(case['historical_sources']['gt_state']['image'])) for case in protocol['cases']}
        quality_views(historical,HISTORICAL,'Historical saved images from the old 923-view benchmark; no current C6000 generation. Not a new matched-runtime comparison.')
        atomic_json(HISTORICAL/'SOURCES.json',dict(cases=protocol['cases'],new_GPU_images=0,training=False))
        print(HISTORICAL/'quality.html',flush=True);return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic_json(output/'PROTOCOL.json',protocol)
        for arm in protocol['arms']:
            subprocess.run([sys.executable,'-u','-m',__package__+'.old_panel_compare','--output',str(output),'--arm',arm],check=True)
        summarize(output,plan,byid,protocol)
        sources={c['id']:{name:output/c['id']/(name+'.png') for name in ('gt',*protocol['arms'])} for c in protocol['cases']}
        quality_views(sources,output/'quality_views','Fresh matched-device Panorama-only / old A10000 / current C6000 on the old six validation images. GT shown for reference; no edits or training.')
        atomic_json(output/'COMPARISON_COMPLETE.json',dict(images=18,val_images=6,training_steps=0,quality_views=True,
            old_steps=10000,current_full_steps=6000,matched_training_budget=False))
        print(output/'quality_views/quality.html',flush=True)


if __name__=='__main__':main()
