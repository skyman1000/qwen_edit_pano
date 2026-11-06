"""User-run natural-state evaluation: six val views, World-off/4000/6000.

No object interventions, training, downloads, or GT image/mask input to inference.
All arms run in one pipeline/runtime; partial output can resume on the same setup.
"""
import argparse
import fcntl
import html
import json
import math
from pathlib import Path
import statistics as st

import numpy as np
import torch
from safetensors.torch import load_file

from ..build_polished_world import read, atomic_json
from ..object_world.data import tensorize, identity, target_image
from ..spatial_world.controls import fully_outside
from . import full_data as fd
from .data import EVAL_PANEL
from .runtime import WorldBranch
from .cfg_policy import world_cfg_policy

VERSION = 'natural_state_val6_v1'
MAJOR_CLASSES = (3,4,5,7,9,10,11,13,14,15,18,21,22,25,26,31,33)


def validation_cases(panel, byid, train_ids):
    selected = [c for c in panel['cases'] if c['split'] == 'val']
    if len(selected) != 6 or len({c['id'] for c in selected}) != 6:
        raise ValueError('Use exactly the six existing development validation views')
    train_ids = set(train_ids)
    cases = []
    for i, case in enumerate(selected):
        row = byid[case['id']]
        if row['split'] != 'val' or row['id'] in train_ids or identity(row) != case['source']:
            raise ValueError('Validation source/split changed or entered training')
        cases.append(dict(id=row['id'],split='val',source=identity(row),seed=20261031+i))
    return cases


def jobs(protocol):
    return [(case,arm) for case in protocol['cases'] for arm in protocol['arms']]


def expected_record(protocol, case, arm):
    checkpoint = protocol['arms'][arm]
    return dict(version=VERSION,id=case['id'],split='val',arm=arm,
        mode='correct' if checkpoint is not None else 'world_off',
        seed=case['seed'],checkpoint=checkpoint)


def prepare(root, steps):
    if len(steps) != 2 or steps[0] <= 0 or steps[0] >= steps[1]:
        raise ValueError('Use two increasing positive Full-C step counts')
    plan, allrows, rows, features = fd.full_inputs()
    byid = {r['id']:r for r in allrows}
    config = fd.configuration(plan, rows)
    checkpoints = {}
    for step in steps:
        found = sorted((root/'train').glob(f'run-*/checkpoint-step{step:06d}/COMPLETE.json'))
        if len(found) != 1:
            raise ValueError(f'Need one unambiguous completed Full-C checkpoint at {step}, found {len(found)}')
        path = found[0].parent
        if read(found[0]) != dict(version=fd.VERSION,step=step,optimizer_step=600+step):
            raise ValueError('Wrong checkpoint completion marker')
        if read(path/'config.json') != config:
            raise ValueError('Checkpoint data/base/architecture/loss do not match current reviewed configuration')
        checkpoints[f'correct_step{step}'] = str(path)
    arch = read(Path(plan['base_snapshot'])/'transformer/config.json')
    world = WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],features,
        height=plan['height'],padding=plan['padding_columns'])
    for path in checkpoints.values():
        state = load_file(str(Path(path)/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Nonfinite checkpoint')
        world.load_state_dict(state,strict=True)
    del world,state
    cases = validation_cases(read(EVAL_PANEL),byid,config['train_ids'])
    arms = dict(pano_only=None,**checkpoints)
    protocol = dict(version=VERSION,full_steps=list(steps),cases=cases,arms=arms,
        checkpoint_configs={arm:fd.file_stat(Path(path)/'config.json') for arm,path in checkpoints.items()},
        height=plan['height'],width=2*plan['height'],padding_columns=plan['padding_columns'],
        prompt=plan['prompt'],negative_prompt=' ',generation_steps=28,text_cfg=4.,
        pano_lora_scale=.5,world_cfg_policy='both',base_snapshot=plan['base_snapshot'],
        base_checkpoint=plan['base_checkpoint'],images=18,optimizer_steps=0,
        GT_RGB_and_masks_enter_forward=False,
        note='Six fixed development val images, not six buildings. Correct states only. No object editing or final blind-test certification.')
    return plan,byid,features,protocol


def checked_pending(output, protocol):
    if (output/'PROTOCOL.json').is_file() and read(output/'PROTOCOL.json') != protocol:
        raise ValueError('Protocol changed; use a new output directory')
    pending = []
    for case,arm in jobs(protocol):
        prefix = output/case['id']/arm
        if prefix.with_suffix('.json').is_file():
            actual = read(prefix.with_suffix('.json'))
            if any(actual.get(k) != v for k,v in expected_record(protocol,case,arm).items()):
                raise ValueError('Saved image metadata does not match the requested case/checkpoint/seed')
            if actual.get('branch_calls') != dict(policy='both',cond=28,uncond=28,disabled_uncond=0):
                raise ValueError('Saved CFG branch record is invalid')
            if not (output/'EXECUTION.json').is_file() or actual.get('execution') != read(output/'EXECUTION.json'):
                raise ValueError('Saved image runtime differs from the output runtime')
        if not all(prefix.with_suffix(suffix).is_file() for suffix in ('.png','.json')):
            pending.append((case,arm))
    return pending


def select_objects(row):
    # Evaluation/display only. Selection depends on GT support, never generated RGB.
    _,geometry,_ = tensorize(row)
    outside = fully_outside(geometry)[0]
    with np.load(row['object_region_supervision']['path'],allow_pickle=False) as npz:
        instance,valid = npz['instance'],npz['object_loss_valid']
    eligible = []
    for slot,obj in enumerate(row['objects'],1):
        if not outside[slot] or obj['category_id'] not in MAJOR_CLASSES:continue
        mask = (instance == int(obj['instance_id'])) & valid
        if int(mask.sum()) >= 64:eligible.append((int(mask.sum()),slot,obj,mask))
    selected=[];seen=set()
    for item in sorted(eligible,key=lambda x:(-x[0],x[1])):
        if item[2]['category_id'] in seen:continue
        selected.append(item);seen.add(item[2]['category_id'])
        if len(selected)==3:break
    return selected


def summarize(output, plan, byid, protocol):
    from PIL import Image,ImageDraw
    from qwen_pano.geometry import perspective
    from qwen_pano.caupano.world_view import local_rays,sample_erp
    if checked_pending(output,protocol):raise ValueError('Finish all requested images before summarizing')
    sections=[];metrics=[];objects=[]
    for case in protocol['cases']:
        sid=case['id'];row=byid[sid];directory=output/sid
        # Read GT only AFTER all inference jobs are finished.
        target_image(row).save(directory/'gt.png')
        with Image.open(directory/'local.png') as im:
            local=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32)/255
        names=['gt',*protocol['arms']];images={}
        for name in names:
            with Image.open(directory/(name+'.png')) as im:images[name]=np.asarray(im.convert('RGB'),dtype=np.float32)/255
            if images[name].shape != (protocol['height'],protocol['width'],3):raise ValueError('Wrong generated image size')
        chosen=select_objects(row)
        sheet=Image.new('RGB',(384*len(names),220+412*len(chosen)),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for col,name in enumerate(names):
            draw.text((col*384+4,5),name,fill='black')
            sheet.paste(Image.fromarray((images[name]*255).round().astype(np.uint8)).resize((384,192)),(col*384,28))
            metrics.append(dict(id=sid,arm=name,metric='local_RGB_MAE',
                value=float(np.abs(sample_erp(images[name],local_rays(512))-local).mean())))
        for j,(area,slot,obj,mask) in enumerate(chosen):
            yy,xx=np.nonzero(mask);lat=np.pi/2-(yy+.5)/mask.shape[0]*np.pi;lon=(xx+.5)/mask.shape[1]*2*np.pi-np.pi
            yaw=math.degrees(math.atan2(float(np.mean(np.cos(lat)*np.sin(lon))),float(np.mean(np.cos(lat)*np.cos(lon)))))
            pitch=math.degrees(math.atan2(float(np.mean(np.sin(lat))),math.hypot(float(np.mean(np.cos(lat)*np.sin(lon))),float(np.mean(np.cos(lat)*np.cos(lon))))))
            label=plan['names'][str(obj['category_id'])]
            objects.append(dict(id=sid,slot=slot,instance_id=obj['instance_id'],category_id=obj['category_id'],name=label,
                visible_pixels=area,yaw=yaw,pitch=pitch))
            fullmask=np.repeat(np.repeat(mask,protocol['height']//mask.shape[0],0),protocol['width']//mask.shape[1],1)
            for col,name in enumerate(names):
                with torch.inference_mode():
                    crop=perspective(torch.from_numpy(images[name]).permute(2,0,1)[None],yaw,pitch,fov=70.,size=384)
                crop=(crop[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
                draw.text((col*384+4,220+j*412+5),label+' | '+name,fill='black')
                sheet.paste(Image.fromarray(crop),(col*384,220+j*412+28))
                if name!='gt':metrics.append(dict(id=sid,arm=name,metric='GT_instance_RGB_MAE',instance_id=obj['instance_id'],
                    category=label,value=float(np.abs(images[name]-images['gt'])[fullmask].mean())))
        sheet.save(directory/'comparison.jpg',quality=94)
        (directory/'input_objects.json').write_text(json.dumps(row['objects'],ensure_ascii=False,indent=2)+'\n')
        links=' | '.join('<a href="'+sid+'/'+name+'.png">'+html.escape(name)+'</a>' for name in names)
        sections.append('<h2>'+html.escape(sid)+'</h2><p>'+links+'</p><img width="220" src="'+sid+'/local.png" alt="Local input">'+
            '<img style="max-width:100%" src="'+sid+'/comparison.jpg">')
    summary={arm:st.mean(m['value'] for m in metrics if m['arm']==arm and m['metric']=='local_RGB_MAE') for arm in protocol['arms']}
    atomic_json(output/'METRICS.json',dict(rows=metrics,local_RGB_MAE_means=summary,selected_GT_objects=objects,
        note='RGB reconstruction auxiliaries, not object recognition/count/position accuracy. GT supports select display views only; no edited state. Six repeatedly used development val images, one seed each.'))
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><h1>Natural state generation: six validation images</h1>'+
        '<p>GT | '+html.escape(' | '.join(protocol['arms']))+'. Same Local, seed, LoRA0.5, CFG4 and runtime. No object edits. '+
        'Object rows: largest supported fully-offscreen distinct GT categories (up to three), fixed 70-degree views; not detection results. '+
        'Inspect class, major-object presence and rough layout; RGB errors are not semantic success.</p>'+''.join(sections))
    atomic_json(output/'COMPLETE.json',dict(version=VERSION,images=len(jobs(protocol)),val_images=len(protocol['cases']),optimizer_steps=0,
        object_success_certified=False,automatic_training=False))
    print(dict(stage='natural_state_evaluation_complete',images=len(jobs(protocol)),local_RGB_MAE_means=summary,output=str(output)),flush=True)


def run(root, steps, output, preflight=False):
    torch.set_num_threads(4)
    plan,byid,features,protocol=prepare(root,steps)
    pending=checked_pending(output,protocol)
    print(dict(stage='natural_state_preflight',val_images=6,images=18,pending=len(pending),steps=list(steps),
        object_edits=False,training=False,hashes_computed=False),flush=True)
    if preflight:return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=checked_pending(output,protocol)
        atomic_json(output/'PROTOCOL.json',protocol)
        if pending:
            from ..cuda_check import require_cuda
            status=require_cuda('natural_state_val6',output)
            from diffusers import AutoencoderKLQwenImage
            from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
            from qwen_edit_pano.paired_data import local_image
            from qwen_edit_pano.inference_state import enforce_inference_eval
            from ..scaled_world.binding import apply_scale
            from ..render_memory import configure_pipeline_memory
            vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True,low_cpu_mem_usage=True)
            pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
            pipe.vae.to(dtype=torch.float32)
            pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
            apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
            for module in (pipe.transformer,pipe.vae,pipe.text_encoder):enforce_inference_eval(module);module.requires_grad_(False)
            hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
            world=WorldBranch(hidden,features,height=plan['height'],padding=plan['padding_columns']).cuda().eval().requires_grad_(False)
            world.attach(pipe.transformer);memory=configure_pipeline_memory(pipe,output)
            signature=dict(device=status['device0'],torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
            if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=signature:
                raise ValueError('Resume on the same device type/runtime/placement, or use a new output')
            atomic_json(output/'EXECUTION.json',signature);loaded=None
            for i,(case,arm) in enumerate(pending,1):
                row=byid[case['id']];directory=output/case['id'];directory.mkdir(exist_ok=True)
                path=protocol['arms'][arm]
                if path is not None and path!=loaded:
                    world.load_state_dict(load_file(str(Path(path)/'world.safetensors'),device='cpu'),strict=True);loaded=path
                state=None if path is None else tuple(t.cuda() for t in tensorize(row,plan['max_objects']))
                local=local_image(row);local.save(directory/'local.png');world.clear_context()
                with torch.inference_mode():
                    world.set_world(state)
                    with world_cfg_policy(pipe.transformer,world,'both') as calls:
                        image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=protocol['height'],width=protocol['width'],
                            num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(case['seed'])).images[0]
                if calls!=dict(policy='both',cond=28,uncond=28,disabled_uncond=0):raise ValueError('Unexpected CFG branch execution')
                tmp=directory/(arm+'.tmp.png');image.save(tmp);tmp.replace(directory/(arm+'.png'))
                atomic_json(directory/(arm+'.json'),dict(**expected_record(protocol,case,arm),branch_calls=calls,execution=signature))
                world.clear_context();del state
                print(dict(stage='natural_state_render',completed=i,total_pending=len(pending),id=case['id'],arm=arm),flush=True)
        summarize(output,plan,byid,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=fd.OUTPUT)
    parser.add_argument('--steps',type=int,nargs=2,default=(4000,6000))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    output=args.output or root/'evaluation'/f'natural-state-{args.steps[0]}-{args.steps[1]}-v1'
    run(root,tuple(args.steps),output.resolve(),args.preflight)
