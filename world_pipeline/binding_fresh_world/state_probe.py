"""User-run controlled state interventions; two models x seven variants = 14.

Local/prompt/seed fixed. All target AABBs remain outside the Local frustum.
Only minimal encoder fields are serialized; no stale edited GT masks/captions.
"""
import argparse
import copy
import fcntl
import html
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch
from safetensors.torch import load_file
from ..build_polished_world import read,atomic_json,validate_objects
from ..object_world.data import identity,tensorize,target_image
from ..id_binding_world.runtime import WorldBranch
from ..id_binding_world.state_probe import changes
from ..id_binding_world.state_probe_review import draw_box
from ..spatial_world.controls import fully_outside
from ..spatial_world.projection import erp_support
from ..grounded_world.cfg_policy import world_cfg_policy
from .evaluate import prepare as natural_prepare
from . import data as d

VERSION='fresh_binding_offscreen_interventions_v1'
SID='X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e'
DEFAULT_OUTPUT=d.OUTPUT/'state_interventions/step10000'


def minimal(row):
    return dict(id=row['id'],frame=row['frame'],objects=copy.deepcopy(row['objects']))


def variants(row):
    original=minimal(row);objects={o['instance_id']:o for o in original['objects']}
    if objects['360']['category_id']!=22 or objects['328']['category_id']!=3:
        raise ValueError('Expected original television360 and dining chair328')
    result=[('correct','Original complete state',original),
        ('permuted','Reverse object list only; identical physical state',dict(original,objects=list(reversed(original['objects']))))]
    for name,description,iid,category,move in (
        ('tv_to_picture','TV360 -> picture6; geometry fixed','360',6,False),
        ('tv_to_mirror','TV360 -> mirror21; geometry fixed','360',21,False),
        ('move_tv','TV360 center x +1 metre along the camera x axis; size/y/z fixed','360',22,True),
        ('remove_tv','Omit TV360 token; NOT an explicit guaranteed deletion command','360',None,False),
        ('chair_to_plant','Dining chair328 -> plant14; geometry fixed','328',14,False)):
        edited=copy.deepcopy(original)
        obj=next(o for o in edited['objects'] if o['instance_id']==iid)
        if category is None:edited['objects']=[o for o in edited['objects'] if o['instance_id']!=iid]
        elif move:obj['center_local_m'][0]+=1.
        else:obj['category_id']=category
        result.append((name,description,edited))
    return result


def validate_variants(row,cases,names,max_objects):
    original={o['instance_id']:o for o in row['objects']}
    slots={o['instance_id']:i for i,o in enumerate(row['objects'],1)}
    outside=fully_outside(tensorize(row,max_objects)[1])[0]
    for iid in ('360','328'):
        if not outside[slots[iid]]:raise ValueError('Intervention conflicts with unchanged Local')
    for name,_,state in cases:
        validate_objects(state['objects'],names);tensorize(state,max_objects)
        current={o['instance_id']:o for o in state['objects']}
        if name in ('correct','permuted'):
            if current!=original:raise ValueError('Control changed physical state')
            continue
        delta=changes(row,state)
        if len(delta['modified'])+len(delta['removed'])!=1 or delta['added']:
            raise ValueError('Edit must change exactly one original object')
        for iid,obj in original.items():
            if not outside[slots[iid]] and current.get(iid)!=obj:
                raise ValueError('Local-intersecting objects must remain unchanged')
        geometry=tensorize(state,max_objects)[1]
        for i,obj in enumerate(state['objects'],1):
            if obj!=original[obj['instance_id']] and not fully_outside(geometry)[0,i]:
                raise ValueError('Moved/modified box must stay fully offscreen')


def prepare(root,step=10000):
    plan,byid,natural=natural_prepare(root,step);row=byid[SID]
    case=next(c for c in natural['cases'] if c['id']==SID)
    cases=variants(row);validate_variants(row,cases,plan['names'],plan['max_objects'])
    models={arm:path for arm,path in natural['arms'].items() if path is not None}
    states={name:state for name,_,state in cases}
    protocol=dict(version=VERSION,id=SID,split='val',source=identity(row),seed=case['seed'],
        step=step,models=models,checkpoint_files=natural['checkpoint_files'],height=plan['height'],width=2*plan['height'],
        prompt=plan['prompt'],negative_prompt=' ',pano_lora_scale=.5,text_cfg=4.,generation_steps=28,
        world_cfg_policy='both',base_checkpoint=plan['base_checkpoint'],base_snapshot=plan['base_snapshot'],
        variants=[dict(name=n,description=desc,object_count=len(state['objects']),changes=changes(row,state)) for n,desc,state in cases],
        images=len(cases)*len(models),local_intersecting_objects_unchanged=True,all_changed_boxes_fully_offscreen=True,
        GT_caption=False,GT_RGB_or_masks_enter_forward=False,optimizer_steps=0,hashes_computed=False,
        note='One repeatedly used validation scene. Single-attribute sensitivity diagnostic, not an editing benchmark or edited-GT reconstruction. Omission does not explicitly supervise absence; movement has no mesh collision certification.')
    return plan,row,states,protocol


def expected(protocol,model,name):
    return dict(version=VERSION,id=SID,model=model,variant=name,seed=protocol['seed'],checkpoint=protocol['models'][model])


def pending_jobs(output,protocol):
    if (output/'PROTOCOL.json').is_file() and read(output/'PROTOCOL.json')!=protocol:
        raise ValueError('Protocol changed; use a NEW output directory')
    jobs=[]
    for model in protocol['models']:
        for case in protocol['variants']:
            name=case['name'];directory=output/model
            if (directory/(name+'.json')).is_file():
                record=read(directory/(name+'.json'))
                if any(record.get(k)!=v for k,v in expected(protocol,model,name).items()):
                    raise ValueError('Existing output metadata changed')
                if (not (output/'EXECUTION.json').is_file() or record['execution']!=read(output/'EXECUTION.json')
                        or record['branch_calls']!=dict(policy='both',cond=28,uncond=28,disabled_uncond=0)):
                    raise ValueError('Existing output runtime/CFG changed')
            if not all((directory/(name+s)).is_file() for s in ('.json','.png')):jobs.append((model,name))
    return jobs


def view(obj,key):
    x,y,z=obj['center_local_m']
    return dict(key=key,yaw=math.degrees(math.atan2(x,z)),pitch=math.degrees(math.atan2(y,math.hypot(x,z))),fov=70.)


def support(objects):
    # Display/scoring proxy only: input AABB ray support, without visibility.
    temp=dict(objects=[dict(o,instance_id=str(i+1)) for i,o in enumerate(objects)])
    _,geometry,mask=tensorize(temp)
    weights,_=erp_support(geometry,mask,256,512,supersample=1)
    return weights.sum(-1).reshape(256,512).numpy()>0


def review(output,plan,row,states,protocol):
    from PIL import Image,ImageDraw
    from qwen_pano.geometry import perspective
    from qwen_pano.caupano.world_view import sample_erp,local_rays
    if pending_jobs(output,protocol):raise ValueError('Finish generation before review')
    original={o['instance_id']:o for o in row['objects']}
    moved=next(o for o in states['move_tv']['objects'] if o['instance_id']=='360')
    views=[dict(key='front',yaw=0.,pitch=0.,fov=90.),view(original['360'],'tv_original'),
        view(moved,'tv_moved'),view(original['328'],'chair')]
    metrics=[];sections=[];files=[]
    # This original GT is explicitly a reference, never the target for edits.
    target_image(row).save(output/'original_gt.png')
    for model in protocol['models']:
        with Image.open(output/model/'correct.png') as image:baseline=np.asarray(image.convert('RGB'),dtype=np.float32)/255
        for case in protocol['variants']:
            name=case['name'];filename=model+'/'+name;files.append(filename)
            with Image.open(output/(filename+'.png')) as image:rgb=np.asarray(image.convert('RGB'),dtype=np.float32)/255
            delta=np.abs(rgb-baseline).mean(-1)
            metric=dict(model=model,variant=name,full_RGB_change=float(delta.mean()),
                Local_RGB_change=float(np.abs(sample_erp(rgb,local_rays(512))-sample_erp(baseline,local_rays(512))).mean()))
            changed=case['changes'];boxes=[m['before'] for m in changed['modified']]+[m['after'] for m in changed['modified']]+changed['removed']
            if boxes:
                mask=support(boxes);mask=np.repeat(np.repeat(mask,plan['height']//256,0),2*plan['height']//512,1)
                metric.update(input_box_union_RGB_change=float(delta[mask].mean()),outside_box_RGB_change=float(delta[~mask].mean()),
                    note='Change magnitude only; projected input boxes ignore occlusion and do not identify generated objects.')
            metrics.append(metric)
            current={o['instance_id']:o for o in states[name]['objects']}
            erp=torch.from_numpy(rgb).permute(2,0,1)[None]
            for camera in views:
                with torch.inference_mode():cut=perspective(erp,camera['yaw'],camera['pitch'],fov=camera['fov'],size=384)
                crop=Image.fromarray((cut[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8))
                directory=output/'views'/camera['key']/model;directory.mkdir(parents=True,exist_ok=True)
                crop.save(directory/(name+'.jpg'),quality=95)
                iid=None if camera['key']=='front' else ('328' if camera['key']=='chair' else '360')
                if iid:
                    obj=current.get(iid);box=obj or original[iid]
                    label=('input '+plan['names'][str(obj['category_id'])]) if obj else 'omitted token: original input box'
                    draw_box(crop.copy(),box,camera,(255,160,20) if obj else (240,80,80),label).save(directory/(name+'_boxes.jpg'),quality=95)
                else:crop.save(directory/(name+'_boxes.jpg'),quality=95)
    for case in protocol['variants']:
        name=case['name']
        if name=='correct':continue
        blocks=[]
        relevant=['front','chair'] if name=='chair_to_plant' else ['front','tv_original','tv_moved']
        for key in relevant:
            cards=[];sheet=Image.new('RGB',(4*384,414),'white');draw=ImageDraw.Draw(sheet)
            for col,(model,variant) in enumerate((m,v) for m in protocol['models'] for v in ('correct',name)):
                path=f'views/{key}/{model}/{variant}'
                with Image.open(output/(path+'.jpg')) as crop:sheet.paste(crop,(col*384,30))
                draw.text((col*384+4,5),model+' / '+variant,fill='black')
                cards.append(f'<figure><img class="crop" src="{path}.jpg" data-plain="{path}.jpg" data-boxes="{path}_boxes.jpg">'
                    f'<figcaption>{html.escape(model+" / "+variant)}</figcaption></figure>')
            sheet.save(output/f'{name}_{key}.jpg',quality=95)
            blocks.append('<h3>'+key+'</h3><div class="cards">'+''.join(cards)+'</div>')
        sections.append('<section><h2>'+html.escape(name)+'</h2><p>'+html.escape(case['description'])+'</p>'+''.join(blocks)+'</section>')
    atomic_json(output/'CHANGE_METRICS.json',dict(rows=metrics,
        note='Differences from same-model correct generation; not edited-GT error, detection, edit success or semantic accuracy. Compare permutation control and inspect images.'))
    options=''.join('<option value="'+f+'">'+html.escape(f)+'</option>' for f in files)
    cameras='<option value="erp">ERP</option>'+''.join('<option value="'+v['key']+'">'+v['key']+'</option>' for v in views)
    page='''<!doctype html><meta charset="utf-8"><title>State control diagnostic</title>
<style>body{font-family:system-ui;margin:24px auto;max-width:1600px;padding:0 16px;background:#f4f5f8}section,.panel{background:white;padding:16px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0}img{max-width:100%}select{margin:8px;padding:6px}.wipe{position:relative}.wipe img{width:100%;display:block}.over{position:absolute;inset:0}input[type=range]{width:100%}figcaption{font-size:12px}</style>
<h1>固定Local与seed的状态干预</h1><p>两组均为全景LoRA0.5、CFG4、28步。GT仅是未编辑参考。橙/红框是输入AABB投影，不是检测结果。
列表置乱应保持物理状态；删除仅是移除token，不能视为已训练的显式删除指令。移动未做mesh碰撞认证。</p>
<div class="panel"><label>底图<select id="left">OPTIONS</select></label><label>叠图<select id="right">OPTIONS</select></label>
<label>视角<select id="view">CAMERAS</select></label><label><input type="checkbox" id="boxes">输入框</label>
<div class="wipe"><img id="under"><img id="over" class="over"></div><input type="range" id="range" value="50" min="0" max="100"></div>
<p><a href="PROTOCOL.json">运行协议/字段变动</a> | <a href="input_states.json">实际输入状态</a> | <a href="CHANGE_METRICS.json">变化幅度（非成功率）</a> | <a href="original_gt.png">未编辑GT</a></p>
SECTIONS<script>const el=id=>document.getElementById(id);function path(file){const v=el('view').value;if(v==='erp')return file+'.png';return 'views/'+v+'/'+file+(el('boxes').checked?'_boxes.jpg':'.jpg');}
function draw(){el('under').src=path(el('left').value);el('over').src=path(el('right').value);el('over').style.clipPath='inset(0 '+(100-el('range').value)+'% 0 0)';}
el('left').value=LEFT_DEFAULT;el('right').value=RIGHT_DEFAULT;for(const id of ['left','right','view','boxes','range'])el(id).addEventListener('input',draw);draw();</script>'''
    models=list(protocol['models'])
    page=page.replace('OPTIONS',options).replace('CAMERAS',cameras).replace('SECTIONS',''.join(sections))
    page=page.replace('LEFT_DEFAULT',json.dumps(models[-1]+'/correct')).replace('RIGHT_DEFAULT',json.dumps(models[-1]+'/tv_to_picture'))
    (output/'review.html').write_text(page)
    atomic_json(output/'COMPLETE.json',dict(version=VERSION,images=protocol['images'],step=protocol['step'],
        training=False,edit_success_certified=False))


def run(root,step,output,preflight=False,review_only=False):
    torch.set_num_threads(4);plan,row,states,protocol=prepare(root,step)
    pending=pending_jobs(output,protocol)
    print(dict(stage='state_intervention_preflight',images=protocol['images'],pending=len(pending),id=SID,
        models=protocol['models'],variants=list(states),all_changed_boxes_fully_offscreen=True,training=False),flush=True)
    if preflight:return
    if review_only and pending:raise ValueError('Review requires all 14 images')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pending=pending_jobs(output,protocol)
        if (output/'input_states.json').is_file() and read(output/'input_states.json')!=states:
            raise ValueError('Saved input states differ; use NEW output')
        atomic_json(output/'PROTOCOL.json',protocol);atomic_json(output/'input_states.json',states)
        if pending:
            from ..cuda_check import require_cuda
            from ..render_memory import configure_pipeline_memory
            from diffusers import AutoencoderKLQwenImage
            from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
            from qwen_edit_pano.paired_data import local_image
            from qwen_edit_pano.inference_state import enforce_inference_eval
            from ..scaled_world.binding import apply_scale
            status=require_cuda('fresh_binding_state_intervention',output)
            vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True,low_cpu_mem_usage=True)
            pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
            pipe.vae.to(dtype=torch.float32)
            pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
            scale=apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
            for module in (pipe.transformer,pipe.vae,pipe.text_encoder):enforce_inference_eval(module);module.requires_grad_(False)
            hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
            world=WorldBranch(hidden).cuda().eval().requires_grad_(False);world.attach(pipe.transformer)
            memory=configure_pipeline_memory(pipe,output);props=torch.cuda.get_device_properties(0)
            execution=dict(device=status['device0'],device_uuid=str(getattr(props,'uuid','unavailable')),hostname=socket.gethostname(),
                torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
            if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=execution:
                raise ValueError('Resume requires same GPU/runtime; use NEW --output on another GPU')
            atomic_json(output/'EXECUTION.json',execution);atomic_json(output/'pano_scale_check.json',scale)
            local=local_image(identity(row));local.save(output/'local.png');loaded=None
            for i,(model,name) in enumerate(pending,1):
                start=time.monotonic();checkpoint=protocol['models'][model]
                if loaded!=checkpoint:
                    world.load_state_dict(load_file(str(Path(checkpoint)/'world.safetensors'),device='cpu'),strict=True);loaded=checkpoint
                world.clear_context()
                with torch.inference_mode():
                    world.set_world(tuple(t.cuda() for t in tensorize(states[name],plan['max_objects'])))
                    with world_cfg_policy(pipe.transformer,world,'both') as calls:
                        image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                            num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(protocol['seed'])).images[0]
                if calls!=dict(policy='both',cond=28,uncond=28,disabled_uncond=0):raise ValueError('Unexpected CFG calls')
                directory=output/model;directory.mkdir(exist_ok=True)
                temp=directory/(name+'.tmp.png');image.save(temp);temp.replace(directory/(name+'.png'))
                atomic_json(directory/(name+'.json'),dict(**expected(protocol,model,name),execution=execution,branch_calls=calls,seconds=time.monotonic()-start))
                world.clear_context()
                print(dict(stage='state_intervention_render',completed=i,total_pending=len(pending),model=model,variant=name,seconds=time.monotonic()-start),flush=True)
        review(output,plan,row,states,protocol)
        print(output/'review.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=d.OUTPUT)
    p.add_argument('--step',type=int,default=10000);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--preflight',action='store_true');p.add_argument('--review-only',action='store_true');a=p.parse_args()
    run(a.root.resolve(),a.step,a.output.resolve(),a.preflight,a.review_only)
