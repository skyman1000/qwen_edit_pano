"""User-run 18-image comparison: Panorama-only, A10000, fresh binding10000.

Same six historical validation views/seeds. No editing, training, or GT caption.
Masks only score read-only attention and select post-generation review crops.
"""
import argparse
import fcntl
import math
from pathlib import Path
import socket
import time

import torch
from safetensors.torch import load_file
from ..build_polished_world import read,atomic_json
from ..object_world.data import load_pack,DEFAULT_PACK,identity,tensorize
from ..id_binding_world.runtime import WorldBranch
from ..grounded_world.cfg_policy import world_cfg_policy
from ..grounded_world.state_evaluate import checked_pending,expected_record,summarize,VERSION as PANEL_VERSION
from ..grounded_world.old_panel_compare import quality_views,CASES
from ..binding_refine_world import data as old
from ..binding_refine_world.evaluate import SamplingProbe as BaseProbe,HISTORICAL
from ..binding_refine_world.supervision import sample_binding
from . import data as d


def prepare(root,step=10000):
    plan,train=d.inputs();_,allrows=load_pack(DEFAULT_PACK);byid={r['id']:r for r in allrows}
    previous=read(HISTORICAL);a_config=read(old.SOURCE/'config.json')
    if (previous['prompt']!=plan['prompt'] or previous['pano_lora_scale']!=.5
            or previous['base_checkpoint']!=plan['base_checkpoint']
            or previous['arms']['A10000']!=str(old.SOURCE)
            or a_config['pack_plan']!=plan or a_config['binding_weight']!=0
            or a_config['region_weight']!=.25 or a_config['pano_lora_scale']!=.5
            or a_config['blocks']!=[20,40,59] or a_config['width']!=256
            or read(old.SOURCE/'COMPLETE.json')['step']!=10000):
        raise ValueError('Original A / shared frozen base / historical panel changed')
    config=d.configuration(plan,False)
    matches=list((root/'train').glob(f'run-*/checkpoint-step{step:06d}/COMPLETE.json'))
    if len(matches)!=1:raise ValueError('Need one unambiguous complete fresh checkpoint at requested step')
    fresh=matches[0].parent
    if d.resume_step(fresh,config)!=step:raise ValueError('Wrong fresh checkpoint')
    ancestry=fresh.parent
    while not (ancestry/'initial_world.safetensors').is_file():
        resume=read(ancestry/'run_plan.json')['resume']
        if not resume:raise ValueError('Missing fresh initialization provenance')
        ancestry=Path(resume).parent
    actual=load_file(str(ancestry/'initial_world.safetensors'),device='cpu')
    source=load_file(str(d.INITIAL),device='cpu')
    if actual.keys()!=source.keys() or not all(torch.equal(v,source[k]) for k,v in actual.items()):
        raise ValueError('Fresh training did not use the original untrained A initialization')
    cases=[];trainids={r['id'] for r in train}
    for _,sid in CASES:
        c=next(c for c in previous['cases'] if c['id']==sid);row=byid[sid]
        if row['split']!='val' or sid in trainids or identity(row)!=c['source']:
            raise ValueError('Historical validation source/split changed')
        cases.append(dict(id=sid,split='val',source=identity(row),seed=c['seed']))
    arms=dict(pano_only=None,A10000=str(old.SOURCE),**{f'fresh_binding{step}':str(fresh)})
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json')
    world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'])
    for path in arms.values():
        if path is None:continue
        state=load_file(str(Path(path)/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite weights')
        world.load_state_dict(state,strict=True)
    protocol=dict(version=PANEL_VERSION,experiment='fresh_binding_natural_val6_v1',step=step,cases=cases,arms=arms,
        checkpoint_files={arm:{name:d.stat(Path(path)/name) for name in ('config.json','COMPLETE.json','world.safetensors')}
            for arm,path in arms.items() if path is not None},
        height=plan['height'],width=2*plan['height'],padding_columns=plan['padding_columns'],prompt=plan['prompt'],
        negative_prompt=' ',generation_steps=28,text_cfg=4.,pano_lora_scale=.5,world_cfg_policy='both',
        base_snapshot=plan['base_snapshot'],base_checkpoint=plan['base_checkpoint'],images=18,optimizer_steps=0,
        GT_RGB_and_masks_enter_forward=False,GT_caption_enters_forward=False,
        attention_probe='Actual block40 Q/K calls 0/1/28/29/54/55; shared evaluation queries, with global mass and instance/category token-top1 diagnostics',
        note='Six repeatedly used development val images, one fixed seed each. Natural correct states only; no edits or blind-test claim.')
    return plan,byid,protocol


def attention_stats(q,k,mask,labels,categories):
    """Sampled Q/K in [1, heads, sampled_queries, width]; CPU, read-only."""
    _,pairs,slots,pos,neg=labels
    if not len(slots):return dict(eligible_support=False)
    logits=(q[0].float()@k[0].float().transpose(-1,-2))/math.sqrt(q.shape[-1])
    probabilities=logits.masked_fill(~mask[0][None,None,:],-torch.inf).softmax(-1).mean(0)
    pair_prob=probabilities[pairs];target=pair_prob.gather(1,slots[:,None])[:,0]
    top=pair_prob.argmax(-1);categories=categories.reshape(-1)
    positive=pos>0;negative=neg>0
    def mean(value,weights,selected):
        return float((value[selected]*weights[selected]).sum()) if selected.any() else None
    return dict(eligible_support=True,
        positive_target_attention_mean=mean(target,pos,positive),
        positive_global_attention_mean=mean(pair_prob[:,0],pos,positive),
        positive_other_object_attention_mean=mean(pair_prob[:,1:].sum(-1)-target,pos,positive),
        positive_correct_instance_token_top1=mean((top==slots).float(),pos,positive),
        positive_correct_category_token_top1=mean(((top!=0)&(categories[top]==categories[slots])).float(),pos,positive),
        negative_target_attention_mean=mean(target,neg,negative),
        negative_global_attention_mean=mean(pair_prob[:,0],neg,negative))


class SamplingProbe(BaseProbe):
    def reset(self,labels,categories=None):
        super().reset(labels);self.categories=categories

    def finish(self,branch,args,output):
        if not self.active or not self.capture:return
        q,k=self.saved['q'],self.saved['k'];heads=branch.heads
        def split(x):return x.reshape(1,-1,heads,x.shape[-1]//heads).transpose(1,2)
        mask=args[2].detach().cpu()
        with torch.no_grad():extra=attention_stats(split(q),split(k),mask,self.labels,self.categories)
        super().finish(branch,args,output)
        self.records[-1].update(extra)


def run(root,step,output,preflight=False):
    torch.set_num_threads(4);plan,byid,protocol=prepare(root,step)
    print(dict(stage='fresh_binding_eval_preflight',val_images=6,images=18,pending=len(checked_pending(output,protocol)),
        training=False,checkpoints=protocol['arms'],hashes_computed=False),flush=True)
    if preflight:return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=checked_pending(output,protocol);atomic_json(output/'PROTOCOL.json',protocol)
        if pending:
            from ..cuda_check import require_cuda
            from ..render_memory import configure_pipeline_memory
            from diffusers import AutoencoderKLQwenImage
            from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
            from qwen_edit_pano.paired_data import local_image
            from qwen_edit_pano.inference_state import enforce_inference_eval
            from ..scaled_world.binding import apply_scale
            status=require_cuda('fresh_binding_val6',output)
            vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,
                local_files_only=True,low_cpu_mem_usage=True)
            pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,
                local_files_only=True,low_cpu_mem_usage=True)
            pipe.vae.to(dtype=torch.float32)
            pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',
                adapter_name='pano',local_files_only=True,use_safetensors=True)
            apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
            for module in (pipe.transformer,pipe.vae,pipe.text_encoder):
                enforce_inference_eval(module);module.requires_grad_(False)
            hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
            world=WorldBranch(hidden).cuda().eval().requires_grad_(False);world.attach(pipe.transformer)
            memory=configure_pipeline_memory(pipe,output);probe=SamplingProbe(world)
            props=torch.cuda.get_device_properties(0)
            execution=dict(device=status['device0'],device_uuid=str(getattr(props,'uuid','unavailable')),hostname=socket.gethostname(),
                torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
            if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=execution:
                raise ValueError('Partial output requires same GPU/runtime; use a new --output on another GPU')
            atomic_json(output/'EXECUTION.json',execution);loaded=None
            for number,(case,arm) in enumerate(pending,1):
                tick=time.monotonic();row=byid[case['id']];directory=output/case['id'];directory.mkdir(exist_ok=True)
                path=protocol['arms'][arm]
                if path is not None and path!=loaded:
                    world.load_state_dict(load_file(str(Path(path)/'world.safetensors'),device='cpu'),strict=True);loaded=path
                state=None if path is None else tuple(t.cuda() for t in tensorize(row,plan['max_objects']))
                labels,meta=sample_binding(row,plan['height'],plan['padding_columns'],case['seed'])
                probe.reset(labels if state is not None else None,categories=state[0].cpu() if state is not None else None)
                local=local_image(identity(row));local.save(directory/'local.png');world.clear_context()
                with torch.inference_mode():
                    world.set_world(state)
                    with world_cfg_policy(pipe.transformer,world,'both') as calls:
                        image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                            num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(case['seed'])).images[0]
                if calls!=dict(policy='both',cond=28,uncond=28,disabled_uncond=0):raise ValueError('Unexpected CFG calls')
                if path is not None and (probe.calls!=56 or len(probe.records)!=6):raise ValueError('Unexpected observer calls')
                temporary=directory/(arm+'.tmp.png');image.save(temporary);temporary.replace(directory/(arm+'.png'))
                atomic_json(directory/(arm+'.json'),dict(**expected_record(protocol,case,arm),execution=execution,branch_calls=calls,
                    sampling_attention=probe.records,probe_support=meta,seconds=time.monotonic()-tick,
                    note='Read-only proxy metrics are NOT visual object recognition. GT masks do not condition generation.'))
                world.clear_context();probe.reset(None);del state
                print(dict(stage='fresh_binding_eval_render',completed=number,total_pending=len(pending),id=case['id'],arm=arm,
                    seconds=time.monotonic()-tick),flush=True)
            probe.close()
        summarize(output,plan,byid,protocol)
        sources={c['id']:{arm:output/c['id']/(arm+'.png') for arm in ('gt',*protocol['arms'])} for c in protocol['cases']}
        quality_views(sources,output/'quality_views',
            'GT | Panorama-only | A10000 | fresh binding. Same Local, state, seed and fixed prompt. Inspect class, position, texture and geometry.')
        records={c['id']:{arm:read(output/c['id']/(arm+'.json'))['sampling_attention'] for arm,p in protocol['arms'].items()
            if p is not None} for c in protocol['cases']}
        atomic_json(output/'SAMPLING_ATTENTION.json',dict(cases=records,
            note='Actual generation Q/K, read-only sampled token probabilities. Token top1/category matches are not image detection accuracy.'))
        atomic_json(output/'FRESH_BINDING_EVALUATION_COMPLETE.json',dict(images=18,val_images=6,step=step,
            optimizer_steps=0,training=False,GT_caption=False,spatial_routing=False,quality_views=True,actual_sampling_attention=True))
        print(output/'quality_views/quality.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=d.OUTPUT)
    p.add_argument('--step',type=int,default=10000);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();root=a.root.resolve();output=a.output or root/'evaluation'/f'step{a.step}'
    run(root,a.step,output.resolve(),a.preflight)
