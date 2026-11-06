"""One user-allocated GPU: 6 fixed val images x off/A10000/control/binding.

No training, object editing, target caption or target masks in the forward.
Read-only attention probes use masks only to score actual sampling Q/K.
"""
import argparse
import fcntl
import math
from pathlib import Path
import socket
import time

import torch
from safetensors.torch import load_file

from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import load_pack,DEFAULT_PACK,identity,tensorize
from ..id_binding_world.runtime import WorldBranch
from ..grounded_world.cfg_policy import world_cfg_policy
from ..grounded_world.state_evaluate import checked_pending,expected_record,summarize,VERSION as PANEL_VERSION
from ..grounded_world.old_panel_compare import quality_views,CASES
from . import data as d
from .supervision import sample_binding,binding_terms

HISTORICAL=ROOT/'qwen_edit_pano/outputs/old_A10000_vs_C6000_val6_v1/PROTOCOL.json'


def prepare(root,step):
    plan,train,old=d.inputs();_,allrows=load_pack(DEFAULT_PACK);byid={r['id']:r for r in allrows}
    previous=read(HISTORICAL)
    if (previous['prompt']!=plan['prompt'] or previous['pano_lora_scale']!=.5
            or previous['base_checkpoint']!=plan['base_checkpoint'] or previous['arms']['A10000']!=str(d.SOURCE)):
        raise ValueError('Historical fixed panel/base changed')
    cases=[];trainids={r['id'] for r in train}
    for _,sid in CASES:
        source=next(c for c in previous['cases'] if c['id']==sid);row=byid[sid]
        if row['split']!='val' or sid in trainids or identity(row)!=source['source']:
            raise ValueError('Panel must remain six validation images, never training IDs')
        cases.append(dict(id=sid,split='val',source=identity(row),seed=source['seed']))
    paths={}
    for arm in ('control','binding'):
        matches=sorted((root/arm/'train').glob(f'run-*/checkpoint-step{step:06d}/COMPLETE.json'))
        if len(matches)!=1:raise ValueError(f'Need one complete {arm} checkpoint at {step}, found {len(matches)}')
        path=matches[0].parent;meta=read(matches[0])
        if (read(path/'config.json')!=d.configuration(plan,old,arm)
                or meta!=dict(version=d.VERSION,step=step,optimizer_step=step)):
            raise ValueError('New checkpoint configuration/completion mismatch')
        paths[arm+str(step)]=str(path)
    architecture=read(Path(plan['base_snapshot'])/'transformer/config.json')
    world=WorldBranch(architecture['num_attention_heads']*architecture['attention_head_dim'])
    arms=dict(pano_only=None,A10000=str(d.SOURCE),**paths)
    for path in arms.values():
        if path is None:continue
        state=load_file(str(Path(path)/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(t).all() for t in state.values()):raise ValueError('Nonfinite weights')
        world.load_state_dict(state,strict=True)
    protocol=dict(version=PANEL_VERSION,experiment='binding_refine_natural_val6_v1',step=step,cases=cases,arms=arms,
        checkpoint_files={arm:{n:d.stat(Path(p)/n) for n in ('config.json','COMPLETE.json','world.safetensors')}
            for arm,p in arms.items() if p is not None},
        height=plan['height'],width=2*plan['height'],padding_columns=plan['padding_columns'],prompt=plan['prompt'],
        negative_prompt=' ',generation_steps=28,text_cfg=4.,pano_lora_scale=.5,world_cfg_policy='both',
        base_snapshot=plan['base_snapshot'],base_checkpoint=plan['base_checkpoint'],images=24,optimizer_steps=0,
        GT_RGB_and_masks_enter_forward=False,GT_caption_enters_forward=False,
        attention_probe='Read-only actual block40 Q/K, calls 0/1/28/29/54/55; same mask-derived evaluation queries for all World checkpoints',
        note='Six historical development val images, one fixed seed each; natural correct GT state only. No editing, new training or blind-test claim.')
    return plan,byid,protocol


class SamplingProbe:
    """Observe actual Q/K; hooks return None and never touch attention output."""
    def __init__(self,world):
        self.branch=world.branches[str(world.binding_block)];self.active=False;self.records=[];self.calls=0;self.saved={}
        self.handles=[self.branch.q.register_forward_hook(self.q_hook),self.branch.k.register_forward_hook(self.k_hook),
            self.branch.register_forward_hook(self.finish)]

    def reset(self,labels):
        self.labels=labels;self.active=labels is not None;self.records=[];self.calls=0;self.saved={}

    def q_hook(self,module,args,output):
        if not self.active:return
        self.index=self.calls;self.calls+=1;self.capture=self.index in (0,1,28,29,54,55)
        if self.capture:self.saved['q']=output[:,self.labels[0].to(output.device)].detach().float().cpu()

    def k_hook(self,module,args,output):
        if self.active and self.capture:self.saved['k']=output.detach().float().cpu()

    def finish(self,branch,args,output):
        if not self.active or not self.capture:return
        q,k=self.saved['q'],self.saved['k'];heads=branch.heads
        def split(x):return x.reshape(1,-1,heads,x.shape[-1]//heads).transpose(1,2)
        labels=(torch.arange(q.shape[1]),*self.labels[1:]);mask=args[2].detach().cpu()
        with torch.no_grad():coverage,concentration=binding_terms(split(q),split(k),mask,labels)
        self.records.append(dict(call_index=self.index,coverage=float(coverage),concentration=float(concentration),
            positive_probability_geomean=math.exp(-float(coverage)),queries=q.shape[1],tokens=k.shape[1]))
        self.saved={}

    def close(self):
        for handle in self.handles:handle.remove()


def run(root,step,output,preflight=False):
    torch.set_num_threads(4);plan,byid,protocol=prepare(root,step);pending=checked_pending(output,protocol)
    print(dict(stage='binding_eval_preflight',val_images=6,images=24,pending=len(pending),training=False,
        checkpoints=protocol['arms'],hashes_computed=False),flush=True)
    if preflight:return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic_json(output/'PROTOCOL.json',protocol)
        if pending:
            from ..cuda_check import require_cuda
            from ..render_memory import configure_pipeline_memory
            from diffusers import AutoencoderKLQwenImage
            from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
            from qwen_edit_pano.paired_data import local_image
            from qwen_edit_pano.inference_state import enforce_inference_eval
            from ..scaled_world.binding import apply_scale
            status=require_cuda('binding_refine_val6',output)
            vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,
                local_files_only=True,low_cpu_mem_usage=True)
            pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,
                local_files_only=True,low_cpu_mem_usage=True)
            pipe.vae.to(dtype=torch.float32)
            pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',
                adapter_name='pano',local_files_only=True,use_safetensors=True)
            apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
            for module in (pipe.transformer,pipe.vae,pipe.text_encoder):enforce_inference_eval(module);module.requires_grad_(False)
            hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
            # Exact old A forward for all three identical-key architectures.
            world=WorldBranch(hidden).cuda().eval().requires_grad_(False);world.attach(pipe.transformer)
            memory=configure_pipeline_memory(pipe,output);probe=SamplingProbe(world)
            props=torch.cuda.get_device_properties(0)
            execution=dict(device=status['device0'],device_uuid=str(getattr(props,'uuid','unavailable')),hostname=socket.gethostname(),
                torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
            if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=execution:
                raise ValueError('Partial outputs require same GPU/runtime; use --output with a new directory on another GPU')
            atomic_json(output/'EXECUTION.json',execution);loaded=None
            for number,(case,arm) in enumerate(pending,1):
                tick=time.monotonic();row=byid[case['id']];directory=output/case['id'];directory.mkdir(exist_ok=True)
                path=protocol['arms'][arm]
                if path is not None and path!=loaded:
                    world.load_state_dict(load_file(str(Path(path)/'world.safetensors'),device='cpu'),strict=True);loaded=path
                state=None if path is None else tuple(t.cuda() for t in tensorize(row,plan['max_objects']))
                labels,meta=sample_binding(row,plan['height'],plan['padding_columns'],case['seed'])
                probe.reset(labels if path is not None else None)
                local=local_image(identity(row));local.save(directory/'local.png');world.clear_context()
                with torch.inference_mode():
                    world.set_world(state)
                    with world_cfg_policy(pipe.transformer,world,'both') as calls:
                        image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                            num_inference_steps=28,true_cfg_scale=4.,generator=torch.Generator(device='cpu').manual_seed(case['seed'])).images[0]
                if calls!=dict(policy='both',cond=28,uncond=28,disabled_uncond=0):raise ValueError('Unexpected CFG calls')
                if path is not None and (probe.calls!=56 or len(probe.records)!=6):raise ValueError('Unexpected actual attention probe calls')
                temporary=directory/(arm+'.tmp.png');image.save(temporary);temporary.replace(directory/(arm+'.png'))
                atomic_json(directory/(arm+'.json'),dict(**expected_record(protocol,case,arm),execution=execution,branch_calls=calls,
                    sampling_attention=probe.records,probe_support=meta,seconds=time.monotonic()-tick,
                    note='GT support used only by read-only observer; never enters forward conditioning'))
                world.clear_context();probe.reset(None)
                print(dict(stage='binding_eval_render',completed=number,total_pending=len(pending),id=case['id'],arm=arm,
                    seconds=time.monotonic()-tick),flush=True)
            probe.close()
        summarize(output,plan,byid,protocol)
        sources={c['id']:{name:output/c['id']/(name+'.png') for name in ('gt',*protocol['arms'])} for c in protocol['cases']}
        quality_views(sources,output/'quality_views',
            'GT | Panorama-only | A10000 | control continuation | binding continuation. Same Local, seed and fixed prompt; no edits/captions. Inspect objects AND perspective quality.')
        records={c['id']:{arm:read(output/c['id']/(arm+'.json'))['sampling_attention'] for arm,p in protocol['arms'].items()
            if p is not None} for c in protocol['cases']}
        atomic_json(output/'SAMPLING_ATTENTION.json',dict(cases=records,
            note='Proxy from actual generation Q/K; same evaluation queries. Not semantic recognition or control success.'))
        atomic_json(output/'BINDING_EVALUATION_COMPLETE.json',dict(images=24,val_images=6,step=step,
            optimizer_steps=0,training=False,GT_caption=False,spatial_routing=False,quality_views=True,actual_sampling_attention=True))
        print(output/'quality_views/quality.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=d.OUTPUT)
    p.add_argument('--step',type=int,default=10500);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();root=a.root.resolve();output=a.output or root/'evaluation'/f'step{a.step}'
    run(root,a.step,output.resolve(),a.preflight)
