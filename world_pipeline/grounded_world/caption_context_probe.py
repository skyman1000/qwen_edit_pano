"""User-run frozen-model 2x2 prompt probe; full-GT captions are oracle input.

Reuse old generic images only on their exact recorded GPU/runtime. Otherwise
generate the entire matrix on the newly allocated GPU. No training or downloads.
"""
import argparse
import fcntl
import html
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from ..build_polished_world import ROOT, read, atomic_json
from ..object_world.data import tensorize
from .old_panel_compare import prepare as historical_inputs, make_world, quality_views, OUTPUT as SOURCE
from .state_evaluate import checked_pending as common_pending, expected_record, jobs, summarize
from .cfg_policy import world_cfg_policy

OUTPUT = ROOT/'qwen_edit_pano/outputs/caption_context_C6000_val6_v1'
GENERIC = dict(pano_generic='pano_only', C6000_generic='C6000')
GROUPS = dict(pano_only=('pano_generic','pano_GTcaption'), C6000=('C6000_generic','C6000_GTcaption'))
CALLS = dict(policy='both',cond=28,uncond=28,disabled_uncond=0)
NOTE = ('ORACLE TEXT PROBE: GT | Panorama generic | Panorama + full-GT caption | '
        'C6000 generic | C6000 + full-GT caption. Full-panorama captions add target information; '
        'this is NOT the normal Local + state task score. Same Local, seed, LoRA0.5, CFG4, 28 steps. '
        'No training, object edits, GT RGB/mask input, or claim of semantic success.')


def caption_prompt(instruction, caption):
    if not isinstance(caption,str) or not caption.strip():
        raise ValueError('Expected a nonempty full-panorama caption')
    return instruction+'\nScene description: '+caption.strip()


def prepare():
    plan,byid,features,old_config,original = historical_inputs()
    if read(SOURCE/'PROTOCOL.json') != original or common_pending(SOURCE,original):
        raise ValueError('The reviewed historical 18-image comparison must be complete and unchanged')
    if read(SOURCE/'COMPARISON_COMPLETE.json')['images'] != 18:
        raise ValueError('Historical comparison incomplete')
    import pyarrow as pa
    cases=[]
    for case in original['cases']:
        ptr=case['source']['image']
        # Read caption cell only; never decode or feed target RGB to the model.
        with pa.memory_map(ptr['arrow_file'],'r') as source:
            for index,batch in enumerate(pa.ipc.open_stream(source)):
                if index==ptr['batch_index']:
                    caption=batch.column(batch.schema.get_field_index('caption'))[ptr['row_index']].as_py()
                    break
            else:raise ValueError('Caption Arrow batch missing')
        cases.append(dict(**case,full_GT_caption=caption,
            prompts=dict(generic=plan['prompt'],GTcaption=caption_prompt(plan['prompt'],caption))))
    protocol={k:v for k,v in original.items() if k not in ('cases','arms','images','comparison','checkpoint_configs')}
    protocol.update(experiment='full_GT_caption_context_probe_v1',cases=cases,
        arms=dict(pano_generic=None,pano_GTcaption=None,C6000_generic=original['arms']['C6000'],C6000_GTcaption=original['arms']['C6000']),
        images=24,checkpoint_configs=dict(C6000=original['checkpoint_configs']['C6000']),
        full_GT_caption_enters_forward=True,GT_RGB_and_masks_enter_forward=False,
        generic_reuse_source=str(SOURCE),source_protocol=original,
        comparison=NOTE,training=False,hashes_computed=False)
    return plan,byid,features,old_config,protocol


def record(protocol,case,arm):
    variant='generic' if arm in GENERIC else 'GTcaption'
    return dict(**expected_record(protocol,case,arm),prompt=case['prompts'][variant],
                prompt_variant=variant,full_GT_caption_enters_forward=variant=='GTcaption')


def pending(output,protocol):
    result=common_pending(output,protocol)
    for case,arm in jobs(protocol):
        path=output/case['id']/(arm+'.json')
        if path.is_file():
            actual=read(path)
            if any(actual.get(k)!=v for k,v in record(protocol,case,arm).items()):
                raise ValueError('Saved prompt/GT-caption provenance changed; use a new output')
    return result


def reusable_generic(protocol,case,arm,execution):
    """Caption arms are always new; generic images require identical execution."""
    if arm not in GENERIC:return None
    source=Path(protocol['generic_reuse_source'])
    if read(source/'EXECUTION.json')!=execution:return None
    original=protocol['source_protocol']
    old_case=next(c for c in original['cases'] if c['id']==case['id'])
    if any(old_case[k]!=case[k] for k in ('seed','source','split')):
        raise ValueError('Historical sample/seed changed')
    old_arm=GENERIC[arm];path=source/case['id']/(old_arm+'.png')
    actual=read(path.with_suffix('.json'))
    expected=expected_record(original,old_case,old_arm)
    if (any(actual.get(k)!=v for k,v in expected.items()) or actual.get('execution')!=execution
            or actual.get('branch_calls')!=CALLS or original['prompt']!=case['prompts']['generic']):
        raise ValueError('Historical generic generation provenance invalid')
    if not path.is_file():raise ValueError('Historical generic PNG missing')
    return path


def run_group(group,output):
    plan,byid,features,old_config,protocol=prepare()
    selected=[entry for entry in pending(output,protocol) if entry[1] in GROUPS[group]]
    if not selected:
        print(dict(group=group,already_complete=True),flush=True);return
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from ..cuda_check import require_cuda
        from diffusers import AutoencoderKLQwenImage
        from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
        from qwen_edit_pano.paired_data import local_image
        from qwen_edit_pano.inference_state import enforce_inference_eval
        from ..scaled_world.binding import apply_scale
        from ..render_memory import configure_pipeline_memory
        status=require_cuda('caption_context_'+group,output)
        vae=AutoencoderKLQwenImage.from_pretrained(plan['base_snapshot'],subfolder='vae',torch_dtype=torch.float32,local_files_only=True,low_cpu_mem_usage=True)
        pipe=PairedPanoPipeline.from_pretrained(plan['base_snapshot'],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
        pipe.vae.to(dtype=torch.float32)
        pipe.load_lora_weights(plan['base_checkpoint'],weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
        apply_scale(pipe.transformer,.5,'pano');pipe.enable_pano(plan['padding_columns'])
        for module in (pipe.transformer,pipe.vae,pipe.text_encoder):
            enforce_inference_eval(module);module.requires_grad_(False)
        hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
        world=make_world(group,hidden,features,plan,old_config)
        if world is not None:
            world=world.cuda().eval().requires_grad_(False)
            world.load_state_dict(load_file(str(Path(protocol['arms']['C6000_generic'])/'world.safetensors'),device='cpu'),strict=True)
            world.attach(pipe.transformer)
        memory=configure_pipeline_memory(pipe,output)
        props=torch.cuda.get_device_properties(0)
        execution=dict(device=status['device0'],device_uuid=str(getattr(props,'uuid','unavailable')),hostname=socket.gethostname(),
            torch=status['torch_version'],cuda=status['torch_cuda_build'],memory_mode=memory['mode'],
            allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
        if (output/'EXECUTION.json').is_file() and read(output/'EXECUTION.json')!=execution:
            raise ValueError('Partial probe requires the same GPU UUID/runtime; use --output with a new directory on another GPU')
        atomic_json(output/'EXECUTION.json',execution)
        print(dict(stage='caption_reuse_policy',reuse_generic=read(SOURCE/'EXECUTION.json')==execution,
            fresh_total_if_no_partial=12 if read(SOURCE/'EXECUTION.json')==execution else 24),flush=True)
        for index,(case,arm) in enumerate(selected,1):
            tick=time.monotonic();directory=output/case['id'];directory.mkdir(exist_ok=True)
            row=byid[case['id']];local=local_image(row);local.save(directory/'local.png')
            source=reusable_generic(protocol,case,arm,execution)
            if source is not None:
                tmp=directory/(arm+'.tmp.png');shutil.copyfile(source,tmp);tmp.replace(directory/(arm+'.png'))
                calls=CALLS.copy();origin=dict(kind='reused_identical_execution',source=str(source))
            else:
                state=None if world is None else tuple(t.cuda() for t in tensorize(row,plan['max_objects']))
                with torch.inference_mode():
                    if world is not None:world.clear_context();world.set_world(state)
                    with world_cfg_policy(pipe.transformer,world if world is not None else SimpleNamespace(current=None),'both') as calls:
                        image=pipe(image=local,prompt=record(protocol,case,arm)['prompt'],negative_prompt=' ',
                            height=plan['height'],width=2*plan['height'],num_inference_steps=28,true_cfg_scale=4.,
                            generator=torch.Generator(device='cpu').manual_seed(case['seed'])).images[0]
                if calls!=CALLS:raise ValueError('Unexpected CFG branch calls')
                tmp=directory/(arm+'.tmp.png');image.save(tmp);tmp.replace(directory/(arm+'.png'))
                if world is not None:world.clear_context()
                origin=dict(kind='generated',seconds=time.monotonic()-tick)
            atomic_json(directory/(arm+'.json'),dict(**record(protocol,case,arm),branch_calls=calls,execution=execution,origin=origin))
            print(dict(stage='caption_context',group=group,arm=arm,id=case['id'],completed=index,total_pending=len(selected),**origin),flush=True)


def summarize_probe(output,plan,byid,protocol):
    if pending(output,protocol):raise ValueError('Complete all four arms before summarizing')
    summarize(output,plan,byid,protocol)
    # The common renderer dynamically allocates all five columns, including GT.
    page=output/'comparison.html'
    banner='<p><strong>'+html.escape(NOTE)+'</strong></p>'
    prompts=''.join('<h3>'+html.escape(c['id'])+'</h3><p>'+html.escape(c['full_GT_caption'])+'</p>' for c in protocol['cases'])
    page.write_text(page.read_text().replace('<h1>',banner+'<h1>',1)+'<h2>Full-GT captions used only in oracle arms</h2>'+prompts)
    sources={c['id']:{name:output/c['id']/(name+'.png') for name in ('gt',*protocol['arms'])} for c in protocol['cases']}
    quality_views(sources,output/'quality_views',NOTE)
    records=[read(output/c['id']/(arm+'.json')) for c,arm in jobs(protocol)]
    generated=sum(r['origin']['kind']=='generated' for r in records)
    atomic_json(output/'CAPTION_PROBE_COMPLETE.json',dict(images=24,val_images=6,generated=generated,reused=24-generated,
        training_steps=0,full_GT_caption_is_extra_target_information=True,normal_task_success_certified=False,
        quality_gallery=str(output/'quality_views/quality.html')))
    print(output/'quality_views/quality.html',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--group',choices=tuple(GROUPS))
    a=parser.parse_args();torch.set_num_threads(4);output=a.output.resolve()
    if a.group:
        if a.preflight:parser.error('Use --preflight without --group')
        if not (output/'PROTOCOL.json').is_file():parser.error('Run the parent driver first')
        run_group(a.group,output);return
    plan,byid,features,old_config,protocol=prepare();todo=pending(output,protocol)
    print(dict(stage='caption_context_preflight',val_images=6,pending=len(todo),matrix_images=24,
        fresh_images_same_execution=12,fresh_images_other_execution=24,training=False,oracle_GT_text=True,hashes_computed=False),flush=True)
    if a.preflight:return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(output/'PROTOCOL.json',protocol)
        atomic_json(output/'PROMPTS.json',{c['id']:c['prompts'] for c in protocol['cases']})
        for group in GROUPS:
            subprocess.run([sys.executable,'-u','-m',__package__+'.caption_context_probe','--output',str(output),'--group',group],check=True)
        summarize_probe(output,plan,byid,protocol)


if __name__=='__main__':main()
