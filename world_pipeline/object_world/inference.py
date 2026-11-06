"""User-run baseline/correct/shuffled Oracle generation from the new checkpoint."""
import argparse
import time
from pathlib import Path

from ..build_polished_world import read,atomic_json
from ..common import new_output
from ..scaled_world.binding import apply_scale
from .runtime import VERSION,WorldBranch
from .data import load_pack,tensorize,target_image
from .evaluation import select_images,resolve_panel,generation_budget,write_crop_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','val'),default='val')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--panel',type=Path,help='CPU-generated fixed image IDs, seeds, donors and GT object eligibility')
    p.add_argument('--preflight',action='store_true',help='Print exact image budget without loading a GPU model')
    p.add_argument('--steps',type=int,default=28)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--modes',nargs='+',choices=('baseline','correct','shuffled'),default=['baseline','correct','shuffled'])
    args=p.parse_args()
    if args.limit<1 or args.steps<1 or args.cfg<1:p.error('Invalid generation settings')
    config=read(args.checkpoint/'config.json');complete=read(args.checkpoint/'COMPLETE.json')
    if config['version']!=VERSION or complete['version']!=VERSION:
        raise ValueError('Use the new semantic/region World checkpoint')
    plan,records=load_pack(config['pack'])
    if config['pack_plan']!=plan:raise ValueError('Training pack configuration differs')
    candidates=[row for row in records if row['split']==args.split]
    panel=read(args.panel) if args.panel else None
    selected=(resolve_panel(panel,records,args.split,args.limit,args.seed) if panel else
              select_images(records,args.split,args.limit))
    budget=generation_budget(len(selected),args.modes,args.steps)
    print(dict(stage='generation_budget',**budget,ids=[r['id'] for r in selected]),flush=True)
    if args.preflight:return
    import torch
    from diffusers import AutoencoderKLQwenImage
    from safetensors.torch import load_file
    from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
    from qwen_edit_pano.paired_data import local_image
    from qwen_edit_pano.inference_state import enforce_inference_eval
    if not torch.cuda.is_available():raise ValueError('Allocate one BF16 CUDA GPU')
    output=new_output(args.output)
    atomic_json(output/'run_config.json',dict(checkpoint=str(args.checkpoint.resolve()),config=config,
        split=args.split,ids=[r['id'] for r in selected],seed=args.seed,steps=args.steps,cfg=args.cfg,
        modes=args.modes,oracle=True,world_cfg='same World in positive/negative branches',
        budget=budget,panel=str(args.panel.resolve()) if args.panel else None,
        note='Diagnostic generation; two-step smoke does not establish quality'))
    if panel:atomic_json(output/'diagnostic_panel.json',panel)
    snapshot=plan['base_snapshot'];base=Path(plan['base_checkpoint'])
    vae=AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe=PairedPanoPipeline.from_pretrained(snapshot,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.vae.to(dtype=torch.float32)
    pipe.load_lora_weights(str(base),weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
    apply_scale(pipe.transformer,config['pano_lora_scale'],'pano');pipe.enable_pano(plan['padding_columns'])
    for module in (pipe.transformer,pipe.vae,pipe.text_encoder):
        enforce_inference_eval(module);module.requires_grad_(False)
    hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
    state=load_file(str(args.checkpoint/'world.safetensors'))
    world=WorldBranch(hidden,state['encoder.class_features'],config['blocks'],config['width']).cuda().eval().requires_grad_(False)
    world.load_state_dict(state,strict=True);world.attach(pipe.transformer);pipe.enable_model_cpu_offload()
    generation_started=time.monotonic();generated_count=0
    for index,row in enumerate(selected):
        directory=output/row['id'];directory.mkdir()
        local=local_image(row);local.save(directory/'local.png')
        entry=panel['samples'][index] if panel else None
        other=(next(r for r in candidates if r['id']==entry['shuffled_source']['id']) if entry else
               next((r for r in candidates if r['scene_id']!=row['scene_id']),None))
        sample_seed=entry['seed'] if entry else args.seed+index
        if 'shuffled' in args.modes and other is None:raise ValueError('Shuffle requires another building')
        for mode in args.modes:
            source=other if mode=='shuffled' else row
            image_started=time.monotonic()
            with torch.inference_mode():
                world.set_world(None if mode=='baseline' else tuple(t.cuda() for t in tensorize(source,plan['max_objects'])))
                image=pipe(image=local,prompt=plan['prompt'],negative_prompt=' ',height=plan['height'],width=2*plan['height'],
                    num_inference_steps=args.steps,true_cfg_scale=args.cfg,
                    generator=torch.Generator(device='cpu').manual_seed(sample_seed)).images[0]
            image.save(directory/(mode+'.png'))
            generated_count+=1
            elapsed=time.monotonic()-generation_started
            atomic_json(directory/(mode+'.json'),dict(seed=sample_seed,world_source=source['id'] if mode!='baseline' else None,
                        diagnostics=world.diagnostics() if mode!='baseline' else {},oracle=mode!='baseline',
                        generation_seconds=time.monotonic()-image_started))
            print(dict(stage='generation',id=row['id'],mode=mode,completed=generated_count,
                total=budget['generated_panoramas'],elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/generated_count*(budget['generated_panoramas']-generated_count)),flush=True)
            world.set_world(None)
        target_image(row).save(directory/'gt_erp.png')
        if entry:write_crop_review(directory,entry,args.modes)
    atomic_json(output/'COMPLETE.json',dict(samples=len(selected),modes=args.modes,budget=budget,
        elapsed_generation_seconds=time.monotonic()-generation_started))


if __name__=='__main__':main()
