"""User-run baseline/correct/shuffled Oracle generation from the new checkpoint."""
import argparse
import shutil
import time
from pathlib import Path

from ..build_polished_world import read,atomic_json
from ..common import new_output
from ..scaled_world.binding import apply_scale
from .runtime import VERSION,WorldBranch
from ..object_world.data import load_pack,tensorize,target_image
from ..object_world.evaluation import select_images,resolve_panel,generation_budget,write_crop_review
from .baseline import check_baselines
from qwen_edit_pano.common import sample_seed as per_id_seed


def requested_images(records, split, limit, path):
    ids=read(path)['ids']
    by_id={r['id']:r for r in records if r['split']==split}
    if len(ids)!=limit or len(set(ids))!=len(ids) or not set(ids)<=set(by_id):
        raise ValueError('Requested IDs must be unique, in this split, and match --limit')
    return [by_id[sid] for sid in ids]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','val'),default='val')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--panel',type=Path,help='CPU-generated fixed image IDs, seeds, donors and GT object eligibility')
    p.add_argument('--baseline-reference',type=Path,help='Reuse matching completed baseline images, never World results')
    p.add_argument('--preflight',action='store_true',help='Print exact image budget without loading a GPU model')
    p.add_argument('--steps',type=int,default=28)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--seed-mode',choices=('sequential','per-id'),default='sequential',
                   help='Use per-id for matched panorama benchmarks; preserves legacy sequential default')
    p.add_argument('--ids-file',type=Path,help='JSON {ids:[...]}: exact image subset, without object-review filtering')
    p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--modes',nargs='+',choices=('baseline','correct','shuffled'),default=['baseline','correct','shuffled'])
    args=p.parse_args()
    if args.limit<1 or args.steps<1 or args.cfg<1:p.error('Invalid generation settings')
    if args.ids_file and args.panel:p.error('Use ids-file or panel, not both')
    if args.panel and args.seed_mode!='sequential':p.error('Legacy panel stores sequential seeds')
    config=read(args.checkpoint/'config.json');complete=read(args.checkpoint/'COMPLETE.json')
    if config['version']!=VERSION or complete['version']!=VERSION:
        raise ValueError('Use the ID/region/binding World checkpoint')
    plan,records=load_pack(config['pack'])
    if config['pack_plan']!=plan:raise ValueError('Training pack configuration differs')
    candidates=[row for row in records if row['split']==args.split]
    panel=read(args.panel) if args.panel else None
    selected=(requested_images(records,args.split,args.limit,args.ids_file) if args.ids_file else
              resolve_panel(panel,records,args.split,args.limit,args.seed) if panel else
              select_images(records,args.split,args.limit))
    reuse=check_baselines(args.baseline_reference,config,selected,panel,args.steps,args.cfg,args.seed,args.split) if args.baseline_reference else None
    if reuse and 'baseline' not in args.modes:raise ValueError('Requested reuse but baseline mode is absent')
    generated_modes=[m for m in args.modes if not (m=='baseline' and reuse)]
    budget=generation_budget(len(selected),generated_modes,args.steps)
    budget['reused_baselines']=len(selected) if reuse else 0
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
        split=args.split,ids=[r['id'] for r in selected],seed=args.seed,seed_mode=args.seed_mode,steps=args.steps,cfg=args.cfg,
        modes=args.modes,oracle=True,world_cfg='same World in positive/negative branches',
        budget=budget,panel=str(args.panel.resolve()) if args.panel else None,
        note='Explicit GT-state Oracle images; no GT RGB or instance masks enter generation; no object-control certification'))
    if panel:atomic_json(output/'diagnostic_panel.json',panel)
    if reuse:atomic_json(output/'BASELINE_REUSE.json',reuse)
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
    world=WorldBranch(hidden,config['blocks'],config['width']).cuda().eval().requires_grad_(False)
    world.load_state_dict(state,strict=True);world.attach(pipe.transformer);pipe.enable_model_cpu_offload()
    generation_started=time.monotonic();generated_count=0
    for index,row in enumerate(selected):
        directory=output/row['id'];directory.mkdir()
        local=local_image(row);local.save(directory/'local.png')
        entry=panel['samples'][index] if panel else None
        other=(next(r for r in candidates if r['id']==entry['shuffled_source']['id']) if entry else
               next((r for r in candidates if r['scene_id']!=row['scene_id']),None))
        sample_seed=(entry['seed'] if entry else per_id_seed(args.seed,row['id'],'per-id')
                     if args.seed_mode=='per-id' else args.seed+index)
        if 'shuffled' in args.modes and other is None:raise ValueError('Shuffle requires another building')
        for mode in args.modes:
            if mode=='baseline' and reuse:
                source_dir=args.baseline_reference/row['id']
                shutil.copy2(source_dir/'baseline.png',directory/'baseline.png')
                atomic_json(directory/'baseline.json',dict(read(source_dir/'baseline.json'),reused_from=str(source_dir.resolve())))
                continue
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
