"""Fixed indoor diagnostic inference; original training/inference code remains unchanged."""
import argparse
from .binding import apply_scale, verify_binding
from pathlib import Path
from ..common import read,sha,write,new_output
from ..oracle_data import load_bundle
from ..oracle_model import tensorize,FEATURE_VERSION
from .runtime import WorldBranch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('train','val'),default='val')
    p.add_argument('--limit',type=int,default=6)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--steps',type=int,default=28)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--modes',nargs='+',choices=('baseline','correct','shuffled'),default=['baseline','correct'])
    args=p.parse_args()
    if args.limit<1 or args.steps<1 or args.cfg<1 or len(set(args.modes))!=len(args.modes):
        p.error('Invalid limit/steps/cfg/modes')
    from qwen_edit_pano.common import require_versions,sample_seed
    require_versions()
    config=read(args.checkpoint/'config.json');complete=read(args.checkpoint/'COMPLETE.json')
    if config['feature_version']!=FEATURE_VERSION or sha(args.checkpoint/'world.safetensors')!=complete['world_sha256']:
        raise ValueError('World checkpoint incomplete/changed')
    if sha(Path(config['bundle'])/'BUNDLE.json')!=config['bundle_sha256']:
        raise ValueError('Bundle changed')
    bundle,entries=load_bundle(config['bundle'])
    verify_binding(config,bundle)
    candidates=[r for r in entries if r['split']==args.split]
    selection=read(args.selection)
    if selection['bundle_sha256']!=sha(Path(config['bundle'])/'BUNDLE.json') or selection['split']!=args.split:
        raise ValueError('Selection bundle/split mismatch')
    lookup={r['id']:r for r in candidates}
    ids=selection['ids']
    if len(ids)!=len(set(ids)) or set(ids)-set(lookup):raise ValueError('Invalid selected IDs')
    entries=[lookup[sid] for sid in ids[:args.limit]]
    for entry in entries:
        other=lookup[selection['shuffled_sources'][entry['id']]]
        if other['paired']['scene_id']==entry['paired']['scene_id']:
            raise ValueError('Diagnostic shuffle must cross buildings')
    if not entries:raise ValueError(f'No reviewed {args.split} samples; use train only for pilot diagnostics')
    if 'shuffled' in args.modes and len(candidates)<2:raise ValueError('Shuffle needs >=2 same-split worlds')
    import torch
    from diffusers import AutoencoderKLQwenImage
    from safetensors.torch import load_file
    from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
    from qwen_edit_pano.paired_data import local_image,target_image
    from qwen_edit_pano.inference_state import enforce_inference_eval
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Need BF16 CUDA GPU')
    output=new_output(args.output)
    write(output/'run_config.json',dict(world_checkpoint=str(args.checkpoint.resolve()),
        world_sha256=complete['world_sha256'],config=config,split=args.split,ids=[e['id'] for e in entries],
        steps=args.steps,cfg=args.cfg,seed=args.seed,modes=args.modes,
        selection=selection,selection_sha256=sha(args.selection),
        sampling_precision='native_pipeline_scoped_world_bf16',pano_lora_scale=config['pano_lora_scale'],
        world_cfg='same world for positive and negative branches',oracle=True,
        note='Train split is fit diagnosis only; not a held-out result'))
    snapshot=bundle['base_snapshot'];base=Path(bundle['base_checkpoint']);pano=read(base/'pano_config.json')
    vae=AutoencoderKLQwenImage.from_pretrained(snapshot,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
    pipe=PairedPanoPipeline.from_pretrained(snapshot,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.vae.to(dtype=torch.float32)
    pipe.load_lora_weights(str(base),weight_name='pytorch_lora_weights.safetensors',adapter_name='pano',local_files_only=True,use_safetensors=True)
    scale_report=apply_scale(pipe.transformer,config['pano_lora_scale'],'pano')
    write(output/'pano_scale_check.json',scale_report)
    print(scale_report,flush=True)
    pipe.enable_pano(bundle['padding_columns'])
    for module in (pipe.transformer,pipe.vae,pipe.text_encoder):
        enforce_inference_eval(module);module.requires_grad_(False)
    hidden=pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
    world=WorldBranch(hidden,config['blocks'],config['width']).cuda().eval().requires_grad_(False)
    world.load_state_dict(load_file(str(args.checkpoint/'world.safetensors')),strict=True)
    world.attach(pipe.transformer)
    # Read-only activation diagnostics: no changes to weights or output tensors.
    from ..inference_diagnostics import RelativeResidualRecorder
    recorder=RelativeResidualRecorder(world)
    pipe.enable_model_cpu_offload()
    for index,entry in enumerate(entries):
        directory=output/entry['id'];directory.mkdir()
        local=local_image(entry['paired']);local.save(directory/'local.png')
        seed=sample_seed(args.seed,entry['id'],'per-id')
        for mode in args.modes:
            source=lookup[selection['shuffled_sources'][entry['id']]] if mode=='shuffled' else entry
            recorder.reset()
            with torch.inference_mode():
                if mode=='baseline':world.set_world(None)
                else:
                    world.set_world(tuple(t.cuda() for t in tensorize(read(source['graph']),
                        bundle['max_objects'],config['condition'])))
                image=pipe(image=local,prompt=pano['prompt'],negative_prompt=' ',height=bundle['height'],
                    width=2*bundle['height'],num_inference_steps=args.steps,true_cfg_scale=args.cfg,
                    generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
            if image.size!=(2*bundle['height'],bundle['height']):raise AssertionError('ERP size')
            image.save(directory/(mode+'.png'))
            write(directory/(mode+'.json'),dict(seed=seed,world_source=None if mode=='baseline' else source['id'],
                diagnostics=world.diagnostics() if mode!='baseline' else {},relative_residual=recorder.summary(),oracle=mode!='baseline'))
            world.set_world(None)
        target_image(entry['paired']).resize((2*bundle['height'],bundle['height'])).save(directory/'gt_erp.png')
    write(output/'COMPLETE.json',dict(samples=len(entries),modes=args.modes,oracle=True,
        note='Generation completed; inspect local fidelity, hidden structure and seams; not quality certification'))


if __name__=='__main__':
    main()
