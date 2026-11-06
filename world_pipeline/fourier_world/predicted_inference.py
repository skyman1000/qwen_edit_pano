"""One predicted scene -> frozen Panorama/World diagnostic, without scene GT.

Compare no World / predicted G_obs / predicted G_full with identical RGB,
prompt, checkpoint, sampling settings and seed. Existing Oracle entry unchanged.
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

from ..common import read, sha, write, new_output, validate_graph
from ..oracle_model import FEATURE_VERSION, tensorize
from .binding import apply_scale, verify_binding


def scene_inputs(directory, vocabulary_hash, max_objects):
    directory = Path(directory).resolve()
    paths = {name: directory/name for name in ('G_obs.json', 'G_full.json', 'local.png')}
    # Completion runs record evidence in input.json; earlier single-image runs
    # use config.json. Read either native format without copying/relabeling graphs.
    metadata = directory/('config.json' if (directory/'config.json').exists() else 'input.json')
    paths['metadata'] = metadata
    config = read(metadata)
    image_path = Path(config['image'])
    source_hashes = {}
    if metadata.name == 'input.json':
        source_hashes = config['source_sha256']
        if any(sha(Path(p)) != digest for p,digest in source_hashes.items()):
            raise ValueError('Completion source image/observation metadata changed')
        image_hash = source_hashes.get(str(image_path.resolve()))
        observed_path = Path(config['state']).resolve()
        if (str(observed_path) not in source_hashes
                or read(observed_path) != read(paths['G_obs.json'])):
            raise ValueError('Completion G_obs differs from its recorded predicted source')
    else:
        image_hash = config['image_sha256']
    if sha(image_path) != image_hash:
        raise ValueError('Original input RGB changed')
    with Image.open(image_path) as handle:
        image = handle.convert('RGB')
    thumbnail = image.copy()
    thumbnail.thumbnail((768, 768))
    with Image.open(paths['local.png']) as handle:
        saved = handle.convert('RGB')
    if thumbnail.size != saved.size or thumbnail.tobytes() != saved.tobytes():
        raise ValueError('Scene-state RGB differs from the original input')
    graphs = {mode: validate_graph(read(paths[name]))
              for mode, name in (('observed', 'G_obs.json'), ('completed', 'G_full.json'))}
    observed, full = graphs['observed'], graphs['completed']
    for graph, scope in ((observed, 'observed'), (full, 'full')):
        if graph['stage'] != 'prediction' or graph['scope'] != scope:
            raise ValueError('Use predicted G_obs/G_full, not Oracle/GT graphs')
        if graph['vocabulary_sha256'] != vocabulary_hash:
            raise ValueError('Prediction vs checkpoint vocabulary mismatch')
        for obj in graph['objects']:
            if not all(obj[k]['valid'] for k in ('category_id', 'center_local_m', 'size_aabb_local_m')):
                raise ValueError('This coarse-box diagnostic requires category/center/full-size for every object')
    count = len(observed['objects'])
    if not count or observed['id'] != full['id'] or full['objects'][:count] != observed['objects']:
        raise ValueError('G_full must preserve every observed object and append new instances')
    if any(o['local_evidence'] != 'visible' for o in observed['objects']):
        raise ValueError('G_obs must contain visible evidence')
    if any(o['local_evidence'] == 'visible' for o in full['objects'][count:]):
        raise ValueError('Appended objects must be unobserved hypotheses')
    tensors = {mode: tensorize(graph, max_objects, 'full') for mode, graph in graphs.items()}
    # Do not use condition='observed' on G_full: it would discard the completion.
    for a, b in zip(tensors['observed'], tensors['completed']):
        if not a[:, :count+1].equal(b[:, :count+1]):
            raise AssertionError('Observed feature prefix changed')
    paths['image'] = image_path
    return image, graphs, tensors, {**source_hashes, **{str(p): sha(p) for p in paths.values()}}


def checkpoint_inputs(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    config = read(checkpoint/'config.json')
    complete = read(checkpoint/'COMPLETE.json')
    if (config['feature_version'] != FEATURE_VERSION or complete['feature_version'] != FEATURE_VERSION
            or sha(checkpoint/'world.safetensors') != complete['world_sha256']):
        raise ValueError('World checkpoint incomplete/changed')
    if config['condition'] != 'full':
        raise ValueError('Use a World checkpoint trained with condition=full')
    bundle_path = Path(config['bundle'])/'BUNDLE.json'
    if sha(bundle_path) != config['bundle_sha256']:
        raise ValueError('Bound bundle metadata changed')
    bundle = read(bundle_path)  # Metadata only; no samples.jsonl or GT graph loading.
    if bundle['feature_version'] != FEATURE_VERSION:
        raise ValueError('Bundle feature version mismatch')
    verify_binding(config, bundle)
    base = Path(bundle['base_checkpoint'])
    protected = bundle['protected_files']
    vocab_paths = [Path(p) for p in protected if Path(p).name == 'vocabulary.json']
    if len(vocab_paths) != 1:
        raise ValueError('Need one checkpoint-bound category vocabulary')
    required = [vocab_paths[0], base/'pano_config.json', base/'pytorch_lora_weights.safetensors']
    for path in required:
        if protected.get(str(path)) != sha(path):
            raise ValueError('Bound vocabulary/Panorama file changed: '+str(path))
    vocabulary_hash = read(vocab_paths[0])['category_mapping_sha256']
    pano = read(base/'pano_config.json')
    if (pano['base_snapshot'] != bundle['base_snapshot'] or pano['height'] != bundle['height']
            or pano['inference_padding_columns'] != bundle['padding_columns']
            or pano['conditioning'] != 'edit2511_native_local_rgb_face2_v1'):
        raise ValueError('Panorama config vs bound bundle mismatch')
    return config, complete, bundle, pano, vocabulary_hash


def comparison(output, modes):
    canvas = Image.new('RGB', (1024, 544*len(modes)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, mode in enumerate(modes):
        draw.text((12, i*544+10), mode+' (same RGB, prompt and seed)', fill='black')
        with Image.open(output/(mode+'.png')) as image:
            canvas.paste(image.resize((1024, 512)), (0, i*544+32))
    canvas.save(output/'comparison.png')


def generate(pipe, world, tensors, image, pano, bundle, output, modes, steps, cfg, seed):
    import torch
    from safetensors.torch import save_file
    from ..inference_diagnostics import RelativeResidualRecorder
    recorder = RelativeResidualRecorder(world)
    for mode in modes:
        print('Panorama diagnostic: '+mode, flush=True)
        recorder.reset()
        try:
            with torch.inference_mode():
                world.set_world(None if mode == 'baseline' else tuple(t.cuda() for t in tensors[mode]))
                result = pipe(image=image, prompt=pano['prompt'], negative_prompt=' ',
                    height=bundle['height'], width=2*bundle['height'],
                    num_inference_steps=steps, true_cfg_scale=cfg,
                    generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
            if result.size != (2*bundle['height'], bundle['height']):
                raise AssertionError('Expected 2:1 ERP output')
            result.save(output/(mode+'.png'))
            if mode != 'baseline':
                save_file(dict(zip(('category', 'geometry', 'mask'), tensors[mode])),
                          str(output/(mode+'_input.safetensors')))
            write(output/(mode+'.json'), dict(seed=seed, oracle=False,
                graph=None if mode == 'baseline' else 'G_obs.json' if mode == 'observed' else 'G_full.json',
                diagnostics={} if mode == 'baseline' else world.diagnostics(),
                relative_residual=recorder.summary(),
                note='Activation/image changes do not certify correct object control'))
        finally:
            world.set_world(None)
    comparison(output, modes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--cfg', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--modes', nargs='+', choices=('baseline', 'observed', 'completed'),
                        default=['baseline', 'observed', 'completed'])
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if (args.steps < 1 or not math.isfinite(args.cfg) or args.cfg < 1
            or not 0 <= args.seed < 2**63 or len(set(args.modes)) != len(args.modes)):
        parser.error('Invalid steps/cfg/seed/modes')
    if args.output.exists():
        parser.error('Output exists; use a new directory')
    from qwen_edit_pano.common import require_versions
    require_versions()
    config, complete, bundle, pano, vocabulary_hash = checkpoint_inputs(args.checkpoint)
    image, graphs, tensors, hashes = scene_inputs(args.scene, vocabulary_hash, bundle['max_objects'])
    counts = {mode: len(graph['objects']) for mode, graph in graphs.items()}
    if args.preflight:
        print(dict(status='preflight_passed', image_size=image.size, objects=counts,
            encoder_masks={k: int(v[2].sum()) for k,v in tensors.items()},
            pano_lora_scale=config['pano_lora_scale'], scene_gt_read=False,
            gpu_model_loaded=False, note='CPU interface checks only; no quality certification'), flush=True)
        return
    import torch
    from diffusers import AutoencoderKLQwenImage
    from safetensors.torch import load_file
    from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
    from qwen_edit_pano.inference_state import enforce_inference_eval
    from .runtime import WorldBranch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Need BF16 CUDA GPU')
    output = new_output(args.output)
    write(output/'run_config.json', dict(scene=str(args.scene.resolve()), source_sha256=hashes,
        world_checkpoint=str(args.checkpoint.resolve()), world_sha256=complete['world_sha256'],
        config=config, objects=counts, image_size=image.size,
        steps=args.steps, cfg=args.cfg, seed=args.seed, modes=args.modes,
        oracle=False, scene_gt_read=False, sampling_precision='native_pipeline_scoped_world_bf16',
        world_cfg='same world for positive and negative text branches',
        note='Predicted-layout diagnostic on one image, not completion/panorama quality certification'))
    image.save(output/'local.png')
    write(output/'G_obs.json', graphs['observed'])
    write(output/'G_full.json', graphs['completed'])
    snapshot = bundle['base_snapshot']
    base = Path(bundle['base_checkpoint'])
    vae = AutoencoderKLQwenImage.from_pretrained(snapshot, subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = PairedPanoPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.to(dtype=torch.float32)
    pipe.load_lora_weights(str(base), weight_name='pytorch_lora_weights.safetensors',
                          adapter_name='pano', local_files_only=True, use_safetensors=True)
    write(output/'pano_scale_check.json', apply_scale(pipe.transformer, config['pano_lora_scale'], 'pano'))
    pipe.enable_pano(bundle['padding_columns'])
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        enforce_inference_eval(module)
        module.requires_grad_(False)
    hidden = pipe.transformer.config.num_attention_heads*pipe.transformer.config.attention_head_dim
    world = WorldBranch(hidden, config['blocks'], config['width']).cuda().eval().requires_grad_(False)
    world.load_state_dict(load_file(str(args.checkpoint/'world.safetensors')), strict=True)
    world.attach(pipe.transformer)
    pipe.enable_model_cpu_offload()
    generate(pipe, world, tensors, image, pano, bundle, output, args.modes, args.steps, args.cfg, args.seed)
    if any(sha(Path(path)) != digest for path,digest in hashes.items()):
        raise ValueError('Scene inputs changed during generation')
    write(output/'COMPLETE.json', dict(modes=args.modes, objects=counts, oracle=False,
        scene_gt_read=False, quality_status='requires_panorama_review',
        note='Inspect preservation and predicted object category/direction; image changes alone are insufficient'))


if __name__ == '__main__':
    main()
