"""Zero-training Local RGB + GT text oracle. Independent of world adapters."""
import argparse
from pathlib import Path
from .common import ROOT, read, rows, sha, new_output
from .checkpoint_io import atomic_json
from .gt_text import spatial_text, shuffle_sources, VERSION


def prepare(args):
    from qwen_edit_pano.paired_data import paired_rows, CONDITIONING
    ckpt = args.checkpoint.resolve()
    pano = read(ckpt/'pano_config.json')
    completed = read(ckpt/'COMPLETE.json')
    if completed.get('completed_epochs', 0) < 1 or pano['conditioning'] != CONDITIONING:
        raise ValueError('Requires completed formal paired Panorama LoRA')
    if pano['width'] != 2*pano['height']:
        raise ValueError('Expected 2:1 ERP')
    training = read(ckpt/'training_config.json')
    if sha(args.pairs/'train.jsonl') != training['panorama_sha256']:
        raise ValueError('Use the checkpoint original paired manifests')
    paired = {s: paired_rows(args.pairs/f'{s}.jsonl', s) for s in ('train', 'val')}
    if {r['scene_id'] for r in paired['train']} & {r['scene_id'] for r in paired['val']}:
        raise ValueError('Building leakage')
    heldout = rows(Path(training['heldout_manifest']))
    if {r['scene_id'] for r in heldout} & {r['scene_id'] for r in paired['train']}:
        raise ValueError('Checkpoint training/heldout building overlap')
    canonical = {r['id']: r for rs in paired.values() for r in rs}
    if len(canonical) != sum(map(len, paired.values())):
        raise ValueError('Duplicate paired identity')
    if sha(args.gt/'gt_manifest.jsonl') != read(args.gt/'EXPORT_COMPLETE.json')['gt_manifest_sha256']:
        raise ValueError('GT export manifest changed')
    review = read(args.review)
    vocabulary = read(args.gt/'vocabulary.json')
    pool, excluded = {}, []
    for record in rows(args.gt/'gt_manifest.jsonl'):
        if record['split'] != args.split:
            continue
        sid = record['id']
        if review.get(sid, {}).get('decision') != 'approved' or record['review_flags']:
            excluded.append(sid)
            continue
        row = canonical[sid]
        directory = args.gt/record['directory']
        graph_path = directory/'G_full.json'
        graph = read(graph_path)
        if sha(graph_path) != record['gt_sha256']['G_full'] or graph['id'] != sid or graph['split'] != args.split or row['split'] != args.split:
            raise ValueError(f'GT identity/hash mismatch: {sid}')
        provenance = read(directory/'provenance.json')
        for key in ('image', 'local_condition', 'target_alignment'):
            if row[key] != provenance['paired_row'][key]:
                raise ValueError(f'Local/ERP/GT provenance mismatch: {sid}/{key}')
        text, projection = spatial_text(graph, vocabulary, args.text_format)
        pool[sid] = dict(id=sid, paired={k: row[k] for k in ('id','split','scene_id','source_view_id','image','local_condition','target_alignment')},
                         graph=str(graph_path.resolve()), graph_sha256=sha(graph_path),
                         structured_text=text, projection=projection)
    ids = sorted(pool)[:args.limit]
    if not ids:
        raise ValueError('No approved, flag-free samples in this split. Current pilot has train only.')
    shuffled = shuffle_sources(pool, args.shuffle_seed)
    prompts = {}
    for sid in ids:
        prompts[sid] = {}
        for mode, source in [('baseline', None), ('correct', sid), ('shuffled', shuffled[sid])]:
            prompt = pano['prompt'] if source is None else pano['prompt'] + '\n\n' + pool[source]['structured_text']
            prompts[sid][mode] = dict(prompt=prompt, source_id=source,
                                     source_graph_sha256=None if source is None else pool[source]['graph_sha256'])
    protected = [ckpt/'pano_config.json', ckpt/'training_config.json', ckpt/'COMPLETE.json',
                 ckpt/'pytorch_lora_weights.safetensors', args.gt/'gt_manifest.jsonl', args.gt/'contract.json',
                 args.gt/'vocabulary.json', args.review, args.pairs/'train.jsonl', args.pairs/'val.jsonl',
                 Path(training['heldout_manifest'])]
    config = dict(experiment='GT-as-Text Oracle', zero_training=True, world_adapter=False,
                  checkpoint=str(ckpt), completed_epochs=completed['completed_epochs'], pano=pano,
                  text_version=VERSION, text_format=args.text_format, split=args.split, ids=ids,
                  seeds=args.seeds, shuffle_seed=args.shuffle_seed, shuffle_mapping=shuffled,
                  steps=args.steps, cfg=args.cfg, negative_prompt=' ', max_text_tokens=args.max_text_tokens,
                  max_encoder_tokens=args.max_encoder_tokens, offload=args.offload,
                  protected_files={str(p.resolve()): sha(p) for p in protected}, excluded=excluded,
                  code={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('gt_text.py')]},
                  note='Train split is a diagnostic, not held-out generalization. Text is a versioned projection of GT, not an LLM caption.')
    return config, pool, prompts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006')
    p.add_argument('--gt', type=Path, default=ROOT/'qwen_edit_pano/data/gt_world_pilot_v1')
    p.add_argument('--review', type=Path, default=ROOT/'qwen_edit_pano/data/gt_world_pilot_v1/assistant_review_decisions_2026-10-02.json')
    p.add_argument('--pairs', type=Path, default=ROOT/'qwen_edit_pano/data/paired_full_v1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train','val'], default='val')
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--shuffle-seed', type=int, default=0)
    p.add_argument('--text-format', choices=['spatial','metric'], default='spatial')
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--cfg', type=float, default=4.)
    p.add_argument('--max-text-tokens', type=int, default=1024)
    p.add_argument('--max-encoder-tokens', type=int, default=4096)
    p.add_argument('--offload', choices=['model','sequential','none'], default='model')
    p.add_argument('--prepare-only', action='store_true', help='Write prompts/provenance only; no model or tokenizer load')
    p.add_argument('--rewrite-file', type=Path, help='Text-only LLM drafts tied to an existing original GT-text run')
    p.add_argument('--modes', nargs='+', choices=['baseline','correct','shuffled','llm_correct','llm_shuffled'],
                   default=['baseline','correct','shuffled'])
    args = p.parse_args()
    if min(args.limit, args.steps, args.max_text_tokens, args.max_encoder_tokens) < 1 or args.cfg < 1 or len(args.seeds) != len(set(args.seeds)):
        p.error('Invalid counts/CFG/seeds')
    config, pool, prompts = prepare(args)
    if len(args.modes) != len(set(args.modes)):
        p.error('Duplicate modes')
    if any(mode.startswith('llm_') for mode in args.modes) and args.rewrite_file is None:
        p.error('LLM modes require --rewrite-file')
    if args.rewrite_file is not None:
        from .gt_text_rewrite import apply_rewrites
        apply_rewrites(config, pool, prompts, args.rewrite_file)
    prompts = {sid: {mode: entries[mode] for mode in args.modes} for sid, entries in prompts.items()}
    config['modes'] = args.modes
    out = new_output(args.output)
    atomic_json(out/'run_config.json', config)
    atomic_json(out/'prompts.json', prompts)
    atomic_json(out/'gt_projection.json', pool)
    if args.prepare_only:
        atomic_json(out/'PREPARED.json', dict(samples=len(prompts), inference_run=False))
        print(f'Prepared {len(prompts)} samples x {len(args.modes)} modes; inspect {out}/prompts.json. No model loaded.')
        return
    import torch
    from diffusers import AutoencoderKLQwenImage
    from qwen_edit_pano.common import require_versions, sample_seed, versions
    from qwen_edit_pano.paired_pipeline import PairedPanoPipeline
    from qwen_edit_pano.paired_data import local_image, target_image
    from qwen_edit_pano.inference_state import enforce_inference_eval
    require_versions()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Requires BF16 CUDA GPU')
    pano = config['pano']
    snapshot = pano['base_snapshot']
    vae = AutoencoderKLQwenImage.from_pretrained(snapshot, subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    pipe = PairedPanoPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.to(dtype=torch.float32)
    pipe.load_lora_weights(str(args.checkpoint), weight_name='pytorch_lora_weights.safetensors', local_files_only=True, use_safetensors=True)
    pipe.enable_pano(pano['padding_columns'])
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        enforce_inference_eval(module)
        module.requires_grad_(False)
    # Never reuse embeddings from the ordinary-task training cache: they do not
    # contain this experiment's GT text. Native Edit encodes Local+prompt anew.
    counts = {sid: {mode: len(pipe.processor.tokenizer(value['prompt'], add_special_tokens=False)['input_ids'])
                    for mode, value in modes.items()} for sid, modes in prompts.items()}
    atomic_json(out/'text_token_counts.json', counts)
    if any(n > args.max_text_tokens for modes in counts.values() for n in modes.values()):
        raise ValueError('Prompt exceeds explicit text budget; no silent object dropping/truncation. Inspect text_token_counts.json.')
    encoded_lengths = []
    def check_encoder_length(module, positional, kwargs):
        token_ids = kwargs.get('input_ids')
        if token_ids is None:
            raise ValueError('Unexpected native Edit text encoder API')
        length = token_ids.shape[-1]
        encoded_lengths.append(int(length))
        if length > args.max_encoder_tokens:
            raise ValueError('Combined image/template/text tokens exceed budget; no silent truncation')
    hook = pipe.text_encoder.register_forward_pre_hook(check_encoder_length, with_kwargs=True)
    if args.offload == 'model':
        pipe.enable_model_cpu_offload()
    elif args.offload == 'sequential':
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to('cuda')
    atomic_json(out/'runtime.json', dict(versions=versions(), device=torch.cuda.get_device_name(),
                transformer_dtype=str(next(pipe.transformer.parameters()).dtype), vae_dtype=str(next(pipe.vae.parameters()).dtype)))
    for sid, modes in prompts.items():
        directory = out/sid
        directory.mkdir()
        local = local_image(pool[sid]['paired'])
        local.save(directory/'local.png')
        for seed_base in args.seeds:
            seed = sample_seed(seed_base, sid, 'per-id')
            seed_dir = directory/f'seed-{seed_base}'
            seed_dir.mkdir()
            for mode, entry in modes.items():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                encoded_lengths.clear()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    image = pipe(image=local, prompt=entry['prompt'], negative_prompt=' ',
                                 height=pano['height'], width=pano['width'], num_inference_steps=args.steps,
                                 true_cfg_scale=args.cfg, max_sequence_length=1024,
                                 generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
                if image.size != (pano['width'], pano['height']):
                    raise ValueError('Unexpected ERP size')
                image.save(seed_dir/f'{mode}.png')
                atomic_json(seed_dir/f'{mode}.json', dict(**entry, seed=seed, text_tokens=counts[sid][mode],
                            encoded_lengths=list(encoded_lengths), image_sha256=sha(seed_dir/f'{mode}.png'),
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
                print(f'{sid} seed={seed_base} mode={mode} completed', flush=True)
        # RGB ground truth is only loaded after all generation for this sample.
        target_image(pool[sid]['paired']).resize((pano['width'], pano['height'])).save(directory/'gt_erp.png')
    hook.remove()
    atomic_json(out/'COMPLETE.json', dict(samples=len(prompts), seeds=args.seeds, modes=args.modes,
                generated_images=len(prompts)*len(args.seeds)*len(args.modes), zero_training=True,
                note='Generation complete, not evidence of improved quality or held-out generalization'))


if __name__ == '__main__':
    main()
