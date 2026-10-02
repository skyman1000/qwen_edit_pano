"""Native Qwen flow-matching LoRA with DiT360 panorama supervision.

Single GPU, batch1/accum4. Frozen base weights are NOT saved in checkpoints.
Checkpoint resume is at completed epoch boundaries. Fixed face2, synchronized flip only.
"""
import argparse
import json
import math
import time
from pathlib import Path

from .common import (DEFAULT_MODEL, DEFAULT_PROMPTS, atomic_json, model_snapshot,
                     require_versions, sha256, source_hashes, text_hash, versions)


def parse_args():
    from qwen_pano.profiles import REPO_PANORAMA
    p = argparse.ArgumentParser(description="Local RGB -> ERP panorama-only LoRA; native Edit conditioning")
    p.add_argument('--manifest', dest='panorama_manifest', type=Path, required=True)
    p.add_argument('--heldout-manifest', type=Path, required=True)
    p.add_argument('--condition-cache', dest='text_cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--revision')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--workers', type=int, default=25)
    p.add_argument('--warmup-steps', type=int, default=2350, help='Optimizer-step linear warmup; 2350 for the confirmed 7521-row/25-epoch run')
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--alignment-reviewed', action='store_true', help='You inspected prepare_pairs previews for direction/content agreement')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--no-flip', dest='augment', action='store_false', default=True)
    args = p.parse_args()
    if not args.alignment_reviewed:
        p.error('Inspect pair previews first, then pass --alignment-reviewed; identity match alone is insufficient')
    if args.epochs <= 0 or args.workers < 0 or args.log_every <= 0 or args.warmup_steps < 0:
        p.error('Invalid epochs/workers/log-every/warmup-steps')
    for key, value in REPO_PANORAMA.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.profile = 'repo-panorama-paired-face2'
    args.perspective_manifest = None
    args.official_full_training_only = False
    args.allow_unverified_split = False
    args.init_lora = None
    args.vae_tiling = False
    return args


def main():
    args = parse_args()
    from .common import require_local_output
    from .paired_data import CONDITIONING, PairedDataset, paired_collate, paired_rows, cache_key
    require_local_output(args.output, "outputs")
    require_local_output(args.text_cache, "cache")
    if args.smoke_test and (args.resume or args.init_lora):
        raise ValueError("Smoke test must start from the base model")
    require_versions()
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedType, set_seed
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline as QwenImagePipeline, QwenImageTransformer2DModel
    from diffusers.training_utils import (_collate_lora_metadata, compute_density_for_timestep_sampling,
                                          compute_loss_weighting_for_sd3)
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, RandomSampler
    from .paired_circular import install_circular_padding
    from .losses import panorama_loss
    from qwen_pano.profiles import lr_multiplier

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation_steps, mixed_precision="bf16")
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU):
        raise ValueError("This entry supports single GPU / DDP; FSDP and DeepSpeed need separate checkpoint handling")
    if accelerator.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a CUDA GPU with bfloat16 support")
    set_seed(args.seed, device_specific=True)
    if accelerator.num_processes != 1:
        raise ValueError('This audited paired entry currently supports one GPU; do not silently change effective batch')
    allow_unverified = False
    panoramas = paired_rows(args.panorama_manifest, 'train')
    from .common import read_jsonl, check_ids
    heldout = read_jsonl(args.heldout_manifest)
    excluded_houses = {r['scene_id'] for r in heldout}
    excluded_views = {r['source_view_id'] for r in heldout}
    if any(r['scene_id'] in excluded_houses or r['source_view_id'] in excluded_views for r in panoramas):
        raise ValueError('Held-out house/view appears in training')
    if args.smoke_test and len(panoramas) < 2 * args.accumulation_steps:
        raise ValueError('Smoke test requires at least 8 training rows')
    perspectives, hf_dataset = [], None
    with accelerator.main_process_first():
        snapshot = model_snapshot(args.model, args.revision, args.local_files_only)
    cache_config = json.loads((args.text_cache / 'cache_config.json').read_text())
    if (cache_config.get('conditioning') != CONDITIONING or cache_config['base_snapshot'] != snapshot
            or cache_config['versions'] != versions() or cache_config['manifest_sha256'] != sha256(args.panorama_manifest)):
        raise ValueError('Condition cache does not match this model, environment or frozen paired manifest')
    if cache_config.get('cache_sources') != {n:sha256(Path(__file__).parent/n) for n in ('cache_paired.py','paired_data.py')}:
        raise ValueError('Condition cache encoder source changed; rebuild in a new directory')
    cache_complete = json.loads((args.text_cache / 'complete.json').read_text())
    if cache_complete['config_sha256'] != sha256(args.text_cache / 'cache_config.json') or cache_complete.get('count') != 2*len(panoramas):
        raise ValueError('Incomplete condition cache')
    for row in panoramas:
        # Validate bytes before using a cache: target images never condition the model.
        from .paired_data import local_image
        local_image(row)
        for flip in (False, True):
            path = args.text_cache / (cache_key(row, flip) + '.safetensors')
            if not path.is_file():
                raise FileNotFoundError(path)
    # User-selected 2350 for 7521 rows, 25 epochs, accumulation 4; explicit in config.
    computed_warmup = args.warmup_steps
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("resume", "output")}
    config.update(conditioning=CONDITIONING, base_snapshot=snapshot, width=2 * args.height, versions=versions(),
                  world_size=accelerator.num_processes, source_hashes=source_hashes(),
                  panorama_sha256=sha256(args.panorama_manifest),
                  perspective_sha256=sha256(args.perspective_manifest) if args.perspective_manifest else None,
                  heldout_sha256=sha256(args.heldout_manifest),
                  text_cache_config_sha256=sha256(args.text_cache / "cache_config.json"),
                  computed_warmup_steps=computed_warmup,
                  allow_missing_house_identity=allow_unverified,
                  actual_panorama_count=len(panoramas), actual_perspective_count=len(perspectives),
                  global_effective_batch=args.batch_size * args.accumulation_steps * accelerator.num_processes,
                  data_backend="huggingface_load_dataset" if hf_dataset is not None else "manifest",
                  persistent_workers=args.schedule == "panorama" and args.workers > 0,
                  split_status="unverified" if any(not (r.get("scene_id") or r.get("scan_id")) for r in panoramas)
                  else "checked_against_supplied_heldout_houses")
    config["augmentation"] = "synchronized_horizontal_flip_only_no_random_yaw" if args.augment else "none"
    config["conditioning_prompt"] = cache_config["prompt"]
    if args.resume:
        from .resume_compat import compatible_config
        old = json.loads((args.resume / "training_config.json").read_text())
        if not compatible_config(old, config):
            raise ValueError("Resume configuration changed. For a new experiment, use base weights and a new output")
        if not (args.resume / "COMPLETE.json").exists():
            raise ValueError("Checkpoint incomplete; use the previous completed epoch checkpoint")
    config_path = args.output / "training_config.json"
    # Validate on every rank before rank zero mutates the shared directory.
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output must be empty for a new training run")
    if config_path.exists():
        from .resume_compat import compatible_config
        if not compatible_config(json.loads(config_path.read_text()), config):
            raise ValueError("Output belongs to another experiment")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)
    accelerator.wait_for_everyone()
    vae = AutoencoderKLQwenImage.from_pretrained(snapshot, subfolder="vae", torch_dtype=getattr(torch, args.vae_dtype),
                                                local_files_only=True).requires_grad_(False).eval().to(accelerator.device)
    if args.vae_tiling:
        vae.enable_tiling()
    vae_scale = 2 ** len(vae.temperal_downsample)
    if vae_scale != 8:
        raise ValueError("This implementation expects Qwen-Image's 8x spatial VAE")
    mean = torch.tensor(vae.config.latents_mean, device=accelerator.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=accelerator.device).reshape(1, -1, 1, 1, 1)
    transformer = QwenImageTransformer2DModel.from_pretrained(snapshot, subfolder="transformer",
                      torch_dtype=torch.bfloat16, local_files_only=True).requires_grad_(False)
    if transformer.config.guidance_embeds or not getattr(transformer, "zero_cond_t", False):
        raise ValueError("Expected non-distilled Edit-2511 with zero_cond_t=True")
    targets = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
    transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout, init_lora_weights="gaussian", target_modules=targets))
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    install_circular_padding(transformer, args.padding_columns)
    transformer.enable_gradient_checkpointing()
    transformer.train()
    from .validation import parameter_report
    parameter_counts = parameter_report(transformer, vae)
    if accelerator.is_main_process:
        atomic_json(args.output / "parameter_report.json", parameter_counts)
    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(.9, .999), weight_decay=1e-5, eps=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda index: lr_multiplier(index, args.lr_schedule, args.epochs, computed_warmup))
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    # Step exactly once per actual optimizer update; do not multiply by GPU count.
    accelerator.register_for_checkpointing(lr_scheduler)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler", local_files_only=True)
    # Use the scheduler's native training timestep/sigma table, matching the
    # official Qwen LoRA example. Inference uses its native resolution shift.
    sigmas_table = scheduler.sigmas.to(accelerator.device)
    timesteps_table = scheduler.timesteps.to(accelerator.device)
    pano_config = dict(conditioning=CONDITIONING, base_snapshot=snapshot, padding_columns=args.padding_columns,
                       rank=args.rank, lora_alpha=args.lora_alpha, height=args.height, width=args.height * 2,
                       max_sequence_length=cache_config["max_sequence_length"],
                       targets=targets, geometry=args.geometry_backend, profile=args.profile,
                       inference_padding_columns=1 if args.profile.startswith("repo-") else args.padding_columns,
                       inference_padding_mode="persistent", schema_version=3, prompt=cache_config["prompt"],
                       augmentation="synchronized_horizontal_flip_only_no_random_yaw", reference_vae_dtype="float32")

    def save_hook(models, weights, directory):
        if accelerator.is_main_process:
            state = {k: v.detach().cpu().contiguous() for k, v in
                     get_peft_model_state_dict(accelerator.unwrap_model(models[0])).items()}
            save_file(state, str(Path(directory) / "adapter_resume.safetensors"))
            QwenImagePipeline.save_lora_weights(directory, transformer_lora_layers=state,
                **_collate_lora_metadata({"transformer": accelerator.unwrap_model(models[0])}))
            atomic_json(Path(directory) / "pano_config.json", pano_config)
            atomic_json(Path(directory) / "training_config.json", config)
        weights.clear()  # Prevent saving the frozen 20B base.

    def load_hook(models, directory):
        model = accelerator.unwrap_model(models.pop())
        result = set_peft_model_state_dict(model, load_file(str(Path(directory) / "adapter_resume.safetensors")))
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("Incomplete or incompatible checkpoint LoRA state")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    start_epoch, step = 0, 0
    if args.resume:
        accelerator.load_state(str(args.resume))
        progress = json.loads((args.resume / "COMPLETE.json").read_text())
        start_epoch, step = progress["completed_epochs"], progress["optimizer_step"]
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint already reached requested epochs; start a new experiment from base weights")
    accelerator.print(f"Trainable LoRA parameters: {sum(p.numel() for p in params):,}; {config['split_status']}")

    def make_loader(rows, persistent=False):
        dataset = PairedDataset(rows, args.height, args.text_cache, augment=args.augment, seed=args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        # Keep shuffle RNG independent of worker startup: persistent workers
        # start once, whereas resumed runs start fresh workers at a later epoch.
        sampler = RandomSampler(dataset, generator=generator)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                            collate_fn=paired_collate, pin_memory=True,
                            generator=torch.Generator().manual_seed(args.seed), drop_last=args.drop_last,
                            persistent_workers=persistent and args.workers > 0)
        loader = accelerator.prepare_data_loader(loader)
        if not len(loader):
            raise ValueError("No training batches remain after distributed sharding/drop_last")
        return dataset, generator, loader

    # Pure panorama training constructs/prepares exactly one loader for the run.
    rows = panoramas
    dataset, generator, loader = make_loader(rows, persistent=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        generator.manual_seed(args.seed + epoch)
        loader.set_epoch(epoch)
        accelerator.print(f"Epoch {epoch + 1}/{args.epochs}: panorama={len(panoramas)}, perspective={len(rows) - len(panoramas)}")
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    z = vae.encode(batch["pixels"].to(vae.dtype).unsqueeze(2)).latent_dist.sample()
                    z = ((z.float() - mean) / std).to(torch.bfloat16)
                    embeddings, text_mask = batch["embeddings"], batch["text_mask"]
                    reference = batch["reference"]
                noise = torch.randn_like(z)
                u = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme,
                            batch_size=len(z), logit_mean=args.logit_mean, logit_std=args.logit_std,
                            mode_scale=args.mode_scale)
                indices = (u * scheduler.config.num_train_timesteps).long().clamp_max(len(timesteps_table) - 1).to(z.device)
                t = timesteps_table[indices]
                sigma = sigmas_table[indices].reshape(-1, 1, 1, 1, 1).to(z.dtype)
                noisy = (1 - sigma) * z + sigma * noise
                packed = QwenImagePipeline._pack_latents(noisy, len(z), z.shape[1], z.shape[3], z.shape[4])
                with accelerator.autocast():
                    prediction = transformer(hidden_states=torch.cat([packed, reference], dim=1), timestep=t / 1000,
                          encoder_hidden_states=embeddings, encoder_hidden_states_mask=text_mask,
                          img_shapes=[[(1, z.shape[3] // 2, z.shape[4] // 2), tuple(batch["ref_shape"])]] * len(z), return_dict=False)[0]
                prediction = prediction[:, :packed.shape[1]]
                prediction = QwenImagePipeline._unpack_latents(prediction, args.height, args.height * 2, vae_scale)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigma)
                pano = torch.ones(len(z), device=z.device, dtype=torch.bool)
                loss, components = panorama_loss(prediction, noise - z, batch["mask"], pano,
                                                  weighting, args.geometry_start_epoch if args.smoke_test else epoch, args, noise=noise, clean=z)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in {batch['ids']}")
                accelerator.backward(loss)
                smoke_probe = None
                if args.smoke_test and accelerator.sync_gradients:
                    from .validation import gradient_report, update_probe
                    gradient_counts = gradient_report(accelerator.unwrap_model(transformer), vae)
                    smoke_probe = update_probe(params)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    if args.lr_schedule != "repo-epoch":
                        lr_scheduler.step()
                    step += 1
                if args.smoke_test and smoke_probe is not None and step >= 2:
                    probe, before = smoke_probe
                    if torch.equal(probe.detach().cpu(), before):
                        raise AssertionError("Nonzero-LR optimizer step did not change the LoRA probe")
                optimizer.zero_grad(set_to_none=True)
            if args.smoke_test and step >= 2:
                from .validation import checkpoint_roundtrip
                checkpoint = args.output / "smoke-checkpoint"
                accelerator.wait_for_everyone()
                accelerator.save_state(str(checkpoint))
                accelerator.wait_for_everyone()
                checkpoint_roundtrip(accelerator, transformer, checkpoint)
                report = dict(status="passed", optimizer_steps=step, loss=float(loss.detach()),
                              components={k: float(v) for k, v in components.items()},
                              parameter_counts=parameter_counts, gradients=gradient_counts,
                              checkpoint_roundtrip=True, optimizer_parameter_changed=True,
                              inference="pending: run paired_inference on smoke-checkpoint")
                if accelerator.is_main_process:
                    atomic_json(checkpoint / "SMOKE_COMPLETE.json", report)
                accelerator.end_training()
                return
            if accelerator.sync_gradients and step % args.log_every == 0:
                average = accelerator.gather(loss.detach().reshape(1)).mean().item()
                if accelerator.is_main_process:
                    elapsed = time.perf_counter() - epoch_started
                    record = dict(epoch=epoch, step=step, loss_last_microbatch_across_ranks=average,
                                  microbatches_seen=batch_index + 1, epoch_elapsed_seconds=elapsed,
                                  estimated_epoch_seconds=elapsed * len(loader) / (batch_index + 1),
                                  lr=lr_scheduler.get_last_lr()[0],
                                  rank0_last_microbatch={k: v.item() for k, v in components.items()})
                    with (args.output / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(record, flush=True)
        if args.lr_schedule == "repo-epoch":
            lr_scheduler.step()
        checkpoint = args.output / f"checkpoint-epoch{epoch + 1:03d}"
        if checkpoint.exists():
            raise ValueError(f"Checkpoint already exists; use a new output when branching a run: {checkpoint}")
        accelerator.wait_for_everyone()
        accelerator.save_state(str(checkpoint))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            atomic_json(checkpoint / "COMPLETE.json", dict(completed_epochs=epoch + 1, optimizer_step=step))
        accelerator.wait_for_everyone()
    accelerator.print(f"Finished. Inference LoRA: {checkpoint}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
