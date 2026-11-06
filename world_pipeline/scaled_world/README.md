# Epoch015 / scale 0.5 World oracle

> Updated after the 2026-10-12 audit: experiment version is now
> `frozen_pano_scale_world_v2`. Use fresh `_v2` output paths. The earlier v1
> configuration cannot be resumed with v2. This changes sampling/RNG/precision
> policy, not GT fields or the adapter architecture.

## Selection and limits

All four Panorama-only runs in `outputs/pano_epoch_scale_v1` completed six images,
with identical manifest hash, seed 0, 28 steps, CFG 4 and 1024x2048 output.
Flat ERP visual review provisionally selects **epoch015 / scale 0.5** for the next
oracle experiment. This is six validation scenes from three buildings, not a
full validation ranking. In particular:

- Stair hall: scale 0.5 reduces some of the extensive corridor/room extrapolation
  seen at scale 1.0, but none recovers the GT hall.
- First living room: all retain the visible sofas; none recovers the complete GT
  kitchen/TV layout. There is no decisive universal winner here.
- SPA: epoch015/0.5 still invents a large blue opening. Epoch006/1.0 is competitive
  and closer to the GT's enclosed room in this example.
- Bar: epoch015/0.5 has plausible, relatively clear furniture details; differences
  are modest and all versions change the unseen space.
- Bedroom: epoch015/0.5 avoids the large open wardrobe in the scale-1.0 versions,
  but still fails to recover the GT bathroom layout and changes bed proportions.
- Second living room: epoch015/0.5 reduces the obvious duplicated sofa arrangement
  present at scale 1.0. Hidden layout still differs from GT.

These observations concern image appearance and visible flat-ERP geometry, not
measured spherical seam quality or certified 3D accuracy. The no-world baseline
cannot uniquely determine the hidden GT room. A nicer baseline does not establish
that correct world conditioning beats shuffled conditioning.

## Implementation

`train.py` is an isolated version of the existing oracle trainer; `inference.py`
starts from the fixed indoor diagnostic entry. Original `oracle_*.py` and old
outputs remain untouched. Core tensorization, WorldBranch, activation storage,
losses, Local RGB conditioning and training schedule are reused.

- Fixed base: completed Panorama epoch015; explicit LoRA scale 0.5.
- New World Encoder/Adapter initialization; no resume from the epoch006 World run.
- Same 1262 training / 168 validation samples and same 160 object slots.
- Same category, local center, local AABB full sizes, validity and padding masks.
- No new room, relation, Observer or Completion fields.
- Same blocks 20/40/59, width 256, LR 1e-4, accumulation 4, warmup 100 and 1000 steps.
- These blocks are an inexpensive baseline, not a literature-validated optimum.
  1000 steps are the first evaluation stage, not a convergence guarantee.
- Sampling now shuffles every complete pass without dropping samples. At 1000
  steps every training example appears 3 or 4 times. Python yaw RNG and PyTorch
  RNG are seeded per absolute optimizer step; World initialization has its own
  fixed seed independent of VAE cache reuse.
- Scale is saved in every World checkpoint, checked on actual LoRA layers, and
  automatically restored by inference. Old checkpoints are rejected by this entry.
- Inference uses native pipeline precision, with BF16 autocast scoped only to the
  World encoder and attention, matching their training precision. The older v1
  scaled-world inference had no World autocast, which changed its numerics versus
  training. Compare correct/shuffled against the new run's own baseline.
- Existing target posterior files are reused through read-only-use symlink paths,
  after checking paired targets/alignment, VAE snapshot, resolution and each file hash.
  Do not delete the old run's `target_posteriors` while the new run depends on it.
  Fresh posterior samples are still drawn at each training microbatch.

## Manual execution

From project root, use `bash qwen_edit_pano/world_pipeline/run_scaled_world.sh`:

1. `bundle` on CPU: new manifest bound to epoch015, unchanged GT and pair records.
2. `preflight` on CPU: complete provenance and reused-posterior hash checks.
3. `smoke` on a GPU: two steps, frozen base, zero-gate identity and gradients.
4. `train` on a GPU: fresh 1000-step run; the smoke checkpoint is not resumed.
5. `eval 000500` and/or `eval 001000` on a GPU: same six indoor images, baseline,
   correct GT and cross-building shuffled GT. Scale is read from the World checkpoint.

6. `eval-train 001000`: fixed eight training images for fit diagnosis only.
7. `eval-val24 001000`: fixed 24 validation images, six from each held-out building;
   this broader set is not indoor-only. This is a diagnostic subset, not all 168.

New training output: `qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_full_v2`.
New bundle: `qwen_edit_pano/data/oracle_supplement_epoch015_scale05_v1`.
Each checkpoint has `world.safetensors`, `optimizer.pt`, `config.json`, `COMPLETE.json`.
Training root has `train_log.jsonl`, `pano_scale_check.json`, memory/parameter reports.
Training does not itself generate panoramas; run `eval` for image output.

For interruption recovery, `train CHECKPOINT NEW_OUTPUT` accepts only this new
experiment's checkpoint and identical configuration/source hashes. To evaluate a
resumed output, call `scaled_world.inference` with its explicit checkpoint path.
`extend CHECKPOINT NEW_OUTPUT TOTAL_STEPS` supports continuing to an absolute
budget such as 2000, without restarting warmup. Only extend after assessing
correct-GT benefit on validation, not because training loss alone declines.
Checkpoints are saved every 250 optimizer steps.

No GPU training or inference was run by the assistant. CPU checks cover runtime
scaling/reload, zero-gate identity, encoder gradients with frozen base, invalid
scales, posterior provenance/corruption and checkpoint binding. Real GPU smoke is
still required to verify this new entry in the user's environment.
