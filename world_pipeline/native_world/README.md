# Native-projection layout adapter

2026-11-04. Version `native_projection_semantic_layout_dual_v1`.

This is a NEW Encoder/Adapter experiment, not a continuation of A10000 or
binding10000. The original Qwen weights AND existing Panorama epoch015 LoRA
remain frozen, with Panorama strength 0.5. New trainable modules start afresh,
Adam starts at zero, and the formal run never inherits the smoke's trained state.
Pretrained semantic anchors/projections are reused; it is not all-random training.

## What changes

- Closed-set category IDs still index the exact existing 37-class name table.
  Reuse the existing 42x3584 raw frozen Qwen name features. Apply the ACTUAL frozen
  native `txt_norm` and `txt_in` to obtain 3072d anchors. Names use no GT caption,
  image-derived scene description, or new LLM training/download/cache preparation.
- A zero-output rank32 semantic residual and Fourier-geometry MLP produce object
  tokens. Geometry input fields/frame and validity rules stay unchanged. The
  geometry output starts at zero; binding provides initial geometry gradients.
- At blocks20/40/59 a DECOUPLED layout attention pass reuses the actual frozen
  image Query, text K/V maps, Q/K normalization, and image output projection
  (including the original frozen Panorama LoRA at scale0.5). K/V have trainable
  zero-initialized rank16 residuals. No new independent random Query/output map.
- Match the block's text LayerNorm and timestep modulation before object K/V.
  The side output enters the native image attention output, then follows the
  block's native image modulation gate and MLP. There is no old external
  target-RMS multiplier. Only target queries get the DIRECT side residual;
  native reference/text outputs are untouched by this addition. Indirect effects
  through subsequent native joint attention are possible, as in other adapters.
- The SIDE pass uses BOTH Q and K BEFORE RoPE. Objects are an unordered set, not
  native text positions; no arbitrary object-order RoPE is assigned. The ORIGINAL
  native attention pass retains its exact RoPE, masks, token count and
  target/reference boundaries. This is not explicit ERP spatial routing or bias,
  and does not guarantee geometry control. Geometry is still learned conditioning.
- Retain slot0 as a learned global, initialized from the projected generic
  class39 vector; actual instances retain their own slots. No object-order PE.

For width3072, ranks16/32, geometry width256 and three layers, there are
1,679,619 trainable parameters. The 42x3072 semantic table is a frozen buffer.
Frozen native projection matrices are REFERENCED, not copied into the optimizer
or checkpoint. Fewer trainable parameters do not prove smaller peak activations.

Pooled name vectors are lexical anchors, not claimed to be identical to complete
native contextual prompts or text-stream representations after many blocks.
This interface is a testable hypothesis, not a faithful LACONIC reproduction or
a guaranteed improvement over the existing models.

## What stays fixed

Single original face2 Local + camera-local metric object state -> aligned
polished ERP. Existing 7390 train / 923 val building-disjoint split and fixed
panorama instruction; original height1024 and circular padding1. No GT masks
enter generation, no GT target caption, no hard projected-box routing, no new
object detector/classification loss, no replacement/deletion training pairs.

`flow + .5*cube + .5*yaw + .25*region + .01*ramp*positive_CE + .001*ramp*reliable_other_BCE`

Binding remains at block40, ramp200, original supported-instance filtering,
eight positive/eight reliable-other queries per eligible instance. It reads
the ACTUAL side-pass Q/K, with global slot0 included in the denominator.
Labels do not modify the forward prediction. Positive/negative support is not a
mask on forward attention. Current binding and region weights are unchanged.

Global batch4 = two replicas x microbatch1 x accumulation2 per rank. FP32 World
masters/Adam, BF16 autocast with cast cache disabled; CPU saved-activation storage
and pure non-reentrant checkpointing. LR1e-4, warmup100, AdamW(.9,.999), eps1e-6,
weight decay1e-5, clip1 after rank mean, saves500. Total10000 = 5.41 data passes.

## User-run command

```bash
sbatch --nodelist=GPUX --qos=xx qwen_edit_pano/world_pipeline/submit_native_world_dual.sh
```

One node with TWO GPUs trains ONE model. The launcher first checks the existing
data, then runs an isolated two-step dual-GPU smoke. It starts formal training
from fresh initialization only if BOTH ranks pass; no automatic evaluation.
Smoke includes the actual 215-object maximum case. It checks zero-gate equality,
label-independent forward, key/semantic/geometry gradients at zero gates,
generation gradients reaching Value deltas after the first update, frozen base,
and equal replica parameters. This is a functionality/memory check, not quality.

Default output: `qwen_edit_pano/outputs/object_world_native_dual_v1/`.
Log: `world-native_JOBID.log`. Complete checkpoints save World, Adam and config;
COMPLETE is written last. Repeating the SAME command resumes only complete own
formal checkpoints toward total10000, never adds another10000. Time-limit
signals request a complete-step save and attempt Slurm requeue. If the cluster
does not requeue, resubmit the same command. Partial saves are ignored. No old
data/weights/results are overwritten. GPUs/downloads are user-run only.

Optional separate short smoke (no formal training):

```bash
srun -p debug --nodelist=GPUX --qos=xx --gres=gpu:2 --cpus-per-task=8 --mem=128G --time=01:00:00 \
 /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m qwen_edit_pano.world_pipeline.native_world.run --stage smoke
```

## Checks and limits

Actual cache/provenance check passed: all7390 train condition/posterior/region/
instance paths exist, names42x3584 finite. No hashing or ZIP CRC scan.
Eight CPU regressions pass using real tiny Qwen-Edit blocks, Panorama LoRA,
circular padding, BF16, pure checkpoint replay, CPU activation storage, explicit
two-rank gradient-average equivalence, strict own Adam restore and driver smoke/
formal isolation. No actual CUDA/NCCL training has been performed by the assistant.
48G GPU peak and training speed remain to be measured by the user's smoke.

An eight-instance CPU visual review is in
`research/audits/native_world_2026-11-04/supervision/`. Most outlines roughly
match visible objects; TV246 in scan5q7pvUzZiYa is visually ambiguous on a bathroom
reflective surface. The original Local has the same appearance and the official
house record already says `tv -> mpcat40 22`; it is not evidence of a new numeric
mapping error or a polished-only replacement. No automatic relabeling/filtering
was done. Eight selected objects do not certify the entire polished dataset.

Acceptance must inspect natural GT-state category/direction/count adherence,
Local preservation and image quality, with training/validation reported
separately. Attention probabilities, RGB edit magnitudes and low reconstruction
loss cannot substitute for generation success. No promise that this interface
alone resolves ambiguous source labels, occlusion, missing room structure or
precise placement. Preserve A10000 and binding10000 as comparisons.
