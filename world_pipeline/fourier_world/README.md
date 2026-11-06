# Geometry Fourier oracle v1 — 2026-10-12

## Purpose and evidence

Isolated ablation of scaled_world v2. Keep existing GT fields, Local RGB native
conditioning, epoch015 Panorama LoRA scale 0.5, flow/cube/yaw losses, target-only
World residuals, injection blocks 20/40/59, BF16 scoped runtime and sampling policy.
Only World geometry representation changes. No production GPU jobs run by Codex.

Current baseline already consumes camera-local center and AABB full sizes in meters,
and Qwen has image-token RoPE. What is missing is explicit object-geometry Fourier PE.
This is not sequence-index PE: object order must not convey spatial position.

LACONIC Appendix A.1.2 Eq.5 specifies 32 sinusoidal frequencies 128**(j/31),
j=0..31, per geometric scalar. GLIGEN Appendix B reports a substantial grounding
AP gap between Fourier and MLP geometry despite similar FID. These motivate an
ablation, not a guarantee of improved Qwen ERP output.

Sources:
- https://arxiv.org/html/2507.03257v1
- https://ar5iv.labs.arxiv.org/html/2301.07093
- https://ar5iv.labs.arxiv.org/html/2308.06721

## Exact implementation and differences from the papers

Six scalars: camera center xyz and camera-axis AABB full lengths xyz. Each uses
sin(f*x), cos(f*x), with x expressed in meters divided by a fixed 1m unit. No 2pi,
clipping, per-scene normalization, fitted validation statistics or angular conversion.
Cartesian coordinates avoid introducing longitude wrap discontinuities, but this
alone does not enforce correspondence with ERP token positions.

Raw six scalars + two validity bits are retained, then 384 PE channels appended:
392 -> width256 -> width256. This raw bypass is our adaptation, not literal LACONIC.
It adds 98,304 parameters at width256, so this is not an exactly parameter-matched
ablation. Category/global/attention initialization and surrounding RNG stream match
the baseline for the same seed. Geometry projection itself is newly initialized.

Unknown center/size: mask both raw and sin/cos features to zero, keep validity bits.
Known zero coordinate still has cos=1. Trigonometry runs FP32 even under autocast.
Frequency buffer is saved in safetensors; exact encoding metadata/version is checked
at inference. Old MLP checkpoints are rejected. Train fresh World Encoder+Adapter;
do not resume baseline or two-step smoke into the formal experiment.

We do not add LACONIC's object self-attention, pretrained semantic embedding, OBB
rotation or floor geometry in this ablation. Its DDPM objective is not copied into
Qwen flow matching. Keep our effective batch4, LR1e-4, warmup100, constant schedule
and initial1000 steps to isolate encoding effects. These are a baseline budget,
not the paper's training configuration or a convergence claim. Highest frequency
has period about 4.9cm; later predicted/noisy geometry may need lower frequencies
or noise robustness training. This version tests clean GT oracle conditions only.

## Run manually, from project root

Use bash qwen_edit_pano/world_pipeline/run_fourier_world.sh ACTION under srun.
No new environment variables. Script selects the existing qwen360 Python.

1. bundle: binds existing GT to epoch015 and creates fixed evaluation selections.
   Reuses bundle if present; does not rebuild GT or download datasets.
2. preflight: CPU provenance/data/cache checks. Expect train1262 / val168.
3. smoke: two optimizer steps, new directory, frozen-base/gradient/save checks.
4. eval-smoke: reload smoke checkpoint for one sample, baseline/correct/shuffled.
   Only an interface test, not a quality result.
5. train: fresh1000 steps, saves every250. Does not resume smoke.
6. eval 000500 and eval 001000: same indoor6, all three modes, fixed seeds.
7. eval-train 001000 and eval-val24 001000: fit/generalization diagnosis.

Suggested resources: bundle/preflight CPU4 RAM16G1h; smoke GPU1 RAM96G2h;
eval-smoke GPU1 RAM64G2h; train GPU1 RAM96G24h; eval GPU1 RAM64G8h.
These are allocations, not execution-time guarantees; request a GPU with capacity
comparable to the previous successful run. Slurm --mem is CPU RAM, not GPU VRAM.

Formal output:
qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1
Evaluation:
qwen_edit_pano/outputs/oracle_epoch015_scale05_indoor6_step001000_fourier_v1

To resume an interrupted Fourier run, use:
  train CHECKPOINT NEW_OUTPUT
To extend a completed Fourier run, use:
  extend CHECKPOINT NEW_OUTPUT TOTAL_STEPS
Default eval actions read the default formal output; for custom resume/extend outputs,
use fourier_world.inference --checkpoint ... with the same selection and parameters.

Existing baseline comparison uses run_scaled_world.sh. Compare equal checkpoints,
data, Local RGB, seeds and Panorama scale. Correct vs shuffled is a condition
sensitivity test, not proof of generalization. Inspect object direction/presence,
Local fidelity and seams. If only texture changes, do not blindly extend training.

CPU verification:
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m unittest \
  qwen_edit_pano.world_pipeline.fourier_world.test_runtime -v

Tests cover PE math/missingness/FP32, old-checkpoint rejection, unchanged adapter
initialization, permutation invariance, zero-gate identity, tiny real Edit backward
with checkpointing and activation offload, frozen backbone, state reload and sampling.
