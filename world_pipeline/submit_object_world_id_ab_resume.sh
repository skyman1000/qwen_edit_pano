#!/usr/bin/env bash
#SBATCH --job-name=world-id-resume
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-id-ab-resume_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
ab_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
ab_group="${1:-}"
ab_tag="${2:-v1_resume1000}"
ab_steps="${4:-2000}"
case "$ab_group" in A|B) ;; *) echo 'Usage: sbatch [Slurm options] submit_object_world_id_ab_resume.sh A|B [NEW_TAG] [CHECKPOINT] [TOTAL_STEPS]' >&2; exit 2;; esac
[[ "$ab_tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a plain new output tag.' >&2; exit 2; }
[[ "$ab_steps" =~ ^[1-9][0-9]{0,5}$ ]] || { echo 'TOTAL_STEPS must be a positive integer, at most six digits.' >&2; exit 2; }
printf -v ab_final_checkpoint 'checkpoint-step%06d' "$ab_steps"
ab_checkpoint="${3:-qwen_edit_pano/outputs/object_world_id_ab_${ab_group}_v1/checkpoint-step001000}"
ab_output="qwen_edit_pano/outputs/object_world_id_ab_${ab_group}_${ab_tag}"
ab_panel=research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json
ab_baseline=qwen_edit_pano/outputs/object_world_layer_pilot_legacy_v1_val6
for ab_path in "$ab_output" "${ab_output}_val6"; do
  [[ ! -e "$ab_path" ]] || { echo "Output exists: $ab_path; use a new tag." >&2; exit 2; }
done
for ab_file in COMPLETE.json config.json world.safetensors optimizer.pt; do
  [[ -f "$ab_checkpoint/$ab_file" ]] || { echo "Incomplete resume checkpoint: $ab_checkpoint/$ab_file" >&2; exit 2; }
done
[[ -f "$ab_panel" && -f "$ab_baseline/COMPLETE.json" ]] || { echo 'Missing fixed panel or baseline reference.' >&2; exit 2; }

# Restore World + Adam. TOTAL_STEPS is the final absolute step, not an additional
# step count. The default remains 2000; pass 1500 to finish the short comparison.
# The trainer validates the complete resume config.
# Do not run smoke or initialize a new World training run before resuming.
"$ab_python" -u -m qwen_edit_pano.world_pipeline.id_binding_world.train \
  --group "$ab_group" --resume "$ab_checkpoint" \
  --steps "$ab_steps" --warmup-steps 100 --binding-ramp-steps 200 \
  --accumulation 4 --lr 1e-4 --region-weight .25 --blocks 20 40 59 \
  --save-every 500 --seed 0 --activation-storage cpu --output "$ab_output"
[[ -f "$ab_output/$ab_final_checkpoint/COMPLETE.json" ]] || { echo "Missing completed step $ab_steps; refusing generation." >&2; exit 1; }
"$ab_python" -u -m qwen_edit_pano.world_pipeline.id_binding_world.inference \
  --checkpoint "$ab_output/$ab_final_checkpoint" --output "${ab_output}_val6" \
  --panel "$ab_panel" --baseline-reference "$ab_baseline" --limit 6 --steps 28 --seed 0 \
  --modes baseline correct shuffled
