#!/usr/bin/env bash
#SBATCH --job-name=world-id-ab
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-id-ab_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
ab_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
ab_group="${1:-}"
ab_tag="${2:-v1}"
case "$ab_group" in A|B) ;; *) echo 'Usage: sbatch [Slurm options] submit_object_world_id_ab.sh A|B [NEW_TAG]' >&2; exit 2;; esac
[[ "$ab_tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a plain output tag, without paths.' >&2; exit 2; }
ab_output="qwen_edit_pano/outputs/object_world_id_ab_${ab_group}_${ab_tag}"
ab_panel=research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json
ab_baseline=qwen_edit_pano/outputs/object_world_layer_pilot_legacy_v1_val6
for ab_path in "$ab_output" "${ab_output}_smoke" "${ab_output}_val6"; do
  [[ ! -e "$ab_path" ]] || { echo "Output exists: $ab_path; use a new tag, do not overwrite." >&2; exit 2; }
done
[[ -f "$ab_panel" && -f "$ab_baseline/COMPLETE.json" ]] || { echo 'Missing fixed panel or completed baseline reference.' >&2; exit 2; }
# Independent jobs: one GPU per arm, no dependency or shared mutable training files.
# The smoke process exits before fresh training, releasing its CUDA allocation.
"$ab_python" -u -m qwen_edit_pano.world_pipeline.id_binding_world.train \
  --group "$ab_group" --smoke --seed 0 --output "${ab_output}_smoke"
[[ -f "${ab_output}_smoke/checkpoint-step000002/SMOKE_COMPLETE.json" ]] || { echo 'GPU smoke did not pass.' >&2; exit 1; }
"$ab_python" -u -m qwen_edit_pano.world_pipeline.id_binding_world.train \
  --group "$ab_group" --steps 2000 --warmup-steps 100 --binding-ramp-steps 200 \
  --accumulation 4 --lr 1e-4 --region-weight .25 --blocks 20 40 59 \
  --save-every 500 --seed 0 --activation-storage cpu --output "$ab_output"
"$ab_python" -u -m qwen_edit_pano.world_pipeline.id_binding_world.inference \
  --checkpoint "$ab_output/checkpoint-step002000" --output "${ab_output}_val6" \
  --panel "$ab_panel" --baseline-reference "$ab_baseline" --limit 6 --steps 28 --seed 0 \
  --modes baseline correct shuffled
