#!/usr/bin/env bash
#SBATCH --job-name=world-id-eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-id-ab-eval_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
ab_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
ab_group="${1:-}"
case "$ab_group" in A|B) ;; *) echo 'Usage: sbatch [Slurm options] submit_object_world_id_ab_eval.sh A|B [CHECKPOINT] [NEW_TAG]' >&2; exit 2;; esac
ab_checkpoint="${2:-qwen_edit_pano/outputs/object_world_id_ab_${ab_group}_v1/checkpoint-step001000}"
ab_tag="${3:-step1000}"
[[ "$ab_tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a plain output tag.' >&2; exit 2; }
ab_output="qwen_edit_pano/outputs/object_world_id_ab_${ab_group}_${ab_tag}_val6"
[[ ! -e "$ab_output" ]] || { echo "Output exists: $ab_output; use a new tag." >&2; exit 2; }
# Evaluation only: no training, optimizer restoration, new data cache or download.
"$ab_python" -u -m qwen_edit_pano.world_pipeline.id_binding_world.inference \
  --checkpoint "$ab_checkpoint" --output "$ab_output" \
  --panel research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json \
  --baseline-reference qwen_edit_pano/outputs/object_world_layer_pilot_legacy_v1_val6 \
  --limit 6 --steps 28 --seed 0 --modes baseline correct shuffled
