#!/usr/bin/env bash
#SBATCH --job-name=world-layer-pilot
#SBATCH --partition=debug
#SBATCH --nodelist=GPU3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-layer-pilot_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
pilot_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
case "${1:-}" in
  legacy) pilot_blocks=(20 40 59);;
  spread) pilot_blocks=(9 29 49);;
  *) echo 'Usage: sbatch submit_object_world_layer_pilot.sh legacy|spread' >&2; exit 2;;
esac
pilot_output="qwen_edit_pano/outputs/object_world_layer_pilot_${1}_v1"
pilot_panel=research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json
[[ -f "$pilot_panel" ]] || { echo "Missing fixed six-image panel: $pilot_panel" >&2; exit 2; }
[[ ! -e "$pilot_output" && ! -e "${pilot_output}_val6" ]] || {
  echo "Pilot output already exists: $pilot_output" >&2; exit 2;
}
# Matched exposure, initialization seed, optimizer and loss. A screening run,
# not an automatic launch of the prior 10000-step chain.
"$pilot_python" -u -m qwen_edit_pano.world_pipeline.object_world.train \
  --steps 500 --warmup-steps 100 --save-every 250 --seed 0 \
  --blocks "${pilot_blocks[@]}" --output "$pilot_output"
"$pilot_python" -u -m qwen_edit_pano.world_pipeline.object_world.inference \
  --checkpoint "$pilot_output/checkpoint-step000500" \
  --output "${pilot_output}_val6" --panel "$pilot_panel" --limit 6 --steps 28 --seed 0 \
  --modes baseline correct shuffled
