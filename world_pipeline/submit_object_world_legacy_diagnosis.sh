#!/usr/bin/env bash
#SBATCH --job-name=world-legacy-diag
#SBATCH --partition=debug
#SBATCH --nodelist=GPU1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-legacy-diag_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
diagnostic_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
diagnostic_output="${1:-qwen_edit_pano/outputs/object_world_legacy_diagnosis_v1}"
"$diagnostic_python" -u -m qwen_edit_pano.world_pipeline.object_world.diagnose \
  --run qwen_edit_pano/outputs/object_world_layer_pilot_legacy_v1 \
  --output "$diagnostic_output" --samples 2 --steps 250 500 --sigmas .2 .5 .8
