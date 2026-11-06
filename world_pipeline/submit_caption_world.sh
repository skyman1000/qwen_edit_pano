#!/usr/bin/env bash
#SBATCH --job-name=world-caption-fresh
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-caption-fresh_%j.log

set -euo pipefail
caption_world_steps="${1:-2000}"
if [[ $# -gt 1 || ! "$caption_world_steps" =~ ^[1-9][0-9]{0,4}$ ]] || (( caption_world_steps > 10000 )); then
  echo 'Usage: sbatch [Slurm options] submit_caption_world.sh [TARGET_STEPS<=10000]' >&2
  exit 2
fi
exec bash /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/world_pipeline/run_caption_world.sh \
  all --gpus 2 --steps "$caption_world_steps"
