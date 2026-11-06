#!/usr/bin/env bash
#SBATCH --job-name=world-bind-refine
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-bind-refine_%j.log
#SBATCH --open-mode=append
set -euo pipefail
bind_arm="${1:-}"
case "$bind_arm" in control|binding) ;; *) echo 'Usage: sbatch [Slurm options] submit_binding_refine.sh control|binding [ADDITIONAL_STEPS]' >&2; exit 2;; esac
bind_steps="${2:-1000}"
exec bash /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/world_pipeline/run_binding_refine.sh \
  "$bind_arm" --additional-steps "$bind_steps"
