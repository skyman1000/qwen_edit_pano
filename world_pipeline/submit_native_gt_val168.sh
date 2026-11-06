#!/usr/bin/env bash
#SBATCH --job-name=caupano-val168-gen
#SBATCH --partition=debug
#SBATCH --nodelist=GPU3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/caupano-val168-gen_%j.log

set -euo pipefail
case "${1:-}" in native|correct) ;; *) echo 'Usage: sbatch SCRIPT native|correct' >&2; exit 2;; esac
cd /data-nfs/gpu1-2/u13529658780/DIT360
exec bash qwen_edit_pano/world_pipeline/run_native_gt_val168.sh generate --arm "$1"
