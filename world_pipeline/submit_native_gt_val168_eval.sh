#!/usr/bin/env bash
#SBATCH --job-name=caupano-val168-eval
#SBATCH --partition=debug
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/caupano-val168-eval_%j.log

set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/run_native_gt_val168.sh manifests
exec bash qwen_edit_pano/world_pipeline/run_native_gt_val168.sh evaluate
