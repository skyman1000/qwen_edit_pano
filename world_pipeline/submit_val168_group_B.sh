#!/usr/bin/env bash
#SBATCH --job-name=val168-B-correct
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
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/val168-B_%j.log

set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
exec bash qwen_edit_pano/world_pipeline/run_native_gt_val168.sh generate --arm correct
