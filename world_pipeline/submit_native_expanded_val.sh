#!/usr/bin/env bash
#SBATCH --job-name=caupano-native-val
#SBATCH --partition=debug
#SBATCH --nodelist=GPU1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --array=0-5%1
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/oracle_epoch015_scale05_indoor6_step000750_fourier_v1/expanded_val_v1/native_edit_v1/logs/native_%A_%a.log

set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
exec /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u \
  -m research.world_evaluation.native_expanded_val run \
  --task "${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
