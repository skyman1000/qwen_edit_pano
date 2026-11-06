#!/bin/bash
#SBATCH -p debug
#SBATCH --array=0-2%2
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=pano-GT923-eval
#SBATCH --output=qwen_edit_pano/outputs/pano-GT923-eval_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
ARMS=(native finetuned gt_state)
exec "$PYTHON_BIN" -u -m qwen_edit_pano.benchmark_gt evaluate --arm "${ARMS[${SLURM_ARRAY_TASK_ID:?Submit with sbatch}]}" "$@"
