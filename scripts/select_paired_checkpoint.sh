#!/bin/bash
# Two independent single-GPU jobs, three settings per GPU. No DDP/model sharding.
#SBATCH -p debug
#SBATCH --nodelist=GPU3
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --job-name=pano-select
#SBATCH --output=qwen_edit_pano/outputs/pano-select_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f qwen_edit_pano/paired_inference.py ]] || {
  echo 'Submit from the DIT360 project root.' >&2; exit 2;
}
export PYTHON_BIN="${PYTHON_BIN:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUTPUT="qwen_edit_pano/outputs/checkpoint_selection_${SLURM_ARRAY_JOB_ID}"
echo "Array ${SLURM_ARRAY_JOB_ID}; worker=${SLURM_ARRAY_TASK_ID}; node=$(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output: $OUTPUT"
exec "$PYTHON_BIN" qwen_edit_pano/scripts/select_paired_checkpoint.py \
  --worker "$SLURM_ARRAY_TASK_ID" --output "$OUTPUT"
