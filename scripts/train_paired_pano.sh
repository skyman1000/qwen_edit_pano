#!/bin/bash
# Submit from the project root after reviewing paired_full_v1 alignment previews.
# Fresh: sbatch qwen_edit_pano/scripts/train_paired_pano.sh
# Resume: sbatch qwen_edit_pano/scripts/train_paired_pano.sh --resume /path/to/checkpoint-epochNNN
# Current debug MaxTime is 3-00:01:00; long has no PartitionTimeLimit override.
#SBATCH -p debug
#SBATCH --nodelist=GPU4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --job-name=qwen-edit-paired
#SBATCH --output=qwen_edit_pano/outputs/qwen-edit-paired_%j.log

set -euo pipefail
# sbatch copies this script, so locate the repository using the submission directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f qwen_edit_pano/paired_train.py ]] || {
  echo "Submit from the DIT360 project root." >&2; exit 2;
}
export PYTHON_BIN="${PYTHON_BIN:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }

RESUME_ARGS=()
if [[ $# -eq 0 ]]; then
  OUTPUT="qwen_edit_pano/outputs/paired_full_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
elif [[ $# -eq 2 && "$1" == --resume ]]; then
  CHECKPOINT="$(realpath -e -- "$2")"
  [[ -f "$CHECKPOINT/COMPLETE.json" ]] || {
    echo "Resume requires a completed epoch checkpoint with COMPLETE.json." >&2; exit 2;
  }
  OUTPUT="$(dirname -- "$CHECKPOINT")"
  RESUME_ARGS=(--resume "$CHECKPOINT")
else
  echo "Usage: sbatch $0 [--resume /path/to/checkpoint-epochNNN]" >&2
  exit 2
fi

echo "Job ${SLURM_JOB_ID:-local}; node=$(hostname); Python=$PYTHON_BIN"
echo "Training output: $OUTPUT"
echo "Paired Local RGB -> ERP; epochs=25; warmup=2350; workers=25"
# paired.sh sets offline mode and invokes paired_train, not the text-only train.py.
# --alignment-reviewed records the prerequisite that paired previews were reviewed.
exec bash qwen_edit_pano/scripts/paired.sh train \
  --manifest qwen_edit_pano/data/paired_full_v1/train.jsonl \
  --heldout-manifest qwen_edit_pano/data/paired_full_v1/heldout.jsonl \
  --condition-cache qwen_edit_pano/cache/paired_full_v1 \
  --output "$OUTPUT" \
  --epochs 25 --workers 25 --warmup-steps 2350 \
  --alignment-reviewed "${RESUME_ARGS[@]}"
