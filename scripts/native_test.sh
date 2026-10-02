#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=GPU3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --job-name=qwen-edit-native
#SBATCH --output=qwen_edit_pano/outputs/qwen-edit-native_%j.log
set -eo pipefail
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate qwen360
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f qwen_edit_pano/native_test.py ]]; do
  [[ "$PWD" != / ]] || { echo 'Submit from the DIT360 project' >&2; exit 2; }
  cd ..
done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
MODEL="${QWEN_MODEL:-Qwen/Qwen-Image-Edit-2511}"
OUTPUT="qwen_edit_pano/outputs/native_edit_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
echo "Model: $MODEL; output: $OUTPUT"
python -u -m qwen_edit_pano.native_test \
  --model "$MODEL" --output "$OUTPUT" \
  --height 1024 --width 2048 --steps 28 --true-cfg-scale 4 \
  --seed 0 --offload model "$@"
