#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=GPU4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --job-name=qwen-edit-smoke
#SBATCH --output=qwen_edit_pano/outputs/qwen-edit-smoke_%j.log
set -eo pipefail
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate qwen360
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f qwen_edit_pano/train.py ]]; do
  [[ "$PWD" != / ]] || exit 2
  cd ..
done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
MODEL="${QWEN_MODEL:-Qwen/Qwen-Image-Edit-2511}"
OUTPUT="qwen_edit_pano/outputs/smoke_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
python -u -m qwen_edit_pano.train \
  --profile repo-panorama --official-full-training-only --smoke-test \
  --model "$MODEL" \
  --panorama-manifest qwen_pano/data/cached_official_full/train.jsonl \
  --text-cache qwen_edit_pano/cache/text_edit2511_official_full \
  --output "$OUTPUT" "$@"
# A fresh process reloads the exported Diffusers adapter and generates one ERP.
python -u -m qwen_edit_pano.inference \
  --model "$MODEL" --lora "$OUTPUT/smoke-checkpoint" \
  --output "$OUTPUT/inference" --height 1024 --width 2048 \
  --steps 28 --true-cfg-scale 4 --seed 0 --seed-mode per-id \
  --prompt-mode verbatim --offload model --limit 1
python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
from PIL import Image
root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root/'inference/generated.jsonl').read_text().splitlines()]
assert len(rows) == 1
with Image.open(root/'inference'/rows[0]['image']) as image:
    assert image.size == (2048, 1024)
report = json.loads((root/'smoke-checkpoint/SMOKE_COMPLETE.json').read_text())
report['inference'] = 'passed: reloaded exported LoRA, generated 2048x1024 ERP'
(root/'VALIDATION_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
print('Full smoke validation passed:', root/'VALIDATION_COMPLETE.json')
PY
