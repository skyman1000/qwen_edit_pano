#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
PANO_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
TEXT_RUN=${TEXT_RUN:-qwen_edit_pano/outputs/pretrained_text_val3_run001}
OBS_ROOT=${OBS_ROOT:-qwen_edit_pano/outputs/observer3d_indoor_val3_run001}
INPUTS=qwen_edit_pano/data/observer3d_indoor_val3_inputs.jsonl
QWEN_LOCAL=/data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
gpu() {
  srun -p debug --nodelist="${TEXT_NODE:-GPU2}" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 "$PANO_PY" -u "$@"
}
case "${1:-}" in
  describe)
    gpu -m qwen_edit_pano.world_pipeline.completion.describe \
      --inputs "$INPUTS" --model "$QWEN_LOCAL" --output "$TEXT_RUN/captions"
    ;;
  compare)
    gpu -m qwen_edit_pano.world_pipeline.completion.compare_pretrained \
      --inputs "$INPUTS" --observations "$OBS_ROOT" --captions "$TEXT_RUN/captions" \
      --output "$TEXT_RUN/comparison" --seeds 0 1 2 \
      --size-convention "${M3D_SIZE_CONVENTION:-full}" --camera-height 1.5
    ;;
  *) echo 'Usage: bash run_pretrained_text.sh describe|compare'; exit 2 ;;
esac
