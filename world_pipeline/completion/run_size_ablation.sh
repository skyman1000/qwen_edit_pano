#!/usr/bin/env bash
# Same observed scene and seed; change only the size convention and text arm.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
OBS_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
SAMPLE_ID=X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c
export HF_HOME=/data-nfs/gpu1-2/u13529658780/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_arm() {
  local arm="$1"
  local scene_text="$2"
  srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=32G --qos=normal --time=02:00:00 \
    "$OBS_PY" -u -m qwen_edit_pano.world_pipeline.completion.complete \
    --observed "qwen_edit_pano/outputs/observer3d_indoor_val3_run001/$SAMPLE_ID/observed.json" \
    --weights qwen_edit_pano/third_party/M3DLayout-code/weights/autoregressive_59000.pth \
    --camera-height 1.5 --size-convention half \
    --max-new-objects 32 --max-objects 160 --seeds 0 \
    --text "$scene_text" \
    --output "qwen_edit_pano/outputs/completion_indoor_half${arm}_run001/$SAMPLE_ID"
}

case "${1:-all}" in
  text) run_arm text 'The room is a living room.' ;;
  notext) run_arm notext '' ;;
  all)
    run_arm text 'The room is a living room.'
    run_arm notext ''
    ;;
  *) echo 'Usage: bash run_size_ablation.sh [all|text|notext]'; exit 2 ;;
esac
