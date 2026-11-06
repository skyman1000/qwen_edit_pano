#!/usr/bin/env bash
# Manual diagnostic run, using the same inference settings as the previous test.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
OBS_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
export HF_HOME=/data-nfs/gpu1-2/u13529658780/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
case "${1:-}" in
  observe)
    srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
      "$OBS_PY" -u -m qwen_edit_pano.world_pipeline.completion.observe \
      --inputs qwen_edit_pano/data/observer3d_indoor_val3_inputs.jsonl \
      --model /data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
      --decoding checkpoint --max-per-category 8 --angle-unit normalized --seed 0 \
      --output qwen_edit_pano/outputs/observer3d_indoor_val3_run001
    ;;
  complete)
    for SAMPLE_ID in \
      8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39 \
      X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c \
      x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6
    do
      srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=32G --qos=normal --time=02:00:00 \
        "$OBS_PY" -u -m qwen_edit_pano.world_pipeline.completion.complete \
        --observed "qwen_edit_pano/outputs/observer3d_indoor_val3_run001/$SAMPLE_ID/observed.json" \
        --weights qwen_edit_pano/third_party/M3DLayout-code/weights/autoregressive_59000.pth \
        --camera-height 1.5 --size-convention full \
        --max-new-objects 32 --max-objects 160 --seeds 0 \
        --output "qwen_edit_pano/outputs/completion_indoor_val3_run001/$SAMPLE_ID"
    done
    ;;
  *) echo 'Usage: bash run_indoor_val3.sh observe|complete'; exit 2 ;;
esac
