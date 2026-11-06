#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
VISIBLE_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
VISIBLE_BUNDLE=${VISIBLE_BUNDLE:-qwen_edit_pano/data/m3dlayout_visible_probe_v1}
VISIBLE_RUN=${VISIBLE_RUN:-qwen_edit_pano/outputs/m3dlayout_visible_probe_run001}
VISIBLE_MODEL=${VISIBLE_MODEL:-/data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
gpu() {
  srun -p debug --nodelist="${VISIBLE_NODE:-GPU2}" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 "$VISIBLE_PY" -u "$@"
}
case "${1:-}" in
  check)
    "$VISIBLE_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_visible_probe -v
    ;;
  prepare)
    "$VISIBLE_PY" -u -m qwen_edit_pano.world_pipeline.completion.visible_probe prepare \
      --bundle "$VISIBLE_BUNDLE" --per-building 3 --selection-seed 0
    ;;
  describe)
    gpu -m qwen_edit_pano.world_pipeline.completion.describe \
      --inputs "$VISIBLE_BUNDLE/inputs.jsonl" --model "$VISIBLE_MODEL" --output "$VISIBLE_RUN/captions"
    ;;
  run)
    gpu -m qwen_edit_pano.world_pipeline.completion.visible_probe run \
      --bundle "$VISIBLE_BUNDLE" --captions "$VISIBLE_RUN/captions" \
      --output "$VISIBLE_RUN/comparison" --seeds 0 1 2
    ;;
  *) echo 'Usage: bash run_visible_probe.sh check|prepare|describe|run'; exit 2 ;;
esac
