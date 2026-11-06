#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
SIMPLE_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
SIMPLE_BUNDLE=qwen_edit_pano/data/qwen_upstream_val12_v1
SIMPLE_RUN=${SIMPLE_WORLD_RUN:-qwen_edit_pano/outputs/qwen_world_simple_run001}
SIMPLE_OBSERVER=${SIMPLE_WORLD_OBSERVER:-$SIMPLE_RUN/observer}
SIMPLE_SOURCE=${SIMPLE_WORLD_SOURCE:-qwen_edit_pano/outputs/qwen_upstream_val12_run003/observer}
SIMPLE_MODEL=/data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
gpu() {
  local nodes=()
  if [[ -n "${SIMPLE_WORLD_NODE:-GPU2}" ]]; then nodes=(--nodelist="${SIMPLE_WORLD_NODE:-GPU2}"); fi
  srun -p debug "${nodes[@]}" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 "$SIMPLE_PY" -u "$@"
}
case "${1:-}" in
  check)
    "$SIMPLE_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_qwen_upstream -v ;;
  observe|observe-marked)
    mark_args=()
    if [[ "$1" == observe-marked ]]; then mark_args=(--mark-target); fi
    gpu -m qwen_edit_pano.world_pipeline.completion.qwen_upstream observe \
      --bundle "$SIMPLE_BUNDLE" --model "$SIMPLE_MODEL" --simple \
      --reuse-instances "$SIMPLE_SOURCE" "${mark_args[@]}" --output "$SIMPLE_RUN/observer" ;;
  complete|complete-gt)
    condition=pred
    if [[ "$1" == complete-gt ]]; then condition=gt; fi
    gpu -m qwen_edit_pano.world_pipeline.completion.qwen_upstream complete \
      --bundle "$SIMPLE_BUNDLE" --model "$SIMPLE_MODEL" --simple \
      --observer "$SIMPLE_OBSERVER" --conditions "$condition" --seeds 0 \
      --max-hidden 12 --max-new-tokens 4096 --output "$SIMPLE_RUN/completion_$condition" ;;
  review)
    "$SIMPLE_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
      --bundle "$SIMPLE_BUNDLE" --observer "$SIMPLE_OBSERVER" \
      --completion "$SIMPLE_RUN/completion_pred" --output "$SIMPLE_RUN/review"
    if [[ -f "$SIMPLE_RUN/completion_gt/config.json" ]]; then
      "$SIMPLE_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
        --bundle "$SIMPLE_BUNDLE" --observer "$SIMPLE_OBSERVER" \
        --completion "$SIMPLE_RUN/completion_gt" --output "$SIMPLE_RUN/review_gt"
    fi ;;
  *) echo 'Usage: bash run_simple_world.sh check|observe|complete|review|complete-gt'; exit 2 ;;
esac
