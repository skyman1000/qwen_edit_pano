#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
UPSTREAM_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
UPSTREAM_BUNDLE=${UPSTREAM_BUNDLE:-qwen_edit_pano/data/qwen_upstream_val12_v1}
UPSTREAM_RUN=${UPSTREAM_RUN:-qwen_edit_pano/outputs/qwen_upstream_val12_run003}
UPSTREAM_MODEL=${UPSTREAM_MODEL:-/data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/caupano_upstream_mpl}"
gpu() {
  local node_args=()
  if [[ -n "${UPSTREAM_NODE:-}" ]]; then node_args=(--nodelist="$UPSTREAM_NODE"); fi
  srun -p debug "${node_args[@]}" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 "$UPSTREAM_PY" -u "$@"
}
case "${1:-}" in
  check)
    "$UPSTREAM_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_qwen_upstream -v ;;
  prepare)
    "$UPSTREAM_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream prepare \
      --bundle "$UPSTREAM_BUNDLE" --per-building 3 --selection-seed 0 ;;
  observe)
    gpu -m qwen_edit_pano.world_pipeline.completion.qwen_upstream observe \
      --bundle "$UPSTREAM_BUNDLE" --model "$UPSTREAM_MODEL" --output "$UPSTREAM_RUN/observer" ;;
  complete-gt|complete-pred)
    condition=${1#complete-}
    gpu -m qwen_edit_pano.world_pipeline.completion.qwen_upstream complete \
      --bundle "$UPSTREAM_BUNDLE" --observer "$UPSTREAM_RUN/observer" --model "$UPSTREAM_MODEL" \
      --conditions "$condition" --seeds 0 1 2 --output "$UPSTREAM_RUN/completion_$condition" ;;
  baseline)
    gpu -m qwen_edit_pano.world_pipeline.completion.observe \
      --inputs "$UPSTREAM_BUNDLE/inputs.jsonl" --model "$UPSTREAM_MODEL" \
      --max-new-tokens 8192 --decoding greedy --output "$UPSTREAM_RUN/baseline_observer" ;;
  review)
    baseline_args=()
    if [[ -d "$UPSTREAM_RUN/baseline_observer" ]]; then
      baseline_args=(--baseline-observer "$UPSTREAM_RUN/baseline_observer")
    fi
    for condition in gt pred; do
      "$UPSTREAM_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
        --bundle "$UPSTREAM_BUNDLE" --observer "$UPSTREAM_RUN/observer" \
        --completion "$UPSTREAM_RUN/completion_$condition" \
        --output "$UPSTREAM_RUN/review_$condition" "${baseline_args[@]}"
    done ;;
  *) echo 'Usage: bash run_qwen_upstream.sh check|prepare|observe|complete-gt|complete-pred|review|baseline'; exit 2 ;;
esac
