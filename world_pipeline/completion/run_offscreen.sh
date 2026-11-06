#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
OFFSCREEN_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
OFFSCREEN_RUN=${OFFSCREEN_RUN:-qwen_edit_pano/outputs/qwen_m3d_offscreen_run001}
OFFSCREEN_OBSERVER=${OFFSCREEN_OBSERVER:-qwen_edit_pano/outputs/qwen_world_simple_run001/observer}
OFFSCREEN_BUNDLE=qwen_edit_pano/data/qwen_upstream_val12_v1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/caupano_offscreen_mpl}
case "${1:-}" in
  propose|run)
    srun -p debug --nodelist="${OFFSCREEN_NODE:-GPU2}" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 "$OFFSCREEN_PY" -u \
      -m qwen_edit_pano.world_pipeline.completion.hybrid "$1" --offscreen \
      --bundle "$OFFSCREEN_BUNDLE" --observer "$OFFSCREEN_OBSERVER" --output "$OFFSCREEN_RUN" ;;
  review)
    if [[ -f "$OFFSCREEN_RUN/completion/config.json" ]]; then
      "$OFFSCREEN_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
        --bundle "$OFFSCREEN_BUNDLE" --observer "$OFFSCREEN_OBSERVER" \
        --completion "$OFFSCREEN_RUN/completion" --output "$OFFSCREEN_RUN/review"
    fi
    "$OFFSCREEN_PY" -m qwen_edit_pano.world_pipeline.completion.offscreen_review \
      --bundle "$OFFSCREEN_BUNDLE" --run "$OFFSCREEN_RUN" ;;
  check)
    "$OFFSCREEN_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_hybrid -v ;;
  *) echo 'Usage: bash run_offscreen.sh propose|review|run|check'; exit 2 ;;
esac
