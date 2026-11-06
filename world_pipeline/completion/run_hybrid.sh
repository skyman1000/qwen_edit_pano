#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
HYBRID_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
HYBRID_RUN=${HYBRID_RUN:-qwen_edit_pano/outputs/qwen_m3d_hybrid_run001}
HYBRID_OBSERVER=${HYBRID_OBSERVER:-qwen_edit_pano/outputs/qwen_world_simple_run001/observer}
HYBRID_BUNDLE=qwen_edit_pano/data/qwen_upstream_val12_v1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/caupano_hybrid_mpl}
case "${1:-}" in
  propose|run)
    srun -p debug --nodelist="${HYBRID_NODE:-GPU2}" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 "$HYBRID_PY" -u \
      -m qwen_edit_pano.world_pipeline.completion.hybrid "$1" \
      --bundle "$HYBRID_BUNDLE" --observer "$HYBRID_OBSERVER" --output "$HYBRID_RUN" ;;
  review)
    "$HYBRID_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
      --bundle "$HYBRID_BUNDLE" --observer "$HYBRID_OBSERVER" \
      --completion "$HYBRID_RUN/completion" --output "$HYBRID_RUN/review" ;;
  check)
    "$HYBRID_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_hybrid -v ;;
  *) echo 'Usage: bash run_hybrid.sh propose|run|review|check'; exit 2 ;;
esac
