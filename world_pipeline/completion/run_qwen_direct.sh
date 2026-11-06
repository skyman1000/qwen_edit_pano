#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
DIRECT_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
DIRECT_RUN=${QWEN_DIRECT_RUN:-qwen_edit_pano/outputs/qwen_direct_offscreen_run001}
DIRECT_OBSERVER=qwen_edit_pano/outputs/qwen_world_simple_run001/observer
DIRECT_BUNDLE=qwen_edit_pano/data/qwen_upstream_val12_v1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/caupano_direct_mpl
case "${1:-}" in
  run)
    srun -p debug --nodelist=GPU1 --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 "$DIRECT_PY" -u \
      -m qwen_edit_pano.world_pipeline.completion.qwen_direct \
      --bundle "$DIRECT_BUNDLE" --observer "$DIRECT_OBSERVER" --output "$DIRECT_RUN/completion" ;;
  review)
    "$DIRECT_PY" -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
      --bundle "$DIRECT_BUNDLE" --observer "$DIRECT_OBSERVER" \
      --completion "$DIRECT_RUN/completion" --output "$DIRECT_RUN/review" ;;
  *) echo 'Usage: bash run_qwen_direct.sh run|review'; exit 2 ;;
esac
