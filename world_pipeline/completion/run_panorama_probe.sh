#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
PROBE_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
PROBE_RUN=qwen_edit_pano/outputs/panorama_qwen_probe_v1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:-}" in
  check)
    "$PROBE_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_panorama_probe -v
    ;;
  replay-plan)
    "$PROBE_PY" -u -m qwen_edit_pano.world_pipeline.completion.panorama_probe \
      replay-plan --source "$PROBE_RUN/plan" --output "$PROBE_RUN/plan_review"
    ;;
  controls|plan|complete)
    srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
      "$PROBE_PY" -u -m qwen_edit_pano.world_pipeline.completion.panorama_probe \
      "$1" --output "$PROBE_RUN/$1"
    ;;
  *)
    echo 'Usage: bash run_panorama_probe.sh check|controls|plan|complete|replay-plan'
    exit 2
    ;;
esac
