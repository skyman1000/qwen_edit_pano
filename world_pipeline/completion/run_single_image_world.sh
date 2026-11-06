#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
SINGLE_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:-}" in
  check)
    "$SINGLE_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_single_image_world \
      qwen_edit_pano.world_pipeline.completion.test_single_image_plan \
      qwen_edit_pano.world_pipeline.completion.test_json_generation -v
    ;;
  preflight)
    shift
    "$SINGLE_PY" -m qwen_edit_pano.world_pipeline.completion.single_image_world --preflight "$@"
    ;;
  run)
    shift
    srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
      "$SINGLE_PY" -u -m qwen_edit_pano.world_pipeline.completion.single_image_world "$@"
    ;;
  observe-preflight)
    shift
    "$SINGLE_PY" -m qwen_edit_pano.world_pipeline.completion.single_image_world --observe-only --preflight "$@"
    ;;
  observe)
    shift
    srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
      "$SINGLE_PY" -u -m qwen_edit_pano.world_pipeline.completion.single_image_world --observe-only "$@"
    ;;
  plan-preflight)
    shift
    "$SINGLE_PY" -m qwen_edit_pano.world_pipeline.completion.single_image_plan --preflight "$@"
    ;;
  plan)
    shift
    srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
      "$SINGLE_PY" -u -m qwen_edit_pano.world_pipeline.completion.single_image_plan "$@"
    ;;
  *)
    echo 'Usage: bash run_single_image_world.sh check|preflight|run|observe-preflight|observe|plan-preflight|plan [options]'
    exit 2
    ;;
esac
