#!/usr/bin/env bash
# User-run entry. This script does not submit jobs or download weights.
set -euo pipefail
cd "$(dirname "$0")/../.."
object_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
object_output=qwen_edit_pano/outputs/object_world_polished_semantic_region_v1
action="${1:-check}"
case "$action" in
  check)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.prepare --preflight
    ;;
  prepare)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.prepare
    ;;
  cache-smoke)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.cache --smoke
    ;;
  cache)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.cache
    ;;
  check-cache)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.cache --check-only
    ;;
  smoke)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.train \
      --smoke --output "${object_output}_smoke"
    ;;
  train)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.train \
      --steps 10000 --warmup-steps 500 --save-every 500 --output "$object_output"
    ;;
  eval-smoke)
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.inference \
      --checkpoint "${object_output}_smoke/checkpoint-step000002" \
      --output "${object_output}_smoke_generation" --limit 2
    ;;
  eval)
    [[ $# -eq 3 ]] || { echo 'Usage: eval CHECKPOINT NEW_OUTPUT' >&2; exit 2; }
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.inference \
      --checkpoint "$2" --output "$3" --limit 6 \
      --panel research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json
    ;;
  extend)
    [[ $# -eq 4 ]] || { echo 'Usage: extend CHECKPOINT NEW_OUTPUT TOTAL_STEPS' >&2; exit 2; }
    "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.train \
      --resume "$2" --output "$3" --steps "$4" --warmup-steps 500 --save-every 500
    ;;
  *) echo 'Usage: check|prepare|cache-smoke|smoke|eval-smoke|cache|check-cache|train|eval CHECKPOINT NEW_OUTPUT|extend CHECKPOINT NEW_OUTPUT TOTAL_STEPS' >&2; exit 2;;
esac
