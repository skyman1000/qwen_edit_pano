#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
BATCH_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
action="${1:-}"
if (( $# )); then shift; fi
case "$action" in
  check)
    "$BATCH_PY" -m unittest \
      qwen_edit_pano.world_pipeline.completion.test_batch_world \
      qwen_edit_pano.world_pipeline.completion.test_protocol_bridge \
      qwen_edit_pano.world_pipeline.completion.test_box_observer \
      qwen_edit_pano.world_pipeline.completion.test_placement_program -v
    ;;
  preflight|review)
    "$BATCH_PY" -u -m qwen_edit_pano.world_pipeline.completion.batch_world "$action" "$@"
    ;;
  run)
    nodes=(); batch_args=()
    while (( $# )); do
      case "$1" in
        --nodelist=*)
          [[ -n "${1#--nodelist=}" ]] || { echo 'Missing --nodelist value' >&2; exit 2; }
          nodes=("$1"); shift ;;
        --nodelist)
          if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
            echo 'Missing --nodelist value' >&2; exit 2
          fi
          nodes=(--nodelist "$2"); shift 2 ;;
        *) batch_args+=("$1"); shift ;;
      esac
    done
    "$BATCH_PY" -u -m qwen_edit_pano.world_pipeline.completion.batch_world preflight "${batch_args[@]}"
    srun -p debug "${nodes[@]}" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=64G --qos=normal --time=04:00:00 \
      "$BATCH_PY" -u -m qwen_edit_pano.world_pipeline.completion.batch_world run "${batch_args[@]}"
    ;;
  *) echo 'Usage: bash run_box_world_val.sh check|preflight|run|review [--nodelist GPUX] --output NEW_BATCH_ROOT' >&2; exit 2 ;;
esac
