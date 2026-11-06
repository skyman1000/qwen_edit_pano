#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
action="${1:-}"
if (( $# )); then shift; fi
BOX_PY="${BOX_PY:-/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python}"
case "$action" in
  check)
    /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python -m unittest \
      qwen_edit_pano.world_pipeline.completion.test_box_observer -v
    ;;
  preflight)
    "$BOX_PY" -m qwen_edit_pano.world_pipeline.completion.box_observer --preflight "$@"
    ;;
  run)
    node_args=()
    model_args=()
    while (( $# )); do
      case "$1" in
        --nodelist=*)
          [[ -n "${1#--nodelist=}" ]] || { echo 'Missing --nodelist value' >&2; exit 2; }
          node_args=("$1"); shift ;;
        --nodelist)
          if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
            echo 'Missing --nodelist value' >&2; exit 2
          fi
          node_args=(--nodelist "$2"); shift 2 ;;
        *) model_args+=("$1"); shift ;;
      esac
    done
    # Check files, imports and cached config before occupying a GPU allocation.
    "$BOX_PY" -m qwen_edit_pano.world_pipeline.completion.box_observer --preflight "${model_args[@]}"
    srun -p debug "${node_args[@]}" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
      "$BOX_PY" -u -m qwen_edit_pano.world_pipeline.completion.box_observer "${model_args[@]}"
    ;;
  *) echo 'Usage: bash run_box_observer.sh check|preflight|run [options]; run accepts --nodelist GPUX' >&2; exit 2 ;;
esac
