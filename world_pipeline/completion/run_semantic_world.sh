#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
WORLD_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
action="${1:-}"
if (( $# )); then shift; fi
case "$action" in
  check)
    "$WORLD_PY" -m unittest qwen_edit_pano.world_pipeline.completion.test_semantic_world qwen_edit_pano.world_pipeline.completion.test_planned_completion qwen_edit_pano.world_pipeline.completion.test_scene_consistency qwen_edit_pano.world_pipeline.completion.test_box_completion qwen_edit_pano.world_pipeline.completion.test_layout_examples qwen_edit_pano.world_pipeline.completion.test_qwen_upstream qwen_edit_pano.world_pipeline.completion.test_placement_program qwen_edit_pano.world_pipeline.completion.test_anchored_completion -v
    ;;
  preflight)
    "$WORLD_PY" -m qwen_edit_pano.world_pipeline.completion.semantic_world --preflight "$@"
    ;;
  finish)
    "$WORLD_PY" -u -m qwen_edit_pano.world_pipeline.completion.semantic_world --postprocess-only "$@"
    ;;
  run)
    # Node selection belongs to Slurm, not the Python scene-state arguments.
    slurm_node_args=()
    world_args=()
    while (( $# )); do
      case "$1" in
        --nodelist=*)
          if [[ -z "${1#--nodelist=}" ]]; then echo 'Missing value for --nodelist' >&2; exit 2; fi
          slurm_node_args=("$1")
          shift
          ;;
        --nodelist)
          if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
            echo 'Missing value for --nodelist' >&2
            exit 2
          fi
          slurm_node_args=(--nodelist "$2")
          shift 2
          ;;
        *) world_args+=("$1"); shift ;;
      esac
    done
    srun -p debug "${slurm_node_args[@]}" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
      "$WORLD_PY" -u -m qwen_edit_pano.world_pipeline.completion.semantic_world "${world_args[@]}"
    ;;
  *)
    echo 'Usage: bash run_semantic_world.sh check|preflight|run|finish [options]; run accepts --nodelist GPUX'
    exit 2
    ;;
esac
