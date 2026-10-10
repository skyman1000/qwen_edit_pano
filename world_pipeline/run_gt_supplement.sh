#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
pipeline_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
pipeline_sources=qwen_edit_pano/data/gt_sources_supplement_v1
pipeline_pairs=qwen_edit_pano/data/paired_gt_supplement_ready_v1
pipeline_gt=qwen_edit_pano/data/gt_world_supplement_v1
pipeline_bundle=qwen_edit_pano/data/oracle_supplement_epoch006_v1
pipeline_checkpoint=qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006
case "${1:-plan}" in
  plan|build)
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.supplement_gt "${1:-plan}"
    ;;
  download)
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.supplement_gt plan
    "$pipeline_python" -u download_mp_py3.py \
      -o benchmark_assets/Matterport3D_raw \
      --scan-list qwen_edit_pano/data/paired_gt_supplement_v1/added_scans.txt \
      --type house_segmentations region_segmentations matterport_camera_intrinsics \
        matterport_camera_poses undistorted_camera_parameters undistorted_color_images \
        undistorted_depth_images undistorted_normal_images \
      --retries 8 --timeout 60 --retry-delay 5 --verify-existing
    ;;
  select)
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.prepare_source_ready_pairs \
      --sources "$pipeline_sources" --pairs qwen_edit_pano/data/paired_gt_supplement_v1 \
      --output "$pipeline_pairs"
    ;;
  export)
    resume_args=()
    if [[ -d "$pipeline_gt" ]]; then resume_args+=(--resume); fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.prepare_gt \
      --pairs "$pipeline_pairs" --world-root "$pipeline_sources" \
      --train-limit 0 --val-limit 0 --threads 4 --require-all-pairs \
      --output "$pipeline_gt" "${resume_args[@]}"
    ;;
  bundle)
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.oracle_data \
      --gt "$pipeline_gt" --review "$pipeline_gt/review_decisions.json" \
      --selection-policy qwen_edit_pano/data/gt_supplement_experiment_selection_v1.json --max-objects 160 \
      --base-checkpoint "$pipeline_checkpoint" --output "$pipeline_bundle"
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.oracle_train \
      --bundle "$pipeline_bundle" --purpose oracle --preflight
    ;;
  train-full|train-constant)
    condition="${1#train-}"
    resume_args=()
    run_output="qwen_edit_pano/outputs/oracle_supplement_epoch006_${condition}_v1"
    # Optional positional resume checkpoint and NEW output directory.
    if [[ $# -ne 1 && $# -ne 3 ]]; then echo 'Usage: train-full|train-constant [resume-checkpoint new-output]' >&2; exit 2; fi
    if [[ $# -eq 3 ]]; then resume_args=(--resume "$2"); run_output="$3"; fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.oracle_train \
      --bundle "$pipeline_bundle" --condition "$condition" --purpose oracle \
      --steps 1000 --warmup-steps 100 --save-every 100 --accumulation 4 --lr 1e-4 --seed 0 \
      --activation-storage cpu --output "$run_output" "${resume_args[@]}"
    ;;
  eval-full|eval-constant)
    condition="${1#eval-}"
    modes=(correct)
    if [[ "$condition" == full ]]; then modes=(baseline correct shuffled); fi
    # Optional checkpoint and NEW output for resumed runs or other saved steps.
    if [[ $# -ne 1 && $# -ne 3 ]]; then echo 'Usage: eval-full|eval-constant [checkpoint new-output]' >&2; exit 2; fi
    eval_checkpoint="qwen_edit_pano/outputs/oracle_supplement_epoch006_${condition}_v1/checkpoint-step001000"
    eval_output="qwen_edit_pano/outputs/oracle_supplement_epoch006_${condition}_val_v1"
    if [[ $# -eq 3 ]]; then eval_checkpoint="$2"; eval_output="$3"; fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.oracle_inference \
      --checkpoint "$eval_checkpoint" --split val --limit 183 \
      --seed 0 --steps 28 --cfg 4 --modes "${modes[@]}" --output "$eval_output"
    ;;
  *) echo 'Usage: plan|download|build|select|export|bundle|train-full|train-constant|eval-full|eval-constant' >&2; exit 2 ;;
esac
