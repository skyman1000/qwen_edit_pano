#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
pipeline_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
pipeline_bundle=qwen_edit_pano/data/oracle_supplement_epoch015_scale05_v1
pipeline_output=qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1
posterior_source=qwen_edit_pano/outputs/oracle_supplement_epoch006_full_v1
case "${1:-preflight}" in
  bundle)
    if [[ ! -f "$pipeline_bundle/BUNDLE.json" ]]; then
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.oracle_data \
      --gt qwen_edit_pano/data/gt_world_supplement_v1 \
      --review qwen_edit_pano/data/gt_world_supplement_v1/review_decisions.json \
      --selection-policy qwen_edit_pano/data/gt_supplement_experiment_selection_v1.json \
      --max-objects 160 \
      --base-checkpoint qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch015 \
      --output "$pipeline_bundle"
    fi
    "$pipeline_python" -m qwen_edit_pano.world_pipeline.scaled_world.prepare_selection --bundle "$pipeline_bundle"
    ;;
  preflight)
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.fourier_world.train \
      --bundle "$pipeline_bundle" --purpose oracle --pano-lora-scale 0.5 \
      --posterior-source "$posterior_source" --steps 1000 --warmup-steps 100 \
      --blocks 20 40 59 --accumulation 4 --preflight
    ;;
  smoke|train|extend)
    extra=()
    train_steps=1000
    if [[ "$1" == smoke ]]; then
      [[ $# -eq 1 ]] || exit 2
      extra=(--smoke)
      pipeline_output=qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_smoke_v1
    elif [[ "$1" == extend ]]; then
      [[ $# -eq 4 ]] || { echo 'Usage: extend CHECKPOINT NEW_OUTPUT TOTAL_STEPS' >&2; exit 2; }
      extra=(--resume "$2");pipeline_output="$3";train_steps="$4"
    elif [[ $# -eq 3 ]]; then
      extra=(--resume "$2")
      pipeline_output="$3"
    elif [[ $# -ne 1 ]]; then
      echo 'Usage: train [scaled-world-checkpoint NEW-output-directory]' >&2; exit 2
    fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.fourier_world.train \
      --bundle "$pipeline_bundle" --purpose oracle --condition full --pano-lora-scale 0.5 \
      --posterior-source "$posterior_source" \
      --steps "$train_steps" --warmup-steps 100 --save-every 250 --accumulation 4 --lr 1e-4 --seed 0 \
      --blocks 20 40 59 --width 256 \
      --activation-storage cpu --output "$pipeline_output" "${extra[@]}"
    ;;
  eval-smoke|eval|eval-train|eval-val24)
    eval_step="${2:-001000}"
    if [[ "$1" == eval-smoke ]]; then
      pipeline_output=qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_smoke_v1
      eval_step=000002
    fi
    case "$eval_step" in 000002|000250|000500|000750|001000) ;; *) echo 'Use 000250, 000500, 000750 or 001000' >&2; exit 2;; esac
    selection=indoor6_selection.json;eval_split=val;eval_count=6;eval_tag=indoor6
    if [[ "$1" == eval-smoke ]]; then eval_count=1;eval_tag=smoke; fi
    if [[ "$1" == eval-train ]]; then selection=train8_selection.json;eval_split=train;eval_count=8;eval_tag=train8; fi
    if [[ "$1" == eval-val24 ]]; then selection=val24_selection.json;eval_count=24;eval_tag=val24; fi
    "$pipeline_python" -u -m qwen_edit_pano.world_pipeline.fourier_world.inference \
      --checkpoint "$pipeline_output/checkpoint-step${eval_step}" \
      --selection "$pipeline_bundle/$selection" \
      --split "$eval_split" --limit "$eval_count" --steps 28 --cfg 4 --seed 0 \
      --modes baseline correct shuffled \
      --output "qwen_edit_pano/outputs/oracle_epoch015_scale05_${eval_tag}_step${eval_step}_fourier_v1"
    ;;
  *) echo 'Usage: bundle|preflight|smoke|train|extend|eval-smoke|eval|eval-train|eval-val24' >&2; exit 2;;
esac
