#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-1000}" in
  1000) checkpoint_step=001000 ;;
  500) checkpoint_step=000500 ;;
  *) echo 'Usage: run_indoor_diagnostic.sh [1000|500]' >&2; exit 2 ;;
esac
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -u -m qwen_edit_pano.world_pipeline.indoor_diagnostic_inference \
  --checkpoint "qwen_edit_pano/outputs/oracle_supplement_epoch006_full_v1/checkpoint-step${checkpoint_step}" \
  --selection qwen_edit_pano/data/oracle_indoor6_selection_v1.json \
  --split val --limit 6 --steps 28 --cfg 4 --seed 0 \
  --modes baseline correct shuffled \
  --output "qwen_edit_pano/outputs/oracle_indoor6_step${checkpoint_step}_v1"
