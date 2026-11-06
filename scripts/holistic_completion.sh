#!/usr/bin/env bash
# Joint inventory / camera organization, shared local/API configuration.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
export CAUPANO_EXPERIMENT_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_holistic_experiment
export CAUPANO_VISUAL_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_holistic_visual
export CAUPANO_TEST_MODULE=qwen_edit_pano.world_pipeline.completion.test_holistic
export CAUPANO_EXPERIMENT_BRANCHES='end_to_end layout_control function_free direct_inventory rgb_inventory'
export CAUPANO_MECHANISM_OUTPUT=${CAUPANO_HOLISTIC_OUTPUT:-qwen_edit_pano/outputs/holistic_completion_$(date +%Y%m%d_%H%M%S)_$$}
exec bash qwen_edit_pano/scripts/mechanism_completion.sh "$@"
