#!/usr/bin/env bash
# Rich semantic catalog, fresh optional functional programs and spatial draws.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
export CAUPANO_EXPERIMENT_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_compositional_experiment
export CAUPANO_VISUAL_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_compositional_visual
export CAUPANO_TEST_MODULE=qwen_edit_pano.world_pipeline.completion.test_compositional
export CAUPANO_EXPERIMENT_BRANCHES='end_to_end layout_control function_free'
export CAUPANO_MECHANISM_OUTPUT=${CAUPANO_COMPOSITIONAL_OUTPUT:-qwen_edit_pano/outputs/compositional_completion_$(date +%Y%m%d_%H%M%S)_$$}
exec bash qwen_edit_pano/scripts/mechanism_completion.sh "$@"
