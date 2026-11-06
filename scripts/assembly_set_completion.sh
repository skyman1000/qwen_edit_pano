#!/usr/bin/env bash
# Observation-conditioned joint candidate assemblies; no training or object quota.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
export CAUPANO_EXPERIMENT_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_set_experiment
export CAUPANO_VISUAL_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_set_visual
export CAUPANO_TEST_MODULE=qwen_edit_pano.world_pipeline.completion.test_assembly_set
export CAUPANO_EXPERIMENT_BRANCHES='end_to_end layout_control function_free direct_inventory tag_only'
export CAUPANO_MECHANISM_OUTPUT=${CAUPANO_ASSEMBLY_SET_OUTPUT:-qwen_edit_pano/outputs/assembly_set_completion_$(date +%Y%m%d_%H%M%S)_$$}
exec bash qwen_edit_pano/scripts/mechanism_completion.sh "$@"
