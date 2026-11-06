#!/usr/bin/env bash
# v9 world-state completion; no RGB panorama renderer or model training.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
export CAUPANO_EXPERIMENT_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_panorama_experiment
export CAUPANO_VISUAL_MODULE=qwen_edit_pano.world_pipeline.completion.assembly_panorama_visual
export CAUPANO_TEST_MODULE=qwen_edit_pano.world_pipeline.completion.test_panorama
export CAUPANO_MECHANISM_OUTPUT=${CAUPANO_PANORAMA_OUTPUT:-qwen_edit_pano/outputs/panorama_completion_$(date +%Y%m%d_%H%M%S)_$$}
exec bash qwen_edit_pano/scripts/mechanism_completion.sh "$@"
