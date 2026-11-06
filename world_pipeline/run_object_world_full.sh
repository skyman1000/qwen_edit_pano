#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
exec /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u -m \
  qwen_edit_pano.world_pipeline.grounded_world.full_run "$@"
