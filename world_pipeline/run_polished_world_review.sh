#!/usr/bin/env bash
# User-run CPU review. Does not download, train, or alter the source dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."
world_review_python="${WORLD_DATA_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
"$world_review_python" -u -m qwen_edit_pano.world_pipeline.review_polished_world "$@"
