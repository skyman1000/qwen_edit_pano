#!/usr/bin/env bash
# User-run CPU pipeline. No scheduler submission or implicit download.
set -euo pipefail
cd "$(dirname "$0")/../.."
GT_PYTHON="${GT_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
GT_SOURCE_ROOT="${GT_SOURCE_ROOT:-qwen_edit_pano/data/gt_sources_full_v1}"
GT_EXPORT_ROOT="${GT_EXPORT_ROOT:-qwen_edit_pano/data/gt_world_full_v1}"
GT_PAIRS_ROOT="${GT_PAIRS_ROOT:-qwen_edit_pano/data/paired_full_v1}"
GT_THREADS="${GT_THREADS:-4}"
export OMP_NUM_THREADS="$GT_THREADS" OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
action="${1:-plan}"

plan_args=(--pairs "$GT_PAIRS_ROOT" --work-root "$GT_SOURCE_ROOT" --threads "$GT_THREADS")
if [[ -d "$GT_SOURCE_ROOT" ]]; then plan_args+=(--resume); fi

case "$action" in
  plan)
    "$GT_PYTHON" -u -m qwen_edit_pano.world_pipeline.expand_gt_sources "${plan_args[@]}"
    ;;
  download)
    "$GT_PYTHON" -u -m qwen_edit_pano.world_pipeline.expand_gt_sources "${plan_args[@]}"
    "$GT_PYTHON" -u download_mp_py3.py \
      -o benchmark_assets/Matterport3D_raw --scan-list "$GT_SOURCE_ROOT/scans.txt" \
      --type house_segmentations region_segmentations matterport_camera_intrinsics \
        matterport_camera_poses undistorted_camera_parameters undistorted_color_images \
        undistorted_depth_images undistorted_normal_images \
      --retries 8 --timeout 60 --retry-delay 5 --verify-existing
    ;;
  build)
    "$GT_PYTHON" -u -m qwen_edit_pano.world_pipeline.expand_gt_sources "${plan_args[@]}" --build
    ;;
  export)
    "$GT_PYTHON" - "$GT_SOURCE_ROOT" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1])/'BUILD_RESULT.json'
if not report.is_file() or not json.loads(report.read_text())['all_requested_built']:
    raise SystemExit('Source build incomplete: inspect inventory.json/build_progress.json; do not silently export a partial full dataset.')
PY
    export_args=(--pairs "$GT_PAIRS_ROOT" --world-root "$GT_SOURCE_ROOT"
      --train-limit 0 --val-limit 0 --threads "$GT_THREADS" --require-all-pairs --output "$GT_EXPORT_ROOT")
    if [[ -d "$GT_EXPORT_ROOT" ]]; then export_args+=(--resume); fi
    "$GT_PYTHON" -u -m qwen_edit_pano.world_pipeline.prepare_gt "${export_args[@]}"
    ;;
  *)
    echo 'Usage: bash qwen_edit_pano/world_pipeline/run_gt_full.sh plan|download|build|export' >&2
    exit 2
    ;;
esac
