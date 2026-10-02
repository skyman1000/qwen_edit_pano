#!/usr/bin/env bash
# Run interactively: the original downloader asks you to acknowledge MP terms.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${MP3D_RAW_ROOT:-$PROJECT_ROOT/benchmark_assets/Matterport3D_raw}"
AUDIT="$PROJECT_ROOT/qwen_edit_pano/audit_skybox_coverage.py"
"$PYTHON_BIN" "$AUDIT" --raw-root "$RAW_ROOT" --fail-on-invalid
# Explicit --type is essential: omitting it downloads all data types.
# Existing ZIPs are skipped. Named .part files resume when HTTP Range is supported.
"$PYTHON_BIN" "$PROJECT_ROOT/download_mp_py3.py" -o "$RAW_ROOT" --id ALL --type matterport_skybox_images
"$PYTHON_BIN" "$AUDIT" --raw-root "$RAW_ROOT" --verify-crc --require-full
