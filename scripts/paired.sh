#!/usr/bin/env bash
# Invoke inside srun; no sbatch directives and no implicit GPU job submission.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python}"
ACTION="${1:?Usage: paired.sh prepare|cache|smoke|train|infer|test [arguments]}"
shift
case "$ACTION" in
  prepare) exec "$PYTHON_BIN" -m qwen_edit_pano.prepare_pairs "$@" ;;
  cache) exec "$PYTHON_BIN" -m qwen_edit_pano.cache_paired --local-files-only "$@" ;;
  smoke) exec "$PYTHON_BIN" -m qwen_edit_pano.paired_train --local-files-only --smoke-test "$@" ;;
  train) exec "$PYTHON_BIN" -m qwen_edit_pano.paired_train --local-files-only "$@" ;;
  infer) exec "$PYTHON_BIN" -m qwen_edit_pano.paired_inference --local-files-only "$@" ;;
  test) exec "$PYTHON_BIN" -m qwen_edit_pano.test_paired_interface "$@" ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
