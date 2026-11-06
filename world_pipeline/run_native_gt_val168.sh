#!/usr/bin/env bash
# User allocates the GPU with srun. Evaluation uses the existing DiT360 venv.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
case "${1:-prepare}" in
  prepare|generate|manifests|evaluate|summarize)
    action=${1:-prepare}
    if [[ $# -gt 0 ]]; then shift; fi
    exec /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u \
      -m research.world_evaluation.native_gt_benchmark "$action" "$@"
    ;;
  *) echo 'Usage: prepare|generate|manifests|evaluate [--dry-run] [--eval-tag NAME]|summarize' >&2; exit 2;;
esac
