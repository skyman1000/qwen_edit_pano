#!/usr/bin/env bash
# Saved coordinates -> numerical search + visuals. No Slurm, GPU, or API key.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: organized_numeric_recovery.sh SOURCE_SUITE [NEW_OUTPUT]' >&2;exit 2; }
nr_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
nr_root=${2:-qwen_edit_pano/outputs/organized_numeric_recovery_$(date +%Y%m%d_%H%M%S)_$$}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
unset CAUPANO_PROGRAM_CATALOG
CUDA_VISIBLE_DEVICES= "$nr_python" -u -m qwen_edit_pano.world_pipeline.completion.assembly_numeric_recovery run \
  --root "$nr_root" --source-root "$1"
echo "Numeric recovery and visuals: $nr_root/index.html"
