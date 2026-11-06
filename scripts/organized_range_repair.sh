#!/usr/bin/env bash
# Recover only missing group distances; original v13 outputs remain intact.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
range_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
range_module=qwen_edit_pano.world_pipeline.completion.assembly_organized_repair
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ ${1:-} == test ]]; then
  CUDA_VISIBLE_DEVICES= "$range_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_organized_range_repair -v
  exit
fi
if [[ $# -lt 3 || $# -gt 4 || ( $1 != prepare && $1 != repair ) ]]; then
  echo 'Usage: organized_range_repair.sh prepare|repair GPU3 ORIGINAL_V13_RUN [NEW_OUTPUT]; test' >&2
  exit 2
fi
range_action=$1
range_node=$2
range_source=$3
range_root=${4:-qwen_edit_pano/outputs/organized_range_repair_$(date +%Y%m%d_%H%M%S)_$$}
CUDA_VISIBLE_DEVICES= "$range_python" -m "$range_module" prepare --root "$range_root" --source-run "$range_source"
if [[ $range_action == prepare ]]; then
  echo "Prepared only; no model/API calls. New root: $range_root"
  exit
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
range_status=0
for range_backend in qwen api; do
  range_pending=$(CUDA_VISIBLE_DEVICES= "$range_python" -m "$range_module" pending --root "$range_root" --backend "$range_backend")
  if [[ $range_pending == 0 ]]; then
    echo "$range_backend: no pending cases; no provider initialized."
    continue
  fi
  if [[ $range_backend == qwen ]]; then
    if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$range_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
        "$range_python" -u -m "$range_module" run --root "$range_root" --backend qwen \
        2>&1 | tee "$range_root/range_repair_qwen.log"; then range_status=1;fi
  else
    if [[ -z ${SILICONFLOW_API_KEY:-} ]]; then
      echo 'API cases are pending: set SILICONFLOW_API_KEY for the frozen SiliconFlow configuration.' >&2
      range_status=1
      continue
    fi
    if ! CUDA_VISIBLE_DEVICES= "$range_python" -u -m "$range_module" run --root "$range_root" --backend api \
        2>&1 | tee "$range_root/range_repair_api.log"; then range_status=1;fi
  fi
done
if ! CUDA_VISIBLE_DEVICES= "$range_python" -m "$range_module" report --root "$range_root"; then range_status=1;fi
echo "Range-only repair comparison: $range_root/index.html (exit_status=$range_status)"
exit "$range_status"
