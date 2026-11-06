#!/usr/bin/env bash
# Full joint inventories; three sampling laws over the EXACT SAME proposals.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
pc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
pc_module=qwen_edit_pano.world_pipeline.completion.assembly_experiment
export MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ ${1:-} == test ]]; then
  CUDA_VISIBLE_DEVICES= "$pc_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_assembly_population -v
  exit
fi
if [[ $# -lt 4 || $# -gt 5 || $1 != compare ]]; then
  echo 'Usage: population_completion.sh compare GPU3 SOURCE_EXPERIMENT qwen|api|both [SEED]; test' >&2
  exit 2
fi
pc_node=$2
pc_source=$3
pc_backend=$4
pc_seed=${5:-3407}
case "$pc_backend" in
  qwen) pc_backends=(qwen) ;;
  api) pc_backends=(api) ;;
  both) pc_backends=(api qwen) ;;
  *) echo 'Backend must be qwen, api or both' >&2; exit 2 ;;
esac
if [[ $pc_backend != qwen && -z ${SILICONFLOW_API_KEY:-} ]]; then
  echo 'Set SILICONFLOW_API_KEY for api/both.' >&2; exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
pc_root=${CAUPANO_POPULATION_OUTPUT:-qwen_edit_pano/outputs/population_completion_$(date +%Y%m%d_%H%M%S)_$$}
mkdir "$pc_root"
pc_status=0
for pc_method in population_posterior joint_reference population_no_function; do
  case "$pc_method" in
    population_posterior) pc_policy=posterior ;;
    joint_reference) pc_policy=reference ;;
    population_no_function) pc_policy=no_function ;;
  esac
  pc_args=(-m "$pc_module" restart --root "$pc_root/$pc_method" --source-root "$pc_source"
    --completion-model population --population-sampling "$pc_policy" --seed "$pc_seed" --worlds "${CAUPANO_WORLDS:-4}")
  if [[ -n ${CAUPANO_POPULATION_PARAMETERS:-} ]]; then pc_args+=(--population-parameters "$CAUPANO_POPULATION_PARAMETERS"); fi
  if [[ $pc_method == population_posterior && -n ${CAUPANO_POPULATION_CANDIDATES:-} ]]; then
    pc_args+=(--candidate-programs-from "$CAUPANO_POPULATION_CANDIDATES")
  fi
  if [[ $pc_method != population_posterior ]]; then pc_args+=(--semantic-mechanisms-from "$pc_root/population_posterior"); fi
  CUDA_VISIBLE_DEVICES= "$pc_python" "${pc_args[@]}"
  for pc_b in "${pc_backends[@]}"; do
    if [[ $pc_method != population_posterior ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$pc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_functional_status \
        check --root "$pc_root/population_posterior" --backend "$pc_b"; then
        CUDA_VISIBLE_DEVICES= "$pc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_functional_status \
          block --root "$pc_root/$pc_method" --source-root "$pc_root/population_posterior" --backend "$pc_b" \
          | tee "$pc_root/$pc_method/$pc_b/completion.log"
        pc_status=1
        continue
      fi
    fi
    pc_args=(-u -m "$pc_module" world --root "$pc_root/$pc_method" --backend "$pc_b")
    if [[ $pc_b == api ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$pc_python" "${pc_args[@]}" 2>&1 | tee "$pc_root/$pc_method/$pc_b/completion.log"; then pc_status=1; fi
    else
      if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$pc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
        "$pc_python" "${pc_args[@]}" 2>&1 | tee "$pc_root/$pc_method/$pc_b/completion.log"; then pc_status=1; fi
    fi
  done
done
CUDA_VISIBLE_DEVICES= "$pc_python" - "$pc_root" <<'PY'
import sys
from pathlib import Path
from qwen_edit_pano.world_pipeline.common import write
write(Path(sys.argv[1])/'pilot.json',dict(methods=['joint_reference','population_posterior','population_no_function'],
    comparison='population_v3_fixed_domain'))
PY
for pc_b in "${pc_backends[@]}"; do
  if ! CUDA_VISIBLE_DEVICES= "$pc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_pilot_visual \
    --root "$pc_root" --backend "$pc_b"; then pc_status=1; fi
done
if [[ $pc_status -eq 0 ]]; then
  echo "Population world comparison: $pc_root"
else
  echo "Population run has failures; inspect summaries and diagnostic gallery: $pc_root" >&2
fi
exit "$pc_status"
