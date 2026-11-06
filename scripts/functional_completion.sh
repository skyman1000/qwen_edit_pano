#!/usr/bin/env bash
# Same frozen observations AND candidate grammar: prior versus cue posterior.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
fc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
fc_module=qwen_edit_pano.world_pipeline.completion.assembly_experiment
export MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ ${1:-} == test ]]; then
  CUDA_VISIBLE_DEVICES= "$fc_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_assembly_functional -v
  exit
fi
if [[ $# -lt 4 || $# -gt 5 || $1 != compare ]]; then
  echo 'Usage: functional_completion.sh compare GPU3 SOURCE_EXPERIMENT qwen|api|both [SEED]; test' >&2
  exit 2
fi
fc_node=$2
fc_source=$3
fc_backend=$4
fc_seed=${5:-3407}
case "$fc_backend" in
  qwen) fc_backends=(qwen) ;;
  api) fc_backends=(api) ;;
  both) fc_backends=(api qwen) ;;
  *) echo 'Backend must be qwen, api or both' >&2; exit 2 ;;
esac
if [[ $fc_backend != qwen && -z ${SILICONFLOW_API_KEY:-} ]]; then
  echo 'Set SILICONFLOW_API_KEY for api/both.' >&2; exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
fc_root=${CAUPANO_FUNCTIONAL_OUTPUT:-qwen_edit_pano/outputs/functional_completion_$(date +%Y%m%d_%H%M%S)_$$}
# new_output inside restart rejects an existing experiment, preserving old runs.
mkdir "$fc_root"
fc_status=0
for fc_method in posterior prior; do
  fc_args=(-m "$fc_module" restart --root "$fc_root/$fc_method" --source-root "$fc_source"
    --completion-model functional --functional-sampling "$fc_method" --seed "$fc_seed" --worlds "${CAUPANO_WORLDS:-6}")
  if [[ -n ${CAUPANO_FUNCTIONAL_PARAMETERS:-} ]]; then fc_args+=(--functional-parameters "$CAUPANO_FUNCTIONAL_PARAMETERS"); fi
  if [[ $fc_method == prior ]]; then fc_args+=(--semantic-mechanisms-from "$fc_root/posterior"); fi
  CUDA_VISIBLE_DEVICES= "$fc_python" "${fc_args[@]}"
  for fc_b in "${fc_backends[@]}"; do
    if [[ $fc_method == prior ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$fc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_functional_status \
        check --root "$fc_root/posterior" --backend "$fc_b"; then
        CUDA_VISIBLE_DEVICES= "$fc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_functional_status \
          block --root "$fc_root/prior" --source-root "$fc_root/posterior" --backend "$fc_b" \
          | tee "$fc_root/prior/$fc_b/completion.log"
        fc_status=1
        continue
      fi
    fi
    fc_args=(-u -m "$fc_module" world --root "$fc_root/$fc_method" --backend "$fc_b")
    if [[ $fc_b == api ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$fc_python" "${fc_args[@]}" 2>&1 | tee "$fc_root/$fc_method/$fc_b/completion.log"; then fc_status=1; fi
    else
      if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$fc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
        "$fc_python" "${fc_args[@]}" 2>&1 | tee "$fc_root/$fc_method/$fc_b/completion.log"; then fc_status=1; fi
    fi
  done
done
CUDA_VISIBLE_DEVICES= "$fc_python" - "$fc_root" <<'PY'
import sys
from pathlib import Path
from qwen_edit_pano.world_pipeline.common import write
write(Path(sys.argv[1])/'pilot.json',dict(methods=['prior','posterior'],comparison='frozen_catalog_probability_ablation'))
PY
for fc_b in "${fc_backends[@]}"; do
  if ! CUDA_VISIBLE_DEVICES= "$fc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_pilot_visual \
    --root "$fc_root" --backend "$fc_b"; then fc_status=1; fi
done
if [[ $fc_status -eq 0 ]]; then
  echo "Functional probability comparison: $fc_root"
else
  echo "Functional run has failures; inspect summaries and diagnostic comparison: $fc_root" >&2
fi
exit "$fc_status"
