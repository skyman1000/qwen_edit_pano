#!/usr/bin/env bash
# Both models share one frozen observer, value domain, budgets and compiler.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
sc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
sc_module=qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment
export MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ ${1:-} == test ]]; then
  CUDA_VISIBLE_DEVICES= "$sc_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_sdscene -v
  exit
fi
if [[ $# -lt 3 || $# -gt 4 || $1 != compare ]]; then
  echo 'Usage: sdscene_compare.sh compare GPU3 SOURCE_EXPERIMENT [SEED]; test' >&2; exit 2
fi
if [[ -z ${SILICONFLOW_API_KEY:-} ]]; then
  echo 'This comparison includes API inference. Set SILICONFLOW_API_KEY in your terminal; never put it in logs or chat.' >&2
  exit 2
fi
sc_node=$2
sc_source=$3
sc_seed=${4:-3407}
sc_api_model='deepseek-ai/DeepSeek-V4.1-Flash'
export SILICONFLOW_API_BASE='https://api.siliconflow.cn/v1'
sc_root=${CAUPANO_SDSCENE_OUTPUT:-qwen_edit_pano/outputs/sdscene_compare_$(date +%Y%m%d_%H%M%S)_$$}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
CUDA_VISIBLE_DEVICES= "$sc_python" -m "$sc_module" prepare --root "$sc_root" --source-root "$sc_source" \
  --api-model "$sc_api_model" --seed "$sc_seed" --worlds "${CAUPANO_WORLDS:-4}" --draws "${CAUPANO_MECHANISM_DRAWS:-3}"
sc_status=0
sc_phase() {
  local action=$1 backend=$2
  local args=(-u -m "$sc_module" "$action" --root "$sc_root" --backend "$backend")
  if [[ $backend == api ]]; then
    CUDA_VISIBLE_DEVICES= "$sc_python" "${args[@]}" 2>&1 | tee "$sc_root/${action}_${backend}.log"
  else
    srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$sc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
      "$sc_python" "${args[@]}" 2>&1 | tee "$sc_root/${action}_${backend}.log"
  fi
}
for sc_b in api qwen; do
  if ! sc_phase propose "$sc_b"; then sc_status=1; fi
done
if CUDA_VISIBLE_DEVICES= "$sc_python" -m "$sc_module" pool --root "$sc_root"; then
  for sc_b in api qwen; do
    if ! sc_phase mechanisms "$sc_b"; then sc_status=1; fi
  done
  if ! CUDA_VISIBLE_DEVICES= "$sc_python" -m "$sc_module" pool-mechanisms --root "$sc_root"; then sc_status=1; fi
else
  sc_status=1
fi
for sc_experiment in end_to_end layout_control; do
  for sc_b in api qwen; do
    if ! CUDA_VISIBLE_DEVICES= "$sc_python" -m "$sc_module" ready --root "$sc_root" --experiment "$sc_experiment" --backend "$sc_b"; then
      CUDA_VISIBLE_DEVICES= "$sc_python" -m "$sc_module" block --root "$sc_root" --experiment "$sc_experiment" --backend "$sc_b"
      sc_status=1
      continue
    fi
    sc_args=(-u -m qwen_edit_pano.world_pipeline.completion.assembly_experiment world --root "$sc_root/$sc_experiment" --backend "$sc_b")
    if [[ $sc_b == api ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$sc_python" "${sc_args[@]}" 2>&1 | tee "$sc_root/${sc_experiment}_${sc_b}.log"; then sc_status=1; fi
    else
      if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$sc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
        "$sc_python" "${sc_args[@]}" 2>&1 | tee "$sc_root/${sc_experiment}_${sc_b}.log"; then sc_status=1; fi
    fi
  done
done
if ! CUDA_VISIBLE_DEVICES= "$sc_python" -m "$sc_module" report --root "$sc_root"; then sc_status=1; fi
if [[ $sc_status -eq 0 ]]; then
  echo "Qwen/API scene comparison: $sc_root/index.html"
else
  echo "Comparison has failed or blocked stages; inspect $sc_root/index.html and phase_errors." >&2
fi
exit "$sc_status"
