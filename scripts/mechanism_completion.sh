#!/usr/bin/env bash
# No-training v8: reference-shrunk causal mechanism + direct inventory control.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
mc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
mc_module=${CAUPANO_EXPERIMENT_MODULE:-qwen_edit_pano.world_pipeline.completion.assembly_mechanism_experiment}
mc_visual_module=${CAUPANO_VISUAL_MODULE:-qwen_edit_pano.world_pipeline.completion.assembly_mechanism_visual}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ ${1:-} == test ]]; then
  CUDA_VISIBLE_DEVICES= "$mc_python" -m unittest "${CAUPANO_TEST_MODULE:-qwen_edit_pano.world_pipeline.completion.test_mechanism}" -v
  exit
fi
if [[ $# -lt 3 || $# -gt 4 || ( $1 != compare && $1 != prepare ) ]]; then
  echo 'Usage: mechanism_completion.sh compare|prepare GPU3 SOURCE_EXPERIMENT [SEED]; test' >&2; exit 2
fi
mc_action=$1;mc_node=$2;mc_source=$3;mc_seed=${4:-3407}
if [[ $mc_action == compare && -z ${SILICONFLOW_API_KEY:-} ]]; then
  echo 'Set SILICONFLOW_API_KEY; this runs SiliconFlow deepseek-ai/DeepSeek-V4.1-Flash and local Qwen.' >&2; exit 2
fi
export SILICONFLOW_API_BASE='https://api.siliconflow.cn/v1'
export SILICONFLOW_MODEL='deepseek-ai/DeepSeek-V4.1-Flash'
mc_root=${CAUPANO_MECHANISM_OUTPUT:-qwen_edit_pano/outputs/mechanism_completion_$(date +%Y%m%d_%H%M%S)_$$}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" prepare --root "$mc_root" --source-root "$mc_source" \
  --seed "$mc_seed" --worlds "${CAUPANO_WORLDS:-4}" --draws "${CAUPANO_MECHANISM_DRAWS:-4}" \
  --concentration "${CAUPANO_CONCENTRATION:-1}" --reference-concentration "${CAUPANO_REFERENCE_CONCENTRATION:-1}"
if [[ $mc_action == prepare ]]; then
  echo "Prepared only; no model/API calls: $mc_root/run.json"
  exit
fi
mc_status=0
mc_phase() {
  local action=$1 backend=$2
  local args=(-u -m "$mc_module" "$action" --root "$mc_root" --backend "$backend")
  if [[ $backend == api ]]; then
    CUDA_VISIBLE_DEVICES= "$mc_python" "${args[@]}" 2>&1 | tee "$mc_root/${action}_${backend}.log"
  else
    srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$mc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
      "$mc_python" "${args[@]}" 2>&1 | tee "$mc_root/${action}_${backend}.log"
  fi
}
for mc_b in api qwen; do
  if ! mc_phase propose "$mc_b"; then mc_status=1; fi
done
if CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" pool --root "$mc_root"; then
  for mc_b in api qwen; do
    if ! mc_phase mechanisms "$mc_b"; then mc_status=1; fi
  done
  if ! CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" pool-mechanisms --root "$mc_root"; then mc_status=1; fi
else mc_status=1
fi
for mc_experiment in ${CAUPANO_EXPERIMENT_BRANCHES:-end_to_end layout_control function_free direct_inventory}; do
  for mc_b in api qwen; do
    if ! CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" ready --root "$mc_root" --experiment "$mc_experiment" --backend "$mc_b"; then
      CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" block --root "$mc_root" --experiment "$mc_experiment" --backend "$mc_b"
      mc_status=1;continue
    fi
    mc_args=(-u -m qwen_edit_pano.world_pipeline.completion.assembly_experiment world --root "$mc_root/$mc_experiment" --backend "$mc_b")
    if [[ $mc_b == api ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$mc_python" "${mc_args[@]}" 2>&1 | tee "$mc_root/${mc_experiment}_${mc_b}.log"; then mc_status=1; fi
    else
      if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$mc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
        "$mc_python" "${mc_args[@]}" 2>&1 | tee "$mc_root/${mc_experiment}_${mc_b}.log"; then mc_status=1; fi
    fi
  done
done
if [[ -n ${CAUPANO_BASELINE_ROOT:-} ]]; then
  if ! CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_visual_module" \
    --root "$mc_root" --baseline-root "$CAUPANO_BASELINE_ROOT"; then mc_status=1; fi
else
  if ! CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" report --root "$mc_root"; then mc_status=1; fi
fi
echo "No-training mechanism comparison: $mc_root/index.html (exit_status=$mc_status)"
exit "$mc_status"
