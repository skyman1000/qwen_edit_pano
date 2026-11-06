#!/usr/bin/env bash
# v13: fixed rich candidates, balanced mechanisms, two-stage geometry.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
org_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
org_module=qwen_edit_pano.world_pipeline.completion.assembly_organized_experiment
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ ${1:-} == test ]]; then
  CUDA_VISIBLE_DEVICES= "$org_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_organized -v
  exit
fi
if [[ ${1:-} == replay ]]; then
  if [[ $# -lt 2 || $# -gt 3 ]]; then echo 'Usage: organized_completion.sh replay V12_RUN [NEW_OUTPUT]' >&2;exit 2;fi
  org_out=${3:-research/audits/organized_replay_$(date +%Y%m%d_%H%M%S)_$$}
  CUDA_VISIBLE_DEVICES= "$org_python" -m qwen_edit_pano.world_pipeline.completion.assembly_organized_replay --source-run "$2" --out "$org_out"
  exit
fi
if [[ $# -lt 3 || $# -gt 4 || ( $1 != prepare && $1 != compare ) ]]; then
  echo 'Usage: organized_completion.sh compare|prepare GPU3 V12_RUN [SEED]; test; replay V12_RUN [NEW_OUTPUT]' >&2;exit 2
fi
org_action=$1;org_node=$2;org_source=$3;org_seed=${4:-3407}
if [[ $org_action == compare && -z ${SILICONFLOW_API_KEY:-} ]]; then
  echo 'Set SILICONFLOW_API_KEY for SiliconFlow deepseek-ai/DeepSeek-V4.1-Flash.' >&2;exit 2
fi
export SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1 SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
org_root=${CAUPANO_ORGANIZED_OUTPUT:-qwen_edit_pano/outputs/organized_completion_$(date +%Y%m%d_%H%M%S)_$$}
CUDA_VISIBLE_DEVICES= "$org_python" -m "$org_module" prepare --root "$org_root" --source-run "$org_source" --seed "$org_seed" \
  --draws "${CAUPANO_MECHANISM_DRAWS:-4}" --concentration "${CAUPANO_CONCENTRATION:-1}" --variant-concentration "${CAUPANO_REFERENCE_CONCENTRATION:-1}"
if [[ $org_action == prepare ]]; then echo "Prepared only; no model/API calls: $org_root/run.json";exit;fi
org_phase() {
  local phase=$1 backend=$2
  shift 2
  local args=(-u -m "$org_module" "$phase" --root "$org_root" --backend "$backend" "$@")
  if [[ $backend == api ]]; then
    CUDA_VISIBLE_DEVICES= "$org_python" "${args[@]}"
  else
    srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$org_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" "$org_python" "${args[@]}"
  fi
}
org_status=0
for org_b in api qwen; do
  if ! org_phase mechanisms "$org_b" 2>&1 | tee "$org_root/mechanisms_${org_b}.log"; then org_status=1;fi
done
if ! CUDA_VISIBLE_DEVICES= "$org_python" -m "$org_module" pool-mechanisms --root "$org_root"; then org_status=1;fi
for org_experiment in end_to_end layout_control function_free geometry_control; do
  for org_b in api qwen; do
    if ! org_phase world "$org_b" --experiment "$org_experiment" 2>&1 | tee "$org_root/${org_experiment}_${org_b}.log"; then org_status=1;fi
  done
done
if ! CUDA_VISIBLE_DEVICES= "$org_python" -m "$org_module" report --root "$org_root"; then org_status=1;fi
echo "v13 organized completion: $org_root/index.html (exit_status=$org_status)"
exit "$org_status"
