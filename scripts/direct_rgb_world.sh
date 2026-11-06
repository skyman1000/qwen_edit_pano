#!/usr/bin/env bash
# Raw local RGB -> one direct structured world per image/backend; no WildDet.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
dr_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
dr_module=qwen_edit_pano.world_pipeline.completion.direct_rgb_world
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
case ${1:-} in
  test) CUDA_VISIBLE_DEVICES= "$dr_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_direct_rgb -v;exit ;;
  report)
    [[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: direct_rgb_world.sh report DIRECT_ROOT [CAUSAL_SUITE]' >&2;exit 2; }
    dr_args=(report --root "$2")
    if [[ $# == 3 ]]; then dr_args+=(--causal-suite "$3");fi
    CUDA_VISIBLE_DEVICES= "$dr_python" -m "$dr_module" "${dr_args[@]}";exit ;;
  prepare|run|resume) ;;
  *) echo 'Usage: direct_rgb_world.sh run|prepare GPU3 [NEW_OUTPUT]; resume GPU3 DIRECT_ROOT; report DIRECT_ROOT [CAUSAL_SUITE]; test' >&2;exit 2 ;;
esac
[[ $# -ge 2 && $# -le 3 ]] || { echo 'Specify a GPU node; resume requires an initialized direct root.' >&2;exit 2; }
dr_action=$1;dr_node=$2
if [[ $dr_action != prepare && -z ${SILICONFLOW_API_KEY:-} ]];then echo 'Set SILICONFLOW_API_KEY for deepseek-ai/DeepSeek-V4.1-Flash on SiliconFlow.' >&2;exit 2;fi
if [[ $dr_action == resume ]];then
  [[ $# == 3 ]] || { echo 'resume requires DIRECT_ROOT' >&2;exit 2; };dr_root=$3
else
  dr_root=${3:-qwen_edit_pano/outputs/direct_rgb_world_$(date +%Y%m%d_%H%M%S)_$$}
  dr_init=(prepare --root "$dr_root" --inputs "${CAUPANO_TEST_INPUTS:-qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl}" --max-tokens "${CAUPANO_DIRECT_MAX_TOKENS:-8192}")
  if [[ -n ${CAUPANO_DIRECT_REFERENCE_SUITE:-} ]];then dr_init+=(--reference-suite "$CAUPANO_DIRECT_REFERENCE_SUITE");fi
  CUDA_VISIBLE_DEVICES= "$dr_python" -m "$dr_module" "${dr_init[@]}"
fi
if [[ $dr_action == prepare ]];then echo "Prepared RGB-only baseline; no model calls: $dr_root";exit;fi
dr_status=0
if ! CUDA_VISIBLE_DEVICES= "$dr_python" -u -m "$dr_module" generate --root "$dr_root" --backend api \
  2>&1 | tee -a "$dr_root/api.log";then dr_status=1;fi
if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$dr_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
  "$dr_python" -u -m "$dr_module" generate --root "$dr_root" --backend qwen \
  2>&1 | tee -a "$dr_root/qwen.log";then dr_status=1;fi
dr_report=(report --root "$dr_root")
if [[ -n ${CAUPANO_DIRECT_REFERENCE_SUITE:-} ]];then dr_report+=(--causal-suite "$CAUPANO_DIRECT_REFERENCE_SUITE");fi
if ! CUDA_VISIBLE_DEVICES= "$dr_python" -m "$dr_module" "${dr_report[@]}";then dr_status=1;fi
echo "Direct RGB world baseline: $dr_root/index.html (exit_status=$dr_status)"
exit "$dr_status"
