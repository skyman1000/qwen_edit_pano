#!/usr/bin/env bash
# Five local inputs; frozen v13 SCM, bounded I/O and recoverable stage gates.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
mc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
mc_wild_python=${CAUPANO_WILDDET_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python}
mc_module=qwen_edit_pano.world_pipeline.completion.assembly_multiscene
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
case ${1:-} in
  test)
    CUDA_VISIBLE_DEVICES= "$mc_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_io_recovery qwen_edit_pano.world_pipeline.completion.test_multiscene qwen_edit_pano.world_pipeline.completion.test_named_recovery qwen_edit_pano.world_pipeline.completion.test_coordinate_transport qwen_edit_pano.world_pipeline.completion.test_observer_recovery -v
    exit ;;
  report)
    [[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: organized_multiscene.sh report SUITE_ROOT [NEW_VISUAL_OUTPUT]' >&2;exit 2; }
    mc_args=(report --root "$2")
    if [[ $# == 3 ]]; then mc_args+=(--out "$3");fi
    CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" "${mc_args[@]}"
    exit ;;
  prepare|prepare-single|observe|generate|run|resume|repair-generate|prepare-repair|repair-coordinates|prepare-coordinates|resume-coordinates) ;;
  *) echo 'Usage: organized_multiscene.sh observe GPU3 [NEW_OUTPUT] (2D only, no API key); resume GPU3 SUITE_ROOT; generate GPU3 [NEW_OUTPUT]; repair-coordinates|prepare-coordinates GPU3 OLD_SUITE_ROOT; resume-coordinates GPU3 PREPARED_ROOT; repair-generate|prepare-repair GPU3 OLD_SUITE_ROOT; run GPU3 [NEW_OUTPUT]; prepare|prepare-single GPU3 [NEW_OUTPUT]; report SUITE_ROOT [NEW_VISUAL_OUTPUT]; test' >&2;exit 2 ;;
esac
[[ $# -ge 2 && $# -le 3 ]] || { echo 'Specify GPU node; resume additionally requires SUITE_ROOT.' >&2;exit 2; }
mc_action=$1;mc_node=$2
if [[ $mc_action != observe && $mc_action != *coordinates && $mc_action != prepare && $mc_action != prepare-single && $mc_action != prepare-repair && -z ${SILICONFLOW_API_KEY:-} ]]; then
  echo 'Set SILICONFLOW_API_KEY for SiliconFlow deepseek-ai/DeepSeek-V4.1-Flash. Both backends generate fresh candidates and tests.' >&2
  exit 2
fi
if [[ $mc_action == repair-generate || $mc_action == prepare-repair || $mc_action == repair-coordinates || $mc_action == prepare-coordinates ]]; then
  [[ $# == 3 ]] || { echo 'Usage: organized_multiscene.sh repair-coordinates|prepare-coordinates|repair-generate|prepare-repair GPU3 OLD_SUITE_ROOT' >&2;exit 2; }
  mc_root=${CAUPANO_RECOVERY_OUTPUT:-qwen_edit_pano/outputs/organized_multiscene_recovery_$(date +%Y%m%d_%H%M%S)_$$}
  CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" fork --root "$mc_root" --source-root "$3" \
    --inputs "${CAUPANO_TEST_INPUTS:-qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl}"
elif [[ $mc_action == resume || $mc_action == resume-coordinates ]]; then
  [[ $# == 3 ]] || { echo 'resume or resume-coordinates requires an initialized SUITE_ROOT' >&2;exit 2; }
  mc_root=$3
else
  mc_root=${3:-qwen_edit_pano/outputs/organized_multiscene_$(date +%Y%m%d_%H%M%S)_$$}
  mc_mode=evaluation;mc_worlds=${CAUPANO_WORLDS:-4}
  if [[ $mc_action == generate || $mc_action == prepare-single || $mc_action == observe ]]; then mc_mode=single;mc_worlds=1;fi
  CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" prepare --root "$mc_root" \
    --inputs "${CAUPANO_TEST_INPUTS:-qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl}" \
    --mode "$mc_mode" --worlds "$mc_worlds" --draws "${CAUPANO_MECHANISM_DRAWS:-4}"
fi
if [[ $mc_action == *coordinates ]]; then
  CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" coordinate-plan --root "$mc_root"
fi
if [[ $mc_action == prepare || $mc_action == prepare-single || $mc_action == prepare-repair || $mc_action == prepare-coordinates ]]; then echo "Prepared; no model/API calls: $mc_root";exit;fi
mc_gpu() {
  srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$mc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" "$@"
}
mc_status=0
mc_phase() {
  local phase=$1 backend=${2:-qwen} interpreter=${3:-$mc_python} location=${4:-cpu}
  local args=(-u -m "$mc_module" "$phase" --root "$mc_root" --backend "$backend")
  if [[ $location == gpu ]]; then
    if ! mc_gpu "$interpreter" "${args[@]}" 2>&1 | tee -a "$mc_root/${phase}_${backend}.log"; then mc_status=1;fi
  else
    if ! CUDA_VISIBLE_DEVICES= "$interpreter" "${args[@]}" 2>&1 | tee -a "$mc_root/${phase}_${backend}.log"; then mc_status=1;fi
  fi
}
if [[ $mc_action == *coordinates ]]; then
  mc_phase worlds qwen "$mc_python" gpu
else
mc_phase observer2d qwen "$mc_python" gpu
CUDA_VISIBLE_DEVICES= "$mc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_observer_recovery --root "$mc_root"
if [[ $mc_status != 0 ]]; then
  echo "Observer2D failed; downstream stages were not launched. Inspect $mc_root/observation_index.html" >&2
  exit "$mc_status"
fi
if [[ $mc_action == observe ]]; then
  echo "Observation-only preview: $mc_root/observation_index.html"
  echo "Continue after reviewing the boxes: bash qwen_edit_pano/scripts/organized_multiscene.sh resume $mc_node $mc_root"
  exit
fi
mc_phase observer3d qwen "$mc_wild_python" gpu
mc_phase prepare-scenes
mc_phase propose api
mc_phase propose qwen "$mc_python" gpu
mc_phase pool
mc_phase mechanisms api
mc_phase mechanisms qwen "$mc_python" gpu
mc_phase pool-mechanisms
mc_phase worlds api
mc_phase worlds qwen "$mc_python" gpu
fi
if ! CUDA_VISIBLE_DEVICES= "$mc_python" -m "$mc_module" report --root "$mc_root";then mc_status=1;fi
echo "Multi-scene test and visuals: $mc_root/index.html (exit_status=$mc_status)"
exit "$mc_status"
