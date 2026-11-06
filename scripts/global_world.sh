#!/usr/bin/env bash
# Local-only global SCM. Separate Qwen/WildDet3D processes; never starts an API.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
gw_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
gw_wild_python=${CAUPANO_WILDDET_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python}
gw_module=qwen_edit_pano.world_pipeline.completion.global_world
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/caupano_global_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
case ${1:-} in
  test)
    CUDA_VISIBLE_DEVICES= "$gw_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_global_world \
      qwen_edit_pano.world_pipeline.completion.test_global_world_v2_replay \
      qwen_edit_pano.world_pipeline.completion.test_global_world_v3_replay \
      qwen_edit_pano.world_pipeline.completion.test_restored_observer -v
    exit ;;
  report|check)
    gw_action=$1;shift
    [[ $# == 1 ]] || { echo 'Specify the run directory' >&2;exit 2; }
    CUDA_VISIBLE_DEVICES= "$gw_python" -m "$gw_module" "$gw_action" --root "$1"
    exit ;;
  prepare|run|resume|local|local-resume) ;;
  *) echo 'Usage: global_world.sh prepare|run GPU3 [NEW_ROOT]; resume GPU3 ROOT; local [NEW_ROOT]; local-resume ROOT; report|check ROOT; test' >&2;exit 2 ;;
esac
gw_action=$1;gw_local=0
if [[ $gw_action == local || $gw_action == local-resume ]];then
  [[ $# -le 2 ]] || { echo 'local [NEW_ROOT]; local-resume ROOT' >&2;exit 2; }
  gw_local=1;gw_node=allocated_gpu
  gw_requested_root=${2:-}
else
  [[ $# -ge 2 && $# -le 3 ]] || { echo 'Specify a GPU node and optional output directory' >&2;exit 2; }
  gw_node=$2;gw_requested_root=${3:-}
fi
if [[ $gw_action == resume || $gw_action == local-resume ]];then
  [[ -n $gw_requested_root ]] || { echo 'resume requires the existing run directory' >&2;exit 2; }
  gw_root=$gw_requested_root
  CUDA_VISIBLE_DEVICES= "$gw_python" -m "$gw_module" check --root "$gw_root"
else
  gw_root=${gw_requested_root:-qwen_edit_pano/outputs/global_world_$(date +%Y%m%d_%H%M%S)_$$}
  gw_args=(--root "$gw_root" --worlds "${CAUPANO_WORLDS:-6}" --hypotheses "${CAUPANO_HYPOTHESES:-3}"
    --alternatives "${CAUPANO_ALTERNATIVES:-2}" --draws "${CAUPANO_DRAWS:-4}"
    --seed "${CAUPANO_SEED:-3407}" --maximum-objects "${CAUPANO_MAX_OBJECTS:-48}"
    --solver-starts "${CAUPANO_SOLVER_STARTS:-3}" --solver-iterations "${CAUPANO_SOLVER_ITERATIONS:-180}")
  if [[ -n ${CAUPANO_IMAGE:-} ]];then
    gw_args+=(--image "$CAUPANO_IMAGE" --hfov "${CAUPANO_HFOV:-90}")
  elif [[ -n ${CAUPANO_TEST_INPUTS:-} ]];then
    gw_args+=(--inputs "$CAUPANO_TEST_INPUTS")
  elif [[ -z ${CAUPANO_RECOVER_FROM:-} ]];then
    gw_args+=(--inputs qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl)
  fi
  if [[ -n ${CAUPANO_QWEN_MODEL:-} ]];then gw_args+=(--local-model "$CAUPANO_QWEN_MODEL");fi
  if [[ -n ${CAUPANO_OBSERVATIONS_FROM:-} ]];then gw_args+=(--observations-from "$CAUPANO_OBSERVATIONS_FROM");fi
  if [[ -n ${CAUPANO_RECOVER_FROM:-} ]];then gw_args+=(--recover-from "$CAUPANO_RECOVER_FROM");fi
  CUDA_VISIBLE_DEVICES= "$gw_python" -m "$gw_module" prepare "${gw_args[@]}"
fi
if [[ $gw_action == prepare ]];then echo "Prepared only, no model calls: $gw_root";exit;fi
gw_phase() {
  local phase=$1 interpreter=$2
  if [[ $gw_local == 1 ]];then
    # Already inside the user's srun allocation. Never submit a nested job.
    "$interpreter" -u -m "$gw_module" "$phase" --root "$gw_root" 2>&1 | tee -a "$gw_root/${phase}.log"
  else
    srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$gw_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem="${CAUPANO_MEMORY:-48G}" --qos=normal --time="${CAUPANO_TIME_LIMIT:-04:00:00}" \
      "$interpreter" -u -m "$gw_module" "$phase" --root "$gw_root" 2>&1 | tee -a "$gw_root/${phase}.log"
  fi
}
gw_status=0
# Failures stay recorded; later stages still process the other valid scenes.
if ! gw_phase observer2d "$gw_python";then gw_status=1;fi
if ! gw_phase observer3d "$gw_wild_python";then gw_status=1;fi
if ! gw_phase complete "$gw_python";then gw_status=1;fi
if ! CUDA_VISIBLE_DEVICES= "$gw_python" -m "$gw_module" report --root "$gw_root";then gw_status=1;fi
echo "Results: $gw_root/index.html (exit_status=$gw_status; see per-scene accepted/rejected/failed counts; 1 means a stage failed or a scene has no accepted world)"
exit "$gw_status"
