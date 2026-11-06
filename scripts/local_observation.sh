#!/usr/bin/env bash
# Only Qwen 2D -> WildDet3D -> observed-state views. No API or completion.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
lo_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
lo_wild_python=${CAUPANO_WILDDET_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python}
lo_module=qwen_edit_pano.world_pipeline.completion.local_observation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
case ${1:-} in
  report) shift;CUDA_VISIBLE_DEVICES= "$lo_python" -m "$lo_module" report "$@";exit ;;
  test) CUDA_VISIBLE_DEVICES= "$lo_python" -m unittest \
    qwen_edit_pano.world_pipeline.completion.test_local_observation \
    qwen_edit_pano.world_pipeline.completion.test_observer_lossless \
    qwen_edit_pano.world_pipeline.completion.test_restored_observer -v;exit ;;
  run|prepare|resume|local|local-resume) ;;
  *) echo 'Usage: local_observation.sh local [NEW_ROOT]; local-resume ROOT; report --source SUITE --out NEW_OUTPUT; run|prepare GPU3 [NEW_ROOT]; resume GPU3 ROOT; test' >&2;exit 2 ;;
esac
lo_action=$1;lo_local=0
if [[ $lo_action == local || $lo_action == local-resume ]];then
  [[ $# -le 2 ]] || { echo 'local [NEW_ROOT]; local-resume ROOT' >&2;exit 2; }
  lo_local=1;lo_node=allocated_gpu;lo_requested_root=${2:-}
  if [[ $lo_action == local-resume ]];then lo_action=resume;fi
else
  [[ $# -ge 2 && $# -le 3 ]] || { echo 'Specify a GPU node and optional root' >&2;exit 2; }
  lo_node=$2;lo_requested_root=${3:-}
fi
if [[ $lo_action == resume ]];then
  [[ -n $lo_requested_root ]] || { echo 'resume requires an observation-only root' >&2;exit 2; }
  lo_root=$lo_requested_root
  CUDA_VISIBLE_DEVICES= "$lo_python" -m "$lo_module" check --root "$lo_root"
else
  lo_root=${lo_requested_root:-qwen_edit_pano/outputs/local_observation_$(date +%Y%m%d_%H%M%S)_$$}
  CUDA_VISIBLE_DEVICES= "$lo_python" -m "$lo_module" prepare --root "$lo_root" \
    --inputs "${CAUPANO_TEST_INPUTS:-qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl}"
fi
if [[ $lo_action == prepare ]];then echo "Prepared local observations; no model calls: $lo_root";exit;fi
lo_phase() {
  local phase=$1 interpreter=$2
  if [[ $lo_local == 1 ]];then
    "$interpreter" -u -m qwen_edit_pano.world_pipeline.completion.assembly_multiscene "$phase" \
      --root "$lo_root" --backend qwen 2>&1 | tee -a "$lo_root/${phase}.log"
  else
    srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$lo_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
      "$interpreter" -u -m qwen_edit_pano.world_pipeline.completion.assembly_multiscene "$phase" \
      --root "$lo_root" --backend qwen 2>&1 | tee -a "$lo_root/${phase}.log"
  fi
}
lo_status=0
if ! lo_phase observer2d "$lo_python";then
  lo_status=1
  echo 'Some 2D scenes failed; the 3D stage will check dependencies per scene and process valid scenes.' >&2
fi
if ! lo_phase observer3d "$lo_wild_python";then lo_status=1;fi
# The gallery is outside the source root, preserving its snapshot.
lo_visual="${lo_root%/}_review_$(date +%Y%m%d_%H%M%S)_$$"
if ! CUDA_VISIBLE_DEVICES= "$lo_python" -m "$lo_module" report --source "$lo_root" --out "$lo_visual";then lo_status=1;fi
echo "Local observation only: $lo_visual/index.html (exit_status=$lo_status)"
exit "$lo_status"
