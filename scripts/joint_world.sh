#!/usr/bin/env bash
# One user-controlled allocation, local Qwen only. No nested srun.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
jw_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
jw_module=qwen_edit_pano.world_pipeline.completion.joint_world
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n ${CAUPANO_PROGRAM_CATALOG:-} ]];then echo 'Unset CAUPANO_PROGRAM_CATALOG for fresh hypotheses.' >&2;exit 2;fi
jw_action=${1:-}
case "$jw_action" in
  repair|preview)
    [[ $# == 3 ]] || { echo 'Usage: joint_world.sh repair|preview NEW_ROOT SOURCE_COMPLETION_RUN' >&2;exit 2; }
    jw_root=$2
    if [[ $jw_action == preview ]];then
      CUDA_VISIBLE_DEVICES= "$jw_python" -m "$jw_module" preview --root "$jw_root" --source-run "$3"
      echo "CPU preview: $jw_root/index.html (missing metric worlds remain pending; no model inference)"
      exit
    fi
    CUDA_VISIBLE_DEVICES= "$jw_python" -m "$jw_module" prepare-repair --root "$jw_root" --source-run "$3"
    ;;
  run|prepare)
    [[ $# == 3 ]] || { echo 'Usage: joint_world.sh run|prepare NEW_ROOT OBSERVATION_SUITE' >&2;exit 2; }
    jw_root=$2
    CUDA_VISIBLE_DEVICES= "$jw_python" -m "$jw_module" prepare --root "$jw_root" --observations-from "$3" \
      --worlds "${CAUPANO_WORLDS:-4}" --draws "${CAUPANO_DRAWS:-4}"
    if [[ $jw_action == prepare ]];then exit;fi
    ;;
  resume|report)
    [[ $# == 2 ]] || { echo 'Usage: joint_world.sh resume|report ROOT' >&2;exit 2; }
    jw_root=$2
    CUDA_VISIBLE_DEVICES= "$jw_python" -m "$jw_module" check --root "$jw_root"
    if [[ $jw_action == report ]];then CUDA_VISIBLE_DEVICES= "$jw_python" -m "$jw_module" report --root "$jw_root";exit;fi
    ;;
  *) echo 'Usage: joint_world.sh run|prepare NEW_ROOT OBSERVATION_SUITE; repair|preview NEW_ROOT SOURCE_COMPLETION_RUN; resume|report ROOT' >&2;exit 2 ;;
esac
jw_status=0
for jw_phase in propose mechanisms worlds;do
  if ! "$jw_python" -u -m "$jw_module" "$jw_phase" --root "$jw_root" 2>&1 | tee -a "$jw_root/$jw_phase.log";then jw_status=1;fi
done
if ! CUDA_VISIBLE_DEVICES= "$jw_python" -m "$jw_module" report --root "$jw_root" 2>&1 | tee -a "$jw_root/report.log";then jw_status=1;fi
echo "Results: $jw_root/index.html (exit_status=$jw_status; see results.json for failed/empty/exported distinctions)"
exit "$jw_status"
