#!/usr/bin/env bash
# Run inside one user-supplied srun allocation. No nested srun or API stages.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
ow_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
ow_module=qwen_edit_pano.world_pipeline.completion.organized_local_world
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n ${CAUPANO_PROGRAM_CATALOG:-} || ${CAUPANO_LAYOUT_RGB:-0} != 0 ]];then
  echo 'Unset CAUPANO_PROGRAM_CATALOG and use CAUPANO_LAYOUT_RGB=0 for the original fresh v13 protocol.' >&2
  exit 2
fi
ow_action=${1:-}
case "$ow_action" in
  run|prepare)
    [[ $# == 3 ]] || { echo 'Usage: organized_local_world.sh run|prepare NEW_ROOT OBSERVATION_SUITE' >&2;exit 2; }
    ow_root=$2
    CUDA_VISIBLE_DEVICES= "$ow_python" -m "$ow_module" prepare --root "$ow_root" --observations-from "$3" \
      --worlds "${CAUPANO_WORLDS:-4}" --draws "${CAUPANO_DRAWS:-4}"
    if [[ $ow_action == prepare ]];then exit;fi
    ;;
  resume|report)
    [[ $# == 2 ]] || { echo 'Usage: organized_local_world.sh resume|report ROOT' >&2;exit 2; }
    ow_root=$2
    CUDA_VISIBLE_DEVICES= "$ow_python" -m "$ow_module" check --root "$ow_root"
    if [[ $ow_action == report ]];then
      CUDA_VISIBLE_DEVICES= "$ow_python" -m "$ow_module" report --root "$ow_root"
      exit
    fi
    ;;
  *) echo 'Usage: organized_local_world.sh run|prepare NEW_ROOT OBSERVATION_SUITE; resume|report ROOT' >&2;exit 2 ;;
esac
ow_status=0
# One process per stage releases model memory; successful scene markers are
# reused on resume. Missing prerequisites are recorded per scene, never faked.
for ow_phase in prepare-scenes propose pool mechanisms worlds;do
  if ! "$ow_python" -u -m "$ow_module" "$ow_phase" --root "$ow_root" 2>&1 | tee -a "$ow_root/$ow_phase.log";then
    ow_status=1
  fi
done
if ! CUDA_VISIBLE_DEVICES= "$ow_python" -m "$ow_module" report --root "$ow_root" 2>&1 | tee -a "$ow_root/report.log";then ow_status=1;fi
echo "Full v13 worlds: $ow_root/index.html (exit_status=$ow_status; see report/results.json for per-world causes)"
exit "$ow_status"
