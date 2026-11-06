#!/usr/bin/env bash
# User supplies the srun allocation. No API, observer inference, or nested jobs.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
pv_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
pv_module=qwen_edit_pano.world_pipeline.completion.plan_repair.workflow
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pv_action=${1:-}
case "$pv_action" in
  run|plan|prepare)
    [[ $# == 3 ]] || { echo 'Usage: planned_v13.sh run|plan|prepare NEW_ROOT OBSERVATION_SUITE' >&2; exit 2; }
    pv_root=$2
    CUDA_VISIBLE_DEVICES= "$pv_python" -m "$pv_module" prepare --root "$pv_root" --observations-from "$3" \
      --plans "${CAUPANO_PLANS:-2}" --worlds "${CAUPANO_WORLDS_PER_PLAN:-2}" \
      --draws "${CAUPANO_DRAWS:-4}" --seed "${CAUPANO_PLAN_SEED:-3407}"
    [[ $pv_action != prepare ]] || exit 0
    ;;
  resume|report)
    [[ $# == 2 ]] || { echo 'Usage: planned_v13.sh resume|report ROOT' >&2; exit 2; }
    pv_root=$2
    CUDA_VISIBLE_DEVICES= "$pv_python" -m "$pv_module" check --root "$pv_root"
    if [[ $pv_action == report ]]; then
      CUDA_VISIBLE_DEVICES= "$pv_python" -m "$pv_module" report --root "$pv_root"
      exit
    fi
    ;;
  *) echo 'Usage: planned_v13.sh run|plan|prepare NEW_ROOT OBSERVATION_SUITE; resume|report ROOT' >&2; exit 2 ;;
esac
pv_status=0
for pv_phase in prepare-scenes plan propose pool mechanisms worlds; do
  if ! "$pv_python" -u -m "$pv_module" "$pv_phase" --root "$pv_root" 2>&1 | tee -a "$pv_root/$pv_phase.log"; then pv_status=2; fi
  # Optional proposal-only inspection; default run continues to actual 3D.
  if [[ $pv_action == plan && $pv_phase == propose ]]; then break; fi
done
pv_report=0
CUDA_VISIBLE_DEVICES= "$pv_python" -m "$pv_module" report --root "$pv_root" 2>&1 | tee -a "$pv_root/report.log" || pv_report=$?
if (( pv_report > pv_status )); then pv_status=$pv_report; fi
echo "Results: $pv_root/index.html (exit_status=$pv_status; 2=execution error, 1=scene lacks nonempty export, 0=all scenes have a nonempty export requiring review)"
exit "$pv_status"
