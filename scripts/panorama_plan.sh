#!/usr/bin/env bash
# Local Qwen only. Does not submit jobs. Separate entry preserves pinned v13.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
pp_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
pp_module=qwen_edit_pano.world_pipeline.completion.panorama_plan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pp_action=${1:-}
case "$pp_action" in
  run|plan|prepare)
    [[ $# == 3 ]] || { echo 'Usage: panorama_plan.sh run|plan|prepare NEW_ROOT OBSERVATION_SUITE' >&2; exit 2; }
    pp_root=$2
    CUDA_VISIBLE_DEVICES= "$pp_python" -m "$pp_module" prepare --root "$pp_root" --observations-from "$3" \
      --hypotheses "${CAUPANO_PLAN_HYPOTHESES:-3}" --seed "${CAUPANO_PLAN_SEED:-3407}"
    [[ $pp_action != prepare ]] || exit 0
    ;;
  resume|layout|report)
    [[ $# == 2 ]] || { echo 'Usage: panorama_plan.sh resume|layout|report ROOT' >&2; exit 2; }
    pp_root=$2
    if [[ $pp_action == report ]]; then
      CUDA_VISIBLE_DEVICES= "$pp_python" -m "$pp_module" report --root "$pp_root"
      exit
    fi
    pp_action=run
    ;;
  *) echo 'Usage: panorama_plan.sh run|plan|prepare NEW_ROOT OBSERVATION_SUITE; resume|layout|report ROOT' >&2; exit 2 ;;
esac
pp_status=0
"$pp_python" -u -m "$pp_module" "$pp_action" --root "$pp_root" 2>&1 | tee -a "$pp_root/$pp_action.log" || pp_status=$?
CUDA_VISIBLE_DEVICES= "$pp_python" -m "$pp_module" report --root "$pp_root" || pp_status=$?
echo "Results: $pp_root/index.html (exit_status=$pp_status; layout/semantic warnings are separate from execution failures)"
exit "$pp_status"
