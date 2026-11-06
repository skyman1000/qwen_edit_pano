#!/usr/bin/env bash
# Run inside the user's srun allocation. Local Qwen only; no new observer pass.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
sw_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
sw_module=qwen_edit_pano.world_pipeline.completion.scope_world.workflow
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
sw_action=${1:-}
case "$sw_action" in
  run|prepare)
    [[ $# == 3 ]] || { echo 'Usage: scope_world.sh run|prepare NEW_ROOT OBSERVATION_SUITE' >&2; exit 2; }
    sw_root=$2
    CUDA_VISIBLE_DEVICES= "$sw_python" -m "$sw_module" prepare --root "$sw_root" --observations-from "$3" --worlds "${CAUPANO_WORLDS:-2}"
    [[ $sw_action != prepare ]] || exit 0
    ;;
  resume|report)
    [[ $# == 2 ]] || { echo 'Usage: scope_world.sh resume|report ROOT' >&2; exit 2; }
    sw_root=$2
    CUDA_VISIBLE_DEVICES= "$sw_python" -m "$sw_module" check --root "$sw_root"
    if [[ $sw_action == report ]]; then CUDA_VISIBLE_DEVICES= "$sw_python" -m "$sw_module" report --root "$sw_root"; exit; fi
    ;;
  *) echo 'Usage: scope_world.sh run|prepare NEW_ROOT OBSERVATION_SUITE; resume|report ROOT' >&2; exit 2 ;;
esac
sw_status=0
for sw_phase in prepare-scenes propose worlds; do
  if ! "$sw_python" -u -m "$sw_module" "$sw_phase" --root "$sw_root" 2>&1 | tee -a "$sw_root/$sw_phase.log"; then sw_status=2; fi
done
sw_report=0
CUDA_VISIBLE_DEVICES= "$sw_python" -m "$sw_module" report --root "$sw_root" 2>&1 | tee -a "$sw_root/report.log" || sw_report=$?
if (( sw_report > sw_status )); then sw_status=$sw_report; fi
echo "Results: $sw_root/index.html (exit_status=$sw_status; see results.json for execution errors / rejected candidates / empty worlds)"
exit "$sw_status"
