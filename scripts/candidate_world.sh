#!/usr/bin/env bash
# One user-supplied srun allocation; original v13 executor, local Qwen only.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
cw_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
cw_module=qwen_edit_pano.world_pipeline.completion.candidate_repair.workflow
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cw_action=${1:-}
case "$cw_action" in
  run|candidates|prepare)
    [[ $# == 3 ]] || { echo 'Usage: candidate_world.sh run|candidates|prepare NEW_ROOT OBSERVATION_SUITE' >&2; exit 2; }
    cw_root=$2
    CUDA_VISIBLE_DEVICES= "$cw_python" -m "$cw_module" prepare --root "$cw_root" --observations-from "$3" \
      --rounds "${CAUPANO_CANDIDATE_ROUNDS:-2}" --worlds "${CAUPANO_WORLDS:-4}" --draws "${CAUPANO_DRAWS:-4}"
    [[ $cw_action != prepare ]] || exit 0
    ;;
  complete|resume|report)
    [[ $# == 2 ]] || { echo 'Usage: candidate_world.sh complete|resume|report ROOT' >&2; exit 2; }
    cw_root=$2
    CUDA_VISIBLE_DEVICES= "$cw_python" -m "$cw_module" check --root "$cw_root"
    if [[ $cw_action == report ]]; then CUDA_VISIBLE_DEVICES= "$cw_python" -m "$cw_module" report --root "$cw_root"; exit; fi
    ;;
  *) echo 'Usage: candidate_world.sh run|candidates|prepare NEW_ROOT OBSERVATION_SUITE; complete|resume|report ROOT' >&2; exit 2 ;;
esac
cw_status=0
for cw_phase in prepare-scenes propose pool mechanisms worlds; do
  if ! "$cw_python" -u -m "$cw_module" "$cw_phase" --root "$cw_root" 2>&1 | tee -a "$cw_root/$cw_phase.log"; then cw_status=2; fi
  if [[ $cw_phase == pool ]]; then
    cw_preview=0
    CUDA_VISIBLE_DEVICES= "$cw_python" -m "$cw_module" candidate-report --root "$cw_root" || cw_preview=$?
    if (( cw_preview > cw_status )); then cw_status=$cw_preview; fi
    if [[ $cw_action == candidates ]]; then
      echo "Candidate objects/dependencies: $cw_root/candidates.html (no 3D layout requested; exit_status=$cw_status)"
      exit "$cw_status"
    fi
  fi
done
cw_report=0
CUDA_VISIBLE_DEVICES= "$cw_python" -m "$cw_module" report --root "$cw_root" 2>&1 | tee -a "$cw_root/report.log" || cw_report=$?
if (( cw_report > cw_status )); then cw_status=$cw_report; fi
echo "Results: $cw_root/index.html (exit_status=$cw_status; execution errors, empty samples and layout rejections are separate in results.json)"
exit "$cw_status"
