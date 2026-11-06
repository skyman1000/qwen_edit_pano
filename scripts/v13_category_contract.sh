#!/usr/bin/env bash
# Opt-in candidate interface; original v13 model/solver and frozen observations.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
vc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
vc_module=qwen_edit_pano.world_pipeline.completion.v13_category_contract
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n ${CAUPANO_PROGRAM_CATALOG:-} || ${CAUPANO_LAYOUT_RGB:-0} != 0 || ${CAUPANO_PROPOSAL_ROUNDS:-1} != 1 ]]; then
  echo 'Use fresh one-round candidates: unset CAUPANO_PROGRAM_CATALOG, CAUPANO_LAYOUT_RGB=0, CAUPANO_PROPOSAL_ROUNDS=1.' >&2
  exit 2
fi
case ${1:-} in
  run|prepare)
    [[ $# == 3 ]] || { echo 'Usage: v13_category_contract.sh run|prepare NEW_ROOT OBSERVATION_SUITE' >&2; exit 2; }
    vc_root=$2
    CUDA_VISIBLE_DEVICES= "$vc_python" -m "$vc_module" prepare --root "$vc_root" --observations-from "$3" \
      --worlds "${CAUPANO_WORLDS:-1}" --draws "${CAUPANO_DRAWS:-4}"
    if [[ $1 == prepare ]]; then exit; fi
    ;;
  resume|report)
    [[ $# == 2 ]] || { echo 'Usage: v13_category_contract.sh resume|report ROOT' >&2; exit 2; }
    vc_root=$2
    CUDA_VISIBLE_DEVICES= "$vc_python" -m "$vc_module" check --root "$vc_root"
    if [[ $1 == report ]]; then CUDA_VISIBLE_DEVICES= "$vc_python" -m "$vc_module" report --root "$vc_root"; exit; fi
    ;;
  *) echo 'Usage: v13_category_contract.sh run|prepare NEW_ROOT OBSERVATION_SUITE; resume|report ROOT' >&2; exit 2 ;;
esac
vc_status=0
for vc_phase in prepare-scenes propose pool mechanisms worlds; do
  if ! "$vc_python" -u -m "$vc_module" "$vc_phase" --root "$vc_root" 2>&1 | tee -a "$vc_root/$vc_phase.log"; then vc_status=1; fi
done
if ! CUDA_VISIBLE_DEVICES= "$vc_python" -m "$vc_module" report --root "$vc_root" 2>&1 | tee -a "$vc_root/report.log"; then vc_status=1; fi
echo "v13 dataset categories and optional relations: $vc_root/index.html (exit_status=$vc_status)"
exit "$vc_status"
