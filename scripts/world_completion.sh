#!/usr/bin/env bash
# Restored default: original v13 only. Experimental workflows have separate entries.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
wc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_world_mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Keep the restored baseline as default. Candidate-contract changes are explicit.
wc_interface=${CAUPANO_PROPOSAL_INTERFACE:-original}
if [[ $wc_interface == closed_optional_v2 ]]; then
  exec bash qwen_edit_pano/scripts/v13_contract_fix.sh "$@"
elif [[ $wc_interface == closed_optional_v1 ]]; then
  exec bash qwen_edit_pano/scripts/v13_category_contract.sh "$@"
elif [[ $wc_interface != original ]]; then
  echo 'CAUPANO_PROPOSAL_INTERFACE must be original, closed_optional_v1, or closed_optional_v2.' >&2
  exit 2
fi
if [[ ( ${1:-} == resume || ${1:-} == report ) && $# == 2 && -f $2/v13_category_contract.json ]]; then
  echo 'This run uses the opt-in category contract. Set CAUPANO_PROPOSAL_INTERFACE=closed_optional_v1.' >&2
  exit 2
fi

# Do not silently change the restored protocol through a previous experiment's env.
if [[ ${CAUPANO_PROPOSAL_ROUNDS:-1} != 1 ]]; then
  echo 'Restored v13 uses one original proposal round. Unset CAUPANO_PROPOSAL_ROUNDS or set it to 1.' >&2
  exit 2
fi
if [[ ( ${1:-} == resume || ${1:-} == report ) && $# == 2 && -f $2/v13_proposal_budget.json ]]; then
  echo 'This directory belongs to the historical two-round experiment, not restored v13. Preserve it; use its original v13_workflow module to inspect it.' >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES= "$wc_python" -m qwen_edit_pano.world_pipeline.completion.v13_workflow verify
exec bash qwen_edit_pano/scripts/organized_local_world.sh "$@"
