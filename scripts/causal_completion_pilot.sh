#!/usr/bin/env bash
# User-run experiment: same observers and layout backend across three methods.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
cc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
cc_action=${1:-test}
if [[ "$cc_action" == test || "$cc_action" == preflight ]]; then
  exec bash qwen_edit_pano/scripts/assembly_compare.sh "$cc_action"
fi
if [[ ( "$cc_action" != compare && "$cc_action" != retry-needs ) || $# -lt 4 || $# -gt 5 ]]; then
  echo 'Usage: causal_completion_pilot.sh compare GPU3 SOURCE_ROOT api|qwen|both [SEED]; retry-needs GPU3 PILOT_ROOT api|qwen|both [SEED]' >&2
  exit 2
fi
cc_node=$2
cc_source=$3
cc_backend=$4
cc_seed=${5:-}
case "$cc_backend" in
  api) cc_backends=(api) ;;
  qwen) cc_backends=(qwen) ;;
  both) cc_backends=(api qwen) ;;
  *) echo 'Backend must be api, qwen or both' >&2; exit 2 ;;
esac
if [[ "$cc_backend" != qwen && -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo 'Set SILICONFLOW_API_KEY before choosing api/both.' >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ "$cc_action" == retry-needs ]]; then
  [[ -f "$cc_source/needs/experiment.json" ]] || { echo 'retry-needs requires a pilot root containing needs/experiment.json' >&2; exit 2; }
  cc_root="${CAUPANO_PILOT_OUTPUT:-qwen_edit_pano/outputs/needs_single_schema_retry_$(date +%Y%m%d_%H%M%S)_$$}"
  cc_init=(-m qwen_edit_pano.world_pipeline.completion.assembly_experiment restart --root "$cc_root" \
    --source-root "$cc_source/needs" --completion-model needs)
  if [[ -n "$cc_seed" ]]; then cc_init+=(--seed "$cc_seed"); fi
  if [[ -n "${CAUPANO_WORLDS:-}" ]]; then cc_init+=(--worlds "$CAUPANO_WORLDS"); fi
  CUDA_VISIBLE_DEVICES= "$cc_python" "${cc_init[@]}"
  cc_methods=(needs)
else
  cc_seed=${cc_seed:-3407}
  cc_root="${CAUPANO_PILOT_OUTPUT:-qwen_edit_pano/outputs/causal_completion_pilot_$(date +%Y%m%d_%H%M%S)_$$_$cc_seed}"
  CUDA_VISIBLE_DEVICES= "$cc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_pilot prepare \
    --root "$cc_root" --source-root "$cc_source" --seed "$cc_seed" --worlds "${CAUPANO_WORLDS:-3}"
  cc_methods=(baseline joint needs)
fi
cc_status=0
for cc_method in "${cc_methods[@]}"; do
  if [[ "$cc_action" == retry-needs ]]; then cc_experiment=$cc_root; else cc_experiment="$cc_root/$cc_method"; fi
  for cc_b in "${cc_backends[@]}"; do
    cc_args=(-u -m qwen_edit_pano.world_pipeline.completion.assembly_experiment world --root "$cc_experiment" --backend "$cc_b")
    if [[ "$cc_b" == api ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$cc_python" "${cc_args[@]}" 2>&1 | tee "$cc_experiment/$cc_b/completion.log"; then cc_status=1; fi
    else
      if ! srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$cc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" \
        "$cc_python" "${cc_args[@]}" 2>&1 | tee "$cc_experiment/$cc_b/completion.log"; then cc_status=1; fi
    fi
  done
done
if [[ "$cc_action" == retry-needs ]]; then
  for cc_b in "${cc_backends[@]}"; do
    CUDA_VISIBLE_DEVICES= "$cc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_pilot_visual \
      --root "$cc_source" --backend "$cc_b" --needs-root "$cc_root"
  done
  echo "New needs experiment: $cc_root"
else
  CUDA_VISIBLE_DEVICES= "$cc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_pilot report --root "$cc_root"
  for cc_b in "${cc_backends[@]}"; do
    CUDA_VISIBLE_DEVICES= "$cc_python" -m qwen_edit_pano.world_pipeline.completion.assembly_pilot_visual --root "$cc_root" --backend "$cc_b"
  done
  echo "Pilot comparison: $cc_root/index.html"
fi
exit "$cc_status"
