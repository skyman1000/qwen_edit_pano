#!/usr/bin/env bash
# Independent API and Qwen pipelines. No historical run.sh or cached domain.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
fc_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
fc_wild_python=${CAUPANO_WILDDET_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python}
fc_module=qwen_edit_pano.world_pipeline.completion.fresh_pipeline
fc_action=${1:-preflight}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
if [[ "$fc_action" == test ]]; then
  CUDA_VISIBLE_DEVICES= "$fc_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_fresh -v
  exit
fi
if [[ "$fc_action" == review ]]; then
  [[ $# == 2 ]] || { echo 'Usage: fresh_compare.sh review EXPERIMENT_ROOT' >&2; exit 2; }
  CUDA_VISIBLE_DEVICES= "$fc_python" -m "$fc_module" review --root "$2"
  exit
fi
[[ "$fc_action" == preflight || "$fc_action" == run ]] || { echo 'Actions: preflight, test, run GPU_NODE, review EXPERIMENT_ROOT' >&2; exit 2; }
CUDA_VISIBLE_DEVICES= "$fc_python" -m "$fc_module" preflight
CUDA_VISIBLE_DEVICES= "$fc_wild_python" -m qwen_edit_pano.world_pipeline.completion.fresh_wilddet --preflight
if [[ "$fc_action" == preflight ]]; then exit; fi
[[ $# == 2 && -n "$2" ]] || { echo 'Usage: fresh_compare.sh run GPU1' >&2; exit 2; }
[[ -n "${SILICONFLOW_API_KEY:-}" ]] || { echo 'Set SILICONFLOW_API_KEY in the environment first.' >&2; exit 2; }
fc_node=$2
fc_init=(-m "$fc_module" init)
if [[ -n "${CAUPANO_OUTPUT:-}" ]]; then fc_init+=(--root "$CAUPANO_OUTPUT"); fi
fc_root=$(CUDA_VISIBLE_DEVICES= "$fc_python" "${fc_init[@]}")
echo "NEW experiment: $fc_root"
fc_status=0
fc_gpu() {
  srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$fc_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" "$@"
}
for fc_backend in api qwen; do
  echo "Starting independent branch: $fc_backend"
  if [[ "$fc_backend" == api ]]; then
    if ! CUDA_VISIBLE_DEVICES= "$fc_python" -u -m "$fc_module" observer2d --root "$fc_root" --backend api 2>&1 | tee "$fc_root/api/observer2d.log"; then
      fc_status=1; continue
    fi
  else
    if ! fc_gpu "$fc_python" -u -m "$fc_module" observer2d --root "$fc_root" --backend qwen 2>&1 | tee "$fc_root/qwen/observer2d.log"; then
      fc_status=1; continue
    fi
  fi
  if ! fc_gpu "$fc_wild_python" -u -m qwen_edit_pano.world_pipeline.completion.fresh_wilddet \
      --root "$fc_root" --backend "$fc_backend" 2>&1 | tee "$fc_root/$fc_backend/observer3d.log"; then
    fc_status=1; continue
  fi
  if [[ "$fc_backend" == api ]]; then
    if ! CUDA_VISIBLE_DEVICES= "$fc_python" -u -m "$fc_module" world --root "$fc_root" --backend api 2>&1 | tee "$fc_root/api/completion.log"; then fc_status=1; fi
  else
    if ! fc_gpu "$fc_python" -u -m "$fc_module" world --root "$fc_root" --backend qwen 2>&1 | tee "$fc_root/qwen/completion.log"; then fc_status=1; fi
  fi
done
CUDA_VISIBLE_DEVICES= "$fc_python" -m "$fc_module" review --root "$fc_root"
echo "Experiment and complete per-stage logs: $fc_root"
exit "$fc_status"
