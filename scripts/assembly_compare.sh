#!/usr/bin/env bash
# Two interpretable comparisons: independent RGB observers, then one frozen G_obs.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
ac_python=${CAUPANO_QWEN_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python}
ac_wild_python=${CAUPANO_WILDDET_PYTHON:-/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python}
ac_module=qwen_edit_pano.world_pipeline.completion.assembly_experiment
ac_action=${1:-preflight}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAM3_COMPILE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/caupano_review_mpl
case "$ac_action" in
  test) CUDA_VISIBLE_DEVICES= "$ac_python" -m unittest qwen_edit_pano.world_pipeline.completion.test_assembly qwen_edit_pano.world_pipeline.completion.test_assembly_inventory qwen_edit_pano.world_pipeline.completion.test_assembly_delta qwen_edit_pano.world_pipeline.completion.test_assembly_needs qwen_edit_pano.world_pipeline.completion.test_assembly_functional qwen_edit_pano.world_pipeline.completion.test_assembly_population qwen_edit_pano.world_pipeline.completion.test_sdscene qwen_edit_pano.world_pipeline.completion.test_modular qwen_edit_pano.world_pipeline.completion.test_activation qwen_edit_pano.world_pipeline.completion.test_realization qwen_edit_pano.world_pipeline.completion.test_mechanism qwen_edit_pano.world_pipeline.completion.test_panorama qwen_edit_pano.world_pipeline.completion.test_assembly_set qwen_edit_pano.world_pipeline.completion.test_support_audit qwen_edit_pano.world_pipeline.completion.test_pilot_visual qwen_edit_pano.world_pipeline.completion.test_fresh -v; exit ;;
  review)
    shift
    [[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: assembly_compare.sh review ROOT [OTHER_ROOT]' >&2; exit 2; }
    ac_args=(-m "$ac_module" review --root "$1")
    if [[ $# == 2 ]]; then ac_args+=(--other-root "$2"); fi
    CUDA_VISIBLE_DEVICES= "$ac_python" "${ac_args[@]}"; exit ;;
  preflight|run|run2|controlled|retry|retry2) ;;
  *) echo 'Actions: preflight; test; run GPU1; run2 GPU1; controlled GPU1 SOURCE_ROOT; retry GPU1 SOURCE_ROOT; retry2 GPU1 FIRST_ROOT SECOND_ROOT; review ROOT [OTHER_ROOT]' >&2; exit 2 ;;
esac
if [[ "$ac_action" != retry && "$ac_action" != retry2 ]]; then
  CUDA_VISIBLE_DEVICES= "$ac_python" -m "$ac_module" preflight
  CUDA_VISIBLE_DEVICES= "$ac_wild_python" -m qwen_edit_pano.world_pipeline.completion.fresh_wilddet --preflight
fi
if [[ "$ac_action" == preflight ]]; then exit; fi
[[ $# -ge 2 && -n "$2" ]] || { echo 'Specify a GPU node, e.g. run2 GPU1' >&2; exit 2; }
[[ -n "${SILICONFLOW_API_KEY:-}" ]] || { echo 'Set SILICONFLOW_API_KEY first.' >&2; exit 2; }
ac_node=$2
ac_gpu() {
  srun -p "${CAUPANO_PARTITION:-debug}" --nodelist="$ac_node" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --qos=normal --time="${CAUPANO_TIME_LIMIT:-02:00:00}" "$@"
}
ac_status=0
ac_one() {
  local source_root=${1:-}
  local ac_init=(-m "$ac_module" init)
  if [[ "$ac_action" == retry || "$ac_action" == retry2 ]]; then
    ac_init=(-m "$ac_module" restart --source-root "$source_root")
  elif [[ -n "$source_root" ]]; then
    ac_init+=(--observations-from "$source_root" --shared-observer "${CAUPANO_SHARED_OBSERVER:-qwen}")
  fi
  if [[ -n "${CAUPANO_COMPLETION_MODEL:-}" ]]; then ac_init+=(--completion-model "$CAUPANO_COMPLETION_MODEL"); fi
  if [[ -n "${CAUPANO_SEED:-}" ]]; then ac_init+=(--seed "$CAUPANO_SEED"); fi
  if [[ -n "${CAUPANO_WORLDS:-}" ]]; then ac_init+=(--worlds "$CAUPANO_WORLDS"); fi
  ac_latest=$(CUDA_VISIBLE_DEVICES= "$ac_python" "${ac_init[@]}")
  echo "NEW experiment: $ac_latest"
  for ac_backend in api qwen; do
    if [[ -z "$source_root" ]]; then
      if [[ "$ac_backend" == api ]]; then
        if ! CUDA_VISIBLE_DEVICES= "$ac_python" -u -m "$ac_module" observer2d --root "$ac_latest" --backend api 2>&1 | tee "$ac_latest/api/observer2d.log"; then ac_status=1; continue; fi
      else
        if ! ac_gpu "$ac_python" -u -m "$ac_module" observer2d --root "$ac_latest" --backend qwen 2>&1 | tee "$ac_latest/qwen/observer2d.log"; then ac_status=1; continue; fi
      fi
      if ! ac_gpu "$ac_wild_python" -u -m "$ac_module" observer3d --root "$ac_latest" --backend "$ac_backend" 2>&1 | tee "$ac_latest/$ac_backend/observer3d.log"; then ac_status=1; continue; fi
    fi
    if [[ "$ac_backend" == api ]]; then
      if ! CUDA_VISIBLE_DEVICES= "$ac_python" -u -m "$ac_module" world --root "$ac_latest" --backend api 2>&1 | tee "$ac_latest/api/completion.log"; then ac_status=1; fi
    else
      if ! ac_gpu "$ac_python" -u -m "$ac_module" world --root "$ac_latest" --backend qwen 2>&1 | tee "$ac_latest/qwen/completion.log"; then ac_status=1; fi
    fi
  done
  CUDA_VISIBLE_DEVICES= "$ac_python" -m "$ac_module" review --root "$ac_latest"
}
if [[ "$ac_action" == retry || "$ac_action" == retry2 ]]; then
  if [[ "$ac_action" == retry ]]; then
    [[ $# == 3 ]] || { echo 'Usage: retry GPU1 SOURCE_ROOT' >&2; exit 2; }
    ac_one "$3"
  else
    [[ $# == 4 ]] || { echo 'Usage: retry2 GPU1 FIRST_ROOT SECOND_ROOT' >&2; exit 2; }
    ac_one "$3"
    ac_first=$ac_latest
    ac_one "$4"
    CUDA_VISIBLE_DEVICES= "$ac_python" -m "$ac_module" review --root "$ac_first" --other-root "$ac_latest"
  fi
elif [[ "$ac_action" == controlled ]]; then
  [[ $# == 3 ]] || { echo 'Usage: controlled GPU1 SOURCE_ROOT' >&2; exit 2; }
  ac_one "$3"
else
  [[ $# == 2 ]] || { echo 'Usage: run|run2 GPU1' >&2; exit 2; }
  ac_one
  ac_first=$ac_latest
  if [[ "$ac_action" == run2 ]]; then
    [[ -f "$ac_first/${CAUPANO_SHARED_OBSERVER:-qwen}/observer3d/G_obs.json" ]] || { echo "Shared observation not completed: $ac_first" >&2; exit 1; }
    ac_one "$ac_first"
    CUDA_VISIBLE_DEVICES= "$ac_python" -m "$ac_module" review --root "$ac_first" --other-root "$ac_latest"
  fi
fi
echo "Last experiment: $ac_latest"
exit "$ac_status"
