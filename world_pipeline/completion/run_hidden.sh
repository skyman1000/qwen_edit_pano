#!/usr/bin/env bash
# Manual entry points. No Qwen/Adapter training is launched here.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
HIDDEN_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
HIDDEN_DATA=qwen_edit_pano/data/hidden_completion_v1
HIDDEN_RUN=${HIDDEN_RUN:-qwen_edit_pano/outputs/hidden_completion_run001}
HIDDEN_VAL=${HIDDEN_VAL:-$HIDDEN_RUN/val}
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
gpu() {
  srun -p debug --nodelist="${HIDDEN_NODE:-GPU2}" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --mem=32G --qos=normal --time=02:00:00 "$HIDDEN_PY" -u "$@"
}
case "${1:-}" in
  prepare)
    "$HIDDEN_PY" -m qwen_edit_pano.world_pipeline.completion.hidden_data --output "$HIDDEN_DATA"
    ;;
  train)
    gpu -m qwen_edit_pano.world_pipeline.completion.train_hidden \
      --data "$HIDDEN_DATA" --output "$HIDDEN_RUN" --steps 20000 --batch-size 16 --validate-every 1000
    ;;
  resume)
    gpu -m qwen_edit_pano.world_pipeline.completion.train_hidden \
      --data "$HIDDEN_DATA" --output "$HIDDEN_RUN" --resume "$HIDDEN_RUN/last.pt" \
      --steps 20000 --batch-size 16 --validate-every 1000
    ;;
  val)
    gpu -m qwen_edit_pano.world_pipeline.completion.sample_hidden \
      --checkpoint "$HIDDEN_RUN/best.pt" --val-data "$HIDDEN_DATA" --seeds 0 \
      --output "$HIDDEN_VAL"
    ;;
  evaluate)
    "$HIDDEN_PY" -m qwen_edit_pano.world_pipeline.completion.evaluate_hidden \
      --data "$HIDDEN_DATA" --predictions "$HIDDEN_VAL"
    ;;
  predict)
    gpu -m qwen_edit_pano.world_pipeline.completion.sample_hidden \
      --checkpoint "$HIDDEN_RUN/best.pt" --observed "${2:?Pass an Observer G_obs.json path}" \
      --seeds 0 1 2 --output "${HIDDEN_PRED:-$HIDDEN_RUN/predicted}"
    ;;
  *) echo 'Usage: bash run_hidden.sh prepare|train|resume|val|evaluate|predict [G_obs.json]'; exit 2 ;;
esac
