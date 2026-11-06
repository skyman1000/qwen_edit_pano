#!/usr/bin/env bash
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
PANO_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
case "${1:-}" in
  prepare)
    "$PANO_PY" -m qwen_edit_pano.world_pipeline.completion.native_probe prepare
    ;;
  run)
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export OMP_NUM_THREADS=4
    srun -p debug --nodelist="${M3D_NODE:-GPU2}" --nodes=1 --ntasks=1 --gres=gpu:1 \
      --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
      "$PANO_PY" -u -m qwen_edit_pano.world_pipeline.completion.native_probe run \
      --output "${M3D_RUN:-qwen_edit_pano/outputs/m3dlayout_native_probe_run001}" --seeds 0 1 2
    ;;
  *) echo 'Usage: bash run_native_probe.sh prepare|run'; exit 2 ;;
esac
