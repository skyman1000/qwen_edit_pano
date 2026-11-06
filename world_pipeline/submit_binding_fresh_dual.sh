#!/usr/bin/env bash
#SBATCH --job-name=world-bind-fresh
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-bind-fresh_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
fresh_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
fresh_steps="${1:-10000}"
[[ "$fresh_steps" =~ ^[1-9][0-9]{0,4}$ ]] && ((fresh_steps>=2 && fresh_steps<=10000)) || {
  echo 'Usage: sbatch [options] submit_binding_fresh_dual.sh [TOTAL_STEPS:2..10000]' >&2; exit 2;
}
fresh_root=qwen_edit_pano/outputs/object_world_binding_fresh_dual_v1
mkdir -p "$fresh_root"
# Hold this across the shell's requeue decision, in addition to the Python lock.
exec 9>"$fresh_root/LAUNCHER.lock"
flock -n 9 || { echo 'Another launcher is using this experiment.' >&2; exit 2; }
fresh_stop="$fresh_root/STOP-${SLURM_JOB_ID:-manual}-${SLURM_RESTART_COUNT:-0}-$(date +%s%N)"
fresh_timeout=0
fresh_cancelled=0
trap 'fresh_timeout=1; : > "$fresh_stop"' USR1
trap 'fresh_cancelled=1; : > "$fresh_stop"' TERM INT
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Fresh binding; two GPUs train one model; global batch=4; target=$fresh_steps; automatic evaluation disabled."
"$fresh_python" -u -m qwen_edit_pano.world_pipeline.binding_fresh_world.run \
  --root "$fresh_root" --steps "$fresh_steps" --stop-file "$fresh_stop" &
fresh_pid=$!
while true; do
  if wait "$fresh_pid"; then fresh_status=0; else fresh_status=$?; fi
  if kill -0 "$fresh_pid" 2>/dev/null; then continue; fi
  break
done
[[ "$fresh_status" == 0 ]] || exit "$fresh_status"
if [[ "$fresh_timeout" == 1 && "$fresh_cancelled" == 0 && ! -f "$fresh_root/STOP" && -n "${SLURM_JOB_ID:-}" ]]; then
  # A resubmission is idempotent, including a stop immediately after the final step.
  echo 'Time limit approaching; saved complete steps, requeueing to resume.'
  scontrol requeue "$SLURM_JOB_ID"
else
  echo 'Job ended. Reuse the same sbatch command to resume any unfinished formal training.'
fi
