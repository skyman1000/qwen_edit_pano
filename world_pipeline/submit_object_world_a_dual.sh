#!/usr/bin/env bash
#SBATCH --job-name=world-A-dual
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
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-A-dual_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
dual_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
dual_steps="${1:-10000}"
dual_tag="${2:-full_v1}"
[[ "$dual_steps" =~ ^[1-9][0-9]{0,5}$ && "$dual_tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Usage: sbatch [options] submit_object_world_a_dual.sh [TOTAL_STEPS] [TAG]' >&2; exit 2; }
dual_root="qwen_edit_pano/outputs/object_world_id_A_dual_${dual_tag}"
mkdir -p "$dual_root"
exec 9>"$dual_root/.launcher.lock"
flock -n 9 || { echo 'Another launcher is using this run root.' >&2; exit 2; }
dual_plan=$("$dual_python" -m qwen_edit_pano.world_pipeline.id_binding_world.restart --root "$dual_root" --steps "$dual_steps")
mapfile -t dual_fields <<< "$dual_plan"
[[ "${dual_fields[0]}" != DONE ]] || { echo "Already reached target: ${dual_fields[2]} steps"; exit 0; }
dual_output="${dual_fields[0]}"
dual_checkpoint="${dual_fields[1]}"
dual_stop="${dual_output}.stop"
dual_timed_out=0
dual_terminated=0
dual_pid=''
trap 'dual_timed_out=1; : > "$dual_stop"' USR1
trap 'dual_terminated=1; : > "$dual_stop"' TERM INT
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "A only; two replicas, effective batch=4; resume=$dual_checkpoint; target=$dual_steps; output=$dual_output"
"$dual_python" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m qwen_edit_pano.world_pipeline.id_binding_world.train \
  --group A --resume "$dual_checkpoint" --output "$dual_output" --steps "$dual_steps" \
  --warmup-steps 100 --binding-ramp-steps 200 --accumulation 4 --lr 1e-4 \
  --pano-lora-scale .5 --region-weight .25 --blocks 20 40 59 --seed 0 \
  --save-every 500 --activation-storage cpu --stop-file "$dual_stop" &
dual_pid=$!
# A signal interrupts bash wait; wait again while the launcher is still alive.
while true; do
  if wait "$dual_pid"; then dual_status=0; else dual_status=$?; fi
  if kill -0 "$dual_pid" 2>/dev/null; then continue; fi
  break
done
[[ "$dual_status" == 0 ]] || exit "$dual_status"
if [[ -f "$dual_output/COMPLETE.json" ]]; then
  echo "A TRAINING COMPLETE: $dual_output; no evaluation performed."
elif [[ -f "$dual_output/STOPPED.json" ]]; then
  echo "Safe checkpoint saved: $dual_output/STOPPED.json"
  if [[ "$dual_timed_out" == 1 && "$dual_terminated" == 0 && -n "${SLURM_JOB_ID:-}" ]]; then
    echo 'Time limit approaching; requeueing this job to resume the latest complete checkpoint.'
    scontrol requeue "$SLURM_JOB_ID"
  else
    echo 'Resubmit the same sbatch command to resume.'
  fi
else
  echo 'Trainer exited without COMPLETE or STOPPED; inspect the log.' >&2
  exit 1
fi
