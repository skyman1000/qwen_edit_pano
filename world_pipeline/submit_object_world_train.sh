#!/usr/bin/env bash
#SBATCH --job-name=object-world
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/object-world_%j.log
#SBATCH --open-mode=append

# Run only when the user submits it. Does not download or submit further jobs.
# Args: TOTAL_STEPS NEW_OUTPUT [COMPLETE_RESUME_CHECKPOINT]
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
object_python=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
if [[ $# -ne 2 && $# -ne 3 ]]; then
    echo 'Usage: sbatch submit_object_world_train.sh TOTAL_STEPS NEW_OUTPUT [RESUME_CHECKPOINT]' >&2
    exit 2
fi
[[ "$1" =~ ^[0-9]+$ && "$1" -gt 500 ]] || { echo 'TOTAL_STEPS must exceed warmup500.' >&2; exit 2; }
[[ ! -e "$2" ]] || { echo "Output already exists: $2; resume into a NEW output." >&2; exit 2; }
train_arguments=(--steps "$1" --output "$2" --warmup-steps 500 --save-every 500)
if [[ $# -eq 3 ]]; then
    [[ -f "$3/COMPLETE.json" && -f "$3/world.safetensors" && -f "$3/optimizer.pt" ]] || {
        echo "Resume requires a committed complete checkpoint: $3" >&2; exit 2;
    }
    train_arguments+=(--resume "$3")
fi
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Total target step: $1; output: $2"
echo 'Frozen epoch015 scale0.5; World semantic + Fourier geometry; object region weight0.25; accumulation4.'
exec "$object_python" -u -m qwen_edit_pano.world_pipeline.object_world.train "${train_arguments[@]}"
