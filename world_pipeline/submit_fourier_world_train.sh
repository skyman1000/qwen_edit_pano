#!/usr/bin/env bash
#SBATCH --job-name=caupano-fourier-world
#SBATCH --partition=debug
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --qos=normal
#SBATCH --time=72:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/caupano-fourier-world_%j.log
#SBATCH --open-mode=append

# Submit using sbatch. No interactive shell, conda activation or new env vars.
# Fresh: sbatch qwen_edit_pano/world_pipeline/submit_fourier_world_train.sh
# Resume (same 1000-step budget): sbatch ...sh CHECKPOINT NEW_OUTPUT
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360

if [[ $# -ne 0 && $# -ne 2 ]]; then
    echo 'Usage: sbatch submit_fourier_world_train.sh [FOURIER_CHECKPOINT NEW_OUTPUT]' >&2
    exit 2
fi
train_output=qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1
if [[ $# -eq 2 ]]; then
    [[ -f "$1/COMPLETE.json" && -f "$1/optimizer.pt" ]] || {
        echo 'Resume requires a complete Fourier training checkpoint.' >&2; exit 2;
    }
    train_output="$2"
fi
if [[ -e "$train_output" ]]; then
    echo "Output already exists: $train_output. Use a complete checkpoint and NEW_OUTPUT to resume." >&2
    exit 2
fi
echo "Job ID: ${SLURM_JOB_ID:-not-submitted-with-sbatch}"
echo "Node: $(hostname)"
echo "Started: $(date -Is)"
echo "Output: $train_output"
echo 'Frozen epoch015 / scale0.5; Fourier World only; 1000 optimizer steps; warmup100; accumulation4.'

# Same audited runner; Python uses -u and the absolute qwen360 interpreter.
# No srun or background shell required: this process is managed by Slurm.
exec bash qwen_edit_pano/world_pipeline/run_fourier_world.sh train "$@"
