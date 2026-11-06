#!/usr/bin/env bash
#SBATCH --job-name=world-full-C
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/world-full-C_%j.log

# Submitted by the user only. Two GPUs train ONE model, global batch four.
# Default: the reviewed first 2000 NEW full-data steps; no automatic evaluation.
# Re-submitting the same command resumes the latest complete Full-C checkpoint.
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360

full_c_steps="${1:-2000}"
if [[ $# -gt 1 || ! "$full_c_steps" =~ ^[1-9][0-9]{0,4}$ ]]; then
    echo 'Usage: sbatch [Slurm options] submit_object_world_full_c.sh [NEW_FULL_DATA_TARGET_STEPS]' >&2
    exit 2
fi
if (( full_c_steps > 10000 )); then
    echo 'Full-C supports at most 10000 new full-data steps; the reviewed first target is 2000.' >&2
    exit 2
fi

echo "Slurm job: ${SLURM_JOB_ID:-manual}; two GPUs; global batch=4; new full-data target=$full_c_steps"
echo 'Frozen Panorama epoch015 scale0.5; initialize from C600 + Adam, or resume latest Full-C checkpoint.'
echo 'Save every100 steps. No automatic evaluation, budget extension or job resubmission.'

exec bash qwen_edit_pano/world_pipeline/run_object_world_full.sh train --gpus 2 --steps "$full_c_steps"
