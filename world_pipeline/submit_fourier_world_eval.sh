#!/usr/bin/env bash
#SBATCH --job-name=caupano-fourier-eval
#SBATCH --partition=debug
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/caupano-fourier-eval_%A_%a.log

set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
steps=(000250 000500 000750 001000)
task_index=${SLURM_ARRAY_TASK_ID:?Submit with sbatch; array index required}
case "$task_index" in 0|1|2|3) ;; *) echo 'Invalid array index' >&2; exit 2;; esac
echo "Checkpoint step: ${steps[$task_index]}"
echo 'Fixed indoor6, per-ID seed0, 28 inference steps, CFG4, Panorama scale from checkpoint.'
exec bash qwen_edit_pano/world_pipeline/run_fourier_world.sh eval "${steps[$task_index]}"
