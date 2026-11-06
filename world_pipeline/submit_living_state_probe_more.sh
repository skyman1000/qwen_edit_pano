#!/usr/bin/env bash
#SBATCH --job-name=caupano-state-probe-more
#SBATCH --partition=debug
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/caupano-state-probe-more_%j.log
set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
exec /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u \
  -m research.world_evaluation.state_probe_more run
