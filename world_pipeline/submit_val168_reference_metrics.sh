#!/usr/bin/env bash
#SBATCH --job-name=caupano-local-iqa
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/caupano-local-iqa_%j.log

set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
# Existing evaluation runtime; pretrained metric weights may download on first use.
exec DiT360/.venv/bin/python -u -m research.world_evaluation.local_reference_metrics \
  --size 512 --cubemap --device cuda:0 "$@"
