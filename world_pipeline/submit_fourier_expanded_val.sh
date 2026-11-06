#!/usr/bin/env bash
#SBATCH --job-name=caupano-expanded-val
#SBATCH --partition=debug
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --array=0-5%1
#SBATCH --chdir=/data-nfs/gpu1-2/u13529658780/DIT360
#SBATCH --output=/data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/oracle_epoch015_scale05_indoor6_step000750_fourier_v1/expanded_val_v1/logs/eval_%A_%a.log

set -euo pipefail
cd /data-nfs/gpu1-2/u13529658780/DIT360
task_index=${SLURM_ARRAY_TASK_ID:?Submit with sbatch; array index required}
case "$task_index" in 0|1|2|3|4|5) ;; *) echo 'Invalid array index' >&2; exit 2;; esac
printf -v group_name 'group%02d' "$((task_index / 2 + 1))"
seed=$((task_index % 2))
eval_root=qwen_edit_pano/outputs/oracle_epoch015_scale05_indoor6_step000750_fourier_v1/expanded_val_v1
echo "Expanded held-out validation: $group_name seed$seed; fixed step750 / scale0.5"
exec /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u \
  -m qwen_edit_pano.world_pipeline.fourier_world.inference \
  --checkpoint qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1/checkpoint-step000750 \
  --selection "$eval_root/selections/$group_name.json" \
  --split val --limit 8 --steps 28 --cfg 4 --seed "$seed" \
  --modes baseline correct shuffled \
  --output "$eval_root/${group_name}_seed${seed}"
