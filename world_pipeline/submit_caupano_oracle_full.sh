#!/usr/bin/env bash
#SBATCH --job-name=caupano-oracle-full
#SBATCH --partition=debug
#SBATCH --nodelist=GPU2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=caupano-oracle-full_%j.log

set -eo pipefail

# 使用提交任务时所在的项目目录
cd "${SLURM_SUBMIT_DIR:?Please submit from DIT360}"

if [[ ! -f qwen_edit_pano/world_pipeline/run_gt_supplement.sh ]]; then
    echo "Please submit from the DIT360 project root." >&2
    exit 2
fi

# 激活 qwen360 环境
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate qwen360
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Started: $(date)"

nvidia-smi

# 调用现有训练脚本，不改变 Full 实验配置
bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh train-full

echo "Training finished: $(date)"
