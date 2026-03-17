#!/bin/bash
#SBATCH -p acd_u
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=bh_downstream
set -euo pipefail

REPO_DIR="/data/user/cwu319/RC/poincare"
CONDA_ENV_DIR="/data/user/cwu319/conda_envs/rec"

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME.}"
SEED="${SEED:?Set SEED.}"
RUN_NAME="${RUN_NAME:-${DATASET_NAME}_seed${SEED}}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix/${RUN_NAME}}"

cd "${REPO_DIR}" || exit 1
mkdir -p history

echo "===== downstream job start ====="
echo "time: $(date)"
echo "pwd: $(pwd)"
echo "job_name: ${SLURM_JOB_NAME:-N/A}"
echo "partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "dataset_name: ${DATASET_NAME}"
echo "seed: ${SEED}"
echo "run_name: ${RUN_NAME}"
echo "run_root: ${RUN_ROOT}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "==============================="

source "${CONDA_ENV_DIR}/bin/activate"

export REPO_DIR
export DATASET_NAME
export SEED
export RUN_NAME
export RUN_ROOT
export RUN_SEMANTIC_STAGE=0
export RUN_DOWNSTREAM_STAGE=1
export USE_SLURM_SRUN=1

bash "${REPO_DIR}/scripts/run_base_hyp_sweep_then_tiger.sh"
