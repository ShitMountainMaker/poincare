#!/bin/bash
#SBATCH -p acd_u
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=bh_tiger_single
set -euo pipefail

REPO_DIR="/data/user/cwu319/RC/hyper"
CONDA_ENV_DIR="/data/user/cwu319/conda_envs/rec"

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME.}"
SEED="${SEED:?Set SEED.}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME.}"
SEMANTIC_ID_PATH="${SEMANTIC_ID_PATH:?Set SEMANTIC_ID_PATH.}"
TIGER_TRAIN_DIR="${TIGER_TRAIN_DIR:?Set TIGER_TRAIN_DIR.}"

cd "${REPO_DIR}" || exit 1
mkdir -p history

echo "===== single candidate downstream job start ====="
echo "time: $(date)"
echo "pwd: $(pwd)"
echo "job_name: ${SLURM_JOB_NAME:-N/A}"
echo "partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "dataset_name: ${DATASET_NAME}"
echo "seed: ${SEED}"
echo "run_name: ${RUN_NAME}"
echo "semantic_id_path: ${SEMANTIC_ID_PATH}"
echo "tiger_train_dir: ${TIGER_TRAIN_DIR}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "==============================================="

source "${CONDA_ENV_DIR}/bin/activate"

TRAIN_DISTRIBUTED_LAUNCHER=()
if [[ "${USE_SLURM_SRUN:-1}" == "1" && -n "${SLURM_NTASKS:-}" ]]; then
  TRAIN_DISTRIBUTED_LAUNCHER=(srun --ntasks="${SLURM_NTASKS}")
  if [[ -n "${SLURM_NTASKS_PER_NODE:-}" ]]; then
    TRAIN_DISTRIBUTED_LAUNCHER+=(--ntasks-per-node="${SLURM_NTASKS_PER_NODE}")
  fi
fi

rm -rf "${TIGER_TRAIN_DIR}"

"${TRAIN_DISTRIBUTED_LAUNCHER[@]}" \
python -m src.train \
  experiment=tiger_train_flat \
  "data_dir=${REPO_DIR}/data/amazon_data/${DATASET_NAME}" \
  "semantic_id_path=${SEMANTIC_ID_PATH}" \
  num_hierarchies=4 \
  "seed=${SEED}" \
  "paths.output_dir=${TIGER_TRAIN_DIR}" \
  "hydra.run.dir=${TIGER_TRAIN_DIR}"

