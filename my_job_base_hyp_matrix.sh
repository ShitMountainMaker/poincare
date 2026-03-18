#!/bin/bash
#SBATCH -p emergency_acd
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --job-name=bh_semantic_matrix
set -euo pipefail

REPO_DIR="/data/user/cwu319/RC/poincare"
CONDA_ENV_DIR="/data/user/cwu319/conda_envs/rec"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix}"
PREFIX_WEIGHT_SWEEP="${PREFIX_WEIGHT_SWEEP:-0.05,0.1,0.2,0.3,0.4}"
MATRIX_RUNS="${MATRIX_RUNS:-beauty:42 beauty:43 toys:42}"
SEMANTIC_EMBEDDING_MODEL="${SEMANTIC_EMBEDDING_MODEL:-${REPO_DIR}/models/google/flan-t5-xl}"

cd "${REPO_DIR}" || exit 1
mkdir -p history

echo "===== semantic matrix job start ====="
echo "time: $(date)"
echo "pwd: $(pwd)"
echo "job_name: ${SLURM_JOB_NAME:-N/A}"
echo "partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "output_root: ${OUTPUT_ROOT}"
echo "prefix_weight_sweep: ${PREFIX_WEIGHT_SWEEP}"
echo "matrix_runs: ${MATRIX_RUNS}"
echo "semantic_embedding_model: ${SEMANTIC_EMBEDDING_MODEL}"
echo "==============================="

source "${CONDA_ENV_DIR}/bin/activate"

export REPO_DIR
export OUTPUT_ROOT
export PREFIX_WEIGHT_SWEEP
export MATRIX_RUNS
export SEMANTIC_EMBEDDING_MODEL
export PROXY_ANALYSIS_SCRIPT
export WEIGHT_SELECTOR_SCRIPT
export USE_SLURM_SRUN=1

bash "${REPO_DIR}/scripts/run_base_hyp_semantic_matrix.sh"
