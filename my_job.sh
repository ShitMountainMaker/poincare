#!/bin/bash
#SBATCH -p acd_u
#SBATCH -o /data/user/cwu319/RC/poincare/history/output.txt
#SBATCH -e /data/user/cwu319/RC/poincare/history/error.txt
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=hyper_sid_prefix
set -euo pipefail

REPO_DIR="/data/user/cwu319/RC/poincare"
CONDA_ENV_DIR="/data/user/cwu319/conda_envs/rec"

cd "${REPO_DIR}" || exit 1
mkdir -p history

echo "===== job start ====="
echo "time: $(date)"
echo "submit_dir: ${SLURM_SUBMIT_DIR:-N/A}"
echo "pwd: $(pwd)"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "====================="

source "${CONDA_ENV_DIR}/bin/activate"

# Step 1: dataset paths
export REPO_DIR
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/amazon_data}"
export DATASET_NAME="${DATASET_NAME:-beauty}"
export DATA_DIR="${DATA_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
export DATA_ITEMS_DIR="${DATA_ITEMS_DIR:-${DATA_DIR}/items}"
export DATA_SEQUENCE_TRAIN_DIR="${DATA_SEQUENCE_TRAIN_DIR:-${DATA_DIR}/training}"
export DATA_SEQUENCE_EVAL_DIR="${DATA_SEQUENCE_EVAL_DIR:-${DATA_DIR}/evaluation}"
export DATA_SEQUENCE_TEST_DIR="${DATA_SEQUENCE_TEST_DIR:-${DATA_DIR}/testing}"

# Step 2: semantic embedding generation
export SEMANTIC_EMBEDDING_OUTPUT_DIR="${SEMANTIC_EMBEDDING_OUTPUT_DIR:-${REPO_DIR}/outputs/semantic_embeddings}"
export SEMANTIC_EMBEDDING_PICKLE_DIR="${SEMANTIC_EMBEDDING_PICKLE_DIR:-${SEMANTIC_EMBEDDING_OUTPUT_DIR}/pickle}"
export EMBEDDING_PATH="${EMBEDDING_PATH:-${SEMANTIC_EMBEDDING_PICKLE_DIR}/merged_predictions_tensor.pt}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"

# Step 3: semantic ID learning and inference
export NUM_HIERARCHIES="${NUM_HIERARCHIES:-3}"
export CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"
export SEMANTIC_ID_STAGE_ROOT="${SEMANTIC_ID_STAGE_ROOT:-${REPO_DIR}/outputs/semantic_id_stage}"
export RUN_ROOT="${RUN_ROOT:-${SEMANTIC_ID_STAGE_ROOT}}"

export BASE_SEMANTIC_ID_TRAIN_DIR="${BASE_SEMANTIC_ID_TRAIN_DIR:-${SEMANTIC_ID_STAGE_ROOT}/base}"
export BASE_SEMANTIC_ID_CHECKPOINT_DIR="${BASE_SEMANTIC_ID_CHECKPOINT_DIR:-${BASE_SEMANTIC_ID_TRAIN_DIR}/checkpoints}"
export BASE_SEMANTIC_ID_CHECKPOINT_PATH="${BASE_SEMANTIC_ID_CHECKPOINT_PATH:-${BASE_SEMANTIC_ID_CHECKPOINT_DIR}/last.ckpt}"
export BASE_SEMANTIC_ID_INFERENCE_DIR="${BASE_SEMANTIC_ID_INFERENCE_DIR:-${SEMANTIC_ID_STAGE_ROOT}/inference/base}"
export BASE_SEMANTIC_ID_PICKLE_DIR="${BASE_SEMANTIC_ID_PICKLE_DIR:-${BASE_SEMANTIC_ID_INFERENCE_DIR}/pickle}"
export BASE_SEMANTIC_ID_PATH="${BASE_SEMANTIC_ID_PATH:-${BASE_SEMANTIC_ID_PICKLE_DIR}/merged_predictions_tensor.pt}"

export EUC_PREFIX_SEMANTIC_ID_TRAIN_DIR="${EUC_PREFIX_SEMANTIC_ID_TRAIN_DIR:-${SEMANTIC_ID_STAGE_ROOT}/euc_prefix}"
export EUC_PREFIX_SEMANTIC_ID_CHECKPOINT_DIR="${EUC_PREFIX_SEMANTIC_ID_CHECKPOINT_DIR:-${EUC_PREFIX_SEMANTIC_ID_TRAIN_DIR}/checkpoints}"
export EUC_PREFIX_SEMANTIC_ID_CHECKPOINT_PATH="${EUC_PREFIX_SEMANTIC_ID_CHECKPOINT_PATH:-${EUC_PREFIX_SEMANTIC_ID_CHECKPOINT_DIR}/last.ckpt}"
export EUC_PREFIX_SEMANTIC_ID_INFERENCE_DIR="${EUC_PREFIX_SEMANTIC_ID_INFERENCE_DIR:-${SEMANTIC_ID_STAGE_ROOT}/inference/euc_prefix}"
export EUC_PREFIX_SEMANTIC_ID_PICKLE_DIR="${EUC_PREFIX_SEMANTIC_ID_PICKLE_DIR:-${EUC_PREFIX_SEMANTIC_ID_INFERENCE_DIR}/pickle}"
export EUC_PREFIX_SEMANTIC_ID_PATH="${EUC_PREFIX_SEMANTIC_ID_PATH:-${EUC_PREFIX_SEMANTIC_ID_PICKLE_DIR}/merged_predictions_tensor.pt}"

export HYP_PREFIX_SEMANTIC_ID_TRAIN_DIR="${HYP_PREFIX_SEMANTIC_ID_TRAIN_DIR:-${SEMANTIC_ID_STAGE_ROOT}/hyp_prefix}"
export HYP_PREFIX_SEMANTIC_ID_CHECKPOINT_DIR="${HYP_PREFIX_SEMANTIC_ID_CHECKPOINT_DIR:-${HYP_PREFIX_SEMANTIC_ID_TRAIN_DIR}/checkpoints}"
export HYP_PREFIX_SEMANTIC_ID_CHECKPOINT_PATH="${HYP_PREFIX_SEMANTIC_ID_CHECKPOINT_PATH:-${HYP_PREFIX_SEMANTIC_ID_CHECKPOINT_DIR}/last.ckpt}"
export HYP_PREFIX_SEMANTIC_ID_INFERENCE_DIR="${HYP_PREFIX_SEMANTIC_ID_INFERENCE_DIR:-${SEMANTIC_ID_STAGE_ROOT}/inference/hyp_prefix}"
export HYP_PREFIX_SEMANTIC_ID_PICKLE_DIR="${HYP_PREFIX_SEMANTIC_ID_PICKLE_DIR:-${HYP_PREFIX_SEMANTIC_ID_INFERENCE_DIR}/pickle}"
export HYP_PREFIX_SEMANTIC_ID_PATH="${HYP_PREFIX_SEMANTIC_ID_PATH:-${HYP_PREFIX_SEMANTIC_ID_PICKLE_DIR}/merged_predictions_tensor.pt}"

# Step 4: TIGER training
export RECOMMENDATION_STAGE_ROOT="${RECOMMENDATION_STAGE_ROOT:-${REPO_DIR}/outputs/recommendation_stage}"
export TIGER_NUM_HIERARCHIES="${TIGER_NUM_HIERARCHIES:-4}"
export DOWNSTREAM_MODE="${DOWNSTREAM_MODE:-base}"

case "${DOWNSTREAM_MODE}" in
  base)
    DEFAULT_TIGER_TRAIN_SEMANTIC_ID_PATH="${BASE_SEMANTIC_ID_PATH}"
    ;;
  euc_prefix)
    DEFAULT_TIGER_TRAIN_SEMANTIC_ID_PATH="${EUC_PREFIX_SEMANTIC_ID_PATH}"
    ;;
  hyp_prefix)
    DEFAULT_TIGER_TRAIN_SEMANTIC_ID_PATH="${HYP_PREFIX_SEMANTIC_ID_PATH}"
    ;;
  *)
    echo "Unsupported DOWNSTREAM_MODE: ${DOWNSTREAM_MODE}. Expected one of: base, euc_prefix, hyp_prefix." >&2
    exit 1
    ;;
esac

export TIGER_TRAIN_DIR="${TIGER_TRAIN_DIR:-${RECOMMENDATION_STAGE_ROOT}/tiger_train_${DOWNSTREAM_MODE}}"
export TIGER_TRAIN_CHECKPOINT_DIR="${TIGER_TRAIN_CHECKPOINT_DIR:-${TIGER_TRAIN_DIR}/checkpoints}"
export TIGER_TRAIN_CHECKPOINT_PATH="${TIGER_TRAIN_CHECKPOINT_PATH:-${TIGER_TRAIN_CHECKPOINT_DIR}/best.ckpt}"
export TIGER_TRAIN_SEMANTIC_ID_PATH="${TIGER_TRAIN_SEMANTIC_ID_PATH:-${DEFAULT_TIGER_TRAIN_SEMANTIC_ID_PATH}}"

# Step 5: TIGER inference
export TIGER_INFERENCE_DIR="${TIGER_INFERENCE_DIR:-${RECOMMENDATION_STAGE_ROOT}/tiger_inference}"
export TIGER_INFERENCE_PICKLE_DIR="${TIGER_INFERENCE_PICKLE_DIR:-${TIGER_INFERENCE_DIR}/pickle}"
export TIGER_INFERENCE_SEMANTIC_ID_PATH="${TIGER_INFERENCE_SEMANTIC_ID_PATH:-${BASE_SEMANTIC_ID_PATH}}"
export TIGER_INFERENCE_CHECKPOINT_PATH="${TIGER_INFERENCE_CHECKPOINT_PATH:-${TIGER_TRAIN_CHECKPOINT_PATH}}"

# Stage controls
export RUN_MODE="${RUN_MODE:-base_only}"
export RUN_PROXY_METRICS="${RUN_PROXY_METRICS:-0}"
export TOP_K="${TOP_K:-5}"
export METADATA_CSV="${METADATA_CSV:-}"
export METADATA_JSON="${METADATA_JSON:-}"
export CATEGORY_FIELD="${CATEGORY_FIELD:-}"
export PREFIX_WEIGHT_SWEEP="${PREFIX_WEIGHT_SWEEP:-0.05,0.1,0.2}"
export COLLISION_WORSE_THRESHOLD="${COLLISION_WORSE_THRESHOLD:-0.01}"
export FRAC_UNIQUE_MIN_IMPROVEMENT="${FRAC_UNIQUE_MIN_IMPROVEMENT:-0.0}"
export ENTROPY_AVG_DROP_THRESHOLD="${ENTROPY_AVG_DROP_THRESHOLD:-0.15}"
export ENTROPY_LAYER_DROP_THRESHOLD="${ENTROPY_LAYER_DROP_THRESHOLD:-0.2}"
export QUANTIZATION_LOSS_RELATIVE_THRESHOLD="${QUANTIZATION_LOSS_RELATIVE_THRESHOLD:-0.05}"
export QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD="${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD:-1.0}"
export LOSS_EXPLOSION_THRESHOLD="${LOSS_EXPLOSION_THRESHOLD:-100000}"
export PREFIX_WORSE_THRESHOLD="${PREFIX_WORSE_THRESHOLD:-0.0}"

echo "===== fixed paths ====="
echo "[step1] data_dir: ${DATA_DIR}"
echo "[step1] items_dir: ${DATA_ITEMS_DIR}"
echo "[step1] training_dir: ${DATA_SEQUENCE_TRAIN_DIR}"
echo "[step1] evaluation_dir: ${DATA_SEQUENCE_EVAL_DIR}"
echo "[step1] testing_dir: ${DATA_SEQUENCE_TEST_DIR}"
echo "[step2] embedding_output_dir: ${SEMANTIC_EMBEDDING_OUTPUT_DIR}"
echo "[step2] embedding_path: ${EMBEDDING_PATH}"
echo "[step3] semantic_id_stage_root: ${SEMANTIC_ID_STAGE_ROOT}"
echo "[step3-base] train_dir: ${BASE_SEMANTIC_ID_TRAIN_DIR}"
echo "[step3-base] checkpoint_path: ${BASE_SEMANTIC_ID_CHECKPOINT_PATH}"
echo "[step3-base] inference_dir: ${BASE_SEMANTIC_ID_INFERENCE_DIR}"
echo "[step3-base] semantic_id_path: ${BASE_SEMANTIC_ID_PATH}"
echo "[step3-euc] train_dir: ${EUC_PREFIX_SEMANTIC_ID_TRAIN_DIR}"
echo "[step3-euc] checkpoint_path: ${EUC_PREFIX_SEMANTIC_ID_CHECKPOINT_PATH}"
echo "[step3-euc] inference_dir: ${EUC_PREFIX_SEMANTIC_ID_INFERENCE_DIR}"
echo "[step3-euc] semantic_id_path: ${EUC_PREFIX_SEMANTIC_ID_PATH}"
echo "[step3-hyp] train_dir: ${HYP_PREFIX_SEMANTIC_ID_TRAIN_DIR}"
echo "[step3-hyp] checkpoint_path: ${HYP_PREFIX_SEMANTIC_ID_CHECKPOINT_PATH}"
echo "[step3-hyp] inference_dir: ${HYP_PREFIX_SEMANTIC_ID_INFERENCE_DIR}"
echo "[step3-hyp] semantic_id_path: ${HYP_PREFIX_SEMANTIC_ID_PATH}"
echo "[step4] tiger_train_dir: ${TIGER_TRAIN_DIR}"
echo "[step4] tiger_train_checkpoint_path: ${TIGER_TRAIN_CHECKPOINT_PATH}"
echo "[step4] tiger_train_semantic_id_path: ${TIGER_TRAIN_SEMANTIC_ID_PATH}"
echo "[mode] downstream_mode: ${DOWNSTREAM_MODE}"
echo "[step5] tiger_inference_dir: ${TIGER_INFERENCE_DIR}"
echo "[step5] tiger_inference_checkpoint_path: ${TIGER_INFERENCE_CHECKPOINT_PATH}"
echo "[step5] tiger_inference_semantic_id_path: ${TIGER_INFERENCE_SEMANTIC_ID_PATH}"
echo "[sid-sweep] prefix_weight_sweep: ${PREFIX_WEIGHT_SWEEP}"
echo "[sid-sweep] collision_worse_threshold: ${COLLISION_WORSE_THRESHOLD}"
echo "[sid-sweep] frac_unique_min_improvement: ${FRAC_UNIQUE_MIN_IMPROVEMENT}"
echo "[sid-sweep] entropy_avg_drop_threshold: ${ENTROPY_AVG_DROP_THRESHOLD}"
echo "[sid-sweep] entropy_layer_drop_threshold: ${ENTROPY_LAYER_DROP_THRESHOLD}"
echo "[sid-sweep] quantization_loss_relative_threshold: ${QUANTIZATION_LOSS_RELATIVE_THRESHOLD}"
echo "[sid-sweep] quantization_loss_absolute_threshold: ${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD}"
echo "[sid-sweep] loss_explosion_threshold: ${LOSS_EXPLOSION_THRESHOLD}"
echo "[sid-sweep] prefix_worse_threshold: ${PREFIX_WORSE_THRESHOLD}"
echo "======================="

# RUN_MODE=all_three RUN_PROXY_METRICS=0 bash "${REPO_DIR}/scripts/run_semantic_id_prefix_experiments.sh"

python -m src.train \
  experiment=tiger_train_flat \
  data_dir="${DATA_DIR}" \
  semantic_id_path="${TIGER_TRAIN_SEMANTIC_ID_PATH}" \
  num_hierarchies="${TIGER_NUM_HIERARCHIES}" \
  paths.output_dir="${TIGER_TRAIN_DIR}" \
  hydra.run.dir="${TIGER_TRAIN_DIR}"
