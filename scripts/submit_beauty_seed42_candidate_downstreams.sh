#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PARTITION="${PARTITION:-acd_u}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix/beauty_seed42}"
HISTORY_DIR="${HISTORY_DIR:-${REPO_DIR}/history}"

cd "${REPO_DIR}" || exit 1
mkdir -p "${HISTORY_DIR}"

declare -A SEMANTIC_PATHS=(
  [base]="${RUN_ROOT}/semantic_id_stage/inference/base/pickle/merged_predictions_tensor.pt"
  [hyp_0_05]="${RUN_ROOT}/semantic_id_stage/sweeps/hyp_prefix/hyp_prefix_w_0_05/inference/pickle/merged_predictions_tensor.pt"
  [hyp_0_2]="${RUN_ROOT}/semantic_id_stage/sweeps/hyp_prefix/hyp_prefix_w_0_2/inference/pickle/merged_predictions_tensor.pt"
  [hyp_0_3]="${RUN_ROOT}/semantic_id_stage/sweeps/hyp_prefix/hyp_prefix_w_0_3/inference/pickle/merged_predictions_tensor.pt"
)

for candidate in base hyp_0_05 hyp_0_2 hyp_0_3; do
  semantic_id_path="${SEMANTIC_PATHS[$candidate]}"
  if [[ ! -f "${semantic_id_path}" ]]; then
    echo "Missing semantic ids for ${candidate}: ${semantic_id_path}" >&2
    exit 1
  fi

  stdout_log="${HISTORY_DIR}/output_downstream_beauty_seed42_${candidate}.txt"
  stderr_log="${HISTORY_DIR}/error_downstream_beauty_seed42_${candidate}.txt"
  tiger_train_dir="${RUN_ROOT}/recommendation_stage/tiger_train_${candidate}"

  sbatch \
    -p "${PARTITION}" \
    -J "b42_${candidate}" \
    -o "${stdout_log}" \
    -e "${stderr_log}" \
    --export=ALL,DATASET_NAME=beauty,SEED=42,RUN_NAME=beauty_seed42,SEMANTIC_ID_PATH="${semantic_id_path}",TIGER_TRAIN_DIR="${tiger_train_dir}" \
    my_job_tiger_single_candidate.sh
done
