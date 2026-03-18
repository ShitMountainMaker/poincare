#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/poincare_band_v35/beauty_seed42}"
SEMANTIC_ROOT="${SEMANTIC_ROOT:-${RUN_ROOT}/semantic_id_stage}"
RECOMMENDATION_ROOT="${RECOMMENDATION_ROOT:-${RUN_ROOT}/recommendation_stage}"
PROXY_OUTPUT_DIR="${PROXY_OUTPUT_DIR:-${SEMANTIC_ROOT}/proxy_teacher_forced_v35}"
SELECTION_JSON="${SELECTION_JSON:-${PROXY_OUTPUT_DIR}/selection_teacher_forced.json}"
SELECTION_MD="${SELECTION_MD:-${PROXY_OUTPUT_DIR}/selection_teacher_forced.md}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/teacher_forced_proxy_logs}"

BASE_SEMANTIC_PICKLE="${BASE_SEMANTIC_PICKLE:-${SEMANTIC_ROOT}/inference/base/pickle}"
BASE_TRAIN_DIR="${BASE_TRAIN_DIR:-${SEMANTIC_ROOT}/base}"
HYP_SWEEP_ROOT="${HYP_SWEEP_ROOT:-${SEMANTIC_ROOT}/sweeps/hyp_prefix}"

BASE_TIGER_DIR="${BASE_TIGER_DIR:-${RECOMMENDATION_ROOT}/tiger_train_base}"
HYP_0_1_TIGER_DIR="${HYP_0_1_TIGER_DIR:-${RECOMMENDATION_ROOT}/tiger_train_hyp_0_1}"
HYP_0_2_TIGER_DIR="${HYP_0_2_TIGER_DIR:-${RECOMMENDATION_ROOT}/tiger_train_hyp_0_2}"
HYP_0_3_TIGER_DIR="${HYP_0_3_TIGER_DIR:-${RECOMMENDATION_ROOT}/tiger_train_hyp_0_3}"

EMBEDDING_PATH="${EMBEDDING_PATH:-${RUN_ROOT}/semantic_embeddings/pickle/merged_predictions_tensor.pt}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"
NUM_HIERARCHIES="${NUM_HIERARCHIES:-3}"
TOP_K="${TOP_K:-5}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}" "${PROXY_OUTPUT_DIR}"

run_tf() {
  local label="$1"
  local train_dir="$2"
  local output_json="${train_dir}/teacher_forced_val.json"
  echo "[teacher-forced] ${label}" | tee -a "${LOG_DIR}/summary.log"
  python scripts/analyze_teacher_forced_tiger.py \
    --train-dir "${train_dir}" \
    --output-json "${output_json}" \
    --split val \
    --device cuda \
    > "${LOG_DIR}/${label}_stdout.log" \
    2> "${LOG_DIR}/${label}_stderr.log"
}

run_tf base "${BASE_TIGER_DIR}"
run_tf hyp_0_1 "${HYP_0_1_TIGER_DIR}"
run_tf hyp_0_2 "${HYP_0_2_TIGER_DIR}"
run_tf hyp_0_3 "${HYP_0_3_TIGER_DIR}"

python scripts/analyze_poincare_proxy.py \
  --output-dir "${PROXY_OUTPUT_DIR}" \
  --embedding-path "${EMBEDDING_PATH}" \
  --codebook-size "${CODEBOOK_WIDTH}" \
  --num-hierarchies "${NUM_HIERARCHIES}" \
  --top-k "${TOP_K}" \
  --run "base=${BASE_SEMANTIC_PICKLE}" \
  --run "hyp_prefix_w_0_1=${HYP_SWEEP_ROOT}/hyp_prefix_w_0_1/inference/pickle" \
  --run "hyp_prefix_w_0_2=${HYP_SWEEP_ROOT}/hyp_prefix_w_0_2/inference/pickle" \
  --run "hyp_prefix_w_0_3=${HYP_SWEEP_ROOT}/hyp_prefix_w_0_3/inference/pickle" \
  --teacher-forced-run "base=${BASE_TIGER_DIR}" \
  --teacher-forced-run "hyp_prefix_w_0_1=${HYP_0_1_TIGER_DIR}" \
  --teacher-forced-run "hyp_prefix_w_0_2=${HYP_0_2_TIGER_DIR}" \
  --teacher-forced-run "hyp_prefix_w_0_3=${HYP_0_3_TIGER_DIR}" \
  > "${LOG_DIR}/proxy_stdout.log" \
  2> "${LOG_DIR}/proxy_stderr.log"

python scripts/select_poincare_weight_v35.py \
  --method-family hyp_prefix \
  --baseline-train-dir "${BASE_TRAIN_DIR}" \
  --baseline-proxy-json "${PROXY_OUTPUT_DIR}/base.json" \
  --output-json "${SELECTION_JSON}" \
  --output-md "${SELECTION_MD}" \
  --candidate "0.1|hyp_prefix_w_0_1|${HYP_SWEEP_ROOT}/hyp_prefix_w_0_1/train|${HYP_SWEEP_ROOT}/hyp_prefix_w_0_1/inference|${PROXY_OUTPUT_DIR}/hyp_prefix_w_0_1.json" \
  --candidate "0.2|hyp_prefix_w_0_2|${HYP_SWEEP_ROOT}/hyp_prefix_w_0_2/train|${HYP_SWEEP_ROOT}/hyp_prefix_w_0_2/inference|${PROXY_OUTPUT_DIR}/hyp_prefix_w_0_2.json" \
  --candidate "0.3|hyp_prefix_w_0_3|${HYP_SWEEP_ROOT}/hyp_prefix_w_0_3/train|${HYP_SWEEP_ROOT}/hyp_prefix_w_0_3/inference|${PROXY_OUTPUT_DIR}/hyp_prefix_w_0_3.json" \
  > "${LOG_DIR}/selector_stdout.log" \
  2> "${LOG_DIR}/selector_stderr.log"

echo "done" | tee -a "${LOG_DIR}/summary.log"
