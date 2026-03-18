#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PARTITION="${PARTITION:-emergency_acd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/poincare_band_v35}"
HISTORY_DIR="${HISTORY_DIR:-${REPO_DIR}/history}"
SEMANTIC_EMBEDDING_MODEL="${SEMANTIC_EMBEDDING_MODEL:-${REPO_DIR}/models/google/flan-t5-xl}"

cd "${REPO_DIR}" || exit 1
mkdir -p "${HISTORY_DIR}"

sbatch \
  -p "${PARTITION}" \
  -J "b42_poincare_v35_sem" \
  -o "${HISTORY_DIR}/output_b42_poincare_v35_semantic.txt" \
  -e "${HISTORY_DIR}/error_b42_poincare_v35_semantic.txt" \
  --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",MATRIX_RUNS="beauty:42",PREFIX_WEIGHT_SWEEP="0.1 0.2 0.3",SEMANTIC_EMBEDDING_MODEL="${SEMANTIC_EMBEDDING_MODEL}",PROXY_ANALYSIS_SCRIPT="scripts/analyze_poincare_proxy.py",WEIGHT_SELECTOR_SCRIPT="scripts/select_poincare_weight_v35.py" \
  my_job_base_hyp_matrix.sh
