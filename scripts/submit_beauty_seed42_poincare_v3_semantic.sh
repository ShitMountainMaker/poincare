#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PARTITION="${PARTITION:-emergency_acd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/poincare_band_v3}"
HISTORY_DIR="${HISTORY_DIR:-${REPO_DIR}/history}"
SEMANTIC_EMBEDDING_MODEL="${SEMANTIC_EMBEDDING_MODEL:-${REPO_DIR}/models/google/flan-t5-xl}"

cd "${REPO_DIR}" || exit 1
mkdir -p "${HISTORY_DIR}"

sbatch \
  -p "${PARTITION}" \
  -J "b42_poincare_v3_sem" \
  -o "${HISTORY_DIR}/output_b42_poincare_v3_semantic.txt" \
  -e "${HISTORY_DIR}/error_b42_poincare_v3_semantic.txt" \
  --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",MATRIX_RUNS="beauty:42",PREFIX_WEIGHT_SWEEP="0.1 0.2 0.3",SEMANTIC_EMBEDDING_MODEL="${SEMANTIC_EMBEDDING_MODEL}" \
  my_job_base_hyp_matrix.sh
