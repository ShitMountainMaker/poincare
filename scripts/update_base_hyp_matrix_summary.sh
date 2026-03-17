#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix}"
SUMMARY_MD="${SUMMARY_MD:-${OUTPUT_ROOT}/SUMMARY.md}"

python "${REPO_DIR}/scripts/build_base_hyp_matrix_summary.py" \
  --root "${OUTPUT_ROOT}" \
  --output-md "${SUMMARY_MD}" \
  --run beauty_seed42 \
  --run beauty_seed43 \
  --run toys_seed42

echo "summary written to ${SUMMARY_MD}"
