#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix}"
PREFIX_WEIGHT_SWEEP="${PREFIX_WEIGHT_SWEEP:-0.05,0.1,0.2,0.3,0.4}"
MATRIX_RUNS="${MATRIX_RUNS:-beauty:42 beauty:43 toys:42}"
SUMMARY_MD="${SUMMARY_MD:-${OUTPUT_ROOT}/SUMMARY.md}"

cd "${REPO_DIR}" || exit 1
mkdir -p "${OUTPUT_ROOT}"

{
  echo "# Base + Hyp Matrix Semantic Stage"
  echo
  echo "- output_root: \`${OUTPUT_ROOT}\`"
  echo "- prefix_weight_sweep: \`${PREFIX_WEIGHT_SWEEP}\`"
  echo "- matrix_runs: \`${MATRIX_RUNS}\`"
  echo
} > "${OUTPUT_ROOT}/SEMANTIC_JOB_SUMMARY.md"

for run_spec in ${MATRIX_RUNS}; do
  dataset_name="${run_spec%%:*}"
  seed="${run_spec##*:}"
  run_name="${dataset_name}_seed${seed}"
  run_root="${OUTPUT_ROOT}/${run_name}"

  {
    echo "## ${run_name}"
    echo
    echo "- dataset: \`${dataset_name}\`"
    echo "- seed: \`${seed}\`"
    echo "- run_root: \`${run_root}\`"
    echo
  } >> "${OUTPUT_ROOT}/SEMANTIC_JOB_SUMMARY.md"

  DATASET_NAME="${dataset_name}" \
  SEED="${seed}" \
  RUN_NAME="${run_name}" \
  RUN_ROOT="${run_root}" \
  PREFIX_WEIGHT_SWEEP="${PREFIX_WEIGHT_SWEEP}" \
  RUN_SEMANTIC_STAGE=1 \
  RUN_DOWNSTREAM_STAGE=0 \
  bash "${REPO_DIR}/scripts/run_base_hyp_sweep_then_tiger.sh"

  echo "- status: completed" >> "${OUTPUT_ROOT}/SEMANTIC_JOB_SUMMARY.md"
  echo >> "${OUTPUT_ROOT}/SEMANTIC_JOB_SUMMARY.md"
done

bash "${REPO_DIR}/scripts/update_base_hyp_matrix_summary.sh"

echo "semantic matrix completed"
echo "summary: ${SUMMARY_MD}"
