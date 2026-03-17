#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PARTITION="${PARTITION:-acd_u}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix}"
HISTORY_DIR="${HISTORY_DIR:-${REPO_DIR}/history}"
MATRIX_RUNS="${MATRIX_RUNS:-beauty:42 beauty:43 toys:42}"

cd "${REPO_DIR}" || exit 1
mkdir -p "${HISTORY_DIR}"

for run_spec in ${MATRIX_RUNS}; do
  dataset_name="${run_spec%%:*}"
  seed="${run_spec##*:}"
  run_name="${dataset_name}_seed${seed}"
  stdout_log="${HISTORY_DIR}/output_downstream_${run_name}.txt"
  stderr_log="${HISTORY_DIR}/error_downstream_${run_name}.txt"

  sbatch \
    -p "${PARTITION}" \
    -J "bh_${run_name}" \
    -o "${stdout_log}" \
    -e "${stderr_log}" \
    --export=ALL,DATASET_NAME="${dataset_name}",SEED="${seed}",RUN_NAME="${run_name}",RUN_ROOT="${OUTPUT_ROOT}/${run_name}" \
    my_job_base_hyp_downstream.sh
done
