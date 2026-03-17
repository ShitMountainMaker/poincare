#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

BASE_SEMANTIC_ID_PATH="${BASE_SEMANTIC_ID_PATH:?Set BASE_SEMANTIC_ID_PATH to the base semantic id inference output.}"
EUC_PREFIX_SEMANTIC_ID_PATH="${EUC_PREFIX_SEMANTIC_ID_PATH:?Set EUC_PREFIX_SEMANTIC_ID_PATH to the Euclidean-prefix semantic id inference output.}"
HYP_PREFIX_SEMANTIC_ID_PATH="${HYP_PREFIX_SEMANTIC_ID_PATH:?Set HYP_PREFIX_SEMANTIC_ID_PATH to the Hyperbolic-prefix semantic id inference output.}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/semantic_id_proxy_metrics}"
EMBEDDING_PATH="${EMBEDDING_PATH:-}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-}"
NUM_HIERARCHIES="${NUM_HIERARCHIES:-}"
TOP_K="${TOP_K:-5}"
METADATA_CSV="${METADATA_CSV:-}"
METADATA_JSON="${METADATA_JSON:-}"
CATEGORY_FIELD="${CATEGORY_FIELD:-}"

cd "${REPO_DIR}" || exit 1
if [[ -e "${OUTPUT_DIR}" ]]; then
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

CMD=(
  python scripts/analyze_semantic_ids.py
  "--run" "base=${BASE_SEMANTIC_ID_PATH}"
  "--run" "euc_prefix=${EUC_PREFIX_SEMANTIC_ID_PATH}"
  "--run" "hyp_prefix=${HYP_PREFIX_SEMANTIC_ID_PATH}"
  "--output-dir" "${OUTPUT_DIR}"
  "--top-k" "${TOP_K}"
)

if [[ -n "${EMBEDDING_PATH}" ]]; then
  CMD+=("--embedding-path" "${EMBEDDING_PATH}")
fi

if [[ -n "${CODEBOOK_SIZE}" ]]; then
  CMD+=("--codebook-size" "${CODEBOOK_SIZE}")
fi

if [[ -n "${NUM_HIERARCHIES}" ]]; then
  CMD+=("--num-hierarchies" "${NUM_HIERARCHIES}")
fi

if [[ -n "${METADATA_CSV}" ]]; then
  CMD+=("--metadata-csv" "${METADATA_CSV}")
fi

if [[ -n "${METADATA_JSON}" ]]; then
  CMD+=("--metadata-json" "${METADATA_JSON}")
fi

if [[ -n "${CATEGORY_FIELD}" ]]; then
  CMD+=("--category-field" "${CATEGORY_FIELD}")
fi

printf 'command: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
