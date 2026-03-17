#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/amazon_data/beauty}"
EMBEDDING_PATH="${EMBEDDING_PATH:-${REPO_DIR}/outputs/semantic_embeddings/pickle/merged_predictions_tensor.pt}"
EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"
NUM_HIERARCHIES="${NUM_HIERARCHIES:-3}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"
RUN_MODE="${RUN_MODE:-base_only}"
RUN_PROXY_METRICS="${RUN_PROXY_METRICS:-0}"
TOP_K="${TOP_K:-5}"
METADATA_CSV="${METADATA_CSV:-}"
METADATA_JSON="${METADATA_JSON:-}"
CATEGORY_FIELD="${CATEGORY_FIELD:-}"
PREFIX_WEIGHT_SWEEP="${PREFIX_WEIGHT_SWEEP:-0.05,0.1,0.2}"
COLLISION_WORSE_THRESHOLD="${COLLISION_WORSE_THRESHOLD:-0.03}"
FRAC_UNIQUE_MIN_IMPROVEMENT="${FRAC_UNIQUE_MIN_IMPROVEMENT:--0.01}"
ENTROPY_AVG_DROP_THRESHOLD="${ENTROPY_AVG_DROP_THRESHOLD:-0.15}"
ENTROPY_LAYER_DROP_THRESHOLD="${ENTROPY_LAYER_DROP_THRESHOLD:-0.2}"
QUANTIZATION_LOSS_RELATIVE_THRESHOLD="${QUANTIZATION_LOSS_RELATIVE_THRESHOLD:-0.05}"
QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD="${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD:-1.0}"
LOSS_EXPLOSION_THRESHOLD="${LOSS_EXPLOSION_THRESHOLD:-100000}"
PREFIX_WORSE_THRESHOLD="${PREFIX_WORSE_THRESHOLD:-0.0}"

cd "${REPO_DIR}" || exit 1
mkdir -p outputs/semantic_id_stage

RUN_ROOT_INPUT="${RUN_ROOT:-}"
RUN_ROOT="${RUN_ROOT_INPUT:-${REPO_DIR}/outputs/semantic_id_stage}"
SUMMARY_FILE="${RUN_ROOT}/SUMMARY.md"
PROXY_OUTPUT_DIR="${RUN_ROOT}/proxy_metrics"
COMPARISON_CSV="${RUN_ROOT}/semantic_id_stage_comparison.csv"
COMPARISON_JSON="${RUN_ROOT}/semantic_id_stage_comparison.json"
COMPARISON_MD="${RUN_ROOT}/semantic_id_stage_comparison.md"
SWEEP_ROOT="${RUN_ROOT}/sweeps"

case "${RUN_MODE}" in
  base_only|all_three|analyze_only)
    ;;
  *)
    echo "Unsupported RUN_MODE: ${RUN_MODE}. Expected one of: base_only, all_three, analyze_only." >&2
    exit 1
    ;;
esac

case "${RUN_PROXY_METRICS}" in
  0|1)
    ;;
  *)
    echo "Unsupported RUN_PROXY_METRICS: ${RUN_PROXY_METRICS}. Expected 0 or 1." >&2
    exit 1
    ;;
esac

if [[ "${RUN_MODE}" == "analyze_only" ]]; then
  if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "RUN_ROOT does not exist for analyze_only mode: ${RUN_ROOT}" >&2
    exit 1
  fi
  if [[ "${RUN_PROXY_METRICS}" != "1" ]]; then
    echo "RUN_MODE=analyze_only requires RUN_PROXY_METRICS=1." >&2
    exit 1
  fi
else
  mkdir -p "${RUN_ROOT}"
fi

if [[ "${RUN_MODE}" != "analyze_only" ]]; then
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "DATA_DIR does not exist: ${DATA_DIR}" >&2
    exit 1
  fi

  if [[ ! -f "${EMBEDDING_PATH}" ]]; then
    echo "EMBEDDING_PATH does not exist: ${EMBEDDING_PATH}" >&2
    exit 1
  fi
elif [[ -n "${EMBEDDING_PATH}" && ! -f "${EMBEDDING_PATH}" ]]; then
  echo "EMBEDDING_PATH does not exist for analyze_only mode, continuing without embedding fallback: ${EMBEDDING_PATH}" >&2
  EMBEDDING_PATH=""
fi

log_summary_line() {
  printf '%s\n' "$1" | tee -a "${SUMMARY_FILE}"
}

reset_path() {
  local target_path="$1"
  if [[ -e "${target_path}" ]]; then
    rm -rf "${target_path}"
  fi
}

latest_metrics_file_for_run() {
  local run_dir="$1"
  local latest_metrics_file=""
  if [[ -d "${run_dir}/csv" ]]; then
    latest_metrics_file="$(find "${run_dir}/csv" -path '*/metrics.csv' | sort -V | tail -n 1 || true)"
  fi
  printf '%s\n' "${latest_metrics_file}"
}

experiment_name_for_label() {
  local label="$1"
  case "${label}" in
    base)
      printf '%s\n' "rkmeans_train_flat"
      ;;
    euc_prefix)
      printf '%s\n' "rkmeans_train_flat_euc_prefix"
      ;;
    hyp_prefix)
      printf '%s\n' "rkmeans_train_flat_hyp_prefix"
      ;;
    *)
      echo "Unsupported label for experiment lookup: ${label}" >&2
      exit 1
      ;;
  esac
}

inference_experiment_name_for_label() {
  local label="$1"
  case "${label}" in
    base)
      printf '%s\n' "rkmeans_inference_flat"
      ;;
    euc_prefix)
      printf '%s\n' "rkmeans_inference_flat_euc_prefix"
      ;;
    hyp_prefix)
      printf '%s\n' "rkmeans_inference_flat_hyp_prefix"
      ;;
    *)
      echo "Unsupported label for inference experiment lookup: ${label}" >&2
      exit 1
      ;;
  esac
}

set_label_value() {
  local prefix="$1"
  local label="$2"
  local value="$3"
  local var_name="${prefix}_${label}"
  printf -v "${var_name}" '%s' "${value}"
}

get_label_value() {
  local prefix="$1"
  local label="$2"
  local var_name="${prefix}_${label}"
  printf '%s\n' "${!var_name-}"
}

append_metrics_summary() {
  local metrics_file="$1"
  python - "$metrics_file" <<'PY'
import csv
import os
import sys

metrics_file = sys.argv[1]
interesting = [
    "step",
    "epoch",
    "train/loss_epoch",
    "train/loss_step",
    "train/loss",
    "train/quantization_loss_epoch",
    "train/quantization_loss_step",
    "train/quantization_loss",
    "train/reconstruction_loss_epoch",
    "train/reconstruction_loss_step",
    "train/reconstruction_loss",
    "train/hierarchy_loss_epoch",
    "train/hierarchy_loss_step",
    "train/hierarchy_loss",
]

if not os.path.exists(metrics_file):
    print("metrics_file: missing")
    raise SystemExit(0)

last_seen = {}
with open(metrics_file, newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        for key, value in row.items():
            if value not in ("", None):
                last_seen[key] = value

for key in interesting:
    print(f"{key}: {last_seen.get(key, 'N/A')}")
PY
}

log_prefixed_metrics_summary() {
  local metrics_summary="$1"
  if [[ -z "${metrics_summary}" ]]; then
    return
  fi

  printf '%s\n' "${metrics_summary}" | while IFS= read -r line; do
    log_summary_line "- ${line}"
  done
}

write_header() {
  {
    echo "# Semantic ID Prefix Regularizer HPC Run"
    echo
    echo "- run_mode: \`${RUN_MODE}\`"
    echo "- run_proxy_metrics: \`${RUN_PROXY_METRICS}\`"
    echo "- run_root: \`${RUN_ROOT}\`"
    echo "- data_dir: \`${DATA_DIR}\`"
    echo "- embedding_path: \`${EMBEDDING_PATH}\`"
    echo "- embedding_dim: \`${EMBEDDING_DIM}\`"
    echo "- num_hierarchies: \`${NUM_HIERARCHIES}\`"
    echo "- codebook_width: \`${CODEBOOK_WIDTH}\`"
    echo "- top_k: \`${TOP_K}\`"
    echo "- metadata_csv: \`${METADATA_CSV:-N/A}\`"
    echo "- metadata_json: \`${METADATA_JSON:-N/A}\`"
    echo "- category_field: \`${CATEGORY_FIELD:-N/A}\`"
    echo "- prefix_weight_sweep: \`${PREFIX_WEIGHT_SWEEP}\`"
    echo "- collision_worse_threshold: \`${COLLISION_WORSE_THRESHOLD}\`"
    echo "- frac_unique_min_improvement: \`${FRAC_UNIQUE_MIN_IMPROVEMENT}\`"
    echo "- entropy_avg_drop_threshold: \`${ENTROPY_AVG_DROP_THRESHOLD}\`"
    echo "- entropy_layer_drop_threshold: \`${ENTROPY_LAYER_DROP_THRESHOLD}\`"
    echo "- quantization_loss_relative_threshold: \`${QUANTIZATION_LOSS_RELATIVE_THRESHOLD}\`"
    echo "- quantization_loss_absolute_threshold: \`${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD}\`"
    echo "- loss_explosion_threshold: \`${LOSS_EXPLOSION_THRESHOLD}\`"
    echo "- prefix_worse_threshold: \`${PREFIX_WORSE_THRESHOLD}\`"
    echo
  } | tee "${SUMMARY_FILE}"
}

print_training_summary() {
  local label="$1"
  local experiment_name="$2"
  local run_dir="$3"
  local metrics_file
  metrics_file="$(latest_metrics_file_for_run "${run_dir}")"
  local stdout_log="${run_dir}/stdout.log"
  local stderr_log="${run_dir}/stderr.log"
  local checkpoint_path="$4"
  local metrics_summary=""

  metrics_summary="$(append_metrics_summary "${metrics_file}")"

  log_summary_line "## ${label} Training"
  log_summary_line "- experiment: \`${experiment_name}\`"
  log_summary_line "- output_dir: \`${run_dir}\`"
  log_summary_line "- checkpoint: \`${checkpoint_path}\`"
  log_summary_line "- stdout_log: \`${stdout_log}\`"
  log_summary_line "- stderr_log: \`${stderr_log}\`"
  log_prefixed_metrics_summary "${metrics_summary}"
  log_summary_line ""

  echo "----- ${label} summary -----"
  echo "experiment: ${experiment_name}"
  echo "output_dir: ${run_dir}"
  echo "checkpoint: ${checkpoint_path}"
  echo "stdout_log: ${stdout_log}"
  echo "stderr_log: ${stderr_log}"
  echo "${metrics_summary}"
  echo
}

print_inference_summary() {
  local label="$1"
  local inference_dir="$2"
  local semantic_id_dir="$3"
  local stdout_log="${inference_dir}/stdout.log"
  local stderr_log="${inference_dir}/stderr.log"

  log_summary_line "## ${label} Inference"
  log_summary_line "- output_dir: \`${inference_dir}\`"
  log_summary_line "- semantic_id_dir: \`${semantic_id_dir}\`"
  log_summary_line "- stdout_log: \`${stdout_log}\`"
  log_summary_line "- stderr_log: \`${stderr_log}\`"
  log_summary_line ""

  echo "----- ${label} inference -----"
  echo "output_dir: ${inference_dir}"
  echo "semantic_id_dir: ${semantic_id_dir}"
  echo "stdout_log: ${stdout_log}"
  echo "stderr_log: ${stderr_log}"
  echo
}

populate_selected_methods() {
  case "${RUN_MODE}" in
    base_only)
      SELECTED_METHODS=("base")
      ;;
    all_three|analyze_only)
      SELECTED_METHODS=("base" "euc_prefix" "hyp_prefix")
      ;;
  esac
}

populate_existing_analysis_inputs() {
  local selected_methods=("$@")

  for label in "${selected_methods[@]}"; do
    local train_dir="${RUN_ROOT}/${label}"
    local inference_dir="${RUN_ROOT}/inference/${label}"
    local semantic_id_dir="${inference_dir}/pickle"

    if [[ ! -d "${train_dir}" ]]; then
      echo "Missing training output for ${label}: ${train_dir}" >&2
      exit 1
    fi
    if [[ ! -d "${inference_dir}" ]]; then
      echo "Missing inference output for ${label}: ${inference_dir}" >&2
      exit 1
    fi
    if [[ ! -f "${semantic_id_dir}/merged_predictions.pkl" && ! -f "${semantic_id_dir}/merged_predictions_tensor.pt" ]]; then
      echo "Missing semantic id output for ${label}: ${semantic_id_dir}" >&2
      exit 1
    fi

    set_label_value "TRAIN_DIR" "${label}" "${train_dir}"
    set_label_value "INFERENCE_DIR" "${label}" "${inference_dir}"
    set_label_value "SEMANTIC_ID_DIR" "${label}" "${semantic_id_dir}"
  done
}

find_checkpoint_path() {
  local run_dir="$1"
  local checkpoint_path="N/A"

  if [[ -d "${run_dir}/checkpoints" ]]; then
    checkpoint_path="$(find "${run_dir}/checkpoints" -name '*.ckpt' | sort | tail -n 1 || true)"
  fi

  if [[ -z "${checkpoint_path}" ]]; then
    checkpoint_path="N/A"
  fi

  printf '%s\n' "${checkpoint_path}"
}

weight_slug() {
  local weight="$1"
  printf '%s\n' "${weight}" | tr '.' '_'
}

parse_weight_sweep() {
  local raw_weights="${PREFIX_WEIGHT_SWEEP}"
  raw_weights="${raw_weights//,/ }"
  SWEEP_WEIGHTS=()
  local weight=""
  for weight in ${raw_weights}; do
    if [[ -n "${weight}" ]]; then
      SWEEP_WEIGHTS+=("${weight}")
    fi
  done
  if [[ "${#SWEEP_WEIGHTS[@]}" -eq 0 ]]; then
    echo "PREFIX_WEIGHT_SWEEP resolved to an empty list." >&2
    exit 1
  fi
}

copy_dir_contents() {
  local source_dir="$1"
  local target_dir="$2"
  reset_path "${target_dir}"
  mkdir -p "${target_dir}"
  cp -a "${source_dir}/." "${target_dir}/"
}

write_value_file() {
  local output_path="$1"
  local value="$2"
  printf '%s\n' "${value}" > "${output_path}"
}

append_markdown_file() {
  local input_path="$1"
  if [[ -f "${input_path}" ]]; then
    cat "${input_path}" >> "${SUMMARY_FILE}"
    printf '\n' >> "${SUMMARY_FILE}"
  fi
}

run_proxy_metrics_for_specs() {
  local output_dir="$1"
  shift
  local cmd=(
    python scripts/analyze_semantic_ids.py
    "--output-dir" "${output_dir}"
    "--top-k" "${TOP_K}"
  )

  local run_spec=""
  for run_spec in "$@"; do
    cmd+=("--run" "${run_spec}")
  done

  if [[ -n "${EMBEDDING_PATH}" ]]; then
    cmd+=("--embedding-path" "${EMBEDDING_PATH}")
  fi
  if [[ -n "${CODEBOOK_WIDTH}" ]]; then
    cmd+=("--codebook-size" "${CODEBOOK_WIDTH}")
  fi
  if [[ -n "${NUM_HIERARCHIES}" ]]; then
    cmd+=("--num-hierarchies" "${NUM_HIERARCHIES}")
  fi
  if [[ -n "${METADATA_CSV}" ]]; then
    cmd+=("--metadata-csv" "${METADATA_CSV}")
  fi
  if [[ -n "${METADATA_JSON}" ]]; then
    cmd+=("--metadata-json" "${METADATA_JSON}")
  fi
  if [[ -n "${CATEGORY_FIELD}" ]]; then
    cmd+=("--category-field" "${CATEGORY_FIELD}")
  fi

  reset_path "${output_dir}"
  mkdir -p "${output_dir}"
  printf 'command: %q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

run_training_experiment() {
  local label="$1"
  local experiment_name="$2"
  local run_dir="$3"
  shift 3
  local stdout_log="${run_dir}/stdout.log"
  local stderr_log="${run_dir}/stderr.log"

  reset_path "${run_dir}"
  mkdir -p "${run_dir}"

  local cmd=(
    python -m src.train
    "experiment=${experiment_name}"
    "data_dir=${DATA_DIR}"
    "embedding_path=${EMBEDDING_PATH}"
    "embedding_dim=${EMBEDDING_DIM}"
    "num_hierarchies=${NUM_HIERARCHIES}"
    "codebook_width=${CODEBOOK_WIDTH}"
    "paths.output_dir=${run_dir}"
    "hydra.run.dir=${run_dir}"
  )
  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi

  echo "===== running ${label} (${experiment_name}) ====="
  printf 'command: %q ' "${cmd[@]}"
  printf '\n'

  if "${cmd[@]}" >"${stdout_log}" 2>"${stderr_log}"; then
    echo "status: success"
  else
    local status=$?
    echo "status: failed (${status})"
    echo "stdout tail:"
    tail -n 40 "${stdout_log}" || true
    echo "stderr tail:"
    tail -n 40 "${stderr_log}" || true
    exit "${status}"
  fi

  local checkpoint_path
  checkpoint_path="$(find_checkpoint_path "${run_dir}")"
  if [[ "${checkpoint_path}" == "N/A" ]]; then
    echo "No checkpoint found in ${run_dir}/checkpoints" >&2
    exit 1
  fi

  set_label_value "TRAIN_DIR" "${label}" "${run_dir}"
  set_label_value "CHECKPOINT_PATH" "${label}" "${checkpoint_path}"
  print_training_summary "${label}" "${experiment_name}" "${run_dir}" "${checkpoint_path}"
}

run_inference_experiment() {
  local label="$1"
  local checkpoint_path="$2"
  local inference_dir="$3"
  local inference_label="$4"
  local inference_experiment_name
  inference_experiment_name="$(inference_experiment_name_for_label "${inference_label}")"
  shift 4
  local stdout_log="${inference_dir}/stdout.log"
  local stderr_log="${inference_dir}/stderr.log"
  local semantic_id_dir="${inference_dir}/pickle"

  reset_path "${inference_dir}"
  mkdir -p "${inference_dir}"

  local cmd=(
    python -m src.inference
    "experiment=${inference_experiment_name}"
    "data_dir=${DATA_DIR}"
    "embedding_path=${EMBEDDING_PATH}"
    "embedding_dim=${EMBEDDING_DIM}"
    "num_hierarchies=${NUM_HIERARCHIES}"
    "codebook_width=${CODEBOOK_WIDTH}"
    "ckpt_path=${checkpoint_path}"
    "callbacks.bq_writer=null"
    "paths.output_dir=${inference_dir}"
    "hydra.run.dir=${inference_dir}"
  )
  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi

  echo "===== running ${label} inference ====="
  printf 'command: %q ' "${cmd[@]}"
  printf '\n'

  if "${cmd[@]}" >"${stdout_log}" 2>"${stderr_log}"; then
    echo "status: success"
  else
    local status=$?
    echo "status: failed (${status})"
    echo "stdout tail:"
    tail -n 40 "${stdout_log}" || true
    echo "stderr tail:"
    tail -n 40 "${stderr_log}" || true
    exit "${status}"
  fi

  if [[ ! -f "${semantic_id_dir}/merged_predictions.pkl" && ! -f "${semantic_id_dir}/merged_predictions_tensor.pt" ]]; then
    echo "Inference output missing merged semantic ids in ${semantic_id_dir}" >&2
    exit 1
  fi

  set_label_value "INFERENCE_DIR" "${label}" "${inference_dir}"
  set_label_value "SEMANTIC_ID_DIR" "${label}" "${semantic_id_dir}"
  print_inference_summary "${label}" "${inference_dir}" "${semantic_id_dir}"
}

run_baseline_experiment() {
  local label="base"
  local train_dir="${RUN_ROOT}/${label}"
  local inference_dir="${RUN_ROOT}/inference/${label}"
  run_training_experiment "${label}" "$(experiment_name_for_label "${label}")" "${train_dir}"
  run_inference_experiment "${label}" "$(get_label_value "CHECKPOINT_PATH" "${label}")" "${inference_dir}" "${label}"
}

run_weight_sweep_for_label() {
  local label="$1"
  local experiment_name
  experiment_name="$(experiment_name_for_label "${label}")"
  local sweep_dir="${SWEEP_ROOT}/${label}"
  local sweep_proxy_dir="${sweep_dir}/proxy_metrics"
  local selection_json="${sweep_dir}/selection.json"
  local selection_md="${sweep_dir}/selection.md"
  local baseline_train_dir
  baseline_train_dir="$(get_label_value "TRAIN_DIR" "base")"
  local baseline_semantic_id_dir
  baseline_semantic_id_dir="$(get_label_value "SEMANTIC_ID_DIR" "base")"
  local base_proxy_dir="${SWEEP_ROOT}/baseline_reference"
  local baseline_proxy_json="${base_proxy_dir}/base.json"

  reset_path "${sweep_dir}"
  mkdir -p "${sweep_dir}"

  run_proxy_metrics_for_specs "${base_proxy_dir}" "base=${baseline_semantic_id_dir}"

  local candidate_specs=()
  local weight=""
  for weight in "${SWEEP_WEIGHTS[@]}"; do
    local weight_slug_value
    weight_slug_value="$(weight_slug "${weight}")"
    local candidate_label="${label}_w_${weight_slug_value}"
    local candidate_dir="${sweep_dir}/${candidate_label}"
    local candidate_train_dir="${candidate_dir}/train"
    local candidate_inference_dir="${candidate_dir}/inference"

    run_training_experiment \
      "${candidate_label}" \
      "${experiment_name}" \
      "${candidate_train_dir}" \
      "model.hierarchy_loss_weight=${weight}"
    run_inference_experiment \
      "${candidate_label}" \
      "$(get_label_value "CHECKPOINT_PATH" "${candidate_label}")" \
      "${candidate_inference_dir}" \
      "${label}"

    candidate_specs+=(
      "${weight}|${candidate_label}|${candidate_train_dir}|${candidate_inference_dir}|${sweep_proxy_dir}/${candidate_label}.json"
    )
  done

  local proxy_run_specs=("base=${baseline_semantic_id_dir}")
  for weight in "${SWEEP_WEIGHTS[@]}"; do
    local weight_slug_value
    weight_slug_value="$(weight_slug "${weight}")"
    local candidate_label="${label}_w_${weight_slug_value}"
    local candidate_semantic_id_dir
    candidate_semantic_id_dir="$(get_label_value "SEMANTIC_ID_DIR" "${candidate_label}")"
    proxy_run_specs+=("${candidate_label}=${candidate_semantic_id_dir}")
  done
  run_proxy_metrics_for_specs "${sweep_proxy_dir}" "${proxy_run_specs[@]}"

  local select_cmd=(
    python scripts/select_semantic_id_weight.py
    "--method-family" "${label}"
    "--baseline-train-dir" "${baseline_train_dir}"
    "--baseline-proxy-json" "${baseline_proxy_json}"
    "--output-json" "${selection_json}"
    "--output-md" "${selection_md}"
    "--collision-worse-threshold" "${COLLISION_WORSE_THRESHOLD}"
    "--frac-unique-min-improvement" "${FRAC_UNIQUE_MIN_IMPROVEMENT}"
    "--entropy-avg-drop-threshold" "${ENTROPY_AVG_DROP_THRESHOLD}"
    "--entropy-layer-drop-threshold" "${ENTROPY_LAYER_DROP_THRESHOLD}"
    "--quantization-loss-relative-threshold" "${QUANTIZATION_LOSS_RELATIVE_THRESHOLD}"
    "--quantization-loss-absolute-threshold" "${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD}"
    "--loss-explosion-threshold" "${LOSS_EXPLOSION_THRESHOLD}"
    "--prefix-worse-threshold" "${PREFIX_WORSE_THRESHOLD}"
  )
  local candidate_spec=""
  for candidate_spec in "${candidate_specs[@]}"; do
    select_cmd+=("--candidate" "${candidate_spec}")
  done

  printf 'command: %q ' "${select_cmd[@]}"
  printf '\n'
  "${select_cmd[@]}"

  local selected_weight selected_train_dir selected_inference_dir selected_method selection_mode
  selected_weight="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_weight_text"])' "${selection_json}")"
  selected_method="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_method"])' "${selection_json}")"
  selected_train_dir="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_train_dir"])' "${selection_json}")"
  selected_inference_dir="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_inference_dir"])' "${selection_json}")"
  selection_mode="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selection_mode"])' "${selection_json}")"

  local final_train_dir="${RUN_ROOT}/${label}"
  local final_inference_dir="${RUN_ROOT}/inference/${label}"
  copy_dir_contents "${selected_train_dir}" "${final_train_dir}"
  copy_dir_contents "${selected_inference_dir}" "${final_inference_dir}"
  cp "${selection_json}" "${final_train_dir}/weight_selection.json"
  cp "${selection_md}" "${final_train_dir}/weight_selection.md"
  write_value_file "${final_train_dir}/selected_weight.txt" "${selected_weight}"
  write_value_file "${final_train_dir}/selection_mode.txt" "${selection_mode}"

  set_label_value "TRAIN_DIR" "${label}" "${final_train_dir}"
  set_label_value "CHECKPOINT_PATH" "${label}" "$(find_checkpoint_path "${final_train_dir}")"
  set_label_value "INFERENCE_DIR" "${label}" "${final_inference_dir}"
  set_label_value "SEMANTIC_ID_DIR" "${label}" "${final_inference_dir}/pickle"
  if [[ "$(get_label_value "CHECKPOINT_PATH" "${label}")" == "N/A" ]]; then
    echo "Failed to promote a checkpoint for ${label} into ${final_train_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${final_inference_dir}/pickle" ]]; then
    echo "Failed to promote semantic id outputs for ${label} into ${final_inference_dir}" >&2
    exit 1
  fi

  log_summary_line "## ${label} Retained Weight"
  log_summary_line "- selected_method: \`${selected_method}\`"
  log_summary_line "- selected_weight: \`${selected_weight}\`"
  log_summary_line "- selection_mode: \`${selection_mode}\`"
  log_summary_line "- promoted_train_dir: \`${final_train_dir}\`"
  log_summary_line "- promoted_inference_dir: \`${final_inference_dir}\`"
  log_summary_line ""
  append_markdown_file "${selection_md}"
}

run_proxy_metrics() {
  reset_path "${COMPARISON_CSV}"
  reset_path "${COMPARISON_JSON}"
  reset_path "${COMPARISON_MD}"

  echo "===== running proxy metrics ====="
  local base_semantic_id_dir
  local euc_semantic_id_dir
  local hyp_semantic_id_dir
  base_semantic_id_dir="$(get_label_value "SEMANTIC_ID_DIR" "base")"
  euc_semantic_id_dir="$(get_label_value "SEMANTIC_ID_DIR" "euc_prefix")"
  hyp_semantic_id_dir="$(get_label_value "SEMANTIC_ID_DIR" "hyp_prefix")"
  run_proxy_metrics_for_specs \
    "${PROXY_OUTPUT_DIR}" \
    "base=${base_semantic_id_dir}" \
    "euc_prefix=${euc_semantic_id_dir}" \
    "hyp_prefix=${hyp_semantic_id_dir}"

  log_summary_line "## Proxy Metrics"
  log_summary_line "- output_dir: \`${PROXY_OUTPUT_DIR}\`"
  log_summary_line "- comparison_json: \`${PROXY_OUTPUT_DIR}/comparison.json\`"
  log_summary_line "- comparison_csv: \`${PROXY_OUTPUT_DIR}/comparison.csv\`"
  log_summary_line "- summary_markdown: \`${PROXY_OUTPUT_DIR}/SUMMARY.md\`"
  log_summary_line ""
}

build_stage_comparison() {
  echo "===== building stage comparison table ====="

  local base_train_dir
  local euc_train_dir
  local hyp_train_dir
  local base_inference_dir
  local euc_inference_dir
  local hyp_inference_dir
  base_train_dir="$(get_label_value "TRAIN_DIR" "base")"
  euc_train_dir="$(get_label_value "TRAIN_DIR" "euc_prefix")"
  hyp_train_dir="$(get_label_value "TRAIN_DIR" "hyp_prefix")"
  base_inference_dir="$(get_label_value "INFERENCE_DIR" "base")"
  euc_inference_dir="$(get_label_value "INFERENCE_DIR" "euc_prefix")"
  hyp_inference_dir="$(get_label_value "INFERENCE_DIR" "hyp_prefix")"

  local cmd=(
    python scripts/build_semantic_id_stage_comparison.py
    "--train-run" "base=${base_train_dir}"
    "--train-run" "euc_prefix=${euc_train_dir}"
    "--train-run" "hyp_prefix=${hyp_train_dir}"
    "--inference-run" "base=${base_inference_dir}"
    "--inference-run" "euc_prefix=${euc_inference_dir}"
    "--inference-run" "hyp_prefix=${hyp_inference_dir}"
    "--proxy-dir" "${PROXY_OUTPUT_DIR}"
    "--output-csv" "${COMPARISON_CSV}"
    "--output-json" "${COMPARISON_JSON}"
    "--output-md" "${COMPARISON_MD}"
  )

  printf 'command: %q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"

  log_summary_line "## Stage Comparison"
  log_summary_line "- comparison_csv: \`${COMPARISON_CSV}\`"
  log_summary_line "- comparison_json: \`${COMPARISON_JSON}\`"
  log_summary_line "- comparison_markdown: \`${COMPARISON_MD}\`"
  log_summary_line ""
  append_markdown_file "${COMPARISON_MD}"
}

write_header
parse_weight_sweep
SELECTED_METHODS=()
case "${RUN_MODE}" in
  analyze_only)
    populate_selected_methods
    populate_existing_analysis_inputs "${SELECTED_METHODS[@]}"
    run_proxy_metrics
    build_stage_comparison
    ;;
  base_only)
    run_baseline_experiment
    if [[ "${RUN_PROXY_METRICS}" == "1" ]]; then
      echo "RUN_PROXY_METRICS=1 was ignored because RUN_MODE=base_only only runs the baseline semantic ID train+inference."
      log_summary_line "## Analysis Skipped"
      log_summary_line "- reason: \`RUN_MODE=base_only ignores RUN_PROXY_METRICS and skips proxy metrics/comparison.\`"
      log_summary_line ""
    fi
    ;;
  all_three)
    if [[ "${RUN_PROXY_METRICS}" == "0" ]]; then
      echo "RUN_PROXY_METRICS=0 is ignored in all_three mode because weight screening requires proxy metrics."
      log_summary_line "## Analysis Mode"
      log_summary_line "- note: \`RUN_MODE=all_three always runs the internal weight sweep analysis and final stage comparison.\`"
      log_summary_line ""
    fi
    run_baseline_experiment
    run_weight_sweep_for_label "euc_prefix"
    run_weight_sweep_for_label "hyp_prefix"
    run_proxy_metrics
    build_stage_comparison
    ;;
esac

echo "all experiments completed successfully"
echo "summary markdown: ${SUMMARY_FILE}"
if [[ "${RUN_MODE}" != "base_only" ]]; then
  echo "proxy metrics dir: ${PROXY_OUTPUT_DIR}"
  echo "stage comparison csv: ${COMPARISON_CSV}"
fi
echo "run root: ${RUN_ROOT}"
