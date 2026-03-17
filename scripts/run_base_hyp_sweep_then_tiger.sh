#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/amazon_data}"
DATASET_NAME="${DATASET_NAME:-beauty}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-${DATASET_NAME}_seed${SEED}}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/base_hyp_matrix/${RUN_NAME}}"

EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"
SEMANTIC_EMBEDDING_MODEL="${SEMANTIC_EMBEDDING_MODEL:-google/flan-t5-xl}"
NUM_HIERARCHIES="${NUM_HIERARCHIES:-3}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"
TIGER_NUM_HIERARCHIES="${TIGER_NUM_HIERARCHIES:-4}"
TOP_K="${TOP_K:-5}"
PREFIX_WEIGHT_SWEEP="${PREFIX_WEIGHT_SWEEP:-0.05,0.1,0.2,0.3,0.4}"
RUN_SEMANTIC_STAGE="${RUN_SEMANTIC_STAGE:-1}"
RUN_DOWNSTREAM_STAGE="${RUN_DOWNSTREAM_STAGE:-1}"

COLLISION_WORSE_THRESHOLD="${COLLISION_WORSE_THRESHOLD:-0.03}"
FRAC_UNIQUE_MIN_IMPROVEMENT="${FRAC_UNIQUE_MIN_IMPROVEMENT:--0.01}"
ENTROPY_AVG_DROP_THRESHOLD="${ENTROPY_AVG_DROP_THRESHOLD:-0.15}"
ENTROPY_LAYER_DROP_THRESHOLD="${ENTROPY_LAYER_DROP_THRESHOLD:-0.2}"
QUANTIZATION_LOSS_RELATIVE_THRESHOLD="${QUANTIZATION_LOSS_RELATIVE_THRESHOLD:-0.05}"
QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD="${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD:-1.0}"
LOSS_EXPLOSION_THRESHOLD="${LOSS_EXPLOSION_THRESHOLD:-100000}"
PREFIX_WORSE_THRESHOLD="${PREFIX_WORSE_THRESHOLD:-0.0}"
ENABLE_DOWNSTREAM_VAL_SURROGATE="${ENABLE_DOWNSTREAM_VAL_SURROGATE:-0}"
SURROGATE_MAX_STEPS="${SURROGATE_MAX_STEPS:-3200}"
SURROGATE_VAL_CHECK_INTERVAL="${SURROGATE_VAL_CHECK_INTERVAL:-800}"
SURROGATE_EARLY_STOPPING_PATIENCE="${SURROGATE_EARLY_STOPPING_PATIENCE:-2}"
SURROGATE_PRIMARY_METRIC="${SURROGATE_PRIMARY_METRIC:-val/recall@5}"
SURROGATE_SECONDARY_METRIC="${SURROGATE_SECONDARY_METRIC:-val/ndcg@10}"

SEMANTIC_EMBEDDING_OUTPUT_DIR="${SEMANTIC_EMBEDDING_OUTPUT_DIR:-${RUN_ROOT}/semantic_embeddings}"
EMBEDDING_PATH="${EMBEDDING_PATH:-${SEMANTIC_EMBEDDING_OUTPUT_DIR}/pickle/merged_predictions_tensor.pt}"

SEMANTIC_ID_STAGE_ROOT="${SEMANTIC_ID_STAGE_ROOT:-${RUN_ROOT}/semantic_id_stage}"
BASE_TRAIN_DIR="${BASE_TRAIN_DIR:-${SEMANTIC_ID_STAGE_ROOT}/base}"
BASE_CKPT_PATH="${BASE_CKPT_PATH:-${BASE_TRAIN_DIR}/checkpoints/last.ckpt}"
BASE_INFERENCE_DIR="${BASE_INFERENCE_DIR:-${SEMANTIC_ID_STAGE_ROOT}/inference/base}"
BASE_SEMANTIC_ID_PATH="${BASE_SEMANTIC_ID_PATH:-${BASE_INFERENCE_DIR}/pickle/merged_predictions_tensor.pt}"
HYP_TRAIN_DIR="${HYP_TRAIN_DIR:-${SEMANTIC_ID_STAGE_ROOT}/hyp_prefix}"
HYP_CKPT_PATH="${HYP_CKPT_PATH:-${HYP_TRAIN_DIR}/checkpoints/last.ckpt}"
HYP_INFERENCE_DIR="${HYP_INFERENCE_DIR:-${SEMANTIC_ID_STAGE_ROOT}/inference/hyp_prefix}"
HYP_SEMANTIC_ID_PATH="${HYP_SEMANTIC_ID_PATH:-${HYP_INFERENCE_DIR}/pickle/merged_predictions_tensor.pt}"
SWEEP_ROOT="${SWEEP_ROOT:-${SEMANTIC_ID_STAGE_ROOT}/sweeps/hyp_prefix}"
SWEEP_PROXY_DIR="${SWEEP_PROXY_DIR:-${SWEEP_ROOT}/proxy_metrics}"
SURROGATE_ROOT="${SURROGATE_ROOT:-${SWEEP_ROOT}/downstream_surrogate}"
PROXY_OUTPUT_DIR="${PROXY_OUTPUT_DIR:-${SEMANTIC_ID_STAGE_ROOT}/proxy_metrics}"
COMPARISON_CSV="${COMPARISON_CSV:-${SEMANTIC_ID_STAGE_ROOT}/semantic_id_stage_comparison.csv}"
COMPARISON_JSON="${COMPARISON_JSON:-${SEMANTIC_ID_STAGE_ROOT}/semantic_id_stage_comparison.json}"
COMPARISON_MD="${COMPARISON_MD:-${SEMANTIC_ID_STAGE_ROOT}/semantic_id_stage_comparison.md}"

RECOMMENDATION_STAGE_ROOT="${RECOMMENDATION_STAGE_ROOT:-${RUN_ROOT}/recommendation_stage}"
BASE_TIGER_DIR="${BASE_TIGER_DIR:-${RECOMMENDATION_STAGE_ROOT}/tiger_train_base}"
HYP_TIGER_DIR="${HYP_TIGER_DIR:-${RECOMMENDATION_STAGE_ROOT}/tiger_train_hyp_prefix}"

SUMMARY_FILE="${SUMMARY_FILE:-${RUN_ROOT}/SUMMARY.md}"

cd "${REPO_DIR}" || exit 1
mkdir -p "${RUN_ROOT}"

log() {
  printf '%s\n' "$1" | tee -a "${SUMMARY_FILE}"
}

reset_path() {
  local target_path="$1"
  if [[ -e "${target_path}" ]]; then
    rm -rf "${target_path}"
  fi
}

build_distributed_launcher() {
  TRAIN_DISTRIBUTED_LAUNCHER=()
  INFERENCE_DISTRIBUTED_LAUNCHER=()
  SURROGATE_DISTRIBUTED_LAUNCHER=()
  if [[ "${USE_SLURM_SRUN:-0}" != "1" ]]; then
    return
  fi

  local ntasks="${SLURM_NTASKS:-}"
  local ntasks_per_node="${SLURM_NTASKS_PER_NODE:-}"

  if [[ -z "${ntasks}" ]]; then
    return
  fi

  TRAIN_DISTRIBUTED_LAUNCHER=(srun --ntasks="${ntasks}")
  if [[ -n "${ntasks_per_node}" ]]; then
    TRAIN_DISTRIBUTED_LAUNCHER+=(--ntasks-per-node="${ntasks_per_node}")
  fi

  INFERENCE_DISTRIBUTED_LAUNCHER=(srun --ntasks=1 --ntasks-per-node=1)
  SURROGATE_DISTRIBUTED_LAUNCHER=(srun --ntasks=1 --ntasks-per-node=1)
}

run_command() {
  local label="$1"
  local stdout_log="$2"
  local stderr_log="$3"
  shift 3

  mkdir -p "$(dirname "${stdout_log}")"
  mkdir -p "$(dirname "${stderr_log}")"

  echo "===== ${label} =====" | tee -a "${SUMMARY_FILE}"
  printf 'command: %q ' "$@" | tee -a "${SUMMARY_FILE}"
  printf '\n' | tee -a "${SUMMARY_FILE}"

  if "$@" >"${stdout_log}" 2>"${stderr_log}"; then
    log "status: success"
  else
    local status=$?
    log "status: failed (${status})"
    log "stdout_log: \`${stdout_log}\`"
    log "stderr_log: \`${stderr_log}\`"
    tail -n 40 "${stdout_log}" || true
    tail -n 40 "${stderr_log}" || true
    exit "${status}"
  fi
  log ""
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
    "train/quantization_loss_epoch",
    "train/reconstruction_loss_epoch",
    "train/hierarchy_loss_epoch",
    "val/recall@5",
    "val/recall@10",
    "val/ndcg@5",
    "val/ndcg@10",
    "test/recall@5",
    "test/recall@10",
    "test/ndcg@5",
    "test/ndcg@10",
]

if not os.path.exists(metrics_file):
    raise SystemExit(0)

last_seen = {}
with open(metrics_file, newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        for key, value in row.items():
            if value not in ("", None):
                last_seen[key] = value

for key in interesting:
    value = last_seen.get(key)
    if value not in (None, ""):
        print(f"{key}: {value}")
PY
}

log_metrics_file() {
  local title="$1"
  local metrics_file="$2"
  if [[ ! -f "${metrics_file}" ]]; then
    return
  fi
  log "## ${title}"
  while IFS= read -r line; do
    log "- ${line}"
  done < <(append_metrics_summary "${metrics_file}")
  log ""
}

latest_metrics_file_for_run() {
  local run_dir="$1"
  local latest_metrics_file=""
  if [[ -d "${run_dir}/csv" ]]; then
    latest_metrics_file="$(find "${run_dir}/csv" -path '*/metrics.csv' | sort -V | tail -n 1 || true)"
  fi
  printf '%s\n' "${latest_metrics_file}"
}

weight_slug() {
  printf '%s\n' "$1" | tr '.' '_'
}

copy_dir_contents() {
  local source_dir="$1"
  local target_dir="$2"
  reset_path "${target_dir}"
  mkdir -p "${target_dir}"
  cp -a "${source_dir}/." "${target_dir}/"
}

parse_weight_sweep() {
  local raw_weights="${PREFIX_WEIGHT_SWEEP//,/ }"
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

write_header() {
  if [[ "${RUN_SEMANTIC_STAGE}" == "1" ]]; then
    reset_path "${SUMMARY_FILE}"
    {
      echo "# Base + Hyp Sweep Run"
      echo
      echo "- dataset: \`${DATASET_NAME}\`"
      echo "- seed: \`${SEED}\`"
      echo "- run_name: \`${RUN_NAME}\`"
      echo "- run_root: \`${RUN_ROOT}\`"
      echo "- data_dir: \`${DATA_DIR}\`"
      echo "- embedding_dim: \`${EMBEDDING_DIM}\`"
      echo "- semantic_embedding_model: \`${SEMANTIC_EMBEDDING_MODEL}\`"
      echo "- semantic_num_hierarchies: \`${NUM_HIERARCHIES}\`"
      echo "- tiger_num_hierarchies: \`${TIGER_NUM_HIERARCHIES}\`"
      echo "- codebook_width: \`${CODEBOOK_WIDTH}\`"
      echo "- prefix_weight_sweep: \`${PREFIX_WEIGHT_SWEEP}\`"
      echo "- semantic checkpoint policy: \`last.ckpt only\`"
      echo "- run_semantic_stage: \`${RUN_SEMANTIC_STAGE}\`"
      echo "- run_downstream_stage: \`${RUN_DOWNSTREAM_STAGE}\`"
      echo
    } > "${SUMMARY_FILE}"
  elif [[ ! -f "${SUMMARY_FILE}" ]]; then
    {
      echo "# Base + Hyp Sweep Run"
      echo
      echo "- dataset: \`${DATASET_NAME}\`"
      echo "- seed: \`${SEED}\`"
      echo "- run_name: \`${RUN_NAME}\`"
      echo "- run_root: \`${RUN_ROOT}\`"
      echo "- data_dir: \`${DATA_DIR}\`"
      echo "- run_semantic_stage: \`${RUN_SEMANTIC_STAGE}\`"
      echo "- run_downstream_stage: \`${RUN_DOWNSTREAM_STAGE}\`"
      echo
    } > "${SUMMARY_FILE}"
  else
    {
      echo
      echo "## Downstream Resume"
      echo
    } >> "${SUMMARY_FILE}"
  fi
}

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi

if [[ "${RUN_SEMANTIC_STAGE}" != "1" && "${RUN_DOWNSTREAM_STAGE}" != "1" ]]; then
  echo "Nothing to do: both RUN_SEMANTIC_STAGE and RUN_DOWNSTREAM_STAGE are disabled." >&2
  exit 1
fi

parse_weight_sweep
build_distributed_launcher
write_header

if [[ "${RUN_SEMANTIC_STAGE}" == "1" ]]; then
  reset_path "${SEMANTIC_EMBEDDING_OUTPUT_DIR}"
  run_command \
    "semantic-embeddings" \
    "${SEMANTIC_EMBEDDING_OUTPUT_DIR}/stdout.log" \
    "${SEMANTIC_EMBEDDING_OUTPUT_DIR}/stderr.log" \
    "${INFERENCE_DISTRIBUTED_LAUNCHER[@]}" \
    python -m src.inference \
    experiment=sem_embeds_inference_flat \
    "data_dir=${DATA_DIR}" \
    "embedding_model=${SEMANTIC_EMBEDDING_MODEL}" \
    "seed=${SEED}" \
    "trainer.devices=1" \
    "paths.output_dir=${SEMANTIC_EMBEDDING_OUTPUT_DIR}" \
    "hydra.run.dir=${SEMANTIC_EMBEDDING_OUTPUT_DIR}"

  if [[ ! -f "${EMBEDDING_PATH}" ]]; then
    echo "Embedding output missing: ${EMBEDDING_PATH}" >&2
    exit 1
  fi

  reset_path "${BASE_TRAIN_DIR}"
  run_command \
    "semantic-base-train" \
    "${BASE_TRAIN_DIR}/stdout.log" \
    "${BASE_TRAIN_DIR}/stderr.log" \
    "${TRAIN_DISTRIBUTED_LAUNCHER[@]}" \
    python -m src.train \
    experiment=rkmeans_train_flat \
    "data_dir=${DATA_DIR}" \
    "embedding_path=${EMBEDDING_PATH}" \
    "embedding_dim=${EMBEDDING_DIM}" \
    "num_hierarchies=${NUM_HIERARCHIES}" \
    "codebook_width=${CODEBOOK_WIDTH}" \
    "seed=${SEED}" \
    "paths.output_dir=${BASE_TRAIN_DIR}" \
    "hydra.run.dir=${BASE_TRAIN_DIR}"

  if [[ ! -f "${BASE_CKPT_PATH}" ]]; then
    echo "Base checkpoint missing: ${BASE_CKPT_PATH}" >&2
    exit 1
  fi
  log_metrics_file "Base Semantic Train" "$(latest_metrics_file_for_run "${BASE_TRAIN_DIR}")"

  reset_path "${BASE_INFERENCE_DIR}"
  run_command \
    "semantic-base-inference" \
    "${BASE_INFERENCE_DIR}/stdout.log" \
    "${BASE_INFERENCE_DIR}/stderr.log" \
    "${INFERENCE_DISTRIBUTED_LAUNCHER[@]}" \
    python -m src.inference \
    experiment=rkmeans_inference_flat \
    "data_dir=${DATA_DIR}" \
    "embedding_path=${EMBEDDING_PATH}" \
    "embedding_dim=${EMBEDDING_DIM}" \
    "num_hierarchies=${NUM_HIERARCHIES}" \
    "codebook_width=${CODEBOOK_WIDTH}" \
    "seed=${SEED}" \
    "ckpt_path=${BASE_CKPT_PATH}" \
    "trainer.devices=1" \
    "callbacks.bq_writer=null" \
    "paths.output_dir=${BASE_INFERENCE_DIR}" \
    "hydra.run.dir=${BASE_INFERENCE_DIR}"

  if [[ ! -f "${BASE_SEMANTIC_ID_PATH}" ]]; then
    echo "Base semantic ids missing: ${BASE_SEMANTIC_ID_PATH}" >&2
    exit 1
  fi

  reset_path "${SWEEP_ROOT}"
  mkdir -p "${SWEEP_ROOT}"

  CANDIDATE_SPECS=()
  PROXY_RUN_SPECS=("base=${BASE_INFERENCE_DIR}/pickle")
  for weight in "${SWEEP_WEIGHTS[@]}"; do
    slug="$(weight_slug "${weight}")"
    candidate_method="hyp_prefix_w_${slug}"
    candidate_root="${SWEEP_ROOT}/${candidate_method}"
    candidate_train_dir="${candidate_root}/train"
    candidate_ckpt_path="${candidate_train_dir}/checkpoints/last.ckpt"
    candidate_inference_dir="${candidate_root}/inference"
    candidate_semantic_id_dir="${candidate_inference_dir}/pickle"

    reset_path "${candidate_train_dir}"
    run_command \
      "${candidate_method}-train" \
      "${candidate_train_dir}/stdout.log" \
      "${candidate_train_dir}/stderr.log" \
      "${TRAIN_DISTRIBUTED_LAUNCHER[@]}" \
      python -m src.train \
      experiment=rkmeans_train_flat_hyp_prefix \
      "data_dir=${DATA_DIR}" \
      "embedding_path=${EMBEDDING_PATH}" \
      "embedding_dim=${EMBEDDING_DIM}" \
      "num_hierarchies=${NUM_HIERARCHIES}" \
      "codebook_width=${CODEBOOK_WIDTH}" \
      "seed=${SEED}" \
      "model.hierarchy_loss_weight=${weight}" \
      "paths.output_dir=${candidate_train_dir}" \
      "hydra.run.dir=${candidate_train_dir}"

    if [[ ! -f "${candidate_ckpt_path}" ]]; then
      echo "Hyp candidate checkpoint missing: ${candidate_ckpt_path}" >&2
      exit 1
    fi
    log_metrics_file "${candidate_method} Semantic Train" "$(latest_metrics_file_for_run "${candidate_train_dir}")"

    reset_path "${candidate_inference_dir}"
    run_command \
      "${candidate_method}-inference" \
      "${candidate_inference_dir}/stdout.log" \
      "${candidate_inference_dir}/stderr.log" \
      "${INFERENCE_DISTRIBUTED_LAUNCHER[@]}" \
      python -m src.inference \
      experiment=rkmeans_inference_flat_hyp_prefix \
      "data_dir=${DATA_DIR}" \
      "embedding_path=${EMBEDDING_PATH}" \
      "embedding_dim=${EMBEDDING_DIM}" \
      "num_hierarchies=${NUM_HIERARCHIES}" \
      "codebook_width=${CODEBOOK_WIDTH}" \
      "seed=${SEED}" \
      "ckpt_path=${candidate_ckpt_path}" \
      "trainer.devices=1" \
      "callbacks.bq_writer=null" \
      "paths.output_dir=${candidate_inference_dir}" \
      "hydra.run.dir=${candidate_inference_dir}"

    if [[ ! -f "${candidate_semantic_id_dir}/merged_predictions_tensor.pt" ]]; then
      echo "Hyp candidate semantic ids missing: ${candidate_semantic_id_dir}/merged_predictions_tensor.pt" >&2
      exit 1
    fi

    candidate_surrogate_json=""
    if [[ "${ENABLE_DOWNSTREAM_VAL_SURROGATE}" == "1" ]]; then
      candidate_surrogate_dir="${SURROGATE_ROOT}/${candidate_method}"
      candidate_surrogate_json="${candidate_surrogate_dir}/val_surrogate.json"
      reset_path "${candidate_surrogate_dir}"
      run_command \
        "${candidate_method}-surrogate-train" \
        "${candidate_surrogate_dir}/stdout.log" \
        "${candidate_surrogate_dir}/stderr.log" \
        "${SURROGATE_DISTRIBUTED_LAUNCHER[@]}" \
        python -m src.train \
        experiment=tiger_train_flat \
        "data_dir=${DATA_DIR}" \
        "semantic_id_path=${candidate_semantic_id_dir}/merged_predictions_tensor.pt" \
        "num_hierarchies=${TIGER_NUM_HIERARCHIES}" \
        "seed=${SEED}" \
        "trainer.devices=1" \
        "trainer.strategy=auto" \
        "trainer.sync_batchnorm=false" \
        "trainer.max_steps=${SURROGATE_MAX_STEPS}" \
        "trainer.val_check_interval=${SURROGATE_VAL_CHECK_INTERVAL}" \
        "callbacks.model_checkpoint.monitor=${SURROGATE_PRIMARY_METRIC}" \
        "callbacks.early_stopping.monitor=${SURROGATE_PRIMARY_METRIC}" \
        "callbacks.early_stopping.patience=${SURROGATE_EARLY_STOPPING_PATIENCE}" \
        "callbacks.restart_job=null" \
        "paths.output_dir=${candidate_surrogate_dir}" \
        "hydra.run.dir=${candidate_surrogate_dir}"

      run_command \
        "${candidate_method}-surrogate-extract" \
        "${candidate_surrogate_dir}/extract_stdout.log" \
        "${candidate_surrogate_dir}/extract_stderr.log" \
        python scripts/extract_downstream_val_surrogate.py \
        --train-dir "${candidate_surrogate_dir}" \
        --output-json "${candidate_surrogate_json}" \
        --primary-metric "${SURROGATE_PRIMARY_METRIC}" \
        --metric "${SURROGATE_SECONDARY_METRIC}"
    fi

    if [[ -n "${candidate_surrogate_json}" ]]; then
      CANDIDATE_SPECS+=("${weight}|${candidate_method}|${candidate_train_dir}|${candidate_inference_dir}|${SWEEP_PROXY_DIR}/${candidate_method}.json|${candidate_surrogate_json}")
    else
      CANDIDATE_SPECS+=("${weight}|${candidate_method}|${candidate_train_dir}|${candidate_inference_dir}|${SWEEP_PROXY_DIR}/${candidate_method}.json")
    fi
    PROXY_RUN_SPECS+=("${candidate_method}=${candidate_semantic_id_dir}")
  done

  reset_path "${SWEEP_PROXY_DIR}"
  proxy_cmd=(
    python scripts/analyze_semantic_ids.py
    --output-dir "${SWEEP_PROXY_DIR}"
    --embedding-path "${EMBEDDING_PATH}"
    --codebook-size "${CODEBOOK_WIDTH}"
    --num-hierarchies "${NUM_HIERARCHIES}"
    --top-k "${TOP_K}"
  )
  for proxy_run_spec in "${PROXY_RUN_SPECS[@]}"; do
    proxy_cmd+=(--run "${proxy_run_spec}")
  done
  run_command \
    "hyp-sweep-proxy" \
    "${SWEEP_ROOT}/proxy_stdout.log" \
    "${SWEEP_ROOT}/proxy_stderr.log" \
    "${proxy_cmd[@]}"

  SELECTION_JSON="${SWEEP_ROOT}/selection.json"
  SELECTION_MD="${SWEEP_ROOT}/selection.md"
  select_cmd=(
    python scripts/select_semantic_id_weight.py
    --method-family hyp_prefix
    --baseline-train-dir "${BASE_TRAIN_DIR}"
    --baseline-proxy-json "${SWEEP_PROXY_DIR}/base.json"
    --output-json "${SELECTION_JSON}"
    --output-md "${SELECTION_MD}"
    --collision-worse-threshold "${COLLISION_WORSE_THRESHOLD}"
    --frac-unique-min-improvement "${FRAC_UNIQUE_MIN_IMPROVEMENT}"
    --entropy-avg-drop-threshold "${ENTROPY_AVG_DROP_THRESHOLD}"
    --entropy-layer-drop-threshold "${ENTROPY_LAYER_DROP_THRESHOLD}"
    --quantization-loss-relative-threshold "${QUANTIZATION_LOSS_RELATIVE_THRESHOLD}"
    --quantization-loss-absolute-threshold "${QUANTIZATION_LOSS_ABSOLUTE_THRESHOLD}"
    --loss-explosion-threshold "${LOSS_EXPLOSION_THRESHOLD}"
    --prefix-worse-threshold "${PREFIX_WORSE_THRESHOLD}"
    --surrogate-primary-metric "${SURROGATE_PRIMARY_METRIC}"
    --surrogate-secondary-metric "${SURROGATE_SECONDARY_METRIC}"
  )
  for candidate_spec in "${CANDIDATE_SPECS[@]}"; do
    select_cmd+=(--candidate "${candidate_spec}")
  done
  run_command \
    "hyp-sweep-select" \
    "${SWEEP_ROOT}/selection_stdout.log" \
    "${SWEEP_ROOT}/selection_stderr.log" \
    "${select_cmd[@]}"

  selected_weight="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_weight_text"])' "${SELECTION_JSON}")"
  selected_train_dir="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_train_dir"])' "${SELECTION_JSON}")"
  selected_inference_dir="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selected_inference_dir"])' "${SELECTION_JSON}")"
  selection_mode="$(python -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["selection_mode"])' "${SELECTION_JSON}")"

  copy_dir_contents "${selected_train_dir}" "${HYP_TRAIN_DIR}"
  copy_dir_contents "${selected_inference_dir}" "${HYP_INFERENCE_DIR}"
  cp "${SELECTION_JSON}" "${HYP_TRAIN_DIR}/weight_selection.json"
  cp "${SELECTION_MD}" "${HYP_TRAIN_DIR}/weight_selection.md"
  printf '%s\n' "${selected_weight}" > "${HYP_TRAIN_DIR}/selected_weight.txt"
  printf '%s\n' "${selection_mode}" > "${HYP_TRAIN_DIR}/selection_mode.txt"

  if [[ ! -f "${HYP_CKPT_PATH}" ]]; then
    echo "Final hyp checkpoint missing: ${HYP_CKPT_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${HYP_SEMANTIC_ID_PATH}" ]]; then
    echo "Final hyp semantic ids missing: ${HYP_SEMANTIC_ID_PATH}" >&2
    exit 1
  fi

  log "## Hyp Selection"
  log "- selected_weight: \`${selected_weight}\`"
  log "- selection_mode: \`${selection_mode}\`"
  log "- promoted_train_dir: \`${HYP_TRAIN_DIR}\`"
  log "- promoted_inference_dir: \`${HYP_INFERENCE_DIR}\`"
  log ""

  reset_path "${PROXY_OUTPUT_DIR}"
  run_command \
    "final-proxy" \
    "${PROXY_OUTPUT_DIR}/stdout.log" \
    "${PROXY_OUTPUT_DIR}/stderr.log" \
    python scripts/analyze_semantic_ids.py \
    --run "base=${BASE_INFERENCE_DIR}/pickle" \
    --run "hyp_prefix=${HYP_INFERENCE_DIR}/pickle" \
    --output-dir "${PROXY_OUTPUT_DIR}" \
    --embedding-path "${EMBEDDING_PATH}" \
    --codebook-size "${CODEBOOK_WIDTH}" \
    --num-hierarchies "${NUM_HIERARCHIES}" \
    --top-k "${TOP_K}"

  reset_path "${COMPARISON_CSV}"
  reset_path "${COMPARISON_JSON}"
  reset_path "${COMPARISON_MD}"
  run_command \
    "semantic-stage-comparison" \
    "${SEMANTIC_ID_STAGE_ROOT}/comparison_stdout.log" \
    "${SEMANTIC_ID_STAGE_ROOT}/comparison_stderr.log" \
    python scripts/build_semantic_id_stage_comparison.py \
    --train-run "base=${BASE_TRAIN_DIR}" \
    --train-run "hyp_prefix=${HYP_TRAIN_DIR}" \
    --inference-run "base=${BASE_INFERENCE_DIR}" \
    --inference-run "hyp_prefix=${HYP_INFERENCE_DIR}" \
    --proxy-dir "${PROXY_OUTPUT_DIR}" \
    --output-csv "${COMPARISON_CSV}" \
    --output-json "${COMPARISON_JSON}" \
    --output-md "${COMPARISON_MD}"
else
  for required_file in \
    "${EMBEDDING_PATH}" \
    "${BASE_CKPT_PATH}" \
    "${BASE_SEMANTIC_ID_PATH}" \
    "${HYP_CKPT_PATH}" \
    "${HYP_SEMANTIC_ID_PATH}" \
    "${COMPARISON_CSV}"; do
    if [[ ! -f "${required_file}" ]]; then
      echo "Downstream-only mode requires existing semantic output: ${required_file}" >&2
      exit 1
    fi
  done
fi

if [[ "${RUN_DOWNSTREAM_STAGE}" == "1" ]]; then
  reset_path "${BASE_TIGER_DIR}"
  run_command \
    "tiger-base-train" \
    "${BASE_TIGER_DIR}/stdout.log" \
    "${BASE_TIGER_DIR}/stderr.log" \
    "${TRAIN_DISTRIBUTED_LAUNCHER[@]}" \
    python -m src.train \
    experiment=tiger_train_flat \
    "data_dir=${DATA_DIR}" \
    "semantic_id_path=${BASE_SEMANTIC_ID_PATH}" \
    "num_hierarchies=${TIGER_NUM_HIERARCHIES}" \
    "seed=${SEED}" \
    "paths.output_dir=${BASE_TIGER_DIR}" \
    "hydra.run.dir=${BASE_TIGER_DIR}"
  log_metrics_file "Base TIGER Train" "$(latest_metrics_file_for_run "${BASE_TIGER_DIR}")"

  reset_path "${HYP_TIGER_DIR}"
  run_command \
    "tiger-hyp-train" \
    "${HYP_TIGER_DIR}/stdout.log" \
    "${HYP_TIGER_DIR}/stderr.log" \
    "${TRAIN_DISTRIBUTED_LAUNCHER[@]}" \
    python -m src.train \
    experiment=tiger_train_flat \
    "data_dir=${DATA_DIR}" \
    "semantic_id_path=${HYP_SEMANTIC_ID_PATH}" \
    "num_hierarchies=${TIGER_NUM_HIERARCHIES}" \
    "seed=${SEED}" \
    "paths.output_dir=${HYP_TIGER_DIR}" \
    "hydra.run.dir=${HYP_TIGER_DIR}"
  log_metrics_file "Hyp TIGER Train" "$(latest_metrics_file_for_run "${HYP_TIGER_DIR}")"
fi

log "## Outputs"
log "- semantic embeddings: \`${SEMANTIC_EMBEDDING_OUTPUT_DIR}\`"
log "- semantic stage root: \`${SEMANTIC_ID_STAGE_ROOT}\`"
log "- recommendation stage root: \`${RECOMMENDATION_STAGE_ROOT}\`"
log "- proxy output: \`${PROXY_OUTPUT_DIR}\`"
log "- comparison markdown: \`${COMPARISON_MD}\`"
log ""

echo "completed run: ${RUN_NAME}"
echo "summary: ${SUMMARY_FILE}"
