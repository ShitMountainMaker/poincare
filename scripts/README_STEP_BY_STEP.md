# poincare Step-by-Step Runbook

这份文档只写固定路径、固定命令。

固定路径已经在 [my_job.sh]里配好了。
下面的命令默认直接复用 `my_job.sh` 里已经 `export` 的变量，适合你直接复制到 `my_job.sh` 里。

这份 runbook 不依赖额外的包装脚本，下面都是直接命令。

说明：
- `run_semantic_id_prefix_experiments.sh` 和 `run_semantic_id_proxy_metrics.sh` 现在会覆写各自目标输出目录。
- 直接手写 `python -m ...` 命令时，不会自动清目录；如果你复用固定输出路径，建议先手动清掉旧目录。

## 0. 默认固定路径

这里默认用的是：

```text
/data/user/cwu319/RC/poincare
/data/user/cwu319/RC/poincare/data/amazon_data
/data/user/cwu319/RC/poincare/data/amazon_data/beauty
outputs/semantic_embeddings
outputs/semantic_id_stage
outputs/recommendation_stage/tiger_train
outputs/recommendation_stage/tiger_inference
```

## 1. 检查数据

确认这些目录存在：

```bash
cd "${REPO_DIR}"
ls "${DATA_ITEMS_DIR}"
ls "${DATA_SEQUENCE_TRAIN_DIR}"
ls "${DATA_SEQUENCE_EVAL_DIR}"
ls "${DATA_SEQUENCE_TEST_DIR}"
```

## 2. 生成 item embeddings

运行：

```bash
cd "${REPO_DIR}"
python -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir="${DATA_DIR}" \
  paths.output_dir="${SEMANTIC_EMBEDDING_OUTPUT_DIR}" \
  hydra.run.dir="${SEMANTIC_EMBEDDING_OUTPUT_DIR}"
```

输出：

```text
outputs/semantic_embeddings/pickle/merged_predictions_tensor.pt
```

## 3. 学 semantic IDs

### 3.1 baseline

推荐写法：

```bash
cd "${REPO_DIR}"
RUN_MODE=base_only RUN_PROXY_METRICS=0 bash "${REPO_DIR}/scripts/run_semantic_id_prefix_experiments.sh"
```

如果你想拆开写成原始命令，就是这两条：

```bash
cd "${REPO_DIR}"
python -m src.train \
  experiment=rkmeans_train_flat \
  data_dir="${DATA_DIR}" \
  embedding_path="${EMBEDDING_PATH}" \
  embedding_dim="${EMBEDDING_DIM}" \
  num_hierarchies="${NUM_HIERARCHIES}" \
  codebook_width="${CODEBOOK_WIDTH}" \
  paths.output_dir="${BASE_SEMANTIC_ID_TRAIN_DIR}" \
  hydra.run.dir="${BASE_SEMANTIC_ID_TRAIN_DIR}"
```

```bash
cd "${REPO_DIR}"
python -m src.inference \
  experiment=rkmeans_inference_flat \
  data_dir="${DATA_DIR}" \
  embedding_path="${EMBEDDING_PATH}" \
  embedding_dim="${EMBEDDING_DIM}" \
  num_hierarchies="${NUM_HIERARCHIES}" \
  codebook_width="${CODEBOOK_WIDTH}" \
  ckpt_path="${BASE_SEMANTIC_ID_CHECKPOINT_PATH}" \
  callbacks.bq_writer=null \
  paths.output_dir="${BASE_SEMANTIC_ID_INFERENCE_DIR}" \
  hydra.run.dir="${BASE_SEMANTIC_ID_INFERENCE_DIR}"
```

输出：

```text
outputs/semantic_id_stage/base
outputs/semantic_id_stage/base/checkpoints/last.ckpt
outputs/semantic_id_stage/inference/base/pickle/merged_predictions_tensor.pt
```

### 3.2 三组 semantic ID

如果你要一次跑三组：

```bash
cd "${REPO_DIR}"
RUN_MODE=all_three RUN_PROXY_METRICS=0 bash "${REPO_DIR}/scripts/run_semantic_id_prefix_experiments.sh"
```

这条脚本会自动选择三组对应的 train / inference experiment：

```text
base -> 直接跑 rkmeans_train_flat + rkmeans_inference_flat
euc_prefix -> 在 0.05 / 0.1 / 0.2 上 sweep，再保留一个权重
hyp_prefix -> 在 0.05 / 0.1 / 0.2 上 sweep，再保留一个权重
```

输出：

```text
outputs/semantic_id_stage/base
outputs/semantic_id_stage/euc_prefix
outputs/semantic_id_stage/hyp_prefix
outputs/semantic_id_stage/inference/base
outputs/semantic_id_stage/inference/euc_prefix
outputs/semantic_id_stage/inference/hyp_prefix
outputs/semantic_id_stage/sweeps/euc_prefix
outputs/semantic_id_stage/sweeps/hyp_prefix
outputs/semantic_id_stage/SUMMARY.md
outputs/semantic_id_stage/semantic_id_stage_comparison.csv
outputs/semantic_id_stage/semantic_id_stage_comparison.md
```

说明：
- `base` 不做权重 sweep，保持原始 baseline。
- `euc_prefix` 和 `hyp_prefix` 会先跑 `PREFIX_WEIGHT_SWEEP=0.05,0.1,0.2`，再根据 semantic-ID 阶段的 proxy 指标筛掉不合格权重。
- 筛选标准和保留原因会写进 `outputs/semantic_id_stage/SUMMARY.md`。
- 最终保留的权重会被提升到固定目录：
  - `outputs/semantic_id_stage/euc_prefix`
  - `outputs/semantic_id_stage/hyp_prefix`
- `selected_weight.txt` 和 `selection_mode.txt` 会写在最终保留的训练目录里。
- 如果你想改 sweep 权重，可以在命令前覆盖：`PREFIX_WEIGHT_SWEEP=0.05,0.1,0.2`

## 4. 训练 TIGER

运行：

```bash
cd "${REPO_DIR}"
python -m src.train \
  experiment=tiger_train_flat \
  data_dir="${DATA_DIR}" \
  semantic_id_path="${TIGER_TRAIN_SEMANTIC_ID_PATH}" \
  num_hierarchies="${TIGER_NUM_HIERARCHIES}" \
  paths.output_dir="${TIGER_TRAIN_DIR}" \
  hydra.run.dir="${TIGER_TRAIN_DIR}"
```

输出：

```text
outputs/recommendation_stage/tiger_train
outputs/recommendation_stage/tiger_train/checkpoints/best.ckpt
```

## 5. 跑 TIGER inference

运行：

```bash
cd "${REPO_DIR}"
python -m src.inference \
  experiment=tiger_inference_flat \
  data_dir="${DATA_DIR}" \
  semantic_id_path="${TIGER_INFERENCE_SEMANTIC_ID_PATH}" \
  ckpt_path="${TIGER_INFERENCE_CHECKPOINT_PATH}" \
  num_hierarchies="${TIGER_NUM_HIERARCHIES}" \
  paths.output_dir="${TIGER_INFERENCE_DIR}" \
  hydra.run.dir="${TIGER_INFERENCE_DIR}"
```

输出：

```text
outputs/recommendation_stage/tiger_inference
```

## Proxy Metrics

这个分析不属于 README 原始 1-5 步主线。

如果你跑的是 `RUN_MODE=all_three`，脚本已经会在内部完成：
- `euc_prefix` / `hyp_prefix` 的权重筛选
- 最终保留版本的 proxy metrics
- 最后的 stage comparison

只有当你想对“已经存在的最终保留结果”单独重建分析时，才需要用 `analyze_only`。

先跑三组 semantic ID 并自动完成筛选：

```bash
cd "${REPO_DIR}"
RUN_MODE=all_three RUN_PROXY_METRICS=0 bash "${REPO_DIR}/scripts/run_semantic_id_prefix_experiments.sh"
```

如果之后只想重建最终分析：

```bash
cd "${REPO_DIR}"
RUN_MODE=analyze_only RUN_PROXY_METRICS=1 bash "${REPO_DIR}/scripts/run_semantic_id_prefix_experiments.sh"
```

如果你要单独直接调 proxy metrics 脚本：

```bash
cd "${REPO_DIR}"
BASE_SEMANTIC_ID_PATH="${BASE_SEMANTIC_ID_PICKLE_DIR}" \
EUC_PREFIX_SEMANTIC_ID_PATH="${EUC_PREFIX_SEMANTIC_ID_PICKLE_DIR}" \
HYP_PREFIX_SEMANTIC_ID_PATH="${HYP_PREFIX_SEMANTIC_ID_PICKLE_DIR}" \
OUTPUT_DIR="${SEMANTIC_ID_STAGE_ROOT}/proxy_metrics" \
EMBEDDING_PATH="${EMBEDDING_PATH}" \
CODEBOOK_SIZE="${CODEBOOK_WIDTH}" \
NUM_HIERARCHIES="${NUM_HIERARCHIES}" \
TOP_K="${TOP_K}" \
METADATA_CSV="${METADATA_CSV}" \
METADATA_JSON="${METADATA_JSON}" \
CATEGORY_FIELD="${CATEGORY_FIELD}" \
bash "${REPO_DIR}/scripts/run_semantic_id_proxy_metrics.sh"
```

输出：

```text
outputs/semantic_id_stage/proxy_metrics
outputs/semantic_id_stage/semantic_id_stage_comparison.csv
```

建议顺序：

1. 先跑第 2 步 embedding
2. 再跑第 3 步 baseline semantic ID
3. baseline 没问题后，再跑第 3 步三组 semantic ID
4. 再做 proxy metrics
5. 只有 proxy 指标看起来有信号，再继续第 4/5 步 TIGER
