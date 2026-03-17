# Semantic ID Proxy Metrics Task

请在 GRID 仓库中为 semantic ID / tokenizer 阶段实现一组代理指标（proxy metrics），用于比较以下三个版本：

1. 原始 rkmeans_train_flat
2. rkmeans + EuclideanPrefixLoss
3. rkmeans + HyperbolicPrefixContrastiveLoss

目标不是改推荐模型，而是分析 semantic ID 本身有没有变得更好、更健康、更有层级结构。

一、要实现的代理指标
请至少实现以下 4 类指标：

1. codebook utilization
定义建议：
- 对每一层 hierarchy，统计实际被使用到的 code 数量 / 总 codebook 大小
- 同时给出 overall utilization 和每层 utilization

输出建议：
- utilization_per_layer
- avg_utilization

2. collision rate
定义建议：
- 一个完整 semantic ID（例如 [c1, c2, c3]）如果对应多个不同 item，则视为 collision
- 统计：
  - 总 collision item 数
  - collision rate
  - 唯一 semantic ID 比例（frac_unique_ids）

输出建议：
- num_items
- num_unique_ids
- num_colliding_items
- collision_rate
- frac_unique_ids

3. prefix purity / hierarchy consistency
请优先做两个版本：

3.1 如果数据中存在 item 的 category / taxonomy / path 信息：
- 计算 prefix-level purity
- 对于每一层 prefix（例如前 1 层、前 2 层、前 3 层），统计共享该 prefix 的 item 集合在 category 上的纯度
- 可以用 majority-label purity
- 输出每一层的 purity

3.2 如果没有可靠 category path：
- 实现一个无监督替代指标
- 例如：
  - prefix cluster 内部 embedding 方差
  - prefix cluster 内部平均 pairwise 距离
  - deeper prefix 是否带来更小的 cluster spread
- 输出每一层 prefix 的平均 cluster compactness

4. per-layer entropy / layer-wise distribution health
定义建议：
- 对每一层 hierarchy，统计 token 分布
- 计算每层 entropy
- 计算是否有严重塌缩（例如某些 code 占比过高）

输出建议：
- entropy_per_layer
- max_token_ratio_per_layer
- topk_token_coverage_per_layer

二、实现位置建议
请优先检查这些位置：
- src/modules/clustering/residual_quantization.py
- src/components/metrics.py（如果没有可新增）
- src/utils/ 或新增一个分析脚本目录
- rkmeans_inference_flat 输出的 semantic ID 结果

你可以新增：
- src/components/semantic_id_metrics.py
- scripts/analyze_semantic_ids.py
- 或一个 inference 后分析模块

三、输入来源
请优先使用 semantic ID inference 阶段输出的结果进行分析，而不是训练阶段中间变量。
也就是说，分析对象应该是：
- item -> semantic ID 的最终映射结果
- 如有需要，再结合 item embedding 或 metadata

四、输出要求
请至少支持以下两种输出：

1. 机器可读格式
- json 或 csv
- 方便后续汇总成表格

2. 人类可读摘要
- 直接打印一个 summary
- 包含：
  - codebook utilization
  - collision rate
  - 每层 entropy
  - prefix purity 或 compactness

五、实验对照要求
请确保这套分析工具可以对以下 3 个实验输出统一格式结果：
1. 原始 rkmeans_train_flat
2. rkmeans + EuclideanPrefixLoss
3. rkmeans + HyperbolicPrefixContrastiveLoss

最终希望得到一张对照表，至少包含：
- 方法名
- avg_utilization
- collision_rate
- frac_unique_ids
- entropy_per_layer
- prefix_purity 或 prefix_compactness

六、注意事项
1. 不要修改 TIGER
2. 不要把 proxy metric 和推荐指标混在一起
3. 当前任务只关注 semantic ID 阶段
4. 如果 category path 在当前数据里不可用，请明确说明，并自动回退到无监督版本
5. 如果已有类似指标在仓库里实现了，请复用；如果没有，再新增最小实现

七、交付内容
1. 新增/修改的代码文件列表
2. 如何运行这套分析的命令
3. 统一格式的输出样例
4. 说明每个指标的定义
5. 如果某些指标无法从当前数据直接获得，请清楚写明原因和 fallback 方案

八、当前实现

1. 新增/修改的代码文件列表
- 新增 `src/components/semantic_id_metrics.py`
- 新增 `scripts/analyze_semantic_ids.py`
- 新增 `scripts/run_semantic_id_proxy_metrics.sh`
- 新增 `scripts/build_semantic_id_stage_comparison.py`
- 修改 `scripts/run_semantic_id_prefix_experiments.sh`
- 修改 `my_job.sh`

2. 设计说明
- 当前实现不修改 TIGER，不修改推荐阶段指标。
- 当前实现优先分析 semantic ID inference 阶段输出：
  - 优先读取 `merged_predictions.pkl`
  - 如果只有 `merged_predictions_tensor.pt`，则回退读取 `.pt`
- 当前实现支持三组实验统一对比：
  - `base`
  - `euc_prefix`
  - `hyp_prefix`
- 当前实现输出：
  - 每个方法一个 json
  - 一个汇总 `comparison.json`
  - 一个汇总 `comparison.csv`
  - 一个可读摘要 `SUMMARY.md`
- 当前实现还会把训练阶段最终损失和 proxy metrics 合并成：
  - `semantic_id_stage_comparison.csv`
  - `semantic_id_stage_comparison.json`
  - `semantic_id_stage_comparison.md`

3. 指标定义

3.1 codebook utilization
- `used_codes_per_layer`：
  - 每一层实际出现过的 code 数量
- `utilization_per_layer`：
  - `used_codes_per_layer[layer] / codebook_size`
- `overall_utilization`：
  - 所有层实际使用 code 总数 / 所有层总容量
- `avg_utilization`：
  - `utilization_per_layer` 的平均值

3.2 collision rate
- 完整 semantic ID 完全相同时，视为 collision
- `num_items`：
  - item 总数
- `num_unique_ids`：
  - 唯一完整 semantic ID 的数量
- `num_colliding_items`：
  - 落在 collision 组里的 item 数量
- `num_collision_groups`：
  - collision group 的数量
- `collision_rate`：
  - `num_colliding_items / num_items`
- `frac_unique_ids`：
  - `num_unique_ids / num_items`

3.3 prefix purity / hierarchy consistency
- 如果提供 category / taxonomy / path metadata：
  - 计算 `prefix_purity_per_layer`
  - 对每个 prefix 深度，用 majority-label purity 做加权平均
- 如果没有可靠 category metadata：
  - 自动回退到无监督版本
  - 计算 `prefix_compactness_per_layer`
  - 做法是对相同 prefix 的 item embedding 计算簇内 centroid spread
- 当前实现还会输出：
  - `prefix_metric_type`
  - `labeled_fraction_per_layer` 或 `non_singleton_fraction_per_layer`

3.4 per-layer entropy / distribution health
- `entropy_per_layer`
  - 每层 token 分布熵
- `max_token_ratio_per_layer`
  - 每层最大 token 占比
- `topk_token_coverage_per_layer`
  - 每层 top-k token 的累计覆盖率

4. 运行命令

4.1 直接运行 Python 分析脚本
```bash
python scripts/analyze_semantic_ids.py \
  --run base=/path/to/base/pickle \
  --run euc_prefix=/path/to/euc_prefix/pickle \
  --run hyp_prefix=/path/to/hyp_prefix/pickle \
  --embedding-path /path/to/merged_embeddings.pt \
  --codebook-size 256 \
  --num-hierarchies 3 \
  --output-dir /path/to/proxy_metrics_output
```

4.2 使用 shell 包装脚本
```bash
BASE_SEMANTIC_ID_PATH=/path/to/base/pickle \
EUC_PREFIX_SEMANTIC_ID_PATH=/path/to/euc_prefix/pickle \
HYP_PREFIX_SEMANTIC_ID_PATH=/path/to/hyp_prefix/pickle \
EMBEDDING_PATH=/path/to/merged_embeddings.pt \
CODEBOOK_SIZE=256 \
NUM_HIERARCHIES=3 \
bash scripts/run_semantic_id_proxy_metrics.sh
```

- 不显式传 `OUTPUT_DIR` 时，当前默认输出到：
  - `outputs/semantic_id_proxy_metrics`

4.3 在 `my_job.sh` 中自动串联运行
- 当前 `my_job.sh` 会直接调用：
  - `scripts/run_semantic_id_prefix_experiments.sh`
- 当前调度层支持：
  - `RUN_MODE=base_only`
  - `RUN_MODE=all_three`
  - `RUN_MODE=analyze_only`
- 默认：
  - `RUN_MODE=base_only`
  - `RUN_PROXY_METRICS=0`
- 默认直接运行：
```bash
sbatch my_job.sh
```
- 这时只会运行 baseline semantic ID train + inference，不会跑 proxy metrics。
- 如需三组一起跑并自动分析：
```bash
RUN_MODE=all_three RUN_PROXY_METRICS=1 sbatch my_job.sh
```
- 如需只对已有三组结果做分析：
```bash
RUN_MODE=analyze_only RUN_PROXY_METRICS=1 RUN_ROOT=/path/to/existing/run_root sbatch my_job.sh
```
- 只有在你想单独脱离总流程跑 proxy 分析脚本时，才需要手动设置：
  - `BASE_SEMANTIC_ID_PATH`
  - `EUC_PREFIX_SEMANTIC_ID_PATH`
  - `HYP_PREFIX_SEMANTIC_ID_PATH`

5. 输出样例

5.1 `comparison.csv` 中的典型字段
- `method`
- `avg_utilization`
- `overall_utilization`
- `collision_rate`
- `frac_unique_ids`
- `utilization_per_layer`
- `entropy_per_layer`
- `max_token_ratio_per_layer`
- `topk_token_coverage_per_layer`
- `prefix_metric_type`
- `prefix_purity_per_layer` 或 `prefix_compactness_per_layer`
- `notes`

5.2 `semantic_id_stage_comparison.csv` 中的典型字段
- `method`
- `final_quantization_loss`
- `final_reconstruction_loss`
- `final_hierarchy_loss`
- `avg_utilization`
- `collision_rate`
- `frac_unique_ids`
- `entropy_layer_1`
- `entropy_layer_2`
- `entropy_layer_3`
- `prefix_metric_type`
- `prefix_metric_layer_1`
- `prefix_metric_layer_2`
- `prefix_metric_layer_3`

5.3 `SUMMARY.md` 样例结构
```text
# Semantic ID Proxy Metrics Summary

## base
Method: base
  num_items: 12345
  avg_utilization: 0.812500
  collision_rate: 0.031200
  frac_unique_ids: 0.968800
  utilization_per_layer: [0.91, 0.80, 0.73]
  entropy_per_layer: [5.12, 4.76, 4.31]
  max_token_ratio_per_layer: [0.08, 0.12, 0.18]
  top5_token_coverage_per_layer: [0.19, 0.27, 0.35]
  prefix_compactness_per_layer: [1.82, 1.11, 0.74]
```

5.4 `semantic_id_stage_comparison.csv` 样例理解
- 这个文件会把：
  - 训练阶段 `metrics.csv` 中提取的最终损失
  - semantic ID inference 上算出的 proxy metrics
  合并成一张对照表
- 目的不是替代 `comparison.csv`
- 目的是真正给“要不要继续接 TIGER”提供一个一眼可读的筛选表

6. category path 不可用时的 fallback
- 当前实现会优先尝试 category metadata：
  - `--metadata-csv`
  - `--metadata-json`
- 如果没有提供 metadata，或者 metadata 中没有可靠 category 字段：
  - 自动回退到 embedding-based compactness
- 如果 metadata 和 embedding tensor 都没有提供：
  - 仍然会输出 utilization / collision / entropy
  - prefix purity / compactness 会标记为 unavailable
  - 原因会写入 `notes`

7. 注意事项
- 为了得到有意义的 collision rate，当前实现优先读 `merged_predictions.pkl`。
- 如果只提供 `merged_predictions_tensor.pt`，分析脚本仍可运行，但可能已经包含 dedup 后缀或转置结果。
- 如有 dedup 后缀列，可通过 `--num-hierarchies` 限制只分析前几层真正的 hierarchy。
- `my_job.sh` 当前会默认把 semantic ID stage 先跑完，再给出最终对照表，推荐先看 `semantic_id_stage_comparison.csv` 再决定是否继续接 TIGER。
- 当前默认行为已经调整为先只跑 baseline semantic ID，便于第一轮排障。
