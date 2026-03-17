# GRID Semantic ID Learning Prefix-Aware Regularizer Prototype

请在 GRID 仓库中实现一个“最小可运行”的层级感知正则原型，用于 semantic ID 学习阶段。

一、目标
在 semantic ID learning 阶段（先以 rkmeans_train_flat 路径为主）加入 prefix-aware regularizer，并支持以下三个实验版本：

1. 原始 rkmeans_train_flat（不改）
2. rkmeans + EuclideanPrefixLoss
3. rkmeans + HyperbolicPrefixContrastiveLoss

注意：当前阶段不要修改 TIGER 生成模型、不要修改 CrossEntropyLoss、不要替换推荐阶段损失。只改 semantic ID 学习阶段。

二、改动范围
优先检查并修改这些文件：
- src/modules/clustering/residual_quantization.py
- src/modules/clustering/vector_quantization.py
- src/components/loss_functions.py
- src/components/distance_functions.py
- configs/experiment/rkmeans_train_flat.yaml

可以新增文件：
- src/components/prefix_losses.py
- src/components/hyperbolic_utils.py
- configs/experiment/rkmeans_train_flat_euc_prefix.yaml
- configs/experiment/rkmeans_train_flat_hyp_prefix.yaml

三、实现要求
1. 实现 EuclideanPrefixLoss
输入：
- embeddings: [B, D]
- cluster_ids: [B, L]

功能：
- 根据 cluster_ids 计算任意两个样本的最长公共前缀长度（Longest Common Prefix, LCP）
- 构造一个 prefix-aware regularizer
- 共享前缀越深的样本，应该被拉得更近
- 完全没有共享前缀的样本，应该被推得更远
- 第一版不要求很复杂，先做一个稳定、易解释的版本即可

建议：
- 先实现 pairwise LCP matrix
- 先用简单的加权距离损失，不要一开始就做太复杂的 margin/temperature 设计

2. 实现 HyperbolicPrefixContrastiveLoss
输入同上：
- embeddings: [B, D]
- cluster_ids: [B, L]

要求：
- 先加一个小 projector，把输入 embedding 从原始维度映射到更低维（建议 64）
- 在低维空间上做双曲几何 regularization
- 采用 Poincaré ball 实现
- 必须做数值保护：project/clipping，保证向量范数严格小于 1（例如 0.99）
- 需要实现稳定的双曲距离函数
- 第一版不要引入 Möbius 加法等复杂操作，尽量最小实现

3. 把 regularizer 接入 semantic ID 训练
在 residual quantization 训练阶段加入：
- hierarchy_regularizer
- hierarchy_loss_weight
- hierarchy_start_step

要求：
- 保留原来的 quantization loss / reconstruction loss，不要替换
- 新增 hierarchy loss 只是辅助项
- 当 global_step < hierarchy_start_step 时，不启用 hierarchy loss
- 训练时 log hierarchy loss

四、实现建议
1. 不要修改 TIGER 相关文件
2. 不要重构大量代码
3. 尽量通过“新增文件 + 小范围接线”的方式完成
4. 如果当前代码结构不方便，优先在 residual_quantization.py 中集中接入 hierarchy loss
5. 如果需要，cluster_ids 可以 detach，避免离散 assignment 上的无意义梯度传播

五、需要你交付的内容
1. 完整代码修改
2. 改动文件列表
3. 新增配置文件列表
4. 三组实验的运行命令：
   - 原始 rkmeans_train_flat
   - rkmeans + EuclideanPrefixLoss
   - rkmeans + HyperbolicPrefixContrastiveLoss
5. 一段简短说明：
   - 你把 hierarchy loss 接在了哪里
   - Euclidean 和 Hyperbolic 两个版本各自怎么实现
   - 可能的数值稳定性风险有哪些

六、非常重要的约束
- 当前只做 semantic ID learning prototype
- 不要碰下游生成推荐损失
- 不要把事情做大
- 优先保证“能跑通、可回退、便于对照实验”

七、HPC 运行方式
- 实验编排脚本位于：
  - `scripts/run_semantic_id_prefix_experiments.sh`
- `my_job.sh` 现在只负责：
  - Slurm 配置
  - 进入仓库目录
  - 激活 conda 环境
  - 导出运行所需环境变量
  - 调用 `scripts/run_semantic_id_prefix_experiments.sh`
- 当前调度层支持三种运行模式：
  - `RUN_MODE=base_only`
  - `RUN_MODE=all_three`
  - `RUN_MODE=analyze_only`
- 默认：
  - `RUN_MODE=base_only`
  - `RUN_PROXY_METRICS=0`
- 在默认 `base_only` 模式下，单次 `sbatch my_job.sh` 只会运行：
  - `rkmeans_train_flat`
  - 基于该 checkpoint 的 `rkmeans_inference_flat`
- 在 `all_three` 模式下，会顺序运行三组 train + inference：
  - `rkmeans_train_flat`
  - `rkmeans_train_flat_euc_prefix`
  - `rkmeans_train_flat_hyp_prefix`
- 只有当：
  - `RUN_MODE=all_three` 或 `RUN_MODE=analyze_only`
  - 且 `RUN_PROXY_METRICS=1`
  时，才会继续运行：
  - proxy metrics 分析
  - `semantic_id_stage_comparison.csv` 汇总表构建
- 运行前只需要在 `my_job.sh` 顶部确认这些路径默认值是否正确：
  - `REPO_DIR`
  - `DATA_DIR`
  - `EMBEDDING_PATH`
  - `CONDA_ENV_DIR`
- 输出目录固定写到：
  - `outputs/semantic_id_stage/base`
  - `outputs/semantic_id_stage/euc_prefix`
  - `outputs/semantic_id_stage/hyp_prefix`
- inference 输出会写到：
  - `outputs/semantic_id_stage/inference/base`
  - `outputs/semantic_id_stage/inference/euc_prefix`
  - `outputs/semantic_id_stage/inference/hyp_prefix`
- proxy metrics 输出会写到：
  - `outputs/semantic_id_stage/proxy_metrics`
- 最终对照表会写到：
  - `outputs/semantic_id_stage/semantic_id_stage_comparison.csv`
  - `outputs/semantic_id_stage/semantic_id_stage_comparison.json`
  - `outputs/semantic_id_stage/semantic_id_stage_comparison.md`
- 每个实验子目录里都会保存：
  - Hydra 训练输出
  - `csv/version_0/metrics.csv`
  - `checkpoints/`
  - `stdout.log`
  - `stderr.log`
- 单次三实验的汇总说明会写到：
  - `outputs/semantic_id_stage/SUMMARY.md`
- 如果需要保留多次独立运行结果，可以手动传：
  - `RUN_ROOT=/path/to/custom_output_dir`
- Slurm 总输出会写到：
  - `history/output_semantic_id_<jobid>.txt`
- `history/output_semantic_id_<jobid>.txt` 中会打印每组实验的结果摘要，包括：
  - 实验名
  - 输出目录
  - checkpoint 路径
  - stdout/stderr 日志路径
  - 最后一次记录到 `metrics.csv` 的训练指标摘要
  - proxy metrics 摘要
  - 最终 stage comparison markdown 表

八、当前实现

1. 实际改动的代码文件
- 修改 `src/modules/clustering/residual_quantization.py`
- 修改 `configs/experiment/rkmeans_train_flat.yaml`
- 新增 `src/components/prefix_losses.py`
- 新增 `src/components/hyperbolic_utils.py`
- 新增 `configs/experiment/rkmeans_train_flat_euc_prefix.yaml`
- 新增 `configs/experiment/rkmeans_train_flat_hyp_prefix.yaml`
- 新增 `scripts/run_semantic_id_prefix_experiments.sh`
- 新增 `scripts/build_semantic_id_stage_comparison.py`
- 修改 `my_job.sh`

2. 当前实现中没有改动的相关文件
- 没有修改 TIGER 相关文件
- 没有修改推荐阶段 `CrossEntropyLoss`
- 没有修改推荐阶段推理逻辑
- 当前最小原型没有改 `src/modules/clustering/vector_quantization.py`
- 当前最小原型没有改 `src/components/loss_functions.py`
- 当前最小原型没有改 `src/components/distance_functions.py`
- 当前实现把 hierarchy loss 集中接在 `residual_quantization.py`，避免把事情做大

3. hierarchy loss 接入位置
- 当前实现把 `hierarchy_regularizer`
- `hierarchy_loss_weight`
- `hierarchy_start_step`
  加到了 `ResidualQuantization` 模块配置中
- 在训练阶段，原有目标保持不变：
  - `quantization_loss_weight * quantization_loss`
  - `reconstruction_loss_weight * reconstruction_loss`
- 新增 hierarchy regularizer 作为辅助项：
  - `hierarchy_loss_weight * hierarchy_loss`
- 当以下条件同时满足时才启用 hierarchy loss：
  - `hierarchy_regularizer` 不为 `null`
  - `hierarchy_loss_weight > 0`
  - `global_step >= hierarchy_start_step`
  - 最后一层 quantizer 已初始化
- 训练时会记录：
  - `train/hierarchy_loss`
- 当前实现对 `cluster_ids` 使用 `detach()` 后再送入 hierarchy loss，避免在离散 assignment 上引入无意义梯度

4. EuclideanPrefixLoss 当前实现
- 输入：
  - `embeddings: [B, D]`
  - `cluster_ids: [B, L]`
- 当前实现先根据 `cluster_ids` 计算 pairwise longest common prefix matrix
- 然后计算 batch 内两两样本的 squared Euclidean distance
- 对 embedding 先做可选归一化
- 用一个简单、稳定、可解释的目标距离：
  - 共享前缀越深，目标距离越小
  - 完全没有共享前缀，目标距离越大
- 最终对真实 pairwise distance 和目标 pairwise distance 做 MSE
- 当前配置文件：
  - `configs/experiment/rkmeans_train_flat_euc_prefix.yaml`

5. HyperbolicPrefixContrastiveLoss 当前实现
- 输入：
  - `embeddings: [B, D]`
  - `cluster_ids: [B, L]`
- 当前实现先用一个小 projector 把 embedding 投到低维空间：
  - 默认 `projector_dim=64`
- 然后将低维向量投影到 Poincare ball 内
- 为了数值稳定，当前实现做了：
  - ball projection
  - `max_ball_norm=0.99`
  - `eps=1e-6`
  - 距离计算前对范数和分母做 clipping / clamp
- 当前实现使用稳定的 pairwise Poincare distance
- 损失构成是一个最小可运行版的 contrastive regularizer：
  - 共享前缀越深的 pair，距离项权重越高
  - 无共享前缀的 pair，使用 `negative_margin` 做推远
- 当前没有引入 Mobius 加法等更复杂的双曲操作
- 当前配置文件：
  - `configs/experiment/rkmeans_train_flat_hyp_prefix.yaml`

6. 配置与实验版本
- 原始对照实验：
  - `configs/experiment/rkmeans_train_flat.yaml`
  - 默认 `hierarchy_regularizer: null`
  - 默认 `hierarchy_loss_weight: 0.0`
  - 默认 `hierarchy_start_step: 0`
- EuclideanPrefixLoss 实验：
  - `configs/experiment/rkmeans_train_flat_euc_prefix.yaml`
  - 默认 `hierarchy_loss_weight: 0.1`
- HyperbolicPrefixContrastiveLoss 实验：
  - `configs/experiment/rkmeans_train_flat_hyp_prefix.yaml`
  - 默认 `hierarchy_loss_weight: 0.05`

7. 三组实验运行命令

7.1 本地直接运行单个实验
```bash
python -m src.train \
  experiment=rkmeans_train_flat \
  data_dir=/path/to/data \
  embedding_path=/path/to/merged_predictions_tensor.pt \
  embedding_dim=2048 \
  num_hierarchies=3 \
  codebook_width=256
```

```bash
python -m src.train \
  experiment=rkmeans_train_flat_euc_prefix \
  data_dir=/path/to/data \
  embedding_path=/path/to/merged_predictions_tensor.pt \
  embedding_dim=2048 \
  num_hierarchies=3 \
  codebook_width=256
```

```bash
python -m src.train \
  experiment=rkmeans_train_flat_hyp_prefix \
  data_dir=/path/to/data \
  embedding_path=/path/to/merged_predictions_tensor.pt \
  embedding_dim=2048 \
  num_hierarchies=3 \
  codebook_width=256
```

7.2 HPC 一次性顺序跑三组实验
```bash
sbatch my_job.sh
```

- 当前 `my_job.sh` 会调用：
  - `scripts/run_semantic_id_prefix_experiments.sh`
- 默认单次提交只会运行 baseline：
  - `rkmeans_train_flat`
  - baseline semantic ID inference
- 如需三组一起跑：
```bash
RUN_MODE=all_three sbatch my_job.sh
```
- 如需三组跑完后立刻做 proxy metrics 和 comparison：
```bash
RUN_MODE=all_three RUN_PROXY_METRICS=1 sbatch my_job.sh
```

8. 当前输出组织方式
- 三组实验会统一写到：
  - `outputs/semantic_id_stage/base`
  - `outputs/semantic_id_stage/euc_prefix`
  - `outputs/semantic_id_stage/hyp_prefix`
- 三组 inference 会统一写到：
  - `outputs/semantic_id_stage/inference/base`
  - `outputs/semantic_id_stage/inference/euc_prefix`
  - `outputs/semantic_id_stage/inference/hyp_prefix`
- proxy metrics 会写到：
  - `outputs/semantic_id_stage/proxy_metrics`
- 每个实验子目录里会保存：
  - Hydra 输出
  - `csv/version_0/metrics.csv`
  - `checkpoints/`
  - `stdout.log`
  - `stderr.log`
- 单次三实验的汇总说明会写到：
  - `outputs/semantic_id_stage/SUMMARY.md`
- 最终对照表会写到：
  - `outputs/semantic_id_stage/semantic_id_stage_comparison.csv`
  - `outputs/semantic_id_stage/semantic_id_stage_comparison.json`
  - `outputs/semantic_id_stage/semantic_id_stage_comparison.md`
- Slurm 汇总输出会写到：
  - `history/output_semantic_id_<jobid>.txt`
- 在默认 `base_only` 模式下，第一轮只需要关注：
  - `outputs/semantic_id_stage/base`
  - `outputs/semantic_id_stage/inference/base/pickle/merged_predictions.pkl`

9. 当前自动汇总表字段
- 当前实现会自动从三组训练目录中的 `metrics.csv` 提取：
  - `final_quantization_loss`
  - `final_reconstruction_loss`
  - `final_hierarchy_loss`
- 当前实现会自动从 proxy metrics 结果中提取：
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
- 最终统一写入：
  - `semantic_id_stage_comparison.csv`
- 目的就是先判断 semantic ID 本身有没有变得更健康，再决定是否继续接 TIGER

10. 当前实现的数值稳定性与已知风险
- Euclidean 版本是 batch 内 pairwise loss，batch 很大时会带来 `O(B^2)` 的额外开销
- Hyperbolic 版本同样依赖 pairwise distance，计算和显存成本也会随 batch size 上升
- Hyperbolic 版本如果向量靠近单位球边界，距离会迅速变大，所以当前实现强制投影到 `max_ball_norm < 1`
- 如果 hierarchy loss 权重过大，可能压制原始 quantization objective，导致 codebook 学习退化
- 如果 `hierarchy_start_step` 设得过早，在 quantizer 尚未稳定时可能引入噪声监督
- 当前版本是“最小可运行原型”，重点是便于对照实验和回退，不追求最强表达能力
