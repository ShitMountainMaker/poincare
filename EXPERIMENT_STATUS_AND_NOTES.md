# Experiment Status And Notes

Last updated: 2026-03-16 (Asia/Shanghai)

This file summarizes the current GRID experiment state in the local `poincare` checkout, the code changes already made, the active run recipes, the main results so far, and the points that must be kept in mind before making new comparisons.

## 1. Scope of What Has Been Changed

Only files inside the local `poincare` checkout were changed. No project-wide structural refactor was made.

Main code change already accepted:

- `scripts/select_semantic_id_weight.py`
  - `quantization_loss` is no longer the main veto for weight selection.
  - The main selection signals are now:
    - `collision_rate`
    - `frac_unique_ids`
    - entropy-related proxy metrics
    - prefix compactness
  - `quantization_loss` is only used to catch obvious failures such as non-finite values or exploding loss.

Main orchestration additions:

- `my_job_base_hyp_matrix.sh`
- `my_job_base_hyp_downstream.sh`
- `my_job_tiger_single_candidate.sh`
- `scripts/run_base_hyp_semantic_matrix.sh`
- `scripts/run_base_hyp_sweep_then_tiger.sh`
- `scripts/submit_base_hyp_downstream_jobs.sh`
- `scripts/submit_beauty_seed42_candidate_downstreams.sh`
- `scripts/update_base_hyp_matrix_summary.sh`
- `scripts/build_base_hyp_matrix_summary.py`

These scripts were added to support:

- staged semantic runs
- staged downstream runs
- candidate-specific downstream evaluation for fixed `hyp` weights
- root-level result summaries

## 2. Current Experimental Recipe

### 2.1 Semantic stage

Current ACD semantic recipe:

- semantic embeddings: `1 GPU`
- semantic base train: `8 GPU`
- semantic hyp sweep train: `8 GPU`
- semantic inference: `1 GPU`
- proxy analysis: CPU / single process

Reason:

- semantic training was aligned to the stronger tokenizer training setup
- semantic inference on `8 GPU` hit NCCL instability on ACD
- forcing inference to `1 GPU` fixed the problem without changing semantic training

Current sweep weights:

- `0.05`
- `0.1`
- `0.2`
- `0.3`
- `0.4`

### 2.2 Downstream TIGER stage

Current ACD downstream recipe:

- `4 GPU`
- DDP via `srun`
- one candidate per job when comparing fixed weights

Reason:

- isolate `base` vs fixed `hyp` candidates cleanly
- avoid overwriting outputs

## 3. Important Configuration Facts

### 3.1 Semantic RKMeans training

Relevant file:

- `configs/experiment/rkmeans_train_flat.yaml`

Important facts:

- `batch_size_per_device = 2048`
- `max_steps = 30`
- `devices = -1`
- `accumulate_grad_batches = 1`
- `hierarchy_loss_weight = 0.0` for `base`
- semantic stage currently uses `save_last: true`
- semantic stage has no early stopping

Implication:

- `1 GPU` semantic global batch = `2048`
- `8 GPU` semantic global batch = `16384`
- changing GPU count changes optimization dynamics materially

### 3.2 TIGER training

Relevant file:

- `configs/experiment/tiger_train_flat.yaml`

Important facts:

- train `batch_size_per_device = 32`
- val/test `batch_size_per_device = 8`
- `devices = -1`
- `accumulate_grad_batches = 16`
- `max_steps = 320000`
- `save_last = false`
- checkpoint monitored on `val/recall@5`
- early stopping is enabled

Implication:

- `1 GPU` effective train batch = `32 * 1 * 16 = 512`
- `2 GPU` effective train batch = `32 * 2 * 16 = 1024`
- `4 GPU` effective train batch = `32 * 4 * 16 = 2048`

DDP synchronizes gradients, but changing GPU count still changes the global batch and therefore the optimization trajectory.

## 4. Why Semantic Results Depend So Much on GPU Count

This is not just a random training fluctuation. The current semantic tokenizer implementation is genuinely sensitive to world size.

Relevant files:

- `src/models/modules/clustering/base_clustering_module.py`
- `src/components/training_loop_functions.py`
- `src/data/loading/datamodules/sequence_datamodule.py`

Main reasons:

- file assignment changes with `trainer.world_size`
- rank-local data order changes
- the clustering initialization buffer depends on local data seen early in training
- initialization is computed on rank 0 only
- only the buffer from the first device is used to initialize centroids

Concrete implementation detail:

- `compute_initial_centroids` is `@rank_zero_only`
- the code explicitly comments that only the first device buffer is used for initialization

Implication:

- semantic `1 GPU` and `8 GPU` are not equivalent runs
- `base` can be much weaker on one recipe and much stronger on another
- a stronger multi-GPU semantic recipe can reduce base collapse substantially

## 5. Current Hyperbolic Loss: What It Does and What It Does Not Do

Relevant files:

- `src/components/prefix_losses.py`
- `src/components/hyperbolic_utils.py`
- `src/modules/clustering/residual_quantization.py`

Current `hyp` loss is:

- a projector from Euclidean embedding to a lower-dimensional space
- projection into the Poincare ball
- pairwise Poincare distance regularization based on longest common prefix depth

What it is good at:

- imposing a prefix-aware structure on quantized embeddings
- reducing collapse risk
- making semantic IDs more organized for downstream generation

What it does not directly optimize:

- explicit collision count
- uniqueness / injectivity
- asymmetric parent-child containment

Important limitation:

- the hierarchy loss uses `cluster_ids.detach()`
- if two items already have the same full semantic ID, the current loss treats them as strongest positives rather than explicitly pushing them apart

This means the current `hyp` loss is a structural regularizer, not a true anti-collision objective.

## 6. Current Multi-GPU Limitation of `hyp`

The current hyperbolic loss is computed on per-rank local batches.

Implication:

- moving from `1 GPU` to `8 GPU` does not automatically give the `hyp` loss access to `8x` more negative pairs
- each rank still computes its loss from its own local batch
- there is no cross-rank `all_gather` of embeddings for the hierarchy loss

So:

- `8 GPU` makes `base` stronger mainly through better semantic training dynamics
- `8 GPU` does not fully strengthen `hyp` in the same way

This is one of the main reasons why base improved a lot under the stronger tokenizer recipe, while `hyp` gains became more moderate.

## 7. Weight Selection Logic

Relevant file:

- `scripts/select_semantic_id_weight.py`

Current rule after the latest local update:

- main semantic proxy signals:
  - `prefix compactness`
  - `per-layer entropy`
  - `avg_sibling_separation`
  - `near_collision_separation`
- `collision_rate` and `frac_unique_ids` are still recorded, but they are now treated as guardrails rather than first-order ranking targets
- `quantization_loss` only flags obvious failure
- final candidate is selected by Pareto filtering and structure-first proxy ordering

Reason for this change:

- the current `hyp` loss does not directly optimize collision count
- it optimizes prefix-consistent geometry
- therefore using `collision_rate` as the main ranking axis was conceptually mismatched

New semantic proxy fields are computed in:

- `src/components/semantic_id_metrics.py`

New selection logic lives in:

- `scripts/select_semantic_id_weight.py`

Important caveat discovered from current runs:

- proxy-optimal weight is not always downstream-optimal weight

Observed example:

- `beauty_seed42` semantic proxy selected `0.05`
- but downstream TIGER was best at `0.3`

So proxy should be treated as a strong heuristic, not as a guaranteed downstream oracle.

### 7.1 Optional small downstream validation surrogate

A new optional hook was added for future runs:

- `scripts/extract_downstream_val_surrogate.py`
- `scripts/run_base_hyp_sweep_then_tiger.sh`

Purpose:

- run a cheap downstream `val` check for each candidate or shortlisted candidates
- use it as a second-stage tie-breaker after semantic proxy filtering
- avoid selecting weights that look best semantically but are not best for TIGER

Important notes:

- this surrogate path is optional and off by default
- it does not change the main training implementation
- it is intended for hyperparameter selection only, not for reporting final test numbers

## 8. Historical Results Already Observed

### 8.1 Earlier hpc2 beauty run

Earlier downstream result on `hpc2`:

- `base`
  - `R@5 = 0.0419`
  - `R@10 = 0.0621`
  - `NDCG@5 = 0.0282`
  - `NDCG@10 = 0.0347`
- best `hyp`
  - selected weight at that time: `0.2`
  - `R@5 = 0.0453`
  - `R@10 = 0.0651`
  - `NDCG@5 = 0.0308`
  - `NDCG@10 = 0.0371`

Interpretation:

- this run showed a strong downstream gain
- but later analysis showed that the old `base` recipe was also weaker and more collapse-prone

### 8.2 Current ACD beauty seed42 semantic proxy

Current semantic run recipe:

- semantic train on `8 GPU`
- semantic inference on `1 GPU`

Proxy summary:

| candidate | collision_rate | frac_unique_ids |
| --- | ---: | ---: |
| `base` | `0.192463` | `0.883150` |
| `hyp_0.05` | `0.178002` | `0.892984` |
| `hyp_0.2` | `0.189571` | `0.886208` |
| `hyp_0.3` | `0.188001` | `0.886704` |

Semantic selector chose:

- `selected_weight = 0.05`
- `selection_mode = pareto_frontier`

### 8.3 Current ACD beauty seed42 downstream

Current downstream recipe:

- `4 GPU`

Final results:

| candidate | R@5 | R@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| `base` | `0.042704` | `0.064392` | `0.028438` | `0.035460` |
| `hyp_0.05` | `0.044180` | `0.064526` | `0.029903` | `0.036427` |
| `hyp_0.2` | `0.043152` | `0.064616` | `0.029474` | `0.036389` |
| `hyp_0.3` | `0.044627` | `0.066091` | `0.030238` | `0.037159` |

Current conclusion for `beauty_seed42`:

- `hyp` still improves downstream under the stronger tokenizer recipe
- best downstream candidate is `0.3`, not `0.05`
- therefore stronger hierarchy weight can help more under stronger semantic training, but the gain is not monotonic

### 8.4 Current ACD toys seed42 semantic status

`toys_seed42` semantic stage is complete.

Confirmed path:

- `/data/user/cwu319/RC/poincare/data/amazon_data/toys`

Semantic selection result:

- `selected_weight = 0.2`
- `selection_mode = pareto_frontier`

### 8.5 Current ACD toys seed42 downstream jobs

As of the last check:

- `239299` `t42_hyp_0_05`
- `239300` `t42_hyp_0_2`
- `239301` `t42_hyp_0_3`
- `239304` `t42_hyp_0_4`

Later queue snapshot still showed:

- `239301` `RUNNING`
- `239304` `RUNNING`

So the toys candidate comparison is in progress.

## 9. Why `hyp` Helps Today

Best current interpretation:

- `hyp` does not mainly help by directly minimizing collision count
- `hyp` helps by making semantic IDs more structurally consistent and less ambiguous
- this makes the semantic token sequence easier for TIGER to model

What seems to be optimized in practice:

- better prefix consistency
- better separation of semantically different items
- more orderly coarse-to-fine structure for autoregressive generation

Observed lesson:

- minimum collision is not automatically the same as maximum downstream recall/NDCG
- downstream quality depends on both separability and generative friendliness

## 10. What Still Looks Weak in the Current Method

Main weakness of the current `hyp` formulation:

- it is Poincare-flavored, but it is still only a symmetric pairwise regularizer

It does not yet explicitly model:

- full collision repulsion
- near-collision hard negatives
- radial hierarchy depth
- asymmetric parent-child relations

So the current method should be viewed as:

- a prefix-structure regularizer

not yet as:

- a collision-aware hierarchical hyperbolic semantic-ID objective

## 11. Most Important Next-Method Ideas

These are not implemented yet. They are the current research directions that looked most promising.

### 11.1 Add an explicit uniqueness / anti-collision term

Borrow the useful part of HiD-VAE, but do not fully copy the framework.

Preferred idea:

- keep our unsupervised / plug-in setting
- add a pre-quantization uniqueness loss for:
  - full collision pairs
  - near-collision pairs

### 11.2 Focus on hard negatives, not just more negatives

Most useful pairs are:

- full-collision pairs
- same long prefix but different leaf
- very close continuous embeddings that should not share a leaf code

### 11.3 Replace fixed margin with prefix-depth-aware margin

Current margin:

- global and fixed

Better idea:

- harder negative if prefixes are more similar
- stronger repulsion for long-prefix mistakes

### 11.4 Make the Poincare hierarchy more explicit

Inspired by Poincare embeddings:

- use radial structure
- let parent-like prefixes stay closer to the center
- let finer leaves move outward
- move from plain pairwise matching to ranking / hierarchy-aware structure

## 12. Relevant Code Map

### 12.1 Core semantic model

- `src/modules/clustering/residual_quantization.py`
  - semantic tokenizer training step
  - combines quantization loss, reconstruction loss, hierarchy loss

- `src/models/modules/clustering/base_clustering_module.py`
  - clustering initialization logic
  - rank-0 initialization behavior

### 12.2 Hyperbolic regularizer

- `src/components/prefix_losses.py`
  - Euclidean prefix loss
  - hyperbolic prefix contrastive loss

- `src/components/hyperbolic_utils.py`
  - Poincare ball projection
  - pairwise Poincare distance

### 12.3 Data loading and world-size effects

- `src/data/loading/datamodules/sequence_datamodule.py`
  - file assignment by `world_size`
  - rank-local batch construction

- `src/components/training_loop_functions.py`
  - special scaling logic for DDP initialization steps

### 12.4 Semantic proxy and weight selection

- `scripts/analyze_semantic_ids.py`
  - computes proxy metrics from semantic ID outputs

- `scripts/select_semantic_id_weight.py`
  - selects the final hyp weight from the sweep

- `scripts/build_semantic_id_stage_comparison.py`
  - builds semantic-stage comparison tables

### 12.5 Experiment orchestration

- `scripts/run_base_hyp_sweep_then_tiger.sh`
  - one end-to-end run
  - can run semantic stage only, downstream stage only, or both

- `scripts/run_base_hyp_semantic_matrix.sh`
  - serial semantic matrix over:
    - `beauty_seed42`
    - `beauty_seed43`
    - `toys_seed42`

- `my_job_base_hyp_matrix.sh`
  - ACD semantic matrix Slurm job

- `my_job_base_hyp_downstream.sh`
  - ACD staged downstream Slurm job for selected `base + hyp`

- `my_job_tiger_single_candidate.sh`
  - single candidate TIGER train job

- `scripts/submit_beauty_seed42_candidate_downstreams.sh`
  - submits fixed candidate downstream jobs for `beauty_seed42`

- `scripts/submit_base_hyp_downstream_jobs.sh`
  - submits staged downstream jobs for the matrix outputs

- `scripts/update_base_hyp_matrix_summary.sh`
  - rebuilds root summary markdown

- `scripts/build_base_hyp_matrix_summary.py`
  - generates the root matrix summary file

## 13. Current Operational Rules To Remember

- only modify files inside local `poincare`
- only modify files inside remote cluster `poincare`
- do not compare runs across different GPU recipes as if they were identical
- semantic `1 GPU` and semantic `8 GPU` are different experimental conditions
- proxy-best weight is not guaranteed to be downstream-best weight
- current `hyp` is useful, but it is not yet a direct collision minimizer

## 14. Practical Summary

The most important state of the project right now is:

- the stronger semantic tokenizer recipe made `base` much better and more stable
- `hyp` still gives real downstream improvements under the stronger recipe
- for `beauty_seed42`, the best downstream weight is currently `0.3`
- this suggests the method still works, but the optimal weight shifts when the semantic training regime changes
- the next research step should not be blind imitation of HiD-VAE
- the next research step should be:
  - keep the plug-in hyperbolic direction
  - add direct anti-collision bias
  - make hierarchy structure more explicit and geometry-aware
