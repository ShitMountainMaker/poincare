# Beauty And Toys Proxy V2 Summary

Last updated: 2026-03-17 (Asia/Shanghai)

This note summarizes:

- current downstream results for `beauty_seed42` and `toys_seed42`
- refreshed `proxy_v2` analysis using the new structure-first proxy
- current status of the `toys_seed42` `hyp_0.4` rerun

## 1. High-Level Summary

| run | old semantic selected weight | new proxy_v2 selected weight | best completed downstream weight | short conclusion |
| --- | --- | --- | --- | --- |
| beauty_seed42 | 0.05 | 0.05 | 0.3 | new proxy still prefers tighter structure; downstream still prefers stronger regularization |
| toys_seed42 | 0.2 | 0.05 | 0.2 | new proxy shifts toward compactness/entropy, but completed downstream still favors 0.2 |

## 2. Beauty Seed42

### 2.1 Semantic proxy_v2

| weight | status | selected | collision | unique | avg entropy | avg prefix metric | avg sibling separation | near-collision separation | q-loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | kept | yes | 0.178002 | 0.892984 | 5.314826 | 0.047555 | 0.111760 | 0.093067 | 17.673742 |
| 0.10 | kept | no | 0.201306 | 0.875713 | 5.302659 | 0.048844 | 0.111125 | 0.093099 | 17.425880 |
| 0.20 | kept | no | 0.189571 | 0.886208 | 5.320649 | 0.048107 | 0.110423 | 0.089362 | 17.370888 |
| 0.30 | kept | no | 0.188001 | 0.886704 | 5.308784 | 0.048572 | 0.115774 | 0.098575 | 17.667610 |
| 0.40 | rejected | no | 0.192959 | 0.882820 | 5.305966 | 0.049260 | 0.115553 | 0.099663 | 17.568813 |

### 2.2 Final proxy_v2 comparison

| method | collision | unique | entropy layer 2 | avg sibling separation | near-collision separation | prefix layer1 | prefix layer2 | prefix layer3 | q-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.192463 | 0.883150 | 5.300820 | 0.114570 | 0.097043 | 0.093009 | 0.037334 | 0.016831 | 17.517517 |
| hyp_prefix (proxy_v2 selected = 0.05) | 0.178002 | 0.892984 | 5.341876 | 0.111760 | 0.093067 | 0.091799 | 0.035581 | 0.015286 | 17.673742 |

### 2.3 Downstream test

| candidate | R@5 | R@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| base | 0.04270 | 0.06439 | 0.02844 | 0.03546 |
| hyp_0.05 | 0.04418 | 0.06453 | 0.02990 | 0.03643 |
| hyp_0.2 | 0.04315 | 0.06462 | 0.02947 | 0.03639 |
| hyp_0.3 | 0.04463 | 0.06609 | 0.03024 | 0.03716 |

### 2.4 Beauty interpretation

- `proxy_v2` still selects `0.05`.
- The main reason is that `0.05` still dominates on compactness and remains strong on entropy.
- Downstream still prefers `0.3`.
- So on `beauty`, structural semantic proxy alone is still not enough to recover the true downstream-optimal weight.

## 3. Toys Seed42

### 3.1 Semantic proxy_v2

| weight | status | selected | collision | unique | avg entropy | avg prefix metric | avg sibling separation | near-collision separation | q-loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | kept | yes | 0.247065 | 0.842754 | 5.309494 | 0.046169 | 0.096556 | 0.071481 | 17.619717 |
| 0.10 | rejected | no | 0.252348 | 0.839064 | 5.297588 | 0.047333 | 0.102243 | 0.082079 | 17.564201 |
| 0.20 | kept | no | 0.242536 | 0.846780 | 5.306900 | 0.046273 | 0.100245 | 0.078162 | 17.471672 |
| 0.30 | rejected | no | 0.255703 | 0.837387 | 5.308342 | 0.046638 | 0.101765 | 0.077646 | 17.534380 |
| 0.40 | rejected | no | 0.255619 | 0.838645 | 5.294041 | 0.047416 | 0.105502 | 0.084760 | 17.662472 |

### 3.2 Final proxy_v2 comparison

| method | collision | unique | entropy layer 2 | avg sibling separation | near-collision separation | prefix layer1 | prefix layer2 | prefix layer3 | q-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.247568 | 0.846109 | 5.316848 | 0.101728 | 0.080463 | 0.089137 | 0.034807 | 0.015905 | 17.603973 |
| hyp_prefix (proxy_v2 selected = 0.05) | 0.247065 | 0.842754 | 5.322787 | 0.096556 | 0.071481 | 0.088696 | 0.033528 | 0.016284 | 17.619717 |

### 3.3 Downstream test

| candidate | R@5 | R@10 | NDCG@5 | NDCG@10 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| base | 0.04116 | 0.06156 | 0.02776 | 0.03426 | completed |
| hyp_0.05 | 0.04281 | 0.06341 | 0.02711 | 0.03381 | completed |
| hyp_0.2 | 0.04348 | 0.06326 | 0.02784 | 0.03419 | completed |
| hyp_0.3 | 0.03858 | 0.05739 | 0.02655 | 0.03263 | completed |
| hyp_0.4 | - | - | - | - | rerun submitted, pending |

### 3.4 Toys interpretation

- `proxy_v2` now selects `0.05`, not the old `0.2`.
- This happens because the new rule prioritizes compactness and entropy more than collision/unique.
- But the completed downstream runs still favor `0.2`.
- So on `toys`, the new proxy is conceptually cleaner, but it still over-preferences the more compact candidate.

## 4. Current Takeaways

- `beauty_seed42`: new proxy still does not recover the downstream-best weight; downstream remains best at `0.3`.
- `toys_seed42`: old proxy and downstream both preferred `0.2`, but new proxy shifts to `0.05`.
- This suggests the new structure-first proxy is a useful cleanup, but not sufficient yet as a final weight selector.
- The remaining gap is now clearer:
  - `beauty`: downstream likes stronger regularization than the semantic proxy suggests.
  - `toys`: downstream likes better separation than the new proxy currently rewards.

## 5. Toys 0.4 Rerun

The previous `toys_seed42 hyp_0.4` run failed because of a DDP port collision:

- error: `EADDRINUSE`
- root cause: `MASTER_PORT` conflict during distributed initialization

It has been resubmitted with a fixed manual port:

- new job id: `239511`
- queue: `acd_u`
- candidate: `toys_seed42 hyp_0.4`
