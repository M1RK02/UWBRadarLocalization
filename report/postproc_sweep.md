# Post-processing hyperparameter sweep

Replay of `logs/20260829-181356/oof` -- 24 out-of-fold windows, 180000 frames, no model and no raw dataset involved.  
Grid: 4 threshold x 3 min_distance x 4 alpha x 7 max_coast x 3 v_max = 1008 configs, scored in 9.4 min on 10 processes.

Pooled F1/P/R aggregate TP/FP/FN over every window, exactly as the graders do. **The empty-room phantom rate is not part of that pool** -- it is the fraction of frames with at least one reported detection, over the windows whose ground truth is empty in every frame, and nothing else. Localization error is on matched TP pairs only (RMSE/MAE exact; median/P90 read off a 512-bin histogram, ~2 mm resolution).


## Windows

Empty-room windows are detected from the ground truth (`people_mask` False for every slot in every frame), never by name.

| window | scenario class (derived) | empty-room |
|---|---|---|
| window_000000 | 2p | no |
| window_000001 | 2p | no |
| window_000002 | 2p | no |
| window_000003 | 2p | no |
| window_000004 | 2p+1seated | no |
| window_000005 | 4p | no |
| window_000006 | 4p | no |
| window_000007 | 4p | no |
| window_000008 | 4p+1seated | no |
| window_000009 | 4p+1seated | no |
| window_000010 | 4p+2seated | no |
| window_000011 | 4p+2seated | no |
| window_000012 | 3p | no |
| window_000013 | 3p | no |
| window_000014 | 3p+1seated | no |
| window_000015 | 3p+2seated | no |
| window_000016 | 1p | no |
| window_000017 | 1p | no |
| window_000018 | 1p | no |
| window_000019 | 1p | no |
| window_000020 | 1p | no |
| window_000021 | 1p+1seated | no |
| window_000022 | empty | **yes** |
| window_000023 | empty | **yes** |

## Grid

- `threshold`: [0.4, 0.45, 0.5, 0.55]
- `min_distance`: [0.5, 0.6, 0.7]
- `alpha`: [0.25, 0.3, 0.35, 0.4]
- `max_coast`: [0, 1, 2, 3, 4, 5, 6]
- `v_max`: [1.5, 2.5, 4.0]
- `max_distance`: derived, not swept -- `0.57 + (max_coast + 1) * 0.04 * v_max` -> [0.63, 0.67, 0.69, 0.73, 0.75, 0.77, 0.81, 0.87, 0.89, 0.93, 0.97, 0.99, 1.05, 1.07, 1.17, 1.21, 1.27, 1.37, 1.53, 1.69]

## Top 10 by pooled F1

| # | thr | min_d | alpha | coast | v_max | max_d | F1 | Prec | Rec | empty-room phantom | RMSE | MAE | median | P90 | nb-mean F1 | nb-min F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.550 | 0.70 | 0.25 | 6 | 4.0 | 1.69 | **0.8820** | 0.8854 | 0.8787 | 1.65% | 0.314 | 0.262 | 0.221 | 0.487 | 0.8811 | 0.8799 |
| 2 | 0.550 | 0.70 | 0.30 | 6 | 4.0 | 1.69 | **0.8820** | 0.8854 | 0.8786 | 1.65% | 0.310 | 0.256 | 0.213 | 0.483 | 0.8813 | 0.8799 |
| 3 | 0.550 | 0.70 | 0.35 | 6 | 4.0 | 1.69 | **0.8818** | 0.8851 | 0.8786 | 1.65% | 0.307 | 0.252 | 0.208 | 0.481 | 0.8811 | 0.8798 |
| 4 | 0.550 | 0.70 | 0.40 | 6 | 4.0 | 1.69 | **0.8814** | 0.8848 | 0.8781 | 1.65% | 0.306 | 0.250 | 0.206 | 0.480 | 0.8807 | 0.8795 |
| 5 | 0.550 | 0.60 | 0.25 | 6 | 4.0 | 1.69 | **0.8814** | 0.8835 | 0.8793 | 1.65% | 0.314 | 0.261 | 0.221 | 0.486 | 0.8808 | 0.8793 |
| 6 | 0.550 | 0.50 | 0.25 | 6 | 4.0 | 1.69 | **0.8814** | 0.8834 | 0.8794 | 1.65% | 0.314 | 0.261 | 0.221 | 0.486 | 0.8806 | 0.8793 |
| 7 | 0.500 | 0.70 | 0.30 | 6 | 4.0 | 1.69 | **0.8814** | 0.8702 | 0.8929 | 2.35% | 0.308 | 0.253 | 0.209 | 0.482 | 0.8803 | 0.8765 |
| 8 | 0.550 | 0.60 | 0.35 | 6 | 4.0 | 1.69 | **0.8814** | 0.8834 | 0.8794 | 1.65% | 0.307 | 0.252 | 0.208 | 0.480 | 0.8808 | 0.8793 |
| 9 | 0.550 | 0.60 | 0.30 | 6 | 4.0 | 1.69 | **0.8813** | 0.8835 | 0.8792 | 1.65% | 0.309 | 0.255 | 0.213 | 0.482 | 0.8809 | 0.8794 |
| 10 | 0.500 | 0.70 | 0.25 | 6 | 4.0 | 1.69 | **0.8813** | 0.8701 | 0.8928 | 2.35% | 0.312 | 0.258 | 0.216 | 0.486 | 0.8803 | 0.8768 |

`nb-mean F1` / `nb-min F1` are the mean and minimum pooled F1 over the config's immediate neighbours (+-1 step on each of the five swept axes). A lone spike has a much lower `nb-min` than a plateau.


## Top 10 by pooled F1, among configs that do not worsen the empty room

Filter: empty-room phantom rate <= the deployed baseline's **2.68%**. 756/1008 configs qualify.

| # | thr | min_d | alpha | coast | v_max | max_d | F1 | Prec | Rec | empty-room phantom | RMSE | MAE | median | P90 | nb-mean F1 | nb-min F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.550 | 0.70 | 0.25 | 6 | 4.0 | 1.69 | **0.8820** | 0.8854 | 0.8787 | 1.65% | 0.314 | 0.262 | 0.221 | 0.487 | 0.8811 | 0.8799 |
| 2 | 0.550 | 0.70 | 0.30 | 6 | 4.0 | 1.69 | **0.8820** | 0.8854 | 0.8786 | 1.65% | 0.310 | 0.256 | 0.213 | 0.483 | 0.8813 | 0.8799 |
| 3 | 0.550 | 0.70 | 0.35 | 6 | 4.0 | 1.69 | **0.8818** | 0.8851 | 0.8786 | 1.65% | 0.307 | 0.252 | 0.208 | 0.481 | 0.8811 | 0.8798 |
| 4 | 0.550 | 0.70 | 0.40 | 6 | 4.0 | 1.69 | **0.8814** | 0.8848 | 0.8781 | 1.65% | 0.306 | 0.250 | 0.206 | 0.480 | 0.8807 | 0.8795 |
| 5 | 0.550 | 0.60 | 0.25 | 6 | 4.0 | 1.69 | **0.8814** | 0.8835 | 0.8793 | 1.65% | 0.314 | 0.261 | 0.221 | 0.486 | 0.8808 | 0.8793 |
| 6 | 0.550 | 0.50 | 0.25 | 6 | 4.0 | 1.69 | **0.8814** | 0.8834 | 0.8794 | 1.65% | 0.314 | 0.261 | 0.221 | 0.486 | 0.8806 | 0.8793 |
| 7 | 0.500 | 0.70 | 0.30 | 6 | 4.0 | 1.69 | **0.8814** | 0.8702 | 0.8929 | 2.35% | 0.308 | 0.253 | 0.209 | 0.482 | 0.8803 | 0.8765 |
| 8 | 0.550 | 0.60 | 0.35 | 6 | 4.0 | 1.69 | **0.8814** | 0.8834 | 0.8794 | 1.65% | 0.307 | 0.252 | 0.208 | 0.480 | 0.8808 | 0.8793 |
| 9 | 0.550 | 0.60 | 0.30 | 6 | 4.0 | 1.69 | **0.8813** | 0.8835 | 0.8792 | 1.65% | 0.309 | 0.255 | 0.213 | 0.482 | 0.8809 | 0.8794 |
| 10 | 0.500 | 0.70 | 0.25 | 6 | 4.0 | 1.69 | **0.8813** | 0.8701 | 0.8928 | 2.35% | 0.312 | 0.258 | 0.216 | 0.486 | 0.8803 | 0.8768 |

## Baseline (currently deployed)

`threshold=0.4, min_distance=0.6, alpha=0.4, max_distance=1.0, max_coast=2`  
-> F1 **0.8677** (P 0.8493, R 0.8869), empty-room phantom **2.68%** (402/15000 frames), RMSE 0.295 m, MAE 0.240 m, median 0.196 m, P90 0.460 m.

Best pooled F1 in the grid: **0.8820** (delta **+0.0143**) at empty-room phantom **1.65%** (delta **-1.03%**).


## Per-axis marginals

Best pooled F1 and best/worst empty-room phantom rate at each value of each axis, with the other axes free. Reading the two side by side is the point: an axis that buys F1 by trading the empty room shows it here.

| axis | value | best F1 | best phantom | worst phantom |
|---|---|---|---|---|
| `threshold` | 0.4 | 0.8686 | 1.05% | 5.38% |
| `threshold` | 0.45 | 0.8773 | 0.72% | 3.72% |
| `threshold` | 0.5 | 0.8814 | 0.47% | 2.35% |
| `threshold` | 0.55 | 0.8820 | 0.34% | 1.65% |
| `min_distance` | 0.5 | 0.8814 | 0.34% | 5.38% |
| `min_distance` | 0.6 | 0.8814 | 0.34% | 5.38% |
| `min_distance` | 0.7 | 0.8820 | 0.34% | 5.38% |
| `alpha` | 0.25 | 0.8820 | 0.34% | 5.38% |
| `alpha` | 0.3 | 0.8820 | 0.34% | 5.38% |
| `alpha` | 0.35 | 0.8818 | 0.34% | 5.38% |
| `alpha` | 0.4 | 0.8814 | 0.34% | 5.38% |
| `max_coast` | 0 | 0.8560 | 0.34% | 1.05% |
| `max_coast` | 1 | 0.8689 | 0.62% | 1.90% |
| `max_coast` | 2 | 0.8734 | 0.88% | 2.68% |
| `max_coast` | 3 | 0.8767 | 1.12% | 3.43% |
| `max_coast` | 4 | 0.8792 | 1.31% | 4.11% |
| `max_coast` | 5 | 0.8810 | 1.49% | 4.77% |
| `max_coast` | 6 | 0.8820 | 1.65% | 5.38% |
| `v_max` | 1.5 | 0.8777 | 0.34% | 5.38% |
| `v_max` | 2.5 | 0.8799 | 0.34% | 5.38% |
| `v_max` | 4.0 | 0.8820 | 0.34% | 5.38% |

## Scenario-class breakdown

Classes are derived from the ground truth (max concurrent persons, plus `+Nseated` counting person slots that never move more than 2 m over the whole recording), so pooling artifacts are visible per class rather than only in the aggregate.

| class | baseline F1 | best-config F1 | delta |
|---|---|---|---|
| 1p | 0.9086 | 0.9534 | +0.0448 |
| 1p+1seated | 0.0155 | 0.0095 | -0.0060 |
| 2p | 0.9371 | 0.9580 | +0.0209 |
| 2p+1seated | 0.7702 | 0.8148 | +0.0447 |
| 3p | 0.8395 | 0.8573 | +0.0178 |
| 3p+1seated | 0.7388 | 0.7444 | +0.0056 |
| 3p+2seated | 0.7570 | 0.7745 | +0.0175 |
| 4p | 0.8755 | 0.8896 | +0.0141 |
| 4p+1seated | 0.8940 | 0.8844 | -0.0097 |
| 4p+2seated | 0.9324 | 0.9365 | +0.0040 |
| empty | 0.0000 | 0.0000 | +0.0000 |

The `empty` class scores F1 0 under every config (no TP is possible with no ground truth) -- that is exactly why the empty-room phantom rate is tracked separately instead of being read off F1.


## Leave-one-window-out stability of the sweep

For each window, the best config is recomputed with that window excluded from the pool. **The overall winner `threshold=0.55, min_distance=0.7, alpha=0.25, max_distance=1.69, max_coast=6` also wins 17/24 hold-outs.**


## Per-window F1

| window | class | baseline F1 | best-config F1 | delta |
|---|---|---|---|---|
| window_000000 | 2p | 0.9766 | 0.9806 | +0.0040 |
| window_000001 | 2p | 0.9532 | 0.9657 | +0.0125 |
| window_000002 | 2p | 0.8554 | 0.9036 | +0.0481 |
| window_000003 | 2p | 0.9748 | 0.9874 | +0.0126 |
| window_000004 | 2p+1seated | 0.7702 | 0.8148 | +0.0447 |
| window_000005 | 4p | 0.8689 | 0.8891 | +0.0201 |
| window_000006 | 4p | 0.9017 | 0.9048 | +0.0031 |
| window_000007 | 4p | 0.8560 | 0.8753 | +0.0192 |
| window_000008 | 4p+1seated | 0.8614 | 0.8488 | -0.0126 |
| window_000009 | 4p+1seated | 0.9270 | 0.9211 | -0.0060 |
| window_000010 | 4p+2seated | 0.9486 | 0.9600 | +0.0114 |
| window_000011 | 4p+2seated | 0.9158 | 0.9117 | -0.0041 |
| window_000012 | 3p | 0.8243 | 0.8447 | +0.0204 |
| window_000013 | 3p | 0.8554 | 0.8709 | +0.0155 |
| window_000014 | 3p+1seated | 0.7388 | 0.7444 | +0.0056 |
| window_000015 | 3p+2seated | 0.7570 | 0.7745 | +0.0175 |
| window_000016 | 1p | 0.9782 | 0.9918 | +0.0136 |
| window_000017 | 1p | 0.8712 | 0.9198 | +0.0485 |
| window_000018 | 1p | 0.8271 | 0.9188 | +0.0917 |
| window_000019 | 1p | 0.9641 | 0.9774 | +0.0133 |
| window_000020 | 1p | 0.9197 | 0.9639 | +0.0441 |
| window_000021 | 1p+1seated | 0.0155 | 0.0095 | -0.0060 |
| window_000022 | empty | 0.0000 | 0.0000 | +0.0000 |
| window_000023 | empty | 0.0000 | 0.0000 | +0.0000 |

> **Boundary note:** the winner sits on the edge of the grid for ['threshold', 'min_distance', 'alpha', 'max_coast', 'v_max']. For `threshold` that means re-run `characterize` and widen. For the four pinned axes it does **not** automatically mean widen -- read the reasoning comments on `SWEEP_AXES` first; each is narrow for a physical reason, and `max_coast` in particular is capped because coasting amplifies empty-room phantoms.

