# ICAS CNN Grid Search Report

## Overview

This report summarizes the latest CNN grid search results stored under
`reports/grid_search/`, with the primary source file:

- `reports/grid_search/grid_search_summary.json`

The current report intentionally focuses on the **latest grid search batch only**
and replaces the earlier mixed summary that combined older augmentation-search
findings. The main ranking metric is **test AUC-ROC**. Test AUC-PR and test F1
are included because several runs show meaningful ranking-versus-threshold
trade-offs.

The latest grid search contains **33 completed experiments** covering:

- `mobilenet` baselines
- `mobilenet` with augmentation
- region attention on/off
- multi-task learning on/off
- pretrained vs. no-pretrained
- face-mask and severity-loss ablations
- `simple` and `deeper` CNN baselines
- three coarse hyperparameter profiles: `profile_a`, `profile_b`, `profile_c`

## Final Ranking Snapshot

The top configurations by **test AUC-ROC** are:

| Rank | Configuration | Test AUC-ROC | Test AUC-PR | Test F1 | Best Epoch | Interpretation |
|---|---|---:|---:|---:|---:|---|
| 1 | `deeper_baseline__profile_a` | 0.6478 | 0.4617 | 0.5333 | 1 | Best overall result in this grid search. |
| 2 | `mobilenet_multi_task__profile_c` | 0.6407 | 0.4293 | 0.4792 | 4 | Strongest MobileNet run by AUC-ROC. |
| 3 | `mobilenet_multi_task__profile_a` | 0.6397 | 0.4558 | 0.5140 | 1 | Best balanced MobileNet result. |
| 4 | `mobilenet_region_attention_multi_task_no_severity__profile_b` | 0.6395 | 0.4235 | 0.5536 | 7 | Highest observed F1 with near-top AUC. |
| 5 | `mobilenet_region_attention_multi_task_no_severity__profile_c` | 0.6197 | 0.4489 | 0.3030 | 25 | Strong AUC, weak threshold behavior. |
| 6 | `mobilenet_region_attention_multi_task_no_pretrained__profile_a` | 0.6128 | 0.4047 | 0.0000 | 13 | Ranking signal exists, but default threshold fails completely. |
| 7 | `mobilenet_region_attention_multi_task__profile_c` | 0.6001 | 0.4905 | 0.4390 | 14 | Best AUC-PR in the whole grid search. |
| 8 | `mobilenet_region_attention__profile_c` | 0.5972 | 0.4532 | 0.3562 | 12 | Best pure region-attention run without multi-task. |

## Main Findings

### 1. The current winner is `deeper_baseline__profile_a`

This run is now the strongest result in the repository's latest grid search:

- `test_auc_roc = 0.6478`
- `test_auc_pr = 0.4617`
- `test_f1 = 0.5333`

Its command-level configuration is:

- `model = deeper`
- `target-size = 64`
- `lr = 0.001`
- `dropout = 0.3`
- `weight-decay = 0.0001`
- no region attention
- no multi-task
- no augmentation

This is an important update because earlier working assumptions in the project
were more favorable to `mobilenet`-based candidates. The new search suggests
that a deeper plain CNN remains highly competitive, and currently sits at the
top.

### 2. `mobilenet_multi_task` remains the most stable MobileNet family

Two `mobilenet_multi_task` runs land in the top three:

- `mobilenet_multi_task__profile_c`
- `mobilenet_multi_task__profile_a`

Although neither beats `deeper_baseline__profile_a`, this family still looks
like the most reliable MobileNet branch because it performs well across more
than one profile rather than winning only in one isolated case.

Among these, `mobilenet_multi_task__profile_a` is still the best balanced
MobileNet configuration:

- `test_auc_roc = 0.6397`
- `test_auc_pr = 0.4558`
- `test_f1 = 0.5140`

### 3. Removing severity supervision can help

The strongest F1 in the entire grid search comes from:

- `mobilenet_region_attention_multi_task_no_severity__profile_b`

with:

- `test_auc_roc = 0.6395`
- `test_f1 = 0.5536`

This is a meaningful result. It suggests that the current severity auxiliary
task or severity-related weighting is not universally helpful and may be
hurting the main ICAS objective under some settings.

This ablation is strong enough that it should be revisited deliberately in the
next round rather than treated as noise.

### 4. Region attention is promising, but mixed

The strongest AUC-PR in the whole search comes from:

- `mobilenet_region_attention_multi_task__profile_c`
  - `test_auc_pr = 0.4905`

This keeps region attention in play. However, the region-attention family is
not consistently dominant across profiles, and some related variants have weak
F1 or unstable threshold behavior.

So the current evidence is:

- region attention can help ranking quality,
- but its benefits are configuration-sensitive,
- and it is not yet a clearly superior default.

### 5. Augmentation performed poorly in this search

The `mobilenet_augment` family is clearly the weakest family overall:

- family mean `test_auc_roc = 0.3400`
- family best `test_auc_roc = 0.3876`

This is not a subtle result. In this grid search batch, augmentation is a
negative finding and should not be enabled by default.

The weaker result may reflect augmentation design, thermal-domain mismatch, or
an interaction with the current training regime, but operationally the
conclusion is simple: **drop augmentation from the next short-list** unless the
augmentation strategy is redesigned.

### 6. Face mask removal also looks harmful

The `mobilenet_region_attention_multi_task_no_mask` family is weak:

- family mean `test_auc_roc = 0.4620`
- family best `test_auc_roc = 0.5161`

Compared with the stronger region-attention and multi-task families, the
no-mask ablation underperforms clearly. The current evidence supports keeping
face masking enabled.

## By-Family Summary

### Best family peaks by test AUC-ROC

| Family | Mean AUC-ROC | Best AUC-ROC | Interpretation |
|---|---:|---:|---|
| `deeper_baseline` | 0.5290 | 0.6478 | Highest single-run peak. |
| `mobilenet_region_attention_multi_task_no_severity` | 0.6079 | 0.6395 | Strong family mean and strongest F1 profile. |
| `mobilenet_multi_task` | 0.6061 | 0.6407 | Most stable MobileNet family. |
| `mobilenet_region_attention_multi_task` | 0.5555 | 0.6001 | Promising, especially for AUC-PR. |
| `mobilenet_region_attention` | 0.5378 | 0.5972 | Some signal, but weaker than stronger multi-task variants. |
| `mobilenet_augment` | 0.3400 | 0.3876 | Clear negative result. |

### Profile-level pattern

By average AUC-ROC across all methods:

| Profile | Mean AUC-ROC | Best AUC-ROC |
|---|---:|---:|
| `profile_a` | 0.5304 | 0.6478 |
| `profile_b` | 0.5256 | 0.6395 |
| `profile_c` | 0.5048 | 0.6407 |

This suggests:

- `profile_a` is still the safest overall coarse recipe,
- `profile_b` can produce very strong threshold behavior in some ablations,
- `profile_c` has slightly lower average quality but still contains several top
  candidates.

## Important Caveats

### 1. Best epoch is often very early

Several of the strongest runs peak very early:

- `deeper_baseline__profile_a`: best epoch `1`
- `mobilenet_multi_task__profile_a`: best epoch `1`
- `simple_baseline__profile_b`: best epoch `1`

This usually indicates one or more of:

- validation noise,
- unstable early stopping selection,
- aggressive optimization relative to dataset size,
- rapid overfitting after the first few epochs.

This does not invalidate the results, but it does reduce confidence that the
single best run is robust.

### 2. Some runs have good AUC but degenerate F1

Examples:

- `mobilenet_region_attention_multi_task_no_pretrained__profile_a`
- `mobilenet_region_attention_multi_task_no_pretrained__profile_b`

Both have non-trivial AUC but `test_f1 = 0.0000`. That means ranking signal can
exist while default-threshold classification is unusable. These runs should not
be promoted without threshold tuning or calibration study.

### 3. This is still single-seed evidence

All rankings here are based on one seed (`42`) per configuration. The next
decision should be based on reproducibility, not just the leaderboard order.

## Recommended Shortlist

If the goal is to decide what to verify next, the most defensible shortlist is:

### AUC-first shortlist

1. `deeper_baseline__profile_a`
2. `mobilenet_multi_task__profile_c`
3. `mobilenet_multi_task__profile_a`

### Balanced-performance shortlist

1. `deeper_baseline__profile_a`
2. `mobilenet_region_attention_multi_task_no_severity__profile_b`
3. `mobilenet_multi_task__profile_a`

If you want only two immediate follow-up candidates, I would keep:

1. `deeper_baseline__profile_a`
2. `mobilenet_multi_task__profile_a`

This pair gives one current overall winner and one stable, strong MobileNet
baseline with better interpretability in the existing project narrative.

## Recommended Next Steps

### 1. Reproduce the top candidates with multiple seeds

Highest priority:

1. `deeper_baseline__profile_a`
2. `mobilenet_multi_task__profile_a`
3. `mobilenet_region_attention_multi_task_no_severity__profile_b`

Run each with at least 3 to 5 seeds and report:

- mean and std of test AUC-ROC
- mean and std of test AUC-PR
- mean and std of test F1

### 2. Re-check threshold behavior

Because some runs show strong ranking but weak default-threshold classification,
the next comparison should explicitly include:

- default-threshold metrics
- threshold-tuned validation-selected test F1
- probability calibration if needed

### 3. Keep augmentation out of the next round

Do not spend the next round on the current augmentation family unless the
augmentation design changes materially.

### 4. Treat severity supervision as an open question

Do not assume that multi-task severity supervision helps by default. The
`no_severity` result is strong enough that this should become an explicit
decision variable in the next focused search.

## Working Conclusion

The latest grid search changes the project picture in three important ways:

1. the current best single run is now `deeper_baseline__profile_a`,
2. `mobilenet_multi_task` remains the most stable MobileNet family, and
3. the current augmentation setup is clearly not helping.

If we want a practical next round, the cleanest plan is to verify:

- `deeper_baseline__profile_a`
- `mobilenet_multi_task__profile_a`
- `mobilenet_region_attention_multi_task_no_severity__profile_b`

and make the next decision from multi-seed evidence rather than from another
large one-shot grid.
