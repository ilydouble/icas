# ICAS Model Search Report

## Overview

This report summarizes the latest CNN model search results for the ICAS thermal
classification project. It consolidates two experiment groups:

1. the coarse architecture and ablation sweep in
   `reports/grid_search/grid_search_summary.json`, and
2. the focused augmentation comparison in
   `reports/augmentation_search/augmentation_summary.json`.

The main ranking metric in this report is **test AUC-ROC**. Test AUC-PR and
test F1 are included as supporting metrics because several runs show clear
ranking-versus-threshold trade-offs.

## Data Sources And Scope

### Grid search scope

The grid search compared:

- MobileNet baselines,
- augmentation on/off,
- region attention on/off,
- multi-task learning on/off,
- pretrained versus no-pretrained variants,
- face-mask and severity ablations, and
- coarse hyperparameter profiles `profile_a`, `profile_b`, and `profile_c`.

### Augmentation search scope

The augmentation search fixed the model family to MobileNet with multi-task
learning and compared:

- `no_augment`
- `mild_baseline`
- `mild_no_flip`
- `tiny_face_cutout`
- `tiny_attention_guided_cutout`

for `profile_a` and `profile_c`.

### Comparability note

Most results are directly comparable because they use the same task, split, and
test metrics. However, repeated runs of the same nominal configuration can still
show noticeable variation, so the tables below should be interpreted as
**evidence for narrowing the search space**, not as a final statistical proof of
superiority.

## Final Comparison Table

The table below highlights the most decision-relevant configurations across the
two searches.

| Rank | Source | Configuration | Test AUC-ROC | Test AUC-PR | Test F1 | Best Epoch | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| 1 | Grid search | `mobilenet_augment__profile_b` | 0.6588 | 0.5195 | 0.5083 | 4 | Best overall AUC-ROC in the coarse search. |
| 2 | Grid search | `simple_baseline__profile_a` | 0.6514 | 0.4539 | 0.0000 | 14 | High ranking performance, but unusable threshold behavior at the default decision point. |
| 3 | Grid search | `mobilenet_multi_task__profile_a` | 0.6397 | 0.4558 | 0.5140 | 1 | Best balanced result among the grid-search candidates. |
| 4 | Augmentation rerun | `profile_a__no_augment` | 0.6397 | 0.4558 | 0.5140 | 1 | Reproduces the strong `profile_a` multi-task baseline without augmentation. |
| 5 | Grid search | `mobilenet_region_attention__profile_c` | 0.6219 | 0.5114 | 0.3429 | 14 | Strong AUC-PR and the best region-attention result. |
| 6 | Augmentation rerun | `profile_c__no_augment` | 0.6065 | 0.4338 | 0.1132 | 4 | Same search family as the `profile_c` multi-task baseline, but with very weak threshold behavior. |
| 7 | Augmentation rerun | `profile_a__tiny_attention_guided_cutout` | 0.5813 | 0.4347 | 0.4848 | 7 | Best tested augmentation variant, but still worse than `profile_a__no_augment`. |
| 8 | Augmentation rerun | `profile_a__mild_no_flip` | 0.5625 | 0.3981 | 0.5257 | 8 | Highest F1 among augmentation variants, but clearly lower ranking quality. |

## Main Findings

### 1. The current best overall candidate is `mobilenet_augment__profile_b`

This configuration achieved the highest observed test AUC-ROC (`0.6588`) and
the highest observed test AUC-PR (`0.5195`) among the coarse grid-search runs.
It is the best current candidate if the priority is ranking quality and overall
retrieval of positive ICAS cases.

### 2. The most balanced candidate is `mobilenet_multi_task__profile_a`

The `mobilenet_multi_task__profile_a` configuration did not win on AUC-ROC, but
it delivered a better balance across all three metrics:

- AUC-ROC `0.6397`
- AUC-PR `0.4558`
- F1 `0.5140`

This makes it the safest baseline for the next round of targeted experiments.

### 3. Region attention is promising, but not yet dominant

`mobilenet_region_attention__profile_c` was the strongest region-attention run
and achieved notably strong AUC-PR (`0.5114`). This suggests that spatially
focused modeling may improve ranking quality, especially for the positive class.
However, its F1 remained modest (`0.3429`), so the present benefit appears to
be stronger in ranking than in default-threshold classification.

### 4. The tested augmentation variants did not beat the no-augmentation baseline

In the dedicated augmentation rerun, both evaluated profiles ranked
`no_augment` first:

| Profile | Best Strategy | Test AUC-ROC | Runner-Up | Delta AUC-ROC vs. `no_augment` |
|---|---|---:|---|---:|
| `profile_a` | `no_augment` | 0.6397 | `tiny_attention_guided_cutout` | -0.0584 |
| `profile_c` | `no_augment` | 0.6065 | `mild_no_flip` | -0.1724 |

This is the clearest negative result in the search: the currently implemented
light augmentation and tiny cutout variants did **not** improve test-set ranking
performance.

### 5. Several runs show metric disagreement

Some configurations achieved good AUC but poor or even degenerate F1. The most
extreme examples are:

- `simple_baseline__profile_a`: AUC-ROC `0.6514`, F1 `0.0000`
- `profile_a__mild_baseline`: AUC-ROC `0.5581`, F1 `0.0000`
- `profile_c__no_augment`: AUC-ROC `0.6065`, F1 `0.1132`

This pattern indicates that model ranking quality and default-threshold
classification quality are not aligned consistently.

## Likely Problems In The Current Search

### 1. Threshold instability and poor calibration

The repeated appearance of high-AUC / low-F1 runs suggests that some models are
learning a useful ranking but produce scores that are poorly calibrated around
the default threshold. This can make a model look strong in AUC metrics while
still failing as a practical classifier.

### 2. High sensitivity to training profile

The search is not showing a single universally strong recipe. For example:

- augmentation helped one coarse-search configuration (`profile_b`) but not the
  focused augmentation reruns,
- multi-task learning was excellent in `profile_a` but weak in other profiles,
- region attention looked strongest in `profile_c`.

This indicates that architecture choices and optimization settings are still
strongly coupled.

### 3. Possible overfitting or unstable early stopping

Many of the strongest runs peaked very early:

- epoch 1 for `mobilenet_multi_task__profile_a`
- epoch 1 for `profile_a__no_augment`
- epoch 4 for `mobilenet_augment__profile_b`

This may reflect a small effective training signal, unstable validation
selection, or an optimization regime that converges sharply and then degrades.

### 4. Search conclusions are still based on single-seed evidence

The current repository artifacts are effectively single-run comparisons. Without
multiple seeds, it is difficult to distinguish a truly better recipe from a
fortunate run.

### 5. Search batches are not yet fully normalized

Although the main summaries are usable, the experiment history shows that some
searches were rerun and some configurations were revisited later. That does not
invalidate the current tables, but it does mean the project still needs a clean,
finalized confirmation phase under one consistent protocol.

## Recommended Next Experiment Plan

### Phase 1: Reproducibility confirmation

Run the top three candidate families with at least 3 to 5 random seeds:

1. `mobilenet_augment__profile_b`
2. `mobilenet_multi_task__profile_a`
3. `mobilenet_region_attention__profile_c`

For each family, report mean and standard deviation for AUC-ROC, AUC-PR, and
F1. This should be the highest-priority next step.

### Phase 2: Threshold and calibration study

For the same finalists, evaluate:

- validation-set threshold tuning,
- probability calibration if needed, and
- threshold-selected test F1 in addition to default-threshold test F1.

This is important because several current runs appear to have usable ranking but
poor decision-threshold behavior.

### Phase 3: Focused refinement around the strongest balanced baseline

Use `mobilenet_multi_task__profile_a` as the primary refinement anchor and test
small, controlled changes one at a time:

- narrower learning-rate adjustments around `1e-3`,
- early-stopping patience and minimum-epoch settings,
- dropout around `0.2` to `0.4`,
- severity-loss weight around the current `lambda-sev = 0.3`.

This phase should aim to improve AUC without destroying the already reasonable
F1.

### Phase 4: Re-test augmentation only if it is redefined

The current augmentation family has produced a negative result overall. The next
augmentation round should be attempted only if it changes the design materially,
for example:

- weaker perturbation magnitude,
- ROI-aware augmentation that preserves thermal structure better, or
- augmentation policies tailored separately for `64x64` and `96x96` inputs.

Blindly repeating the same augmentation family is unlikely to be productive.

### Phase 5: Investigate the region-attention path further

Because `mobilenet_region_attention__profile_c` achieved strong ranking metrics,
it should remain in scope. The next tests should determine whether its lower F1
comes mainly from thresholding, calibration, or genuinely weaker separation near
the operating point.

## Recommended Working Conclusion

At the current stage, the project should treat:

- `mobilenet_augment__profile_b` as the **best AUC-first candidate**,
- `mobilenet_multi_task__profile_a` as the **best balanced candidate**, and
- `mobilenet_region_attention__profile_c` as the **most promising alternative
  architecture for further validation**.

The focused augmentation search does not currently justify enabling augmentation
by default for the multi-task MobileNet baseline. The next round should
prioritize reproducibility, threshold analysis, and a tighter local refinement
around the best balanced configuration.
