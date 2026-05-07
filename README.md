# icas

## Rebuild `datasets/full_data`

To rebuild the harmonized thermal dataset with the latest matching logic, run:

```bash
python scripts/dataset_builder.py build-full --output datasets/full_data
```

The explicit form is:

```bash
python scripts/dataset_builder.py build-full \
  --datasets datasets \
  --clinical-xlsx "datasets/临床特征核对后最终版.xlsx" \
  --multimodal-csv datasets/patient_level_multimodal_data.csv \
  --output datasets/full_data
```

Key outputs will be refreshed under `datasets/full_data/`, including:

- `manifest.csv`
- `excluded_samples.csv`
- `source_issues.csv`
- `analysis_report.md`

## Refresh Features And Data Split

After rebuilding `datasets/full_data`, refresh the downstream sample-level
feature table and patient-level split configuration in this order:

### 0. Extract ASR features and rank clinical associations

```bash
python scripts/extract_asr_features.py
python scripts/analyze_asr_feature_correlations.py
python scripts/select_asr_candidate_features.py
python scripts/compare_asr_clinical_models.py --no-search
python scripts/analyze_clinical_feature_correlations.py
python scripts/select_clinical_candidate_features.py
python scripts/compare_filtered_asr_clinical_models.py --no-search
python scripts/compare_topk_filtered_asr_clinical_models.py --no-search --top-k-asr 3 --top-k-clinical 3
python scripts/compare_late_fusion_asr_clinical.py --no-search --top-k-clinical 3 --asr-model GradientBoosting --asr-strategy standard --clinical-model LogisticRegression --clinical-strategy standard
```

This produces:

- `datasets/asr_2025_features.csv`
- `reports/asr_feature_correlation_scores.csv`
- `reports/asr_feature_correlation_report.md`
- `reports/asr_candidate_feature_list.csv`
- `reports/asr_candidate_modeling_subset.csv`
- `reports/asr_clinical_model_comparison_<timestamp>.csv`
- `reports/clinical_feature_correlation_scores.csv`
- `reports/clinical_feature_correlation_report.md`
- `reports/clinical_candidate_feature_list.csv`
- `reports/clinical_candidate_modeling_subset.csv`
- `reports/filtered_asr_clinical_model_comparison_<timestamp>.csv`
- `reports/topk_filtered_asr_clinical_model_comparison_<timestamp>.csv`
- `reports/late_fusion_asr_clinical_<timestamp>.csv`

The ASR feature table is merged by `canonical_patient_id` with
`datasets/full_data/patient_clinical_data.csv`, so it can be used directly for
multi-task supervision, correlation analysis, and speech feature screening.
The candidate subset script keeps the top-ranked ASR features and removes
highly redundant ones with correlation pruning before writing a compact
modeling table.
The baseline comparison script then evaluates `asr_only`, `clinical_only`, and
`fusion` feature sets with classical models for a quick sanity check before
integrating ASR features into the larger pipeline.
The clinical correlation scripts provide the same ranking and pruning workflow
for structured patient variables, so ASR and clinical features can be screened
symmetrically before any joint model is trained.
The filtered comparison script then evaluates only the screened ASR and
screened clinical subsets, which is usually a better fusion sanity check than
feeding all raw features into the same baseline model.
The top-k ablation script is a stricter sanity check that keeps only the top
few screened features from each branch before re-running the same classical
model comparison.
The late-fusion script trains ASR and clinical branches separately, then tunes
the branch weight on the validation split before reporting fused test metrics.

### 1. Re-extract temperature features

```bash
python scripts/extract_temperature_features.py
```

The explicit form is:

```bash
python scripts/extract_temperature_features.py \
  --annotations outputs/annotations/annotations.json \
  --excluded configs/excluded_samples.json \
  --output datasets/temperature_features.csv \
  --failures datasets/temperature_features_failures.csv \
  --repo-root .
```

### 2. Rebuild the train/val/test split

```bash
python scripts/make_data_split.py
```

The explicit form is:

```bash
python scripts/make_data_split.py \
  --clinical datasets/full_data/patient_clinical_data.csv \
  --features datasets/temperature_features.csv \
  --output configs/data_split.json
```

This refreshes:

- `datasets/temperature_features.csv`
- `datasets/temperature_features_failures.csv`
- `configs/data_split.json`

## Generate Face ROI Annotations

To generate face segmentation and facial ROI annotations for the current
harmonized dataset, run:

```bash
python scripts/face_roi_annotation.py \
  --manifest datasets/full_data/manifest.csv \
  --face-seg-model models/4-segmentation.pt \
  --roi-seg-model models/8-re_analyze.pt \
  --output outputs/annotations
```

Useful optional arguments:

- `--device cpu` or `--device cuda`
- `--face-conf 0.25`
- `--roi-conf 0.25`
- `--padding 20`
- `--roi-size 512`
- `--limit 100` for a small test run

This writes the main annotation artifacts under `outputs/annotations/`,
including:

- `annotations.json`
- `annotations.jsonl`
- `summary.csv`
- `failures.csv`
- `qc/index.html`

## Early Best CNN Configuration

The strongest early `train_cnn_v2` run currently archived in `reports/` is
`cnn_v2_results_20260505_113654.json`. Its key settings were:

- `model=mobilenet`
- `epochs=50`
- `batch_size=32`
- `lr=0.001`
- `dropout=0.3`
- `target_size=64`
- `region_attention=true`
- `multi_task=true`
- `lambda_sev=0.3`
- `use_face_mask=true`
- `use_severity_weighting=true`
- `augment=false`

Its sample-level test metrics were:

- `AUC-ROC = 0.6913`
- `AUC-PR = 0.5386`
- `F1 = 0.5352`

The current local experiment runner now keeps the same champion ranking target:

- best checkpoint selection defaults to `selection-metric=auc_roc`
- experiment summaries stay ranked by `test_auc_roc`

To rerun that configuration directly:

```bash
python scripts/train_cnn_v2.py \
  --model mobilenet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --dropout 0.3 \
  --target-size 64 \
  --selection-metric auc_roc \
  --region-attention \
  --multi-task \
  --lambda-sev 0.3
```

## Multimodal CNN Training

To train a sample-level thermal CNN with patient-level structured features
broadcast onto each sample, run:

```bash
python scripts/train_cnn_multimodal.py --device cuda
```

This multimodal script currently uses:

- thermal image branch: one image per sample
- ASR branch: filtered 9-dim patient-level ASR features
- clinical branch: top-3 patient-level clinical features
- fairness rule: each sample gets weight `1 / image_count(patient)` so patients
  with 3 thermal images do not dominate those with 1 image

Useful variants:

```bash
python scripts/train_cnn_multimodal.py --model deeper --multi-task --device cuda
python scripts/train_cnn_multimodal.py --model mobilenet --multi-task --freeze-backbone-epochs 3 --device cuda
python scripts/train_cnn_multimodal.py --region-attention --multi-task --device cuda
```

Recommended refined-search-aligned starting commands:

```bash
python scripts/train_cnn_multimodal.py \
  --model deeper \
  --multi-task \
  --dropout 0.2 \
  --lr 0.001 \
  --init-checkpoint reports/best_cnn_v3.pt \
  --device cuda

python scripts/train_cnn_multimodal.py \
  --model mobilenet \
  --multi-task \
  --dropout 0.3 \
  --lr 0.001 \
  --lambda-sev 0.3 \
  --init-checkpoint reports/best_cnn_v3.pt \
  --device cuda
```

If multimodal fine-tuning is unstable right after loading a strong thermal-only
checkpoint, try a very short thermal freeze instead of the older 3-epoch
MobileNet backbone freeze:

```bash
python scripts/train_cnn_multimodal.py \
  --model mobilenet \
  --multi-task \
  --init-checkpoint reports/best_cnn_v3.pt \
  --freeze-thermal-epochs 1 \
  --device cuda
```

The script reads structured features from:

- `reports/asr_candidate_modeling_subset.csv`
- `reports/clinical_candidate_modeling_subset.csv`

and writes:

- `reports/best_cnn_multimodal.pt`
- `reports/cnn_multimodal_results_<timestamp>.json`
- `reports/cnn_multimodal_history_<timestamp>.json`

## Coarse CNN Grid Search

Before running `scripts/run_refined_search.py` or `scripts/run_local_search.py`
for local refinement, use the coarse mixed grid search to decide the rough
winning recipe across method switches and representative hyperparameter
profiles.

Recommended command:

```bash
python scripts/run_grid_search.py --preset coarse
```

Useful variants:

```bash
python scripts/run_grid_search.py --preset quick --dry-run
python scripts/run_grid_search.py --preset coarse --start-from 10
python scripts/run_grid_search.py --preset coarse --limit 12 --device cuda
```

The coarse preset currently runs **33 experiments** and is designed to compare
whether to keep or drop:

- `region-attention`
- `multi-task`
- `augment`
- `pretrained`
- `face mask`
- `severity weighting`
- coarse backbone choice (`mobilenet`, `simple`, `deeper`)

It writes logs and summaries under:

- `reports/grid_search/grid_plan.json`
- `reports/grid_search/grid_search_summary.json`

Use this stage to choose the approximate winning family, then switch to
`python scripts/run_refined_search.py --preset repro` for shortlist
reproducibility checks, or `python scripts/run_refined_search.py --preset focused`
for a compact fine search around the latest shortlist.

Useful refined-search commands:

```bash
python scripts/run_refined_search.py --preset repro --dry-run
python scripts/run_refined_search.py --preset repro --device cuda
python scripts/run_refined_search.py --preset focused --start-from 7 --limit 6
```

The refined search currently targets these shortlist families:

- `deeper_baseline__profile_a`
- `mobilenet_multi_task__profile_a`
- `mobilenet_region_attention_multi_task_no_severity__profile_b`

It writes logs and summaries under:

- `reports/refined_search/`

Keep `python scripts/run_local_search.py` as an additional fallback when you
want a broader manual neighborhood search around a chosen MobileNet-style
champion family.
