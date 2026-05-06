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

To rerun that configuration directly:

```bash
python scripts/train_cnn_v2.py \
  --model mobilenet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --dropout 0.3 \
  --target-size 64 \
  --region-attention \
  --multi-task \
  --lambda-sev 0.3
```
