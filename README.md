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
