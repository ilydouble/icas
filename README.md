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
