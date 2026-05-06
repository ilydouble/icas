# ICAS Thermal Training Data Construction and Distribution Report

## Overview

This report documents how the thermal imaging training dataset was constructed for the ICAS risk prediction project, based on the current repository code and the artifacts stored under `datasets/full_data/`. The description below reflects the logic implemented in `scripts/dataset_builder.py`, especially the `build-full` workflow that creates the `full_data` dataset bundle used downstream.

The project currently uses a sample-level prediction setup: one thermal image corresponds to one model input at inference time. However, image matching, metadata linking, and dataset bookkeeping are organized at the patient level to avoid identifier drift and to preserve clinical provenance.

## Source Data and Build Outputs

The `build-full` pipeline integrates three data sources:

1. Raw thermal image and temperature CSV files stored under `datasets/2024/` and `datasets/2025/`.
2. A clinical spreadsheet (`datasets/临床特征核对后最终版.xlsx`) used for canonical patient identifiers and clinical descriptors.
3. A multimodal CSV file (`datasets/patient_level_multimodal_data.csv`) used to supplement labels and derived anthropometric variables.

The resulting `datasets/full_data/` directory contains:

- `manifest.csv`: sample-level inclusion manifest.
- `patient_summary.csv`: patient-level image count summary.
- `patient_clinical_data.csv`: merged clinical table for included patients.
- `excluded_samples.csv`: extractable samples that were not linked to a known patient record.
- `source_issues.csv`: raw data quality and pairing issues observed during dataset construction.
- `images/` and `temperature/`: copied image and temperature files for included samples.

## Dataset Construction Rules

### 2024 data

For each patient directory under `datasets/2024/`, the builder searches for the sequence-1 frontal image only. The accepted filename patterns are `ID1` and `ID-1`. A sample is included only when:

- a sequence-1 image exists, and
- a same-stem temperature CSV exists alongside the image.

If multiple sequence-1 images are found, the first one in sorted order is selected and a source issue is logged.

### 2025 data

For the 2025 cohort, the builder scans `datasets/2025/热成像照片/` and parses filenames into patient ID, view, and sequence. Only frontal (`正`) images are eligible for inclusion. For each patient:

- all frontal images are considered,
- inclusion requires at least one frontal sequence-1 image to exist,
- each included frontal image must have a same-stem temperature CSV under `datasets/2025/热成像温度数据/`.

Images with unparseable names, missing frontal views, missing frontal sequence-1 images, missing temperature CSV files, or temperature CSV files without a matching image are recorded in `source_issues.csv`.

### Patient identity matching

After extractable samples are identified, each sample is matched to a canonical patient record. The matching logic is:

1. Direct match by patient ID against the union of known IDs from the clinical spreadsheet and multimodal CSV.
2. If direct ID matching fails, extract a Chinese name candidate from the source folder or filename and attempt a unique match against the clinical spreadsheet.
3. If neither method yields a unique match, the sample is excluded and written to `excluded_samples.csv`.

In the current `full_data` build, all included samples were matched directly by patient ID; no final sample required name-based rescue matching.

## Final Dataset Size

### Full matched dataset

The final `full_data` bundle contains:

- **1,419 included image samples**
- **538 included patients**
- **513 samples from 2024**
- **906 samples from 2025**

Patient-level image count patterns are highly regular:

- **428 patients** contributed **3 images** each (`1` image from 2024 and `2` frontal images from 2025)
- **85 patients** contributed **1 image** each (`1` image from 2024 only)
- **25 patients** contributed **2 images** each (`2` frontal images from 2025 only)

Equivalently, the cross-year pattern distribution is:

| Pattern | Patients |
|---|---:|
| `2024=1, 2025=2` | 428 |
| `2024=1, 2025=0` | 85 |
| `2024=0, 2025=2` | 25 |

### Training-ready labeled subset

Not every included patient in `patient_clinical_data.csv` has a binary modeling label. For model training and evaluation, the usable labeled subset is:

- **1,382 labeled samples**
- **522 labeled patients**
- **880 labeled samples from 2025**
- **502 labeled samples from 2024**

The remaining **37 samples from 16 patients** are retained in `full_data`, but do not currently carry a usable binary training label in `patient_clinical_data.csv`.

## Label Distribution

### Patient-level label distribution

Among the **522 labeled patients**:

| Binary label | Meaning | Patients | Percent |
|---|---|---:|---:|
| `0` | non-ICAS | 355 | 68.0% |
| `1` | ICAS | 167 | 32.0% |

This indicates a moderate class imbalance at the patient level.

### Sample-level label distribution

Among the **1,382 labeled samples**:

| Binary label | Samples | Percent |
|---|---:|---:|
| `0` | 940 | 68.0% |
| `1` | 442 | 32.0% |

Because most patients contribute either one or three images, the sample-level class ratio is effectively the same as the patient-level class ratio in this build.

### Stenosis severity distribution

Using the `stenosis_multiclass` field for the labeled subset:

| Severity class | Samples | Percent |
|---|---:|---:|
| `0` | 939 | 68.1% |
| `1` | 158 | 11.4% |
| `2` | 96 | 6.9% |
| `3` | 185 | 13.4% |

Two entries in the merged clinical table have missing severity metadata and are not counted in the table above.

## Demographic and Metadata Coverage

Across all **538 included patients** in `patient_clinical_data.csv`:

- **339 female**
- **198 male**
- **1 missing sex entry**

Age is available for all included patients:

- **Mean age:** 67.25 years
- **Standard deviation:** 9.02 years
- **Median age:** 67 years
- **Range:** 45 to 89 years

For the **522 labeled patients** specifically:

- **326 female**
- **195 male**
- **1 missing sex entry**
- **Mean age:** 67.28 years
- **Standard deviation:** 9.07 years
- **Median age:** 68 years
- **Range:** 45 to 89 years

### Clinical source coverage

Among the **538 included patients**:

- **521 patients** have both clinical spreadsheet data and multimodal CSV data
- **16 patients** have clinical spreadsheet data only
- **1 patient** (`BY019`) has multimodal CSV data only

Label availability is strongly tied to multimodal coverage:

- all **521 patients** with both sources are labeled,
- the single multimodal-only patient is labeled,
- the **16 clinical-only patients** are currently unlabeled for binary model training.

## Excluded Samples

### Final exclusions after extractability

The final build excludes **21 otherwise extractable samples** from **13 patient IDs**. Every final exclusion shares the same reason:

- **`no_patient_info_match`**: the sample could not be linked to a known patient record by patient ID or by a uniquely recoverable Chinese name.

The excluded patient IDs are:

- `AW006` (2 samples)
- `BQ018` (2 samples)
- `BV003` (2 samples)
- `CA021` (2 samples)
- `CE007` (2 samples)
- `CJ007` (2 samples)
- `CJ008` (2 samples)
- `GJ005` (2 samples)
- `CN093` (1 sample)
- `GCOO5` (1 sample)
- `GM104` (1 sample)
- `GM106` (1 sample)
- `GP105` (1 sample)

These exclusions are documented in `datasets/full_data/excluded_samples.csv`.

### Included but currently unlabeled patients

The following **16 included patients** are present in `full_data` but do not currently contribute to the binary training subset because the merged patient table has no usable binary label:

`007DH`, `BL002`, `BL004`, `BN001`, `BR011`, `CD059`, `CE160`, `CH008`, `CH014`, `CH023`, `CH046`, `CK031`, `CN029`, `YY033`, `YY037`, `YY053`

These cases are not excluded from the storage bundle, but they are effectively outside the current supervised training set unless labels are completed later.

## Raw Source Data Issues

The builder logged **297 raw source issues** in `datasets/full_data/source_issues.csv`. These issues reflect source-data completeness and naming problems before patient matching.

### Issue counts by type

| Issue code | Count |
|---|---:|
| `unparseable_image_name` | 267 |
| `missing_sequence_1_image` | 11 |
| `orphan_temperature_csv` | 9 |
| `missing_temperature_csv` | 8 |
| `missing_front_image` | 1 |
| `missing_front_sequence_1` | 1 |

### Issue counts by year

- **2024:** 17 issues
  - `missing_sequence_1_image`: 11
  - `missing_temperature_csv`: 6
- **2025:** 280 issues
  - `unparseable_image_name`: 267
  - `orphan_temperature_csv`: 9
  - `missing_temperature_csv`: 2
  - `missing_front_image`: 1
  - `missing_front_sequence_1`: 1

The dominant raw-data problem is therefore filename parsing failure in the 2025 image folder, not missing thermal CSV files.

## Interpretation for Modeling

For downstream ICAS modeling, two dataset definitions should be distinguished clearly in any manuscript:

1. **The full matched storage dataset**
   - 1,419 samples from 538 patients
   - used as the canonical harmonized dataset bundle

2. **The labeled training/evaluation subset**
   - 1,382 samples from 522 patients
   - the actual subset usable for supervised binary ICAS prediction under the current labels

Because inference is performed on a single image at a time in the current codebase, the effective modeling unit is the **sample**, not the patient. However, patient identifiers remain essential for provenance tracking, cross-year linkage, and leakage-free dataset splitting.

## Files to Cite in the Repository

- `scripts/dataset_builder.py`: dataset construction logic
- `datasets/full_data/manifest.csv`: sample-level inclusion manifest
- `datasets/full_data/patient_summary.csv`: patient-level image coverage summary
- `datasets/full_data/patient_clinical_data.csv`: merged clinical and label table
- `datasets/full_data/excluded_samples.csv`: post-extraction exclusions
- `datasets/full_data/source_issues.csv`: raw data quality issues

