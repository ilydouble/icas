# Face ROI Annotation Pipeline Usage

## Overview

This pipeline performs face segmentation and facial region (ROI) annotation on thermal imaging datasets using two YOLO models:

1. **Face Segmentation Model** (4-segmentation): Segments the entire face region
2. **Facial ROI Segmentation Model** (8-re_analyze): Segments facial sub-regions (eyes, nose, forehead, cheeks)

## Dependencies

Install required dependencies:

```bash
pip install ultralytics opencv-python numpy
```

Or if using conda:

```bash
conda install -c conda-forge ultralytics opencv numpy
```

## Usage

### Basic Command

```bash
python3 scripts/face_roi_annotation.py \
  --manifest datasets/full_data/manifest.csv \
  --face-seg-model /Users/liruirui/Documents/code/IR-image/4-segmentation/thermal_segmentation/yolo11_thermal_macos/weights/best.pt \
  --roi-seg-model /Users/liruirui/Documents/code/IR-image/8-re_analyze/best.pt \
  --output datasets/full_data/face_roi_annotations
```

### Test with Limited Samples

```bash
python3 scripts/face_roi_annotation.py \
  --manifest datasets/full_data/manifest.csv \
  --face-seg-model /Users/liruirui/Documents/code/IR-image/4-segmentation/thermal_segmentation/yolo11_thermal_macos/weights/best.pt \
  --roi-seg-model /Users/liruirui/Documents/code/IR-image/8-re_analyze/best.pt \
  --output datasets/full_data/face_roi_annotations_test \
  --limit 5
```

### All Options

- `--manifest`: Path to manifest.csv (required)
- `--face-seg-model`: Path to face segmentation model .pt file (required)
- `--roi-seg-model`: Path to facial ROI segmentation model .pt file (required)
- `--output`: Output directory for annotations (required)
- `--face-conf`: Face detection confidence threshold (default: 0.25)
- `--roi-conf`: ROI detection confidence threshold (default: 0.25)
- `--padding`: Padding around face crop in pixels (default: 20)
- `--roi-size`: ROI input size for the model (default: 512)
- `--limit`: Limit number of samples to process (for testing)
- `--device`: Device for inference - 'cpu' or 'cuda' (default: cpu)

## Output Structure

```
datasets/full_data/face_roi_annotations/
├── annotations.json          # Complete JSON with all annotations
├── annotations.jsonl         # Line-delimited JSON (one sample per line)
├── summary.csv               # Summary table with detection status
├── failures.csv              # Failed samples with error messages
├── masks/                    # Face segmentation masks
│   └── <sample_id>_face.png
├── crops/                    # 512x512 preprocessed face crops
│   └── <sample_id>_face_512.jpg
├── overlays/                 # Visualization overlays
│   └── <sample_id>.jpg
└── qc/                       # HTML quality control pages
    ├── index.html
    └── pages/
        └── page_001.html
```

## Output File Formats

### annotations.json

Complete dataset with metadata and per-sample annotations including:
- Face segmentation results (bbox, polygon, confidence)
- Facial region detections (left_eye, right_eye, nose, forehead, left_cheek, right_cheek)
- All coordinates in original image space
- Crop transformation parameters for reproducibility
- Quality control information (missing regions, warnings)

### summary.csv

One row per sample with columns:
- Sample identification (sample_id, patient_id, year)
- Paths (image_path, temperature_path)
- Status (ok / no_face / read_error)
- Face detection info (detected, confidence, area)
- Per-region detection info (6 regions x 3 metrics = 18 columns)
- Quality flags (missing_regions, warnings)

### failures.csv

Samples that failed processing with error details.

## Quality Control

Open `<output>/qc/index.html` in a browser to:
- View summary statistics
- Browse annotated images (100 per page)
- Filter by status (ok / missing regions / errors)
- Click through to full-resolution overlays

Samples are color-coded:
- **Green border**: All regions detected
- **Orange border**: Some regions missing
- **Red border**: Processing error or no face detected

## Running Tests

Unit tests cover coordinate transformation, bilateral region assignment, and edge cases:

```bash
python3 -m unittest tests.test_face_roi_annotation -v
```

## Processing Pipeline

1. **Load Models**: Load both YOLO segmentation models
2. **For Each Sample**:
   a. Read image
   b. Segment face (4-segmentation model)
   c. Crop face region with padding
   d. Mask non-face areas
   e. Scale and pad to 512x512
   f. Segment facial regions (8-re_analyze model)
   g. Assign left/right for bilateral regions (eyes, cheeks)
   h. Transform all coordinates back to original image space
   i. Save masks, crops, overlays
3. **Generate Outputs**: Write JSON/JSONL/CSV and HTML QC pages

## Coordinate Systems

All output coordinates are in **original image pixel coordinates** (top-left origin).

The `crop_transform` field records the transformation from original to ROI input space, enabling:
- Verification of coordinate mapping
- Reproduction of ROI inputs
- Debugging coordinate issues

## Troubleshooting

**ImportError: No module named 'ultralytics'**
```bash
pip install ultralytics
```

**CUDA out of memory**
```bash
# Use CPU instead
--device cpu
```

**No face detected for many samples**
- Try lowering `--face-conf` threshold (e.g., 0.15)
- Check that images are thermal images matching training data
- Verify model path is correct

**Missing many facial regions**
- Try lowering `--roi-conf` threshold
- Check ROI model is trained on cropped face images
- Verify `--roi-size` matches model training size
