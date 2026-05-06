#!/usr/bin/env python3
"""Face and facial ROI annotation pipeline for thermal imaging dataset."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class BBox:
    """Bounding box in xyxy format."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class CropTransform:
    """Record of crop/scale/pad transformation."""
    crop_xyxy: list[float]
    scale: float
    pad_x: int
    pad_y: int
    input_size: int

    def inverse_transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform point from ROI input space back to original image space."""
        # Remove padding
        x_unpad = x - self.pad_x
        y_unpad = y - self.pad_y
        # Unscale
        x_orig = x_unpad / self.scale
        y_orig = y_unpad / self.scale
        # Uncrop
        x_final = x_orig + self.crop_xyxy[0]
        y_final = y_orig + self.crop_xyxy[1]
        return x_final, y_final

    def inverse_transform_polygon(self, polygon: list[list[float]]) -> list[list[float]]:
        """Transform polygon from ROI input space back to original image space."""
        return [list(self.inverse_transform_point(x, y)) for x, y in polygon]


@dataclass
class RegionDetection:
    """Detection result for a facial region."""
    model: str
    confidence: float
    bbox_xyxy: list[float]
    centroid: list[float]
    polygon: list[list[float]]
    area: float


@dataclass
class FaceDetection:
    """Face segmentation result."""
    model: str
    confidence: float
    bbox_xyxy: list[float]
    polygon: list[list[float]]
    area: float
    crop_transform: dict[str, Any]


@dataclass
class QualityInfo:
    """Quality control information."""
    missing_regions: list[str] = field(default_factory=list)
    duplicate_regions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SampleAnnotation:
    """Complete annotation for one sample."""
    sample_id: str
    patient_id: str
    year: int
    image_path: str
    temperature_path: str
    image_size: dict[str, int]
    status: str
    face: FaceDetection | None = None
    regions: dict[str, RegionDetection] = field(default_factory=dict)
    quality: QualityInfo = field(default_factory=QualityInfo)


def load_yolo_model(model_path: Path, device: str = "cpu") -> YOLO:
    """Load YOLO segmentation model."""
    return YOLO(str(model_path))


def segment_face(image: np.ndarray, model: YOLO, conf_threshold: float = 0.25) -> FaceDetection | None:
    """
    Segment face using the face segmentation model.
    Returns the highest confidence 'target' detection.
    """
    results = model(image, conf=conf_threshold, verbose=False)
    
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return None
    
    # Find highest confidence detection
    best_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))
    box = result.boxes[best_idx]
    mask = result.masks[best_idx]
    
    # Extract polygon from mask
    if hasattr(mask, 'xy') and len(mask.xy) > 0:
        polygon = mask.xy[0].tolist()
    else:
        # Fallback: use bbox as polygon
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    
    bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()
    confidence = float(box.conf.item())
    
    # Calculate area
    area = cv2.contourArea(np.array(polygon, dtype=np.float32))
    
    return FaceDetection(
        model="4-segmentation",
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        polygon=polygon,
        area=area,
        crop_transform={}
    )


def prepare_roi_input(
    image: np.ndarray,
    face: FaceDetection,
    padding: int = 20,
    target_size: int = 512
) -> tuple[np.ndarray, CropTransform]:
    """
    Prepare ROI input for facial region segmentation.

    Steps:
    1. Crop face region with padding
    2. Mask out non-face regions (set to black)
    3. Scale and center-pad to target_size x target_size
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = face.bbox_xyxy

    # Add padding and clip to image bounds
    crop_x1 = max(0, int(x1) - padding)
    crop_y1 = max(0, int(y1) - padding)
    crop_x2 = min(w, int(x2) + padding)
    crop_y2 = min(h, int(y2) + padding)

    # Crop image
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Create mask for face polygon
    crop_h, crop_w = cropped.shape[:2]
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    # Transform polygon to crop coordinates
    polygon_crop = []
    for x, y in face.polygon:
        px = int(x - crop_x1)
        py = int(y - crop_y1)
        polygon_crop.append([px, py])

    cv2.fillPoly(mask, [np.array(polygon_crop, dtype=np.int32)], 255)

    # Apply mask (set non-face to black)
    masked = cropped.copy()
    masked[mask == 0] = 0

    # Scale to fit target size while maintaining aspect ratio
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    scaled = cv2.resize(masked, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center pad to target_size
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled

    transform = CropTransform(
        crop_xyxy=[crop_x1, crop_y1, crop_x2, crop_y2],
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        input_size=target_size
    )

    return result, transform


def segment_facial_regions(
    roi_input: np.ndarray,
    model: YOLO,
    conf_threshold: float = 0.25
) -> dict[str, list[dict[str, Any]]]:
    """
    Segment facial regions (eyes, nose, forehead, cheeks) from ROI input.
    Returns detections grouped by class name.
    """
    results = model(roi_input, conf=conf_threshold, verbose=False)

    if not results or len(results) == 0:
        return {}

    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return {}

    detections_by_class: dict[str, list[dict[str, Any]]] = {}

    for i in range(len(result.masks)):
        box = result.boxes[i]
        mask = result.masks[i]
        cls_id = int(box.cls.item())
        cls_name = result.names[cls_id]

        # Extract polygon
        if hasattr(mask, 'xy') and len(mask.xy) > 0:
            polygon = mask.xy[0].tolist()
        else:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf.item())

        # Calculate centroid
        polygon_arr = np.array(polygon, dtype=np.float32)
        M = cv2.moments(polygon_arr)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
            cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2

        area = cv2.contourArea(polygon_arr)

        detection = {
            'confidence': confidence,
            'bbox_xyxy': bbox_xyxy,
            'centroid': [cx, cy],
            'polygon': polygon,
            'area': area
        }

        detections_by_class.setdefault(cls_name, []).append(detection)

    return detections_by_class


def assign_bilateral_regions(
    detections: dict[str, list[dict[str, Any]]],
    face_bbox: list[float]
) -> dict[str, dict[str, Any]]:
    """
    Assign bilateral regions (Eye, Cheek) to left/right based on centroid x position.
    Assign singular regions (Nose, Forehead) by highest confidence.
    """
    assigned: dict[str, dict[str, Any]] = {}

    # Face center x
    face_center_x = (face_bbox[0] + face_bbox[2]) / 2

    # Process bilateral regions: Eye, Cheek
    for region_type in ['Eye', 'Cheek']:
        if region_type in detections:
            dets = detections[region_type]
            # Sort by centroid x
            dets_sorted = sorted(dets, key=lambda d: d['centroid'][0])

            if len(dets_sorted) >= 2:
                # Leftmost is left, rightmost is right
                assigned[f'left_{region_type.lower()}'] = dets_sorted[0]
                assigned[f'right_{region_type.lower()}'] = dets_sorted[-1]
            elif len(dets_sorted) == 1:
                # Single detection - assign based on x position
                det = dets_sorted[0]
                side = 'left' if det['centroid'][0] < face_center_x else 'right'
                assigned[f'{side}_{region_type.lower()}'] = det

    # Process singular regions: Nose, Forehead
    for region_type in ['Nose', 'Forehead']:
        if region_type in detections:
            dets = detections[region_type]
            # Take highest confidence
            best = max(dets, key=lambda d: d['confidence'])
            assigned[region_type.lower()] = best

    return assigned


def process_sample(
    sample_row: dict[str, str],
    face_model: YOLO,
    roi_model: YOLO,
    face_conf: float = 0.25,
    roi_conf: float = 0.25,
    padding: int = 20,
    roi_size: int = 512
) -> SampleAnnotation:
    """Process a single sample and generate complete annotation."""
    sample_id = sample_row['sample_id']
    patient_id = sample_row['canonical_patient_id']
    year = int(sample_row['year'])
    image_path = sample_row['image_path']
    temperature_path = sample_row['temperature_path']

    annotation = SampleAnnotation(
        sample_id=sample_id,
        patient_id=patient_id,
        year=year,
        image_path=image_path,
        temperature_path=temperature_path,
        image_size={'width': 0, 'height': 0},
        status='error'
    )

    # Read image
    try:
        image = cv2.imread(image_path)
        if image is None:
            annotation.status = 'read_error'
            annotation.quality.warnings.append(f'Failed to read image: {image_path}')
            return annotation
    except Exception as e:
        annotation.status = 'read_error'
        annotation.quality.warnings.append(f'Exception reading image: {e}')
        return annotation

    h, w = image.shape[:2]
    annotation.image_size = {'width': w, 'height': h}

    # Step 1: Face segmentation
    face = segment_face(image, face_model, face_conf)
    if face is None:
        annotation.status = 'no_face'
        annotation.quality.warnings.append('No face detected')
        return annotation

    annotation.face = face

    # Step 2: Prepare ROI input
    roi_input, transform = prepare_roi_input(image, face, padding, roi_size)
    face.crop_transform = asdict(transform)

    # Step 3: Segment facial regions
    roi_detections = segment_facial_regions(roi_input, roi_model, roi_conf)

    # Step 4: Assign bilateral regions
    assigned_roi = assign_bilateral_regions(roi_detections, face.bbox_xyxy)

    # Step 5: Transform regions back to original image coordinates
    expected_regions = ['left_eye', 'right_eye', 'nose', 'forehead', 'left_cheek', 'right_cheek']

    for region_name in expected_regions:
        if region_name in assigned_roi:
            det = assigned_roi[region_name]
            # Transform polygon and centroid back
            polygon_orig = transform.inverse_transform_polygon(det['polygon'])
            centroid_orig = transform.inverse_transform_point(det['centroid'][0], det['centroid'][1])

            # Transform bbox
            bbox_p1 = transform.inverse_transform_point(det['bbox_xyxy'][0], det['bbox_xyxy'][1])
            bbox_p2 = transform.inverse_transform_point(det['bbox_xyxy'][2], det['bbox_xyxy'][3])
            bbox_orig = [bbox_p1[0], bbox_p1[1], bbox_p2[0], bbox_p2[1]]

            # Recalculate area in original coordinates
            area_orig = cv2.contourArea(np.array(polygon_orig, dtype=np.float32))

            annotation.regions[region_name] = RegionDetection(
                model='8-re_analyze',
                confidence=det['confidence'],
                bbox_xyxy=bbox_orig,
                centroid=list(centroid_orig),
                polygon=polygon_orig,
                area=area_orig
            )
        else:
            annotation.quality.missing_regions.append(region_name)

    annotation.status = 'ok'
    return annotation


def save_masks_and_crops(
    sample: SampleAnnotation,
    image: np.ndarray,
    roi_input: np.ndarray,
    output_dir: Path
) -> None:
    """Save face mask and ROI crop images."""
    masks_dir = output_dir / 'masks'
    crops_dir = output_dir / 'crops'
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if sample.face:
        # Save face mask
        mask = np.zeros((sample.image_size['height'], sample.image_size['width']), dtype=np.uint8)
        polygon = np.array(sample.face.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        cv2.imwrite(str(masks_dir / f'{sample.sample_id}_face.png'), mask)

        # Save ROI crop
        cv2.imwrite(str(crops_dir / f'{sample.sample_id}_face_512.jpg'), roi_input)


def save_overlay(
    sample: SampleAnnotation,
    image: np.ndarray,
    output_dir: Path
) -> None:
    """Save visualization overlay with all annotations."""
    overlays_dir = output_dir / 'overlays'
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Darken the original image to make annotations stand out
    overlay = (image * 0.4).astype(np.uint8)

    # Draw face mask with semi-transparent fill
    if sample.face:
        face_mask = np.zeros_like(image, dtype=np.uint8)
        polygon = np.array(sample.face.polygon, dtype=np.int32)
        cv2.fillPoly(face_mask, [polygon], (100, 150, 200))
        overlay = cv2.addWeighted(overlay, 1.0, face_mask, 0.4, 0)

        # Draw face boundary with thick line
        cv2.polylines(overlay, [polygon], True, (150, 200, 255), 3)

    # Draw ROI regions with brighter colors and fills
    region_colors = {
        'left_eye': (0, 255, 0),        # Green
        'right_eye': (0, 255, 255),     # Yellow
        'nose': (255, 128, 0),          # Orange
        'forehead': (255, 0, 255),      # Magenta
        'left_cheek': (0, 200, 255),    # Light orange
        'right_cheek': (255, 100, 255)  # Pink
    }

    region_labels = {
        'left_eye': 'L eye',
        'right_eye': 'R eye',
        'nose': 'nose',
        'forehead': 'forehead',
        'left_cheek': 'L cheek',
        'right_cheek': 'R cheek'
    }

    # First pass: draw filled semi-transparent regions
    for region_name, region in sample.regions.items():
        color = region_colors.get(region_name, (255, 255, 255))
        polygon = np.array(region.polygon, dtype=np.int32)

        # Draw semi-transparent fill
        region_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(region_mask, [polygon], color)
        overlay = cv2.addWeighted(overlay, 1.0, region_mask, 0.3, 0)

    # Second pass: draw outlines and labels
    for region_name, region in sample.regions.items():
        color = region_colors.get(region_name, (255, 255, 255))
        polygon = np.array(region.polygon, dtype=np.int32)

        # Draw thick outline
        cv2.polylines(overlay, [polygon], True, color, 4)

        # Draw label at centroid with background
        cx, cy = int(region.centroid[0]), int(region.centroid[1])
        label = region_labels.get(region_name, region_name)

        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(overlay,
                     (cx - 5, cy - text_h - 5),
                     (cx + text_w + 5, cy + baseline + 5),
                     (0, 0, 0), -1)

        # Draw text
        cv2.putText(overlay, label, (cx, cy), font, font_scale, color, thickness)

    # Draw sample info with background
    info_text = f'{sample.sample_id} | {sample.status}'
    if sample.quality.missing_regions:
        info_text += f' | Missing: {", ".join(sample.quality.missing_regions)}'

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)

    # Draw background rectangle for info
    cv2.rectangle(overlay, (5, 5), (text_w + 15, text_h + baseline + 15), (0, 0, 0), -1)
    cv2.putText(overlay, info_text, (10, 30), font, font_scale, (0, 255, 255), thickness)

    cv2.imwrite(str(overlays_dir / f'{sample.sample_id}.jpg'), overlay)


def to_serializable(obj: Any) -> Any:
    """Convert dataclass instances to dictionaries for JSON serialization."""
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    return obj


def write_outputs(
    samples: list[SampleAnnotation],
    output_dir: Path,
    metadata: dict[str, Any]
) -> None:
    """Write JSON, JSONL, and CSV outputs."""
    # Prepare data
    samples_data = [to_serializable(s) for s in samples]

    # Write annotations.json
    output_data = {
        'metadata': metadata,
        'samples': samples_data
    }
    with (output_dir / 'annotations.json').open('w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Write annotations.jsonl
    with (output_dir / 'annotations.jsonl').open('w', encoding='utf-8') as f:
        for sample in samples_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Write summary.csv
    summary_rows = []
    for sample in samples:
        row = {
            'sample_id': sample.sample_id,
            'patient_id': sample.patient_id,
            'year': sample.year,
            'image_path': sample.image_path,
            'temperature_path': sample.temperature_path,
            'status': sample.status,
            'face_detected': sample.face is not None,
            'face_confidence': sample.face.confidence if sample.face else 0.0,
            'face_area': sample.face.area if sample.face else 0.0,
        }

        for region_name in ['left_eye', 'right_eye', 'nose', 'forehead', 'left_cheek', 'right_cheek']:
            region = sample.regions.get(region_name)
            row[f'{region_name}_detected'] = region is not None
            row[f'{region_name}_confidence'] = region.confidence if region else 0.0
            row[f'{region_name}_area'] = region.area if region else 0.0

        row['missing_regions'] = ','.join(sample.quality.missing_regions)
        row['warnings'] = ','.join(sample.quality.warnings)
        summary_rows.append(row)

    with (output_dir / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write failures.csv
    failures = [s for s in samples if s.status != 'ok']
    failure_rows = []
    for sample in failures:
        failure_rows.append({
            'sample_id': sample.sample_id,
            'patient_id': sample.patient_id,
            'status': sample.status,
            'warnings': ','.join(sample.quality.warnings),
            'missing_regions': ','.join(sample.quality.missing_regions)
        })

    with (output_dir / 'failures.csv').open('w', encoding='utf-8', newline='') as f:
        if failure_rows:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'patient_id', 'status', 'warnings', 'missing_regions'])
            writer.writeheader()
            writer.writerows(failure_rows)


def generate_html_qc(samples: list[SampleAnnotation], output_dir: Path) -> None:
    """Generate HTML quality control pages."""
    qc_dir = output_dir / 'qc'
    pages_dir = qc_dir / 'pages'
    qc_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    samples_per_page = 100
    num_pages = (len(samples) + samples_per_page - 1) // samples_per_page

    # Generate individual pages
    for page_num in range(num_pages):
        start_idx = page_num * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(samples))
        page_samples = samples[start_idx:end_idx]

        html_content = generate_page_html(page_samples, page_num + 1, num_pages, output_dir)
        page_file = pages_dir / f'page_{page_num + 1:03d}.html'
        page_file.write_text(html_content, encoding='utf-8')

    # Generate index page
    index_html = generate_index_html(samples, num_pages, output_dir)
    (qc_dir / 'index.html').write_text(index_html, encoding='utf-8')


def generate_page_html(samples: list[SampleAnnotation], page_num: int, total_pages: int, output_dir: Path) -> str:
    """Generate HTML for a single QC page."""
    cards = []
    for sample in samples:
        status_class = 'error' if sample.status != 'ok' else ('warning' if sample.quality.missing_regions else 'ok')
        missing_text = f'<br>Missing: {", ".join(sample.quality.missing_regions)}' if sample.quality.missing_regions else ''

        overlay_path = f'../../overlays/{sample.sample_id}.jpg'
        card = f'''
        <div class="card {status_class}">
            <a href="{overlay_path}" target="_blank">
                <img src="{overlay_path}" alt="{sample.sample_id}">
            </a>
            <div class="info">
                <strong>{sample.sample_id}</strong><br>
                Patient: {sample.patient_id} ({sample.year})<br>
                Status: {sample.status}{missing_text}
            </div>
        </div>
        '''
        cards.append(card)

    nav = f'<div class="nav">Page {page_num} of {total_pages}'
    if page_num > 1:
        nav += f' | <a href="page_{page_num-1:03d}.html">Previous</a>'
    if page_num < total_pages:
        nav += f' | <a href="page_{page_num+1:03d}.html">Next</a>'
    nav += ' | <a href="../index.html">Index</a></div>'

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>QC Page {page_num}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .nav {{ margin-bottom: 20px; padding: 10px; background: white; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 10px; border-radius: 5px; }}
        .card.error {{ border: 3px solid red; }}
        .card.warning {{ border: 3px solid orange; }}
        .card.ok {{ border: 1px solid #ddd; }}
        .card img {{ width: 100%; height: auto; }}
        .info {{ margin-top: 10px; font-size: 12px; }}
    </style>
</head>
<body>
    {nav}
    <div class="grid">
        {"".join(cards)}
    </div>
    {nav}
</body>
</html>'''


def generate_index_html(samples: list[SampleAnnotation], num_pages: int, output_dir: Path) -> str:
    """Generate HTML index page."""
    total = len(samples)
    ok_count = sum(1 for s in samples if s.status == 'ok' and not s.quality.missing_regions)
    missing_count = sum(1 for s in samples if s.status == 'ok' and s.quality.missing_regions)
    error_count = sum(1 for s in samples if s.status != 'ok')

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Face ROI Annotation QC</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .summary {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
        .pages {{ background: white; padding: 20px; border-radius: 5px; }}
        .pages a {{ display: inline-block; margin: 5px; padding: 10px; background: #007bff; color: white; text-decoration: none; border-radius: 3px; }}
        .pages a:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <h1>Face ROI Annotation Quality Control</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total samples: {total}</p>
        <p>OK (complete): {ok_count}</p>
        <p>OK (missing regions): {missing_count}</p>
        <p>Errors: {error_count}</p>
    </div>
    <div class="pages">
        <h2>Browse Pages</h2>
        {"".join(f'<a href="pages/page_{i:03d}.html">Page {i}</a>' for i in range(1, num_pages + 1))}
    </div>
</body>
</html>'''


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Path to manifest.csv')
    parser.add_argument('--face-seg-model', type=Path, required=True, help='Path to face segmentation model')
    parser.add_argument('--roi-seg-model', type=Path, required=True, help='Path to facial ROI segmentation model')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--face-conf', type=float, default=0.25, help='Face detection confidence threshold')
    parser.add_argument('--roi-conf', type=float, default=0.25, help='ROI detection confidence threshold')
    parser.add_argument('--padding', type=int, default=20, help='Padding around face crop in pixels')
    parser.add_argument('--roi-size', type=int, default=512, help='ROI input size')
    parser.add_argument('--limit', type=int, help='Limit number of samples to process (for testing)')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference (cpu or cuda)')

    args = parser.parse_args()

    # Load models
    print(f'Loading face segmentation model from {args.face_seg_model}...')
    face_model = load_yolo_model(args.face_seg_model, args.device)

    print(f'Loading ROI segmentation model from {args.roi_seg_model}...')
    roi_model = load_yolo_model(args.roi_seg_model, args.device)

    # Load manifest
    print(f'Loading manifest from {args.manifest}...')
    with args.manifest.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    if args.limit:
        manifest_rows = manifest_rows[:args.limit]
        print(f'Limited to {args.limit} samples for testing')

    print(f'Processing {len(manifest_rows)} samples...')

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process samples
    samples: list[SampleAnnotation] = []
    for i, row in enumerate(manifest_rows, 1):
        if i % 50 == 0:
            print(f'  Processed {i}/{len(manifest_rows)} samples...')

        annotation = process_sample(
            row,
            face_model,
            roi_model,
            args.face_conf,
            args.roi_conf,
            args.padding,
            args.roi_size
        )
        samples.append(annotation)

        # Save intermediate outputs
        if annotation.status == 'ok' or annotation.face:
            try:
                image = cv2.imread(annotation.image_path)
                if image is not None:
                    # Prepare ROI input for saving
                    if annotation.face:
                        roi_input, _ = prepare_roi_input(
                            image,
                            annotation.face,
                            args.padding,
                            args.roi_size
                        )
                        save_masks_and_crops(annotation, image, roi_input, args.output)

                    save_overlay(annotation, image, args.output)
            except Exception as e:
                print(f'  Warning: Failed to save outputs for {annotation.sample_id}: {e}')

    print(f'Processed {len(samples)} samples')

    # Generate metadata
    metadata = {
        'source_manifest': str(args.manifest),
        'image_count': len(manifest_rows),
        'face_seg_model': str(args.face_seg_model),
        'roi_seg_model': str(args.roi_seg_model),
        'face_seg_conf_threshold': args.face_conf,
        'roi_seg_conf_threshold': args.roi_conf,
        'face_crop_padding': args.padding,
        'roi_input_size': args.roi_size,
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Write outputs
    print('Writing JSON, JSONL, and CSV outputs...')
    write_outputs(samples, args.output, metadata)

    # Generate HTML QC
    print('Generating HTML QC pages...')
    generate_html_qc(samples, args.output)

    # Print summary
    ok_count = sum(1 for s in samples if s.status == 'ok')
    no_face_count = sum(1 for s in samples if s.status == 'no_face')
    error_count = sum(1 for s in samples if s.status not in ['ok', 'no_face'])
    missing_regions_count = sum(1 for s in samples if s.status == 'ok' and s.quality.missing_regions)

    print('\n=== Summary ===')
    print(f'Total samples: {len(samples)}')
    print(f'OK: {ok_count}')
    print(f'  - Complete: {ok_count - missing_regions_count}')
    print(f'  - Missing regions: {missing_regions_count}')
    print(f'No face detected: {no_face_count}')
    print(f'Errors: {error_count}')
    print(f'\nOutputs written to: {args.output}')
    print(f'QC page: {args.output / "qc" / "index.html"}')


if __name__ == '__main__':
    main()
