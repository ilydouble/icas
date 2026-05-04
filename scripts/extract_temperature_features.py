#!/usr/bin/env python3
"""Extract temperature features from thermal CSVs using face ROI annotations.

For each sample, this script:
1. Loads the per-pixel temperature matrix (CSV) and matches it to the image grid.
2. For each ROI polygon (face + 6 sub-regions) builds a binary mask.
3. Computes per-region statistics and cross-region features designed for
   downstream ICAS classification (left/right asymmetry, hot/cold contrasts).

Input:
- annotations.json produced by scripts/face_roi_annotation.py
- temperature CSV files referenced by `temperature_path` in each sample
- configs/excluded_samples.json with sample_ids to skip

Output:
- datasets/temperature_features.csv  (one row per included sample)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REGION_NAMES = [
    "forehead",
    "nose",
    "left_eye",
    "right_eye",
    "left_cheek",
    "right_cheek",
]

# Per-region statistics produced by `region_stats`.
STAT_NAMES = [
    "mean", "std", "median", "min", "max", "range",
    "p5", "p25", "p75", "p95",
    "skew", "kurtosis",
]


def parse_temperature_csv(path: Path) -> np.ndarray:
    """Parse a thermal CSV file into a 2D temperature matrix.

    File layout:
        line 1: chinese metadata header (rule, max, mean, min, ...)
        line 2: metadata values
        line 3: blank
        line 4: column index header (",1,2,...,W,")
        line 5+: data rows ("r,t11,t12,...,t1W,")

    Trailing empty fields (caused by a trailing comma) are stripped.
    """
    with path.open("r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    rows: list[list[float]] = []
    for line in lines[4:]:
        parts = line.rstrip("\r\n").split(",")
        if len(parts) <= 1 or parts[0] == "":
            continue
        values = [v for v in parts[1:] if v != ""]
        if not values:
            continue
        rows.append([float(v) for v in values])

    if not rows:
        raise ValueError(f"No temperature rows parsed from {path}")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"Inconsistent row widths in {path}")

    return np.asarray(rows, dtype=np.float32)


def build_polygon_mask(polygon: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon (list of [x, y] floats) to a uint8 mask of `shape` (H, W)."""
    mask = np.zeros(shape, dtype=np.uint8)
    if not polygon:
        return mask
    pts = np.asarray(polygon, dtype=np.float32).round().astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def align_polygon_to_temperature(
    polygon: list[list[float]],
    image_size: dict,
    temp_shape: tuple[int, int],
) -> list[list[float]]:
    """Rescale polygon coordinates from image space to temperature-matrix space.

    Coordinates in `annotations.json` are in original image pixel space
    (top-left origin). The temperature matrix may have a different resolution
    (e.g. 1024 x 1280 vs the same image size). We map by simple uniform scaling.
    """
    img_w = float(image_size.get("width") or 0)
    img_h = float(image_size.get("height") or 0)
    if img_w <= 0 or img_h <= 0:
        return polygon

    temp_h, temp_w = temp_shape
    sx = temp_w / img_w
    sy = temp_h / img_h
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return polygon
    return [[x * sx, y * sy] for x, y in polygon]


def _safe_skew(values: np.ndarray) -> float:
    """Population skewness; returns 0 when std is zero."""
    if values.size < 2:
        return 0.0
    mu = float(values.mean())
    sigma = float(values.std())
    if sigma < 1e-9:
        return 0.0
    return float(((values - mu) ** 3).mean() / (sigma ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis (Fisher); returns 0 when std is zero."""
    if values.size < 2:
        return 0.0
    mu = float(values.mean())
    sigma = float(values.std())
    if sigma < 1e-9:
        return 0.0
    return float(((values - mu) ** 4).mean() / (sigma ** 4) - 3.0)


def region_stats(values: np.ndarray) -> dict[str, float]:
    """Compute the standard set of statistics for a 1D temperature sample."""
    if values.size == 0:
        return {name: float("nan") for name in STAT_NAMES}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "skew": _safe_skew(values),
        "kurtosis": _safe_kurtosis(values),
    }


def asymmetry_features(left: dict[str, float], right: dict[str, float], prefix: str) -> dict[str, float]:
    """Bilateral asymmetry features for paired regions (e.g. left/right eye)."""
    lm = left.get("mean", float("nan"))
    rm = right.get("mean", float("nan"))
    diff = lm - rm
    abs_diff = abs(diff)
    denom = (abs(lm) + abs(rm)) / 2.0
    ratio = (lm / rm) if rm not in (0.0, float("nan")) and not np.isnan(rm) else float("nan")
    asym = (abs_diff / denom) if denom > 1e-9 else float("nan")
    # std difference captures distribution-shape asymmetry
    ls = left.get("std", float("nan"))
    rs = right.get("std", float("nan"))
    return {
        f"{prefix}_diff_mean": diff,
        f"{prefix}_abs_diff_mean": abs_diff,
        f"{prefix}_ratio_mean": ratio,
        f"{prefix}_asymmetry_index": asym,
        f"{prefix}_diff_std": ls - rs,
    }


def hotspot_features(temp: np.ndarray, mask: np.ndarray, bbox: list[float]) -> dict[str, float]:
    """Spatial features inside the face mask: hot-spot location (relative) and gradient.

    `bbox` is the face bounding box [x1, y1, x2, y2] in temperature coordinates.
    """
    h, w = temp.shape
    if mask.sum() == 0 or temp.size == 0:
        return {
            "face_hotspot_x_rel": float("nan"),
            "face_hotspot_y_rel": float("nan"),
            "face_thermal_gradient_mean": float("nan"),
            "face_thermal_gradient_std": float("nan"),
        }

    # Hot-spot: argmax inside the mask
    masked = np.where(mask > 0, temp, -np.inf)
    hot_idx = int(np.argmax(masked))
    hy, hx = divmod(hot_idx, w)

    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    hx_rel = (hx - x1) / bw
    hy_rel = (hy - y1) / bh

    # Gradient magnitude inside the mask
    gx = cv2.Sobel(temp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(temp, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_in = grad[mask > 0]
    return {
        "face_hotspot_x_rel": float(hx_rel),
        "face_hotspot_y_rel": float(hy_rel),
        "face_thermal_gradient_mean": float(grad_in.mean()) if grad_in.size else float("nan"),
        "face_thermal_gradient_std": float(grad_in.std()) if grad_in.size else float("nan"),
    }


@dataclass
class SampleResult:
    sample_id: str
    patient_id: str
    year: int
    status: str
    features: dict[str, float]


def _flatten(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": stats.get(name, float("nan")) for name in STAT_NAMES}


def _empty_stats() -> dict[str, float]:
    return {name: float("nan") for name in STAT_NAMES}


def extract_sample_features(sample: dict, repo_root: Path) -> SampleResult:
    """Compute the full feature vector for one annotation sample."""
    sample_id = sample["sample_id"]
    patient_id = sample.get("patient_id", "")
    year = int(sample.get("year", 0) or 0)
    image_size = sample.get("image_size", {}) or {}
    face = sample.get("face")
    regions = sample.get("regions") or {}

    if sample.get("status") != "ok" or not face:
        return SampleResult(sample_id, patient_id, year, sample.get("status", "skipped"), {})

    temp_path = repo_root / sample["temperature_path"]
    if not temp_path.exists():
        return SampleResult(sample_id, patient_id, year, "missing_temperature_csv", {})

    try:
        temp = parse_temperature_csv(temp_path)
    except Exception as exc:  # noqa: BLE001
        return SampleResult(sample_id, patient_id, year, f"temperature_parse_error: {exc}", {})

    temp_shape = temp.shape  # (H, W)

    # --- Face region ---
    face_polygon_temp = align_polygon_to_temperature(face["polygon"], image_size, temp_shape)
    face_mask = build_polygon_mask(face_polygon_temp, temp_shape)
    face_values = temp[face_mask > 0]
    if face_values.size == 0:
        return SampleResult(sample_id, patient_id, year, "empty_face_mask", {})

    face_stats = region_stats(face_values)
    features: dict[str, float] = {}
    features.update(_flatten("face", face_stats))

    # --- Sub-regions ---
    region_stat_map: dict[str, dict[str, float]] = {}
    for name in REGION_NAMES:
        region = regions.get(name)
        if not region or not region.get("polygon"):
            stats = _empty_stats()
        else:
            poly_temp = align_polygon_to_temperature(region["polygon"], image_size, temp_shape)
            mask = build_polygon_mask(poly_temp, temp_shape)
            values = temp[mask > 0]
            stats = region_stats(values) if values.size > 0 else _empty_stats()
        region_stat_map[name] = stats
        features.update(_flatten(name, stats))

    # --- Bilateral asymmetry ---
    features.update(asymmetry_features(region_stat_map["left_eye"], region_stat_map["right_eye"], "eye"))
    features.update(asymmetry_features(region_stat_map["left_cheek"], region_stat_map["right_cheek"], "cheek"))

    # --- Region-vs-face contrasts ---
    face_mean = face_stats["mean"]
    for name in REGION_NAMES:
        features[f"{name}_minus_face_mean"] = region_stat_map[name]["mean"] - face_mean

    # --- Inter-region contrasts (clinically meaningful pairs) ---
    nose_m = region_stat_map["nose"]["mean"]
    forehead_m = region_stat_map["forehead"]["mean"]
    eyes_m = np.nanmean([region_stat_map["left_eye"]["mean"], region_stat_map["right_eye"]["mean"]])
    cheeks_m = np.nanmean([region_stat_map["left_cheek"]["mean"], region_stat_map["right_cheek"]["mean"]])
    features["nose_minus_forehead_mean"] = nose_m - forehead_m
    features["nose_minus_cheeks_mean"] = nose_m - cheeks_m
    features["forehead_minus_eyes_mean"] = forehead_m - eyes_m
    features["eyes_minus_cheeks_mean"] = eyes_m - cheeks_m
    features["eyes_mean"] = float(eyes_m)
    features["cheeks_mean"] = float(cheeks_m)

    # --- Spatial / hotspot features ---
    bbox_temp_x1, bbox_temp_y1 = align_polygon_to_temperature(
        [[face["bbox_xyxy"][0], face["bbox_xyxy"][1]]], image_size, temp_shape
    )[0]
    bbox_temp_x2, bbox_temp_y2 = align_polygon_to_temperature(
        [[face["bbox_xyxy"][2], face["bbox_xyxy"][3]]], image_size, temp_shape
    )[0]
    features.update(hotspot_features(temp, face_mask, [bbox_temp_x1, bbox_temp_y1, bbox_temp_x2, bbox_temp_y2]))

    return SampleResult(sample_id, patient_id, year, "ok", features)


def load_excluded_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("excluded_samples", [])
    ids: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("sample_id"):
            ids.add(str(item["sample_id"]))
        elif isinstance(item, str):
            ids.add(item)
    return ids


def feature_columns() -> list[str]:
    """Stable column order for the output CSV."""
    cols: list[str] = []
    for region in ["face"] + REGION_NAMES:
        cols.extend(f"{region}_{s}" for s in STAT_NAMES)
    for prefix in ("eye", "cheek"):
        cols.extend([
            f"{prefix}_diff_mean",
            f"{prefix}_abs_diff_mean",
            f"{prefix}_ratio_mean",
            f"{prefix}_asymmetry_index",
            f"{prefix}_diff_std",
        ])
    cols.extend(f"{name}_minus_face_mean" for name in REGION_NAMES)
    cols.extend([
        "nose_minus_forehead_mean",
        "nose_minus_cheeks_mean",
        "forehead_minus_eyes_mean",
        "eyes_minus_cheeks_mean",
        "eyes_mean",
        "cheeks_mean",
        "face_hotspot_x_rel",
        "face_hotspot_y_rel",
        "face_thermal_gradient_mean",
        "face_thermal_gradient_std",
    ])
    return cols


def write_features_csv(
    results: Iterable[SampleResult],
    output_path: Path,
) -> tuple[int, int]:
    """Write per-sample feature rows; returns (written, skipped)."""
    cols = feature_columns()
    fieldnames = ["sample_id", "patient_id", "year", "status"] + cols
    written = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r.status != "ok":
                skipped += 1
                continue
            row: dict[str, object] = {
                "sample_id": r.sample_id,
                "patient_id": r.patient_id,
                "year": r.year,
                "status": r.status,
            }
            for c in cols:
                v = r.features.get(c, float("nan"))
                row[c] = "" if (isinstance(v, float) and np.isnan(v)) else v
            writer.writerow(row)
            written += 1
    return written, skipped


def write_failures_csv(results: Iterable[SampleResult], path: Path) -> int:
    failures = [r for r in results if r.status != "ok"]
    if not failures:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "patient_id", "year", "status"])
        writer.writeheader()
        for r in failures:
            writer.writerow({
                "sample_id": r.sample_id,
                "patient_id": r.patient_id,
                "year": r.year,
                "status": r.status,
            })
    return len(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/full_data/face_roi_annotations/annotations.json"),
    )
    parser.add_argument(
        "--excluded",
        type=Path,
        default=Path("configs/excluded_samples.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/temperature_features.csv"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("datasets/temperature_features_failures.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, help="Process only the first N samples (debug).")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    samples = annotations.get("samples", [])
    excluded_ids = load_excluded_ids(args.excluded)
    print(f"Loaded {len(samples)} annotated samples; "
          f"{len(excluded_ids)} sample_ids in exclusion list.")

    if args.limit:
        samples = samples[: args.limit]

    results: list[SampleResult] = []
    excluded_count = 0
    for i, sample in enumerate(samples, 1):
        sid = sample.get("sample_id", "")
        if sid in excluded_ids:
            excluded_count += 1
            continue
        results.append(extract_sample_features(sample, repo_root))
        if i % 100 == 0:
            print(f"  Processed {i}/{len(samples)} samples...")

    written, skipped = write_features_csv(results, args.output)
    failures = write_failures_csv(results, args.failures)

    print("\n=== Summary ===")
    print(f"Annotated samples         : {len(samples)}")
    print(f"Excluded by config        : {excluded_count}")
    print(f"Feature rows written      : {written}")
    print(f"Skipped (status != ok)    : {skipped}")
    print(f"Failure rows              : {failures}")
    print(f"Output CSV                : {args.output}")
    if failures:
        print(f"Failures CSV              : {args.failures}")


if __name__ == "__main__":
    main()

