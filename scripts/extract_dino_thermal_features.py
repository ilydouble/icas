#!/usr/bin/env python3
"""Extract DINO features from masked facial thermal images.

This script:
1. Loads the same masked thermal samples used by the CNN pipeline.
2. Converts each 1-channel thermal map into a 3-channel DINO input.
3. Runs a Hugging Face DINOv2 backbone to get one feature vector per sample.
4. Writes a CSV for downstream classical modeling and fusion.

Typical usage:
  python scripts/extract_dino_thermal_features.py \
    --model-id facebook/dinov2-base \
    --output-csv reports/dino_thermal_features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_cnn_v3 import TemperatureDataset, load_data


def scale_thermal_to_uint8(thermal: np.ndarray) -> np.ndarray:
    """Map a thermal matrix into an 8-bit image for DINO preprocessing."""
    thermal = np.asarray(thermal, dtype=np.float32)

    finite_mask = np.isfinite(thermal)
    if not finite_mask.any():
        return np.zeros_like(thermal, dtype=np.uint8)

    finite_vals = thermal[finite_mask]
    if finite_vals.min() >= 0.0 and finite_vals.max() <= 1.0:
        scaled = thermal
    else:
        lo = float(finite_vals.min())
        hi = float(finite_vals.max())
        if hi <= lo:
            scaled = np.zeros_like(thermal, dtype=np.float32)
        else:
            scaled = (thermal - lo) / (hi - lo)

    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def prepare_dino_input(thermal: np.ndarray) -> np.ndarray:
    """Repeat one thermal channel into a 3-channel uint8 image."""
    if thermal.ndim != 2:
        raise ValueError(f"Expected 2D thermal map, got shape={thermal.shape}")
    thermal_u8 = scale_thermal_to_uint8(thermal)
    return np.repeat(thermal_u8[:, :, None], 3, axis=2)


def load_dino_components(model_id: str, *, local_files_only: bool):
    """Load the Hugging Face image processor and DINO backbone."""
    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = Dinov2Model.from_pretrained(model_id, local_files_only=local_files_only)
    return processor, model


@torch.no_grad()
def extract_feature_rows(
    dataset: TemperatureDataset,
    *,
    processor,
    model,
    device: torch.device,
) -> list[dict]:
    """Extract one DINO feature row per dataset sample."""
    model.eval()
    model.to(device)
    rows: list[dict] = []

    for idx in range(len(dataset)):
        x, y, sev, sample_id = dataset[idx]
        thermal = x[0].cpu().numpy()
        rgb = prepare_dino_input(thermal)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        sample = dataset.samples[idx]
        row = {
            "sample_id": str(sample_id),
            "patient_id": str(sample["canonical_patient_id"]),
            "year": int(sample["year"]),
            "label": int(y.item()),
            "stenosis_multiclass": float(sev.item()),
        }
        for feat_idx, value in enumerate(features):
            row[f"dino_{feat_idx:03d}"] = float(value)
        rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--model-id", type=str, default="facebook/dinov2-base")
    parser.add_argument("--local-files-only", action="store_true", help="Require the DINO model to be available in local cache.")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/dino_thermal_features.csv"))
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = load_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    all_samples = data["train"] + data["val"] + data["test"]
    repo_root = Path(".").resolve()
    npy_dir = args.npy_dir if args.npy_dir.exists() else None
    dataset = TemperatureDataset(
        all_samples,
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size=(args.target_size, args.target_size),
        use_mask=not args.no_mask,
        augment=False,
        region_attention=False,
        npy_dir=npy_dir,
    )

    processor, model = load_dino_components(args.model_id, local_files_only=args.local_files_only)
    rows = extract_feature_rows(dataset, processor=processor, model=model, device=device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    meta_path = args.output_csv.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "local_files_only": args.local_files_only,
                "target_size": args.target_size,
                "use_face_mask": not args.no_mask,
                "n_samples": len(rows),
                "feature_dim": len([key for key in rows[0].keys() if key.startswith("dino_")]) if rows else 0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Output CSV: {args.output_csv}")
    print(f"Metadata JSON: {meta_path}")


if __name__ == "__main__":
    main()
