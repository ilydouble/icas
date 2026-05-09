#!/usr/bin/env python3
"""Compare sample-level thermal, patient-level thermal, and thermal+clinical fusion.

This script produces one report-ready result table with three rows:
1. `thermal_sample_level`
2. `thermal_patient_fusion`
3. `thermal_plus_clinical`

All rows are evaluated on the same patient pool with complete clinical top-3
features so the comparison is directly reportable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_asr_clinical_models import binary_metrics
from scripts.compare_fusion_models import build_model_from_results, load_results_json
from scripts.evaluate_patient_level_fusion import (
    aggregate_patient_predictions,
    build_model,
    build_sample_metadata_lookup,
    load_checkpoint_for_inference,
    predict_sample_probabilities,
)
from scripts.train_cnn_v3 import TemperatureDataset, find_best_threshold, load_data

CLINICAL_TOP3 = [
    "waist_hip_ratio",
    "gender_encoded",
    "height",
]


def load_clinical_top3_table(clinical_path: Path) -> pd.DataFrame:
    """Load the report-ready clinical top-3 feature table."""
    clinical_df = pd.read_csv(clinical_path)
    keep_cols = ["canonical_patient_id", "label", *CLINICAL_TOP3]
    clinical_df = clinical_df.loc[:, keep_cols].copy()
    clinical_df["canonical_patient_id"] = clinical_df["canonical_patient_id"].astype(str)
    clinical_df["label"] = pd.to_numeric(clinical_df["label"], errors="coerce")
    for col in CLINICAL_TOP3:
        clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce")
    clinical_df = clinical_df.dropna(subset=["label", *CLINICAL_TOP3]).reset_index(drop=True)
    clinical_df["label"] = clinical_df["label"].astype(int)
    return clinical_df


def build_sample_probability_frame(
    sample_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    metadata: dict[str, dict],
    patient_pool: set[str],
) -> pd.DataFrame:
    """Create a sample-level probability frame restricted to one patient pool."""
    rows: list[dict] = []
    for sample_id, label, prob in zip(sample_ids, labels.tolist(), probs.tolist()):
        meta = metadata[str(sample_id)]
        patient_id = str(meta["canonical_patient_id"])
        if patient_id not in patient_pool:
            continue
        rows.append(
            {
                "sample_id": str(sample_id),
                "canonical_patient_id": patient_id,
                "year": int(meta["year"]),
                "label": int(label),
                "thermal_prob": float(prob),
            }
        )
    return pd.DataFrame(rows)


def build_patient_probability_frame(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge patient-level thermal probabilities with clinical top-3 data."""
    thermal_df = pd.DataFrame(
        {
            "canonical_patient_id": patient_ids.astype(str),
            "label": labels.astype(int),
            "thermal_prob": probs.astype(float),
        }
    )
    merged = thermal_df.merge(
        clinical_df.loc[:, ["canonical_patient_id", *CLINICAL_TOP3]],
        on="canonical_patient_id",
        how="inner",
    )
    return merged


def _sample_arrays(frame: pd.DataFrame, patient_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sub = frame[frame["canonical_patient_id"].isin(patient_ids)].copy()
    return sub["label"].to_numpy(dtype=int), sub["thermal_prob"].to_numpy(dtype=float)


def _patient_arrays(frame: pd.DataFrame, patient_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = frame[frame["canonical_patient_id"].isin(patient_ids)].copy()
    X = sub[["thermal_prob", *CLINICAL_TOP3]].to_numpy(dtype=np.float32)
    y = sub["label"].to_numpy(dtype=int)
    thermal_prob = sub["thermal_prob"].to_numpy(dtype=float)
    return X, y, thermal_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--results-json", type=Path, help="Optional training results JSON used to reconstruct the thermal model config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--model", type=str, default="mobilenet", choices=["simple", "deeper", "mobilenet", "resnet50"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--region-attention", action="store_true")
    parser.add_argument("--multi-task", action="store_true")
    parser.add_argument("--soft-label", action="store_true")
    parser.add_argument("--year-2025-weight", type=float, default=0.7)
    parser.add_argument("--patient-strategy", type=str, default="year_weighted_mean", choices=["prob_mean", "prob_max", "logit_mean", "year_weighted_mean"])
    parser.add_argument("--fusion-method", type=str, default="logistic_stacking", choices=["logistic_stacking"])
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    split = json.loads(args.split.read_text(encoding="utf-8"))

    data = load_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    repo_root = Path(".").resolve()
    npy_dir = args.npy_dir if args.npy_dir.exists() else None
    target_size = (args.target_size, args.target_size)

    clinical_df = load_clinical_top3_table(args.clinical)
    patient_pool = set(clinical_df["canonical_patient_id"].astype(str))

    val_dataset = TemperatureDataset(
        data["val"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size,
        use_mask=not args.no_mask,
        region_attention=args.region_attention,
        npy_dir=npy_dir,
    )
    test_dataset = TemperatureDataset(
        data["test"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size,
        use_mask=not args.no_mask,
        region_attention=args.region_attention,
        npy_dir=npy_dir,
    )

    if args.results_json:
        results = load_results_json(args.results_json)
        model = build_model_from_results(results).to(device)
    else:
        model = build_model(args).to(device)
    load_checkpoint_for_inference(model, args.checkpoint, device)

    multi_task = bool(getattr(model, "multi_task", False))
    soft_label = bool(getattr(model, "soft_label", False))
    val_ids, val_labels, val_probs = predict_sample_probabilities(
        model, val_dataset, device, soft_label=soft_label, multi_task=multi_task
    )
    test_ids, test_labels, test_probs = predict_sample_probabilities(
        model, test_dataset, device, soft_label=soft_label, multi_task=multi_task
    )

    val_metadata = build_sample_metadata_lookup(data["val"])
    test_metadata = build_sample_metadata_lookup(data["test"])

    sample_val_frame = build_sample_probability_frame(val_ids, val_labels, val_probs, val_metadata, patient_pool)
    sample_test_frame = build_sample_probability_frame(test_ids, test_labels, test_probs, test_metadata, patient_pool)

    comparable_val_patients = sorted(set(sample_val_frame["canonical_patient_id"]).intersection(patient_pool))
    comparable_test_patients = sorted(set(sample_test_frame["canonical_patient_id"]).intersection(patient_pool))

    sample_val_y, sample_val_prob = _sample_arrays(sample_val_frame, comparable_val_patients)
    sample_test_y, sample_test_prob = _sample_arrays(sample_test_frame, comparable_test_patients)
    sample_threshold, _ = find_best_threshold(sample_val_y, sample_val_prob, metric="f1")
    sample_test_pred = (sample_test_prob >= sample_threshold).astype(int)
    sample_metrics = binary_metrics(sample_test_y, sample_test_pred, sample_test_prob, prefix="test")

    patient_val_ids, patient_val_y, patient_val_prob = aggregate_patient_predictions(
        sample_val_frame["sample_id"].tolist(),
        sample_val_frame["label"].to_numpy(dtype=int),
        sample_val_frame["thermal_prob"].to_numpy(dtype=float),
        val_metadata,
        strategy=args.patient_strategy,
        year_2025_weight=args.year_2025_weight,
    )
    patient_test_ids, patient_test_y, patient_test_prob = aggregate_patient_predictions(
        sample_test_frame["sample_id"].tolist(),
        sample_test_frame["label"].to_numpy(dtype=int),
        sample_test_frame["thermal_prob"].to_numpy(dtype=float),
        test_metadata,
        strategy=args.patient_strategy,
        year_2025_weight=args.year_2025_weight,
    )

    patient_threshold, _ = find_best_threshold(patient_val_y, patient_val_prob, metric="f1")
    patient_test_pred = (patient_test_prob >= patient_threshold).astype(int)
    patient_metrics = binary_metrics(patient_test_y, patient_test_pred, patient_test_prob, prefix="test")

    patient_val_frame = build_patient_probability_frame(patient_val_ids, patient_val_y, patient_val_prob, clinical_df)
    patient_test_frame = build_patient_probability_frame(patient_test_ids, patient_test_y, patient_test_prob, clinical_df)
    train_ids = [pid for pid in split["train_patient_ids"] if pid in patient_pool]
    val_ids_split = [pid for pid in split["val_patient_ids"] if pid in set(patient_val_frame["canonical_patient_id"])]
    test_ids_split = [pid for pid in split["test_patient_ids"] if pid in set(patient_test_frame["canonical_patient_id"])]

    all_train_ids, all_train_labels, all_train_probs = predict_sample_probabilities(
        model,
        TemperatureDataset(
            data["train"],
            data["annotations"],
            data["labels"],
            data["severities"],
            repo_root,
            args.masks_dir,
            target_size,
            use_mask=not args.no_mask,
            region_attention=args.region_attention,
            npy_dir=npy_dir,
        ),
        device,
        soft_label=soft_label,
        multi_task=multi_task,
    )
    train_metadata = build_sample_metadata_lookup(data["train"])
    sample_train_frame = build_sample_probability_frame(all_train_ids, all_train_labels, all_train_probs, train_metadata, patient_pool)
    patient_train_ids, patient_train_y, patient_train_prob = aggregate_patient_predictions(
        sample_train_frame["sample_id"].tolist(),
        sample_train_frame["label"].to_numpy(dtype=int),
        sample_train_frame["thermal_prob"].to_numpy(dtype=float),
        train_metadata,
        strategy=args.patient_strategy,
        year_2025_weight=args.year_2025_weight,
    )
    patient_train_frame = build_patient_probability_frame(patient_train_ids, patient_train_y, patient_train_prob, clinical_df)

    X_tr, y_tr, _ = _patient_arrays(patient_train_frame, train_ids)
    X_va, y_va, thermal_val_prob_frame = _patient_arrays(patient_val_frame, val_ids_split)
    X_te, y_te, thermal_test_prob_frame = _patient_arrays(patient_test_frame, test_ids_split)
    del thermal_val_prob_frame, thermal_test_prob_frame

    if args.fusion_method != "logistic_stacking":
        raise ValueError(f"Unsupported fusion_method: {args.fusion_method}")
    fusion_model = LogisticRegression(max_iter=2000)
    fusion_model.fit(X_tr, y_tr)
    fusion_val_prob = fusion_model.predict_proba(X_va)[:, 1]
    fusion_test_prob = fusion_model.predict_proba(X_te)[:, 1]
    fusion_threshold, _ = find_best_threshold(y_va, fusion_val_prob, metric="f1")
    fusion_test_pred = (fusion_test_prob >= fusion_threshold).astype(int)
    fusion_metrics = binary_metrics(y_te, fusion_test_pred, fusion_test_prob, prefix="test")

    rows = [
        {
            "method": "thermal_sample_level",
            "evaluation_level": "sample",
            "thermal_strategy": "direct_cnn",
            "clinical_fusion": "",
            **sample_metrics,
        },
        {
            "method": "thermal_patient_fusion",
            "evaluation_level": "patient",
            "thermal_strategy": args.patient_strategy,
            "clinical_fusion": "",
            **patient_metrics,
        },
        {
            "method": "thermal_plus_clinical",
            "evaluation_level": "patient",
            "thermal_strategy": args.patient_strategy,
            "clinical_fusion": args.fusion_method,
            **fusion_metrics,
        },
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output / f"patient_level_thermal_clinical_{timestamp}.csv"
    out_json = args.output / f"patient_level_thermal_clinical_{timestamp}.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "results_json": str(args.results_json) if args.results_json else None,
                "model": args.model,
                "patient_strategy": args.patient_strategy,
                "fusion_method": args.fusion_method,
                "year_2025_weight": args.year_2025_weight,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Output CSV: {out_csv}")
    print(f"Output JSON: {out_json}")


if __name__ == "__main__":
    main()
