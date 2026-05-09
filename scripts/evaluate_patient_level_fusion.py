#!/usr/bin/env python3
"""Evaluate patient-level shallow fusion from a trained single-image thermal CNN.

This script reuses the single-image checkpoint from `scripts/train_cnn_v3.py`,
runs sample-level inference, and aggregates predictions to the patient level.

Supported fusion rules:
1. `prob_mean`: mean of per-image probabilities
2. `prob_max`: max of per-image probabilities
3. `logit_mean`: mean of per-image logits, then sigmoid/softmax
4. `year_weighted_mean`: weighted mean that favors 2025 images
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train_cnn_v3 import (
    AUGMENTATION_STRATEGIES,
    DeeperCNN,
    MobileNetV3Small,
    ResNet50Backbone,
    SimpleCNN,
    TemperatureDataset,
    compute_metrics,
    find_best_threshold,
    load_data,
)

FUSION_STRATEGIES = ("prob_mean", "prob_max", "logit_mean", "year_weighted_mean")


def build_sample_metadata_lookup(samples: list[dict]) -> dict[str, dict]:
    """Map sample IDs onto patient/year metadata for aggregation."""
    lookup: dict[str, dict] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        lookup[sample_id] = {
            "canonical_patient_id": str(sample["canonical_patient_id"]),
            "year": int(sample["year"]),
        }
    return lookup


def _safe_logit(prob: float) -> float:
    clipped = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def aggregate_patient_predictions(
    sample_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metadata: dict[str, dict],
    *,
    strategy: str,
    year_2025_weight: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sample predictions into patient-level probabilities."""
    if strategy not in FUSION_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    groups: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for sample_id, label, prob in zip(sample_ids, y_true.tolist(), y_prob.tolist()):
        meta = metadata[str(sample_id)]
        groups[meta["canonical_patient_id"]].append((float(prob), int(label), int(meta["year"])))

    patient_ids = sorted(groups)
    patient_labels: list[int] = []
    patient_probs: list[float] = []

    for patient_id in patient_ids:
        rows = groups[patient_id]
        labels = {label for _, label, _ in rows}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels found for patient {patient_id}")
        patient_labels.append(next(iter(labels)))

        probs = np.array([prob for prob, _, _ in rows], dtype=float)
        years = np.array([year for _, _, year in rows], dtype=int)

        if strategy == "prob_mean":
            patient_probs.append(float(probs.mean()))
        elif strategy == "prob_max":
            patient_probs.append(float(probs.max()))
        elif strategy == "logit_mean":
            logits = np.array([_safe_logit(prob) for prob in probs], dtype=float)
            mean_logit = float(logits.mean())
            patient_probs.append(float(1.0 / (1.0 + np.exp(-mean_logit))))
        else:
            weights = np.where(years == 2025, float(year_2025_weight), 1.0 - float(year_2025_weight))
            patient_probs.append(float(np.average(probs, weights=weights)))

    return np.asarray(patient_ids), np.asarray(patient_labels), np.asarray(patient_probs)


@torch.no_grad()
def predict_sample_probabilities(
    model: torch.nn.Module,
    dataset: TemperatureDataset,
    device: torch.device,
    *,
    soft_label: bool = False,
    multi_task: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    labels: list[int] = []
    probs: list[float] = []

    for idx in range(len(dataset)):
        x, y, _, sample_id = dataset[idx]
        x = x.unsqueeze(0).to(device)
        if multi_task:
            logits_cls, _ = model(x)
        else:
            logits_cls = model(x)

        if soft_label:
            prob = float(torch.sigmoid(logits_cls.squeeze(-1)).cpu().item())
        else:
            prob = float(F.softmax(logits_cls, dim=1)[:, 1].cpu().item())

        sample_ids.append(str(sample_id))
        labels.append(int(y.item()))
        probs.append(prob)

    return sample_ids, np.asarray(labels), np.asarray(probs)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    in_channels = 2 if args.region_attention else 1
    multi_task = args.multi_task or args.soft_label
    if args.model == "mobilenet":
        return MobileNetV3Small(
            num_classes=2,
            dropout=args.dropout,
            in_channels=in_channels,
            img_size=args.target_size,
            multi_task=multi_task,
            soft_label=args.soft_label,
            pretrained=not args.no_pretrained,
        )
    if args.model == "resnet50":
        return ResNet50Backbone(
            num_classes=2,
            dropout=args.dropout,
            in_channels=in_channels,
            img_size=args.target_size,
            multi_task=multi_task,
            soft_label=args.soft_label,
            pretrained=not args.no_pretrained,
        )
    if args.model == "deeper":
        return DeeperCNN(
            num_classes=2,
            dropout=args.dropout,
            in_channels=in_channels,
            img_size=args.target_size,
            multi_task=multi_task,
            soft_label=args.soft_label,
        )
    return SimpleCNN(
        num_classes=2,
        dropout=args.dropout,
        in_channels=in_channels,
        img_size=args.target_size,
        multi_task=multi_task,
        soft_label=args.soft_label,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/best_cnn_v3.pt"))
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--strategies",
        type=str,
        default="prob_mean,prob_max,logit_mean,year_weighted_mean",
        help="Comma-separated patient-level fusion strategies",
    )
    parser.add_argument(
        "--augmentation-strategy",
        type=str,
        default="baseline",
        choices=AUGMENTATION_STRATEGIES,
        help="Unused placeholder to match the single-image config surface.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    data = load_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    repo_root = Path(".").resolve()
    npy_dir = args.npy_dir if args.npy_dir.exists() else None
    target_size = (args.target_size, args.target_size)

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

    model = build_model(args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    multi_task = args.multi_task or args.soft_label
    val_ids, val_labels, val_probs = predict_sample_probabilities(
        model, val_dataset, device, soft_label=args.soft_label, multi_task=multi_task
    )
    test_ids, test_labels, test_probs = predict_sample_probabilities(
        model, test_dataset, device, soft_label=args.soft_label, multi_task=multi_task
    )

    val_metadata = build_sample_metadata_lookup(data["val"])
    test_metadata = build_sample_metadata_lookup(data["test"])

    rows: list[dict] = []
    for strategy in strategies:
        val_patient_ids, val_y, val_prob = aggregate_patient_predictions(
            val_ids,
            val_labels,
            val_probs,
            val_metadata,
            strategy=strategy,
            year_2025_weight=args.year_2025_weight,
        )
        test_patient_ids, test_y, test_prob = aggregate_patient_predictions(
            test_ids,
            test_labels,
            test_probs,
            test_metadata,
            strategy=strategy,
            year_2025_weight=args.year_2025_weight,
        )

        threshold, val_threshold_metrics = find_best_threshold(val_y, val_prob, metric="f1")
        test_preds = (test_prob >= threshold).astype(int)
        test_metrics = compute_metrics(test_y, test_preds, test_prob)

        row = {
            "strategy": strategy,
            "val_patients": int(len(val_patient_ids)),
            "test_patients": int(len(test_patient_ids)),
            "threshold": float(threshold),
            "val_auc_roc": float(val_threshold_metrics["auc_roc"]),
            "val_f1": float(val_threshold_metrics["f1"]),
            "test_auc_roc": float(test_metrics["auc_roc"]),
            "test_auc_pr": float(test_metrics["auc_pr"]),
            "test_acc": float(test_metrics["acc"]),
            "test_bal_acc": float(test_metrics["bal_acc"]),
            "test_f1": float(test_metrics["f1"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
        }
        rows.append(row)
        print(
            f"{strategy}: test_auc={row['test_auc_roc']:.4f}, "
            f"test_f1={row['test_f1']:.4f}, threshold={row['threshold']:.4f}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output / f"patient_level_fusion_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "model": args.model,
                "strategies": rows,
                "year_2025_weight": args.year_2025_weight,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
