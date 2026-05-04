#!/usr/bin/env python3
"""Train CNN on masked temperature matrices for ICAS classification.

Key features:
1. Uses pre-computed face masks from face_roi_annotation.py
2. Only uses temperature values within face region
3. Severity-weighted loss: positive samples weighted by stenosis severity

Usage:
    python scripts/train_cnn_v2.py
    python scripts/train_cnn_v2.py --epochs 100 --device cuda
    python scripts/train_cnn_v2.py --augment
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

SEVERITY_MULTIPLIER = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}


def parse_temperature_csv(path: Path) -> np.ndarray:
    """Parse temperature CSV into 2D matrix."""
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

    return np.asarray(rows, dtype=np.float32)


def load_face_mask(mask_path: Path) -> np.ndarray:
    """Load face mask at original resolution."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.float32)


TEMP_RANGE_MIN = 15.0
TEMP_RANGE_MAX = 40.0


class TemperatureAugmentation:
    """Data augmentation for temperature heatmaps.

    Applies temperature-level and geometric augmentations.
    Operates on normalized [0, 1] matrices.
    """

    def __init__(
        self,
        temp_offset_range: tuple[float, float] = (-2.0, 2.0),
        temp_scale_range: tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.2,
        rotation_range: float = 5.0,
        translation_range: int = 5,
    ):
        self.temp_offset_range = temp_offset_range
        self.temp_scale_range = temp_scale_range
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        offset = np.random.uniform(*self.temp_offset_range)
        scale = np.random.uniform(*self.temp_scale_range)
        offset_norm = offset / (TEMP_RANGE_MAX - TEMP_RANGE_MIN)

        augmented = matrix * scale + offset_norm

        if self.noise_std > 0:
            augmented = augmented + np.random.randn(*augmented.shape).astype(np.float32) * self.noise_std

        if self.rotation_range > 0 or self.translation_range > 0:
            h, w = augmented.shape
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            if self.translation_range > 0:
                tx = np.random.randint(-self.translation_range, self.translation_range + 1)
                ty = np.random.randint(-self.translation_range, self.translation_range + 1)
                M[:, 2] += [tx, ty]
            augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return augmented.clip(0.0, 1.0).astype(np.float32)


def apply_face_mask(
    temp_matrix: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply face mask to temperature matrix with fixed-range normalization.

    Steps:
    1. Resize mask to temperature matrix size (align in temp space)
    2. Normalize temperature using fixed range [20, 45] °C
    3. Mask out non-face regions
    4. Resize result to target_size
    """
    temp_h, temp_w = temp_matrix.shape

    if mask is not None and mask.shape[0] != temp_h or mask.shape[1] != temp_w:
        mask = cv2.resize(mask, (temp_w, temp_h), interpolation=cv2.INTER_NEAREST)

    temp_norm = np.clip(
        (temp_matrix - TEMP_RANGE_MIN) / (TEMP_RANGE_MAX - TEMP_RANGE_MIN), 0.0, 1.0
    )

    if mask is not None:
        masked_temp = temp_norm * mask + fill_value * (1 - mask)
    else:
        masked_temp = temp_norm

    masked_temp = cv2.resize(masked_temp, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    return masked_temp.astype(np.float32)


class SeverityWeightedLoss(nn.Module):
    """Binary cross-entropy with severity weighting for positive samples.

    Loss = -[ w_neg * y*log(p) + w_pos * severity_weight * (1-y)*log(1-p) ]

    For positive samples (y=1), the loss is weighted by severity:
        - stenosis=0 (negative): weight = class_weight[0]
        - stenosis=1 (mild):    weight = class_weight[1] * 1.0
        - stenosis=2 (moderate): weight = class_weight[1] * 2.0
        - stenosis=3 (severe):   weight = class_weight[1] * 3.0
    """

    def __init__(self, class_weights: Tensor, severity_weights: dict[int, float]):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.severity_weights = severity_weights

    def forward(self, logits: Tensor, targets: Tensor, severities: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=1)
        probs_pos = probs[:, 1].clamp(min=1e-7, max=1 - 1e-7)

        ce_loss = -targets.float() * torch.log(probs_pos) - (1 - targets.float()) * torch.log(1 - probs_pos)

        sample_weights = torch.ones_like(ce_loss)
        for i in range(len(targets)):
            if targets[i] == 1:
                sev = int(severities[i].item()) if not torch.isnan(severities[i]) else 1
                sev_weight = self.severity_weights.get(sev, 1.0)
                sample_weights[i] = self.class_weights[1] * sev_weight
            else:
                sample_weights[i] = self.class_weights[0]

        return (ce_loss * sample_weights).mean()


class TemperatureDataset(Dataset):
    """Dataset with face-masked temperature matrices."""

    def __init__(
        self,
        samples: list[dict],
        annotations: dict[str, dict],
        labels: dict[str, int],
        severities: dict[str, int],
        repo_root: Path,
        masks_dir: Path,
        target_size: tuple[int, int] = (128, 128),
        use_mask: bool = True,
        augment: bool = False,
    ):
        self.samples = samples
        self.annotations = annotations
        self.labels = labels
        self.severities = severities
        self.repo_root = repo_root
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.use_mask = use_mask
        self.augment = augment
        self.augmentation = TemperatureAugmentation() if augment else None
        self._temp_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, str]:
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        patient_id = sample["canonical_patient_id"]
        temp_path = self.repo_root / sample["temperature_path"]

        if sample_id in self._temp_cache:
            temp_matrix = self._temp_cache[sample_id]
        else:
            try:
                temp_matrix = parse_temperature_csv(temp_path)
                self._temp_cache[sample_id] = temp_matrix
            except Exception as e:
                print(f"Warning: Failed to load {sample_id}: {e}")
                temp_matrix = np.zeros((1024, 1280), dtype=np.float32)

        if self.use_mask:
            mask_path = self.masks_dir / f"{sample_id}_face.png"
            if sample_id in self._mask_cache:
                mask = self._mask_cache[sample_id]
            else:
                mask = load_face_mask(mask_path)
                self._mask_cache[sample_id] = mask

            masked_temp = apply_face_mask(temp_matrix, mask, self.target_size, fill_value=0.0)
        else:
            temp_resized = cv2.resize(temp_matrix, (self.target_size[1], self.target_size[0]))
            masked_temp = np.clip(
                (temp_resized - TEMP_RANGE_MIN) / (TEMP_RANGE_MAX - TEMP_RANGE_MIN), 0.0, 1.0
            ).astype(np.float32)

        if self.augmentation is not None:
            masked_temp = self.augmentation(masked_temp)

        x = torch.from_numpy(masked_temp).unsqueeze(0).float()
        y = torch.tensor(self.labels.get(patient_id, 0), dtype=torch.long)
        sev = torch.tensor(self.severities.get(patient_id, 0), dtype=torch.float)

        return x, y, sev, sample_id


class SimpleCNN(nn.Module):
    """Simple CNN for temperature matrix classification."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self._init_fc(num_classes)

    def _init_fc(self, num_classes: int):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            dummy = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
            dummy = self.pool(F.relu(self.bn3(self.conv3(dummy))))
            self.flat_size = dummy.numel()
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DeeperCNN(nn.Module):
    """Deeper CNN with more capacity."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self._init_classifier(num_classes, dropout)

    def _init_classifier(self, num_classes: int, dropout: float):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            dummy = self.features(dummy)
            self.flat_size = dummy.numel()
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


def load_data(
    manifest_path: Path,
    clinical_path: Path,
    split_path: Path,
    excluded_path: Path | None,
    annotations_path: Path,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    clinical = pd.read_csv(clinical_path, dtype=str)
    clinical["label"] = pd.to_numeric(clinical["label"], errors="coerce")
    clinical["stenosis_multiclass"] = pd.to_numeric(clinical["stenosis_multiclass"], errors="coerce")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    excluded_ids = load_excluded_ids(excluded_path)

    annotations_data = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations = {s["sample_id"]: s for s in annotations_data.get("samples", [])}

    clinical_valid = clinical.dropna(subset=["label"])
    labels = dict(zip(clinical_valid["canonical_patient_id"], clinical_valid["label"].astype(int)))
    severities = dict(zip(
        clinical_valid["canonical_patient_id"],
        clinical_valid["stenosis_multiclass"].fillna(0).astype(int)
    ))

    def filter_samples(patient_ids: list[str]) -> list[dict]:
        samples = []
        for _, row in manifest.iterrows():
            sid = row["sample_id"]
            pid = row["canonical_patient_id"]
            if sid in excluded_ids:
                continue
            if pid not in patient_ids:
                continue
            if sid not in annotations:
                continue
            if annotations[sid].get("status") != "ok":
                continue
            samples.append(row.to_dict())
        return samples

    return {
        "train": filter_samples(split["train_patient_ids"]),
        "val": filter_samples(split["val_patient_ids"]),
        "test": filter_samples(split["test_patient_ids"]),
        "labels": labels,
        "severities": severities,
        "annotations": annotations,
        "split_info": split.get("summary", {}),
    }


def compute_class_weights(dataset: TemperatureDataset, device: torch.device) -> Tensor:
    labels = [dataset.labels.get(s["canonical_patient_id"], 0) for s in dataset.samples]
    labels = np.array(labels)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    weight_neg = (n_neg + n_pos) / (2 * n_neg) if n_neg > 0 else 1.0
    weight_pos = (n_neg + n_pos) / (2 * n_pos) if n_pos > 0 else 1.0
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SeverityWeightedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y, sev, _ in loader:
        x, y, sev = x.to(device), y.to(device), sev.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y, sev)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, list[str]]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []

    for x, y, _, sample_ids in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(y.numpy())
        all_ids.extend(sample_ids)

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    return metrics, all_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "deeper"])
    parser.add_argument("--no-mask", action="store_true", help="Disable face masking")
    parser.add_argument("--no-severity", action="store_true", help="Disable severity weighting")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Use face mask: {not args.no_mask}")
    print(f"Use severity weighting: {not args.no_severity}")
    print(f"Use augmentation: {args.augment}")

    repo_root = Path(".").resolve()

    print("\nLoading data...")
    data = load_data(
        args.manifest, args.clinical, args.split, args.excluded, args.annotations
    )
    print(f"  Train: {len(data['train'])} samples")
    print(f"  Val:   {len(data['val'])} samples")
    print(f"  Test:  {len(data['test'])} samples")

    target_size = (args.target_size, args.target_size)
    use_mask = not args.no_mask

    train_dataset = TemperatureDataset(
        data["train"], data["annotations"], data["labels"], data["severities"],
        repo_root, args.masks_dir, target_size, use_mask=use_mask, augment=args.augment
    )
    val_dataset = TemperatureDataset(
        data["val"], data["annotations"], data["labels"], data["severities"],
        repo_root, args.masks_dir, target_size, use_mask=use_mask
    )
    test_dataset = TemperatureDataset(
        data["test"], data["annotations"], data["labels"], data["severities"],
        repo_root, args.masks_dir, target_size, use_mask=use_mask
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "simple":
        model = SimpleCNN(num_classes=2, dropout=args.dropout)
    else:
        model = DeeperCNN(num_classes=2, dropout=args.dropout)
    model = model.to(device)

    class_weights = compute_class_weights(train_dataset, device)

    if args.no_severity:
        criterion = SeverityWeightedLoss(class_weights, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    else:
        criterion = SeverityWeightedLoss(class_weights, SEVERITY_MULTIPLIER)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    best_epoch = 0
    history: list[dict] = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_val_auc,
            }, args.output / "best_cnn_v2.pt")

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                  f"val_auc={val_metrics['auc_roc']:.4f} "
                  f"val_f1={val_metrics['f1']:.4f}")

    print(f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(args.output / "best_cnn_v2.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, _ = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print("Test Results")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model": args.model,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "use_face_mask": use_mask,
        "use_severity_weighting": not args.no_severity,
        "use_augmentation": args.augment,
        "target_size": args.target_size,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "test_metrics": test_metrics,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / f"cnn_v2_results_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    history_path = args.output / f"cnn_v2_history_{timestamp}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
