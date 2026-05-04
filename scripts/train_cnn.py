#!/usr/bin/env python3
"""Train CNN on raw temperature matrices for ICAS classification.

This script:
1. Loads temperature CSV files and converts them to 2D matrices
2. Excludes samples listed in configs/excluded_samples.json
3. Uses patient-level train/val/test split from configs/data_split.json
4. Trains a simple CNN for binary classification
5. Evaluates on validation and test sets

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --epochs 100 --batch-size 32
    python scripts/train_cnn.py --no-train  # Only evaluate
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def parse_temperature_csv(path: Path) -> np.ndarray:
    """Parse temperature CSV into 2D matrix.
    
    File layout:
        line 1-2: metadata header and values
        line 3: blank
        line 4: column index header
        line 5+: data rows
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

    return np.asarray(rows, dtype=np.float32)


def normalize_temperature(matrix: np.ndarray) -> np.ndarray:
    """Normalize temperature matrix to [0, 1] range."""
    min_val = matrix.min()
    max_val = matrix.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)


def resize_matrix(matrix: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize matrix to target size using bilinear interpolation."""
    import cv2
    return cv2.resize(matrix, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


class TemperatureDataset(Dataset):
    """PyTorch Dataset for temperature matrices."""

    def __init__(
        self,
        samples: list[dict],
        labels: dict[str, int],
        repo_root: Path,
        target_size: tuple[int, int] = (128, 128),
        transform: Optional[callable] = None,
    ):
        self.samples = samples
        self.labels = labels
        self.repo_root = repo_root
        self.target_size = target_size
        self.transform = transform
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        patient_id = sample["canonical_patient_id"]
        temp_path = self.repo_root / sample["temperature_path"]

        if sample_id in self._cache:
            matrix = self._cache[sample_id]
        else:
            try:
                matrix = parse_temperature_csv(temp_path)
                matrix = normalize_temperature(matrix)
                matrix = resize_matrix(matrix, self.target_size)
                self._cache[sample_id] = matrix
            except Exception as e:
                print(f"Warning: Failed to load {sample_id}: {e}")
                matrix = np.zeros(self.target_size, dtype=np.float32)

        if self.transform:
            matrix = self.transform(matrix)

        x = torch.from_numpy(matrix).unsqueeze(0).float()
        y = torch.tensor(self.labels.get(patient_id, 0), dtype=torch.long)

        return x, y, sample_id


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
    """Deeper CNN with residual-style connections."""

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
    """Load excluded sample IDs from config."""
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
) -> dict:
    """Load all data and return split samples with labels."""
    manifest = pd.read_csv(manifest_path)
    clinical = pd.read_csv(clinical_path, dtype=str)
    clinical["label"] = pd.to_numeric(clinical["label"], errors="coerce")
    
    split = json.loads(split_path.read_text(encoding="utf-8"))
    excluded_ids = load_excluded_ids(excluded_path)

    clinical_valid = clinical.dropna(subset=["label"])
    labels = dict(zip(clinical_valid["canonical_patient_id"], clinical_valid["label"].astype(int)))

    def filter_samples(patient_ids: list[str]) -> list[dict]:
        samples = []
        for _, row in manifest.iterrows():
            sid = row["sample_id"]
            pid = row["canonical_patient_id"]
            if sid in excluded_ids:
                continue
            if pid in patient_ids:
                samples.append(row.to_dict())
        return samples

    return {
        "train": filter_samples(split["train_patient_ids"]),
        "val": filter_samples(split["val_patient_ids"]),
        "test": filter_samples(split["test_patient_ids"]),
        "labels": labels,
        "split_info": split.get("summary", {}),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics."""
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
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[Tensor] = None,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        if class_weights is not None:
            weight = class_weights[y]
            loss = nn.CrossEntropyLoss(reduction="none")(logits, y)
            loss = (loss * weight).mean()
        else:
            loss = criterion(logits, y)
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
    """Evaluate model, return metrics and sample IDs."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []

    for x, y, sample_ids in loader:
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


def compute_class_weights(dataset: TemperatureDataset, device: torch.device) -> Tensor:
    """Compute class weights for imbalanced data."""
    labels = [dataset.labels.get(s["canonical_patient_id"], 0) for s in dataset.samples]
    labels = np.array(labels)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    
    weight_neg = (n_neg + n_pos) / (2 * n_neg) if n_neg > 0 else 1.0
    weight_pos = (n_neg + n_pos) / (2 * n_pos) if n_pos > 0 else 1.0
    
    weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=device)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "deeper"])
    parser.add_argument("--no-train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--checkpoint", type=Path, help="Load checkpoint for evaluation")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    repo_root = Path(".").resolve()

    print("Loading data...")
    data = load_data(args.manifest, args.clinical, args.split, args.excluded)
    print(f"  Train: {len(data['train'])} samples")
    print(f"  Val:   {len(data['val'])} samples")
    print(f"  Test:  {len(data['test'])} samples")

    target_size = (args.target_size, args.target_size)

    train_dataset = TemperatureDataset(data["train"], data["labels"], repo_root, target_size)
    val_dataset = TemperatureDataset(data["val"], data["labels"], repo_root, target_size)
    test_dataset = TemperatureDataset(data["test"], data["labels"], repo_root, target_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "simple":
        model = SimpleCNN(num_classes=2, dropout=args.dropout)
    else:
        model = DeeperCNN(num_classes=2, dropout=args.dropout)
    model = model.to(device)

    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    best_epoch = 0
    history: list[dict] = []

    if not args.no_train:
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                }, args.output / "best_cnn.pt")

            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                      f"val_auc={val_metrics['auc_roc']:.4f} "
                      f"val_f1={val_metrics['f1']:.4f}")

        print(f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

        checkpoint = torch.load(args.output / "best_cnn.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    test_metrics, test_ids = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print("Test Results")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "model": args.model,
        "epochs": args.epochs if not args.no_train else "loaded",
        "best_epoch": best_epoch if not args.no_train else "N/A",
        "target_size": args.target_size,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "test_metrics": test_metrics,
    }
    
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / f"cnn_results_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    if history:
        history_path = args.output / f"cnn_history_{timestamp}.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
