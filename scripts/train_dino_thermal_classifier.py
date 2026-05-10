#!/usr/bin/env python3
"""Train a DINO-based thermal classifier with an MLP head.

Design:
1. Reuse the same masked thermal samples as the CNN pipeline.
2. Convert each 1-channel thermal map into a 3-channel DINO input.
3. Keep the DINO backbone frozen by default and train only a small MLP head.
4. Optionally add a severity regression head for multi-task learning.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_dino_thermal_features import load_dino_components, prepare_dino_input
from scripts.train_cnn_v3 import (
    EarlyStopping,
    MultiTaskLoss,
    SeverityWeightedLoss,
    SEVERITY_MULTIPLIER,
    TemperatureDataset,
    compute_class_weights,
    compute_metrics,
    compute_selection_score,
    find_best_threshold,
    load_data,
)


class DinoThermalDataset(Dataset):
    """Wrap masked thermal tensors into DINO processor-ready pixel values."""

    def __init__(self, base_dataset: TemperatureDataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y, sev, sample_id = self.base_dataset[idx]
        rgb = prepare_dino_input(x[0].cpu().numpy())
        encoded = self.processor(images=rgb, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return pixel_values, y, sev, sample_id


class DinoThermalClassifier(nn.Module):
    """Frozen-or-trainable DINO backbone plus lightweight task heads."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_dim: int,
        dropout: float,
        multi_task: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.multi_task = multi_task
        backbone_dim = int(getattr(self.backbone.config, "hidden_size"))

        self.head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier_head = nn.Linear(hidden_dim, 2)
        if multi_task:
            self.severity_head = nn.Linear(hidden_dim, 1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        hidden = self.head(features)
        logits_cls = self.classifier_head(hidden)
        if self.multi_task:
            logits_sev = self.severity_head(hidden)
            return logits_cls, logits_sev
        return logits_cls


def build_optimizer_param_groups(
    model: DinoThermalClassifier,
    *,
    head_lr: float,
    backbone_lr: float,
):
    """Split backbone and head params so optional fine-tuning can use a lower LR."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("backbone.") and p.requires_grad
    ]
    groups: list[dict] = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups


def train_epoch(
    model: DinoThermalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    multi_task: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for pixel_values, y, sev, _ in loader:
        pixel_values = pixel_values.to(device)
        y = y.to(device)
        sev = sev.to(device)
        optimizer.zero_grad()
        if multi_task:
            logits_cls, logits_sev = model(pixel_values)
            loss, _, _ = criterion(logits_cls, logits_sev, y, sev)
        else:
            logits = model(pixel_values)
            loss = criterion(logits, y, sev)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * pixel_values.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: DinoThermalClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    multi_task: bool,
) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_ids: list[str] = []

    for pixel_values, y, _, sample_ids in loader:
        pixel_values = pixel_values.to(device)
        logits_cls = model(pixel_values)[0] if multi_task else model(pixel_values)
        probs = torch.softmax(logits_cls, dim=1)[:, 1].cpu().numpy()
        preds = logits_cls.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())
        all_ids.extend(sample_ids)

    labels = np.asarray(all_labels, dtype=int)
    probs = np.asarray(all_probs, dtype=float)
    metrics = compute_metrics(labels, np.asarray(all_preds, dtype=int), probs)
    return metrics, all_ids, labels, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--model-id", type=str, default="facebook/dinov2-base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-severity", action="store_true")
    parser.add_argument("--multi-task", action="store_true")
    parser.add_argument("--lambda-sev", type=float, default=0.3)
    parser.add_argument("--severity-beta", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--selection-metric", type=str, default="auc_roc", choices=["auc_roc", "f1"])
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["f1", "bal_acc"])
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(".").resolve()
    use_mask = not args.no_mask

    data = load_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    npy_dir = args.npy_dir if args.npy_dir.exists() else None

    base_train = TemperatureDataset(
        data["train"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size=(args.target_size, args.target_size),
        use_mask=use_mask,
        augment=False,
        region_attention=False,
        npy_dir=npy_dir,
    )
    base_val = TemperatureDataset(
        data["val"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size=(args.target_size, args.target_size),
        use_mask=use_mask,
        augment=False,
        region_attention=False,
        npy_dir=npy_dir,
    )
    base_test = TemperatureDataset(
        data["test"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size=(args.target_size, args.target_size),
        use_mask=use_mask,
        augment=False,
        region_attention=False,
        npy_dir=npy_dir,
    )

    processor, backbone = load_dino_components(args.model_id, local_files_only=args.local_files_only)
    train_dataset = DinoThermalDataset(base_train, processor)
    val_dataset = DinoThermalDataset(base_val, processor)
    test_dataset = DinoThermalDataset(base_test, processor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DinoThermalClassifier(
        backbone=backbone,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        multi_task=args.multi_task,
    ).to(device)
    model.set_backbone_trainable(args.unfreeze_backbone)

    class_weights = compute_class_weights(base_train, device)
    if args.multi_task:
        criterion = MultiTaskLoss(
            class_weights,
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0} if args.no_severity else SEVERITY_MULTIPLIER,
            lambda_sev=args.lambda_sev,
            severity_beta=args.severity_beta,
        )
    else:
        criterion = SeverityWeightedLoss(
            class_weights,
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0} if args.no_severity else SEVERITY_MULTIPLIER,
        )

    param_groups = build_optimizer_param_groups(
        model,
        head_lr=args.lr,
        backbone_lr=args.backbone_lr,
    )
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_epochs=args.early_stop_min_epochs,
        min_delta=args.early_stop_min_delta,
    )

    best_selection_score = float("-inf")
    best_epoch = 0
    history: list[dict] = []
    ckpt_path = args.output / "best_dino_thermal_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            multi_task=args.multi_task,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, val_labels, val_probs = evaluate(
            model,
            val_loader,
            device,
            multi_task=args.multi_task,
        )
        scheduler.step()

        selection_score = compute_selection_score(val_metrics, args.selection_metric)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "selection_score": selection_score,
        })

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "selection_metric": args.selection_metric,
                    "selection_score": best_selection_score,
                    "model_id": args.model_id,
                },
                ckpt_path,
            )

        if early_stopper.step(epoch, val_metrics["auc_roc"]):
            break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, _, val_labels, val_probs = evaluate(model, val_loader, device, multi_task=args.multi_task)
    best_threshold, val_threshold_metrics = find_best_threshold(
        val_labels,
        val_probs,
        metric=args.threshold_metric,
    )
    test_metrics, _, test_labels, test_probs = evaluate(model, test_loader, device, multi_task=args.multi_task)
    test_threshold_preds = (test_probs >= best_threshold).astype(int)
    test_threshold_metrics = compute_metrics(test_labels, test_threshold_preds, test_probs)
    test_threshold_metrics["threshold"] = best_threshold
    test_threshold_metrics["optimized_metric"] = args.threshold_metric

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model": "dinov2_mlp",
        "model_id": args.model_id,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "use_face_mask": use_mask,
        "use_severity_weighting": not args.no_severity,
        "use_multi_task": args.multi_task,
        "target_size": args.target_size,
        "dropout": args.dropout,
        "head_hidden_dim": args.head_hidden_dim,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr if args.unfreeze_backbone else 0.0,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "selection_score": best_selection_score,
        "threshold_metric": args.threshold_metric,
        "best_threshold": best_threshold,
        "backbone_frozen": not args.unfreeze_backbone,
        "val_metrics": val_metrics,
        "val_threshold_metrics": val_threshold_metrics,
        "test_metrics": test_metrics,
        "test_threshold_metrics": test_threshold_metrics,
    }

    (args.output / f"dino_thermal_classifier_results_{timestamp}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output / f"dino_thermal_classifier_history_{timestamp}.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Best checkpoint: {ckpt_path}")
    print(f"Best validation {args.selection_metric}: {best_selection_score:.4f} at epoch {best_epoch}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"Threshold-tuned Test F1: {test_threshold_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
