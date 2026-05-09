#!/usr/bin/env python3
"""Train a cross-year patient-pair thermal CNN for ICAS classification.

Design:
1. Build patient pairs using exactly one 2024 image and one 2025 image.
2. Pass the two thermal maps through a shared-weight backbone.
3. Concatenate the two pooled feature vectors for classification.
4. Reweight each pair by 1 / pair_count(patient) so each patient contributes
   roughly equally when some patients have more valid cross-year pairs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from scripts.train_cnn_multimodal import build_thermal_backbone, extract_thermal_features
from scripts.train_cnn_v3 import (
    AUGMENTATION_STRATEGIES,
    EarlyStopping,
    MultiTaskLoss,
    MultiTaskSoftLabelLoss,
    SeverityWeightedLoss,
    SEVERITY_MULTIPLIER,
    TemperatureDataset,
    compute_metrics,
    compute_sample_weights,
    compute_selection_score,
    find_best_threshold,
    load_data,
)


def _normalize_year(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_cross_year_patient_pairs(
    samples: list[dict],
    year_a: int = 2024,
    year_b: int = 2025,
) -> list[dict]:
    """Build all valid cross-year pairs for each patient."""
    by_patient: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        patient_id = str(sample["canonical_patient_id"])
        year = _normalize_year(sample.get("year"))
        if year is None:
            continue
        by_patient[patient_id][year].append(sample)

    pairs: list[dict] = []
    for patient_id, grouped in by_patient.items():
        year_a_samples = sorted(grouped.get(year_a, []), key=lambda item: str(item["sample_id"]))
        year_b_samples = sorted(grouped.get(year_b, []), key=lambda item: str(item["sample_id"]))
        for sample_a in year_a_samples:
            for sample_b in year_b_samples:
                pair_id = f"{patient_id}_{sample_a['sample_id']}__{sample_b['sample_id']}"
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "canonical_patient_id": patient_id,
                        "sample_id_2024": str(sample_a["sample_id"]),
                        "sample_id_2025": str(sample_b["sample_id"]),
                        "sample_2024": sample_a,
                        "sample_2025": sample_b,
                        "year_a": year_a,
                        "year_b": year_b,
                    }
                )
    return pairs


def build_patient_pair_weight_lookup(pairs: list[dict]) -> dict[str, float]:
    counts = Counter(str(pair["canonical_patient_id"]) for pair in pairs)
    return {
        str(pair["pair_id"]): 1.0 / float(counts[str(pair["canonical_patient_id"])])
        for pair in pairs
    }


class PairTemperatureDataset(Dataset):
    """Wrap two per-sample thermal inputs into one patient pair example."""

    def __init__(
        self,
        pairs: list[dict],
        base_dataset: TemperatureDataset,
        pair_weights: dict[str, float],
    ):
        self.pairs = pairs
        self.base_dataset = base_dataset
        self.pair_weights = pair_weights
        self.sample_index = {
            str(sample["sample_id"]): idx for idx, sample in enumerate(self.base_dataset.samples)
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        pair = self.pairs[idx]
        x_2024, y_2024, sev_2024, _ = self.base_dataset[self.sample_index[pair["sample_id_2024"]]]
        x_2025, y_2025, sev_2025, _ = self.base_dataset[self.sample_index[pair["sample_id_2025"]]]
        if int(y_2024.item()) != int(y_2025.item()):
            raise ValueError(f"Mismatched labels within pair {pair['pair_id']}")
        if int(sev_2024.item()) != int(sev_2025.item()):
            raise ValueError(f"Mismatched severities within pair {pair['pair_id']}")
        pair_weight = self.pair_weights.get(str(pair["pair_id"]), 1.0)
        return (
            x_2024,
            x_2025,
            y_2024,
            sev_2024,
            torch.tensor(pair_weight, dtype=torch.float32),
            str(pair["pair_id"]),
        )


def compute_pair_class_weights(
    pairs: list[dict],
    labels: dict[str, int],
    device: torch.device,
) -> Tensor:
    pair_labels = np.array([labels.get(str(pair["canonical_patient_id"]), 0) for pair in pairs])
    n_neg = int((pair_labels == 0).sum())
    n_pos = int((pair_labels == 1).sum())
    weight_neg = (n_neg + n_pos) / (2 * n_neg) if n_neg > 0 else 1.0
    weight_pos = (n_neg + n_pos) / (2 * n_pos) if n_pos > 0 else 1.0
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=device)


class PairSeverityWeightedLoss(nn.Module):
    """Severity-weighted classification loss with extra patient-pair balancing."""

    def __init__(self, class_weights: Tensor, severity_weights: dict[int, float]):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.severity_weights = severity_weights
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        severities: Tensor,
        pair_weights: Tensor,
    ) -> Tensor:
        ce_loss = self.ce(logits, targets)
        sample_weights = compute_sample_weights(
            targets, severities, self.class_weights, self.severity_weights
        )
        return (ce_loss * sample_weights * pair_weights).mean()


class PairMultiTaskLoss(nn.Module):
    """Multi-task loss for pair inputs with patient balancing."""

    def __init__(
        self,
        class_weights: Tensor,
        severity_weights: dict[int, float],
        lambda_sev: float = 0.3,
        severity_beta: float = 0.25,
    ):
        super().__init__()
        self.base_loss = MultiTaskLoss(
            class_weights,
            severity_weights,
            lambda_sev=lambda_sev,
            severity_beta=severity_beta,
        )

    def forward(
        self,
        logits_cls: Tensor,
        logits_sev: Tensor,
        targets_cls: Tensor,
        targets_sev: Tensor,
        pair_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cls_losses = self.base_loss.cls_loss_fn(logits_cls, targets_cls)
        sample_weights = compute_sample_weights(
            targets_cls,
            targets_sev,
            self.base_loss.class_weights,
            self.base_loss.severity_weights,
        )
        loss_cls = (cls_losses * sample_weights * pair_weights).mean()

        pos_mask = targets_cls == 1
        if pos_mask.sum() > 0:
            pos_logits_sev = logits_sev[pos_mask].squeeze(-1)
            sev_targets = targets_sev[pos_mask]
            from scripts.train_cnn_v3 import severity_to_regression_target

            loss_sev = self.base_loss.sev_loss_fn(
                pos_logits_sev,
                severity_to_regression_target(sev_targets),
            )
        else:
            loss_sev = torch.tensor(0.0, device=logits_cls.device)

        total_loss = loss_cls + self.base_loss.lambda_sev * loss_sev
        return total_loss, loss_cls, loss_sev


class PairSharedBackboneClassifier(nn.Module):
    """Two-branch shared-backbone classifier with feature concatenation."""

    def __init__(
        self,
        model_name: str,
        dropout: float,
        in_channels: int,
        img_size: int,
        multi_task: bool = False,
        soft_label: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.multi_task = multi_task
        self.soft_label = soft_label
        self.backbone = build_thermal_backbone(
            model_name=model_name,
            dropout=dropout,
            in_channels=in_channels,
            img_size=img_size,
            pretrained=pretrained,
        )
        self.feature_dim = self._infer_feature_dim(in_channels, img_size)
        cls_out = 1 if soft_label else 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, cls_out),
        )
        if multi_task:
            self.severity_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def _infer_feature_dim(self, in_channels: int, img_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            features = extract_thermal_features(self.backbone, dummy)
        return int(features.shape[1])

    def _extract_pair_features(self, x_2024: Tensor, x_2025: Tensor) -> Tensor:
        f_2024 = extract_thermal_features(self.backbone, x_2024)
        f_2025 = extract_thermal_features(self.backbone, x_2025)
        return torch.cat([f_2024, f_2025], dim=1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        if hasattr(self.backbone, "set_backbone_trainable"):
            self.backbone.set_backbone_trainable(trainable)

    def forward(self, x_2024: Tensor, x_2025: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        features = self._extract_pair_features(x_2024, x_2025)
        logits_cls = self.classifier(features)
        if self.multi_task:
            logits_sev = self.severity_head(features)
            return logits_cls, logits_sev
        return logits_cls


def build_pair_datasets(
    data: dict,
    *,
    repo_root: Path,
    masks_dir: Path,
    target_size: tuple[int, int],
    use_mask: bool,
    augment: bool,
    augmentation_strategy: str,
    region_attention: bool,
    npy_dir: Path | None,
) -> tuple[PairTemperatureDataset, PairTemperatureDataset, PairTemperatureDataset]:
    train_pairs = build_cross_year_patient_pairs(data["train"])
    val_pairs = build_cross_year_patient_pairs(data["val"])
    test_pairs = build_cross_year_patient_pairs(data["test"])

    train_base = TemperatureDataset(
        data["train"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        masks_dir,
        target_size,
        use_mask=use_mask,
        augment=augment,
        augmentation_strategy=augmentation_strategy,
        region_attention=region_attention,
        npy_dir=npy_dir,
    )
    val_base = TemperatureDataset(
        data["val"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        masks_dir,
        target_size,
        use_mask=use_mask,
        region_attention=region_attention,
        npy_dir=npy_dir,
    )
    test_base = TemperatureDataset(
        data["test"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        masks_dir,
        target_size,
        use_mask=use_mask,
        region_attention=region_attention,
        npy_dir=npy_dir,
    )

    return (
        PairTemperatureDataset(train_pairs, train_base, build_patient_pair_weight_lookup(train_pairs)),
        PairTemperatureDataset(val_pairs, val_base, build_patient_pair_weight_lookup(val_pairs)),
        PairTemperatureDataset(test_pairs, test_base, build_patient_pair_weight_lookup(test_pairs)),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    multi_task: bool = False,
    soft_label: bool = False,
    grad_clip: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    for x_2024, x_2025, y, sev, pair_weights, _ in loader:
        x_2024 = x_2024.to(device)
        x_2025 = x_2025.to(device)
        y = y.to(device)
        sev = sev.to(device)
        pair_weights = pair_weights.to(device)
        optimizer.zero_grad()
        if soft_label:
            logits_cls, logits_sev = model(x_2024, x_2025)
            loss, _, _ = criterion(logits_cls, logits_sev, y, sev)
        elif multi_task:
            logits_cls, logits_sev = model(x_2024, x_2025)
            loss, _, _ = criterion(logits_cls, logits_sev, y, sev, pair_weights)
        else:
            logits = model(x_2024, x_2025)
            loss = criterion(logits, y, sev, pair_weights)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x_2024.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    multi_task: bool = False,
    soft_label: bool = False,
) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_ids: list[str] = []

    for x_2024, x_2025, y, _, _, pair_ids in loader:
        x_2024 = x_2024.to(device)
        x_2025 = x_2025.to(device)
        if multi_task or soft_label:
            outputs = model(x_2024, x_2025)
            logits_cls = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            logits_cls = model(x_2024, x_2025)

        if soft_label:
            probs = torch.sigmoid(logits_cls.squeeze(-1)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
        else:
            probs = F.softmax(logits_cls, dim=1)[:, 1].cpu().numpy()
            preds = logits_cls.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())
        all_ids.extend(list(pair_ids))

    labels = np.array(all_labels)
    probs = np.array(all_probs)
    metrics = compute_metrics(labels, np.array(all_preds), probs)
    return metrics, all_ids, labels, probs


def main() -> None:
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="mobilenet", choices=["simple", "deeper", "mobilenet"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-severity", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--augmentation-strategy",
        type=str,
        default="baseline",
        choices=AUGMENTATION_STRATEGIES,
    )
    parser.add_argument("--region-attention", action="store_true")
    parser.add_argument("--multi-task", action="store_true")
    parser.add_argument("--soft-label", action="store_true")
    parser.add_argument("--lambda-sev", type=float, default=0.3)
    parser.add_argument("--severity-beta", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["f1", "bal_acc"])
    parser.add_argument("--selection-metric", type=str, default="f1", choices=["auc_roc", "f1"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(".").resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    train_dataset, val_dataset, test_dataset = build_pair_datasets(
        data,
        repo_root=repo_root,
        masks_dir=args.masks_dir,
        target_size=(args.target_size, args.target_size),
        use_mask=not args.no_mask,
        augment=args.augment,
        augmentation_strategy=args.augmentation_strategy,
        region_attention=args.region_attention,
        npy_dir=args.npy_dir if args.npy_dir.exists() else None,
    )

    print(f"Train pairs: {len(train_dataset)}")
    print(f"Val pairs: {len(val_dataset)}")
    print(f"Test pairs: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PairSharedBackboneClassifier(
        model_name=args.model,
        dropout=args.dropout,
        in_channels=2 if args.region_attention else 1,
        img_size=args.target_size,
        multi_task=args.multi_task or args.soft_label,
        soft_label=args.soft_label,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = compute_pair_class_weights(train_dataset.pairs, data["labels"], device)
    multi_task = args.multi_task or args.soft_label

    if args.soft_label:
        criterion = MultiTaskSoftLabelLoss(lambda_sev=args.lambda_sev)
    elif multi_task:
        criterion = PairMultiTaskLoss(
            class_weights,
            SEVERITY_MULTIPLIER if not args.no_severity else {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            lambda_sev=args.lambda_sev,
            severity_beta=args.severity_beta,
        )
    else:
        criterion = PairSeverityWeightedLoss(
            class_weights,
            SEVERITY_MULTIPLIER if not args.no_severity else {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_epochs=args.early_stop_min_epochs,
        min_delta=args.early_stop_min_delta,
    )

    best_selection_score = float("-inf")
    best_epoch = 0
    history: list[dict] = []
    backbone_trainable = True
    checkpoint_path = args.output / "best_cnn_pair.pt"

    for epoch in range(1, args.epochs + 1):
        should_train_backbone = not (
            args.model == "mobilenet"
            and args.freeze_backbone_epochs > 0
            and epoch <= args.freeze_backbone_epochs
        )
        if args.model == "mobilenet" and should_train_backbone != backbone_trainable:
            model.set_backbone_trainable(should_train_backbone)
            backbone_trainable = should_train_backbone

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            multi_task=multi_task,
            soft_label=args.soft_label,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, val_labels, val_probs = evaluate(
            model,
            val_loader,
            device,
            multi_task=multi_task,
            soft_label=args.soft_label,
        )
        scheduler.step()
        selection_score = compute_selection_score(val_metrics, args.selection_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "selection_score": selection_score,
            }
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "selection_metric": args.selection_metric,
                    "selection_score": best_selection_score,
                    "val_auc": val_metrics["auc_roc"],
                },
                checkpoint_path,
            )
        if early_stopper.step(epoch, val_metrics["auc_roc"]):
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)["model_state_dict"])
    val_metrics, _, val_labels, val_probs = evaluate(model, val_loader, device, multi_task=multi_task, soft_label=args.soft_label)
    best_threshold, val_threshold_metrics = find_best_threshold(val_labels, val_probs, metric=args.threshold_metric)
    test_metrics, _, test_labels, test_probs = evaluate(model, test_loader, device, multi_task=multi_task, soft_label=args.soft_label)
    test_threshold_preds = (test_probs >= best_threshold).astype(int)
    test_threshold_metrics = compute_metrics(test_labels, test_threshold_preds, test_probs)
    test_threshold_metrics["threshold"] = best_threshold
    test_threshold_metrics["optimized_metric"] = args.threshold_metric

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "cross_year_patient_pair",
        "pairing_rule": "all valid 2024 x 2025 pairs within patient",
        "model": args.model,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "use_face_mask": not args.no_mask,
        "use_augmentation": args.augment,
        "use_region_attention": args.region_attention,
        "use_multi_task": multi_task,
        "use_soft_label": args.soft_label,
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "test_pairs": len(test_dataset),
        "test_metrics": test_metrics,
        "val_threshold_metrics": val_threshold_metrics,
        "test_threshold_metrics": test_threshold_metrics,
    }
    (args.output / f"cnn_pair_results_{timestamp}.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    (args.output / f"cnn_pair_history_{timestamp}.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
