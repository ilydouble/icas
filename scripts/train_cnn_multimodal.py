#!/usr/bin/env python3
"""Train a sample-level thermal CNN with patient-level ASR and clinical features.

Design:
1. Keep the thermal branch at the single-image sample level.
2. Broadcast patient-level ASR and clinical features to each sample.
3. Reweight each sample by 1 / image_count(patient) so patients contribute
   roughly equally even when some have more thermal images.
4. Fuse thermal and structured representations before the ICAS/severity heads.

Usage:
    python scripts/train_cnn_multimodal.py --device cuda
    python scripts/train_cnn_multimodal.py --model deeper --multi-task
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_cnn_v3 import (
    AUGMENTATION_STRATEGIES,
    EarlyStopping,
    MobileNetV3Small,
    SimpleCNN,
    DeeperCNN,
    TemperatureDataset,
    compute_class_weights,
    compute_metrics,
    compute_selection_score,
    find_best_threshold,
    load_data,
    severity_to_regression_target,
)

warnings.filterwarnings("ignore")

ASR_TOP9_FEATURES = [
    "asr_speech_rate_min",
    "asr_chars_per_sentence_mean",
    "asr_chars_per_second",
    "asr_emotion_median",
    "asr_long_pause_sentence_ratio",
    "asr_pause_sentence_ratio",
    "asr_sentence_duration_ms_mean",
    "asr_sentence_duration_ms_min",
    "asr_silence_duration_ms_mean",
]

CLINICAL_TOP3_FEATURES = [
    "waist_hip_ratio",
    "gender_encoded",
    "height",
]

SEVERITY_MULTIPLIER = {0: 1.0, 1: 1.0, 2: 1.5, 3: 2.0}


def load_structured_feature_table(
    asr_subset_path: Path,
    clinical_subset_path: Path,
    asr_features: list[str] | None = None,
    clinical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Load patient-level ASR + clinical features for multimodal fusion."""
    asr_features = asr_features or ASR_TOP9_FEATURES
    clinical_features = clinical_features or CLINICAL_TOP3_FEATURES

    asr_df = pd.read_csv(asr_subset_path)
    clinical_df = pd.read_csv(clinical_subset_path)

    if "clinical_match_status" in asr_df.columns:
        asr_df = asr_df.loc[asr_df["clinical_match_status"] == "matched"].copy()

    asr_keep = ["canonical_patient_id", "label", "stenosis_multiclass", *asr_features]
    clinical_keep = ["canonical_patient_id", *clinical_features]

    merged = asr_df.loc[:, asr_keep].merge(
        clinical_df.loc[:, clinical_keep],
        on="canonical_patient_id",
        how="inner",
    )
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
    merged["stenosis_multiclass"] = pd.to_numeric(merged["stenosis_multiclass"], errors="coerce")
    feature_cols = asr_features + clinical_features
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["label", *feature_cols]).reset_index(drop=True)
    return merged.loc[:, ["canonical_patient_id", "label", "stenosis_multiclass", *feature_cols]]


def build_patient_feature_lookup(
    structured_df: pd.DataFrame,
    asr_features: list[str] | None = None,
    clinical_features: list[str] | None = None,
) -> dict[str, np.ndarray]:
    asr_features = asr_features or ASR_TOP9_FEATURES
    clinical_features = clinical_features or CLINICAL_TOP3_FEATURES
    feature_cols = asr_features + clinical_features
    lookup: dict[str, np.ndarray] = {}
    for _, row in structured_df.iterrows():
        lookup[str(row["canonical_patient_id"])] = row.loc[feature_cols].to_numpy(dtype=np.float32)
    return lookup


def build_patient_sample_weight_lookup(samples: list[dict]) -> dict[str, float]:
    counts = Counter(str(sample["canonical_patient_id"]) for sample in samples)
    return {
        str(sample["sample_id"]): 1.0 / float(counts[str(sample["canonical_patient_id"])])
        for sample in samples
    }


def broadcast_patient_features_to_samples(
    samples: list[dict],
    patient_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sample_features: dict[str, np.ndarray] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        patient_id = str(sample["canonical_patient_id"])
        if patient_id not in patient_features:
            raise KeyError(f"Missing structured features for patient {patient_id}")
        sample_features[sample_id] = patient_features[patient_id]
    return sample_features


def load_initial_thermal_weights(
    model: "ThermalStructuredFusionModel",
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    """Load matching thermal-branch weights from a prior thermal checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.thermal_backbone.state_dict()

    matched: dict[str, Tensor] = {}
    for key, value in source_state.items():
        if key.startswith("thermal_backbone."):
            candidate_key = key.split("thermal_backbone.", 1)[1]
        else:
            candidate_key = key
        if candidate_key in target_state and tuple(target_state[candidate_key].shape) == tuple(value.shape):
            matched[candidate_key] = value

    if not matched:
        return 0

    target_state.update(matched)
    model.thermal_backbone.load_state_dict(target_state)
    return len(matched)


class MultimodalTemperatureDataset(Dataset):
    """Wrap the thermal dataset with patient-level structured features."""

    def __init__(
        self,
        base_dataset: TemperatureDataset,
        sample_features: dict[str, np.ndarray],
        sample_weights: dict[str, float],
    ):
        self.base_dataset = base_dataset
        self.sample_features = sample_features
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        x_img, y, sev, sample_id = self.base_dataset[idx]
        features = self.sample_features[str(sample_id)]
        sample_weight = self.sample_weights.get(str(sample_id), 1.0)
        return (
            x_img,
            torch.from_numpy(features).float(),
            y,
            sev,
            torch.tensor(sample_weight, dtype=torch.float32),
            sample_id,
        )


def extract_thermal_features(model: nn.Module, x_img: Tensor) -> Tensor:
    """Expose the penultimate thermal representation from the chosen CNN."""
    if isinstance(model, MobileNetV3Small):
        return model.backbone(x_img)
    if isinstance(model, DeeperCNN):
        z = model.features(x_img)
        z = z.view(z.size(0), -1)
        return model.shared(z)

    z = model.pool(F.relu(model.bn1(model.conv1(x_img))))
    z = model.pool(F.relu(model.bn2(model.conv2(z))))
    z = model.pool(F.relu(model.bn3(model.conv3(z))))
    z = z.view(z.size(0), -1)
    return model.dropout(F.relu(model.fc1(z)))


class ThermalStructuredFusionModel(nn.Module):
    """Fuse single-image thermal features with structured ASR + clinical inputs."""

    def __init__(
        self,
        model_name: str,
        structured_dim: int,
        dropout: float,
        in_channels: int,
        img_size: int,
        multi_task: bool,
        pretrained: bool = True,
        structured_hidden: int = 32,
        fusion_hidden: int = 128,
    ):
        super().__init__()
        self.multi_task = multi_task
        if model_name == "mobilenet":
            self.thermal_backbone = MobileNetV3Small(
                num_classes=2,
                dropout=dropout,
                in_channels=in_channels,
                img_size=img_size,
                multi_task=False,
                soft_label=False,
                pretrained=pretrained,
            )
        elif model_name == "deeper":
            self.thermal_backbone = DeeperCNN(
                num_classes=2,
                dropout=dropout,
                in_channels=in_channels,
                img_size=img_size,
                multi_task=False,
                soft_label=False,
            )
        else:
            self.thermal_backbone = SimpleCNN(
                num_classes=2,
                dropout=dropout,
                in_channels=in_channels,
                img_size=img_size,
                multi_task=False,
                soft_label=False,
            )

        with torch.no_grad():
            dummy_img = torch.zeros(1, in_channels, img_size, img_size)
            thermal_dim = int(extract_thermal_features(self.thermal_backbone, dummy_img).shape[1])

        self.structured_mlp = nn.Sequential(
            nn.Linear(structured_dim, structured_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(thermal_dim + structured_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier_head = nn.Linear(fusion_hidden, 2)
        if self.multi_task:
            self.severity_head = nn.Linear(fusion_hidden, 1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.thermal_backbone.parameters():
            param.requires_grad = trainable
        if isinstance(self.thermal_backbone, MobileNetV3Small):
            self.thermal_backbone.set_backbone_trainable(trainable)

    def forward(self, x_img: Tensor, x_struct: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        thermal_feat = extract_thermal_features(self.thermal_backbone, x_img)
        structured_feat = self.structured_mlp(x_struct)
        fused = self.fusion_mlp(torch.cat([thermal_feat, structured_feat], dim=1))
        logits_cls = self.classifier_head(fused)
        if self.multi_task:
            logits_sev = self.severity_head(fused)
            return logits_cls, logits_sev
        return logits_cls


def compute_structured_loss(
    logits_cls: Tensor,
    logits_sev: Tensor | None,
    targets: Tensor,
    severities: Tensor,
    patient_weights: Tensor,
    class_weights: Tensor,
    use_severity_weighting: bool,
    multi_task: bool,
    lambda_sev: float,
    severity_beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    cls_losses = F.cross_entropy(logits_cls, targets, reduction="none")
    sample_weights = class_weights[targets]
    if use_severity_weighting:
        pos_mask = targets == 1
        if pos_mask.any():
            pos_sev = severities[pos_mask].long()
            sev_scale = torch.ones_like(pos_sev, dtype=torch.float32)
            for sev_value, factor in SEVERITY_MULTIPLIER.items():
                sev_scale = torch.where(
                    pos_sev == int(sev_value),
                    torch.full_like(sev_scale, float(factor)),
                    sev_scale,
                )
            sample_weights[pos_mask] = sample_weights[pos_mask] * sev_scale

    total_weights = sample_weights * patient_weights
    loss_cls = (cls_losses * total_weights).sum() / total_weights.sum().clamp_min(1e-8)

    if not multi_task or logits_sev is None:
        zero = torch.tensor(0.0, device=logits_cls.device)
        return loss_cls, loss_cls, zero

    pos_mask = targets == 1
    if pos_mask.any():
        sev_targets = severity_to_regression_target(severities[pos_mask])
        sev_losses = F.smooth_l1_loss(
            logits_sev[pos_mask].squeeze(-1),
            sev_targets,
            beta=severity_beta,
            reduction="none",
        )
        sev_weights = patient_weights[pos_mask]
        loss_sev = (sev_losses * sev_weights).sum() / sev_weights.sum().clamp_min(1e-8)
    else:
        loss_sev = torch.tensor(0.0, device=logits_cls.device)

    total_loss = loss_cls + lambda_sev * loss_sev
    return total_loss, loss_cls, loss_sev


def train_epoch(
    model: ThermalStructuredFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Tensor,
    use_severity_weighting: bool,
    multi_task: bool,
    lambda_sev: float,
    severity_beta: float,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x_img, x_struct, y, sev, patient_weights, _ in loader:
        x_img = x_img.to(device)
        x_struct = x_struct.to(device)
        y = y.to(device)
        sev = sev.to(device)
        patient_weights = patient_weights.to(device)

        optimizer.zero_grad()
        if multi_task:
            logits_cls, logits_sev = model(x_img, x_struct)
        else:
            logits_cls = model(x_img, x_struct)
            logits_sev = None
        loss, _, _ = compute_structured_loss(
            logits_cls,
            logits_sev,
            y,
            sev,
            patient_weights,
            class_weights,
            use_severity_weighting,
            multi_task,
            lambda_sev,
            severity_beta,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = x_img.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: ThermalStructuredFusionModel,
    loader: DataLoader,
    device: torch.device,
    multi_task: bool,
) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    model.eval()
    all_ids: list[str] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    for x_img, x_struct, y, _, _, sample_ids in loader:
        x_img = x_img.to(device)
        x_struct = x_struct.to(device)
        if multi_task:
            logits_cls, _ = model(x_img, x_struct)
        else:
            logits_cls = model(x_img, x_struct)
        probs = F.softmax(logits_cls, dim=1)[:, 1].cpu().numpy()
        preds = logits_cls.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())
        all_ids.extend(list(sample_ids))

    labels = np.asarray(all_labels, dtype=np.int64)
    probs = np.asarray(all_probs, dtype=np.float32)
    preds = np.asarray(all_preds, dtype=np.int64)
    metrics = compute_metrics(labels, preds, probs)
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
    parser.add_argument("--asr-subset", type=Path, default=Path("reports/asr_candidate_modeling_subset.csv"))
    parser.add_argument("--clinical-subset", type=Path, default=Path("reports/clinical_candidate_modeling_subset.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="mobilenet", choices=["simple", "deeper", "mobilenet"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights for MobileNet")
    parser.add_argument("--no-mask", action="store_true", help="Disable face masking")
    parser.add_argument("--no-severity", action="store_true", help="Disable severity-weighted classification")
    parser.add_argument("--augment", action="store_true", help="Enable thermal augmentation")
    parser.add_argument(
        "--augmentation-strategy",
        type=str,
        default="baseline",
        choices=AUGMENTATION_STRATEGIES,
        help="Augmentation recipe to apply when --augment is enabled",
    )
    parser.add_argument("--region-attention", action="store_true", help="Use region attention as second thermal channel")
    parser.add_argument("--multi-task", action="store_true", help="Enable ICAS + severity joint training")
    parser.add_argument("--lambda-sev", type=float, default=0.3, help="Severity regression loss weight")
    parser.add_argument("--severity-beta", type=float, default=0.25, help="SmoothL1 beta for severity regression")
    parser.add_argument("--init-checkpoint", type=Path, help="Optional prior thermal-only checkpoint used to initialize the thermal branch")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; 0 disables clipping")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--freeze-thermal-epochs", type=int, default=0, help="Freeze the thermal branch for the first N epochs after initialization")
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["f1", "bal_acc"])
    parser.add_argument("--selection-metric", type=str, default="f1", choices=["auc_roc", "f1"])
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_datasets(args: argparse.Namespace, repo_root: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, float], dict[str, float], dict[str, float]]:
    data = load_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    structured_df = load_structured_feature_table(args.asr_subset, args.clinical_subset)
    patient_features = build_patient_feature_lookup(structured_df)

    def filter_complete_cases(samples: list[dict]) -> list[dict]:
        return [sample for sample in samples if str(sample["canonical_patient_id"]) in patient_features]

    data["train"] = filter_complete_cases(data["train"])
    data["val"] = filter_complete_cases(data["val"])
    data["test"] = filter_complete_cases(data["test"])

    all_samples = data["train"] + data["val"] + data["test"]
    sample_features = broadcast_patient_features_to_samples(all_samples, patient_features)
    sample_weights = build_patient_sample_weight_lookup(all_samples)

    target_size = (args.target_size, args.target_size)
    use_mask = not args.no_mask
    npy_dir = args.npy_dir if args.npy_dir.exists() else None

    train_base = TemperatureDataset(
        data["train"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size,
        use_mask=use_mask,
        augment=args.augment,
        augmentation_strategy=args.augmentation_strategy,
        region_attention=args.region_attention,
        npy_dir=npy_dir,
    )
    val_base = TemperatureDataset(
        data["val"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size,
        use_mask=use_mask,
        region_attention=args.region_attention,
        npy_dir=npy_dir,
    )
    test_base = TemperatureDataset(
        data["test"],
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size,
        use_mask=use_mask,
        region_attention=args.region_attention,
        npy_dir=npy_dir,
    )

    datasets = {
        "train": MultimodalTemperatureDataset(train_base, sample_features, sample_weights),
        "val": MultimodalTemperatureDataset(val_base, sample_features, sample_weights),
        "test": MultimodalTemperatureDataset(test_base, sample_features, sample_weights),
    }
    split_counts = {name: len(ds) for name, ds in datasets.items()}
    patient_counts = {
        name: len({sample["canonical_patient_id"] for sample in data[name]})
        for name in ("train", "val", "test")
    }
    return datasets, split_counts, patient_counts, sample_weights, data["split_info"]


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

    repo_root = Path(".").resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Use multi-task: {args.multi_task}")
    print(f"Use severity weighting: {not args.no_severity}")
    print("Structured branch: 9 ASR + top-3 clinical")

    datasets, split_counts, patient_counts, _, split_info = build_datasets(args, repo_root)
    print(f"  Train: {split_counts['train']} samples / {patient_counts['train']} patients")
    print(f"  Val:   {split_counts['val']} samples / {patient_counts['val']} patients")
    print(f"  Test:  {split_counts['test']} samples / {patient_counts['test']} patients")

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_channels = 2 if args.region_attention else 1
    model = ThermalStructuredFusionModel(
        model_name=args.model,
        structured_dim=len(ASR_TOP9_FEATURES) + len(CLINICAL_TOP3_FEATURES),
        dropout=args.dropout,
        in_channels=in_channels,
        img_size=args.target_size,
        multi_task=args.multi_task,
        pretrained=not args.no_pretrained,
    ).to(device)

    if args.init_checkpoint:
        loaded = load_initial_thermal_weights(model, args.init_checkpoint, device)
        print(f"Loaded {loaded} thermal parameters from {args.init_checkpoint}")

    class_weights = compute_class_weights(datasets["train"].base_dataset, device)
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
    thermal_trainable = True
    ckpt_path = args.output / "best_cnn_multimodal.pt"

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        should_train_thermal = epoch > args.freeze_thermal_epochs
        if should_train_thermal != thermal_trainable:
            model.set_backbone_trainable(should_train_thermal)
            thermal_trainable = should_train_thermal
            state = "trainable" if should_train_thermal else "frozen"
            print(f"  Epoch {epoch:3d}: thermal branch is now {state}")

        should_train_backbone = not (
            args.model == "mobilenet"
            and args.freeze_backbone_epochs > 0
            and epoch <= args.freeze_backbone_epochs
        )
        if args.model == "mobilenet" and should_train_backbone != should_train_thermal:
            model.thermal_backbone.set_backbone_trainable(should_train_backbone and should_train_thermal)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            use_severity_weighting=not args.no_severity,
            multi_task=args.multi_task,
            lambda_sev=args.lambda_sev,
            severity_beta=args.severity_beta,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, val_labels, val_probs = evaluate(model, val_loader, device, multi_task=args.multi_task)
        scheduler.step()

        selection_score = compute_selection_score(val_metrics, args.selection_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
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
                ckpt_path,
            )

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                f"val_auc={val_metrics['auc_roc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        if early_stopper.step(epoch, val_metrics["auc_roc"]):
            print(
                f"Early stopping at epoch {epoch}: no val_auc improvement "
                f"for {args.early_stop_patience} epochs."
            )
            break

    print(f"\nBest validation {args.selection_metric}: {best_selection_score:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, _, val_labels, val_probs = evaluate(model, val_loader, device, multi_task=args.multi_task)
    best_threshold, val_threshold_metrics = find_best_threshold(
        val_labels, val_probs, metric=args.threshold_metric
    )
    test_metrics, _, test_labels, test_probs = evaluate(model, test_loader, device, multi_task=args.multi_task)
    test_threshold_preds = (test_probs >= best_threshold).astype(int)
    test_threshold_metrics = compute_metrics(test_labels, test_threshold_preds, test_probs)
    test_threshold_metrics["threshold"] = best_threshold
    test_threshold_metrics["optimized_metric"] = args.threshold_metric

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model": args.model,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "target_size": args.target_size,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "use_face_mask": not args.no_mask,
        "use_region_attention": args.region_attention,
        "use_severity_weighting": not args.no_severity,
        "use_augmentation": args.augment,
        "use_multi_task": args.multi_task,
        "lambda_sev": args.lambda_sev if args.multi_task else None,
        "severity_beta": args.severity_beta if args.multi_task else None,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "freeze_backbone_epochs": args.freeze_backbone_epochs if args.model == "mobilenet" else 0,
        "freeze_thermal_epochs": args.freeze_thermal_epochs,
        "selection_metric": args.selection_metric,
        "threshold_metric": args.threshold_metric,
        "structured_asr_features": ASR_TOP9_FEATURES,
        "structured_clinical_features": CLINICAL_TOP3_FEATURES,
        "structured_feature_dim": len(ASR_TOP9_FEATURES) + len(CLINICAL_TOP3_FEATURES),
        "patient_balanced_loss": True,
        "split_sample_counts": split_counts,
        "split_patient_counts": patient_counts,
        "split_info": split_info,
        "test_metrics": test_metrics,
        "val_threshold_metrics": val_threshold_metrics,
        "test_threshold_metrics": test_threshold_metrics,
    }

    results_path = args.output / f"cnn_multimodal_results_{timestamp}.json"
    history_path = args.output / f"cnn_multimodal_history_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
