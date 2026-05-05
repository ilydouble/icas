#!/usr/bin/env python3
"""Train CNN on masked temperature matrices for ICAS classification.

Key features:
1. Uses pre-computed face masks from face_roi_annotation.py
2. Only uses temperature values within face region
3. Severity-weighted loss: positive samples weighted by stenosis severity
4. Optional region attention: dual-channel input (temp + attention map)
5. Optional multi-task learning: classification + severity regression
6. Optional soft labels: continuous labels based on severity (0, 0.6, 0.8, 1.0)
7. Transfer learning: MobileNetV3-Small with pretrained ImageNet weights

Usage:
    python scripts/train_cnn_v2.py
    python scripts/train_cnn_v2.py --epochs 100 --device cuda
    python scripts/train_cnn_v2.py --augment
    python scripts/train_cnn_v2.py --region-attention
    python scripts/train_cnn_v2.py --multi-task --lambda-sev 0.3
    python scripts/train_cnn_v2.py --soft-label --lambda-sev 0.5
    python scripts/train_cnn_v2.py --model mobilenet --soft-label
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

SEVERITY_MULTIPLIER = {0: 1.0, 1: 1.0, 2: 1.5, 3: 2.0}


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


def severity_to_regression_target(severities: Tensor) -> Tensor:
    """Map stenosis grades onto a compact [0, 1] regression target.

    The auxiliary task is only evaluated on positive samples. Mapping
    mild/moderate/severe to 0.0/0.5/1.0 keeps the target well-scaled and
    reduces the tendency of raw MSE on {1,2,3} to dominate classification.
    """
    targets = torch.zeros_like(severities, dtype=torch.float32)
    targets = torch.where(severities >= 3, torch.ones_like(targets), targets)
    targets = torch.where((severities >= 2) & (severities < 3), torch.full_like(targets, 0.5), targets)
    return targets


def compute_sample_weights(
    targets: Tensor,
    severities: Tensor,
    class_weights: Tensor,
    severity_weights: dict[int, float],
) -> Tensor:
    """Combine class balancing with a mild severity emphasis for positives."""
    sample_weights = torch.ones_like(targets, dtype=torch.float32)
    neg_mask = targets == 0
    sample_weights[neg_mask] = class_weights[0]

    pos_mask = targets == 1
    if pos_mask.any():
        pos_severities = severities[pos_mask]
        severity_scale = torch.ones_like(pos_severities, dtype=torch.float32)
        for sev, scale in severity_weights.items():
            severity_scale = torch.where(
                pos_severities == float(sev),
                torch.full_like(severity_scale, float(scale)),
                severity_scale,
            )
        sample_weights[pos_mask] = class_weights[1] * severity_scale
    return sample_weights


class EarlyStopping:
    """Stop training after sustained validation stagnation."""

    def __init__(self, patience: int, min_epochs: int, min_delta: float = 0.0):
        self.patience = max(0, patience)
        self.min_epochs = max(1, min_epochs)
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def step(self, epoch: int, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False

        if epoch < self.min_epochs:
            return False

        self.bad_epochs += 1
        return self.bad_epochs > self.patience


TEMP_RANGE_MIN = 15.0
TEMP_RANGE_MAX = 40.0

REGION_ATTENTION_WEIGHTS = {
    "forehead": 1.0,
    "left_cheek": 1.0,
    "right_cheek": 1.0,
    "left_eye": 0.5,
    "right_eye": 0.5,
    "nose": 0.3,
}


def build_region_attention_map(
    annotation: dict,
    temp_shape: tuple[int, int],
    target_size: tuple[int, int],
    image_size: dict,
) -> np.ndarray:
    """Build a spatial attention map from region polygons.

    Each pixel gets the max attention weight across overlapping regions.
    Face area outside any sub-region gets a base weight of 0.2.
    Background (outside face) gets 0.0.
    """
    temp_h, temp_w = temp_shape
    attention = np.zeros((temp_h, temp_w), dtype=np.float32)

    face = annotation.get("face")
    regions = annotation.get("regions") or {}

    if face and face.get("polygon"):
        face_poly = face["polygon"]
        sx = temp_w / float(image_size.get("width") or temp_w)
        sy = temp_h / float(image_size.get("height") or temp_h)
        face_poly_t = [[x * sx, y * sy] for x, y in face_poly]
        face_mask = np.zeros((temp_h, temp_w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [np.array(face_poly_t, dtype=np.int32)], 1)
        attention[face_mask > 0] = 0.2

    for region_name, weight in REGION_ATTENTION_WEIGHTS.items():
        region = regions.get(region_name)
        if not region or not region.get("polygon"):
            continue
        poly = region["polygon"]
        sx = temp_w / float(image_size.get("width") or temp_w)
        sy = temp_h / float(image_size.get("height") or temp_h)
        poly_t = [[x * sx, y * sy] for x, y in poly]
        region_mask = np.zeros((temp_h, temp_w), dtype=np.uint8)
        cv2.fillPoly(region_mask, [np.array(poly_t, dtype=np.int32)], 1)
        attention[region_mask > 0] = np.maximum(attention[region_mask > 0], weight)

    attention = cv2.resize(attention, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return attention.astype(np.float32)


class TemperatureAugmentation:
    """Data augmentation for temperature heatmaps.

    Applies temperature-level and geometric augmentations.
    Operates on normalized [0, 1] matrices.
    Synchronously transforms both temperature and attention map.
    """

    def __init__(
        self,
        temp_offset_range: tuple[float, float] = (-1.0, 1.0),
        temp_scale_range: tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.02,
        rotation_range: float = 5.0,
        translation_range: int = 5,
        p_flip: float = 0.5,
    ):
        self.temp_offset_range = temp_offset_range
        self.temp_scale_range = temp_scale_range
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.p_flip = p_flip

    def __call__(
        self,
        matrix: np.ndarray,
        attention_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        offset = np.random.uniform(*self.temp_offset_range)
        scale = np.random.uniform(*self.temp_scale_range)
        offset_norm = offset / (TEMP_RANGE_MAX - TEMP_RANGE_MIN)

        aug_matrix = matrix * scale + offset_norm

        if self.noise_std > 0:
            aug_matrix = aug_matrix + np.random.randn(*aug_matrix.shape).astype(np.float32) * self.noise_std

        aug_matrix = aug_matrix.clip(0.0, 1.0).astype(np.float32)
        aug_attention = attention_map.copy() if attention_map is not None else None

        if np.random.rand() < self.p_flip:
            aug_matrix = cv2.flip(aug_matrix, 1)
            if aug_attention is not None:
                aug_attention = cv2.flip(aug_attention, 1)

        if self.rotation_range > 0 or self.translation_range > 0:
            h, w = aug_matrix.shape
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            if self.translation_range > 0:
                tx = np.random.randint(-self.translation_range, self.translation_range + 1)
                ty = np.random.randint(-self.translation_range, self.translation_range + 1)
                M[:, 2] += [tx, ty]
            aug_matrix = cv2.warpAffine(aug_matrix, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            if aug_attention is not None:
                aug_attention = cv2.warpAffine(aug_attention, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        return aug_matrix, aug_attention


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

    if mask is not None and (mask.shape[0] != temp_h or mask.shape[1] != temp_w):
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
    """Cross-entropy with severity-weighted sample weights.

    Weight formula per sample:
        negative (label=0): w = class_weight[0]
        positive stenosis=1: w = class_weight[1] * 1.0
        positive stenosis=2: w = class_weight[1] * 2.0
        positive stenosis=3: w = class_weight[1] * 3.0
    """

    def __init__(self, class_weights: Tensor, severity_weights: dict[int, float]):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.severity_weights = severity_weights
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: Tensor, targets: Tensor, severities: Tensor) -> Tensor:
        ce_loss = self.ce(logits, targets)
        sample_weights = compute_sample_weights(
            targets, severities, self.class_weights, self.severity_weights
        )
        return (ce_loss * sample_weights).mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss: binary classification + severity regression.

    Main task: binary classification (ICAS positive/negative)
    Auxiliary task: severity regression (only computed on positive samples)
    """

    def __init__(
        self,
        class_weights: Tensor,
        severity_weights: dict[int, float],
        lambda_sev: float = 0.3,
        severity_beta: float = 0.25,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.severity_weights = severity_weights
        self.lambda_sev = lambda_sev
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.sev_loss_fn = nn.SmoothL1Loss(beta=severity_beta)

    def forward(
        self,
        logits_cls: Tensor,
        logits_sev: Tensor,
        targets_cls: Tensor,
        targets_sev: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cls_losses = self.cls_loss_fn(logits_cls, targets_cls)
        sample_weights = compute_sample_weights(
            targets_cls, targets_sev, self.class_weights, self.severity_weights
        )
        loss_cls = (cls_losses * sample_weights).mean()

        pos_mask = targets_cls == 1
        if pos_mask.sum() > 0:
            pos_logits_sev = logits_sev[pos_mask].squeeze(-1)
            pos_targets_sev = severity_to_regression_target(targets_sev[pos_mask])
            loss_sev = self.sev_loss_fn(pos_logits_sev, pos_targets_sev)
        else:
            loss_sev = torch.tensor(0.0, device=logits_cls.device)

        total_loss = loss_cls + self.lambda_sev * loss_sev
        return total_loss, loss_cls, loss_sev


class MultiTaskSoftLabelLoss(nn.Module):
    """Multi-task loss with soft labels: classification + severity regression.

    Main task: binary classification with soft labels based on severity
    Auxiliary task: severity regression (only computed on positive samples)

    Soft label mapping:
        0 (negative): 0.0
        1 (mild): 0.6
        2 (moderate): 0.8
        3 (severe): 1.0
    """

    SEVERITY_TO_SOFT_LABEL = {
        0: 0.0,
        1: 0.6,
        2: 0.8,
        3: 1.0,
    }

    def __init__(self, lambda_sev: float = 0.5):
        super().__init__()
        self.lambda_sev = lambda_sev
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.sev_loss_fn = nn.MSELoss()

    def forward(
        self,
        logits_cls: Tensor,
        logits_sev: Tensor,
        targets_cls: Tensor,
        targets_sev: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits_cls = logits_cls.squeeze(-1)
        logits_sev = logits_sev.squeeze(-1)

        soft_targets = torch.zeros_like(logits_cls)
        for i in range(len(targets_sev)):
            sev = int(targets_sev[i].item())
            soft_targets[i] = self.SEVERITY_TO_SOFT_LABEL.get(sev, 1.0)

        loss_cls = self.cls_loss_fn(logits_cls, soft_targets)

        pos_mask = targets_cls == 1
        if pos_mask.sum() > 0:
            pos_logits_sev = logits_sev[pos_mask]
            pos_targets_sev = targets_sev[pos_mask].float()
            loss_sev = self.sev_loss_fn(pos_logits_sev, pos_targets_sev)
        else:
            loss_sev = torch.tensor(0.0, device=logits_cls.device)

        total_loss = loss_cls + self.lambda_sev * loss_sev
        return total_loss, loss_cls, loss_sev


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
        region_attention: bool = False,
        npy_dir: Path | None = None,
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
        self.region_attention = region_attention
        self.npy_dir = npy_dir
        self.augmentation = TemperatureAugmentation() if augment else None
        self._temp_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        self._attention_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_attention_map(self, sample_id: str, temp_shape: tuple[int, int]) -> np.ndarray:
        if sample_id in self._attention_cache:
            return self._attention_cache[sample_id]
        annotation = self.annotations.get(sample_id)
        if not annotation or annotation.get("status") != "ok":
            attn = np.zeros(self.target_size, dtype=np.float32)
            self._attention_cache[sample_id] = attn
            return attn
        sample = next((s for s in self.samples if s["sample_id"] == sample_id), None)
        image_size = (sample or {}).get("image_size", {})
        attn = build_region_attention_map(
            annotation, temp_shape, self.target_size, image_size
        )
        self._attention_cache[sample_id] = attn
        return attn

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, str]:
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        patient_id = sample["canonical_patient_id"]

        if sample_id in self._temp_cache:
            temp_matrix = self._temp_cache[sample_id]
        elif self.npy_dir is not None:
            npy_path = self.npy_dir / f"{sample_id}.npy"
            if npy_path.exists():
                temp_matrix = np.load(npy_path)
                self._temp_cache[sample_id] = temp_matrix
            else:
                temp_path = self.repo_root / sample["temperature_path"]
                try:
                    temp_matrix = parse_temperature_csv(temp_path)
                    self._temp_cache[sample_id] = temp_matrix
                except Exception as e:
                    print(f"Warning: Failed to load {sample_id}: {e}")
                    temp_matrix = np.zeros((1024, 1280), dtype=np.float32)
        else:
            temp_path = self.repo_root / sample["temperature_path"]
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

        if self.region_attention:
            attention_map = self._get_attention_map(sample_id, temp_matrix.shape)
        else:
            attention_map = None

        if self.augmentation is not None:
            masked_temp, attention_map = self.augmentation(masked_temp, attention_map)

        if self.region_attention:
            if attention_map is None:
                attention_map = np.zeros(self.target_size, dtype=np.float32)
            x = torch.from_numpy(np.stack([masked_temp, attention_map], axis=0)).float()
        else:
            x = torch.from_numpy(masked_temp).unsqueeze(0).float()

        y = torch.tensor(self.labels.get(patient_id, 0), dtype=torch.long)
        sev = torch.tensor(self.severities.get(patient_id, 0), dtype=torch.float)

        return x, y, sev, sample_id


class SimpleCNN(nn.Module):
    """Simple CNN for temperature matrix classification."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        in_channels: int = 1,
        img_size: int = 64,
        multi_task: bool = False,
        soft_label: bool = False,
    ):
        super().__init__()
        self.multi_task = multi_task
        self.soft_label = soft_label
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self._init_fc(num_classes, in_channels, img_size)

    def _init_fc(self, num_classes: int, in_channels: int, img_size: int):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            dummy = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
            dummy = self.pool(F.relu(self.bn3(self.conv3(dummy))))
            self.flat_size = dummy.numel()
        self.fc1 = nn.Linear(self.flat_size, 256)
        cls_out = 1 if self.soft_label else num_classes
        self.classifier_head = nn.Linear(256, cls_out)
        if self.multi_task:
            self.severity_head = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        features = self.dropout(F.relu(self.fc1(x)))
        logits_cls = self.classifier_head(features)
        if self.multi_task:
            logits_sev = self.severity_head(features)
            return logits_cls, logits_sev
        return logits_cls


class DeeperCNN(nn.Module):
    """Deeper CNN with more capacity."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        in_channels: int = 1,
        img_size: int = 64,
        multi_task: bool = False,
        soft_label: bool = False,
    ):
        super().__init__()
        self.multi_task = multi_task
        self.soft_label = soft_label
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
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
        self._init_classifier(num_classes, dropout, in_channels, img_size)

    def _init_classifier(self, num_classes: int, dropout: float, in_channels: int, img_size: int):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            dummy = self.features(dummy)
            self.flat_size = dummy.numel()
        self.shared = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        cls_out = 1 if self.soft_label else num_classes
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, cls_out),
        )
        if self.multi_task:
            self.severity_head = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.shared(x)
        logits_cls = self.classifier_head(features)
        if self.multi_task:
            logits_sev = self.severity_head(features)
            return logits_cls, logits_sev
        return logits_cls


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small with pretrained weights for transfer learning.

    Smallest pretrained model in torchvision (~2.5M params).
    Adapts first conv layer for 1 or 2 channel input.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        in_channels: int = 1,
        img_size: int = 64,
        multi_task: bool = False,
        soft_label: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.multi_task = multi_task
        self.soft_label = soft_label
        self.in_channels = in_channels

        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                elif in_channels == 2:
                    new_conv.weight.data[:, :2, :, :] = original_conv.weight.data[:, :2, :, :]
                else:
                    new_conv.weight.data = original_conv.weight.data.clone()
        self.backbone.features[0][0] = new_conv

        feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        cls_out = 1 if soft_label else num_classes
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, cls_out),
        )
        if multi_task:
            self.severity_head = nn.Linear(feature_dim, 1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = trainable

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        features = self.backbone(x)
        logits_cls = self.classifier_head(features)
        if self.multi_task:
            logits_sev = self.severity_head(features)
            return logits_cls, logits_sev
        return logits_cls


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


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, dict]:
    """Pick a probability threshold on the validation set.

    We optimize a thresholded metric while preserving AUC reporting separately.
    For this project, F1 is the default because the ranking quality can be good
    while the default 0.5 operating point is poor.
    """
    if len(y_true) == 0:
        return 0.5, compute_metrics(y_true, y_true, y_prob)

    thresholds = sorted({0.0, 1.0, *[float(x) for x in y_prob.tolist()]})
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
    best_score = best_metrics[metric]

    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, preds, y_prob)
        score = metrics[metric]
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    best_metrics = dict(best_metrics)
    best_metrics["threshold"] = float(best_threshold)
    best_metrics["optimized_metric"] = metric
    return float(best_threshold), best_metrics


def compute_selection_score(metrics: dict[str, float], metric_name: str) -> float:
    """Read the metric used to select the best checkpoint."""
    return float(metrics.get(metric_name, float("-inf")))


def aggregate_patient_predictions(
    sample_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_to_patient: dict[str, str],
    method: str = "mean",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate sample-level predictions into patient-level scores."""
    patient_order: list[str] = []
    patient_prob_buckets: dict[str, list[float]] = {}
    patient_labels: dict[str, int] = {}

    for sample_id, label, prob in zip(sample_ids, y_true, y_prob):
        patient_id = sample_to_patient[sample_id]
        if patient_id not in patient_prob_buckets:
            patient_order.append(patient_id)
            patient_prob_buckets[patient_id] = []
            patient_labels[patient_id] = int(label)
        patient_prob_buckets[patient_id].append(float(prob))

    labels = np.array([patient_labels[pid] for pid in patient_order], dtype=np.int64)
    def _reduce(values: list[float]) -> float:
        if method == "mean":
            return float(np.mean(values))
        if method == "max":
            return float(np.max(values))
        if method == "top2_mean":
            topk = sorted(values, reverse=True)[:2]
            return float(np.mean(topk))
        raise ValueError(f"Unsupported patient aggregation method: {method}")

    probs = np.array([_reduce(patient_prob_buckets[pid]) for pid in patient_order], dtype=np.float32)
    return patient_order, labels, probs


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    multi_task: bool = False,
    grad_clip: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y, sev, _ in loader:
        x, y, sev = x.to(device), y.to(device), sev.to(device)
        optimizer.zero_grad()
        if multi_task:
            logits_cls, logits_sev = model(x)
            loss, _, _ = criterion(logits_cls, logits_sev, y, sev)
        else:
            logits = model(x)
            loss = criterion(logits, y, sev)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
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
    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []

    for x, y, _, sample_ids in loader:
        x = x.to(device)
        if multi_task:
            logits_cls, _ = model(x)
        else:
            logits_cls = model(x)

        if soft_label:
            probs = torch.sigmoid(logits_cls.squeeze(-1)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
        else:
            probs = F.softmax(logits_cls, dim=1)[:, 1].cpu().numpy()
            preds = logits_cls.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(y.numpy())
        all_ids.extend(sample_ids)

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    return metrics, all_ids, labels, probs


def build_patient_metrics(
    sample_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    sample_to_patient: dict[str, str],
    threshold: float,
    aggregation_method: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    _, patient_labels, patient_probs = aggregate_patient_predictions(
        sample_ids,
        labels,
        probs,
        sample_to_patient,
        method=aggregation_method,
    )
    patient_preds = (patient_probs >= threshold).astype(int)
    metrics = compute_metrics(patient_labels, patient_preds, patient_probs)
    return metrics, patient_labels, patient_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"), help="Directory with pre-converted NPY temperature files")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="mobilenet", choices=["simple", "deeper", "mobilenet"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights (for mobilenet)")
    parser.add_argument("--no-mask", action="store_true", help="Disable face masking")
    parser.add_argument("--no-severity", action="store_true", help="Disable severity weighting")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--region-attention", action="store_true", help="Use region attention map as second input channel")
    parser.add_argument("--multi-task", action="store_true", help="Enable multi-task learning (classification + severity regression)")
    parser.add_argument("--soft-label", action="store_true", help="Use soft labels for classification (requires --multi-task)")
    parser.add_argument("--lambda-sev", type=float, default=0.3, help="Severity loss weight for multi-task learning")
    parser.add_argument("--severity-beta", type=float, default=0.25, help="SmoothL1 beta for severity regression")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; 0 disables clipping")
    parser.add_argument("--early-stop-patience", type=int, default=8, help="Stop after this many non-improving epochs")
    parser.add_argument("--early-stop-min-epochs", type=int, default=8, help="Do not early-stop before this epoch")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum AUC gain to reset patience")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Freeze MobileNet features for the first N epochs")
    parser.add_argument("--threshold-metric", type=str, default="f1", choices=["f1", "bal_acc"], help="Validation metric used to choose the classification threshold")
    parser.add_argument("--selection-metric", type=str, default="f1", choices=["auc_roc", "f1", "patient_auc_roc", "patient_f1"], help="Validation metric used to select the best checkpoint")
    parser.add_argument("--patient-aggregation", type=str, default="mean", choices=["mean", "max", "top2_mean"], help="How to combine multiple samples from the same patient")
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
    print(f"Device: {device}")
    print(f"Use face mask: {not args.no_mask}")
    print(f"Use severity weighting: {not args.no_severity}")
    print(f"Use augmentation: {args.augment}")
    print(f"Use region attention: {args.region_attention}")
    print(f"Use multi-task learning: {args.multi_task}")
    print(f"Use soft labels: {args.soft_label}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Patient aggregation: {args.patient_aggregation}")
    if args.model == "mobilenet":
        print(f"Use pretrained: {not args.no_pretrained}")
        print(f"Freeze backbone epochs: {args.freeze_backbone_epochs}")

    repo_root = Path(".").resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    data = load_data(
        args.manifest, args.clinical, args.split, args.excluded, args.annotations
    )
    print(f"  Train: {len(data['train'])} samples")
    print(f"  Val:   {len(data['val'])} samples")
    print(f"  Test:  {len(data['test'])} samples")

    target_size = (args.target_size, args.target_size)
    use_mask = not args.no_mask
    npy_dir = args.npy_dir

    if npy_dir and npy_dir.exists():
        print(f"Using NPY cache: {npy_dir}")

    train_dataset = TemperatureDataset(
        data["train"], data["annotations"], data["labels"], data["severities"],
        repo_root, args.masks_dir, target_size, use_mask=use_mask, augment=args.augment,
        region_attention=args.region_attention, npy_dir=npy_dir,
    )
    val_dataset = TemperatureDataset(
        data["val"], data["annotations"], data["labels"], data["severities"],
        repo_root, args.masks_dir, target_size, use_mask=use_mask,
        region_attention=args.region_attention, npy_dir=npy_dir,
    )
    test_dataset = TemperatureDataset(
        data["test"], data["annotations"], data["labels"], data["severities"],
        repo_root, args.masks_dir, target_size, use_mask=use_mask,
        region_attention=args.region_attention, npy_dir=npy_dir,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_channels = 2 if args.region_attention else 1
    soft_label = args.soft_label
    multi_task = args.multi_task or args.soft_label

    if args.model == "mobilenet":
        model = MobileNetV3Small(
            num_classes=2,
            dropout=args.dropout,
            in_channels=in_channels,
            img_size=args.target_size,
            multi_task=multi_task,
            soft_label=soft_label,
            pretrained=not args.no_pretrained,
        )
    elif args.model == "simple":
        model = SimpleCNN(
            num_classes=2,
            dropout=args.dropout,
            in_channels=in_channels,
            img_size=args.target_size,
            multi_task=multi_task,
            soft_label=soft_label,
        )
    else:
        model = DeeperCNN(
            num_classes=2,
            dropout=args.dropout,
            in_channels=in_channels,
            img_size=args.target_size,
            multi_task=multi_task,
            soft_label=soft_label,
        )
    model = model.to(device)

    class_weights = compute_class_weights(train_dataset, device)

    if soft_label:
        criterion = MultiTaskSoftLabelLoss(lambda_sev=args.lambda_sev)
    elif multi_task:
        criterion = MultiTaskLoss(
            class_weights,
            SEVERITY_MULTIPLIER,
            lambda_sev=args.lambda_sev,
            severity_beta=args.severity_beta,
        )
    elif args.no_severity:
        criterion = SeverityWeightedLoss(class_weights, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    else:
        criterion = SeverityWeightedLoss(class_weights, SEVERITY_MULTIPLIER)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_epochs=args.early_stop_min_epochs,
        min_delta=args.early_stop_min_delta,
    )

    sample_to_patient = {
        sample["sample_id"]: sample["canonical_patient_id"]
        for sample in (data["val"] + data["test"])
    }

    best_selection_score = float("-inf")
    best_epoch = 0
    history: list[dict] = []
    backbone_trainable = True

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        should_train_backbone = not (
            args.model == "mobilenet"
            and args.freeze_backbone_epochs > 0
            and epoch <= args.freeze_backbone_epochs
        )
        if args.model == "mobilenet" and should_train_backbone != backbone_trainable:
            model.set_backbone_trainable(should_train_backbone)
            backbone_trainable = should_train_backbone
            state = "trainable" if should_train_backbone else "frozen"
            print(f"  Epoch {epoch:3d}: backbone is now {state}")

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            multi_task=multi_task,
            grad_clip=args.grad_clip,
        )
        val_metrics, val_ids, val_labels, val_probs = evaluate(
            model, val_loader, device, multi_task=multi_task, soft_label=soft_label
        )
        val_patient_metrics, _, _ = build_patient_metrics(
            val_ids,
            val_labels,
            val_probs,
            sample_to_patient,
            threshold=0.5,
            aggregation_method=args.patient_aggregation,
        )
        scheduler.step()

        selection_metrics = dict(val_metrics)
        selection_metrics["patient_auc_roc"] = val_patient_metrics["auc_roc"]
        selection_metrics["patient_f1"] = val_patient_metrics["f1"]
        selection_score = compute_selection_score(selection_metrics, args.selection_metric)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_patient_auc_roc": val_patient_metrics["auc_roc"],
            "val_patient_f1": val_patient_metrics["f1"],
            "selection_score": selection_score,
        })

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "selection_metric": args.selection_metric,
                "selection_score": best_selection_score,
                "val_auc": val_metrics["auc_roc"],
                "val_patient_auc": val_patient_metrics["auc_roc"],
            }, args.output / "best_cnn_v2.pt")

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                  f"val_auc={val_metrics['auc_roc']:.4f} "
                  f"val_f1={val_metrics['f1']:.4f}")

        if early_stopper.step(epoch, val_metrics["auc_roc"]):
            print(f"Early stopping at epoch {epoch}: no val_auc improvement for {args.early_stop_patience} epochs.")
            break

    print(f"\nBest validation {args.selection_metric}: {best_selection_score:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(args.output / "best_cnn_v2.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_ids, val_labels, val_probs = evaluate(
        model, val_loader, device, multi_task=multi_task, soft_label=soft_label
    )
    best_threshold, val_threshold_metrics = find_best_threshold(
        val_labels, val_probs, metric=args.threshold_metric
    )
    test_metrics, test_ids, test_labels, test_probs = evaluate(
        model, test_loader, device, multi_task=multi_task, soft_label=soft_label
    )
    test_threshold_preds = (test_probs >= best_threshold).astype(int)
    test_threshold_metrics = compute_metrics(test_labels, test_threshold_preds, test_probs)
    test_threshold_metrics["threshold"] = best_threshold
    test_threshold_metrics["optimized_metric"] = args.threshold_metric

    _, val_patient_labels, val_patient_probs = aggregate_patient_predictions(
        val_ids,
        val_labels,
        val_probs,
        sample_to_patient,
        method=args.patient_aggregation,
    )
    _, test_patient_labels, test_patient_probs = aggregate_patient_predictions(
        test_ids,
        test_labels,
        test_probs,
        sample_to_patient,
        method=args.patient_aggregation,
    )
    patient_default_preds = (test_patient_probs >= 0.5).astype(int)
    patient_metrics = compute_metrics(
        test_patient_labels,
        patient_default_preds,
        test_patient_probs,
    )
    patient_threshold, val_patient_threshold_metrics = find_best_threshold(
        val_patient_labels,
        val_patient_probs,
        metric=args.threshold_metric,
    )
    patient_threshold_preds = (test_patient_probs >= patient_threshold).astype(int)
    patient_threshold_metrics = compute_metrics(
        test_patient_labels,
        patient_threshold_preds,
        test_patient_probs,
    )
    patient_threshold_metrics["threshold"] = patient_threshold
    patient_threshold_metrics["optimized_metric"] = args.threshold_metric

    print(f"\n{'='*50}")
    print("Test Results")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Threshold-tuned Test Results")
    for k, v in test_threshold_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("Patient-level Test Results")
    for k, v in patient_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Patient-level Threshold-tuned Test Results")
    for k, v in patient_threshold_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model": args.model,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "use_face_mask": use_mask,
        "use_severity_weighting": not args.no_severity,
        "use_augmentation": args.augment,
        "use_region_attention": args.region_attention,
        "use_multi_task": multi_task,
        "use_soft_label": soft_label,
        "lambda_sev": args.lambda_sev if multi_task else None,
        "severity_beta": args.severity_beta if multi_task and not soft_label else None,
        "target_size": args.target_size,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "grad_clip": args.grad_clip,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_epochs": args.early_stop_min_epochs,
        "freeze_backbone_epochs": args.freeze_backbone_epochs if args.model == "mobilenet" else 0,
        "threshold_metric": args.threshold_metric,
        "selection_metric": args.selection_metric,
        "patient_aggregation": args.patient_aggregation,
        "batch_size": args.batch_size,
        "test_metrics": test_metrics,
        "val_threshold_metrics": val_threshold_metrics,
        "test_threshold_metrics": test_threshold_metrics,
        "patient_test_metrics": patient_metrics,
        "val_patient_threshold_metrics": val_patient_threshold_metrics,
        "patient_test_threshold_metrics": patient_threshold_metrics,
    }

    results_path = args.output / f"cnn_v2_results_{timestamp}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    history_path = args.output / f"cnn_v2_history_{timestamp}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
