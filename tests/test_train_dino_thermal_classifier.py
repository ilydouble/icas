"""Unit tests for train_dino_thermal_classifier helpers."""

from __future__ import annotations

import types
import unittest

import torch

from scripts.train_dino_thermal_classifier import (
    DinoThermalClassifier,
    build_optimizer_param_groups,
)


class _FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(4, hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        pooled = self.proj(pixel_values.mean(dim=(2, 3)))
        return types.SimpleNamespace(last_hidden_state=pooled.unsqueeze(1))


class DinoThermalClassifierTests(unittest.TestCase):
    def test_forward_returns_multitask_outputs_when_enabled(self):
        model = DinoThermalClassifier(
            backbone=_FakeBackbone(hidden_size=8),
            hidden_dim=16,
            dropout=0.2,
            multi_task=True,
        )
        x = torch.randn(2, 4, 8, 8)

        logits_cls, logits_sev = model(x)

        self.assertEqual(tuple(logits_cls.shape), (2, 2))
        self.assertEqual(tuple(logits_sev.shape), (2, 1))

    def test_set_backbone_trainable_freezes_and_unfreezes_backbone_only(self):
        model = DinoThermalClassifier(
            backbone=_FakeBackbone(hidden_size=8),
            hidden_dim=16,
            dropout=0.2,
            multi_task=False,
        )

        model.set_backbone_trainable(False)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier_head.parameters()))

        model.set_backbone_trainable(True)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))


class OptimizerParamGroupTests(unittest.TestCase):
    def test_build_optimizer_param_groups_excludes_frozen_backbone(self):
        model = DinoThermalClassifier(
            backbone=_FakeBackbone(hidden_size=8),
            hidden_dim=16,
            dropout=0.2,
            multi_task=True,
        )
        model.set_backbone_trainable(False)

        groups = build_optimizer_param_groups(model, head_lr=1e-3, backbone_lr=1e-5)

        self.assertEqual(len(groups), 1)
        params = list(groups[0]["params"])
        self.assertTrue(params)
        self.assertTrue(all(p.requires_grad for p in params))
        frozen_ids = {id(p) for p in model.backbone.parameters()}
        self.assertTrue(all(id(p) not in frozen_ids for p in params))

    def test_build_optimizer_param_groups_includes_backbone_when_trainable(self):
        model = DinoThermalClassifier(
            backbone=_FakeBackbone(hidden_size=8),
            hidden_dim=16,
            dropout=0.2,
            multi_task=False,
        )
        model.set_backbone_trainable(True)

        groups = build_optimizer_param_groups(model, head_lr=1e-3, backbone_lr=1e-5)

        self.assertEqual(len(groups), 2)
        lrs = sorted(group["lr"] for group in groups)
        self.assertEqual(lrs, [1e-5, 1e-3])


if __name__ == "__main__":
    unittest.main()
