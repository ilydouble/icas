"""Unit tests for cross-year patient pair thermal CNN helpers."""

from __future__ import annotations

import unittest

import torch

from scripts.train_cnn_pair import (
    PairSharedBackboneClassifier,
    build_cross_year_patient_pairs,
    build_patient_pair_weight_lookup,
)
from scripts.train_cnn_v3 import ResNet50Backbone


class CrossYearPairBuilderTests(unittest.TestCase):
    def test_build_cross_year_patient_pairs_only_keeps_2024_2025_pairs(self):
        samples = [
            {"sample_id": "2024_P1_1", "canonical_patient_id": "P1", "year": 2024},
            {"sample_id": "2025_P1_1", "canonical_patient_id": "P1", "year": 2025},
            {"sample_id": "2025_P1_2", "canonical_patient_id": "P1", "year": 2025},
            {"sample_id": "2024_P2_1", "canonical_patient_id": "P2", "year": 2024},
            {"sample_id": "2024_P2_2", "canonical_patient_id": "P2", "year": 2024},
            {"sample_id": "2025_P3_1", "canonical_patient_id": "P3", "year": 2025},
        ]

        pairs = build_cross_year_patient_pairs(samples)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(
            {(pair["sample_id_2024"], pair["sample_id_2025"]) for pair in pairs},
            {
                ("2024_P1_1", "2025_P1_1"),
                ("2024_P1_1", "2025_P1_2"),
            },
        )

    def test_build_patient_pair_weight_lookup_balances_patients_by_pair_count(self):
        pairs = [
            {"pair_id": "P1_pair_1", "canonical_patient_id": "P1"},
            {"pair_id": "P1_pair_2", "canonical_patient_id": "P1"},
            {"pair_id": "P2_pair_1", "canonical_patient_id": "P2"},
        ]

        weights = build_patient_pair_weight_lookup(pairs)

        self.assertAlmostEqual(weights["P1_pair_1"], 0.5, places=6)
        self.assertAlmostEqual(weights["P1_pair_2"], 0.5, places=6)
        self.assertAlmostEqual(weights["P2_pair_1"], 1.0, places=6)


class PairModelTests(unittest.TestCase):
    def test_pair_shared_backbone_classifier_returns_logits(self):
        model = PairSharedBackboneClassifier(
            model_name="deeper",
            dropout=0.3,
            in_channels=1,
            img_size=64,
            multi_task=True,
            pretrained=False,
        )
        x_2024 = torch.randn(2, 1, 64, 64)
        x_2025 = torch.randn(2, 1, 64, 64)

        logits_cls, logits_sev = model(x_2024, x_2025)

        self.assertEqual(tuple(logits_cls.shape), (2, 2))
        self.assertEqual(tuple(logits_sev.shape), (2, 1))

    def test_pair_shared_backbone_classifier_supports_resnet50(self):
        model = PairSharedBackboneClassifier(
            model_name="resnet50",
            dropout=0.3,
            in_channels=1,
            img_size=64,
            multi_task=False,
            pretrained=False,
        )
        self.assertIsInstance(model.backbone, ResNet50Backbone)
        x_2024 = torch.randn(2, 1, 64, 64)
        x_2025 = torch.randn(2, 1, 64, 64)

        logits_cls = model(x_2024, x_2025)

        self.assertEqual(tuple(logits_cls.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
