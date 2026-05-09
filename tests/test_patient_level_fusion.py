"""Unit tests for patient-level shallow fusion of thermal predictions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_patient_level_fusion import (
    aggregate_patient_predictions,
    build_sample_metadata_lookup,
    load_checkpoint_for_inference,
)
from scripts.train_cnn_v3 import MobileNetV3Small


class CheckpointLoadingTests(unittest.TestCase):
    def test_load_checkpoint_for_inference_ignores_extra_multitask_head(self):
        model = MobileNetV3Small(
            num_classes=2,
            dropout=0.3,
            in_channels=1,
            img_size=64,
            multi_task=False,
            soft_label=False,
            pretrained=False,
        )
        original_weight = model.state_dict()["classifier_head.1.weight"].clone()
        checkpoint_state = dict(model.state_dict())
        checkpoint_state["severity_head.weight"] = torch.randn(1, 576)
        checkpoint_state["severity_head.bias"] = torch.randn(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            torch.save({"model_state_dict": checkpoint_state}, ckpt_path)
            checkpoint = load_checkpoint_for_inference(model, ckpt_path, torch.device("cpu"))

        self.assertIn("model_state_dict", checkpoint)
        self.assertTrue(torch.allclose(model.state_dict()["classifier_head.1.weight"], original_weight))


class SampleMetadataTests(unittest.TestCase):
    def test_build_sample_metadata_lookup_keeps_patient_and_year(self):
        samples = [
            {"sample_id": "2024_P1_1", "canonical_patient_id": "P1", "year": 2024},
            {"sample_id": "2025_P1_1", "canonical_patient_id": "P1", "year": 2025},
        ]

        lookup = build_sample_metadata_lookup(samples)

        self.assertEqual(lookup["2024_P1_1"]["canonical_patient_id"], "P1")
        self.assertEqual(lookup["2025_P1_1"]["year"], 2025)


class PatientAggregationTests(unittest.TestCase):
    def setUp(self):
        self.sample_ids = ["2024_P1_1", "2025_P1_1", "2025_P1_2", "2025_P2_1"]
        self.probs = np.array([0.20, 0.70, 0.50, 0.40], dtype=float)
        self.labels = np.array([1, 1, 1, 0], dtype=int)
        self.metadata = {
            "2024_P1_1": {"canonical_patient_id": "P1", "year": 2024},
            "2025_P1_1": {"canonical_patient_id": "P1", "year": 2025},
            "2025_P1_2": {"canonical_patient_id": "P1", "year": 2025},
            "2025_P2_1": {"canonical_patient_id": "P2", "year": 2025},
        }

    def test_prob_mean_averages_patient_probabilities(self):
        patient_ids, y_true, y_prob = aggregate_patient_predictions(
            self.sample_ids,
            self.labels,
            self.probs,
            self.metadata,
            strategy="prob_mean",
        )

        self.assertEqual(patient_ids.tolist(), ["P1", "P2"])
        np.testing.assert_allclose(y_true, np.array([1, 0]))
        np.testing.assert_allclose(y_prob, np.array([0.46666667, 0.4]), atol=1e-6)

    def test_prob_max_uses_highest_sample_probability(self):
        _, _, y_prob = aggregate_patient_predictions(
            self.sample_ids,
            self.labels,
            self.probs,
            self.metadata,
            strategy="prob_max",
        )

        np.testing.assert_allclose(y_prob, np.array([0.7, 0.4]), atol=1e-6)

    def test_logit_mean_averages_logits_before_sigmoid(self):
        _, _, y_prob = aggregate_patient_predictions(
            self.sample_ids,
            self.labels,
            self.probs,
            self.metadata,
            strategy="logit_mean",
        )

        self.assertGreater(y_prob[0], 0.45)
        self.assertLess(y_prob[0], 0.46)

    def test_year_weighted_mean_prefers_2025_samples(self):
        _, _, y_prob = aggregate_patient_predictions(
            self.sample_ids,
            self.labels,
            self.probs,
            self.metadata,
            strategy="year_weighted_mean",
            year_2025_weight=0.7,
        )

        expected_p1 = (0.2 * 0.3 + 0.7 * 0.7 + 0.5 * 0.7) / (0.3 + 0.7 + 0.7)
        np.testing.assert_allclose(y_prob, np.array([expected_p1, 0.4]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
