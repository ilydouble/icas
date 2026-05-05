"""Unit tests for train_cnn_v2 utilities."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.train_cnn_v2 import (
    EarlyStopping,
    aggregate_patient_predictions,
    apply_face_mask,
    compute_selection_score,
    find_best_threshold,
    severity_to_regression_target,
)


class SeverityTargetTests(unittest.TestCase):
    def test_severity_targets_are_normalized_for_positive_cases(self):
        values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        targets = severity_to_regression_target(values)
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(targets, expected))


class ApplyFaceMaskTests(unittest.TestCase):
    def test_apply_face_mask_handles_missing_mask(self):
        temp = np.array([[20.0, 25.0], [30.0, 35.0]], dtype=np.float32)
        out = apply_face_mask(temp, None, target_size=(2, 2))
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.isfinite(out).all())


class EarlyStoppingTests(unittest.TestCase):
    def test_early_stopping_waits_for_patience_after_min_epochs(self):
        stopper = EarlyStopping(patience=2, min_epochs=3, min_delta=1e-4)
        self.assertFalse(stopper.step(1, 0.60))
        self.assertFalse(stopper.step(2, 0.62))
        self.assertFalse(stopper.step(3, 0.61))
        self.assertFalse(stopper.step(4, 0.61))
        self.assertTrue(stopper.step(5, 0.61))


class ThresholdSelectionTests(unittest.TestCase):
    def test_find_best_threshold_maximizes_f1(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_prob = np.array([0.2, 0.4, 0.6, 0.9], dtype=np.float32)
        threshold, metrics = find_best_threshold(y_true, y_prob)
        self.assertAlmostEqual(threshold, 0.5, places=6)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=6)


class PatientAggregationTests(unittest.TestCase):
    def test_patient_aggregation_uses_mean_probability(self):
        sample_ids = ["A_1", "A_2", "B_1"]
        y_true = np.array([1, 1, 0], dtype=np.int64)
        y_prob = np.array([0.8, 0.4, 0.2], dtype=np.float32)
        sample_to_patient = {"A_1": "A", "A_2": "A", "B_1": "B"}
        patient_ids, patient_labels, patient_probs = aggregate_patient_predictions(
            sample_ids, y_true, y_prob, sample_to_patient, method="mean"
        )
        self.assertEqual(patient_ids, ["A", "B"])
        np.testing.assert_allclose(patient_labels, np.array([1, 0], dtype=np.int64))
        np.testing.assert_allclose(patient_probs, np.array([0.6, 0.2], dtype=np.float32))

    def test_patient_aggregation_supports_max_and_top2_mean(self):
        sample_ids = ["A_1", "A_2", "A_3", "B_1"]
        y_true = np.array([1, 1, 1, 0], dtype=np.int64)
        y_prob = np.array([0.8, 0.4, 0.6, 0.2], dtype=np.float32)
        sample_to_patient = {"A_1": "A", "A_2": "A", "A_3": "A", "B_1": "B"}

        _, _, patient_probs_max = aggregate_patient_predictions(
            sample_ids, y_true, y_prob, sample_to_patient, method="max"
        )
        _, _, patient_probs_top2 = aggregate_patient_predictions(
            sample_ids, y_true, y_prob, sample_to_patient, method="top2_mean"
        )

        np.testing.assert_allclose(patient_probs_max, np.array([0.8, 0.2], dtype=np.float32))
        np.testing.assert_allclose(patient_probs_top2, np.array([0.7, 0.2], dtype=np.float32))


class SelectionScoreTests(unittest.TestCase):
    def test_selection_score_can_use_patient_level_metric(self):
        metrics = {
            "auc_roc": 0.61,
            "f1": 0.48,
            "patient_auc_roc": 0.73,
            "patient_f1": 0.38,
        }
        self.assertAlmostEqual(compute_selection_score(metrics, "auc_roc"), 0.61)
        self.assertAlmostEqual(compute_selection_score(metrics, "patient_auc_roc"), 0.73)


if __name__ == "__main__":
    unittest.main()
