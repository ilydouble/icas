"""Unit tests for thermal + clinical late-fusion helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.compare_thermal_clinical_late_fusion import (
    CLINICAL_TOP3,
    build_probability_frame,
    make_meta_features,
    pick_best_alpha,
)


class ProbabilityFrameTests(unittest.TestCase):
    def test_build_probability_frame_merges_cnn_prob_and_clinical_top3(self):
        deep_df = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "patient_id": ["P1", "P2"],
                "label": [1, 0],
                "stenosis_multiclass": [2, 0],
                "cnn_prob": [0.7, 0.2],
            }
        )
        clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "waist_hip_ratio": [0.9, 0.8],
                "gender_encoded": [1, 0],
                "height": [170.0, 160.0],
            }
        )

        merged = build_probability_frame(deep_df, clinical_df, {"P1", "P2"})

        self.assertEqual(
            merged.columns.tolist(),
            ["sample_id", "patient_id", "label", "stenosis_multiclass", "cnn_prob", *CLINICAL_TOP3],
        )
        self.assertEqual(merged["patient_id"].tolist(), ["P1", "P2"])


class AlphaSelectionTests(unittest.TestCase):
    def test_pick_best_alpha_maximizes_validation_auc(self):
        thermal = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
        clinical = np.array([0.4, 0.6, 0.5, 0.7], dtype=float)
        y_val = np.array([0, 1, 0, 1], dtype=int)

        alpha, auc = pick_best_alpha(thermal, clinical, y_val, [0.0, 0.5, 1.0])

        self.assertIn(alpha, {0.0, 0.5, 1.0})
        self.assertGreaterEqual(auc, 0.5)


class MetaFeatureTests(unittest.TestCase):
    def test_make_meta_features_stacks_branch_probabilities(self):
        thermal = np.array([0.2, 0.8], dtype=float)
        clinical = np.array([0.3, 0.7], dtype=float)

        X = make_meta_features(thermal, clinical)

        self.assertEqual(X.shape, (2, 2))
        self.assertTrue(np.allclose(X[:, 0], thermal))
        self.assertTrue(np.allclose(X[:, 1], clinical))


if __name__ == "__main__":
    unittest.main()
