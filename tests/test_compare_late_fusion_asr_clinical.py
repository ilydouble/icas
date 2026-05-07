"""Unit tests for late fusion between filtered ASR and top-k clinical branches."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.compare_late_fusion_asr_clinical import (
    blend_probabilities,
    build_late_fusion_frame,
    pick_best_alpha,
    select_top_k_features,
)


class FeatureSelectionTests(unittest.TestCase):
    def test_select_top_k_features_uses_selected_rows(self):
        score_df = pd.DataFrame(
            [
                {"feature_name": "f1", "combined_score": 0.3, "selected": 1},
                {"feature_name": "f2", "combined_score": 0.2, "selected": 1},
                {"feature_name": "f3", "combined_score": 0.4, "selected": 0},
            ]
        )
        chosen = select_top_k_features(score_df, top_k=2)
        self.assertEqual(chosen, ["f1", "f2"])


class MergeTests(unittest.TestCase):
    def test_build_late_fusion_frame_merges_requested_features(self):
        asr_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "clinical_match_status": ["matched", "matched"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "asr_a": [1.0, 2.0],
                "asr_b": [3.0, 4.0],
            }
        )
        cli_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "c_a": [5.0, 6.0],
                "c_b": [7.0, 8.0],
            }
        )
        merged = build_late_fusion_frame(asr_df, cli_df, ["asr_a"], ["c_a"])
        self.assertEqual(
            merged.columns.tolist(),
            ["canonical_patient_id", "clinical_match_status", "has_icas", "label", "stenosis_multiclass", "asr_a", "c_a"],
        )


class FusionRuleTests(unittest.TestCase):
    def test_blend_probabilities_is_convex_combination(self):
        asr = np.array([0.2, 0.6])
        clinical = np.array([0.8, 0.4])
        out = blend_probabilities(asr, clinical, alpha=0.25)
        np.testing.assert_allclose(out, np.array([0.65, 0.45]))

    def test_pick_best_alpha_prefers_best_auc(self):
        y_val = np.array([0, 0, 1, 1])
        asr_prob = np.array([0.2, 0.3, 0.7, 0.8])
        clinical_prob = np.array([0.4, 0.6, 0.5, 0.7])
        alpha, auc = pick_best_alpha(asr_prob, clinical_prob, y_val, alpha_grid=[0.0, 0.5, 1.0])
        self.assertIn(alpha, {0.5, 1.0})
        self.assertGreaterEqual(auc, 0.99)


if __name__ == "__main__":
    unittest.main()
