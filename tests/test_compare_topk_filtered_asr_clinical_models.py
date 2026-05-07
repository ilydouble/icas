"""Unit tests for top-k filtered ASR/clinical comparison helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.compare_topk_filtered_asr_clinical_models import (
    build_feature_sets,
    build_topk_fusion_frame,
    select_top_k_features,
)


class TopKSelectionTests(unittest.TestCase):
    def test_select_top_k_features_uses_selected_rows_and_score_order(self):
        score_df = pd.DataFrame(
            [
                {"feature_name": "f1", "combined_score": 0.30, "selected": 1},
                {"feature_name": "f2", "combined_score": 0.20, "selected": 1},
                {"feature_name": "f3", "combined_score": 0.10, "selected": 0},
                {"feature_name": "f4", "combined_score": 0.15, "selected": 1},
            ]
        )
        chosen = select_top_k_features(score_df, top_k=2)
        self.assertEqual(chosen, ["f1", "f2"])


class FusionFrameTests(unittest.TestCase):
    def test_build_topk_fusion_frame_keeps_only_requested_feature_columns(self):
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
        merged = build_topk_fusion_frame(asr_df, cli_df, ["asr_a"], ["c_a"])
        self.assertEqual(
            merged.columns.tolist(),
            ["canonical_patient_id", "clinical_match_status", "has_icas", "label", "stenosis_multiclass", "asr_a", "c_a"],
        )


class FeatureSetTests(unittest.TestCase):
    def test_build_feature_sets_returns_topk_fusion(self):
        feature_sets = build_feature_sets(["asr_a"], ["c_a"])
        self.assertEqual(feature_sets["asr_only_topk"], ["asr_a"])
        self.assertEqual(feature_sets["clinical_only_topk"], ["c_a"])
        self.assertEqual(feature_sets["topk_fusion"], ["asr_a", "c_a"])


if __name__ == "__main__":
    unittest.main()
