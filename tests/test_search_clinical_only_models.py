"""Unit tests for refined clinical-only search helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.search_clinical_only_models import (
    build_feature_sets,
    build_summary,
    parse_feature_set_sizes,
    ranked_features_from_scores,
    tuned_threshold,
)


class RankedFeatureTests(unittest.TestCase):
    def test_ranked_features_respects_scores_and_numeric_columns(self):
        feature_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2", "P3"],
                "label": [0, 1, 0],
                "stenosis_multiclass": [0, 2, 1],
                "gender": ["男", "女", "男"],
                "age": [60, 70, 65],
                "height": [170, 160, 168],
                "constant": [1, 1, 1],
            }
        )
        score_df = pd.DataFrame(
            {
                "feature_name": ["constant", "height", "age"],
                "combined_score": [0.9, 0.8, 0.7],
                "binary_corr_abs": [0.9, 0.8, 0.7],
                "severity_spearman_abs": [0.9, 0.8, 0.7],
            }
        )
        ranked = ranked_features_from_scores(score_df, feature_df)
        self.assertEqual(ranked, ["height", "age"])


class FeatureSetTests(unittest.TestCase):
    def test_parse_feature_set_sizes_supports_all_alias(self):
        self.assertEqual(parse_feature_set_sizes("3,5,all"), [3, 5, "all"])
        self.assertEqual(parse_feature_set_sizes("3, full"), [3, "all"])

    def test_build_feature_sets_clamps_sizes(self):
        feature_sets = build_feature_sets(["a", "b", "c"], [2, 5, "all"])
        self.assertEqual(feature_sets["top_2"], ["a", "b"])
        self.assertEqual(feature_sets["top_3"], ["a", "b", "c"])
        self.assertEqual(feature_sets["top_all"], ["a", "b", "c"])


class ThresholdTests(unittest.TestCase):
    def test_tuned_threshold_returns_search_result(self):
        prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        y_true = np.array([0, 0, 1, 1], dtype=int)
        threshold, score = tuned_threshold(prob, y_true, "f1")
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLessEqual(threshold, 0.9)
        self.assertGreaterEqual(score, 0.99)


class SummaryTests(unittest.TestCase):
    def test_build_summary_tracks_best_auc_and_tuned_f1(self):
        df = pd.DataFrame(
            [
                {"model": "A", "test_auc_roc": 0.70, "test_tuned_f1": 0.60},
                {"model": "B", "test_auc_roc": 0.75, "test_tuned_f1": 0.58},
                {"model": "C", "test_auc_roc": 0.72, "test_tuned_f1": 0.66},
            ]
        )
        summary = build_summary(df)
        self.assertEqual(summary["best_auc_row"]["model"], "B")
        self.assertEqual(summary["best_tuned_f1_row"]["model"], "C")


if __name__ == "__main__":
    unittest.main()
