"""Unit tests for deep-feature and clinical fusion helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.compare_fusion_models import (
    build_feature_sets,
    select_clinical_feature_columns,
)


class ClinicalColumnSelectionTests(unittest.TestCase):
    def test_select_clinical_feature_columns_excludes_leakage_and_text(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1"],
                "label": [1],
                "has_icas": [1],
                "stenosis_multiclass": [2],
                "icas_detail": ["detail"],
                "name": ["patient"],
                "gender": ["男"],
                "data_sources": ["clinical"],
                "gender_encoded": [0],
                "age": [60],
                "bmi": [24.1],
                "image_count": [3],
                "images_2024": [1],
            }
        )
        cols = select_clinical_feature_columns(df)
        self.assertEqual(cols, ["gender_encoded", "age", "bmi"])


class FeatureSetTests(unittest.TestCase):
    def test_build_feature_sets_returns_deep_clinical_and_fusion(self):
        feature_sets = build_feature_sets(
            deep_cols=["deep_000", "deep_001", "cnn_prob"],
            clinical_cols=["age", "bmi"],
        )
        self.assertEqual(feature_sets["deep_only"], ["deep_000", "deep_001", "cnn_prob"])
        self.assertEqual(feature_sets["clinical_only"], ["age", "bmi"])
        self.assertEqual(
            feature_sets["fusion"],
            ["deep_000", "deep_001", "cnn_prob", "age", "bmi"],
        )


if __name__ == "__main__":
    unittest.main()
