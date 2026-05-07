"""Unit tests for filtered ASR + clinical baseline comparison helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.compare_filtered_asr_clinical_models import (
    build_feature_sets,
    build_filtered_fusion_frame,
    extract_feature_columns,
)


class MergeTests(unittest.TestCase):
    def test_build_filtered_fusion_frame_inner_joins_on_patient_id(self):
        asr_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "asr_speed": [100, 80],
            }
        )
        clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P3"],
                "has_icas": [0, 0],
                "label": [0, 0],
                "stenosis_multiclass": [0, 1],
                "bmi": [24.0, 28.0],
            }
        )
        merged = build_filtered_fusion_frame(asr_df, clinical_df)
        self.assertEqual(merged["canonical_patient_id"].tolist(), ["P1"])
        self.assertIn("asr_speed", merged.columns)
        self.assertIn("bmi", merged.columns)


class FeatureColumnTests(unittest.TestCase):
    def test_extract_feature_columns_splits_asr_and_clinical(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1"],
                "has_icas": [0],
                "label": [0],
                "stenosis_multiclass": [0],
                "asr_speed": [100],
                "asr_pause": [0.1],
                "bmi": [24.0],
                "waist_hip_ratio": [0.9],
            }
        )
        asr_cols, clinical_cols = extract_feature_columns(df)
        self.assertEqual(asr_cols, ["asr_speed", "asr_pause"])
        self.assertEqual(clinical_cols, ["bmi", "waist_hip_ratio"])

    def test_build_feature_sets_returns_filtered_fusion(self):
        feature_sets = build_feature_sets(["asr_speed"], ["bmi", "waist"])
        self.assertEqual(feature_sets["asr_only"], ["asr_speed"])
        self.assertEqual(feature_sets["clinical_only"], ["bmi", "waist"])
        self.assertEqual(feature_sets["filtered_fusion"], ["asr_speed", "bmi", "waist"])


if __name__ == "__main__":
    unittest.main()
