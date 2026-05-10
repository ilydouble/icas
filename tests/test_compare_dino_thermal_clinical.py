"""Unit tests for DINO thermal + clinical comparison helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.compare_dino_thermal_clinical import build_dino_fusion_frame


class DinoFusionFrameTests(unittest.TestCase):
    def test_build_dino_fusion_frame_merges_features_with_clinical_columns(self):
        dino_df = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "patient_id": ["P1", "P2"],
                "year": [2024, 2025],
                "label": [1, 0],
                "stenosis_multiclass": [2, 0],
                "dino_000": [0.1, 0.2],
                "dino_001": [0.3, 0.4],
            }
        )
        clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "age": [60, 55],
                "bmi": [24.0, 22.0],
            }
        )

        merged = build_dino_fusion_frame(dino_df, clinical_df, ["age", "bmi"])

        self.assertEqual(
            merged.columns.tolist(),
            ["sample_id", "patient_id", "year", "label", "stenosis_multiclass", "dino_000", "dino_001", "age", "bmi"],
        )

    def test_build_dino_fusion_frame_coerces_clinical_values_to_numeric(self):
        dino_df = pd.DataFrame(
            {
                "sample_id": ["S1"],
                "patient_id": ["P1"],
                "year": [2024],
                "label": [1],
                "stenosis_multiclass": [2],
                "dino_000": [0.1],
            }
        )
        clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1"],
                "age": ["60"],
                "bmi": ["not_a_number"],
            }
        )

        merged = build_dino_fusion_frame(dino_df, clinical_df, ["age", "bmi"])

        self.assertEqual(float(merged.loc[0, "age"]), 60.0)
        self.assertTrue(pd.isna(merged.loc[0, "bmi"]))


if __name__ == "__main__":
    unittest.main()
