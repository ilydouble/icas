"""Unit tests for patient-level thermal + clinical comparison helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.compare_patient_level_thermal_clinical import (
    CLINICAL_TOP3,
    build_patient_probability_frame,
    build_sample_probability_frame,
)


class FrameBuilderTests(unittest.TestCase):
    def setUp(self):
        self.clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "waist_hip_ratio": [0.9, 0.8],
                "gender_encoded": [1, 0],
                "height": [170.0, 160.0],
            }
        )

    def test_build_patient_probability_frame_merges_probs_with_clinical_top3(self):
        patient_ids = np.array(["P1", "P2"])
        labels = np.array([1, 0])
        probs = np.array([0.7, 0.2])

        frame = build_patient_probability_frame(patient_ids, labels, probs, self.clinical_df)

        self.assertEqual(
            frame.columns.tolist(),
            ["canonical_patient_id", "label", "thermal_prob", *CLINICAL_TOP3],
        )
        self.assertEqual(frame["canonical_patient_id"].tolist(), ["P1", "P2"])

    def test_build_sample_probability_frame_filters_to_patient_pool(self):
        sample_ids = ["S1", "S2", "S3"]
        labels = np.array([1, 1, 0])
        probs = np.array([0.8, 0.6, 0.3])
        metadata = {
            "S1": {"canonical_patient_id": "P1", "year": 2024},
            "S2": {"canonical_patient_id": "P1", "year": 2025},
            "S3": {"canonical_patient_id": "P3", "year": 2025},
        }

        frame = build_sample_probability_frame(
            sample_ids,
            labels,
            probs,
            metadata,
            patient_pool={"P1"},
        )

        self.assertEqual(frame["canonical_patient_id"].tolist(), ["P1", "P1"])
        np.testing.assert_allclose(frame["thermal_prob"].to_numpy(), np.array([0.8, 0.6]))


if __name__ == "__main__":
    unittest.main()
