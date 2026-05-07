"""Unit tests for complete-case modality comparison helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.compare_complete_case_modalities import (
    CLINICAL_TOP3,
    build_complete_case_frame,
    load_complete_case_patient_ids,
    restrict_split_to_patient_pool,
)


class CompleteCasePatientTests(unittest.TestCase):
    def test_load_complete_case_patient_ids_intersects_asr_and_clinical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asr_csv = root / "asr.csv"
            clinical_csv = root / "clinical.csv"

            pd.DataFrame({"canonical_patient_id": ["P1", "P2", "P3"]}).to_csv(asr_csv, index=False)
            pd.DataFrame({"canonical_patient_id": ["P2", "P3", "P4"]}).to_csv(clinical_csv, index=False)

            patient_ids = load_complete_case_patient_ids(asr_csv, clinical_csv)

            self.assertEqual(patient_ids, {"P2", "P3"})


class SplitRestrictionTests(unittest.TestCase):
    def test_restrict_split_to_patient_pool_keeps_only_complete_case_ids(self):
        split = {
            "train_patient_ids": ["P1", "P2", "P3"],
            "val_patient_ids": ["P4", "P5"],
            "test_patient_ids": ["P6", "P7"],
            "summary": {"train": {"n_patients": 3}},
        }

        restricted = restrict_split_to_patient_pool(split, {"P2", "P5", "P7"})

        self.assertEqual(restricted["train_patient_ids"], ["P2"])
        self.assertEqual(restricted["val_patient_ids"], ["P5"])
        self.assertEqual(restricted["test_patient_ids"], ["P7"])
        self.assertEqual(restricted["summary"], split["summary"])


class CompleteCaseFrameTests(unittest.TestCase):
    def test_build_complete_case_frame_merges_deep_and_clinical_top3(self):
        deep_df = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "patient_id": ["P1", "P2", "P3"],
                "label": [1, 0, 1],
                "stenosis_multiclass": [2, 0, 3],
                "cnn_prob": [0.7, 0.2, 0.8],
                "cnn_severity_pred": [0.3, 0.1, 0.9],
                "deep_000": [0.1, 0.2, 0.3],
            }
        )
        clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2", "P4"],
                "waist_hip_ratio": [0.9, 0.8, 0.7],
                "gender_encoded": [1, 0, 1],
                "height": [170.0, 160.0, 168.0],
            }
        )

        merged = build_complete_case_frame(deep_df, clinical_df, {"P1", "P2"})

        self.assertEqual(merged["patient_id"].tolist(), ["P1", "P2"])
        for col in CLINICAL_TOP3:
            self.assertIn(col, merged.columns)


if __name__ == "__main__":
    unittest.main()
