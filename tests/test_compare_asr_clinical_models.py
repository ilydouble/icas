"""Unit tests for ASR/clinical baseline comparison helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.compare_asr_clinical_models import (
    apply_split_feature_set,
    build_feature_sets,
    build_fusion_frame,
    load_split,
    normalize_feature_set_names,
    select_clinical_feature_columns,
)


class ClinicalColumnSelectionTests(unittest.TestCase):
    def test_select_clinical_feature_columns_excludes_labels_and_text(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1"],
                "name": ["张三"],
                "gender": ["男"],
                "label": [1],
                "has_icas": [1],
                "stenosis_multiclass": [2],
                "images_2025": [2],
                "gender_encoded": [0],
                "age": [61],
                "bmi": [24.5],
            }
        )
        cols = select_clinical_feature_columns(df)
        self.assertEqual(cols, ["gender_encoded", "age", "bmi"])


class FusionFrameTests(unittest.TestCase):
    def test_build_fusion_frame_merges_asr_and_clinical(self):
        asr_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "clinical_match_status": ["matched", "matched"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "asr_speed": [100, 80],
            }
        )
        clinical_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "age": [60, 70],
                "bmi": [24.0, 26.0],
            }
        )
        merged = build_fusion_frame(asr_df, clinical_df, ["age", "bmi"])
        self.assertEqual(merged["age"].tolist(), [60, 70])
        self.assertEqual(merged["bmi"].tolist(), [24.0, 26.0])


class FeatureSetTests(unittest.TestCase):
    def test_build_feature_sets_returns_asr_clinical_and_fusion(self):
        feature_sets = build_feature_sets(["asr_speed", "asr_pause"], ["age", "bmi"])
        self.assertEqual(feature_sets["asr_only"], ["asr_speed", "asr_pause"])
        self.assertEqual(feature_sets["clinical_only"], ["age", "bmi"])
        self.assertEqual(feature_sets["fusion"], ["asr_speed", "asr_pause", "age", "bmi"])

    def test_normalize_feature_set_names_defaults_to_all(self):
        names = normalize_feature_set_names(None, ["asr_only", "clinical_only", "fusion"])
        self.assertEqual(names, ["asr_only", "clinical_only", "fusion"])


class SplitTests(unittest.TestCase):
    def test_load_split_reads_patient_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({
                "train_patient_ids": ["P1"],
                "val_patient_ids": ["P2"],
                "test_patient_ids": ["P3"],
            }), encoding="utf-8")
            split = load_split(path)
            self.assertEqual(split["train_patient_ids"], ["P1"])

    def test_apply_split_feature_set_uses_canonical_patient_id(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2", "P3"],
                "label": [0, 1, 0],
                "stenosis_multiclass": [0, 2, 1],
                "asr_speed": [100, 80, 95],
                "age": [60, 70, 66],
            }
        )
        split = {
            "train_patient_ids": ["P1"],
            "val_patient_ids": ["P2"],
            "test_patient_ids": ["P3"],
        }
        X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te = apply_split_feature_set(
            df=df,
            split=split,
            feature_cols=["asr_speed", "age"],
            patient_id_col="canonical_patient_id",
        )
        self.assertEqual(X_tr.tolist(), [[100.0, 60.0]])
        self.assertEqual(y_tr.tolist(), [0])
        self.assertEqual(sev_tr.tolist(), [0])
        self.assertEqual(X_va.tolist(), [[80.0, 70.0]])
        self.assertEqual(y_te.tolist(), [0])
        self.assertEqual(groups_tr.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
