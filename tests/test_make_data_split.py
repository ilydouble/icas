"""Tests for make_data_split and compare_models utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_data_split import (
    load_patients,
    split_summary,
    stratified_patient_split,
)
from scripts.compare_models import (
    apply_split,
    binary_metrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_clinical(path: Path, patients: list[dict]) -> None:
    pd.DataFrame(patients).to_csv(path, index=False)


def _write_features(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_clinical_row(pid: str, label, stenosis="0") -> dict:
    return {
        "canonical_patient_id": pid,
        "has_basic_clinical_data": 1,
        "data_sources": "clinical",
        "clinical_source_available": 1,
        "multimodal_source_available": 1,
        "name": "test",
        "gender": "男", "gender_encoded": 0,
        "age": 50, "height": 170, "weight": 60,
        "waist": 80, "hip": 90, "neck": 35,
        "bmi": 20, "waist_hip_ratio": 0.8,
        "waist_height_ratio": 0.5,
        "neck_height_ratio": 0.2,
        "bmi_category": 1, "age_group": 2,
        "has_icas": 0 if label == 0 else 1,
        "label": label,
        "stenosis_multiclass": stenosis,
        "icas_detail": "",
        "image_count": 2,
        "images_2024": 1,
        "images_2025": 1,
    }


def _make_feature_row(sample_id: str, patient_id: str, extra: int = 0) -> dict:
    return {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "year": 2024,
        "status": "ok",
        "face_mean": 34.0 + extra,
        "face_std": 0.5,
    }


class SplitNoLeakageTests(unittest.TestCase):
    """Core invariant: no patient appears in two splits."""

    def _make_patients_df(self, n_pos: int, n_neg: int) -> pd.DataFrame:
        rows = []
        for i in range(n_pos):
            rows.append({"canonical_patient_id": f"POS{i:03d}", "label": 1.0,
                         "stenosis_multiclass": "3", "feature_sample_count": 2})
        for i in range(n_neg):
            rows.append({"canonical_patient_id": f"NEG{i:03d}", "label": 0.0,
                         "stenosis_multiclass": "0", "feature_sample_count": 2})
        return pd.DataFrame(rows)

    def test_no_patient_in_two_splits(self):
        df = self._make_patients_df(60, 140)
        train, val, test, unlabelled = stratified_patient_split(df, seed=42)
        sets = [set(train), set(val), set(test)]
        # Pairwise intersection must be empty
        self.assertEqual(sets[0] & sets[1], set())
        self.assertEqual(sets[0] & sets[2], set())
        self.assertEqual(sets[1] & sets[2], set())

    def test_all_labelled_patients_assigned(self):
        df = self._make_patients_df(40, 60)
        train, val, test, unlabelled = stratified_patient_split(df, seed=42)
        total_in_splits = len(train) + len(val) + len(test) + len(unlabelled)
        self.assertEqual(total_in_splits, len(df))

    def test_unlabelled_patients_isolated(self):
        df = self._make_patients_df(30, 70)
        df = pd.concat([df, pd.DataFrame([{
            "canonical_patient_id": "UNLABELLED",
            "label": float("nan"),
            "stenosis_multiclass": None,
            "feature_sample_count": 1,
        }])], ignore_index=True)
        train, val, test, unlabelled = stratified_patient_split(df, seed=42)
        self.assertIn("UNLABELLED", unlabelled)
        self.assertNotIn("UNLABELLED", train)
        self.assertNotIn("UNLABELLED", val)
        self.assertNotIn("UNLABELLED", test)

    def test_fractions_are_approximately_correct(self):
        df = self._make_patients_df(80, 120)  # 200 total labelled
        train, val, test, _ = stratified_patient_split(
            df, train_frac=0.75, val_frac=0.10, seed=42)
        n = len(train) + len(val) + len(test)
        self.assertAlmostEqual(len(train) / n, 0.75, delta=0.03)
        self.assertAlmostEqual(len(val) / n, 0.10, delta=0.03)
        self.assertAlmostEqual(len(test) / n, 0.15, delta=0.03)

    def test_positive_rate_preserved_in_each_split(self):
        df = self._make_patients_df(40, 60)
        train, val, test, _ = stratified_patient_split(df, seed=42)
        for ids in (train, val, test):
            sub = df[df["canonical_patient_id"].isin(ids)]
            pos_rate = sub["label"].mean()
            # Should be close to global 40% ± 10pp
            self.assertAlmostEqual(pos_rate, 0.40, delta=0.12)

    def test_split_is_reproducible(self):
        df = self._make_patients_df(50, 50)
        r1 = stratified_patient_split(df, seed=7)
        r2 = stratified_patient_split(df, seed=7)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[1], r2[1])
        self.assertEqual(r1[2], r2[2])

    def test_different_seeds_produce_different_splits(self):
        df = self._make_patients_df(50, 50)
        r1 = stratified_patient_split(df, seed=1)
        r2 = stratified_patient_split(df, seed=99)
        self.assertNotEqual(set(r1[0]), set(r2[0]))


class LoadPatientsTests(unittest.TestCase):
    def test_load_patients_joins_sample_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "clinical.csv"
            fp = Path(tmp) / "features.csv"
            _write_clinical(cp, [_make_clinical_row("P001", 0), _make_clinical_row("P002", 1)])
            _write_features(fp, [
                _make_feature_row("P001_1", "P001"),
                _make_feature_row("P001_2", "P001"),
                _make_feature_row("P002_1", "P002"),
            ])
            df = load_patients(cp, fp)
            p1 = df[df["canonical_patient_id"] == "P001"].iloc[0]
            self.assertEqual(int(p1["feature_sample_count"]), 2)


class ApplySplitTests(unittest.TestCase):
    def test_apply_split_returns_correct_sizes(self):
        patients = ["P001", "P002", "P003", "P004"]
        feat_rows = []
        for i, pid in enumerate(patients):
            feat_rows.append({
                "sample_id": f"{pid}_1", "patient_id": pid, "year": 2024,
                "status": "ok", "feat_a": float(i),
            })
        df = pd.DataFrame(feat_rows)
        df["label"] = [0, 1, 0, 1]
        df["stenosis_multiclass"] = [0, 3, 0, 3]
        df["canonical_patient_id"] = df["patient_id"]

        split = {
            "train_patient_ids": ["P001", "P002"],
            "val_patient_ids": ["P003"],
            "test_patient_ids": ["P004"],
        }
        X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te = apply_split(df, split)
        self.assertEqual(len(X_tr), 2)
        self.assertEqual(len(groups_tr), 2)  # one group index per training sample
        self.assertEqual(len(X_va), 1)
        self.assertEqual(len(X_te), 1)


class MetricsTests(unittest.TestCase):
    def test_binary_metrics_perfect_prediction(self):
        y = np.array([0, 0, 1, 1])
        m = binary_metrics(y, y, np.array([0.0, 0.0, 1.0, 1.0]), prefix="val")
        self.assertEqual(m["val_acc"], 1.0)
        self.assertEqual(m["val_f1"], 1.0)
        self.assertAlmostEqual(m["val_auc_roc"], 1.0)


if __name__ == "__main__":
    unittest.main()
