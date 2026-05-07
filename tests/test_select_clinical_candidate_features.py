"""Unit tests for selecting clinical candidate feature subsets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.select_clinical_candidate_features import (
    apply_correlation_pruning,
    build_modeling_subset,
    select_candidate_features,
    write_selection_outputs,
)


class CandidateSelectionTests(unittest.TestCase):
    def test_select_candidate_features_prefers_high_score_and_deduplicates(self):
        scores = pd.DataFrame(
            [
                {"feature_name": "age", "combined_score": 0.20},
                {"feature_name": "age_dup", "combined_score": 0.19},
                {"feature_name": "bmi", "combined_score": 0.15},
                {"feature_name": "waist", "combined_score": 0.10},
            ]
        )
        feature_df = pd.DataFrame(
            {
                "age": [50, 60, 70, 80],
                "age_dup": [50, 60, 70, 80],
                "bmi": [22.0, 27.0, 24.0, 30.0],
                "waist": [75, 82, 88, 99],
            }
        )
        selected = select_candidate_features(scores, feature_df, top_k=3, corr_threshold=0.95)
        self.assertEqual(selected, ["age", "bmi"])

    def test_apply_correlation_pruning_keeps_first_ranked_feature(self):
        ranked = ["age", "age_dup", "bmi"]
        feature_df = pd.DataFrame(
            {
                "age": [50, 60, 70, 80],
                "age_dup": [50, 60, 70, 80],
                "bmi": [22.0, 27.0, 24.0, 30.0],
            }
        )
        selected = apply_correlation_pruning(ranked, feature_df, corr_threshold=0.99)
        self.assertEqual(selected, ["age", "bmi"])


class ModelingSubsetTests(unittest.TestCase):
    def test_build_modeling_subset_keeps_ids_labels_and_selected_features(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "age": [60, 70],
                "bmi": [24.0, 28.0],
            }
        )
        subset = build_modeling_subset(df, ["age", "bmi"])
        self.assertEqual(
            subset.columns.tolist(),
            ["canonical_patient_id", "has_icas", "label", "stenosis_multiclass", "age", "bmi"],
        )


class OutputTests(unittest.TestCase):
    def test_write_selection_outputs_creates_files(self):
        score_df = pd.DataFrame(
            [
                {"feature_name": "age", "combined_score": 0.2},
                {"feature_name": "bmi", "combined_score": 0.1},
            ]
        )
        feature_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2", "P3", "P4"],
                "has_icas": [0, 1, 0, 1],
                "label": [0, 1, 0, 1],
                "stenosis_multiclass": [0, 2, 1, 3],
                "age": [60, 70, 62, 74],
                "bmi": [24.0, 28.0, 23.0, 29.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            selected, list_df, subset_df = write_selection_outputs(
                score_df=score_df,
                feature_df=feature_df,
                feature_list_csv=tmpdir / "list.csv",
                modeling_subset_csv=tmpdir / "subset.csv",
                top_k=2,
                corr_threshold=0.95,
            )
            self.assertEqual(selected, ["age", "bmi"])
            self.assertTrue((tmpdir / "list.csv").exists())
            self.assertTrue((tmpdir / "subset.csv").exists())
            self.assertEqual(int(list_df["selected"].sum()), 2)
            self.assertEqual(subset_df.shape[1], 6)


if __name__ == "__main__":
    unittest.main()
