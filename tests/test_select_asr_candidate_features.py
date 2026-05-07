"""Unit tests for selecting an ASR candidate feature subset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.select_asr_candidate_features import (
    apply_correlation_pruning,
    build_modeling_subset,
    select_candidate_features,
    write_selection_outputs,
)


class CandidateSelectionTests(unittest.TestCase):
    def test_select_candidate_features_prefers_high_score_and_deduplicates(self):
        scores = pd.DataFrame(
            [
                {"feature_name": "asr_speed", "combined_score": 0.20},
                {"feature_name": "asr_speed_dup", "combined_score": 0.19},
                {"feature_name": "asr_pause", "combined_score": 0.15},
                {"feature_name": "asr_emotion", "combined_score": 0.05},
            ]
        )
        feature_df = pd.DataFrame(
            {
                "asr_speed": [1, 2, 3, 4],
                "asr_speed_dup": [1, 2, 3, 4],
                "asr_pause": [4, 1, 3, 2],
                "asr_emotion": [0.1, 0.2, 0.3, 0.4],
            }
        )
        selected = select_candidate_features(
            score_df=scores,
            feature_df=feature_df,
            top_k=3,
            corr_threshold=0.95,
        )
        self.assertEqual(selected, ["asr_speed", "asr_pause"])

    def test_apply_correlation_pruning_keeps_first_ranked_feature(self):
        ranked = ["asr_a", "asr_b", "asr_c"]
        feature_df = pd.DataFrame(
            {
                "asr_a": [1, 2, 3, 4],
                "asr_b": [1, 2, 3, 4],
                "asr_c": [4, 3, 2, 1],
            }
        )
        selected = apply_correlation_pruning(ranked, feature_df, corr_threshold=0.99)
        self.assertEqual(selected, ["asr_a"])


class ModelingSubsetTests(unittest.TestCase):
    def test_build_modeling_subset_keeps_metadata_labels_and_selected_features(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "clinical_match_status": ["matched", "matched"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "asr_speed": [100, 80],
                "asr_pause": [0.1, 0.2],
                "name": ["a", "b"],
            }
        )
        subset = build_modeling_subset(df, ["asr_speed", "asr_pause"])
        self.assertEqual(
            subset.columns.tolist(),
            [
                "canonical_patient_id",
                "clinical_match_status",
                "has_icas",
                "label",
                "stenosis_multiclass",
                "asr_speed",
                "asr_pause",
            ],
        )


class OutputTests(unittest.TestCase):
    def test_write_selection_outputs_creates_list_and_subset_csv(self):
        score_df = pd.DataFrame(
            [
                {"feature_name": "asr_speed", "combined_score": 0.2},
                {"feature_name": "asr_pause", "combined_score": 0.1},
            ]
        )
        feature_df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2", "P3", "P4"],
                "clinical_match_status": ["matched", "matched", "matched", "matched"],
                "has_icas": [0, 1, 0, 1],
                "label": [0, 1, 0, 1],
                "stenosis_multiclass": [0, 2, 1, 3],
                "asr_speed": [100, 80, 95, 78],
                "asr_pause": [0.1, 0.2, 0.15, 0.12],
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
            self.assertEqual(selected, ["asr_speed", "asr_pause"])
            self.assertTrue((tmpdir / "list.csv").exists())
            self.assertTrue((tmpdir / "subset.csv").exists())
            self.assertIn("selected", list_df.columns)
            self.assertEqual(int(list_df["selected"].sum()), 2)
            self.assertEqual(subset_df.shape[1], 7)


if __name__ == "__main__":
    unittest.main()
