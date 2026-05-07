"""Unit tests for ASR feature correlation analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_asr_feature_correlations import (
    build_feature_score_table,
    select_asr_numeric_feature_columns,
    write_analysis_outputs,
)


class FeatureSelectionTests(unittest.TestCase):
    def test_select_asr_numeric_feature_columns_keeps_only_numeric_asr_features(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "asr_speech_rate_mean": [100.0, 120.0],
                "asr_pause_sentence_ratio": [0.1, 0.2],
                "asr_transcript": ["a", "b"],
                "label": [0, 1],
                "name": ["x", "y"],
            }
        )
        cols = select_asr_numeric_feature_columns(df)
        self.assertEqual(cols, ["asr_speech_rate_mean", "asr_pause_sentence_ratio"])


class ScoreTableTests(unittest.TestCase):
    def test_build_feature_score_table_ranks_signal_features(self):
        df = pd.DataFrame(
            {
                "clinical_match_status": ["matched"] * 6,
                "has_icas": [0, 0, 0, 1, 1, 1],
                "label": [0, 0, 0, 1, 1, 1],
                "stenosis_multiclass": [0, 0, 1, 2, 3, 3],
                "asr_speech_rate_mean": [220, 210, 205, 140, 130, 120],
                "asr_pause_sentence_ratio": [0.02, 0.03, 0.04, 0.21, 0.25, 0.30],
                "asr_noise_feature": [1, 2, 1, 2, 1, 2],
            }
        )
        scores = build_feature_score_table(df)
        self.assertGreaterEqual(len(scores), 3)
        top = scores.iloc[0]
        self.assertIn(top["feature_name"], {"asr_speech_rate_mean", "asr_pause_sentence_ratio"})
        self.assertIn("binary_corr", scores.columns)
        self.assertIn("severity_spearman_rho", scores.columns)
        self.assertIn("combined_score", scores.columns)

    def test_build_feature_score_table_tolerates_constant_feature(self):
        df = pd.DataFrame(
            {
                "clinical_match_status": ["matched"] * 6,
                "has_icas": [0, 0, 0, 1, 1, 1],
                "label": [0, 0, 0, 1, 1, 1],
                "stenosis_multiclass": [0, 0, 1, 2, 3, 3],
                "asr_constant_feature": [1, 1, 1, 1, 1, 1],
                "asr_signal_feature": [10, 9, 8, 3, 2, 1],
            }
        )
        scores = build_feature_score_table(df)
        const_row = scores.loc[scores["feature_name"] == "asr_constant_feature"].iloc[0]
        self.assertTrue(pd.isna(const_row["binary_corr"]))
        self.assertTrue(pd.isna(const_row["severity_spearman_rho"]))


class OutputTests(unittest.TestCase):
    def test_write_analysis_outputs_writes_csv_and_report(self):
        df = pd.DataFrame(
            {
                "clinical_match_status": ["matched"] * 4,
                "has_icas": [0, 0, 1, 1],
                "label": [0, 0, 1, 1],
                "stenosis_multiclass": [0, 1, 2, 3],
                "asr_speech_rate_mean": [200, 190, 150, 130],
                "asr_pause_sentence_ratio": [0.01, 0.05, 0.18, 0.22],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            csv_path = out_dir / "scores.csv"
            md_path = out_dir / "report.md"
            scores = write_analysis_outputs(
                df=df,
                output_csv=csv_path,
                output_md=md_path,
            )
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual(len(scores), 2)
            report_text = md_path.read_text(encoding="utf-8")
            self.assertIn("ASR Feature Correlation Analysis", report_text)
            self.assertIn("Top Features For Binary ICAS", report_text)


if __name__ == "__main__":
    unittest.main()
