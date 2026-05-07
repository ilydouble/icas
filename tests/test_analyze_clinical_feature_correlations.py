"""Unit tests for clinical feature correlation analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_clinical_feature_correlations import (
    build_feature_score_table,
    select_clinical_numeric_feature_columns,
    write_analysis_outputs,
)


class FeatureSelectionTests(unittest.TestCase):
    def test_select_clinical_numeric_feature_columns_excludes_labels_and_text(self):
        df = pd.DataFrame(
            {
                "canonical_patient_id": ["P1", "P2"],
                "name": ["a", "b"],
                "gender": ["男", "女"],
                "has_icas": [0, 1],
                "label": [0, 1],
                "stenosis_multiclass": [0, 2],
                "age": [60, 70],
                "bmi": [24.0, 28.0],
                "waist": [80.0, 90.0],
            }
        )
        cols = select_clinical_numeric_feature_columns(df)
        self.assertEqual(cols, ["age", "bmi", "waist"])


class ScoreTableTests(unittest.TestCase):
    def test_build_feature_score_table_ranks_signal_features(self):
        df = pd.DataFrame(
            {
                "has_icas": [0, 0, 0, 1, 1, 1],
                "label": [0, 0, 0, 1, 1, 1],
                "stenosis_multiclass": [0, 0, 1, 2, 3, 3],
                "age": [55, 58, 60, 72, 75, 78],
                "bmi": [22.0, 23.0, 24.0, 29.0, 30.0, 31.0],
                "gender_encoded": [0, 1, 0, 1, 0, 1],
            }
        )
        scores = build_feature_score_table(df)
        self.assertGreaterEqual(len(scores), 3)
        self.assertIn("feature_name", scores.columns)
        self.assertIn("binary_corr", scores.columns)
        self.assertIn("severity_spearman_rho", scores.columns)
        self.assertIn("combined_score", scores.columns)


class OutputTests(unittest.TestCase):
    def test_write_analysis_outputs_writes_csv_and_report(self):
        df = pd.DataFrame(
            {
                "has_icas": [0, 0, 1, 1],
                "label": [0, 0, 1, 1],
                "stenosis_multiclass": [0, 1, 2, 3],
                "age": [60, 62, 73, 75],
                "bmi": [22.0, 23.0, 28.0, 29.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            csv_path = out_dir / "scores.csv"
            md_path = out_dir / "report.md"
            scores = write_analysis_outputs(df, csv_path, md_path)
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual(len(scores), 2)
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("Clinical Feature Correlation Analysis", text)
            self.assertIn("Top Features For Binary ICAS", text)


if __name__ == "__main__":
    unittest.main()
