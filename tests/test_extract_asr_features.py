"""Unit tests for ASR feature extraction and clinical matching."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.extract_asr_features import (
    REFERENCE_TEXT,
    compute_reference_deviation_features,
    compute_sentence_level_features,
    extract_asr_row,
    normalize_patient_id_from_filename,
    normalize_text_for_comparison,
    write_feature_csv,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FilenameParsingTests(unittest.TestCase):
    def test_normalize_patient_id_handles_suffix_name(self):
        self.assertEqual(
            normalize_patient_id_from_filename("001CF_001CF郭庆兰.json"),
            "001CF",
        )

    def test_normalize_patient_id_handles_plain_id(self):
        self.assertEqual(normalize_patient_id_from_filename("001BD.json"), "001BD")


class TextNormalizationTests(unittest.TestCase):
    def test_normalize_text_removes_punctuation_and_space(self):
        self.assertEqual(normalize_text_for_comparison("北风， 和 太阳。"), "北风和太阳")

    def test_reference_deviation_smaller_for_reference_like_text(self):
        good = compute_reference_deviation_features(REFERENCE_TEXT)
        bad = compute_reference_deviation_features("啊啊啊啊啊")
        self.assertLess(good["asr_reference_char_error_rate"], bad["asr_reference_char_error_rate"])
        self.assertGreater(good["asr_reference_sequence_ratio"], bad["asr_reference_sequence_ratio"])


class SentenceFeatureTests(unittest.TestCase):
    def test_sentence_level_features_compute_rates_and_pauses(self):
        detailed_results = [
            {
                "text": "啊。",
                "duration": 1000,
                "raw_sentence": {"SpeechRate": 90, "SilenceDuration": 2, "EmotionValue": 6.5},
            },
            {
                "text": "北风和太阳。",
                "duration": 3000,
                "raw_sentence": {"SpeechRate": 120, "SilenceDuration": 8, "EmotionValue": 7.0},
            },
        ]
        features = compute_sentence_level_features(detailed_results)
        self.assertEqual(features["asr_sentence_count"], 2)
        self.assertAlmostEqual(features["asr_sentence_duration_ms_mean"], 2000.0)
        self.assertAlmostEqual(features["asr_speech_rate_mean"], 105.0)
        self.assertEqual(features["asr_pause_sentence_count"], 2)
        self.assertEqual(features["asr_long_pause_sentence_count"], 1)


class EndToEndTests(unittest.TestCase):
    def test_extract_row_and_write_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asr_dir = root / "asr"
            asr_dir.mkdir()

            clinical_path = root / "patient_clinical_data.csv"
            pd.DataFrame(
                [
                    {
                        "canonical_patient_id": "001BD",
                        "name": "柳志方",
                        "has_icas": 1,
                        "label": 1,
                        "stenosis_multiclass": 2,
                        "images_2025": 2,
                    }
                ]
            ).to_csv(clinical_path, index=False)

            asr_path = asr_dir / "001BD.json"
            _write_json(
                asr_path,
                {
                    "text": "啊。北风和太阳。",
                    "detailed_results": [
                        {
                            "text": "啊。",
                            "duration": 1000,
                            "begin_time": 0,
                            "end_time": 1000,
                            "raw_sentence": {"SpeechRate": 80, "SilenceDuration": 1, "EmotionValue": 6.0},
                        },
                        {
                            "text": "北风和太阳。",
                            "duration": 2000,
                            "begin_time": 1000,
                            "end_time": 3000,
                            "raw_sentence": {"SpeechRate": 100, "SilenceDuration": 0, "EmotionValue": 7.0},
                        },
                    ],
                    "source_file": "001BD.wav",
                },
            )

            clinical_df = pd.read_csv(clinical_path)
            row = extract_asr_row(asr_path, clinical_df)
            self.assertEqual(row["canonical_patient_id"], "001BD")
            self.assertEqual(row["clinical_match_status"], "matched")
            self.assertEqual(row["has_icas"], 1)
            self.assertIn("asr_speech_rate_mean", row)
            self.assertIn("asr_reference_char_error_rate", row)

            unmatched_path = asr_dir / "AW006.json"
            _write_json(
                unmatched_path,
                {"text": "啊。", "detailed_results": [], "source_file": "AW006.wav"},
            )

            out_path = root / "features.csv"
            matched, unmatched = write_feature_csv(
                asr_dir=asr_dir,
                clinical_csv=clinical_path,
                output_csv=out_path,
            )
            self.assertEqual(matched, 1)
            self.assertEqual(unmatched, 1)
            with out_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["canonical_patient_id"], "001BD")
            self.assertEqual(rows[1]["clinical_match_status"], "unmatched")


if __name__ == "__main__":
    unittest.main()
