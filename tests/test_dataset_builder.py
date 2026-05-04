import csv
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_builder import (
    analyze_sources,
    build_dataset,
    parse_2025_stem,
    parse_sequence_for_patient,
)


class DatasetBuilderTests(unittest.TestCase):
    def test_parse_2024_sequence_accepts_hyphenated_and_compact_names(self):
        self.assertEqual(parse_sequence_for_patient("001BD1", "001BD"), 1)
        self.assertEqual(parse_sequence_for_patient("001BD-1", "001BD"), 1)
        self.assertEqual(parse_sequence_for_patient("001BD-12", "001BD"), 12)
        self.assertIsNone(parse_sequence_for_patient("other-1", "001BD"))

    def test_parse_2025_stem_extracts_patient_view_and_sequence(self):
        self.assertEqual(parse_2025_stem("001BD-正-1"), ("001BD", "正", 1))
        self.assertEqual(parse_2025_stem("001BD-正1"), ("001BD", "正", 1))
        self.assertEqual(parse_2025_stem("047FF张秀英-仰-2"), ("047FF", "仰", 2))
        self.assertEqual(parse_2025_stem("孔村+3朱士苓-正-1"), (None, "正", 1))

    def test_analysis_and_build_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            datasets = root / "datasets"
            (datasets / "2024" / "001BD").mkdir(parents=True)
            (datasets / "2024" / "001BD" / "001BD1.jpg").write_text("jpg", encoding="utf-8")
            (datasets / "2024" / "001BD" / "001BD1.csv").write_text("csv", encoding="utf-8")
            (datasets / "2024" / "002BD").mkdir(parents=True)
            (datasets / "2024" / "002BD" / "002BD2.jpg").write_text("jpg", encoding="utf-8")
            (datasets / "2024" / "002BD" / "002BD2.csv").write_text("csv", encoding="utf-8")

            photo_dir = datasets / "2025" / "热成像照片"
            temp_dir = datasets / "2025" / "热成像温度数据"
            photo_dir.mkdir(parents=True)
            temp_dir.mkdir(parents=True)
            (photo_dir / "003ED-正-1.jpg").write_text("jpg", encoding="utf-8")
            (temp_dir / "003ED-正-1.csv").write_text("csv", encoding="utf-8")
            (photo_dir / "003ED-正-2.jpg").write_text("jpg", encoding="utf-8")
            (temp_dir / "003ED-正-2.csv").write_text("csv", encoding="utf-8")
            (photo_dir / "004DD-仰-1.jpg").write_text("jpg", encoding="utf-8")
            (temp_dir / "004DD-仰-1.csv").write_text("csv", encoding="utf-8")
            (photo_dir / "005FD-正-2.jpg").write_text("jpg", encoding="utf-8")
            (temp_dir / "005FD-正-2.csv").write_text("csv", encoding="utf-8")
            (photo_dir / "006AA-正-1.jpg").write_text("jpg", encoding="utf-8")

            analysis = analyze_sources(datasets, clinical_ids={"001BD", "003ED", "999ZZ"})
            self.assertEqual(analysis.summary["selected_samples"], 3)
            self.assertIn("002BD", analysis.issues_by_patient)
            self.assertIn("004DD", analysis.issues_by_patient)
            self.assertIn("005FD", analysis.issues_by_patient)
            self.assertIn("006AA", analysis.issues_by_patient)

            output = root / "processed"
            result1 = build_dataset(datasets, output, clinical_ids={"001BD", "003ED"})
            result2 = build_dataset(datasets, output, clinical_ids={"001BD", "003ED"})

            self.assertEqual(result1["copied"], 6)
            self.assertEqual(result2["copied"], 0)
            self.assertEqual(result2["skipped_existing"], 6)

            with (output / "manifest.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["sample_id"] for row in rows], ["2024_001BD_1", "2025_003ED_1", "2025_003ED_2"])


if __name__ == "__main__":
    unittest.main()
