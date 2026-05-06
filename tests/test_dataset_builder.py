import csv
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_builder import (
    analyze_sources,
    build_dataset,
    build_full_dataset,
    load_multimodal_lookup,
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
        self.assertEqual(parse_2025_stem("孔村+3朱士苓-正-1"), ("孔村+3朱士苓", "正", 1))
        self.assertEqual(parse_2025_stem("6362403辛秉太-正1"), ("6362403辛秉太", "正", 1))
        self.assertEqual(parse_2025_stem("FY001正-1"), ("FY001", "正", 1))
        self.assertEqual(parse_2025_stem("CN026-正 -1"), ("CN026", "正", 1))

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

    def test_full_dataset_includes_id_and_name_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            datasets = root / "datasets"
            for patient_id in ["001BD", "BY019", "QIU GUOFU邱国福", "UNKNOWN"]:
                patient_dir = datasets / "2024" / patient_id
                patient_dir.mkdir(parents=True)
                (patient_dir / f"{patient_id}1.jpg").write_text("jpg", encoding="utf-8")
                (patient_dir / f"{patient_id}1.csv").write_text("csv", encoding="utf-8")

            photo_dir = datasets / "2025" / "热成像照片"
            temp_dir = datasets / "2025" / "热成像温度数据"
            photo_dir.mkdir(parents=True)
            temp_dir.mkdir(parents=True)
            for stem in ["CH046-正-1", "CH046-正-2"]:
                (photo_dir / f"{stem}.jpg").write_text("jpg", encoding="utf-8")
                (temp_dir / f"{stem}.csv").write_text("csv", encoding="utf-8")

            output = root / "full_data"
            result = build_full_dataset(
                datasets,
                output,
                clinical_ids={"001BD", "CH046"},
                multimodal_ids={"BY019"},
                name_to_clinical_ids={"邱国福": ["CH046"]},
            )

            self.assertEqual(result["included_samples"], 5)
            self.assertEqual(result["included_patients"], 3)
            self.assertEqual(result["excluded_samples"], 1)

            with (output / "manifest.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["canonical_patient_id"] for row in rows], ["001BD", "BY019", "CH046", "CH046", "CH046"])
            self.assertEqual(rows[2]["matched_by"], "name")

            with (output / "excluded_samples.csv").open(encoding="utf-8") as f:
                excluded = list(csv.DictReader(f))
            self.assertEqual(excluded[0]["patient_id"], "UNKNOWN")

    def test_full_dataset_recovers_2025_name_only_stems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            datasets = root / "datasets"
            photo_dir = datasets / "2025" / "热成像照片"
            temp_dir = datasets / "2025" / "热成像温度数据"
            photo_dir.mkdir(parents=True)
            temp_dir.mkdir(parents=True)
            for stem in ["孔村+3朱士苓-正-1", "6362403辛秉太-正1", "FY001正-1", "CN026-正 -1"]:
                (photo_dir / f"{stem}.jpg").write_text("jpg", encoding="utf-8")
                (temp_dir / f"{stem}.csv").write_text("csv", encoding="utf-8")

            output = root / "full_data"
            result = build_full_dataset(
                datasets,
                output,
                clinical_ids={"CH046", "XB001", "FY001", "CN026"},
                multimodal_ids=set(),
                name_to_clinical_ids={"朱士苓": ["CH046"], "辛秉太": ["XB001"]},
            )

            self.assertEqual(result["included_samples"], 4)
            self.assertEqual(result["excluded_samples"], 0)

            with (output / "manifest.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            summary = {row["canonical_patient_id"]: row["matched_by"] for row in rows}
            self.assertEqual(
                summary,
                {
                    "CH046": "name",
                    "XB001": "name",
                    "FY001": "patient_id",
                    "CN026": "patient_id",
                },
            )

    def test_load_multimodal_lookup_extracts_names_from_asr_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "multimodal.csv"
            csv_path.write_text(
                "patient_id,asr_file\n"
                "XB001,6362403辛秉太.json | 6362403辛秉太.json\n"
                "CH046,孔村+3朱士苓.json\n",
                encoding="utf-8",
            )

            multimodal_ids, name_to_ids = load_multimodal_lookup(csv_path)
            self.assertEqual(multimodal_ids, {"XB001", "CH046"})
            self.assertEqual(name_to_ids["辛秉太"], ["XB001"])
            self.assertEqual(name_to_ids["朱士苓"], ["CH046"])

    def test_full_dataset_recovers_exact_ocr_variant_patient_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            datasets = root / "datasets"
            patient_dir = datasets / "2024" / "GCOO5"
            patient_dir.mkdir(parents=True)
            (patient_dir / "GCOO51.jpg").write_text("jpg", encoding="utf-8")
            (patient_dir / "GCOO51.csv").write_text("csv", encoding="utf-8")

            output = root / "full_data"
            result = build_full_dataset(
                datasets,
                output,
                clinical_ids={"GC005"},
                multimodal_ids=set(),
                name_to_clinical_ids={},
            )

            self.assertEqual(result["included_samples"], 1)
            self.assertEqual(result["excluded_samples"], 0)

            with (output / "manifest.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["canonical_patient_id"], "GC005")
            self.assertEqual(rows[0]["matched_by"], "patient_id")


if __name__ == "__main__":
    unittest.main()
