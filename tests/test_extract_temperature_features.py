"""Unit tests for temperature feature extraction."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.extract_temperature_features import (
    REGION_NAMES,
    STAT_NAMES,
    align_polygon_to_temperature,
    asymmetry_features,
    build_polygon_mask,
    coldspot_features,
    extract_sample_features,
    feature_columns,
    hotspot_features,
    load_excluded_ids,
    parse_temperature_csv,
    region_stats,
    write_features_csv,
)


def _write_temp_csv(path: Path, matrix: np.ndarray) -> None:
    """Write a thermal CSV in the same format as the real dataset."""
    h, w = matrix.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("\ufeff测温规则名称,最高温,平均温,最低温,发射率,大气透过率,反射温度,环境温度,距离\n")
        f.write(f"G,{matrix.max():.1f},{matrix.mean():.1f},{matrix.min():.1f},1.00,1.00,25.0,25.0,4.0\n")
        f.write("\n")
        f.write("," + ",".join(str(i + 1) for i in range(w)) + ",\n")
        for r in range(h):
            row_vals = ",".join(f"{matrix[r, c]:.1f}" for c in range(w))
            f.write(f"{r + 1},{row_vals},\n")


class TemperatureCsvParserTests(unittest.TestCase):
    def test_parse_handles_trailing_comma_and_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            mat = np.arange(20, dtype=np.float32).reshape(4, 5)
            _write_temp_csv(path, mat)
            parsed = parse_temperature_csv(path)
            self.assertEqual(parsed.shape, (4, 5))
            np.testing.assert_allclose(parsed, mat, atol=0.05)


class MaskAndAlignmentTests(unittest.TestCase):
    def test_polygon_mask_covers_expected_pixels(self):
        polygon = [[1, 1], [1, 4], [4, 4], [4, 1]]
        mask = build_polygon_mask(polygon, (6, 6))
        self.assertEqual(mask.shape, (6, 6))
        self.assertEqual(int((mask > 0).sum()), 16)

    def test_align_polygon_no_op_when_sizes_match(self):
        polygon = [[10.0, 20.0], [30.0, 40.0]]
        out = align_polygon_to_temperature(polygon, {"width": 100, "height": 100}, (100, 100))
        self.assertEqual(out, polygon)

    def test_align_polygon_scales_coordinates(self):
        polygon = [[10.0, 20.0]]
        out = align_polygon_to_temperature(polygon, {"width": 100, "height": 100}, (50, 200))
        self.assertAlmostEqual(out[0][0], 20.0)  # 10 * 200/100
        self.assertAlmostEqual(out[0][1], 10.0)  # 20 * 50/100


class StatsTests(unittest.TestCase):
    def test_region_stats_basic_values(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        stats = region_stats(values)
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["median"], 3.0)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 5.0)
        self.assertAlmostEqual(stats["range"], 4.0)
        # IQR: p75 - p25
        self.assertAlmostEqual(stats["iqr"], stats["p75"] - stats["p25"])
        # CV: std / mean
        self.assertAlmostEqual(stats["cv"], stats["std"] / stats["mean"], places=5)
        for name in STAT_NAMES:
            self.assertIn(name, stats)

    def test_region_stats_zero_std_skew_kurtosis_is_zero(self):
        values = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        stats = region_stats(values)
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["iqr"], 0.0)
        self.assertAlmostEqual(stats["cv"], 0.0)  # std/mean = 0/3 = 0
        self.assertEqual(stats["skew"], 0.0)
        self.assertEqual(stats["kurtosis"], 0.0)

    def test_region_stats_empty_returns_nan(self):
        stats = region_stats(np.array([], dtype=np.float32))
        for name in STAT_NAMES:
            self.assertTrue(np.isnan(stats[name]))


class AsymmetryTests(unittest.TestCase):
    def test_asymmetry_features_compute_diff_and_ratio(self):
        left = {"mean": 35.0, "std": 0.5, "median": 35.1, "p25": 34.5, "p75": 35.5, "iqr": 1.0}
        right = {"mean": 34.0, "std": 0.4, "median": 34.0, "p25": 33.5, "p75": 34.5, "iqr": 1.1}
        feats = asymmetry_features(left, right, "eye")
        self.assertAlmostEqual(feats["eye_diff_mean"], 1.0)
        self.assertAlmostEqual(feats["eye_abs_diff_mean"], 1.0)
        self.assertAlmostEqual(feats["eye_ratio_mean"], 35.0 / 34.0)
        self.assertGreater(feats["eye_asymmetry_index"], 0)
        self.assertAlmostEqual(feats["eye_diff_std"], 0.1, places=5)
        # New robust asymmetry features
        self.assertAlmostEqual(feats["eye_diff_median"], 1.1, places=5)
        self.assertAlmostEqual(feats["eye_diff_p25"], 1.0, places=5)
        self.assertAlmostEqual(feats["eye_diff_p75"], 1.0, places=5)
        self.assertAlmostEqual(feats["eye_diff_iqr"], -0.1, places=5)


class ColdspotTests(unittest.TestCase):
    def test_coldspot_locates_min_inside_mask(self):
        temp = np.full((10, 10), 35.0, dtype=np.float32)
        temp[6, 7] = 20.0  # cold inside bbox
        temp[1, 1] = 25.0  # slightly warmer, outside mask
        mask = np.zeros_like(temp, dtype=np.uint8)
        mask[5:9, 5:9] = 255
        feats = coldspot_features(temp, mask, [5.0, 5.0, 9.0, 9.0])
        self.assertAlmostEqual(feats["face_coldspot_x_rel"], (7 - 5) / 4)
        self.assertAlmostEqual(feats["face_coldspot_y_rel"], (6 - 5) / 4)

    def test_coldspot_returns_nan_for_empty_mask(self):
        temp = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=np.uint8)
        feats = coldspot_features(temp, mask, [0.0, 0.0, 5.0, 5.0])
        self.assertTrue(np.isnan(feats["face_coldspot_x_rel"]))
        self.assertTrue(np.isnan(feats["face_coldspot_y_rel"]))


class HotspotTests(unittest.TestCase):
    def test_hotspot_locates_max_inside_mask(self):
        temp = np.zeros((10, 10), dtype=np.float32)
        temp[6, 7] = 99.0  # hot inside bbox
        temp[1, 1] = 50.0  # warm outside bbox
        mask = np.zeros_like(temp, dtype=np.uint8)
        mask[5:9, 5:9] = 255
        feats = hotspot_features(temp, mask, [5.0, 5.0, 9.0, 9.0])
        self.assertAlmostEqual(feats["face_hotspot_x_rel"], (7 - 5) / 4)
        self.assertAlmostEqual(feats["face_hotspot_y_rel"], (6 - 5) / 4)
        self.assertGreaterEqual(feats["face_thermal_gradient_mean"], 0.0)


class ExcludedIdsTests(unittest.TestCase):
    def test_load_excluded_supports_object_and_string_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ex.json"
            path.write_text(json.dumps({
                "excluded_samples": [
                    {"sample_id": "A_1"},
                    {"sample_id": "B_2", "reason": "x"},
                    "C_3",
                ]
            }), encoding="utf-8")
            ids = load_excluded_ids(path)
            self.assertEqual(ids, {"A_1", "B_2", "C_3"})

    def test_load_excluded_returns_empty_when_missing(self):
        self.assertEqual(load_excluded_ids(Path("/non/existent.json")), set())


def _square(x: float, y: float, size: float) -> list[list[float]]:
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


def _make_sample(temp_path: Path, image_size: tuple[int, int]) -> dict:
    """Build a synthetic annotation sample with all 6 sub-regions."""
    w, h = image_size
    return {
        "sample_id": "TEST_001",
        "patient_id": "TEST",
        "year": 2024,
        "image_path": "n/a",
        "temperature_path": str(temp_path),
        "image_size": {"width": w, "height": h},
        "status": "ok",
        "face": {
            "polygon": _square(2, 2, 26),
            "bbox_xyxy": [2, 2, 28, 28],
        },
        "regions": {
            "forehead":    {"polygon": _square(8, 4, 6)},
            "nose":        {"polygon": _square(13, 13, 4)},
            "left_eye":    {"polygon": _square(6, 9, 3)},
            "right_eye":   {"polygon": _square(20, 9, 3)},
            "left_cheek":  {"polygon": _square(5, 18, 5)},
            "right_cheek": {"polygon": _square(20, 18, 5)},
        },
    }


class EndToEndTests(unittest.TestCase):
    def test_extract_and_write_features_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            # Build a 30x30 thermal matrix where the left half is hotter than
            # the right half - this should produce non-zero asymmetry features.
            mat = np.full((30, 30), 30.0, dtype=np.float32)
            mat[:, :15] = 36.0  # left side hot
            mat[:, 15:] = 33.0  # right side cooler
            temp_path = tmp_root / "TEST_001.csv"
            _write_temp_csv(temp_path, mat)

            sample = _make_sample(temp_path, image_size=(30, 30))
            result = extract_sample_features(sample, repo_root=tmp_root)

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.sample_id, "TEST_001")

            # All declared columns must be present in the feature dict.
            for col in feature_columns():
                self.assertIn(col, result.features, msg=f"missing column {col}")

            # Bilateral asymmetry: left side hotter than right => positive diff.
            self.assertGreater(result.features["eye_diff_mean"], 0.0)
            self.assertGreater(result.features["cheek_diff_mean"], 0.0)
            self.assertGreater(result.features["eye_abs_diff_mean"], 0.0)

            # CSV writer should produce a single non-skipped row.
            out_path = tmp_root / "features.csv"
            written, skipped = write_features_csv([result], out_path)
            self.assertEqual(written, 1)
            self.assertEqual(skipped, 0)
            with out_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "TEST_001")
            self.assertIn("face_mean", rows[0])

    def test_skipped_samples_excluded_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            from scripts.extract_temperature_features import SampleResult

            ok = SampleResult("OK_1", "P", 2024, "ok", {c: 0.0 for c in feature_columns()})
            bad = SampleResult("BAD_1", "P", 2024, "missing_temperature_csv", {})
            written, skipped = write_features_csv([ok, bad], Path(tmp) / "out.csv")
            self.assertEqual(written, 1)
            self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
