"""Unit tests for DINO thermal feature extraction helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.extract_dino_thermal_features import prepare_dino_input


class DinoInputPreparationTests(unittest.TestCase):
    def test_prepare_dino_input_repeats_single_channel_to_rgb(self):
        thermal = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

        rgb = prepare_dino_input(thermal)

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(rgb[:, :, 0], rgb[:, :, 1]))
        self.assertTrue(np.array_equal(rgb[:, :, 1], rgb[:, :, 2]))

    def test_prepare_dino_input_scales_normalized_heatmap_to_uint8(self):
        thermal = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

        rgb = prepare_dino_input(thermal)

        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(int(rgb[0, 0, 0]), 0)
        self.assertEqual(int(rgb[0, 1, 0]), 128)
        self.assertEqual(int(rgb[1, 0, 0]), 255)
        self.assertEqual(int(rgb[1, 1, 0]), 64)

    def test_prepare_dino_input_uses_min_max_scaling_for_raw_temperatures(self):
        thermal = np.array([[20.0, 25.0], [30.0, 35.0]], dtype=np.float32)

        rgb = prepare_dino_input(thermal)

        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(int(rgb[0, 0, 0]), 0)
        self.assertEqual(int(rgb[1, 1, 0]), 255)


if __name__ == "__main__":
    unittest.main()
