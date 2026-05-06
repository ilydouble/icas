"""Unit tests for train_cnn_v3 augmentation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.train_cnn_v3 import build_cutout_candidate_mask, get_augmentation_strategy_config


class BuildCutoutCandidateMaskTests(unittest.TestCase):
    def test_face_cutout_uses_face_mask(self):
        face_mask = np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
        attn = np.zeros_like(face_mask, dtype=np.float32)

        candidate = build_cutout_candidate_mask("face_cutout", face_mask, attn)

        self.assertTrue(np.array_equal(candidate, face_mask > 0.5))

    def test_attention_guided_cutout_prefers_low_attention_face_pixels(self):
        face_mask = np.ones((3, 3), dtype=np.float32)
        attn = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.2, 0.2, 0.2],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        candidate = build_cutout_candidate_mask("attention_guided_cutout", face_mask, attn)

        expected = np.array(
            [
                [False, False, False],
                [True, True, True],
                [False, False, False],
            ]
        )
        self.assertTrue(np.array_equal(candidate, expected))


class AugmentationConfigTests(unittest.TestCase):
    def test_mild_no_flip_disables_horizontal_flip(self):
        config = get_augmentation_strategy_config("mild_no_flip")
        self.assertEqual(config["p_flip"], 0.0)
        self.assertLess(config["noise_std"], 0.02)

    def test_tiny_attention_guided_cutout_uses_small_low_prob_cutout(self):
        config = get_augmentation_strategy_config("tiny_attention_guided_cutout")
        self.assertEqual(config["strategy"], "attention_guided_cutout")
        self.assertLess(config["cutout_prob"], 0.5)
        self.assertEqual(config["cutout_scale_range"], (0.06, 0.1))


if __name__ == "__main__":
    unittest.main()
