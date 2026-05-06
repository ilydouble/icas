"""Unit tests for augmentation strategy comparison helpers."""

from __future__ import annotations

import unittest

from scripts.compare_augmentation_strategies import generate_experiments, parse_args


class GenerateExperimentsTests(unittest.TestCase):
    def test_focused_preset_has_expected_size(self):
        experiments = generate_experiments("focused")
        self.assertEqual(len(experiments), 10)

    def test_quick_preset_has_expected_size(self):
        experiments = generate_experiments("quick")
        self.assertEqual(len(experiments), 5)

    def test_focused_preset_includes_tiny_attention_guided_variant(self):
        experiments = generate_experiments("focused")
        guided = next(item for item in experiments if item["strategy_name"] == "tiny_attention_guided_cutout")
        self.assertTrue(guided["params"]["augment"])
        self.assertEqual(guided["params"]["augmentation-strategy"], "tiny_attention_guided_cutout")


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_accepts_runner_controls(self):
        args = parse_args(["--preset", "quick", "--start-from", "3", "--batch-size", "64"])
        self.assertEqual(args.preset, "quick")
        self.assertEqual(args.start_from, 3)
        self.assertEqual(args.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
