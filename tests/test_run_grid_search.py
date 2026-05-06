"""Unit tests for run_grid_search helpers."""

from __future__ import annotations

import unittest

from scripts.run_grid_search import generate_experiments, parse_args


class GenerateExperimentsTests(unittest.TestCase):
    def test_coarse_preset_has_expected_size(self):
        experiments = generate_experiments("coarse")
        self.assertEqual(len(experiments), 33)

    def test_quick_preset_has_expected_size(self):
        experiments = generate_experiments("quick")
        self.assertEqual(len(experiments), 12)

    def test_non_multitask_variants_do_not_keep_lambda_sev(self):
        experiments = generate_experiments("coarse")
        baseline = next(item for item in experiments if item["method_name"] == "mobilenet_baseline")
        self.assertNotIn("lambda-sev", baseline["params"])

    def test_multitask_variants_keep_lambda_sev(self):
        experiments = generate_experiments("coarse")
        multitask = next(item for item in experiments if item["method_name"] == "mobilenet_multi_task")
        self.assertEqual(multitask["params"]["lambda-sev"], 0.3)


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_accepts_grid_search_controls(self):
        args = parse_args(["--preset", "quick", "--start-from", "5", "--limit", "3", "--dry-run"])
        self.assertEqual(args.preset, "quick")
        self.assertEqual(args.start_from, 5)
        self.assertEqual(args.limit, 3)
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
