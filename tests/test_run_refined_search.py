"""Unit tests for refined CNN search helpers."""

from __future__ import annotations

import unittest

from scripts.run_refined_search import generate_experiments, parse_args


class GenerateExperimentsTests(unittest.TestCase):
    def test_repro_preset_has_expected_size(self):
        experiments = generate_experiments("repro")
        self.assertEqual(len(experiments), 9)

    def test_focused_preset_has_expected_size(self):
        experiments = generate_experiments("focused")
        self.assertEqual(len(experiments), 18)

    def test_names_include_family_and_variant(self):
        experiments = generate_experiments("repro")
        names = [item["name"] for item in experiments]
        self.assertTrue(any("deeper_profile_a__seed_42" in name for name in names))
        self.assertTrue(any("mobilenet_multi_task_profile_a__seed_1337" in name for name in names))

    def test_multitask_variants_keep_lambda_sev_when_enabled(self):
        experiments = generate_experiments("focused")
        mt = next(item for item in experiments if item["family_name"] == "mobilenet_multi_task_profile_a")
        self.assertEqual(mt["params"]["lambda-sev"], 0.3)

    def test_no_severity_variant_keeps_flag(self):
        experiments = generate_experiments("focused")
        no_sev = next(item for item in experiments if item["family_name"] == "region_attention_multi_task_no_severity_profile_b")
        self.assertTrue(no_sev["params"]["no-severity"])


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_accepts_refined_controls(self):
        args = parse_args(["--preset", "repro", "--start-from", "2", "--limit", "4", "--dry-run"])
        self.assertEqual(args.preset, "repro")
        self.assertEqual(args.start_from, 2)
        self.assertEqual(args.limit, 4)
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
