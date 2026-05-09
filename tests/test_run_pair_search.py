"""Unit tests for pair-search helpers."""

from __future__ import annotations

import unittest

from scripts.run_pair_search import build_command, parse_args, resolve_experiments


class BuildCommandTests(unittest.TestCase):
    def test_build_command_targets_pair_training_script(self):
        cmd = build_command({"model": "resnet50"})
        self.assertEqual(cmd[:2], [cmd[0], "scripts/train_cnn_pair.py"])

    def test_build_command_includes_overrides(self):
        cmd = build_command(
            {"model": "mobilenet", "multi-task": True},
            npy_dir="datasets/npy_temperature",
            batch_size=8,
            device="cuda:0",
        )
        self.assertIn("--multi-task", cmd)
        self.assertIn("--batch-size", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--device", cmd)
        self.assertIn("cuda:0", cmd)


class ResolveExperimentsTests(unittest.TestCase):
    def test_quick_preset_contains_resnet_and_mobilenet(self):
        experiments = resolve_experiments("quick")
        models = {exp["model"] for exp in experiments}
        self.assertIn("mobilenet", models)
        self.assertIn("resnet50", models)


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_accepts_limit_and_device(self):
        args = parse_args(["--preset", "quick", "--limit", "2", "--device", "cuda"])
        self.assertEqual(args.preset, "quick")
        self.assertEqual(args.limit, 2)
        self.assertEqual(args.device, "cuda")


if __name__ == "__main__":
    unittest.main()
