"""Unit tests for run_experiments helpers."""

from __future__ import annotations

import unittest

from scripts.run_experiments import build_command, parse_args


class BuildCommandTests(unittest.TestCase):
    def test_build_command_includes_cli_overrides(self):
        cmd = build_command(
            {
                "model": "mobilenet",
                "batch-size": 128,
                "device": "cuda:1",
            },
            npy_dir="datasets/npy_temperature",
        )
        self.assertIn("--batch-size", cmd)
        self.assertIn("128", cmd)
        self.assertIn("--device", cmd)
        self.assertIn("cuda:1", cmd)

    def test_build_command_defaults_to_sample_level_selection(self):
        cmd = build_command({"model": "mobilenet"})
        idx = cmd.index("--selection-metric")
        self.assertEqual(cmd[idx + 1], "f1")


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_accepts_batch_size_and_device(self):
        args = parse_args(["--batch-size", "128", "--device", "cuda:1"])
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
