#!/usr/bin/env python3
"""Focused experiment runner for the current CNN champion family.

Default search space stays close to the best observed setup:
MobileNetV3-Small + multi-task + region-attention.

It searches regularization and optimization knobs that are more likely to
matter on this dataset than broad architecture sweeps.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --preset champion --start-from 5
    python scripts/run_experiments.py --preset legacy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CHAMPION_EXPERIMENTS = [
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "lambda-sev": 0.1},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "lambda-sev": 0.2},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "lambda-sev": 0.4},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "lr": 0.0003},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "lr": 0.0005},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "weight-decay": 0.0005},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "weight-decay": 0.001},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "dropout": 0.4},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "dropout": 0.5},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "freeze-backbone-epochs": 3},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "freeze-backbone-epochs": 5},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "target-size": 96, "lr": 0.0005},
    {"model": "mobilenet", "epochs": 30, "multi-task": True, "region-attention": True, "augment": True, "lr": 0.0005, "dropout": 0.4},
    {"model": "mobilenet", "epochs": 35, "multi-task": True, "region-attention": True, "lambda-sev": 0.2, "lr": 0.0005, "dropout": 0.4, "weight-decay": 0.0005, "freeze-backbone-epochs": 3},
    {"model": "mobilenet", "epochs": 35, "multi-task": True, "region-attention": True, "lambda-sev": 0.1, "lr": 0.0003, "dropout": 0.5, "weight-decay": 0.001, "freeze-backbone-epochs": 5},
]

QUICK_EXPERIMENTS = [
    {"model": "mobilenet", "epochs": 24, "multi-task": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 24, "multi-task": True, "region-attention": True, "lambda-sev": 0.2},
    {"model": "mobilenet", "epochs": 24, "multi-task": True, "region-attention": True, "lr": 0.0005, "dropout": 0.4},
    {"model": "mobilenet", "epochs": 24, "multi-task": True, "region-attention": True, "freeze-backbone-epochs": 3},
]

LEGACY_EXPERIMENTS = [
    {"model": "mobilenet", "epochs": 50},
    {"model": "mobilenet", "epochs": 50, "augment": True},
    {"model": "mobilenet", "epochs": 50, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "augment": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True, "augment": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True, "augment": True, "region-attention": True},
]

DEFAULT_TRAINING_ARGS = {
    "early-stop-patience": 6,
    "early-stop-min-epochs": 8,
    "early-stop-min-delta": 0.001,
    "grad-clip": 1.0,
    "severity-beta": 0.25,
    "seed": 42,
}


def build_command(params: dict, npy_dir: str | None = None) -> list[str]:
    cmd = [sys.executable, "scripts/train_cnn_v2.py"]
    if npy_dir:
        cmd.extend(["--npy-dir", npy_dir])

    merged = dict(DEFAULT_TRAINING_ARGS)
    merged.update(params)
    for key, value in merged.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    return cmd


def run_experiment(exp_id: int, params: dict, output_dir: Path, npy_dir: str | None = None) -> dict:
    cmd = build_command(params, npy_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{exp_id:03d}_{timestamp}"
    log_file = output_dir / f"{exp_name}.log"

    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id}: {params}")
    print(f"{'=' * 60}")

    result = {
        "exp_id": exp_id,
        "params": params,
        "timestamp": timestamp,
        "status": "pending",
        "test_auc_roc": None,
        "test_auc_pr": None,
        "test_f1": None,
    }

    try:
        with log_file.open("w", encoding="utf-8") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Params: {json.dumps(params, ensure_ascii=False)}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if proc.returncode != 0:
            result["status"] = "failed"
            print(f"  FAILED with return code {proc.returncode}")
        else:
            results_files = sorted(output_dir.parent.glob("cnn_v2_results_*.json"), reverse=True)
            if results_files:
                with results_files[0].open(encoding="utf-8") as f:
                    train_result = json.load(f)
                result["status"] = "completed"
                result["test_auc_roc"] = train_result.get("test_metrics", {}).get("auc_roc")
                result["test_auc_pr"] = train_result.get("test_metrics", {}).get("auc_pr")
                result["test_f1"] = train_result.get("test_metrics", {}).get("f1")
                result["best_epoch"] = train_result.get("best_epoch")
                print(
                    f"  COMPLETED: AUC-ROC={result['test_auc_roc']:.4f}, "
                    f"AUC-PR={result['test_auc_pr']:.4f}, F1={result['test_f1']:.4f}"
                )
            else:
                result["status"] = "no_results"
                print("  COMPLETED but no results file found")

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"  ERROR: {exc}")

    return result


def resolve_experiments(preset: str, quick: bool) -> list[dict]:
    if quick:
        return QUICK_EXPERIMENTS
    if preset == "legacy":
        return LEGACY_EXPERIMENTS
    return CHAMPION_EXPERIMENTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["champion", "legacy"], default="champion")
    parser.add_argument("--quick", action="store_true", help="Run a smaller champion-focused subset")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--start-from", type=int, default=1, help="Start from experiment ID")
    parser.add_argument("--npy-dir", type=str, default="datasets/npy_temperature", help="NPY cache directory")
    args = parser.parse_args()

    experiments = resolve_experiments(args.preset, args.quick)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_dir = args.npy_dir if Path(args.npy_dir).exists() else None
    if npy_dir:
        print(f"Using NPY cache: {npy_dir}")
    else:
        print("NPY cache not found, using CSV (slower)")

    print(f"Preset: {args.preset}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now()}")

    all_results = []
    for i, params in enumerate(experiments, 1):
        if i < args.start_from:
            print(f"Skipping experiment {i}")
            continue
        result = run_experiment(i, params, output_dir, npy_dir)
        all_results.append(result)

        summary_path = output_dir / "experiment_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"End time: {datetime.now()}")
    print(f"{'=' * 60}")

    completed = [r for r in all_results if r["status"] == "completed"]
    if completed:
        completed.sort(key=lambda x: x.get("test_auc_roc") or 0, reverse=True)
        print("\nTop 5 by AUC-ROC:")
        for r in completed[:5]:
            print(
                f"  Exp {r['exp_id']}: AUC={r['test_auc_roc']:.4f}, "
                f"PR={r['test_auc_pr']:.4f}, F1={r['test_f1']:.4f}"
            )
            print(f"    Params: {r['params']}")


if __name__ == "__main__":
    main()
