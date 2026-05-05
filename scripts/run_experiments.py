#!/usr/bin/env python3
"""Batch experiment runner for CNN hyperparameter search.

Runs multiple training experiments with different parameter combinations
and saves results for comparison.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --quick  # Only key combinations
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    # ==================== MobileNetV3 (Transfer Learning) ====================
    # Base experiments
    {"model": "mobilenet", "epochs": 50},
    {"model": "mobilenet", "epochs": 50, "augment": True},
    {"model": "mobilenet", "epochs": 50, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "augment": True, "region-attention": True},

    # Multi-task learning
    {"model": "mobilenet", "epochs": 50, "multi-task": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True, "augment": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "multi-task": True, "augment": True, "region-attention": True},

    # Soft labels (different lambda values)
    {"model": "mobilenet", "epochs": 50, "soft-label": True},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "lambda-sev": 0.5},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "augment": True},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "augment": True, "region-attention": True},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "augment": True, "region-attention": True, "lambda-sev": 0.5},

    # Without pretrained (for comparison)
    {"model": "mobilenet", "epochs": 50, "no-pretrained": True},
    {"model": "mobilenet", "epochs": 50, "no-pretrained": True, "soft-label": True},

    # ==================== SimpleCNN (from scratch) ====================
    {"model": "simple", "epochs": 50},
    {"model": "simple", "epochs": 50, "augment": True},
    {"model": "simple", "epochs": 50, "soft-label": True},
    {"model": "simple", "epochs": 50, "soft-label": True, "augment": True},

    # ==================== DeeperCNN (from scratch) ====================
    {"model": "deeper", "epochs": 50},
    {"model": "deeper", "epochs": 50, "augment": True},
    {"model": "deeper", "epochs": 50, "soft-label": True},
    {"model": "deeper", "epochs": 50, "soft-label": True, "augment": True},

    # ==================== Different learning rates ====================
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "lr": 0.0005},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "lr": 0.002},

    # ==================== Different dropout ====================
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "dropout": 0.1},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "dropout": 0.5},

    # ==================== Different target sizes ====================
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "target-size": 32},
    {"model": "mobilenet", "epochs": 50, "soft-label": True, "target-size": 96},

    # ==================== Best guess combination ====================
    {"model": "mobilenet", "epochs": 80, "soft-label": True, "augment": True, "region-attention": True, "lambda-sev": 0.4, "lr": 0.0008, "dropout": 0.2},
]

QUICK_EXPERIMENTS = [
    {"model": "mobilenet", "epochs": 30},
    {"model": "mobilenet", "epochs": 30, "soft-label": True},
    {"model": "mobilenet", "epochs": 30, "soft-label": True, "augment": True, "region-attention": True},
    {"model": "simple", "epochs": 30, "soft-label": True},
    {"model": "deeper", "epochs": 30, "soft-label": True},
]


def build_command(params: dict, npy_dir: str | None = None) -> list[str]:
    cmd = [sys.executable, "scripts/train_cnn_v2.py"]
    if npy_dir:
        cmd.extend(["--npy-dir", npy_dir])
    for key, value in params.items():
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

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}: {params}")
    print(f"{'='*60}")

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
        with log_file.open("w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Params: {json.dumps(params)}\n")
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
                with results_files[0].open() as f:
                    train_result = json.load(f)
                result["status"] = "completed"
                result["test_auc_roc"] = train_result.get("test_metrics", {}).get("auc_roc")
                result["test_auc_pr"] = train_result.get("test_metrics", {}).get("auc_pr")
                result["test_f1"] = train_result.get("test_metrics", {}).get("f1")
                result["best_epoch"] = train_result.get("best_epoch")
                print(f"  COMPLETED: AUC-ROC={result['test_auc_roc']:.4f}, AUC-PR={result['test_auc_pr']:.4f}, F1={result['test_f1']:.4f}")
            else:
                result["status"] = "no_results"
                print("  COMPLETED but no results file found")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run only key experiments")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--start-from", type=int, default=1, help="Start from experiment ID")
    parser.add_argument("--npy-dir", type=str, default="datasets/npy_temperature", help="NPY cache directory")
    args = parser.parse_args()

    experiments = QUICK_EXPERIMENTS if args.quick else EXPERIMENTS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_dir = args.npy_dir if Path(args.npy_dir).exists() else None
    if npy_dir:
        print(f"Using NPY cache: {npy_dir}")
    else:
        print("NPY cache not found, using CSV (slower)")

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
        with summary_path.open("w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"End time: {datetime.now()}")
    print(f"{'='*60}")

    completed = [r for r in all_results if r["status"] == "completed"]
    if completed:
        completed.sort(key=lambda x: x.get("test_auc_roc") or 0, reverse=True)
        print("\nTop 5 by AUC-ROC:")
        for r in completed[:5]:
            print(f"  Exp {r['exp_id']}: AUC={r['test_auc_roc']:.4f}, PR={r['test_auc_pr']:.4f}, F1={r['test_f1']:.4f}")
            print(f"    Params: {r['params']}")


if __name__ == "__main__":
    main()
