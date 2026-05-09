#!/usr/bin/env python3
"""Search runner for cross-year patient-pair thermal CNN experiments.

This script batches a small, conservative search space around the new
pair-input setting. It prioritizes:

1. Backbone choice: MobileNetV3-Small vs pretrained ResNet-50
2. Task head: single-task vs multi-task
3. Augmentation intensity: none vs light online augmentation

Usage:
    python scripts/run_pair_search.py
    python scripts/run_pair_search.py --preset focused --device cuda
    python scripts/run_pair_search.py --preset quick --limit 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

QUICK_EXPERIMENTS = [
    {"model": "mobilenet", "epochs": 30, "pairing-mode": "cross_year_first"},
    {"model": "mobilenet", "epochs": 30, "pairing-mode": "within_2025_first2"},
    {"model": "resnet50", "epochs": 30, "pairing-mode": "cross_year_all", "lr": 0.0003, "batch-size": 8},
    {"model": "resnet50", "epochs": 30, "pairing-mode": "within_2025_first2", "multi-task": True, "lambda-sev": 0.2, "lr": 0.0003, "batch-size": 8},
]

FOCUSED_EXPERIMENTS = [
    {"model": "mobilenet", "epochs": 30, "pairing-mode": "cross_year_first"},
    {"model": "mobilenet", "epochs": 30, "pairing-mode": "cross_year_all"},
    {"model": "mobilenet", "epochs": 30, "pairing-mode": "within_2025_first2"},
    {"model": "mobilenet", "epochs": 30, "pairing-mode": "within_2025_first2", "augment": True, "augmentation-strategy": "mild_no_flip"},
    {"model": "resnet50", "epochs": 30, "pairing-mode": "cross_year_first", "lr": 0.0003, "batch-size": 8},
    {"model": "resnet50", "epochs": 30, "pairing-mode": "cross_year_all", "lr": 0.0003, "batch-size": 8},
    {"model": "resnet50", "epochs": 30, "pairing-mode": "within_2025_first2", "lr": 0.0003, "batch-size": 8},
    {"model": "resnet50", "epochs": 30, "pairing-mode": "within_2025_first2", "multi-task": True, "lambda-sev": 0.2, "lr": 0.0003, "batch-size": 8, "freeze-backbone-epochs": 3},
]

DEFAULT_TRAINING_ARGS = {
    "early-stop-patience": 6,
    "early-stop-min-epochs": 8,
    "early-stop-min-delta": 0.001,
    "grad-clip": 1.0,
    "selection-metric": "auc_roc",
    "severity-beta": 0.25,
    "seed": 42,
}


def resolve_experiments(preset: str) -> list[dict]:
    if preset == "quick":
        return QUICK_EXPERIMENTS
    return FOCUSED_EXPERIMENTS


def build_command(
    params: dict,
    npy_dir: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> list[str]:
    cmd = [sys.executable, "scripts/train_cnn_pair.py"]
    if npy_dir:
        cmd.extend(["--npy-dir", npy_dir])

    merged = dict(DEFAULT_TRAINING_ARGS)
    merged.update(params)
    if batch_size is not None:
        merged["batch-size"] = batch_size
    if device is not None:
        merged["device"] = device

    for key, value in merged.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("cnn_pair_results_*.json"))


def run_experiment(
    exp_id: int,
    params: dict,
    output_dir: Path,
    npy_dir: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    cmd = build_command(params, npy_dir, batch_size=batch_size, device=device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"pair_exp_{exp_id:03d}_{timestamp}"
    log_file = output_dir / f"{exp_name}.log"
    results_dir = output_dir.parent
    before = {path.resolve() for path in _result_files(results_dir)}

    print(f"\n{'=' * 72}")
    print(f"Pair experiment {exp_id}: {params}")
    print(f"{'=' * 72}")

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
            f.write("=" * 72 + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            result["status"] = "failed"
            print(f"  FAILED with return code {proc.returncode}")
            return result

        new_results = [path for path in _result_files(results_dir) if path.resolve() not in before]
        if not new_results:
            result["status"] = "no_results"
            print("  COMPLETED but no new results file was detected")
            return result

        latest = new_results[-1]
        with latest.open(encoding="utf-8") as f:
            train_result = json.load(f)
        metrics = train_result.get("test_metrics", {})
        result["status"] = "completed"
        result["results_file"] = str(latest)
        result["test_auc_roc"] = metrics.get("auc_roc")
        result["test_auc_pr"] = metrics.get("auc_pr")
        result["test_f1"] = metrics.get("f1")
        result["best_epoch"] = train_result.get("best_epoch")
        print(
            f"  COMPLETED: AUC-ROC={result['test_auc_roc']:.4f}, "
            f"AUC-PR={result['test_auc_pr']:.4f}, F1={result['test_f1']:.4f}"
        )
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"  ERROR: {exc}")
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick", "focused"], default="focused")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pair_search"))
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Run only the first N experiments after start-from")
    parser.add_argument("--npy-dir", type=str, default="datasets/npy_temperature", help="NPY cache directory")
    parser.add_argument("--batch-size", type=int, help="Override batch size for all experiments")
    parser.add_argument("--device", type=str, help="Override device for all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing training")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    experiments = resolve_experiments(args.preset)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_dir = args.npy_dir if Path(args.npy_dir).exists() else None
    print(f"Preset: {args.preset}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Output directory: {output_dir}")
    if npy_dir:
        print(f"Using NPY cache: {npy_dir}")
    else:
        print("NPY cache not found, using CSV (slower)")

    if args.dry_run:
        for i, params in enumerate(experiments, 1):
            cmd = build_command(params, npy_dir, batch_size=args.batch_size, device=args.device)
            print(f"{i:02d}. {' '.join(cmd)}")
        return

    all_results = []
    run_count = 0
    for i, params in enumerate(experiments, 1):
        if i < args.start_from:
            print(f"Skipping experiment {i}")
            continue
        if args.limit is not None and run_count >= args.limit:
            break

        result = run_experiment(
            i,
            params,
            output_dir,
            npy_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_results.append(result)
        run_count += 1

        summary_path = output_dir / "pair_search_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 72}")
    print("Pair search completed")
    print(f"End time: {datetime.now()}")
    print(f"{'=' * 72}")

    completed = [item for item in all_results if item["status"] == "completed"]
    if completed:
        completed.sort(key=lambda item: item.get("test_auc_roc") or 0, reverse=True)
        print("\nTop runs by AUC-ROC:")
        for item in completed[:5]:
            print(
                f"  Exp {item['exp_id']}: AUC={item['test_auc_roc']:.4f}, "
                f"PR={item['test_auc_pr']:.4f}, F1={item['test_f1']:.4f}"
            )
            print(f"    Params: {item['params']}")


if __name__ == "__main__":
    main()
