#!/usr/bin/env python3
"""Targeted AUC-first comparison runner for augmentation strategies.

This runner keeps the model family fixed and compares only augmentation
behavior, making it easier to judge whether a strategy improves ranking
quality on ICAS classification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_local_search import DEFAULT_TRAINING_ARGS, build_command


BASE_PROFILES = [
    {
        "name": "profile_a",
        "params": {
            "model": "mobilenet",
            "epochs": 30,
            "multi-task": True,
            "lambda-sev": 0.3,
            "lr": 1e-3,
            "dropout": 0.3,
            "target-size": 64,
            "weight-decay": 1e-4,
        },
    },
    {
        "name": "profile_c",
        "params": {
            "model": "mobilenet",
            "epochs": 30,
            "multi-task": True,
            "lambda-sev": 0.3,
            "lr": 5e-4,
            "dropout": 0.3,
            "target-size": 96,
            "weight-decay": 1e-4,
        },
    },
]


AUGMENTATION_VARIANTS = [
    {"name": "no_augment", "params": {}},
    {"name": "baseline_augment", "params": {"augment": True}},
    {"name": "face_cutout", "params": {"augment": True, "augmentation-strategy": "face_cutout"}},
    {
        "name": "attention_guided_cutout",
        "params": {"augment": True, "augmentation-strategy": "attention_guided_cutout"},
    },
    {
        "name": "attention_guided_mixed",
        "params": {"augment": True, "augmentation-strategy": "attention_guided_mixed"},
    },
]


def generate_experiments(preset: str) -> list[dict]:
    profiles = BASE_PROFILES if preset == "focused" else BASE_PROFILES[:1]
    experiments: list[dict] = []
    for profile in profiles:
        for strategy in AUGMENTATION_VARIANTS:
            params = dict(profile["params"])
            params.update(strategy["params"])
            experiments.append(
                {
                    "name": f"{profile['name']}__{strategy['name']}",
                    "profile_name": profile["name"],
                    "strategy_name": strategy["name"],
                    "params": params,
                }
            )
    return experiments


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("cnn_v3_results_*.json"))


def run_experiment(
    exp_id: int,
    experiment: dict,
    output_dir: Path,
    npy_dir: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    params = experiment["params"]
    cmd = build_command(params, npy_dir, batch_size=batch_size, device=device)
    log_file = output_dir / f"aug_{exp_id:03d}_{experiment['name']}.log"
    results_dir = output_dir.parent
    before = {path.resolve() for path in _result_files(results_dir)}
    result = {
        "exp_id": exp_id,
        "name": experiment["name"],
        "profile_name": experiment["profile_name"],
        "strategy_name": experiment["strategy_name"],
        "params": params,
        "status": "pending",
        "test_auc_roc": None,
        "test_auc_pr": None,
        "test_f1": None,
        "best_epoch": None,
    }

    print(f"\n{'=' * 72}")
    print(f"Augmentation experiment {exp_id}: {experiment['name']}")
    print(f"Params: {params}")
    print(f"{'=' * 72}")

    try:
        with log_file.open("w", encoding="utf-8") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Params: {json.dumps(params, ensure_ascii=False)}\n")
            f.write(f"Shared defaults: {DEFAULT_TRAINING_ARGS}\n")
            f.write("=" * 72 + "\n\n")
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
            return result

        after = [path.resolve() for path in _result_files(results_dir) if path.resolve() not in before]
        if not after:
            result["status"] = "no_results"
            print("  COMPLETED but no new results file found")
            return result

        latest_result = max(after, key=lambda path: path.stat().st_mtime)
        with latest_result.open(encoding="utf-8") as f:
            train_result = json.load(f)

        test_metrics = train_result.get("test_metrics", {})
        result["status"] = "completed"
        result["results_file"] = str(latest_result)
        result["test_auc_roc"] = test_metrics.get("auc_roc")
        result["test_auc_pr"] = test_metrics.get("auc_pr")
        result["test_f1"] = test_metrics.get("f1")
        result["best_epoch"] = train_result.get("best_epoch")
        print(
            f"  COMPLETED: AUC-ROC={result['test_auc_roc']:.4f}, "
            f"AUC-PR={result['test_auc_pr']:.4f}, F1={result['test_f1']:.4f}"
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"  ERROR: {exc}")

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["focused", "quick"], default="focused")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/augmentation_search"))
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--npy-dir", type=str, default="datasets/npy_temperature")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=str)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    experiments = generate_experiments(args.preset)
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
    for i, experiment in enumerate(experiments, 1):
        if i < args.start_from:
            print(f"Skipping experiment {i}")
            continue
        result = run_experiment(
            i,
            experiment,
            output_dir,
            npy_dir=npy_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_results.append(result)

        summary_path = output_dir / "augmentation_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 72}")
    print("Augmentation comparison completed")
    print(f"End time: {datetime.now()}")
    print(f"{'=' * 72}")

    completed = [row for row in all_results if row["status"] == "completed"]
    if completed:
        completed.sort(key=lambda row: row.get("test_auc_roc") or 0.0, reverse=True)
        print("\nTop 5 by AUC-ROC:")
        for row in completed[:5]:
            print(
                f"  {row['name']}: AUC={row['test_auc_roc']:.4f} | "
                f"PR={row['test_auc_pr']:.4f} | F1={row['test_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
