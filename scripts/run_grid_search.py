#!/usr/bin/env python3
"""Coarse mixed grid search for CNN method selection.

This script is intentionally broader than `scripts/run_local_search.py`.
Use it first to determine the rough winning recipe:

- which backbone family is worth keeping,
- whether region attention helps,
- whether multi-task learning helps,
- whether augmentation, pretrained weights, face masking, and
  severity weighting should remain enabled,
- and which coarse hyperparameter profile looks promising.

After a promising family is identified, switch to
`scripts/run_local_search.py` for local refinement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.run_local_search import DEFAULT_TRAINING_ARGS, build_command


METHOD_VARIANTS = [
    {"name": "mobilenet_baseline", "params": {"model": "mobilenet", "epochs": 30}},
    {"name": "mobilenet_augment", "params": {"model": "mobilenet", "epochs": 30, "augment": True}},
    {"name": "mobilenet_region_attention", "params": {"model": "mobilenet", "epochs": 30, "region-attention": True}},
    {"name": "mobilenet_multi_task", "params": {"model": "mobilenet", "epochs": 30, "multi-task": True, "lambda-sev": 0.3}},
    {
        "name": "mobilenet_region_attention_multi_task",
        "params": {"model": "mobilenet", "epochs": 30, "region-attention": True, "multi-task": True, "lambda-sev": 0.3},
    },
    {
        "name": "mobilenet_region_attention_multi_task_augment",
        "params": {
            "model": "mobilenet",
            "epochs": 30,
            "region-attention": True,
            "multi-task": True,
            "augment": True,
            "lambda-sev": 0.3,
        },
    },
    {
        "name": "mobilenet_region_attention_multi_task_no_pretrained",
        "params": {
            "model": "mobilenet",
            "epochs": 30,
            "region-attention": True,
            "multi-task": True,
            "lambda-sev": 0.3,
            "no-pretrained": True,
        },
    },
    {
        "name": "mobilenet_region_attention_multi_task_no_mask",
        "params": {
            "model": "mobilenet",
            "epochs": 30,
            "region-attention": True,
            "multi-task": True,
            "lambda-sev": 0.3,
            "no-mask": True,
        },
    },
    {
        "name": "mobilenet_region_attention_multi_task_no_severity",
        "params": {
            "model": "mobilenet",
            "epochs": 30,
            "region-attention": True,
            "multi-task": True,
            "lambda-sev": 0.3,
            "no-severity": True,
        },
    },
    {"name": "simple_baseline", "params": {"model": "simple", "epochs": 30}},
    {"name": "deeper_baseline", "params": {"model": "deeper", "epochs": 30}},
]


HYPERPARAMETER_PROFILES = [
    {
        "name": "profile_a",
        "params": {
            "epochs": 30,
            "lr": 1e-3,
            "dropout": 0.3,
            "target-size": 64,
            "weight-decay": 1e-4,
        },
    },
    {
        "name": "profile_b",
        "params": {
            "epochs": 30,
            "lr": 5e-4,
            "dropout": 0.4,
            "target-size": 64,
            "weight-decay": 5e-4,
        },
    },
    {
        "name": "profile_c",
        "params": {
            "epochs": 40,
            "lr": 5e-4,
            "dropout": 0.3,
            "target-size": 96,
            "weight-decay": 1e-4,
        },
    },
]


QUICK_METHOD_VARIANTS = [
    METHOD_VARIANTS[0],
    METHOD_VARIANTS[2],
    METHOD_VARIANTS[4],
    METHOD_VARIANTS[5],
    METHOD_VARIANTS[6],
    METHOD_VARIANTS[9],
]


QUICK_HYPERPARAMETER_PROFILES = [
    HYPERPARAMETER_PROFILES[0],
    HYPERPARAMETER_PROFILES[1],
]


def _clean_params(params: dict) -> dict:
    cleaned = dict(params)
    if not cleaned.get("multi-task"):
        cleaned.pop("lambda-sev", None)
    return cleaned


def generate_experiments(preset: str) -> list[dict]:
    if preset == "quick":
        method_variants = QUICK_METHOD_VARIANTS
        profiles = QUICK_HYPERPARAMETER_PROFILES
    else:
        method_variants = METHOD_VARIANTS
        profiles = HYPERPARAMETER_PROFILES

    experiments: list[dict] = []
    for method in method_variants:
        for profile in profiles:
            params = dict(profile["params"])
            params.update(method["params"])
            params = _clean_params(params)
            experiments.append(
                {
                    "name": f"{method['name']}__{profile['name']}",
                    "method_name": method["name"],
                    "profile_name": profile["name"],
                    "params": params,
                }
            )
    return experiments


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("cnn_v3_results_*.json"))


def run_search_experiment(
    exp_id: int,
    experiment: dict,
    output_dir: Path,
    npy_dir: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    params = experiment["params"]
    cmd = build_command(params, npy_dir, batch_size=batch_size, device=device)
    log_file = output_dir / f"grid_{exp_id:03d}_{experiment['name']}.log"
    results_dir = output_dir.parent

    print(f"\n{'=' * 72}")
    print(f"Grid experiment {exp_id}: {experiment['name']}")
    print(f"Params: {params}")
    print(f"{'=' * 72}")

    before = {path.resolve() for path in _result_files(results_dir)}
    result = {
        "exp_id": exp_id,
        "name": experiment["name"],
        "method_name": experiment["method_name"],
        "profile_name": experiment["profile_name"],
        "params": params,
        "status": "pending",
        "test_auc_roc": None,
        "test_auc_pr": None,
        "test_f1": None,
        "best_epoch": None,
    }

    try:
        with log_file.open("w", encoding="utf-8") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Params: {json.dumps(params, ensure_ascii=False)}\n")
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
    parser.add_argument("--preset", choices=["coarse", "quick"], default="coarse")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/grid_search"))
    parser.add_argument("--start-from", type=int, default=1, help="Start from experiment ID")
    parser.add_argument("--limit", type=int, help="Run only the first N experiments after start-from")
    parser.add_argument("--npy-dir", type=str, default="datasets/npy_temperature", help="NPY cache directory")
    parser.add_argument("--batch-size", type=int, help="Override batch size for all experiments")
    parser.add_argument("--device", type=str, help="Override device for all experiments, e.g. cuda or cuda:1")
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment plan without executing training")
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
    print(f"Total experiments in preset: {len(experiments)}")
    print(f"Output directory: {output_dir}")
    print(f"Shared defaults: {DEFAULT_TRAINING_ARGS}")
    if args.batch_size is not None:
        print(f"Batch size override: {args.batch_size}")
    if args.device is not None:
        print(f"Device override: {args.device}")
    print(f"Start time: {datetime.now()}")

    selected = experiments[args.start_from - 1 :]
    if args.limit is not None:
        selected = selected[: args.limit]

    plan_path = output_dir / "grid_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Saved plan to {plan_path}")

    if args.dry_run:
        for idx, experiment in enumerate(selected, args.start_from):
            print(f"{idx:03d} {experiment['name']}: {experiment['params']}")
        return

    all_results = []
    for i, experiment in enumerate(selected, args.start_from):
        result = run_search_experiment(
            i,
            experiment,
            output_dir,
            npy_dir=npy_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_results.append(result)

        summary_path = output_dir / "grid_search_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 72}")
    print("Grid search completed")
    print(f"End time: {datetime.now()}")
    print(f"{'=' * 72}")

    completed = [row for row in all_results if row["status"] == "completed"]
    if completed:
        completed.sort(key=lambda row: row.get("test_auc_roc") or 0.0, reverse=True)
        print("\nTop 8 by AUC-ROC:")
        for row in completed[:8]:
            print(
                f"  Exp {row['exp_id']:03d} | {row['name']} | "
                f"AUC={row['test_auc_roc']:.4f} | PR={row['test_auc_pr']:.4f} | F1={row['test_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
