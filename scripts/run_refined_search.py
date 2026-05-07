#!/usr/bin/env python3
"""Refined CNN search around the latest grid-search shortlist.

This script is narrower than `scripts/run_grid_search.py` and more structured
than the legacy local search. It targets the three most decision-relevant
families from the current `docs/model_search_report.md` conclusion:

1. `deeper_baseline__profile_a`
2. `mobilenet_multi_task__profile_a`
3. `mobilenet_region_attention_multi_task_no_severity__profile_b`

Two presets are provided:

- `repro`: multi-seed confirmation only
- `focused`: multi-seed confirmation plus small local perturbations
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_local_search import build_command


SHORTLIST_FAMILIES = [
    {
        "family_name": "deeper_profile_a",
        "base_params": {
            "model": "deeper",
            "epochs": 30,
            "lr": 0.001,
            "dropout": 0.3,
            "target-size": 64,
            "weight-decay": 0.0001,
        },
        "repro_variants": [
            {"variant_name": "seed_42", "params": {"seed": 42}},
            {"variant_name": "seed_1337", "params": {"seed": 1337}},
            {"variant_name": "seed_2025", "params": {"seed": 2025}},
        ],
        "focused_variants": [
            {"variant_name": "seed_42", "params": {"seed": 42}},
            {"variant_name": "seed_1337", "params": {"seed": 1337}},
            {"variant_name": "seed_2025", "params": {"seed": 2025}},
            {"variant_name": "lr_7e4", "params": {"lr": 0.0007}},
            {"variant_name": "dropout_02", "params": {"dropout": 0.2}},
            {"variant_name": "weight_decay_5e4", "params": {"weight-decay": 0.0005}},
        ],
    },
    {
        "family_name": "mobilenet_multi_task_profile_a",
        "base_params": {
            "model": "mobilenet",
            "epochs": 30,
            "lr": 0.001,
            "dropout": 0.3,
            "target-size": 64,
            "weight-decay": 0.0001,
            "multi-task": True,
            "lambda-sev": 0.3,
        },
        "repro_variants": [
            {"variant_name": "seed_42", "params": {"seed": 42}},
            {"variant_name": "seed_1337", "params": {"seed": 1337}},
            {"variant_name": "seed_2025", "params": {"seed": 2025}},
        ],
        "focused_variants": [
            {"variant_name": "seed_42", "params": {"seed": 42}},
            {"variant_name": "seed_1337", "params": {"seed": 1337}},
            {"variant_name": "seed_2025", "params": {"seed": 2025}},
            {"variant_name": "lambda_02", "params": {"lambda-sev": 0.2}},
            {"variant_name": "dropout_04", "params": {"dropout": 0.4}},
            {"variant_name": "freeze_backbone_3", "params": {"freeze-backbone-epochs": 3}},
        ],
    },
    {
        "family_name": "region_attention_multi_task_no_severity_profile_b",
        "base_params": {
            "model": "mobilenet",
            "epochs": 30,
            "lr": 0.0005,
            "dropout": 0.4,
            "target-size": 64,
            "weight-decay": 0.0005,
            "region-attention": True,
            "multi-task": True,
            "lambda-sev": 0.3,
            "no-severity": True,
        },
        "repro_variants": [
            {"variant_name": "seed_42", "params": {"seed": 42}},
            {"variant_name": "seed_1337", "params": {"seed": 1337}},
            {"variant_name": "seed_2025", "params": {"seed": 2025}},
        ],
        "focused_variants": [
            {"variant_name": "seed_42", "params": {"seed": 42}},
            {"variant_name": "seed_1337", "params": {"seed": 1337}},
            {"variant_name": "seed_2025", "params": {"seed": 2025}},
            {"variant_name": "lr_3e4", "params": {"lr": 0.0003}},
            {"variant_name": "dropout_03", "params": {"dropout": 0.3}},
            {"variant_name": "weight_decay_1e4", "params": {"weight-decay": 0.0001}},
        ],
    },
]


def _clean_params(params: dict) -> dict:
    cleaned = dict(params)
    if not cleaned.get("multi-task"):
        cleaned.pop("lambda-sev", None)
    return cleaned


def generate_experiments(preset: str) -> list[dict]:
    variant_key = "repro_variants" if preset == "repro" else "focused_variants"
    experiments: list[dict] = []
    for family in SHORTLIST_FAMILIES:
        for variant in family[variant_key]:
            params = dict(family["base_params"])
            params.update(variant["params"])
            params = _clean_params(params)
            experiments.append(
                {
                    "name": f"{family['family_name']}__{variant['variant_name']}",
                    "family_name": family["family_name"],
                    "variant_name": variant["variant_name"],
                    "params": params,
                }
            )
    return experiments


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("cnn_v3_results_*.json"))


def run_refined_experiment(
    exp_id: int,
    experiment: dict,
    output_dir: Path,
    npy_dir: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    params = experiment["params"]
    cmd = build_command(params, npy_dir, batch_size=batch_size, device=device)
    log_file = output_dir / f"refined_{exp_id:03d}_{experiment['name']}.log"
    results_dir = output_dir.parent
    before = {path.resolve() for path in _result_files(results_dir)}

    print(f"\n{'=' * 72}")
    print(f"Refined experiment {exp_id}: {experiment['name']}")
    print(f"Params: {params}")
    print(f"{'=' * 72}")

    result = {
        "exp_id": exp_id,
        "name": experiment["name"],
        "family_name": experiment["family_name"],
        "variant_name": experiment["variant_name"],
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
    parser.add_argument("--preset", choices=["repro", "focused"], default="focused")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/refined_search"))
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Run only the first N experiments after start-from")
    parser.add_argument("--npy-dir", type=str, default="datasets/npy_temperature", help="NPY cache directory")
    parser.add_argument("--batch-size", type=int, help="Override batch size for all experiments")
    parser.add_argument("--device", type=str, help="Override device for all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment plan without executing training")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    experiments = generate_experiments(args.preset)
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
        for i, exp in enumerate(experiments, 1):
            print(f"{i:02d}. {exp['name']}: {exp['params']}")
        return

    all_results = []
    run_count = 0
    for i, exp in enumerate(experiments, 1):
        if i < args.start_from:
            print(f"Skipping experiment {i}")
            continue
        if args.limit is not None and run_count >= args.limit:
            break
        result = run_refined_experiment(
            i,
            exp,
            output_dir,
            npy_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_results.append(result)
        run_count += 1

        summary_path = output_dir / "refined_search_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    completed = [r for r in all_results if r["status"] == "completed"]
    if completed:
        completed.sort(key=lambda x: x.get("test_auc_roc") or 0, reverse=True)
        print(f"\n{'=' * 72}")
        print("Top refined runs by AUC-ROC:")
        for r in completed[:5]:
            print(
                f"  Exp {r['exp_id']}: AUC={r['test_auc_roc']:.4f}, "
                f"PR={r['test_auc_pr']:.4f}, F1={r['test_f1']:.4f}"
            )
            print(f"    {r['name']}")


if __name__ == "__main__":
    main()
