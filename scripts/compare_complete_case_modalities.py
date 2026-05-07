#!/usr/bin/env python3
"""Compare thermal-only, clinical-only, and thermal+clinical on one complete-case pool.

This script exists to answer a fairness question directly: when we restrict the
evaluation pool to the same complete-case patients used by multimodal fusion,
how do `clinical_only`, `thermal_only`, and `thermal+clinical` compare?
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_fusion_models import (
    apply_split_feature_set,
    build_fusion_frame,
    build_model_from_results,
    extract_deep_feature_frame,
    fit_model,
    load_clinical_table,
    load_results_json,
    model_configs,
    run_feature_set_comparison,
)
from scripts.train_cnn_v3 import TemperatureDataset, load_data as load_cnn_data

CLINICAL_TOP3 = [
    "waist_hip_ratio",
    "gender_encoded",
    "height",
]


def load_complete_case_patient_ids(asr_subset_path: Path, clinical_subset_path: Path) -> set[str]:
    asr_df = pd.read_csv(asr_subset_path)
    clinical_df = pd.read_csv(clinical_subset_path)
    return set(asr_df["canonical_patient_id"].astype(str)) & set(clinical_df["canonical_patient_id"].astype(str))


def restrict_split_to_patient_pool(split: dict, patient_ids: set[str]) -> dict:
    restricted = dict(split)
    for key in ("train_patient_ids", "val_patient_ids", "test_patient_ids"):
        restricted[key] = [pid for pid in split[key] if pid in patient_ids]
    return restricted


def build_complete_case_frame(
    deep_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    patient_ids: set[str],
) -> pd.DataFrame:
    deep_sub = deep_df.loc[deep_df["patient_id"].astype(str).isin(patient_ids)].copy()
    cli_sub = clinical_df.loc[
        clinical_df["canonical_patient_id"].astype(str).isin(patient_ids),
        ["canonical_patient_id", *CLINICAL_TOP3],
    ].copy()
    return build_fusion_frame(deep_sub, cli_sub, CLINICAL_TOP3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/best_cnn_v3.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--asr-subset", type=Path, default=Path("reports/asr_candidate_modeling_subset.csv"))
    parser.add_argument("--clinical-subset", type=Path, default=Path("reports/clinical_candidate_modeling_subset.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--feature-csv", type=Path, help="Optional path to save complete-case deep+clinical table")
    parser.add_argument("--no-search", action="store_true", help="Skip GridSearchCV for a faster comparison")
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results_json(args.results_json)
    split = json.loads(args.split.read_text(encoding="utf-8"))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model_from_results(results).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    repo_root = Path(".").resolve()
    data = load_cnn_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    all_samples = data["train"] + data["val"] + data["test"]

    dataset = TemperatureDataset(
        all_samples,
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size=(int(results.get("target_size", 64)), int(results.get("target_size", 64))),
        use_mask=bool(results.get("use_face_mask", True)),
        augment=False,
        region_attention=bool(results.get("use_region_attention")),
        npy_dir=args.npy_dir if args.npy_dir.exists() else None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample_to_patient = {sample["sample_id"]: sample["canonical_patient_id"] for sample in all_samples}
    sample_to_year = {sample["sample_id"]: int(sample["year"]) for sample in all_samples}
    deep_df = extract_deep_feature_frame(model, loader, device, sample_to_patient, sample_to_year)

    complete_case_patients = load_complete_case_patient_ids(args.asr_subset, args.clinical_subset)
    restricted_split = restrict_split_to_patient_pool(split, complete_case_patients)

    clinical_df = load_clinical_table(args.clinical_subset)
    complete_case_df = build_complete_case_frame(deep_df, clinical_df, complete_case_patients)

    thermal_cols = [c for c in complete_case_df.columns if c.startswith("deep_")] + ["cnn_prob", "cnn_severity_pred"]
    feature_sets = {
        "thermal_only": thermal_cols,
        "clinical_only": list(CLINICAL_TOP3),
        "thermal_clinical": thermal_cols + list(CLINICAL_TOP3),
    }

    rows: list[dict] = []
    for feature_set_name, feature_cols in feature_sets.items():
        X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te = apply_split_feature_set(
            complete_case_df,
            restricted_split,
            feature_cols,
        )
        for cfg in model_configs():
            rows.extend(
                run_feature_set_comparison(
                    cfg,
                    feature_set_name,
                    X_tr,
                    y_tr,
                    sev_tr,
                    groups_tr,
                    X_va,
                    y_va,
                    X_te,
                    y_te,
                    do_search=not args.no_search,
                    cv_folds=args.cv_folds,
                    n_jobs=args.n_jobs,
                )
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output.mkdir(parents=True, exist_ok=True)

    feature_csv = args.feature_csv or (args.output / f"complete_case_modal_features_{timestamp}.csv")
    compare_csv = args.output / f"complete_case_modality_comparison_{timestamp}.csv"

    complete_case_df.to_csv(feature_csv, index=False)
    pd.DataFrame(rows).to_csv(compare_csv, index=False)

    summary = {
        "complete_case_patients": len(complete_case_patients),
        "train_patients": len(restricted_split["train_patient_ids"]),
        "val_patients": len(restricted_split["val_patient_ids"]),
        "test_patients": len(restricted_split["test_patient_ids"]),
        "feature_table": str(feature_csv),
        "comparison_csv": str(compare_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
