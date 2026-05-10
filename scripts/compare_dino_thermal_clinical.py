#!/usr/bin/env python3
"""Compare DINO thermal features, clinical variables, and their fusion.

This mirrors `scripts/compare_fusion_models.py`, but uses pre-extracted DINO
features instead of CNN backbone embeddings.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_fusion_models import (
    apply_split_feature_set,
    build_feature_sets,
    load_clinical_table,
    model_configs,
    normalize_feature_set_names,
    run_feature_set_comparison,
    select_clinical_feature_columns,
)


def build_dino_fusion_frame(
    dino_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    clinical_cols: list[str],
) -> pd.DataFrame:
    """Merge per-sample DINO rows with patient-level clinical features."""
    clinical_sub = clinical_df[["canonical_patient_id", *clinical_cols]].copy()
    for col in clinical_cols:
        clinical_sub[col] = pd.to_numeric(clinical_sub[col], errors="coerce")
    merged = dino_df.merge(
        clinical_sub,
        left_on="patient_id",
        right_on="canonical_patient_id",
        how="left",
    )
    return merged.drop(columns=["canonical_patient_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, required=True, help="CSV from extract_dino_thermal_features.py")
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--feature-sets", type=str, help="Comma-separated subset: deep_only,clinical_only,fusion")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--no-search", action="store_true", help="Skip GridSearchCV for a faster comparison")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dino_df = pd.read_csv(args.feature_csv)
    clinical_df = load_clinical_table(args.clinical)
    clinical_cols = select_clinical_feature_columns(clinical_df.copy())
    fusion_df = build_dino_fusion_frame(dino_df, clinical_df, clinical_cols)

    deep_cols = [col for col in fusion_df.columns if col.startswith("dino_")]
    feature_sets = build_feature_sets(deep_cols, clinical_cols)
    selected_feature_sets = normalize_feature_set_names(args.feature_sets, list(feature_sets.keys()))
    split = __import__("json").loads(args.split.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for feature_set_name in selected_feature_sets:
        feature_cols = feature_sets[feature_set_name]
        X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te = apply_split_feature_set(
            fusion_df, split, feature_cols
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
    out_csv = args.output / f"dino_thermal_clinical_comparison_{timestamp}.csv"
    fusion_df.to_csv(args.output / f"dino_thermal_clinical_features_{timestamp}.csv", index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
