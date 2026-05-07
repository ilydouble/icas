#!/usr/bin/env python3
"""Compare classical baselines on filtered ASR-only, clinical-only, and fusion sets."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_asr_clinical_models import (
    apply_split_feature_set,
    load_split,
    model_configs,
    normalize_feature_set_names,
    run_feature_set_comparison,
)


META_COLS = {"canonical_patient_id", "has_icas", "label", "stenosis_multiclass", "clinical_match_status"}


def build_filtered_fusion_frame(asr_df: pd.DataFrame, clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join filtered ASR and filtered clinical subsets on patient id."""
    clinical_non_meta = [c for c in clinical_df.columns if c not in META_COLS]
    return asr_df.merge(
        clinical_df[["canonical_patient_id", *clinical_non_meta]],
        on="canonical_patient_id",
        how="inner",
    )


def extract_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    asr_cols = [c for c in df.columns if c.startswith("asr_")]
    clinical_cols = [c for c in df.columns if c not in META_COLS and not c.startswith("asr_")]
    return asr_cols, clinical_cols


def build_feature_sets(asr_cols: list[str], clinical_cols: list[str]) -> dict[str, list[str]]:
    return {
        "asr_only": asr_cols,
        "clinical_only": clinical_cols,
        "filtered_fusion": asr_cols + clinical_cols,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-subset", type=Path, default=repo_root / "reports/asr_candidate_modeling_subset.csv")
    parser.add_argument("--clinical-subset", type=Path, default=repo_root / "reports/clinical_candidate_modeling_subset.csv")
    parser.add_argument("--split", type=Path, default=repo_root / "configs/data_split.json")
    parser.add_argument("--feature-sets", type=str, help="Comma-separated subset: asr_only,clinical_only,filtered_fusion")
    parser.add_argument("--output", type=Path, default=repo_root / "reports")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asr_df = pd.read_csv(args.asr_subset)
    if "clinical_match_status" in asr_df.columns:
        asr_df = asr_df[asr_df["clinical_match_status"] == "matched"].copy()
    clinical_df = pd.read_csv(args.clinical_subset)

    fusion_df = build_filtered_fusion_frame(asr_df, clinical_df)
    asr_cols, clinical_cols = extract_feature_columns(fusion_df)
    feature_sets = build_feature_sets(asr_cols, clinical_cols)
    selected_feature_sets = normalize_feature_set_names(args.feature_sets, list(feature_sets.keys()))
    split = load_split(args.split)

    rows: list[dict] = []
    for feature_set_name in selected_feature_sets:
        feature_cols = feature_sets[feature_set_name]
        X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te = apply_split_feature_set(
            df=fusion_df,
            split=split,
            feature_cols=feature_cols,
            patient_id_col="canonical_patient_id",
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

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output / f"filtered_asr_clinical_model_comparison_{timestamp}.csv"
    pd.DataFrame(rows).sort_values(["feature_set", "test_auc_roc"], ascending=[True, False]).to_csv(
        out_csv, index=False, encoding="utf-8"
    )
    print(f"Wrote comparison rows: {len(rows)}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
