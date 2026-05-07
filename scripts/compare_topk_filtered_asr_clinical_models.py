#!/usr/bin/env python3
"""Run a compact top-k ASR + clinical fusion ablation baseline."""

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


META_COLS = ["canonical_patient_id", "clinical_match_status", "has_icas", "label", "stenosis_multiclass"]


def select_top_k_features(score_df: pd.DataFrame, top_k: int) -> list[str]:
    selected = score_df.copy()
    if "selected" in selected.columns:
        selected = selected[selected["selected"] == 1].copy()
    return selected.sort_values("combined_score", ascending=False)["feature_name"].head(top_k).tolist()


def build_topk_fusion_frame(
    asr_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    asr_features: list[str],
    clinical_features: list[str],
) -> pd.DataFrame:
    asr_keep = [c for c in META_COLS if c in asr_df.columns] + asr_features
    clinical_keep = ["canonical_patient_id"] + clinical_features
    return asr_df.loc[:, asr_keep].merge(clinical_df.loc[:, clinical_keep], on="canonical_patient_id", how="inner")


def build_feature_sets(asr_cols: list[str], clinical_cols: list[str]) -> dict[str, list[str]]:
    return {
        "asr_only_topk": asr_cols,
        "clinical_only_topk": clinical_cols,
        "topk_fusion": asr_cols + clinical_cols,
    }


def parse_args() -> argparse.Namespace:
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-subset", type=Path, default=repo_root / "reports/asr_candidate_modeling_subset.csv")
    parser.add_argument("--clinical-subset", type=Path, default=repo_root / "reports/clinical_candidate_modeling_subset.csv")
    parser.add_argument("--asr-score-csv", type=Path, default=repo_root / "reports/asr_candidate_feature_list.csv")
    parser.add_argument("--clinical-score-csv", type=Path, default=repo_root / "reports/clinical_candidate_feature_list.csv")
    parser.add_argument("--top-k-asr", type=int, default=3)
    parser.add_argument("--top-k-clinical", type=int, default=3)
    parser.add_argument("--split", type=Path, default=repo_root / "configs/data_split.json")
    parser.add_argument("--feature-sets", type=str, help="Comma-separated subset: asr_only_topk,clinical_only_topk,topk_fusion")
    parser.add_argument("--output", type=Path, default=repo_root / "reports")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asr_df = pd.read_csv(args.asr_subset)
    clinical_df = pd.read_csv(args.clinical_subset)
    asr_scores = pd.read_csv(args.asr_score_csv)
    clinical_scores = pd.read_csv(args.clinical_score_csv)

    if "clinical_match_status" in asr_df.columns:
        asr_df = asr_df[asr_df["clinical_match_status"] == "matched"].copy()

    asr_features = select_top_k_features(asr_scores, args.top_k_asr)
    clinical_features = select_top_k_features(clinical_scores, args.top_k_clinical)
    fusion_df = build_topk_fusion_frame(asr_df, clinical_df, asr_features, clinical_features)
    feature_sets = build_feature_sets(asr_features, clinical_features)
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
    out_csv = args.output / f"topk_filtered_asr_clinical_model_comparison_{timestamp}.csv"
    pd.DataFrame(rows).sort_values(["feature_set", "test_auc_roc"], ascending=[True, False]).to_csv(
        out_csv, index=False, encoding="utf-8"
    )
    print(f"Top-k ASR features: {', '.join(asr_features)}")
    print(f"Top-k clinical features: {', '.join(clinical_features)}")
    print(f"Wrote comparison rows: {len(rows)}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
