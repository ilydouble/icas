#!/usr/bin/env python3
"""Select a compact ASR feature subset for downstream modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


META_COLS = [
    "canonical_patient_id",
    "clinical_match_status",
    "has_icas",
    "label",
    "stenosis_multiclass",
]


def apply_correlation_pruning(
    ranked_features: list[str],
    feature_df: pd.DataFrame,
    corr_threshold: float,
) -> list[str]:
    """Greedily keep high-ranked features while dropping near-duplicates."""
    selected: list[str] = []
    if "clinical_match_status" in feature_df.columns:
        matched = feature_df.loc[feature_df["clinical_match_status"] == "matched"].copy()
    else:
        matched = feature_df.copy()

    for feature in ranked_features:
        if feature not in matched.columns:
            continue
        series = matched[feature]
        if series.dropna().nunique() < 2:
            continue

        keep = True
        for chosen in selected:
            corr = matched[[feature, chosen]].corr(method="spearman").iloc[0, 1]
            if pd.notna(corr) and abs(float(corr)) >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(feature)
    return selected


def select_candidate_features(
    score_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    top_k: int,
    corr_threshold: float,
) -> list[str]:
    """Select top-ranked features and prune redundancy."""
    sort_cols = [col for col in ["combined_score", "binary_corr_abs", "severity_spearman_abs"] if col in score_df.columns]
    ranked = score_df.sort_values(sort_cols, ascending=False)["feature_name"].tolist()
    ranked = [name for name in ranked if name in feature_df.columns][:top_k]
    return apply_correlation_pruning(ranked, feature_df, corr_threshold)


def build_modeling_subset(feature_df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Return a reduced dataset with metadata/labels plus selected ASR features."""
    keep_cols = [col for col in META_COLS if col in feature_df.columns] + selected_features
    return feature_df.loc[:, keep_cols].copy()


def write_selection_outputs(
    score_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_list_csv: Path,
    modeling_subset_csv: Path,
    top_k: int,
    corr_threshold: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Write both the selected feature list and the reduced modeling table."""
    selected = select_candidate_features(score_df, feature_df, top_k=top_k, corr_threshold=corr_threshold)

    score_out = score_df.copy()
    score_out["selected"] = score_out["feature_name"].isin(selected).astype(int)
    subset_df = build_modeling_subset(feature_df, selected)

    feature_list_csv.parent.mkdir(parents=True, exist_ok=True)
    modeling_subset_csv.parent.mkdir(parents=True, exist_ok=True)
    score_out.to_csv(feature_list_csv, index=False, encoding="utf-8")
    subset_df.to_csv(modeling_subset_csv, index=False, encoding="utf-8")
    return selected, score_out, subset_df


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=repo_root / "datasets/asr_2025_features.csv",
        help="Merged ASR feature table.",
    )
    parser.add_argument(
        "--score-csv",
        type=Path,
        default=repo_root / "reports/asr_feature_correlation_scores.csv",
        help="Feature ranking CSV produced by analyze_asr_feature_correlations.py.",
    )
    parser.add_argument(
        "--feature-list-csv",
        type=Path,
        default=repo_root / "reports/asr_candidate_feature_list.csv",
        help="Output CSV listing all ranked features and selected flags.",
    )
    parser.add_argument(
        "--modeling-subset-csv",
        type=Path,
        default=repo_root / "reports/asr_candidate_modeling_subset.csv",
        help="Output CSV containing labels and selected ASR features only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many top-ranked features to consider before redundancy pruning.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Absolute Spearman threshold above which a lower-ranked feature is dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_df = pd.read_csv(args.feature_csv)
    score_df = pd.read_csv(args.score_csv)
    selected, _, subset_df = write_selection_outputs(
        score_df=score_df,
        feature_df=feature_df,
        feature_list_csv=args.feature_list_csv,
        modeling_subset_csv=args.modeling_subset_csv,
        top_k=args.top_k,
        corr_threshold=args.corr_threshold,
    )
    print(f"Selected ASR features: {len(selected)}")
    print(f"Feature list CSV: {args.feature_list_csv}")
    print(f"Modeling subset CSV: {args.modeling_subset_csv}")
    print("Selected feature names:")
    for name in selected:
        print(f"- {name}")
    print(f"Modeling subset rows: {len(subset_df)}")


if __name__ == "__main__":
    main()
