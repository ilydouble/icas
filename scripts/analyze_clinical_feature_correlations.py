#!/usr/bin/env python3
"""Rank clinical features by association with ICAS labels and stenosis severity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


EXCLUDE_COLS = {
    "canonical_patient_id",
    "has_basic_clinical_data",
    "data_sources",
    "clinical_source_available",
    "multimodal_source_available",
    "name",
    "gender",
    "has_icas",
    "label",
    "stenosis_multiclass",
    "icas_detail",
    "image_count",
    "images_2024",
    "images_2025",
}


def select_clinical_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _cohens_d(negative: pd.Series, positive: pd.Series) -> float:
    if len(negative) < 2 or len(positive) < 2:
        return float("nan")
    mean_diff = positive.mean() - negative.mean()
    neg_var = negative.var(ddof=1)
    pos_var = positive.var(ddof=1)
    pooled_denom = ((len(negative) - 1) * neg_var + (len(positive) - 1) * pos_var)
    pooled_denom /= max(len(negative) + len(positive) - 2, 1)
    pooled_std = np.sqrt(pooled_denom) if pooled_denom >= 0 else np.nan
    if pd.isna(pooled_std) or pooled_std == 0:
        return float("nan")
    return float(mean_diff / pooled_std)


def _safe_abs(value: float) -> float:
    return float(abs(value)) if pd.notna(value) else float("nan")


def _binary_feature_summary(values: pd.Series, labels: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "label": labels}).dropna()
    if frame["label"].nunique() < 2 or len(frame) < 3 or frame["value"].nunique() < 2:
        neg = frame.loc[frame["label"] == 0, "value"]
        pos = frame.loc[frame["label"] == 1, "value"]
        return {
            "binary_n": len(frame),
            "binary_corr": float("nan"),
            "binary_corr_pvalue": float("nan"),
            "binary_auc": float("nan"),
            "binary_effect_size_d": float("nan"),
            "binary_mean_neg": float(neg.mean()) if len(neg) else float("nan"),
            "binary_mean_pos": float(pos.mean()) if len(pos) else float("nan"),
        }

    corr, corr_p = pearsonr(frame["value"], frame["label"])
    auc = roc_auc_score(frame["label"], frame["value"])
    neg = frame.loc[frame["label"] == 0, "value"]
    pos = frame.loc[frame["label"] == 1, "value"]
    return {
        "binary_n": len(frame),
        "binary_corr": float(corr),
        "binary_corr_pvalue": float(corr_p),
        "binary_auc": float(auc),
        "binary_effect_size_d": _cohens_d(neg, pos),
        "binary_mean_neg": float(neg.mean()),
        "binary_mean_pos": float(pos.mean()),
    }


def _severity_feature_summary(values: pd.Series, severity: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "severity": severity}).dropna()
    if frame["severity"].nunique() < 2 or len(frame) < 3 or frame["value"].nunique() < 2:
        return {
            "severity_n": len(frame),
            "severity_spearman_rho": float("nan"),
            "severity_spearman_pvalue": float("nan"),
            "severity_kruskal_stat": float("nan"),
            "severity_kruskal_pvalue": float("nan"),
        }

    rho, rho_p = spearmanr(frame["value"], frame["severity"])
    groups = [grp["value"].to_numpy() for _, grp in frame.groupby("severity") if len(grp) > 0]
    if len(groups) >= 2:
        kw_stat, kw_p = kruskal(*groups)
    else:
        kw_stat, kw_p = float("nan"), float("nan")
    return {
        "severity_n": len(frame),
        "severity_spearman_rho": float(rho),
        "severity_spearman_pvalue": float(rho_p),
        "severity_kruskal_stat": float(kw_stat),
        "severity_kruskal_pvalue": float(kw_p),
    }


def build_feature_score_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = select_clinical_numeric_feature_columns(df)
    rows: list[dict[str, float | str]] = []
    for feature in feature_cols:
        binary_stats = _binary_feature_summary(df[feature], df["has_icas"])
        severity_stats = _severity_feature_summary(df[feature], df["stenosis_multiclass"])
        row: dict[str, float | str] = {"feature_name": feature}
        row.update(binary_stats)
        row.update(severity_stats)
        row["binary_corr_abs"] = _safe_abs(row["binary_corr"])  # type: ignore[arg-type]
        row["severity_spearman_abs"] = _safe_abs(row["severity_spearman_rho"])  # type: ignore[arg-type]
        auc = row["binary_auc"]  # type: ignore[assignment]
        row["binary_auc_distance"] = float(abs(float(auc) - 0.5)) if pd.notna(auc) else float("nan")
        d_val = row["binary_effect_size_d"]  # type: ignore[assignment]
        row["binary_effect_size_abs"] = _safe_abs(float(d_val)) if pd.notna(d_val) else float("nan")
        score_parts = [v for v in [
            row["binary_corr_abs"],
            row["severity_spearman_abs"],
            row["binary_auc_distance"],
            row["binary_effect_size_abs"],
        ] if pd.notna(v)]
        row["combined_score"] = float(np.mean(score_parts)) if score_parts else float("nan")
        rows.append(row)
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    return scores.sort_values(
        by=["combined_score", "binary_corr_abs", "severity_spearman_abs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _format_top_table(scores: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    top = scores.loc[:, columns].head(limit).copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for _, row in top.iterrows():
        vals: list[str] = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.6f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_analysis_outputs(df: pd.DataFrame, output_csv: Path, output_md: Path) -> pd.DataFrame:
    scores = build_feature_score_table(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_csv, index=False, encoding="utf-8")

    report = "\n".join([
        "# Clinical Feature Correlation Analysis",
        "",
        "## Dataset Summary",
        "",
        f"- Total patient rows: {len(df)}",
        f"- Binary ICAS positives: {int(df['has_icas'].fillna(0).sum())}",
        f"- Candidate clinical numeric features: {len(select_clinical_numeric_feature_columns(df))}",
        "",
        "## Top Features For Binary ICAS",
        "",
        _format_top_table(scores.sort_values("binary_corr_abs", ascending=False), [
            "feature_name", "binary_corr", "binary_auc", "binary_effect_size_d", "combined_score"
        ]),
        "",
        "## Top Features For Stenosis Severity",
        "",
        _format_top_table(scores.sort_values("severity_spearman_abs", ascending=False), [
            "feature_name", "severity_spearman_rho", "severity_kruskal_pvalue", "combined_score"
        ]),
        "",
        "## Notes",
        "",
        "- `binary_corr` is Pearson correlation against `has_icas`.",
        "- `binary_auc` measures single-feature discrimination ability for ICAS.",
        "- `severity_spearman_rho` measures monotonic association with `stenosis_multiclass`.",
        "",
    ])
    output_md.write_text(report, encoding="utf-8")
    return scores


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=repo_root / "datasets/full_data/patient_clinical_data.csv")
    parser.add_argument("--output-csv", type=Path, default=repo_root / "reports/clinical_feature_correlation_scores.csv")
    parser.add_argument("--output-md", type=Path, default=repo_root / "reports/clinical_feature_correlation_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    scores = write_analysis_outputs(df, args.output_csv, args.output_md)
    print(f"Analyzed rows: {len(df)}")
    print(f"Ranked clinical features: {len(scores)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output report: {args.output_md}")


if __name__ == "__main__":
    main()
