#!/usr/bin/env python3
"""Late-fuse filtered 9-dim ASR and top-k clinical branch probabilities."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_asr_clinical_models import (
    apply_split_feature_set,
    binary_metrics,
    compute_severity_weights,
    fit_model,
    load_split,
    model_configs,
)


META_COLS = ["canonical_patient_id", "clinical_match_status", "has_icas", "label", "stenosis_multiclass"]


def select_top_k_features(score_df: pd.DataFrame, top_k: int) -> list[str]:
    selected = score_df.copy()
    if "selected" in selected.columns:
        selected = selected[selected["selected"] == 1].copy()
    return selected.sort_values("combined_score", ascending=False)["feature_name"].head(top_k).tolist()


def build_late_fusion_frame(
    asr_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    asr_features: list[str],
    clinical_features: list[str],
) -> pd.DataFrame:
    asr_keep = [c for c in META_COLS if c in asr_df.columns] + asr_features
    cli_keep = ["canonical_patient_id"] + clinical_features
    return asr_df.loc[:, asr_keep].merge(clinical_df.loc[:, cli_keep], on="canonical_patient_id", how="inner")


def find_model_config(model_name: str) -> dict:
    for cfg in model_configs():
        if cfg["name"] == model_name:
            return cfg
    raise ValueError(f"Unknown model name: {model_name}")


def fit_branch_model(
    cfg: dict,
    strategy: str,
    X_tr,
    y_tr,
    sev_tr,
    groups_tr,
    do_search: bool,
    cv_folds: int,
    n_jobs: int,
):
    if strategy == "severity_weighted" and not cfg["supports_sample_weight"]:
        raise ValueError(f"Model {cfg['name']} does not support strategy={strategy}")
    pipe = clone(cfg["pipeline"] if strategy == "standard" else cfg["pipeline_sw"])
    sw = None if strategy == "standard" else compute_severity_weights(y_tr, sev_tr)
    return fit_model(
        pipe,
        X_tr,
        y_tr,
        cfg["param_grid"],
        do_search=do_search,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        sample_weight=sw,
        groups=groups_tr,
    )


def predict_prob(model, X) -> np.ndarray:
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        preds = model.predict(X)
        return np.asarray(preds, dtype=float)


def blend_probabilities(asr_prob: np.ndarray, clinical_prob: np.ndarray, alpha: float) -> np.ndarray:
    """alpha weights the ASR branch; (1-alpha) weights the clinical branch."""
    return alpha * np.asarray(asr_prob, dtype=float) + (1.0 - alpha) * np.asarray(clinical_prob, dtype=float)


def pick_best_alpha(
    asr_prob: np.ndarray,
    clinical_prob: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: list[float] | np.ndarray,
) -> tuple[float, float]:
    best_alpha = float(alpha_grid[0])
    best_auc = float("-inf")
    for alpha in alpha_grid:
        blended = blend_probabilities(asr_prob, clinical_prob, alpha=float(alpha))
        auc = roc_auc_score(y_val, blended) if len(np.unique(y_val)) > 1 else float("nan")
        if auc > best_auc:
            best_auc = float(auc)
            best_alpha = float(alpha)
    return best_alpha, best_auc


def parse_args() -> argparse.Namespace:
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-subset", type=Path, default=repo_root / "reports/asr_candidate_modeling_subset.csv")
    parser.add_argument("--clinical-subset", type=Path, default=repo_root / "reports/clinical_candidate_modeling_subset.csv")
    parser.add_argument("--asr-score-csv", type=Path, default=repo_root / "reports/asr_candidate_feature_list.csv")
    parser.add_argument("--clinical-score-csv", type=Path, default=repo_root / "reports/clinical_candidate_feature_list.csv")
    parser.add_argument("--top-k-clinical", type=int, default=3)
    parser.add_argument("--split", type=Path, default=repo_root / "configs/data_split.json")
    parser.add_argument("--asr-model", type=str, default="GradientBoosting")
    parser.add_argument("--asr-strategy", type=str, default="standard", choices=["standard", "severity_weighted"])
    parser.add_argument("--clinical-model", type=str, default="LogisticRegression")
    parser.add_argument("--clinical-strategy", type=str, default="standard", choices=["standard", "severity_weighted"])
    parser.add_argument("--alpha-grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
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

    asr_features = asr_scores[asr_scores["selected"] == 1]["feature_name"].tolist()
    clinical_features = select_top_k_features(clinical_scores, args.top_k_clinical)
    fusion_df = build_late_fusion_frame(asr_df, clinical_df, asr_features, clinical_features)
    split = load_split(args.split)
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    X_tr_asr, y_tr, sev_tr, groups_tr, X_va_asr, y_va, X_te_asr, y_te = apply_split_feature_set(
        fusion_df, split, asr_features, patient_id_col="canonical_patient_id"
    )
    X_tr_cli, _, _, _, X_va_cli, _, X_te_cli, _ = apply_split_feature_set(
        fusion_df, split, clinical_features, patient_id_col="canonical_patient_id"
    )

    asr_cfg = find_model_config(args.asr_model)
    cli_cfg = find_model_config(args.clinical_model)
    asr_model, asr_params, asr_cv = fit_branch_model(
        asr_cfg, args.asr_strategy, X_tr_asr, y_tr, sev_tr, groups_tr, not args.no_search, args.cv_folds, args.n_jobs
    )
    cli_model, cli_params, cli_cv = fit_branch_model(
        cli_cfg, args.clinical_strategy, X_tr_cli, y_tr, sev_tr, groups_tr, not args.no_search, args.cv_folds, args.n_jobs
    )

    asr_val_prob = predict_prob(asr_model, X_va_asr)
    cli_val_prob = predict_prob(cli_model, X_va_cli)
    best_alpha, best_val_auc = pick_best_alpha(asr_val_prob, cli_val_prob, y_va, alpha_grid)

    asr_test_prob = predict_prob(asr_model, X_te_asr)
    cli_test_prob = predict_prob(cli_model, X_te_cli)
    fused_test_prob = blend_probabilities(asr_test_prob, cli_test_prob, best_alpha)
    fused_test_pred = (fused_test_prob >= 0.5).astype(int)
    fused_metrics = binary_metrics(y_te, fused_test_pred, fused_test_prob, "test")
    asr_test_metrics = binary_metrics(y_te, (asr_test_prob >= 0.5).astype(int), asr_test_prob, "test_asr")
    cli_test_metrics = binary_metrics(y_te, (cli_test_prob >= 0.5).astype(int), cli_test_prob, "test_clinical")

    row = {
        "asr_model": args.asr_model,
        "asr_strategy": args.asr_strategy,
        "asr_best_params": str(asr_params),
        "asr_cv_auc_roc": asr_cv,
        "clinical_model": args.clinical_model,
        "clinical_strategy": args.clinical_strategy,
        "clinical_best_params": str(cli_params),
        "clinical_cv_auc_roc": cli_cv,
        "top_k_clinical": args.top_k_clinical,
        "clinical_features": ",".join(clinical_features),
        "asr_features": ",".join(asr_features),
        "best_alpha_asr_weight": best_alpha,
        "val_fused_auc_roc": best_val_auc,
    }
    row.update(asr_test_metrics)
    row.update(cli_test_metrics)
    row.update(fused_metrics)

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output / f"late_fusion_asr_clinical_{timestamp}.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"ASR features: {', '.join(asr_features)}")
    print(f"Clinical features: {', '.join(clinical_features)}")
    print(f"Best alpha (ASR weight): {best_alpha:.3f}")
    print(f"Validation fused AUC: {best_val_auc:.6f}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
