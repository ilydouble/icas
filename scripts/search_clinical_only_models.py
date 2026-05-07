#!/usr/bin/env python3
"""Run a refined clinical-only model search on ranked structured features."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_asr_clinical_models import (
    binary_metrics,
    compute_severity_weights,
    fit_model,
    load_split,
    select_clinical_feature_columns,
)

LABEL_COLS = ["has_icas", "label", "stenosis_multiclass"]


def build_preprocessor(scale: bool) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def make_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline([*build_preprocessor(scale), ("clf", estimator)])


def model_configs(include_elasticnet: bool = False) -> list[dict]:
    configs = [
        {
            "name": "LogisticRegression_L2",
            "pipeline": make_pipeline(
                LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42),
                scale=True,
            ),
            "pipeline_sw": make_pipeline(
                LogisticRegression(max_iter=4000, random_state=42),
                scale=True,
            ),
            "param_grid": {"clf__C": [0.01, 0.1, 1.0, 10.0, 30.0]},
            "supports_sample_weight": True,
        },
        {
            "name": "SVM_RBF",
            "pipeline": make_pipeline(
                SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
                scale=True,
            ),
            "pipeline_sw": make_pipeline(
                SVC(kernel="rbf", probability=True, random_state=42),
                scale=True,
            ),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0, 30.0],
                "clf__gamma": ["scale", "auto"],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "RandomForest",
            "pipeline": make_pipeline(
                RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
                scale=False,
            ),
            "pipeline_sw": make_pipeline(RandomForestClassifier(random_state=42), scale=False),
            "param_grid": {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 3, 5],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "ExtraTrees",
            "pipeline": make_pipeline(
                ExtraTreesClassifier(class_weight="balanced", random_state=42),
                scale=False,
            ),
            "pipeline_sw": make_pipeline(ExtraTreesClassifier(random_state=42), scale=False),
            "param_grid": {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 3, 5],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "GradientBoosting",
            "pipeline": make_pipeline(GradientBoostingClassifier(random_state=42), scale=False),
            "pipeline_sw": make_pipeline(GradientBoostingClassifier(random_state=42), scale=False),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "HistGradientBoosting",
            "pipeline": make_pipeline(HistGradientBoostingClassifier(random_state=42), scale=False),
            "pipeline_sw": make_pipeline(HistGradientBoostingClassifier(random_state=42), scale=False),
            "param_grid": {
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [None, 3, 5],
                "clf__max_leaf_nodes": [15, 31],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "KNN",
            "pipeline": make_pipeline(KNeighborsClassifier(), scale=True),
            "pipeline_sw": None,
            "param_grid": {"clf__n_neighbors": [3, 5, 9, 15]},
            "supports_sample_weight": False,
        },
        {
            "name": "GaussianNB",
            "pipeline": make_pipeline(GaussianNB(), scale=False),
            "pipeline_sw": None,
            "param_grid": {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
            "supports_sample_weight": False,
        },
    ]
    if include_elasticnet:
        configs.insert(
            1,
            {
                "name": "LogisticRegression_EN",
                "pipeline": make_pipeline(
                    LogisticRegression(
                        max_iter=6000,
                        solver="saga",
                        penalty="elasticnet",
                        class_weight="balanced",
                        random_state=42,
                    ),
                    scale=True,
                ),
                "pipeline_sw": make_pipeline(
                    LogisticRegression(
                        max_iter=6000,
                        solver="saga",
                        penalty="elasticnet",
                        random_state=42,
                    ),
                    scale=True,
                ),
                "param_grid": {
                    "clf__C": [0.01, 0.1, 1.0, 10.0],
                    "clf__l1_ratio": [0.2, 0.5, 0.8],
                },
                "supports_sample_weight": True,
            },
        )
    return configs


def load_clinical_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        if col in {"canonical_patient_id", "name", "gender", "data_sources", "icas_detail"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ranked_features_from_scores(score_df: pd.DataFrame, feature_df: pd.DataFrame) -> list[str]:
    ranked = score_df.sort_values(
        [c for c in ["combined_score", "binary_corr_abs", "severity_spearman_abs"] if c in score_df.columns],
        ascending=False,
    )["feature_name"].tolist()
    available = set(select_clinical_feature_columns(feature_df))
    selected: list[str] = []
    for name in ranked:
        if name in available and name not in selected:
            series = pd.to_numeric(feature_df[name], errors="coerce")
            if series.dropna().nunique() >= 2:
                selected.append(name)
    return selected


def parse_feature_set_sizes(spec: str) -> list[int | str]:
    values: list[int | str] = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"all", "full"}:
            values.append("all")
        else:
            values.append(int(item))
    return values


def build_feature_sets(ranked_features: list[str], size_spec: list[int | str]) -> dict[str, list[str]]:
    feature_sets: dict[str, list[str]] = {}
    for item in size_spec:
        if item == "all":
            feature_sets["top_all"] = ranked_features.copy()
            continue
        count = min(int(item), len(ranked_features))
        feature_sets[f"top_{count}"] = ranked_features[:count]
    return feature_sets


def apply_split(df: pd.DataFrame, split: dict, feature_cols: list[str]) -> tuple:
    def arrays(ids: list[str]):
        sub = df[df["canonical_patient_id"].isin(ids)].dropna(subset=["label"])
        X = sub[feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        y = sub["label"].values.astype(int)
        sev = sub["stenosis_multiclass"].values
        return X, y, sev

    X_tr, y_tr, sev_tr = arrays(split["train_patient_ids"])
    X_va, y_va, sev_va = arrays(split["val_patient_ids"])
    X_te, y_te, sev_te = arrays(split["test_patient_ids"])
    return X_tr, y_tr, sev_tr, X_va, y_va, sev_va, X_te, y_te, sev_te


def predict_prob(model, X: np.ndarray) -> np.ndarray:
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        scores = model.predict(X)
        return np.asarray(scores, dtype=float)


def tuned_threshold(prob: np.ndarray, y_true: np.ndarray, metric: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in np.linspace(0.1, 0.9, 33):
        preds = (prob >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def score_with_threshold(y_true: np.ndarray, prob: np.ndarray, prefix: str, threshold: float) -> dict:
    preds = (prob >= threshold).astype(int)
    metrics = binary_metrics(y_true, preds, prob, prefix)
    metrics[f"{prefix}_threshold"] = threshold
    return metrics


def choose_model_pipeline(cfg: dict, strategy: str):
    if strategy == "severity_weighted":
        if not cfg["supports_sample_weight"] or cfg["pipeline_sw"] is None:
            return None
        return clone(cfg["pipeline_sw"])
    return clone(cfg["pipeline"])


def fit_single_run(
    cfg: dict,
    strategy: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sev_tr: np.ndarray,
    cv_folds: int,
    n_jobs: int,
    do_search: bool,
):
    pipe = choose_model_pipeline(cfg, strategy)
    if pipe is None:
        return None
    sample_weight = None
    if strategy == "severity_weighted":
        sample_weight = compute_severity_weights(y_tr, sev_tr)
    model, best_params, cv_auc = fit_model(
        pipe=pipe,
        X_tr=X_tr,
        y_tr=y_tr,
        param_grid=cfg["param_grid"],
        do_search=do_search,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        sample_weight=sample_weight,
        groups=None,
    )
    return model, best_params, cv_auc


def evaluate_config(
    feature_set_name: str,
    feature_cols: list[str],
    cfg: dict,
    strategy: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sev_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    cv_folds: int,
    n_jobs: int,
    do_search: bool,
    threshold_metric: str,
) -> dict | None:
    fitted = fit_single_run(cfg, strategy, X_tr, y_tr, sev_tr, cv_folds, n_jobs, do_search)
    if fitted is None:
        return None
    model, best_params, cv_auc = fitted
    val_prob = predict_prob(model, X_va)
    test_prob = predict_prob(model, X_te)
    best_threshold, best_threshold_score = tuned_threshold(val_prob, y_va, threshold_metric)
    row = {
        "feature_set": feature_set_name,
        "feature_count": len(feature_cols),
        "feature_names": ",".join(feature_cols),
        "model": cfg["name"],
        "strategy": strategy,
        "best_params": json.dumps(best_params, ensure_ascii=False),
        "cv_auc_roc": cv_auc,
        "val_best_threshold_metric": threshold_metric,
        "val_best_threshold_metric_score": best_threshold_score,
    }
    row.update(score_with_threshold(y_va, val_prob, "val", 0.5))
    row.update(score_with_threshold(y_te, test_prob, "test", 0.5))
    row.update(score_with_threshold(y_va, val_prob, "val_tuned", best_threshold))
    row.update(score_with_threshold(y_te, test_prob, "test_tuned", best_threshold))
    return row


def build_summary(rows_df: pd.DataFrame) -> dict:
    if rows_df.empty:
        return {"best_auc_row": None, "best_tuned_f1_row": None}
    best_auc_row = rows_df.sort_values("test_auc_roc", ascending=False).iloc[0].to_dict()
    best_f1_row = rows_df.sort_values("test_tuned_f1", ascending=False).iloc[0].to_dict()
    return {"best_auc_row": best_auc_row, "best_tuned_f1_row": best_f1_row}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, default=REPO_ROOT / "datasets/full_data/patient_clinical_data.csv")
    parser.add_argument("--score-csv", type=Path, default=REPO_ROOT / "reports/clinical_feature_correlation_scores.csv")
    parser.add_argument("--split", type=Path, default=REPO_ROOT / "configs/data_split.json")
    parser.add_argument("--feature-set-sizes", type=str, default="3,5,8,12,all")
    parser.add_argument("--threshold-metric", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--include-elasticnet", action="store_true")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clinical_df = load_clinical_frame(args.clinical)
    score_df = pd.read_csv(args.score_csv)
    split = load_split(args.split)
    ranked_features = ranked_features_from_scores(score_df, clinical_df)
    feature_sets = build_feature_sets(ranked_features, parse_feature_set_sizes(args.feature_set_sizes))

    rows: list[dict] = []
    for feature_set_name, feature_cols in feature_sets.items():
        X_tr, y_tr, sev_tr, X_va, y_va, _, X_te, y_te, _ = apply_split(clinical_df, split, feature_cols)
        for cfg in model_configs(include_elasticnet=args.include_elasticnet):
            for strategy in ("standard", "severity_weighted"):
                row = evaluate_config(
                    feature_set_name=feature_set_name,
                    feature_cols=feature_cols,
                    cfg=cfg,
                    strategy=strategy,
                    X_tr=X_tr,
                    y_tr=y_tr,
                    sev_tr=sev_tr,
                    X_va=X_va,
                    y_va=y_va,
                    X_te=X_te,
                    y_te=y_te,
                    cv_folds=args.cv_folds,
                    n_jobs=args.n_jobs,
                    do_search=not args.no_search,
                    threshold_metric=args.threshold_metric,
                )
                if row is not None:
                    rows.append(row)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_df = pd.DataFrame(rows).sort_values(["test_auc_roc", "test_tuned_f1"], ascending=[False, False])
    out_csv = out_dir / f"clinical_only_refined_search_{timestamp}.csv"
    out_json = out_dir / f"clinical_only_refined_summary_{timestamp}.json"
    rows_df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(build_summary(rows_df), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote refined clinical search rows: {len(rows_df)}")
    print(f"Output CSV: {out_csv}")
    print(f"Output summary: {out_json}")


if __name__ == "__main__":
    main()
