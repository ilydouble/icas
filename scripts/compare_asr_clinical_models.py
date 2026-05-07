#!/usr/bin/env python3
"""Compare classical baselines on selected ASR features, clinical data, and fusion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLINICAL_EXCLUDE_COLS = {
    "canonical_patient_id",
    "name",
    "gender",
    "data_sources",
    "clinical_source_available",
    "multimodal_source_available",
    "has_basic_clinical_data",
    "has_icas",
    "label",
    "stenosis_multiclass",
    "icas_detail",
    "image_count",
    "images_2024",
    "images_2025",
}
SEVERITY_MULTIPLIER = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}
_DEFAULT_PERCENTILE = 50
_PERCENTILE_GRID = [30, 50, 70]


def _pipe(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectPercentile(f_classif, percentile=_DEFAULT_PERCENTILE)),
        ("clf", estimator),
    ])


def model_configs() -> list[dict]:
    pg = _PERCENTILE_GRID
    return [
        {
            "name": "LogisticRegression",
            "pipeline": _pipe(LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
            "pipeline_sw": _pipe(LogisticRegression(max_iter=2000, random_state=42)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__C": [0.01, 0.1, 1, 10],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "SVM_RBF",
            "pipeline": _pipe(SVC(kernel="rbf", random_state=42, class_weight="balanced", probability=True)),
            "pipeline_sw": _pipe(SVC(kernel="rbf", random_state=42, probability=True)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", "auto"],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "KNN",
            "pipeline": _pipe(KNeighborsClassifier()),
            "pipeline_sw": None,
            "param_grid": {
                "selector__percentile": pg,
                "clf__n_neighbors": [3, 5, 9, 15],
            },
            "supports_sample_weight": False,
        },
        {
            "name": "RandomForest",
            "pipeline": _pipe(RandomForestClassifier(random_state=42, class_weight="balanced_subsample")),
            "pipeline_sw": _pipe(RandomForestClassifier(random_state=42)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "GradientBoosting",
            "pipeline": _pipe(GradientBoostingClassifier(random_state=42)),
            "pipeline_sw": _pipe(GradientBoostingClassifier(random_state=42)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
            },
            "supports_sample_weight": True,
        },
    ]


def select_clinical_feature_columns(clinical: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in clinical.columns:
        if col in CLINICAL_EXCLUDE_COLS:
            continue
        series = pd.to_numeric(clinical[col], errors="coerce")
        if not series.notna().any():
            continue
        cols.append(col)
    return cols


def build_fusion_frame(asr_df: pd.DataFrame, clinical_df: pd.DataFrame, clinical_cols: list[str]) -> pd.DataFrame:
    merged = asr_df.merge(
        clinical_df[["canonical_patient_id", *clinical_cols]],
        on="canonical_patient_id",
        how="left",
    )
    return merged


def build_feature_sets(asr_cols: list[str], clinical_cols: list[str]) -> dict[str, list[str]]:
    return {
        "asr_only": asr_cols,
        "clinical_only": clinical_cols,
        "fusion": asr_cols + clinical_cols,
    }


def normalize_feature_set_names(spec: str | None, available: list[str]) -> list[str]:
    if not spec:
        return available
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ValueError(f"Unknown feature set(s): {', '.join(unknown)}")
    return wanted


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_split_feature_set(
    df: pd.DataFrame,
    split: dict,
    feature_cols: list[str],
    patient_id_col: str = "canonical_patient_id",
):
    def arrays(ids: list[str]):
        sub = df[df[patient_id_col].isin(ids)].dropna(subset=["label"])
        X = sub[feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        y = sub["label"].values.astype(int)
        sev = sub["stenosis_multiclass"].values
        pids = sub[patient_id_col].values
        return X, y, sev, pids

    X_tr, y_tr, sev_tr, pids_tr = arrays(split["train_patient_ids"])
    X_va, y_va, _, _ = arrays(split["val_patient_ids"])
    X_te, y_te, _, _ = arrays(split["test_patient_ids"])
    _, groups_tr = np.unique(pids_tr, return_inverse=True)
    return X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te


def compute_severity_weights(y: np.ndarray, stenosis: np.ndarray) -> np.ndarray:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    base_pos = n_neg / n_pos if n_pos > 0 else 1.0
    weights = np.ones(len(y), dtype=np.float64)
    for i, (label, sev) in enumerate(zip(y, stenosis)):
        if label == 1:
            sev_num = pd.to_numeric(pd.Series([sev]), errors="coerce").iloc[0]
            sev_int = int(sev_num) if pd.notna(sev_num) else 1
            weights[i] = base_pos * SEVERITY_MULTIPLIER.get(sev_int, 1.0)
    return weights


def binary_metrics(y_true, y_pred, y_prob, prefix: str) -> dict:
    metrics: dict = {f"{prefix}_acc": accuracy_score(y_true, y_pred)}
    metrics[f"{prefix}_bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics[f"{prefix}_f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics[f"{prefix}_prec"] = precision_score(y_true, y_pred, zero_division=0)
    metrics[f"{prefix}_rec"] = recall_score(y_true, y_pred, zero_division=0)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics[f"{prefix}_auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics[f"{prefix}_auc_pr"] = average_precision_score(y_true, y_prob)
    else:
        metrics[f"{prefix}_auc_roc"] = float("nan")
        metrics[f"{prefix}_auc_pr"] = float("nan")
    return metrics


def score_split(pipeline, X, y, prefix: str) -> dict:
    y_pred = pipeline.predict(X)
    try:
        y_prob = pipeline.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None
    return binary_metrics(y, y_pred, y_prob, prefix)


def fit_model(pipe, X_tr, y_tr, param_grid: dict, do_search: bool, cv_folds: int, n_jobs: int, sample_weight=None, groups=None):
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight

    if do_search:
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=cv_folds)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=True)
        gs.fit(X_tr, y_tr, groups=groups, **fit_kwargs)
        return gs.best_estimator_, gs.best_params_, float(gs.best_score_)

    pipe.fit(X_tr, y_tr, **fit_kwargs)
    return pipe, {}, float("nan")


def run_feature_set_comparison(
    cfg: dict,
    feature_set_name: str,
    X_tr,
    y_tr,
    sev_tr,
    groups_tr,
    X_va,
    y_va,
    X_te,
    y_te,
    do_search: bool,
    cv_folds: int,
    n_jobs: int,
) -> list[dict]:
    rows: list[dict] = []
    for strategy in ("standard", "severity_weighted"):
        if strategy == "severity_weighted" and not cfg["supports_sample_weight"]:
            continue
        pipe = cfg["pipeline"] if strategy == "standard" else cfg["pipeline_sw"]
        sw = None if strategy == "standard" else compute_severity_weights(y_tr, sev_tr)
        best, best_params, cv_auc = fit_model(
            pipe, X_tr, y_tr, cfg["param_grid"], do_search, cv_folds, n_jobs, sample_weight=sw, groups=groups_tr
        )
        row = {
            "feature_set": feature_set_name,
            "model": cfg["name"],
            "strategy": strategy,
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "cv_auc_roc": cv_auc,
        }
        row.update(score_split(best, X_va, y_va, "val"))
        row.update(score_split(best, X_te, y_te, "test"))
        rows.append(row)
    return rows


def load_asr_subset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "clinical_match_status" in df.columns:
        df = df[df["clinical_match_status"] == "matched"].copy()
    return df


def load_clinical_table(path: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path, dtype=str)
    for col in clinical.columns:
        if col in {"canonical_patient_id", "name", "gender", "data_sources", "icas_detail"}:
            continue
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    return clinical


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-subset", type=Path, default=repo_root / "reports/asr_candidate_modeling_subset.csv")
    parser.add_argument("--clinical", type=Path, default=repo_root / "datasets/full_data/patient_clinical_data.csv")
    parser.add_argument("--split", type=Path, default=repo_root / "configs/data_split.json")
    parser.add_argument("--feature-sets", type=str, help="Comma-separated subset: asr_only,clinical_only,fusion")
    parser.add_argument("--output", type=Path, default=repo_root / "reports")
    parser.add_argument("--no-search", action="store_true", help="Skip GridSearchCV for a faster baseline run")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asr_df = load_asr_subset(args.asr_subset)
    clinical_df = load_clinical_table(args.clinical)
    clinical_cols = select_clinical_feature_columns(clinical_df)
    fusion_df = build_fusion_frame(asr_df, clinical_df, clinical_cols)

    asr_cols = [c for c in asr_df.columns if c.startswith("asr_")]
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

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"asr_clinical_model_comparison_{timestamp}.csv"
    pd.DataFrame(rows).sort_values(["feature_set", "test_auc_roc"], ascending=[True, False]).to_csv(
        out_csv, index=False, encoding="utf-8"
    )
    print(f"Wrote comparison rows: {len(rows)}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
