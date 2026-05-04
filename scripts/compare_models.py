#!/usr/bin/env python3
"""Compare multiple classifiers on thermal temperature features.

Two tasks are evaluated in sequence:
  - binary:     predict `label`  (0 = no ICAS, 1 = ICAS)
  - multiclass: predict `stenosis_multiclass` (0/1/2/3)

The train/val/test split is loaded from configs/data_split.json (patient-level,
pre-stratified) so results are free of data leakage.

For each model a small hyperparameter grid is searched via GridSearchCV on the
training fold; the best estimator is then scored on the validation fold
(model selection) and the test fold (final, unbiased evaluation).

Results are saved to reports/model_comparison_<timestamp>.csv.

Usage
-----
  python scripts/compare_models.py                   # both tasks
  python scripts/compare_models.py --task binary
  python scripts/compare_models.py --task multiclass
  python scripts/compare_models.py --task binary --no-search  # skip grid search
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

FEATURE_META_COLS = {"sample_id", "patient_id", "year", "status"}

# ── Model definitions ──────────────────────────────────────────────────────────

def _pipe(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     estimator),
    ])


def model_configs(task: str) -> list[dict]:
    """Return list of {name, pipeline, param_grid} dicts."""
    is_binary = task == "binary"
    proba_svc = dict(probability=True)

    return [
        {
            "name": "LogisticRegression",
            "pipeline": _pipe(LogisticRegression(max_iter=2000, random_state=42,
                                                 class_weight="balanced" if is_binary else None)),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        {
            "name": "SVM_RBF",
            "pipeline": _pipe(SVC(kernel="rbf", random_state=42,
                                  class_weight="balanced" if is_binary else None,
                                  **proba_svc)),
            "param_grid": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
        },
        {
            "name": "KNN",
            "pipeline": _pipe(KNeighborsClassifier()),
            "param_grid": {"clf__n_neighbors": [3, 5, 9, 15]},
        },
        {
            "name": "RandomForest",
            "pipeline": _pipe(RandomForestClassifier(random_state=42,
                                                      class_weight="balanced_subsample" if is_binary else None)),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
            },
        },
        {
            "name": "GradientBoosting",
            "pipeline": _pipe(GradientBoostingClassifier(random_state=42)),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
            },
        },
        {
            "name": "XGBoost",
            "pipeline": _pipe(xgb.XGBClassifier(
                random_state=42, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
            )),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 6],
            },
        },
        {
            "name": "LightGBM",
            "pipeline": _pipe(lgb.LGBMClassifier(
                random_state=42, verbosity=-1,
                class_weight="balanced" if is_binary else None,
            )),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__num_leaves": [31, 63],
            },
        },
    ]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(features_path: Path, clinical_path: Path, split_path: Path) -> dict:
    feats = pd.read_csv(features_path)
    clinical = pd.read_csv(clinical_path, dtype=str)
    clinical["label"] = pd.to_numeric(clinical["label"], errors="coerce")
    clinical["stenosis_multiclass"] = clinical["stenosis_multiclass"].where(
        clinical["stenosis_multiclass"] != "missing", other=None
    )
    clinical["stenosis_multiclass"] = pd.to_numeric(clinical["stenosis_multiclass"], errors="coerce")

    merged = feats.merge(
        clinical[["canonical_patient_id", "label", "stenosis_multiclass"]],
        left_on="patient_id", right_on="canonical_patient_id", how="left",
    )

    split = json.loads(split_path.read_text(encoding="utf-8"))
    feature_cols = [c for c in feats.columns if c not in FEATURE_META_COLS]

    return {
        "df": merged,
        "feature_cols": feature_cols,
        "split": split,
    }


def apply_split(df: pd.DataFrame, split: dict, label_col: str) -> tuple:
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays."""
    def subset(patient_ids, df, label_col):
        sub = df[df["patient_id"].isin(patient_ids)].dropna(subset=[label_col])
        X = sub[[c for c in df.columns if c not in FEATURE_META_COLS
                  and c not in {"label", "stenosis_multiclass", "canonical_patient_id"}]]
        y = sub[label_col]
        return X.values.astype(np.float32), y.values

    X_tr, y_tr = subset(split["train_patient_ids"], df, label_col)
    X_va, y_va = subset(split["val_patient_ids"],   df, label_col)
    X_te, y_te = subset(split["test_patient_ids"],  df, label_col)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ── Metrics ────────────────────────────────────────────────────────────────────

def binary_metrics(y_true, y_pred, y_prob, prefix: str) -> dict:
    m: dict = {f"{prefix}_acc": accuracy_score(y_true, y_pred)}
    m[f"{prefix}_bal_acc"] = balanced_accuracy_score(y_true, y_pred)
    m[f"{prefix}_f1"]      = f1_score(y_true, y_pred, zero_division=0)
    m[f"{prefix}_prec"]    = precision_score(y_true, y_pred, zero_division=0)
    m[f"{prefix}_rec"]     = recall_score(y_true, y_pred, zero_division=0)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m[f"{prefix}_auc_roc"] = roc_auc_score(y_true, y_prob)
        m[f"{prefix}_auc_pr"]  = average_precision_score(y_true, y_prob)
    else:
        m[f"{prefix}_auc_roc"] = float("nan")
        m[f"{prefix}_auc_pr"]  = float("nan")
    return m


def multiclass_metrics(y_true, y_pred, y_prob, prefix: str, classes) -> dict:
    m: dict = {f"{prefix}_acc": accuracy_score(y_true, y_pred)}
    m[f"{prefix}_macro_f1"]   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m[f"{prefix}_macro_prec"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m[f"{prefix}_macro_rec"]  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    # Per-class recall
    per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=sorted(classes))
    for cls, r in zip(sorted(classes), per_class):
        m[f"{prefix}_rec_cls{int(cls)}"] = r
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            m[f"{prefix}_auc_roc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=sorted(classes),
            )
        except Exception:
            m[f"{prefix}_auc_roc_ovr"] = float("nan")
    else:
        m[f"{prefix}_auc_roc_ovr"] = float("nan")
    return m


# ── Training & evaluation ──────────────────────────────────────────────────────

def score_split(pipeline, X, y, task, classes, prefix) -> dict:
    y_pred = pipeline.predict(X)
    try:
        y_prob = pipeline.predict_proba(X)
    except Exception:
        y_prob = None

    if task == "binary":
        prob1 = y_prob[:, 1] if y_prob is not None else None
        return binary_metrics(y, y_pred, prob1, prefix)
    else:
        return multiclass_metrics(y, y_pred, y_prob, prefix, classes)


def run_model(cfg: dict, X_tr, y_tr, X_va, y_va, X_te, y_te,
              task: str, classes, do_search: bool, cv_folds: int) -> dict:
    pipe = cfg["pipeline"]

    if do_search:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "roc_auc" if task == "binary" else "f1_macro"
        gs = GridSearchCV(pipe, cfg["param_grid"], scoring=scoring,
                          cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        best_params = gs.best_params_
    else:
        pipe.fit(X_tr, y_tr)
        best = pipe
        best_params = {}

    row = {
        "model": cfg["name"],
        "task": task,
        "best_params": json.dumps(best_params, ensure_ascii=False),
    }
    row.update(score_split(best, X_va, y_va, task, classes, prefix="val"))
    row.update(score_split(best, X_te, y_te, task, classes, prefix="test"))
    return row


def run_task(task: str, label_col: str, data: dict,
             do_search: bool, cv_folds: int) -> list[dict]:
    df   = data["df"]
    split = data["split"]

    X_tr, y_tr, X_va, y_va, X_te, y_te = apply_split(df, split, label_col)

    if task == "multiclass":
        # Encode string labels to integers for XGB/LGBM
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr)
        y_va = le.transform(y_va)
        y_te = le.transform(y_te)
        classes = le.classes_
    else:
        classes = np.unique(y_tr)

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  label: {label_col}")
    print(f"  Train: {len(y_tr)} samples  (pos={int((y_tr==1).sum()) if task=='binary' else 'N/A'})")
    print(f"  Val  : {len(y_va)} samples")
    print(f"  Test : {len(y_te)} samples")
    print(f"{'='*60}")

    rows = []
    configs = model_configs(task)
    for cfg in configs:
        print(f"  Training {cfg['name']}...", end=" ", flush=True)
        try:
            row = run_model(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te,
                            task, classes, do_search, cv_folds)
            rows.append(row)
            if task == "binary":
                print(f"val_auc={row.get('val_auc_roc', float('nan')):.3f}  "
                      f"test_auc={row.get('test_auc_roc', float('nan')):.3f}")
            else:
                print(f"val_macro_f1={row.get('val_macro_f1', float('nan')):.3f}  "
                      f"test_macro_f1={row.get('test_macro_f1', float('nan')):.3f}")
        except Exception as exc:
            print(f"FAILED: {exc}")
    return rows


def save_results(rows: list[dict], output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"model_comparison_{timestamp}.csv"
    df = pd.DataFrame(rows)
    # Sort by task, then by best validation metric
    df.to_csv(path, index=False)
    return path


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for task, grp in df.groupby("task"):
        print(f"\n{'─'*60}")
        print(f"RESULTS — Task: {task}")
        print(f"{'─'*60}")
        if task == "binary":
            cols = ["model", "val_auc_roc", "val_bal_acc", "val_f1",
                    "test_auc_roc", "test_bal_acc", "test_f1"]
        else:
            cols = ["model", "val_macro_f1", "val_acc", "val_auc_roc_ovr",
                    "test_macro_f1", "test_acc", "test_auc_roc_ovr"]
        show = grp[[c for c in cols if c in grp.columns]].copy()
        for c in show.columns[1:]:
            show[c] = show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
        print(show.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features",  type=Path, default=Path("datasets/temperature_features.csv"))
    parser.add_argument("--clinical",  type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split",     type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--output",    type=Path, default=Path("reports"))
    parser.add_argument("--task",      choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--no-search", action="store_true", help="Skip grid search, use default params")
    parser.add_argument("--cv-folds",  type=int, default=5)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = load_data(args.features, args.clinical, args.split)
    do_search = not args.no_search

    all_rows: list[dict] = []
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]

    for task in tasks:
        label_col = "label" if task == "binary" else "stenosis_multiclass"
        rows = run_task(task, label_col, data, do_search=do_search, cv_folds=args.cv_folds)
        all_rows.extend(rows)

    path = save_results(all_rows, args.output, timestamp)
    print_summary(all_rows)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()

