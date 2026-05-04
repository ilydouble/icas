#!/usr/bin/env python3
"""Compare classifiers for ICAS binary classification using severity-weighted training.

Task
----
Predict `label` (0 = no ICAS, 1 = has ICAS).

The `stenosis_multiclass` field (0=none, 1=mild, 2=moderate, 3=severe) is used
as an **auxiliary supervision signal**: positive training samples are weighted
by their stenosis severity so the model is penalised more for misclassifying
severe cases.  Evaluation always uses the binary label only.

Two training strategies are compared for every model:

  standard          — class_weight="balanced", no severity weighting.
  severity_weighted — sample weights that encode both class imbalance and
                      stenosis severity (see compute_severity_weights).

Models that do not support sample_weight (KNN) are only run with `standard`.

Feature selection
-----------------
Each pipeline includes a SelectPercentile(f_classif) step between the scaler
and the classifier.  The percentile threshold is tuned by GridSearchCV strictly
inside the training fold, preventing any information from val/test leaking into
the feature-selection step.

Model selection
---------------
The winning model for each strategy is chosen by `cv_auc_roc` — the
cross-validated ROC-AUC on the **training set** produced by GridSearchCV.
Val and test sets are evaluated for reporting only and are never used
for selection decisions.

Split
-----
Loaded from configs/data_split.json (patient-level, pre-stratified).

Output
------
reports/model_comparison_<timestamp>.csv — one row per (model × strategy).

Usage
-----
  python scripts/compare_models.py                  # full run with grid search
  python scripts/compare_models.py --no-search      # default params, faster
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

warnings.filterwarnings("ignore")

FEATURE_META_COLS = {"sample_id", "patient_id", "year", "status"}

# Severity → relative weight multiplier for positive (ICAS) samples.
# stenosis=0 negatives always get weight 1.0; positives are scaled by severity.
SEVERITY_MULTIPLIER = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}


# Default percentile for SelectPercentile when grid search is disabled.
_DEFAULT_PERCENTILE = 50

# Percentile values explored during grid search (% of top-scoring features to keep).
_PERCENTILE_GRID = [30, 50, 70]


# ── Pipeline factory ───────────────────────────────────────────────────────────

def _pipe(estimator) -> Pipeline:
    """Build a pipeline: impute → scale → select features → classify.

    SelectPercentile(f_classif) filters features by ANOVA F-value, keeping only
    the top `percentile` percent.  It is placed inside the pipeline so that
    feature selection is re-fitted on each training fold and never sees val/test
    data.
    """
    return Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("scaler",   StandardScaler()),
        ("selector", SelectPercentile(f_classif, percentile=_DEFAULT_PERCENTILE)),
        ("clf",      estimator),
    ])


def model_configs() -> list[dict]:
    """Return list of model descriptors for binary classification.

    Each entry has:
      name                   – display name
      pipeline               – sklearn Pipeline (standard variant with class_weight)
      pipeline_sw            – sklearn Pipeline for severity_weighted (no class_weight,
                               sample_weight will be passed at fit time)
      param_grid             – hyperparameter grid for GridSearchCV; always includes
                               selector__percentile so feature selection is jointly tuned
      supports_sample_weight – whether clf.fit() accepts sample_weight
    """
    pg = _PERCENTILE_GRID  # shorthand

    configs = [
        {
            "name": "LogisticRegression",
            "pipeline":    _pipe(LogisticRegression(max_iter=2000, random_state=42,
                                                    class_weight="balanced")),
            "pipeline_sw": _pipe(LogisticRegression(max_iter=2000, random_state=42)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__C": [0.01, 0.1, 1, 10],
            },
            "supports_sample_weight": True,
        },
        {
            "name": "SVM_RBF",
            "pipeline":    _pipe(SVC(kernel="rbf", random_state=42,
                                     class_weight="balanced", probability=True)),
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
            "pipeline":    _pipe(KNeighborsClassifier()),
            "pipeline_sw": None,   # KNN has no sample_weight
            "param_grid": {
                "selector__percentile": pg,
                "clf__n_neighbors": [3, 5, 9, 15],
            },
            "supports_sample_weight": False,
        },
        {
            "name": "RandomForest",
            "pipeline":    _pipe(RandomForestClassifier(random_state=42,
                                                         class_weight="balanced_subsample")),
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
            "pipeline":    _pipe(GradientBoostingClassifier(random_state=42)),
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

    if _HAS_XGB:
        configs.append({
            "name": "XGBoost",
            "pipeline":    _pipe(xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                                                    use_label_encoder=False, verbosity=0)),
            "pipeline_sw": _pipe(xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                                                    use_label_encoder=False, verbosity=0)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 6],
            },
            "supports_sample_weight": True,
        })

    if _HAS_LGB:
        configs.append({
            "name": "LightGBM",
            "pipeline":    _pipe(lgb.LGBMClassifier(random_state=42, verbosity=-1,
                                                     class_weight="balanced")),
            "pipeline_sw": _pipe(lgb.LGBMClassifier(random_state=42, verbosity=-1)),
            "param_grid": {
                "selector__percentile": pg,
                "clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.05, 0.1],
                "clf__num_leaves": [31, 63],
            },
            "supports_sample_weight": True,
        })

    return configs


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(features_path: Path, clinical_path: Path, split_path: Path) -> dict:
    feats = pd.read_csv(features_path)
    clinical = pd.read_csv(clinical_path, dtype=str)
    clinical["label"] = pd.to_numeric(clinical["label"], errors="coerce")
    clinical["stenosis_multiclass"] = clinical["stenosis_multiclass"].where(
        clinical["stenosis_multiclass"] != "missing", other=None
    )
    clinical["stenosis_multiclass"] = pd.to_numeric(
        clinical["stenosis_multiclass"], errors="coerce"
    )
    merged = feats.merge(
        clinical[["canonical_patient_id", "label", "stenosis_multiclass"]],
        left_on="patient_id", right_on="canonical_patient_id", how="left",
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {"df": merged, "split": split}





def apply_split(df: pd.DataFrame, split: dict) -> tuple:
    """Return arrays for train/val/test, including stenosis severity for weighting.

    Returns:
        X_tr, y_tr, sev_tr,   – train features, labels, stenosis severity
        X_va, y_va,           – val features, labels
        X_te, y_te            – test features, labels
    """
    label_cols = {"label", "stenosis_multiclass", "canonical_patient_id"}
    feat_cols = [c for c in df.columns if c not in FEATURE_META_COLS and c not in label_cols]

    def arrays(ids):
        sub = df[df["patient_id"].isin(ids)].dropna(subset=["label"])
        X = sub[feat_cols].values.astype(np.float32)
        y = sub["label"].values.astype(int)
        sev = sub["stenosis_multiclass"].values  # may contain NaN
        return X, y, sev

    X_tr, y_tr, sev_tr = arrays(split["train_patient_ids"])
    X_va, y_va, _      = arrays(split["val_patient_ids"])
    X_te, y_te, _      = arrays(split["test_patient_ids"])
    return X_tr, y_tr, sev_tr, X_va, y_va, X_te, y_te


def compute_severity_weights(y: np.ndarray, stenosis: np.ndarray) -> np.ndarray:
    """Compute per-sample training weights that encode both class balance and severity.

    Weight formula for sample i:
        w_i = balance_factor(y_i)  ×  severity_multiplier(stenosis_i)

    balance_factor mirrors sklearn's class_weight="balanced":
        neg → n_pos / n_neg  (upweight negatives relative to majority class)
        pos → 1.0
    Wait — actually we want positives to be at least as important as negatives.
    We set:
        neg → 1.0
        pos → (n_neg / n_pos) × severity_multiplier

    This ensures:
      • Class imbalance is corrected (same total weight for each class).
      • Among positives, severe cases carry more weight than mild ones.
    """
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    base_pos = n_neg / n_pos if n_pos > 0 else 1.0

    weights = np.ones(len(y), dtype=np.float64)
    for i, (label, sev) in enumerate(zip(y, stenosis)):
        if label == 1:
            sev_int = int(sev) if not (isinstance(sev, float) and np.isnan(sev)) else 1
            multiplier = SEVERITY_MULTIPLIER.get(sev_int, 1.0)
            weights[i] = base_pos * multiplier
    return weights


# ── Metrics (binary only) ────────────────────────────────────────────────────

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


def score_split(pipeline, X, y, prefix: str) -> dict:
    y_pred = pipeline.predict(X)
    try:
        y_prob = pipeline.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None
    return binary_metrics(y, y_pred, y_prob, prefix)


# ── Training & evaluation ──────────────────────────────────────────────────────

def _fit(pipe: Pipeline, X_tr, y_tr,
         param_grid: dict, do_search: bool, cv_folds: int,
         sample_weight=None) -> tuple:
    """Fit pipeline, optionally with GridSearchCV.

    Returns (best_estimator, best_params, cv_auc_roc).
    `cv_auc_roc` is GridSearchCV's best_score_ (mean ROC-AUC across CV folds on
    the training set).  It is used for model selection so that val/test sets are
    never touched during selection.  Returns nan when do_search=False.
    """
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight

    if do_search:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, param_grid, scoring="roc_auc",
                          cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr, **fit_kwargs)
        return gs.best_estimator_, gs.best_params_, float(gs.best_score_)
    else:
        pipe.fit(X_tr, y_tr, **fit_kwargs)
        return pipe, {}, float("nan")


def run_one_model_strategy(
    cfg: dict,
    strategy: str,
    X_tr, y_tr, sev_tr,
    X_va, y_va,
    X_te, y_te,
    do_search: bool,
    cv_folds: int,
) -> dict:
    """Train and evaluate one (model, strategy) combination.

    Model selection uses `cv_auc_roc` (training-set cross-validation score).
    Val and test scores are recorded for reporting only.
    """
    if strategy == "standard":
        pipe = cfg["pipeline"]
        sw = None
    else:  # severity_weighted
        pipe = cfg["pipeline_sw"]
        sw = compute_severity_weights(y_tr, sev_tr)

    best, best_params, cv_auc = _fit(pipe, X_tr, y_tr, cfg["param_grid"],
                                     do_search, cv_folds, sample_weight=sw)

    row = {
        "model": cfg["name"],
        "strategy": strategy,
        "best_params": json.dumps(best_params, ensure_ascii=False),
        "cv_auc_roc": cv_auc,   # training-CV score → used for model selection
    }
    row.update(score_split(best, X_va, y_va, prefix="val"))
    row.update(score_split(best, X_te, y_te, prefix="test"))
    return row


def _pos_w(y_tr, sev_level: int) -> float:
    """Helper: effective weight of a positive sample with given severity."""
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    base = n_neg / n_pos if n_pos > 0 else 1.0
    return base * SEVERITY_MULTIPLIER.get(sev_level, 1.0)


def run_comparison(data: dict, do_search: bool, cv_folds: int) -> list[dict]:
    """Run all (model × strategy) combinations for binary ICAS classification."""
    df    = data["df"]
    split = data["split"]

    X_tr, y_tr, sev_tr, X_va, y_va, X_te, y_te = apply_split(df, split)

    print(f"\n{'='*60}")
    print("Task: binary ICAS classification  (label: 0=no ICAS, 1=ICAS)")
    print(f"  Train : {len(y_tr)} samples  (pos={int((y_tr==1).sum())}, "
          f"neg={int((y_tr==0).sum())})")
    print(f"  Val   : {len(y_va)} samples  (pos={int((y_va==1).sum())})")
    print(f"  Test  : {len(y_te)} samples  (pos={int((y_te==1).sum())})")
    print(f"\n  Severity-weighted: neg→1.0, mild→{_pos_w(y_tr,1):.2f}, "
          f"mod→{_pos_w(y_tr,2):.2f}, severe→{_pos_w(y_tr,3):.2f}")
    print(f"{'='*60}")

    rows: list[dict] = []
    for cfg in model_configs():
        for strategy in ("standard", "severity_weighted"):
            if strategy == "severity_weighted" and not cfg["supports_sample_weight"]:
                continue
            tag = f"{cfg['name']}[{strategy}]"
            print(f"  {tag:<40}", end=" ", flush=True)
            try:
                row = run_one_model_strategy(
                    cfg, strategy,
                    X_tr, y_tr, sev_tr,
                    X_va, y_va,
                    X_te, y_te,
                    do_search, cv_folds,
                )
                rows.append(row)
                cv_str = f"{row['cv_auc_roc']:.3f}" if pd.notna(row["cv_auc_roc"]) else " n/a"
                print(f"cv_auc={cv_str}  val_auc={row['val_auc_roc']:.3f}  test_auc={row['test_auc_roc']:.3f}")
            except Exception as exc:
                print(f"FAILED: {exc}")
    return rows


# ── Output ─────────────────────────────────────────────────────────────────────

def save_results(rows: list[dict], output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"model_comparison_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cols = ["model", "strategy",
            "cv_auc_roc",
            "val_auc_roc", "val_auc_pr", "val_bal_acc", "val_f1",
            "test_auc_roc", "test_auc_pr", "test_bal_acc", "test_f1"]
    show = df[[c for c in cols if c in df.columns]].copy()

    print(f"\n{'─'*100}")
    print("RESULTS — Binary ICAS classification")
    print(f"  cv_auc_roc : training-set cross-validation AUC (used for model selection)")
    print(f"  val_auc_roc: held-out validation AUC          (reporting only)")
    print(f"  test_auc_roc: held-out test AUC               (final unbiased estimate)")
    print(f"{'─'*100}")
    for c in show.columns[2:]:
        show[c] = show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
    print(show.to_string(index=False))

    print(f"\n{'─'*100}")
    print("Best model per strategy (selected by cv_auc_roc — training CV, NOT val):")
    for strat, grp in df.groupby("strategy"):
        best_idx = grp["cv_auc_roc"].idxmax()
        best = grp.loc[best_idx]
        cv_str  = f"{best['cv_auc_roc']:.4f}" if pd.notna(best["cv_auc_roc"]) else "n/a"
        print(f"  {strat:<20}  {best['model']:<22}"
              f"  cv_auc={cv_str}"
              f"  val_auc={best['val_auc_roc']:.4f}"
              f"  test_auc={best['test_auc_roc']:.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--features", type=Path,
                        default=Path("datasets/temperature_features.csv"))
    parser.add_argument("--clinical", type=Path,
                        default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split",    type=Path,
                        default=Path("configs/data_split.json"))
    parser.add_argument("--output",   type=Path, default=Path("reports"))
    parser.add_argument("--no-search", action="store_true",
                        help="Skip GridSearchCV; use default hyperparameters (faster)")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = load_data(args.features, args.clinical, args.split)
    rows = run_comparison(data, do_search=not args.no_search, cv_folds=args.cv_folds)

    path = save_results(rows, args.output, timestamp)
    print_summary(rows)
    print(f"\nFull results saved to: {path}")


if __name__ == "__main__":
    main()

