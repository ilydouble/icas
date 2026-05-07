#!/usr/bin/env python3
"""Compare thermal CNN, clinical top-3, and shallow thermal+clinical fusion.

This script keeps the deep thermal model fixed and compares:
1. thermal_only: direct CNN probability on the selected patient pool
2. clinical_only: a simple classical model on top-3 clinical variables
3. weighted_late_fusion: validation-tuned alpha blend
4. logistic_stacking: shallow meta-model over [p_thermal, p_clinical]
5. tree_stacking: shallow decision tree over [p_thermal, p_clinical]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_asr_clinical_models import binary_metrics, compute_severity_weights, fit_model, load_split, model_configs
from scripts.compare_fusion_models import (
    build_model_from_results,
    extract_deep_feature_frame,
    load_clinical_table,
    load_results_json,
)
from scripts.train_cnn_v3 import TemperatureDataset, load_data as load_cnn_data

CLINICAL_TOP3 = [
    "waist_hip_ratio",
    "gender_encoded",
    "height",
]
ASR_TOP9 = [
    "asr_speech_rate_min",
    "asr_chars_per_sentence_mean",
    "asr_chars_per_second",
    "asr_emotion_median",
    "asr_long_pause_sentence_ratio",
    "asr_pause_sentence_ratio",
    "asr_sentence_duration_ms_mean",
    "asr_sentence_duration_ms_min",
    "asr_silence_duration_ms_mean",
]


def load_patient_pool(clinical_subset_path: Path) -> set[str]:
    clinical_df = pd.read_csv(clinical_subset_path)
    return set(clinical_df["canonical_patient_id"].astype(str))


def build_probability_frame(
    deep_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    patient_pool: set[str],
) -> pd.DataFrame:
    deep_sub = deep_df.loc[deep_df["patient_id"].astype(str).isin(patient_pool), ["sample_id", "patient_id", "label", "stenosis_multiclass", "cnn_prob"]].copy()
    cli_sub = clinical_df.loc[
        clinical_df["canonical_patient_id"].astype(str).isin(patient_pool),
        ["canonical_patient_id", *CLINICAL_TOP3],
    ].copy()
    merged = deep_sub.merge(cli_sub, left_on="patient_id", right_on="canonical_patient_id", how="inner")
    return merged.drop(columns=["canonical_patient_id"])


def build_three_way_probability_frame(
    deep_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    asr_prob_df: pd.DataFrame,
    patient_pool: set[str],
) -> pd.DataFrame:
    base = build_probability_frame(deep_df, clinical_df, patient_pool)
    asr_sub = asr_prob_df.loc[
        asr_prob_df["canonical_patient_id"].astype(str).isin(patient_pool),
        ["canonical_patient_id", "asr_prob"],
    ].copy()
    merged = base.merge(asr_sub, left_on="patient_id", right_on="canonical_patient_id", how="inner")
    return merged.drop(columns=["canonical_patient_id"])


def apply_split(df: pd.DataFrame, split: dict, feature_cols: list[str]) -> tuple:
    def arrays(ids: list[str]):
        sub = df[df["patient_id"].isin(ids)].dropna(subset=["label"])
        X = sub[feature_cols].values.astype(np.float32)
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
        preds = model.predict(X)
        return np.asarray(preds, dtype=float)


def blend_probabilities(thermal_prob: np.ndarray, clinical_prob: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * np.asarray(thermal_prob, dtype=float) + (1.0 - alpha) * np.asarray(clinical_prob, dtype=float)


def pick_best_alpha(
    thermal_prob: np.ndarray,
    clinical_prob: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: list[float] | np.ndarray,
) -> tuple[float, float]:
    best_alpha = float(alpha_grid[0])
    best_auc = float("-inf")
    for alpha in alpha_grid:
        fused = blend_probabilities(thermal_prob, clinical_prob, float(alpha))
        auc = roc_auc_score(y_val, fused) if len(np.unique(y_val)) > 1 else float("nan")
        if auc > best_auc:
            best_auc = float(auc)
            best_alpha = float(alpha)
    return best_alpha, best_auc


def make_meta_features(
    thermal_prob: np.ndarray,
    clinical_prob: np.ndarray,
    asr_prob: np.ndarray | None = None,
) -> np.ndarray:
    cols = [np.asarray(thermal_prob, dtype=float), np.asarray(clinical_prob, dtype=float)]
    if asr_prob is not None:
        cols.append(np.asarray(asr_prob, dtype=float))
    return np.column_stack(cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True, help="Thermal CNN results JSON to describe the fixed deep model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint matching --results-json")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--clinical-subset", type=Path, default=Path("reports/clinical_candidate_modeling_subset.csv"))
    parser.add_argument("--asr-subset", type=Path, default=Path("reports/asr_candidate_modeling_subset.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clinical-model", type=str, default="LogisticRegression")
    parser.add_argument("--clinical-strategy", type=str, default="standard", choices=["standard", "severity_weighted"])
    parser.add_argument("--asr-model", type=str, default="GradientBoosting")
    parser.add_argument("--asr-strategy", type=str, default="standard", choices=["standard", "severity_weighted"])
    parser.add_argument("--alpha-grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results_json(args.results_json)
    split = load_split(args.split)
    patient_pool = load_patient_pool(args.clinical_subset)
    split = {
        **split,
        "train_patient_ids": [pid for pid in split["train_patient_ids"] if pid in patient_pool],
        "val_patient_ids": [pid for pid in split["val_patient_ids"] if pid in patient_pool],
        "test_patient_ids": [pid for pid in split["test_patient_ids"] if pid in patient_pool],
    }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model_from_results(results).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    repo_root = Path(".").resolve()
    data = load_cnn_data(args.manifest, args.clinical, args.split, args.excluded, args.annotations)
    all_samples = data["train"] + data["val"] + data["test"]

    dataset = TemperatureDataset(
        all_samples,
        data["annotations"],
        data["labels"],
        data["severities"],
        repo_root,
        args.masks_dir,
        target_size=(int(results.get("target_size", 64)), int(results.get("target_size", 64))),
        use_mask=bool(results.get("use_face_mask", True)),
        augment=False,
        region_attention=bool(results.get("use_region_attention")),
        npy_dir=args.npy_dir if args.npy_dir.exists() else None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sample_to_patient = {sample["sample_id"]: sample["canonical_patient_id"] for sample in all_samples}
    sample_to_year = {sample["sample_id"]: int(sample["year"]) for sample in all_samples}
    deep_df = extract_deep_feature_frame(model, loader, device, sample_to_patient, sample_to_year)

    clinical_df = load_clinical_table(args.clinical_subset)
    prob_df = build_probability_frame(deep_df, clinical_df, patient_pool)

    # Thermal branch: fixed CNN probability.
    thermal_cols = ["cnn_prob"]
    X_tr_th, y_tr, sev_tr, X_va_th, y_va, _, X_te_th, y_te, _ = apply_split(prob_df, split, thermal_cols)
    thermal_val_prob = X_va_th[:, 0]
    thermal_test_prob = X_te_th[:, 0]

    # Clinical branch: fit a shallow clinical model on top-3 features.
    X_tr_cli, _, _, X_va_cli, _, _, X_te_cli, _, _ = apply_split(prob_df, split, CLINICAL_TOP3)
    clinical_cfg = next(cfg for cfg in model_configs() if cfg["name"] == args.clinical_model)
    if args.clinical_strategy == "severity_weighted" and not clinical_cfg["supports_sample_weight"]:
        raise ValueError(f"Model {args.clinical_model} does not support strategy={args.clinical_strategy}")
    pipe = clinical_cfg["pipeline"] if args.clinical_strategy == "standard" else clinical_cfg["pipeline_sw"]
    sw = None if args.clinical_strategy == "standard" else compute_severity_weights(y_tr, sev_tr)
    clinical_model, clinical_params, clinical_cv = fit_model(
        pipe,
        X_tr_cli,
        y_tr,
        clinical_cfg["param_grid"],
        do_search=not args.no_search,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        sample_weight=sw,
        groups=None,
    )
    clinical_val_prob = predict_prob(clinical_model, X_va_cli)
    clinical_test_prob = predict_prob(clinical_model, X_te_cli)

    # Optional ASR branch for 3-way stacking.
    asr_df = pd.read_csv(args.asr_subset)
    if "clinical_match_status" in asr_df.columns:
        asr_df = asr_df.loc[asr_df["clinical_match_status"] == "matched"].copy()
    keep_cols = ["canonical_patient_id", "label", "stenosis_multiclass", *ASR_TOP9]
    asr_df = asr_df.loc[:, keep_cols].copy()
    for col in ASR_TOP9:
        asr_df[col] = pd.to_numeric(asr_df[col], errors="coerce")
    asr_df["label"] = pd.to_numeric(asr_df["label"], errors="coerce")
    asr_df["stenosis_multiclass"] = pd.to_numeric(asr_df["stenosis_multiclass"], errors="coerce")
    asr_df = asr_df.dropna(subset=["label", *ASR_TOP9]).reset_index(drop=True)

    asr_patient_pool = patient_pool & set(asr_df["canonical_patient_id"].astype(str))
    split_asr = {
        **split,
        "train_patient_ids": [pid for pid in split["train_patient_ids"] if pid in asr_patient_pool],
        "val_patient_ids": [pid for pid in split["val_patient_ids"] if pid in asr_patient_pool],
        "test_patient_ids": [pid for pid in split["test_patient_ids"] if pid in asr_patient_pool],
    }
    three_way_df = build_three_way_probability_frame(
        deep_df,
        clinical_df,
        pd.DataFrame(
            {
                "canonical_patient_id": asr_df["canonical_patient_id"].astype(str),
                # placeholder to be replaced after fitting ASR
                "asr_prob": np.zeros(len(asr_df), dtype=float),
            }
        ),
        asr_patient_pool,
    )
    X_tr_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["train_patient_ids"])][ASR_TOP9].values.astype(np.float32)
    y_tr_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["train_patient_ids"])]["label"].astype(int).values
    sev_tr_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["train_patient_ids"])]["stenosis_multiclass"].values
    X_va_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["val_patient_ids"])][ASR_TOP9].values.astype(np.float32)
    y_va_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["val_patient_ids"])]["label"].astype(int).values
    X_te_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["test_patient_ids"])][ASR_TOP9].values.astype(np.float32)
    y_te_asr = asr_df[asr_df["canonical_patient_id"].isin(split_asr["test_patient_ids"])]["label"].astype(int).values

    asr_cfg = next(cfg for cfg in model_configs() if cfg["name"] == args.asr_model)
    if args.asr_strategy == "severity_weighted" and not asr_cfg["supports_sample_weight"]:
        raise ValueError(f"Model {args.asr_model} does not support strategy={args.asr_strategy}")
    asr_pipe = asr_cfg["pipeline"] if args.asr_strategy == "standard" else asr_cfg["pipeline_sw"]
    asr_sw = None if args.asr_strategy == "standard" else compute_severity_weights(y_tr_asr, sev_tr_asr)
    asr_model, asr_params, asr_cv = fit_model(
        asr_pipe,
        X_tr_asr,
        y_tr_asr,
        asr_cfg["param_grid"],
        do_search=not args.no_search,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        sample_weight=asr_sw,
        groups=None,
    )
    asr_val_prob = predict_prob(asr_model, X_va_asr)
    asr_test_prob = predict_prob(asr_model, X_te_asr)

    # Weighted late fusion.
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    best_alpha, best_val_auc = pick_best_alpha(thermal_val_prob, clinical_val_prob, y_va, alpha_grid)
    fused_test_prob = blend_probabilities(thermal_test_prob, clinical_test_prob, best_alpha)
    fused_test_pred = (fused_test_prob >= 0.5).astype(int)

    # Meta models fit only on validation branch probabilities.
    X_meta_val = make_meta_features(thermal_val_prob, clinical_val_prob)
    X_meta_test = make_meta_features(thermal_test_prob, clinical_test_prob)

    meta_logit = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    meta_logit.fit(X_meta_val, y_va)
    logit_test_prob = meta_logit.predict_proba(X_meta_test)[:, 1]
    logit_test_pred = (logit_test_prob >= 0.5).astype(int)

    meta_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    meta_tree.fit(X_meta_val, y_va)
    tree_test_prob = meta_tree.predict_proba(X_meta_test)[:, 1]
    tree_test_pred = (tree_test_prob >= 0.5).astype(int)

    rows = []
    rows.append({
        "method": "thermal_only",
        **binary_metrics(y_te, (thermal_test_prob >= 0.5).astype(int), thermal_test_prob, "test"),
    })
    rows.append({
        "method": "clinical_only",
        "clinical_model": args.clinical_model,
        "clinical_strategy": args.clinical_strategy,
        "clinical_best_params": json.dumps(clinical_params, ensure_ascii=False),
        "clinical_cv_auc_roc": clinical_cv,
        **binary_metrics(y_te, (clinical_test_prob >= 0.5).astype(int), clinical_test_prob, "test"),
    })
    rows.append({
        "method": "weighted_late_fusion",
        "best_alpha_thermal_weight": best_alpha,
        "val_fused_auc_roc": best_val_auc,
        **binary_metrics(y_te, fused_test_pred, fused_test_prob, "test"),
    })
    rows.append({
        "method": "logistic_stacking",
        **binary_metrics(y_te, logit_test_pred, logit_test_prob, "test"),
    })
    rows.append({
        "method": "tree_stacking_depth2",
        **binary_metrics(y_te, tree_test_pred, tree_test_prob, "test"),
    })

    # Three-way fusion and stacking on the ASR-complete subset only.
    thermal_prob_lookup = dict(zip(prob_df["patient_id"], prob_df["cnn_prob"]))
    clinical_prob_lookup = dict(zip(prob_df["patient_id"], predict_prob(clinical_model, prob_df[CLINICAL_TOP3].values.astype(np.float32))))
    asr_prob_lookup = dict(zip(asr_df["canonical_patient_id"].astype(str), predict_prob(asr_model, asr_df[ASR_TOP9].values.astype(np.float32))))

    def build_three_way_arrays(ids: list[str]):
        sample_sub = deep_df[deep_df["patient_id"].isin(ids)].copy()
        sample_sub = sample_sub[sample_sub["patient_id"].isin(asr_patient_pool)].copy()
        y = sample_sub["label"].astype(int).values
        thermal_prob = sample_sub["cnn_prob"].astype(float).values
        clinical_prob = sample_sub["patient_id"].map(clinical_prob_lookup).astype(float).values
        asr_prob = sample_sub["patient_id"].map(asr_prob_lookup).astype(float).values
        return y, thermal_prob, clinical_prob, asr_prob

    y_va_3, thermal_va_3, clinical_va_3, asr_va_3 = build_three_way_arrays(split_asr["val_patient_ids"])
    y_te_3, thermal_te_3, clinical_te_3, asr_te_3 = build_three_way_arrays(split_asr["test_patient_ids"])
    X_meta_val_3 = make_meta_features(thermal_va_3, clinical_va_3, asr_va_3)
    X_meta_test_3 = make_meta_features(thermal_te_3, clinical_te_3, asr_te_3)

    meta_logit_3 = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    meta_logit_3.fit(X_meta_val_3, y_va_3)
    logit_test_prob_3 = meta_logit_3.predict_proba(X_meta_test_3)[:, 1]
    logit_test_pred_3 = (logit_test_prob_3 >= 0.5).astype(int)

    meta_tree_3 = DecisionTreeClassifier(max_depth=2, random_state=42)
    meta_tree_3.fit(X_meta_val_3, y_va_3)
    tree_test_prob_3 = meta_tree_3.predict_proba(X_meta_test_3)[:, 1]
    tree_test_pred_3 = (tree_test_prob_3 >= 0.5).astype(int)

    rows.append({
        "method": "logistic_stacking_3way",
        "asr_model": args.asr_model,
        "asr_strategy": args.asr_strategy,
        "asr_best_params": json.dumps(asr_params, ensure_ascii=False),
        "asr_cv_auc_roc": asr_cv,
        **binary_metrics(y_te_3, logit_test_pred_3, logit_test_prob_3, "test"),
    })
    rows.append({
        "method": "tree_stacking_depth2_3way",
        **binary_metrics(y_te_3, tree_test_pred_3, tree_test_prob_3, "test"),
    })

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output / f"thermal_clinical_late_fusion_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Patient pool size: {len(patient_pool)}")
    print(f"Three-way ASR-complete patient pool size: {len(asr_patient_pool)}")
    print(f"Thermal results JSON: {args.results_json}")
    print(f"Thermal checkpoint: {args.checkpoint}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
