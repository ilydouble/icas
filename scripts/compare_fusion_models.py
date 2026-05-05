#!/usr/bin/env python3
"""Compare classical models on CNN deep features fused with clinical data.

Pipeline:
1. Load a trained CNN checkpoint and its results JSON.
2. Extract per-sample deep embeddings from the CNN backbone.
3. Merge those embeddings with patient-level clinical variables from full_data.
4. Compare multiple sklearn models on:
   - deep_only
   - clinical_only
   - fusion (deep + clinical)

Usage:
  python scripts/compare_fusion_models.py \
    --results-json reports/cnn_v2_results_20260505_222801.json \
    --checkpoint reports/best_cnn_v2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_models import (
    FEATURE_META_COLS,
    _fit,
    binary_metrics,
    compute_severity_weights,
    model_configs,
)
from scripts.train_cnn_v2 import (
    DeeperCNN,
    MobileNetV3Small,
    SimpleCNN,
    TemperatureDataset,
    load_data as load_cnn_data,
)

warnings.filterwarnings("ignore")

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


def select_clinical_feature_columns(clinical: pd.DataFrame) -> list[str]:
    """Keep numeric clinical columns that are available at inference time."""
    cols: list[str] = []
    for col in clinical.columns:
        if col in CLINICAL_EXCLUDE_COLS:
            continue
        series = pd.to_numeric(clinical[col], errors="coerce")
        if not series.notna().any():
            continue
        cols.append(col)
    return cols


def build_feature_sets(deep_cols: list[str], clinical_cols: list[str]) -> dict[str, list[str]]:
    return {
        "deep_only": deep_cols,
        "clinical_only": clinical_cols,
        "fusion": deep_cols + clinical_cols,
    }


def load_results_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_from_results(results: dict) -> torch.nn.Module:
    model_name = results["model"]
    multi_task = bool(results.get("use_multi_task"))
    soft_label = bool(results.get("use_soft_label"))
    in_channels = 2 if results.get("use_region_attention") else 1
    common = dict(
        num_classes=2,
        dropout=float(results.get("dropout", 0.3)),
        in_channels=in_channels,
        img_size=int(results.get("target_size", 64)),
        multi_task=multi_task,
        soft_label=soft_label,
    )
    if model_name == "mobilenet":
        return MobileNetV3Small(pretrained=False, **common)
    if model_name == "deeper":
        return DeeperCNN(**common)
    return SimpleCNN(**common)


def extract_backbone_features(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(model, MobileNetV3Small):
        features = model.backbone(x)
        logits_cls = model.classifier_head(features)
        logits_sev = model.severity_head(features) if model.multi_task else None
        return features, logits_cls, logits_sev
    if isinstance(model, DeeperCNN):
        z = model.features(x)
        z = z.view(z.size(0), -1)
        features = model.shared(z)
        logits_cls = model.classifier_head(features)
        logits_sev = model.severity_head(features) if model.multi_task else None
        return features, logits_cls, logits_sev

    z = model.pool(F.relu(model.bn1(model.conv1(x))))
    z = model.pool(F.relu(model.bn2(model.conv2(z))))
    z = model.pool(F.relu(model.bn3(model.conv3(z))))
    z = z.view(z.size(0), -1)
    features = model.dropout(F.relu(model.fc1(z)))
    logits_cls = model.classifier_head(features)
    logits_sev = model.severity_head(features) if model.multi_task else None
    return features, logits_cls, logits_sev


@torch.no_grad()
def extract_deep_feature_frame(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sample_to_patient: dict[str, str],
    sample_to_year: dict[str, int],
) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    for x, y, sev, sample_ids in loader:
        x = x.to(device)
        features, logits_cls, logits_sev = extract_backbone_features(model, x)
        features_np = features.cpu().numpy()

        if logits_cls.ndim == 2 and logits_cls.shape[1] == 2:
            probs = F.softmax(logits_cls, dim=1)[:, 1].cpu().numpy()
        else:
            probs = torch.sigmoid(logits_cls.squeeze(-1)).cpu().numpy()
        sev_pred = logits_sev.squeeze(-1).cpu().numpy() if logits_sev is not None else np.full(len(sample_ids), np.nan)

        for i, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "patient_id": sample_to_patient[sample_id],
                "year": sample_to_year[sample_id],
                "label": int(y[i].item()),
                "stenosis_multiclass": float(sev[i].item()),
                "cnn_prob": float(probs[i]),
                "cnn_severity_pred": float(sev_pred[i]),
            }
            for j, value in enumerate(features_np[i]):
                row[f"deep_{j:03d}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def load_clinical_table(path: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path, dtype=str)
    for col in clinical.columns:
        if col in {"canonical_patient_id", "name", "gender", "data_sources", "icas_detail"}:
            continue
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
    return clinical


def build_fusion_frame(
    deep_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    clinical_cols: list[str],
) -> pd.DataFrame:
    merged = deep_df.merge(
        clinical_df[["canonical_patient_id", *clinical_cols]],
        left_on="patient_id",
        right_on="canonical_patient_id",
        how="left",
    )
    return merged.drop(columns=["canonical_patient_id"])


def apply_split_feature_set(df: pd.DataFrame, split: dict, feature_cols: list[str]) -> tuple:
    def arrays(ids: list[str]):
        sub = df[df["patient_id"].isin(ids)].dropna(subset=["label"])
        X = sub[feature_cols].values.astype(np.float32)
        y = sub["label"].values.astype(int)
        sev = sub["stenosis_multiclass"].values
        pids = sub["patient_id"].values
        return X, y, sev, pids

    X_tr, y_tr, sev_tr, pids_tr = arrays(split["train_patient_ids"])
    X_va, y_va, _, _ = arrays(split["val_patient_ids"])
    X_te, y_te, _, _ = arrays(split["test_patient_ids"])
    _, groups_tr = np.unique(pids_tr, return_inverse=True)
    return X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te


def score_split(pipeline, X, y, prefix: str) -> dict:
    y_pred = pipeline.predict(X)
    try:
        y_prob = pipeline.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None
    return binary_metrics(y, y_pred, y_prob, prefix)


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
) -> list[dict]:
    rows: list[dict] = []
    for strategy in ("standard", "severity_weighted"):
        if strategy == "severity_weighted" and not cfg["supports_sample_weight"]:
            continue
        pipe = cfg["pipeline"] if strategy == "standard" else cfg["pipeline_sw"]
        sw = None if strategy == "standard" else compute_severity_weights(y_tr, sev_tr)
        best, best_params, cv_auc = _fit(
            pipe,
            X_tr,
            y_tr,
            cfg["param_grid"],
            do_search,
            cv_folds,
            sample_weight=sw,
            groups=groups_tr,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True, help="Training results JSON describing the best CNN architecture")
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/best_cnn_v2.pt"), help="Path to the CNN checkpoint")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--clinical", type=Path, default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/data_split.json"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/annotations/annotations.json"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/annotations/masks"))
    parser.add_argument("--npy-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--feature-csv", type=Path, help="Optional path to save extracted deep+clinical features")
    parser.add_argument("--no-search", action="store_true", help="Skip GridSearchCV for a faster comparison")
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results_json(args.results_json)

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

    clinical_df = load_clinical_table(args.clinical)
    clinical_cols = select_clinical_feature_columns(clinical_df)
    fusion_df = build_fusion_frame(deep_df, clinical_df, clinical_cols)

    deep_cols = [c for c in fusion_df.columns if c.startswith("deep_")] + ["cnn_prob", "cnn_severity_pred"]
    feature_sets = build_feature_sets(deep_cols, clinical_cols)
    split = json.loads(args.split.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for feature_set_name, feature_cols in feature_sets.items():
        X_tr, y_tr, sev_tr, groups_tr, X_va, y_va, X_te, y_te = apply_split_feature_set(
            fusion_df, split, feature_cols
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
                )
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output.mkdir(parents=True, exist_ok=True)

    feature_csv = args.feature_csv or (args.output / f"deep_clinical_fusion_features_{timestamp}.csv")
    fusion_df.to_csv(feature_csv, index=False)

    compare_csv = args.output / f"fusion_model_comparison_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(compare_csv, index=False)

    print(f"Deep+clinical feature table saved to: {feature_csv}")
    print(f"Fusion comparison results saved to: {compare_csv}")


if __name__ == "__main__":
    main()
