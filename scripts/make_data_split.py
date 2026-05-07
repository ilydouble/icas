#!/usr/bin/env python3
"""Generate a patient-level stratified train/val/test split.

Splits are done at the **patient** level to prevent data leakage across samples
from the same patient.  Stratification uses the binary `label` column
(has_icas 0/1) so that each split has a similar positive rate.

Patients without a valid label are excluded from the split but listed separately
so downstream scripts can decide how to handle them.

Output: configs/data_split.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_patients(
    clinical_path: Path,
    features_path: Path,
) -> pd.DataFrame:
    """Return one row per patient with label, stenosis, and image_count."""
    clinical = pd.read_csv(clinical_path, dtype=str)
    feats = pd.read_csv(features_path, dtype=str)

    # Count samples per patient in the feature file
    sample_counts = feats.groupby("patient_id").size().rename("feature_sample_count")

    # Normalise types on the clinical columns we need
    clinical["label"] = pd.to_numeric(clinical["label"], errors="coerce")
    clinical["stenosis_multiclass"] = clinical["stenosis_multiclass"].where(
        clinical["stenosis_multiclass"] != "missing", other=None
    )

    df = clinical[["canonical_patient_id", "label", "stenosis_multiclass"]].copy()
    df = df.join(sample_counts, on="canonical_patient_id")
    return df


def stratified_patient_split(
    patients: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split patients into train / val / test, stratified by binary label.

    Returns (train_ids, val_ids, test_ids, unlabelled_ids).
    """
    labelled = patients.dropna(subset=["label"]).copy()
    unlabelled = patients[patients["label"].isna()]["canonical_patient_id"].tolist()

    labelled["label_int"] = labelled["label"].astype(int)

    # First split off test set (size = 1 - train_frac - val_frac)
    test_frac = 1.0 - train_frac - val_frac
    train_val_ids, test_ids = train_test_split(
        labelled["canonical_patient_id"].tolist(),
        test_size=test_frac,
        stratify=labelled["label_int"].tolist(),
        random_state=seed,
    )

    # From the remainder, split out val
    train_val_labels = (
        labelled[labelled["canonical_patient_id"].isin(train_val_ids)]
        .set_index("canonical_patient_id")["label_int"]
    )
    val_relative = val_frac / (train_frac + val_frac)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_relative,
        stratify=[int(train_val_labels[pid]) for pid in train_val_ids],
        random_state=seed,
    )

    return sorted(train_ids), sorted(val_ids), sorted(test_ids), sorted(unlabelled)


def split_summary(patients: pd.DataFrame, ids: list[str]) -> dict:
    sub = patients[patients["canonical_patient_id"].isin(ids)]
    n = len(sub)
    label_counts = sub["label"].value_counts(dropna=False).to_dict()
    # sample counts (images) per split
    samples = int(sub["feature_sample_count"].sum(skipna=True))
    pos = int(label_counts.get(1.0, 0))
    neg = int(label_counts.get(0.0, 0))
    return {
        "n_patients": n,
        "n_samples": samples,
        "n_positive": pos,
        "n_negative": neg,
        "positive_rate": round(pos / (pos + neg), 4) if (pos + neg) > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path,
                        default=Path("datasets/full_data/patient_clinical_data.csv"))
    parser.add_argument("--features", type=Path,
                        default=Path("datasets/temperature_features.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("configs/data_split.json"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patients = load_patients(args.clinical, args.features)
    print(f"Total patients: {len(patients)}")
    print(f"  labelled: {patients['label'].notna().sum()}")
    print(f"  unlabelled: {patients['label'].isna().sum()}")

    train_ids, val_ids, test_ids, unlabelled_ids = stratified_patient_split(
        patients,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    config = {
        "description": (
            "Patient-level stratified train/val/test split. "
            "Split by canonical_patient_id to prevent data leakage."
        ),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": round(1.0 - args.train_frac - args.val_frac, 4),
        },
        "sources": {
            "clinical": str(args.clinical),
            "features": str(args.features),
        },
        "summary": {
            "train": split_summary(patients, train_ids),
            "val": split_summary(patients, val_ids),
            "test": split_summary(patients, test_ids),
            "unlabelled": {
                "n_patients": len(unlabelled_ids),
            },
        },
        "train_patient_ids": train_ids,
        "val_patient_ids": val_ids,
        "test_patient_ids": test_ids,
        "unlabelled_patient_ids": unlabelled_ids,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Split Summary ===")
    for split_name in ("train", "val", "test"):
        s = config["summary"][split_name]
        print(
            f"  {split_name:5s}: {s['n_patients']:3d} patients | "
            f"{s['n_samples']:4d} samples | "
            f"pos_rate={s['positive_rate']:.3f}"
        )
    unlabelled_n = config["summary"]["unlabelled"]["n_patients"]
    print(f"  unlabelled: {unlabelled_n} patients (excluded from split)")
    print(f"\nConfig written to: {args.output}")


if __name__ == "__main__":
    main()
