#!/usr/bin/env python3
"""Convert temperature CSV files to NPY binary format for fast loading.

CSV parsing is slow (text parsing, string-to-float conversion).
NPY is direct memory-mapped binary, ~100x faster to load.

Usage:
    python scripts/convert_csv_to_npy.py
    python scripts/convert_csv_to_npy.py --output-dir datasets/npy_temperature
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def parse_temperature_csv(path: Path) -> np.ndarray:
    """Parse temperature CSV into 2D matrix."""
    with path.open("r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    rows: list[list[float]] = []
    for line in lines[4:]:
        parts = line.rstrip("\r\n").split(",")
        if len(parts) <= 1 or parts[0] == "":
            continue
        values = [v for v in parts[1:] if v != ""]
        if not values:
            continue
        rows.append([float(v) for v in values])

    if not rows:
        raise ValueError(f"No temperature rows parsed from {path}")

    return np.asarray(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/full_data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/npy_temperature"))
    parser.add_argument("--excluded", type=Path, default=Path("configs/excluded_samples.json"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)

    excluded_ids: set[str] = set()
    if args.excluded.exists():
        data = json.loads(args.excluded.read_text(encoding="utf-8"))
        items = data.get("excluded_samples", [])
        for item in items:
            if isinstance(item, dict) and item.get("sample_id"):
                excluded_ids.add(str(item["sample_id"]))
            elif isinstance(item, str):
                excluded_ids.add(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting CSV to NPY...")
    print(f"  Input: {args.manifest}")
    print(f"  Output: {args.output_dir}")
    print(f"  Excluded: {len(excluded_ids)} samples")

    success = 0
    failed = 0
    skipped = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        sample_id = row["sample_id"]
        if sample_id in excluded_ids:
            skipped += 1
            continue

        csv_path = Path(row["temperature_path"])
        if not csv_path.is_absolute():
            csv_path = Path(".") / csv_path

        npy_path = args.output_dir / f"{sample_id}.npy"

        if npy_path.exists():
            skipped += 1
            continue

        try:
            temp_matrix = parse_temperature_csv(csv_path)
            np.save(npy_path, temp_matrix)
            success += 1
        except Exception as e:
            print(f"\n  Failed: {sample_id} - {e}")
            failed += 1

    print(f"\nDone!")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")

    sizes = {}
    for npy_file in args.output_dir.glob("*.npy"):
        arr = np.load(npy_file)
        key = f"{arr.shape[1]}x{arr.shape[0]}"
        sizes[key] = sizes.get(key, 0) + 1

    print(f"\nTemperature matrix sizes:")
    for k, v in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} samples")


if __name__ == "__main__":
    main()
