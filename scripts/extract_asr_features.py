#!/usr/bin/env python3
"""Extract interpretable ASR features and merge them with patient clinical data.

The ASR folder contains one JSON file per patient. This script converts each JSON
into a patient-level feature row and joins it with `datasets/full_data/
patient_clinical_data.csv`, so the resulting CSV can be used directly for:

1. multi-task learning with clinical supervision;
2. feature correlation analysis;
3. speech-related feature screening.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


REFERENCE_TEXT = (
    "啊。阿姨。一零一二三四五六七八九十。跑步，说话，唱歌，微笑，鼓掌。"
    "画图，数数，流星，书本，我出去买菜。"
    "我去地里摘玫瑰，今天的菜卖完了。"
    "住在附近的李红带来了新鲜的蘑菇，明明坐上了刚到的货车，手放不进办公室里的桌子里。"
    "北风和太阳争论谁的本领大。"
    "他们看见有一个人。"
    "身上穿着一件厚衣服。"
    "谁能叫他把衣服脱下来。"
    "就算谁有本领。"
    "北风拼命地吹啊吹啊。"
    "他吹得越厉害。"
    "那个人把衣服裹得越紧。"
    "太阳出来一晒。"
    "那个人马上把衣服脱下来了。"
    "所以北风不得不承认。"
    "太阳比他有本领。"
)

FILLER_CHARS = set("啊嗯呃哦")
PUNCT_PATTERN = re.compile(r"[，。！？；：、,.!?;:\s]+")
NORMALIZE_PATTERN = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def normalize_patient_id_from_filename(filename: str) -> str:
    """Extract the canonical patient id from an ASR JSON filename."""
    stem = Path(filename).stem
    return stem.split("_", 1)[0].strip()


def normalize_text_for_comparison(text: str) -> str:
    """Keep only CJK, alnum chars for robust transcript comparison."""
    return NORMALIZE_PATTERN.sub("", text or "")


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    mean_val = _safe_mean(values)
    return float(math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values)))


def _safe_min(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(min(values))


def _safe_max(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(max(values))


def _safe_median(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _coerce_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def compute_reference_deviation_features(text: str) -> dict[str, float]:
    """Compare transcript against the fixed reading prompt as a rough error proxy."""
    observed = normalize_text_for_comparison(text)
    reference = normalize_text_for_comparison(REFERENCE_TEXT)
    distance = _edit_distance(observed, reference)
    ref_len = max(len(reference), 1)
    obs_len = max(len(observed), 1)
    return {
        "asr_reference_length": float(len(reference)),
        "asr_observed_length": float(len(observed)),
        "asr_reference_edit_distance": float(distance),
        "asr_reference_char_error_rate": float(distance / ref_len),
        "asr_reference_length_ratio": float(len(observed) / ref_len),
        "asr_reference_sequence_ratio": float(SequenceMatcher(None, observed, reference).ratio()),
        "asr_reference_deletion_proxy": float(max(len(reference) - len(observed), 0) / ref_len),
        "asr_reference_insertion_proxy": float(max(len(observed) - len(reference), 0) / obs_len),
    }


def compute_sentence_level_features(detailed_results: list[dict]) -> dict[str, float]:
    """Aggregate sentence timing/rate statistics from the ASR JSON."""
    sentences = [item for item in detailed_results if isinstance(item, dict)]
    durations = [_coerce_float(item.get("duration")) for item in sentences]
    durations = [v for v in durations if v is not None]

    speech_rates: list[float] = []
    silence_durations: list[float] = []
    emotion_values: list[float] = []
    chars_per_sentence: list[float] = []

    pause_sentence_count = 0
    long_pause_sentence_count = 0

    for item in sentences:
        sentence_text = str(item.get("text") or "")
        normalized_sentence = normalize_text_for_comparison(sentence_text)
        if normalized_sentence:
            chars_per_sentence.append(float(len(normalized_sentence)))

        raw = item.get("raw_sentence") or {}
        rate = _coerce_float(raw.get("SpeechRate"))
        silence = _coerce_float(raw.get("SilenceDuration"))
        emotion = _coerce_float(raw.get("EmotionValue"))

        if rate is not None:
            speech_rates.append(rate)
        if silence is not None:
            silence_durations.append(silence)
            if silence > 0:
                pause_sentence_count += 1
            if silence >= 5:
                long_pause_sentence_count += 1
        if emotion is not None:
            emotion_values.append(emotion)

    total_duration_ms = sum(durations)
    total_silence_ms = sum(silence_durations)

    return {
        "asr_sentence_count": float(len(sentences)),
        "asr_sentence_duration_ms_total": float(total_duration_ms),
        "asr_sentence_duration_ms_mean": _safe_mean(durations),
        "asr_sentence_duration_ms_median": _safe_median(durations),
        "asr_sentence_duration_ms_std": _safe_std(durations),
        "asr_sentence_duration_ms_min": _safe_min(durations),
        "asr_sentence_duration_ms_max": _safe_max(durations),
        "asr_speech_rate_mean": _safe_mean(speech_rates),
        "asr_speech_rate_median": _safe_median(speech_rates),
        "asr_speech_rate_std": _safe_std(speech_rates),
        "asr_speech_rate_min": _safe_min(speech_rates),
        "asr_speech_rate_max": _safe_max(speech_rates),
        "asr_silence_duration_ms_total": float(total_silence_ms),
        "asr_silence_duration_ms_mean": _safe_mean(silence_durations),
        "asr_silence_duration_ms_max": _safe_max(silence_durations),
        "asr_pause_sentence_count": float(pause_sentence_count),
        "asr_long_pause_sentence_count": float(long_pause_sentence_count),
        "asr_pause_sentence_ratio": float(pause_sentence_count / len(sentences)) if sentences else float("nan"),
        "asr_long_pause_sentence_ratio": (
            float(long_pause_sentence_count / len(sentences)) if sentences else float("nan")
        ),
        "asr_silence_to_duration_ratio": (
            float(total_silence_ms / total_duration_ms) if total_duration_ms else float("nan")
        ),
        "asr_emotion_mean": _safe_mean(emotion_values),
        "asr_emotion_median": _safe_median(emotion_values),
        "asr_emotion_std": _safe_std(emotion_values),
        "asr_chars_per_sentence_mean": _safe_mean(chars_per_sentence),
        "asr_chars_per_sentence_std": _safe_std(chars_per_sentence),
    }


def compute_transcript_features(text: str, sentence_count: float, total_duration_ms: float) -> dict[str, float]:
    """Compute transcript-level features reflecting articulation/fluency proxies."""
    normalized = normalize_text_for_comparison(text)
    punct_tokens = [token for token in PUNCT_PATTERN.split(text or "") if token]
    filler_count = sum(1 for ch in normalized if ch in FILLER_CHARS)
    unique_chars = len(set(normalized))
    repeated_adjacent = sum(1 for i in range(1, len(normalized)) if normalized[i] == normalized[i - 1])
    total_duration_s = total_duration_ms / 1000.0 if total_duration_ms else float("nan")

    return {
        "asr_transcript_char_count": float(len(text or "")),
        "asr_normalized_char_count": float(len(normalized)),
        "asr_unique_char_count": float(unique_chars),
        "asr_unique_char_ratio": float(unique_chars / len(normalized)) if normalized else float("nan"),
        "asr_token_count_by_punctuation": float(len(punct_tokens)),
        "asr_filler_char_count": float(filler_count),
        "asr_filler_char_ratio": float(filler_count / len(normalized)) if normalized else float("nan"),
        "asr_repeated_adjacent_char_count": float(repeated_adjacent),
        "asr_repeated_adjacent_char_ratio": (
            float(repeated_adjacent / len(normalized)) if normalized else float("nan")
        ),
        "asr_chars_per_second": (
            float(len(normalized) / total_duration_s)
            if total_duration_s and not math.isnan(total_duration_s) and total_duration_s > 0
            else float("nan")
        ),
        "asr_chars_per_sentence": (
            float(len(normalized) / sentence_count) if sentence_count and not math.isnan(sentence_count) else float("nan")
        ),
    }


def _build_clinical_lookup(clinical_df: pd.DataFrame) -> dict[str, dict]:
    rows = clinical_df.to_dict(orient="records")
    return {str(row["canonical_patient_id"]): row for row in rows}


def extract_asr_row(asr_path: Path, clinical_df: pd.DataFrame) -> dict[str, object]:
    """Build a merged row from one ASR JSON and the clinical table."""
    with asr_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    clinical_lookup = _build_clinical_lookup(clinical_df)
    canonical_patient_id = normalize_patient_id_from_filename(asr_path.name)
    clinical_row = clinical_lookup.get(canonical_patient_id, {})

    text = str(payload.get("text") or "")
    detailed_results = payload.get("detailed_results") or []

    sentence_features = compute_sentence_level_features(detailed_results)
    transcript_features = compute_transcript_features(
        text=text,
        sentence_count=sentence_features["asr_sentence_count"],
        total_duration_ms=sentence_features["asr_sentence_duration_ms_total"],
    )
    reference_features = compute_reference_deviation_features(text)

    row: dict[str, object] = {
        "canonical_patient_id": canonical_patient_id,
        "asr_file": asr_path.name,
        "asr_source_file": payload.get("source_file"),
        "clinical_match_status": "matched" if clinical_row else "unmatched",
        "asr_transcript": text,
    }
    row.update(sentence_features)
    row.update(transcript_features)
    row.update(reference_features)

    for key, value in clinical_row.items():
        if key == "canonical_patient_id":
            continue
        row[key] = value

    return row


def write_feature_csv(asr_dir: Path, clinical_csv: Path, output_csv: Path) -> tuple[int, int]:
    """Extract all ASR rows, merge with clinical info, and write CSV."""
    clinical_df = pd.read_csv(clinical_csv)
    rows = [
        extract_asr_row(path, clinical_df)
        for path in sorted(asr_dir.glob("*.json"))
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    matched = sum(1 for row in rows if row["clinical_match_status"] == "matched")
    unmatched = len(rows) - matched
    return matched, unmatched


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=repo_root / "datasets/ASR/json_results",
        help="Directory containing one ASR JSON per patient.",
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=repo_root / "datasets/full_data/patient_clinical_data.csv",
        help="Clinical supervision table keyed by canonical_patient_id.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "datasets/asr_2025_features.csv",
        help="Output CSV path for merged ASR + clinical features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matched, unmatched = write_feature_csv(
        asr_dir=Path(args.asr_dir),
        clinical_csv=Path(args.clinical_csv),
        output_csv=Path(args.output_csv),
    )
    total = matched + unmatched
    print(f"ASR rows written: {total}")
    print(f"Matched clinical rows: {matched}")
    print(f"Unmatched clinical rows: {unmatched}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
