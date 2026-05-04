#!/usr/bin/env python3
"""Analyze raw thermal image data and build a paired training dataset."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PATIENT_ID_RE = re.compile(r"^(?:\d{3}[A-Z]{2}|[A-Z]{1,3}\d{3})")
VIEW_RE = re.compile(r"^(?P<raw_patient>.+?)-(?P<view>正|仰|左|右)-?(?P<sequence>\d+)$")
CHINESE_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,5}")


@dataclass(frozen=True)
class Sample:
    year: str
    patient_id: str
    view: str
    sequence: int
    image_path: Path
    temperature_path: Path
    clinical_available: bool = False

    @property
    def sample_id(self) -> str:
        return f"{self.year}_{self.patient_id}_{self.sequence}"


@dataclass(frozen=True)
class Issue:
    year: str
    patient_id: str
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class MatchInfo:
    canonical_patient_id: str
    matched_by: str
    matched_name: str = ""
    clinical_available: bool = False
    multimodal_available: bool = False
    notes: str = ""


@dataclass
class AnalysisResult:
    samples: list[Sample] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    selected_missing_clinical: list[str] = field(default_factory=list)
    clinical_missing_selected: list[str] = field(default_factory=list)

    @property
    def issues_by_patient(self) -> dict[str, list[Issue]]:
        grouped: dict[str, list[Issue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.patient_id, []).append(issue)
        return grouped


def parse_sequence_for_patient(stem: str, patient_id: str) -> int | None:
    compact = re.fullmatch(re.escape(patient_id) + r"-?(\d+)", stem)
    if not compact:
        return None
    return int(compact.group(1))


def parse_2025_stem(stem: str) -> tuple[str | None, str | None, int | None]:
    match = VIEW_RE.match(stem)
    if not match:
        return None, None, None

    raw_patient = match.group("raw_patient").strip()
    patient_match = PATIENT_ID_RE.match(raw_patient)
    patient_id = patient_match.group(0) if patient_match else None
    return patient_id, match.group("view"), int(match.group("sequence"))


def load_clinical_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading clinical .xlsx requires pandas/openpyxl.") from exc

    df = pd.read_excel(path, sheet_name=0, usecols=["编号"], dtype={"编号": str})
    return {str(value).strip().upper() for value in df["编号"].dropna() if str(value).strip()}


def load_clinical_lookup(path: Path | None) -> tuple[set[str], dict[str, list[str]]]:
    if not path:
        return set(), {}
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading clinical .xlsx requires pandas/openpyxl.") from exc

    df = pd.read_excel(path, sheet_name=0, usecols=["编号", "姓名"], dtype=str).fillna("")
    clinical_ids = {_normalize_id(value) for value in df["编号"] if _normalize_id(value)}
    name_to_ids: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        patient_id = _normalize_id(row["编号"])
        name = _normalize_name(row["姓名"])
        if patient_id and name:
            name_to_ids.setdefault(name, []).append(patient_id)
    return clinical_ids, {name: sorted(set(ids)) for name, ids in name_to_ids.items()}


def load_multimodal_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "patient_id" not in reader.fieldnames:
            return set()
        return {_normalize_id(row.get("patient_id", "")) for row in reader if _normalize_id(row.get("patient_id", ""))}


def analyze_sources(datasets_dir: Path, clinical_ids: set[str] | None = None) -> AnalysisResult:
    clinical_ids = clinical_ids or set()
    result = AnalysisResult()
    result.samples.extend(_analyze_2024(datasets_dir / "2024", clinical_ids, result.issues))
    result.samples.extend(_analyze_2025(datasets_dir / "2025", clinical_ids, result.issues))

    selected_patients = {sample.patient_id for sample in result.samples}
    result.selected_missing_clinical = sorted(selected_patients - clinical_ids) if clinical_ids else []
    result.clinical_missing_selected = sorted(clinical_ids - selected_patients) if clinical_ids else []
    result.summary = {
        "selected_samples": len(result.samples),
        "selected_patients": len(selected_patients),
        "issues": len(result.issues),
        "clinical_rows": len(clinical_ids),
        "selected_patients_missing_clinical": len(result.selected_missing_clinical),
        "clinical_patients_missing_selected_sample": len(result.clinical_missing_selected),
    }
    return result


def _analyze_2024(year_dir: Path, clinical_ids: set[str], issues: list[Issue]) -> list[Sample]:
    samples: list[Sample] = []
    if not year_dir.exists():
        issues.append(Issue("2024", "", "missing_year_dir", "datasets/2024 does not exist.", str(year_dir)))
        return samples

    for patient_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
        patient_id = patient_dir.name.strip().upper()
        seq1_images = []
        for image_path in sorted(_iter_files(patient_dir, IMAGE_EXTENSIONS)):
            sequence = parse_sequence_for_patient(image_path.stem.upper(), patient_id)
            if sequence == 1:
                seq1_images.append(image_path)

        if not seq1_images:
            issues.append(Issue("2024", patient_id, "missing_sequence_1_image", "No first/front image was found.", str(patient_dir)))
            continue
        if len(seq1_images) > 1:
            issues.append(Issue("2024", patient_id, "multiple_sequence_1_images", "Multiple sequence 1 images found; first sorted file is used.", ";".join(str(p) for p in seq1_images)))

        image_path = seq1_images[0]
        temperature_path = image_path.with_suffix(".csv")
        if not temperature_path.exists():
            issues.append(Issue("2024", patient_id, "missing_temperature_csv", "Sequence 1 image has no matching CSV.", str(image_path)))
            continue

        samples.append(Sample("2024", patient_id, "正", 1, image_path, temperature_path, patient_id in clinical_ids))

    return samples


def _analyze_2025(year_dir: Path, clinical_ids: set[str], issues: list[Issue]) -> list[Sample]:
    samples: list[Sample] = []
    photo_dir = year_dir / "热成像照片"
    temperature_dir = year_dir / "热成像温度数据"
    if not photo_dir.exists():
        issues.append(Issue("2025", "", "missing_photo_dir", "2025 photo directory does not exist.", str(photo_dir)))
        return samples
    if not temperature_dir.exists():
        issues.append(Issue("2025", "", "missing_temperature_dir", "2025 temperature directory does not exist.", str(temperature_dir)))
        return samples

    images_by_patient: dict[str, list[tuple[Path, str, int]]] = {}
    seen_image_stems: set[str] = set()

    for image_path in sorted(_iter_files(photo_dir, IMAGE_EXTENSIONS)):
        seen_image_stems.add(image_path.stem)
        patient_id, view, sequence = parse_2025_stem(image_path.stem.upper())
        if patient_id is None or view is None or sequence is None:
            issues.append(Issue("2025", "", "unparseable_image_name", "Cannot parse patient id/view/sequence from image name.", str(image_path)))
            continue
        images_by_patient.setdefault(patient_id, []).append((image_path, view, sequence))

    for patient_id, entries in sorted(images_by_patient.items()):
        front_entries = [(path, sequence) for path, view, sequence in entries if view == "正"]
        if not front_entries:
            issues.append(Issue("2025", patient_id, "missing_front_image", "No front image was found for this patient.", ""))
            continue
        if not any(sequence == 1 for _, sequence in front_entries):
            issues.append(Issue("2025", patient_id, "missing_front_sequence_1", "Front images exist, but no sequence 1 front image was found.", ";".join(str(path) for path, _ in front_entries)))
            continue

        for image_path, sequence in sorted(front_entries, key=lambda item: (item[1], str(item[0]))):
            temperature_path = temperature_dir / f"{image_path.stem}.csv"
            if not temperature_path.exists():
                issues.append(Issue("2025", patient_id, "missing_temperature_csv", "Front image has no matching temperature CSV.", str(image_path)))
                continue
            samples.append(Sample("2025", patient_id, "正", sequence, image_path, temperature_path, patient_id in clinical_ids))

    for temperature_path in sorted(temperature_dir.glob("*.csv")):
        if temperature_path.stem not in seen_image_stems:
            patient_id, _, _ = parse_2025_stem(temperature_path.stem.upper())
            issues.append(Issue("2025", patient_id or "", "orphan_temperature_csv", "Temperature CSV has no matching image file.", str(temperature_path)))

    return samples


def build_dataset(datasets_dir: Path, output_dir: Path, clinical_ids: set[str] | None = None) -> dict[str, int]:
    analysis = analyze_sources(datasets_dir, clinical_ids)
    image_out = output_dir / "images"
    temperature_out = output_dir / "temperature"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_out.mkdir(parents=True, exist_ok=True)
    temperature_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_existing = 0
    manifest_rows = []
    for sample in analysis.samples:
        image_dest = image_out / f"{sample.sample_id}{sample.image_path.suffix.lower()}"
        temp_dest = temperature_out / f"{sample.sample_id}.csv"
        image_copied = _copy_once(sample.image_path, image_dest)
        temp_copied = _copy_once(sample.temperature_path, temp_dest)
        copied += image_copied + temp_copied
        skipped_existing += int(image_copied == 0) + int(temp_copied == 0)
        manifest_rows.append(_sample_row(sample, image_dest, temp_dest))

    _write_manifest(output_dir / "manifest.csv", manifest_rows)
    _write_issues(output_dir / "extraction_log.csv", analysis.issues)
    return {"samples": len(analysis.samples), "copied": copied, "skipped_existing": skipped_existing, "issues": len(analysis.issues)}


def build_full_dataset(
    datasets_dir: Path,
    output_dir: Path,
    clinical_ids: set[str] | None = None,
    multimodal_ids: set[str] | None = None,
    name_to_clinical_ids: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    clinical_ids = clinical_ids or set()
    multimodal_ids = multimodal_ids or set()
    name_to_clinical_ids = name_to_clinical_ids or {}
    all_known_ids = clinical_ids | multimodal_ids
    analysis = analyze_sources(datasets_dir, clinical_ids)

    image_out = output_dir / "images"
    temperature_out = output_dir / "temperature"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_out.mkdir(parents=True, exist_ok=True)
    temperature_out.mkdir(parents=True, exist_ok=True)

    included_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    copied = 0
    skipped_existing = 0

    for sample in analysis.samples:
        match = _match_sample(sample, all_known_ids, clinical_ids, multimodal_ids, name_to_clinical_ids)
        if match is None:
            excluded_rows.append(_excluded_row(sample, "no_patient_info_match", "No patient info found by patient_id or extracted name."))
            continue

        sample_stem = f"{sample.year}_{match.canonical_patient_id}_{sample.sequence}"
        image_dest = image_out / f"{sample_stem}{sample.image_path.suffix.lower()}"
        temp_dest = temperature_out / f"{sample_stem}.csv"
        image_copied = _copy_once(sample.image_path, image_dest)
        temp_copied = _copy_once(sample.temperature_path, temp_dest)
        copied += image_copied + temp_copied
        skipped_existing += int(image_copied == 0) + int(temp_copied == 0)
        included_rows.append(_full_sample_row(sample, match, image_dest, temp_dest))

    patient_rows = _patient_summary_rows(included_rows)
    _write_csv(output_dir / "manifest.csv", _full_manifest_fields(), included_rows)
    _write_csv(output_dir / "excluded_samples.csv", _excluded_fields(), excluded_rows)
    _write_csv(output_dir / "patient_summary.csv", _patient_summary_fields(), patient_rows)
    _write_issues(output_dir / "source_issues.csv", analysis.issues)
    _write_full_report(output_dir / "analysis_report.md", included_rows, excluded_rows, patient_rows, analysis.issues)

    return {
        "included_samples": len(included_rows),
        "included_patients": len(patient_rows),
        "excluded_samples": len(excluded_rows),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "source_issues": len(analysis.issues),
    }


def write_report(analysis: AnalysisResult, report_path: Path, issues_path: Path | None = None, selected_path: Path | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    issue_counts: dict[str, int] = {}
    for issue in analysis.issues:
        issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
    year_counts: dict[str, int] = {}
    for sample in analysis.samples:
        year_counts[sample.year] = year_counts.get(sample.year, 0) + 1

    lines = [
        "# 数据完整性分析报告",
        "",
        "## 汇总",
        "",
        f"- 可提取样本数: {analysis.summary.get('selected_samples', 0)}",
        f"- 可提取患者数: {analysis.summary.get('selected_patients', 0)}",
        f"- 需人工核对问题数: {analysis.summary.get('issues', 0)}",
        f"- 临床特征表患者数: {analysis.summary.get('clinical_rows', 0)}",
        f"- 可提取患者中缺少临床特征: {analysis.summary.get('selected_patients_missing_clinical', 0)}",
        f"- 临床特征表中缺少可提取样本: {analysis.summary.get('clinical_patients_missing_selected_sample', 0)}",
        "",
        "## 可提取样本按年份",
        "",
    ]
    for year, count in sorted(year_counts.items()):
        lines.append(f"- {year}: {count}")
    lines.extend([
        "",
        "## 问题类型统计",
        "",
    ])
    for code, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {code}: {count}")
    if not issue_counts:
        lines.append("- None")
    lines.extend([
        "",
        "## 临床特征覆盖",
        "",
        f"- 可提取但不在临床表中的患者: {', '.join(analysis.selected_missing_clinical) if analysis.selected_missing_clinical else '无'}",
        f"- 在临床表中但没有可提取样本的患者: {', '.join(analysis.clinical_missing_selected) if analysis.clinical_missing_selected else '无'}",
        "",
        "## 提取规则",
        "",
        "- 2024: 使用每个患者文件夹中的 1 号图像（支持 `ID1` 和 `ID-1`）及同名 CSV。",
        "- 2025: 使用所有正面（`正`）JPG/JPEG/PNG 图像，以及 `热成像温度数据` 下同名 CSV。",
        "- 2025 患者如果没有正面图像，或有正面但没有 1 号正面图像，先只写入日志，留待人工判定。",
        "- 缺少图像/温度配对、无法解析患者编号的文件均写入问题明细 CSV。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if issues_path:
        _write_issues(issues_path, analysis.issues)
    if selected_path:
        _write_manifest(selected_path, [_sample_row(sample, sample.image_path, sample.temperature_path) for sample in analysis.samples])


def _normalize_id(value: object) -> str:
    return str(value).strip().upper()


def _normalize_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def _clean_name_candidate(text: str) -> str:
    text = re.sub(r"[（(]有重名[）)]", "", str(text))
    text = re.sub(r"\s+", "", text)
    if "+" in text:
        text = re.sub(r"^.*?\+\d*", "", text)
    text = re.sub(r"^\d+", "", text)
    text = re.sub(r"^[A-Z]{1,3}\d{3}", "", text, flags=re.IGNORECASE)
    text = text.strip("_-")
    matches = CHINESE_NAME_RE.findall(text)
    return matches[-1] if matches else ""


def _sample_candidate_name(sample: Sample) -> str:
    patient_name = _clean_name_candidate(sample.patient_id)
    if patient_name:
        return patient_name
    stem = re.sub(r"-(正|仰|左|右)-?\d+$", "", sample.image_path.stem)
    stem = re.sub(r"(正|仰|左|右)-?\d+$", "", stem)
    return _clean_name_candidate(stem)


def _match_sample(
    sample: Sample,
    all_known_ids: set[str],
    clinical_ids: set[str],
    multimodal_ids: set[str],
    name_to_clinical_ids: dict[str, list[str]],
) -> MatchInfo | None:
    patient_id = _normalize_id(sample.patient_id)
    if patient_id in all_known_ids:
        return MatchInfo(
            canonical_patient_id=patient_id,
            matched_by="patient_id",
            clinical_available=patient_id in clinical_ids,
            multimodal_available=patient_id in multimodal_ids,
        )

    candidate_name = _sample_candidate_name(sample)
    candidate_ids = name_to_clinical_ids.get(candidate_name, []) if candidate_name else []
    if len(candidate_ids) == 1:
        canonical_id = candidate_ids[0]
        return MatchInfo(
            canonical_patient_id=canonical_id,
            matched_by="name",
            matched_name=candidate_name,
            clinical_available=canonical_id in clinical_ids,
            multimodal_available=canonical_id in multimodal_ids,
        )
    if len(candidate_ids) > 1:
        return None
    return None


def _iter_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _copy_once(src: Path, dest: Path) -> int:
    if dest.exists():
        return 0
    shutil.copy2(src, dest)
    return 1


def _sample_row(sample: Sample, image_path: Path, temperature_path: Path) -> dict[str, str]:
    return {
        "sample_id": sample.sample_id,
        "year": sample.year,
        "patient_id": sample.patient_id,
        "view": sample.view,
        "sequence": str(sample.sequence),
        "image_path": str(image_path),
        "temperature_path": str(temperature_path),
        "clinical_available": "1" if sample.clinical_available else "0",
        "source_image_path": str(sample.image_path),
        "source_temperature_path": str(sample.temperature_path),
    }


def _full_sample_row(sample: Sample, match: MatchInfo, image_path: Path, temperature_path: Path) -> dict[str, str]:
    return {
        "sample_id": f"{sample.year}_{match.canonical_patient_id}_{sample.sequence}",
        "canonical_patient_id": match.canonical_patient_id,
        "source_patient_id": sample.patient_id,
        "year": sample.year,
        "view": sample.view,
        "sequence": str(sample.sequence),
        "matched_by": match.matched_by,
        "matched_name": match.matched_name,
        "clinical_available": "1" if match.clinical_available else "0",
        "multimodal_available": "1" if match.multimodal_available else "0",
        "image_path": str(image_path),
        "temperature_path": str(temperature_path),
        "source_image_path": str(sample.image_path),
        "source_temperature_path": str(sample.temperature_path),
    }


def _excluded_row(sample: Sample, reason_code: str, reason: str) -> dict[str, str]:
    candidate_name = _sample_candidate_name(sample)
    return {
        "patient_id": sample.patient_id,
        "candidate_name": candidate_name,
        "year": sample.year,
        "view": sample.view,
        "sequence": str(sample.sequence),
        "reason_code": reason_code,
        "reason": reason,
        "source_image_path": str(sample.image_path),
        "source_temperature_path": str(sample.temperature_path),
    }


def _full_manifest_fields() -> list[str]:
    return [
        "sample_id",
        "canonical_patient_id",
        "source_patient_id",
        "year",
        "view",
        "sequence",
        "matched_by",
        "matched_name",
        "clinical_available",
        "multimodal_available",
        "image_path",
        "temperature_path",
        "source_image_path",
        "source_temperature_path",
    ]


def _excluded_fields() -> list[str]:
    return ["patient_id", "candidate_name", "year", "view", "sequence", "reason_code", "reason", "source_image_path", "source_temperature_path"]


def _patient_summary_fields() -> list[str]:
    return ["canonical_patient_id", "image_count", "images_2024", "images_2025", "matched_by_values", "source_patient_ids"]


def _patient_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["canonical_patient_id"], []).append(row)

    summary_rows = []
    for patient_id, patient_rows in sorted(grouped.items()):
        images_2024 = sum(1 for row in patient_rows if row["year"] == "2024")
        images_2025 = sum(1 for row in patient_rows if row["year"] == "2025")
        summary_rows.append({
            "canonical_patient_id": patient_id,
            "image_count": str(len(patient_rows)),
            "images_2024": str(images_2024),
            "images_2025": str(images_2025),
            "matched_by_values": ";".join(sorted({row["matched_by"] for row in patient_rows})),
            "source_patient_ids": ";".join(sorted({row["source_patient_id"] for row in patient_rows})),
        })
    return summary_rows


def _write_full_report(
    path: Path,
    included_rows: list[dict[str, str]],
    excluded_rows: list[dict[str, str]],
    patient_rows: list[dict[str, str]],
    source_issues: list[Issue],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_2024 = sum(1 for row in included_rows if row["year"] == "2024")
    total_2025 = sum(1 for row in included_rows if row["year"] == "2025")
    matched_by_counts: dict[str, int] = {}
    for row in included_rows:
        matched_by_counts[row["matched_by"]] = matched_by_counts.get(row["matched_by"], 0) + 1
    excluded_counts: dict[str, int] = {}
    for row in excluded_rows:
        excluded_counts[row["reason_code"]] = excluded_counts.get(row["reason_code"], 0) + 1
    image_count_distribution: dict[str, int] = {}
    year_pattern_distribution: dict[str, int] = {}
    for row in patient_rows:
        image_count_distribution[row["image_count"]] = image_count_distribution.get(row["image_count"], 0) + 1
        pattern = f"2024={row['images_2024']},2025={row['images_2025']}"
        year_pattern_distribution[pattern] = year_pattern_distribution.get(pattern, 0) + 1

    lines = [
        "# full_data 数据集分析报告",
        "",
        "## 汇总",
        "",
        f"- 纳入患者数: {len(patient_rows)}",
        f"- 纳入图像样本数: {len(included_rows)}",
        f"- 2024 图像样本数: {total_2024}",
        f"- 2025 图像样本数: {total_2025}",
        f"- 排除可提取样本数: {len(excluded_rows)}",
        f"- 原始数据源问题日志数: {len(source_issues)}",
        "",
        "## 纳入规则",
        "",
        "- 先按患者编号匹配临床特征表或 `patient_level_multimodal_data.csv`。",
        "- 编号匹配不到时，从患者目录名或文件名抽取中文姓名；若姓名在临床表中唯一匹配，则纳入并使用临床表编号作为 canonical_patient_id。",
        "- 编号和姓名都无法匹配，或姓名匹配存在歧义，则不纳入训练数据，并写入 `excluded_samples.csv`。",
        "",
        "## 纳入方式分布",
        "",
    ]
    for key, count in sorted(matched_by_counts.items()):
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## 排除原因", ""])
    if excluded_counts:
        for key, count in sorted(excluded_counts.items()):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- 无")
    excluded_patient_reasons: dict[str, set[str]] = {}
    for row in excluded_rows:
        excluded_patient_reasons.setdefault(row["patient_id"], set()).add(row["reason_code"])
    lines.extend(["", "## 未纳入患者", ""])
    if excluded_patient_reasons:
        for patient_id, reasons in sorted(excluded_patient_reasons.items()):
            patient_excluded_rows = [row for row in excluded_rows if row["patient_id"] == patient_id]
            years = ",".join(sorted({row["year"] for row in patient_excluded_rows}))
            sequences = ",".join(row["sequence"] for row in patient_excluded_rows)
            candidate_names = sorted({row["candidate_name"] for row in patient_excluded_rows if row["candidate_name"]})
            name_text = f"，候选姓名: {';'.join(candidate_names)}" if candidate_names else ""
            lines.append(f"- {patient_id}: {len(patient_excluded_rows)} 张未纳入，年份: {years}，序号: {sequences}，原因: {';'.join(sorted(reasons))}{name_text}")
    else:
        lines.append("- 无")
    lines.extend(["", "## 每位患者图像数量分布", ""])
    for image_count, patient_count in sorted(image_count_distribution.items(), key=lambda item: int(item[0])):
        lines.append(f"- {image_count} 张: {patient_count} 位患者")
    lines.extend(["", "## 2024/2025 图像组合分布", ""])
    for pattern, patient_count in sorted(year_pattern_distribution.items()):
        lines.append(f"- {pattern}: {patient_count} 位患者")
    lines.extend(["", "## 输出文件", ""])
    lines.extend([
        "- `manifest.csv`: 纳入样本明细。",
        "- `patient_summary.csv`: 患者级图像数量统计，包含每位患者总图像数、2024 图像数、2025 图像数。",
        "- `excluded_samples.csv`: 可提取但未纳入的样本及原因。",
        "- `source_issues.csv`: 原始数据配对/命名问题日志。",
        "- `images/`: 复制后的图像文件。",
        "- `temperature/`: 复制后的温度 CSV 文件。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["sample_id", "year", "patient_id", "view", "sequence", "image_path", "temperature_path", "clinical_available", "source_image_path", "source_temperature_path"]
    _write_csv(path, fieldnames, rows)


def _write_issues(path: Path, issues: list[Issue]) -> None:
    rows = [issue.__dict__ for issue in issues]
    _write_csv(path, ["year", "patient_id", "code", "message", "path"], rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze source data completeness.")
    analyze_parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    analyze_parser.add_argument("--clinical-xlsx", type=Path)
    analyze_parser.add_argument("--report", type=Path, default=Path("reports/dataset_integrity_report.md"))
    analyze_parser.add_argument("--issues", type=Path, default=Path("reports/dataset_integrity_issues.csv"))
    analyze_parser.add_argument("--selected", type=Path, default=Path("reports/dataset_integrity_selected_samples.csv"))

    build_parser = subparsers.add_parser("build", help="Build paired dataset without recopying existing files.")
    build_parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    build_parser.add_argument("--clinical-xlsx", type=Path)
    build_parser.add_argument("--output", type=Path, default=Path("processed/icas_dataset"))

    full_parser = subparsers.add_parser("build-full", help="Build the maximally matched full_data dataset.")
    full_parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    full_parser.add_argument("--clinical-xlsx", type=Path, default=Path("datasets/临床特征核对后最终版.xlsx"))
    full_parser.add_argument("--multimodal-csv", type=Path, default=Path("datasets/patient_level_multimodal_data.csv"))
    full_parser.add_argument("--output", type=Path, default=Path("datasets/full_data"))

    args = parser.parse_args()
    if args.command == "analyze":
        clinical_ids = load_clinical_ids(args.clinical_xlsx)
        analysis = analyze_sources(args.datasets, clinical_ids)
        write_report(analysis, args.report, args.issues, args.selected)
        print(f"Wrote report: {args.report}")
        print(f"Selected samples: {analysis.summary['selected_samples']}")
        print(f"Issues: {analysis.summary['issues']}")
    elif args.command == "build":
        clinical_ids = load_clinical_ids(args.clinical_xlsx)
        result = build_dataset(args.datasets, args.output, clinical_ids)
        print(result)
    elif args.command == "build-full":
        clinical_ids, name_to_clinical_ids = load_clinical_lookup(args.clinical_xlsx)
        multimodal_ids = load_multimodal_ids(args.multimodal_csv)
        result = build_full_dataset(args.datasets, args.output, clinical_ids, multimodal_ids, name_to_clinical_ids)
        print(result)


if __name__ == "__main__":
    main()
