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

    args = parser.parse_args()
    clinical_ids = load_clinical_ids(args.clinical_xlsx)

    if args.command == "analyze":
        analysis = analyze_sources(args.datasets, clinical_ids)
        write_report(analysis, args.report, args.issues, args.selected)
        print(f"Wrote report: {args.report}")
        print(f"Selected samples: {analysis.summary['selected_samples']}")
        print(f"Issues: {analysis.summary['issues']}")
    elif args.command == "build":
        result = build_dataset(args.datasets, args.output, clinical_ids)
        print(result)


if __name__ == "__main__":
    main()
