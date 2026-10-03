from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .schema import FeatureSchema
from .shandong_external import ShandongExternalJSONReader, _number, _text


MAIN_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
DOCUMENT_RELATIONSHIP_ID = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
)
SUMMARY_SHEET = "基线数据"
EXCLUSION_COLORS = {
    "FFFFFF00": "yellow",
    "FF92D050": "green",
}
REQUIRED_HEADERS = {
    "治疗前后间隔时间",
    "patient_SN",
    "年龄",
    "性别",
    "体重（kg）",
    "烟龄",
    "酒龄",
    "肿瘤病理类型",
    "基线时间",
    "第一次PETCT影像时间",
    "第二次PETCT影像时间",
    "第一次TBR",
    "第二次TBR",
    "第一次agatston评分",
    "第二次agatston评分",
}
FEATURE_SOURCE = {
    "BMI": "bmi",
    "血压": "systolic_bp",
    "葡萄糖": "glucose",
    "胆固醇": "total_cholesterol",
    "低密度脂蛋白胆固醇": "ldl",
    "甘油三酯": "triglycerides",
    "高密度脂蛋白胆固醇": "hdl",
    "D-二聚体": "d_dimer",
    "血小板": "platelets",
    "中性粒细胞": "neutrophils",
    "淋巴细胞": "lymphocytes",
    "NLR(中性粒细胞/淋巴细胞）": "nlr",
    "NLR（中性粒细胞/淋巴细胞）": "nlr",
    "CRP": "crp",
    "IL-6": "il6",
    "肾小球滤过率": "egfr",
    "肌酐": "creatinine",
}
TREATMENT_SOURCE = {
    "化疗": "chemotherapy",
    "放疗": "radiotherapy",
    "手术治疗": "surgery",
    "免疫治疗": "immunotherapy",
    "靶向治疗": "targeted_therapy",
}


@dataclass(frozen=True)
class ShandongWorkbookAudit:
    source_sha256: str
    source_sheet_count: int
    source_patient_rows: int
    yellow_excluded_patients: int
    green_excluded_patients: int
    yellow_green_overlap_patients: int
    excluded_unique_patients: int
    modeled_patients: int
    unique_patient_ids: int
    missing_patient_sheets: tuple[str, ...]
    extra_patient_sheets: tuple[str, ...]
    sheet_patient_id_mismatches: tuple[str, ...]
    exclusion_fill_colors: dict[str, str]
    excluded_psns: tuple[str, ...]
    modeled_psn_min: int
    modeled_psn_max: int
    model_input_audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_number(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    result = 0
    for letter in letters.group(0):
        result = result * 26 + ord(letter) - ord("A") + 1
    return result


def _row_number(reference: str) -> int:
    digits = re.search(r"\d+", reference)
    if digits is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    return int(digits.group(0))


def _normalized_argb(value: str | None) -> str:
    raw = "" if value is None else str(value).upper()
    return "FF" + raw if len(raw) == 6 else raw


def _treatment_present(value: Any) -> bool:
    text = _text(value)
    if not text:
        return False
    compact = re.sub(r"\s+", "", text).lower()
    if compact in {"0", "0.0", "无", "否", "未", "未行", "未治疗", "none", "no", "na", "n/a"}:
        return False
    parsed = _number(text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        return parsed is not None and parsed > 0
    return True


class _XLSXReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.archive = ZipFile(self.path)
        self.shared_strings = self._read_shared_strings()
        self.sheet_targets = self._read_sheet_targets()
        self.style_fill_ids, self.fill_colors = self._read_style_fills()

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "_XLSXReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _read_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.archive.namelist():
            return []
        root = ET.fromstring(self.archive.read("xl/sharedStrings.xml"))
        return [
            "".join(node.text or "" for node in item.iterfind(".//m:t", MAIN_NS))
            for item in root.findall("m:si", MAIN_NS)
        ]

    def _read_sheet_targets(self) -> dict[str, str]:
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            self.archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            node.attrib["Id"]: node.attrib["Target"]
            for node in relationships.findall("r:Relationship", REL_NS)
        }
        result = {}
        for sheet in workbook.find("m:sheets", MAIN_NS):
            target = targets[sheet.attrib[DOCUMENT_RELATIONSHIP_ID]].replace("\\", "/")
            target = target.lstrip("/")
            result[sheet.attrib["name"]] = (
                target if target.startswith("xl/") else "xl/" + target
            )
        return result

    def _read_style_fills(self) -> tuple[list[int], list[str]]:
        root = ET.fromstring(self.archive.read("xl/styles.xml"))
        colors = []
        for fill in root.find("m:fills", MAIN_NS):
            pattern = fill.find("m:patternFill", MAIN_NS)
            foreground = (
                pattern.find("m:fgColor", MAIN_NS) if pattern is not None else None
            )
            colors.append(
                _normalized_argb(foreground.attrib.get("rgb"))
                if foreground is not None
                else ""
            )
        xfs = root.find("m:cellXfs", MAIN_NS)
        return [int(xf.attrib.get("fillId", 0)) for xf in xfs], colors

    def _value(self, cell: ET.Element) -> Any:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            node = cell.find("m:is", MAIN_NS)
            return (
                "".join(item.text or "" for item in node.iterfind(".//m:t", MAIN_NS))
                if node is not None
                else ""
            )
        value = cell.find("m:v", MAIN_NS)
        if value is None or value.text is None:
            return None
        if cell_type == "s":
            index = int(value.text)
            if index >= len(self.shared_strings):
                raise ValueError(f"Shared string index is out of range: {index}")
            return self.shared_strings[index]
        if cell_type in {"str", "e"}:
            return value.text
        try:
            return float(value.text)
        except ValueError:
            return value.text

    def sheet_cells(
        self, name: str
    ) -> tuple[dict[tuple[int, int], Any], dict[tuple[int, int], int]]:
        if name not in self.sheet_targets:
            raise KeyError(name)
        root = ET.fromstring(self.archive.read(self.sheet_targets[name]))
        values: dict[tuple[int, int], Any] = {}
        styles: dict[tuple[int, int], int] = {}
        for cell in root.iterfind(".//m:c", MAIN_NS):
            reference = cell.attrib["r"]
            coordinate = (_row_number(reference), _column_number(reference))
            style = int(cell.attrib.get("s", 0))
            styles[coordinate] = style
            value = self._value(cell)
            if value not in (None, ""):
                values[coordinate] = value
        return values, styles

    def cell_exclusion_color(self, style_id: int) -> str | None:
        fill_id = self.style_fill_ids[style_id] if style_id < len(self.style_fill_ids) else 0
        color = self.fill_colors[fill_id] if fill_id < len(self.fill_colors) else ""
        return EXCLUSION_COLORS.get(color)


class ShandongExternalWorkbookReader:
    """Read the revised Shandong XLSX and exclude color-flagged patients.

    Yellow and green summary-row fills are source quality-control exclusions,
    never model features.  The retained records are converted to the exact
    deidentified payload consumed by :class:`ShandongExternalJSONReader`.
    """

    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def prepare_payload(
        self, path: str | Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source = Path(path)
        with _XLSXReader(source) as workbook:
            if SUMMARY_SHEET not in workbook.sheet_targets:
                raise ValueError(f"Missing required sheet: {SUMMARY_SHEET}")
            summary, summary_styles = workbook.sheet_cells(SUMMARY_SHEET)
            headers = {
                _text(value): column
                for (row, column), value in summary.items()
                if row == 1 and _text(value)
            }
            missing_headers = REQUIRED_HEADERS - set(headers)
            if missing_headers:
                raise ValueError(f"Missing required headers: {sorted(missing_headers)}")
            maximum_row = max(row for row, _ in summary)
            source_rows = [
                row for row in range(2, maximum_row + 1) if _text(summary.get((row, 1)))
            ]
            psn_by_row = {
                row: _text(summary[(row, 1)]).upper().replace(" ", "")
                for row in source_rows
            }
            colors_by_row: dict[int, set[str]] = {row: set() for row in source_rows}
            for (row, _), style_id in summary_styles.items():
                if row not in colors_by_row:
                    continue
                color = workbook.cell_exclusion_color(style_id)
                if color is not None:
                    colors_by_row[row].add(color)
            excluded_rows = {row for row, colors in colors_by_row.items() if colors}
            retained_rows = [row for row in source_rows if row not in excluded_rows]
            yellow_psns = {
                psn_by_row[row]
                for row, colors in colors_by_row.items()
                if "yellow" in colors
            }
            green_psns = {
                psn_by_row[row]
                for row, colors in colors_by_row.items()
                if "green" in colors
            }
            excluded_psns = yellow_psns | green_psns
            patient_sheets = set(workbook.sheet_targets) - {SUMMARY_SHEET}
            source_psns = set(psn_by_row.values())
            missing_sheets = tuple(sorted(source_psns - patient_sheets))
            extra_sheets = tuple(sorted(patient_sheets - source_psns))
            if missing_sheets or extra_sheets:
                raise ValueError(
                    f"Summary/patient sheet mismatch: missing={missing_sheets}, extra={extra_sheets}"
                )

            records = []
            id_mismatches = []
            source_patient_ids = []
            for row in retained_rows:
                psn = psn_by_row[row]
                source_patient_id = _text(
                    summary.get((row, headers["patient_SN"]))
                ).lower()
                if not re.fullmatch(r"[0-9a-f]{32}", source_patient_id):
                    raise ValueError(f"Invalid pseudonymous patient ID for {psn}")
                values, _ = workbook.sheet_cells(psn)
                sheet_patient_id = _text(values.get((1, 1))).lower()
                if sheet_patient_id != source_patient_id:
                    id_mismatches.append(psn)
                    continue
                source_patient_ids.append(source_patient_id)
                labels = {
                    _text(value): event_row
                    for (event_row, column), value in values.items()
                    if column == 1 and _text(value)
                }
                max_column = max((column for _, column in values), default=1)
                events = []
                for column in range(2, max_column + 1):
                    raw_date = values.get((2, column))
                    if not _text(raw_date) or "无住院记录" in _text(raw_date):
                        continue
                    event: dict[str, Any] = {"date": raw_date}
                    for label, source_name in FEATURE_SOURCE.items():
                        source_row = labels.get(label)
                        if source_row is None:
                            continue
                        value = _number(values.get((source_row, column)))
                        if value is not None:
                            event[source_name] = value
                    treatment: dict[str, float] = {}
                    for label, source_name in TREATMENT_SOURCE.items():
                        source_row = labels.get(label)
                        if source_row is not None and _treatment_present(
                            values.get((source_row, column))
                        ):
                            treatment[source_name] = 1.0
                    if treatment:
                        event["treatment"] = treatment
                    events.append(event)
                record = {
                    # Keep the same stable external patient key used by the
                    # original 710-patient validation.  The independent
                    # 32-character source ID is still required above to prove
                    # the summary row and patient worksheet refer to the same
                    # person, but it is not exposed to prediction outputs.
                    "patient_id": psn,
                    "age": summary.get((row, headers["年龄"])),
                    "sex_code": summary.get((row, headers["性别"])),
                    "weight_kg": summary.get((row, headers["体重（kg）"])),
                    "smoking_years": summary.get((row, headers["烟龄"])),
                    "drinking_years": summary.get((row, headers["酒龄"])),
                    "pathology_type": summary.get((row, headers["肿瘤病理类型"])),
                    "baseline_time": summary.get((row, headers["基线时间"])),
                    "first_petct_time": summary.get(
                        (row, headers["第一次PETCT影像时间"])
                    ),
                    "second_petct_time": summary.get(
                        (row, headers["第二次PETCT影像时间"])
                    ),
                    "baseline_tbr": summary.get((row, headers["第一次TBR"])),
                    "endpoint_tbr": summary.get((row, headers["第二次TBR"])),
                    "baseline_cac": summary.get(
                        (row, headers["第一次agatston评分"])
                    ),
                    "endpoint_cac": summary.get(
                        (row, headers["第二次agatston评分"])
                    ),
                    "followup_interval_months": summary.get(
                        (row, headers["治疗前后间隔时间"])
                    ),
                    "events": events,
                }
                records.append(record)
            if id_mismatches:
                raise ValueError(f"Patient sheet identifiers differ: {id_mismatches}")
            if len(source_patient_ids) != len(set(source_patient_ids)):
                duplicates = [
                    value
                    for value, count in Counter(source_patient_ids).items()
                    if count > 1
                ]
                raise ValueError(
                    f"Duplicate retained source patient IDs: {duplicates}"
                )
            patient_ids = [record["patient_id"] for record in records]
            if len(patient_ids) != len(set(patient_ids)):
                duplicates = [
                    value for value, count in Counter(patient_ids).items() if count > 1
                ]
                raise ValueError(f"Duplicate retained patient IDs: {duplicates}")
            payload = {
                "schema_version": "shandong_external_v37_deidentified_v1",
                "records": records,
            }
            workbook_audit = {
                "source_sha256": _sha256(source),
                "source_sheet_count": len(workbook.sheet_targets),
                "source_patient_rows": len(source_rows),
                "yellow_excluded_patients": len(yellow_psns),
                "green_excluded_patients": len(green_psns),
                "yellow_green_overlap_patients": len(yellow_psns & green_psns),
                "excluded_unique_patients": len(excluded_psns),
                "modeled_patients": len(records),
                "unique_patient_ids": len(set(patient_ids)),
                "missing_patient_sheets": missing_sheets,
                "extra_patient_sheets": extra_sheets,
                "sheet_patient_id_mismatches": tuple(id_mismatches),
                "exclusion_fill_colors": dict(EXCLUSION_COLORS),
                "excluded_psns": tuple(sorted(excluded_psns)),
                "modeled_psn_min": min(
                    int(re.search(r"\d+", psn_by_row[row]).group())
                    for row in retained_rows
                ),
                "modeled_psn_max": max(
                    int(re.search(r"\d+", psn_by_row[row]).group())
                    for row in retained_rows
                ),
            }
        return payload, workbook_audit

    def prepare_arrays(
        self, path: str | Path
    ) -> tuple[dict[str, Any], ShandongWorkbookAudit]:
        payload, workbook_audit = self.prepare_payload(path)
        arrays, model_audit = ShandongExternalJSONReader(self.schema).prepare_payload(
            payload
        )
        audit = ShandongWorkbookAudit(
            **workbook_audit,
            model_input_audit=model_audit.to_dict(),
        )
        if audit.source_patient_rows != 879:
            raise ValueError(
                f"Revised Shandong workbook must contain 879 source rows; got {audit.source_patient_rows}"
            )
        if audit.excluded_unique_patients != 124 or audit.modeled_patients != 755:
            raise ValueError(
                "Revised Shandong cohort lock changed: expected 124 color exclusions and 755 retained patients"
            )
        if int(model_audit.complete_baseline_and_endpoints) != 755:
            raise ValueError("All 755 retained patients must have complete primary endpoints")
        return arrays, audit
