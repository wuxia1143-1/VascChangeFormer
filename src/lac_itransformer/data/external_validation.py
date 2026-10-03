from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from .schema import FeatureSchema


class ExternalValidationGuardrailError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkbookAudit:
    baseline_rows: int
    followup_rows: int
    longitudinal_sheets: int
    matched_baseline_followup_ids: int
    longitudinal_hashed_ids: int
    longitudinal_missing_hashed_ids: int
    invalid_date_cells: int
    nonmonotonic_sheets: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", _text(value).replace(",", ""))
    return float(match.group()) if match else None


class ExternalWorkbookReader:
    """Reader/auditor for the center-A legacy workbook.

    This adapter is intentionally separated from training. Its public API only
    exposes audit and locked external-evaluation preparation; it has no ``fit``
    method and refuses any purpose other than external validation.
    """

    REQUIRED_SHEETS = {"备注", "治疗前", "治疗后"}

    def __init__(self, purpose: str = "external_validation"):
        if purpose != "external_validation":
            raise ExternalValidationGuardrailError("This workbook is external-validation-only")

    def audit(self, path: str | Path) -> WorkbookAudit:
        import xlrd

        workbook = xlrd.open_workbook(str(path), on_demand=True)
        missing = self.REQUIRED_SHEETS - set(workbook.sheet_names())
        if missing:
            raise ValueError(f"Missing required sheets: {sorted(missing)}")
        before = workbook.sheet_by_name("治疗前")
        after = workbook.sheet_by_name("治疗后")
        before_ids = {_text(before.cell_value(row, 2)).lower() for row in range(2, before.nrows) if _text(before.cell_value(row, 2))}
        after_ids = {_text(after.cell_value(row, 1)).lower() for row in range(1, after.nrows) if _text(after.cell_value(row, 1))}
        longitudinal = [name for name in workbook.sheet_names() if name not in self.REQUIRED_SHEETS]
        hashed = 0
        bad_dates = 0
        nonmonotonic = 0
        for name in longitudinal:
            sheet = workbook.sheet_by_name(name)
            first = _text(sheet.cell_value(0, 0)) if sheet.nrows and sheet.ncols else ""
            hashed += int(bool(re.fullmatch(r"[0-9a-fA-F]{32}", first)))
            dates: list[float] = []
            for col in range(1, sheet.ncols):
                raw = sheet.cell_value(0, col)
                value = _number(raw)
                if value is not None and 30000 <= value <= 60000:
                    dates.append(value)
                elif _text(raw):
                    bad_dates += 1
            nonmonotonic += int(any(b < a for a, b in zip(dates, dates[1:])))
        warnings: list[str] = []
        if before_ids != after_ids:
            warnings.append("baseline/follow-up identifier sets differ")
        if bad_dates:
            warnings.append("invalid longitudinal date cells require review")
        if nonmonotonic:
            warnings.append("non-monotonic visit dates will be sorted during preparation")
        if len(longitudinal) != len(before_ids):
            warnings.append("longitudinal sheet count differs from patient count; duplicate/supplement sheets may exist")
        return WorkbookAudit(
            baseline_rows=max(0, before.nrows - 2),
            followup_rows=max(0, after.nrows - 1),
            longitudinal_sheets=len(longitudinal),
            matched_baseline_followup_ids=len(before_ids & after_ids),
            longitudinal_hashed_ids=hashed,
            longitudinal_missing_hashed_ids=len(longitudinal) - hashed,
            invalid_date_cells=bad_dates,
            nonmonotonic_sheets=nonmonotonic,
            warnings=tuple(warnings),
        )

    def prepare_locked_arrays(self, path: str | Path, schema: FeatureSchema) -> dict[str, np.ndarray]:
        """Convert the legacy workbook to the raw model contract without fitting.

        All normalization/imputation must subsequently use a serialized
        training-center ``FoldPreprocessor``. Patient rows never need to be
        written to the repository.
        """
        import xlrd

        workbook = xlrd.open_workbook(str(path), on_demand=True)
        before = workbook.sheet_by_name("治疗前")
        after = workbook.sheet_by_name("治疗后")
        header = [_text(before.cell_value(1, col)) for col in range(before.ncols)]
        groups = [_text(before.cell_value(0, col)) for col in range(before.ncols)]
        header_index = {name: idx for idx, name in enumerate(header) if name}
        tbr_col = next(i for i, (g, h) in enumerate(zip(groups, header)) if g == "TBR值" and h == "胸主动脉")
        cac_col = next(i for i, (g, h) in enumerate(zip(groups, header)) if g == "Agatston评分" and h == "胸主动脉")
        time_col = header_index.get("检验时间")
        static_columns = {
            "age": header_index["年龄"],
            "sex": header_index["性别"],
            "weight": header_index["体重（kg）"],
            "smoking_years": header_index["烟龄"],
            "drinking_years": header_index["酒龄"],
            "pathology_type": header_index["肿瘤病理类型"],
        }
        after_header = {_text(after.cell_value(0, col)): col for col in range(after.ncols)}
        after_by_id = {
            _text(after.cell_value(row, 1)).lower(): row
            for row in range(1, after.nrows)
            if _text(after.cell_value(row, 1))
        }
        baseline_records: list[dict[str, Any]] = []
        psn_to_id: dict[str, str] = {}
        for row in range(2, before.nrows):
            patient_id = _text(before.cell_value(row, 2)).lower()
            psn = _text(before.cell_value(row, 0)).upper().replace(" ", "")
            if not patient_id or patient_id not in after_by_id:
                continue
            psn_to_id[psn] = patient_id
            baseline_records.append({
                "patient_id": patient_id,
                "psn": psn,
                "row": row,
                "baseline_time": _number(before.cell_value(row, time_col)) if time_col is not None else None,
            })

        longitudinal_map = {
            "BMI": "BMI",
            "血压": "blood_pressure",
            "葡萄糖": "glucose",
            "胆固醇": "cholesterol",
            "低密度脂蛋白胆固醇": "ldl",
            "甘油三酯": "triglyceride",
            "高密度脂蛋白胆固醇": "hdl",
            "D-二聚体": "d_dimer",
            "血小板": "platelet",
            "中性粒细胞": "neutrophil",
            "淋巴细胞": "lymphocyte",
            "NLR(中性粒细胞/淋巴细胞）": "nlr",
            "NLR（中性粒细胞/淋巴细胞）": "nlr",
            "CRP": "crp",
            "IL-6": "il6",
            "肾小球滤过率": "egfr",
            "肌酐": "creatinine",
        }
        treatment_map = {
            "化疗": "chemotherapy",
            "放疗": "radiotherapy",
            "手术治疗": "surgery",
            "免疫治疗": "immunotherapy",
            "靶向治疗": "targeted_therapy",
        }
        events: dict[str, dict[float, dict[str, Any]]] = {}
        for sheet_name in workbook.sheet_names():
            if sheet_name in self.REQUIRED_SHEETS:
                continue
            sheet = workbook.sheet_by_name(sheet_name)
            raw_id = _text(sheet.cell_value(0, 0)).lower() if sheet.nrows and sheet.ncols else ""
            if not re.fullmatch(r"[0-9a-f]{32}", raw_id):
                digits = re.findall(r"\d+", sheet_name)
                raw_id = psn_to_id.get(f"PSN{digits[-1]}" if digits else "", "")
            if not raw_id:
                continue
            patient_events = events.setdefault(raw_id, {})
            for col in range(1, sheet.ncols):
                date = _number(sheet.cell_value(0, col))
                if date is None or not 30000 <= date <= 60000:
                    continue
                record = patient_events.setdefault(date, {})
                for row in range(1, sheet.nrows):
                    label = _text(sheet.cell_value(row, 0))
                    value = sheet.cell_value(row, col)
                    if label in longitudinal_map and _number(value) is not None:
                        record[longitudinal_map[label]] = _number(value)
                    elif label in treatment_map and _text(value):
                        record[treatment_map[label]] = 1.0

        n, p = len(baseline_records), schema.time_patches
        event_length = schema.max_events
        d, a = len(schema.longitudinal_features), len(schema.treatment_features)
        arrays = {
            "static": np.full((n, len(schema.static_features)), np.nan, dtype=np.float32),
            "baseline": np.full((n, 2), np.nan, dtype=np.float32),
            "values": np.zeros((n, p, d), dtype=np.float32),
            "mask": np.zeros((n, p, d), dtype=np.float32),
            "delta": np.zeros((n, p, d), dtype=np.float32),
            "times": np.tile(np.linspace(1 / (2 * p), 1 - 1 / (2 * p), p), (n, 1)).astype(np.float32),
            "treatments": np.zeros((n, p, a), dtype=np.float32),
            "irregular_values": np.zeros((n, event_length, d), dtype=np.float32),
            "irregular_mask": np.zeros((n, event_length, d), dtype=np.float32),
            "irregular_times": np.zeros((n, event_length), dtype=np.float32),
            "irregular_treatments": np.zeros((n, event_length, a), dtype=np.float32),
            "targets": np.full((n, 2), np.nan, dtype=np.float32),
            "patient_ids": np.asarray([record["patient_id"] for record in baseline_records]),
        }
        feature_index = {name: idx for idx, name in enumerate(schema.longitudinal_features)}
        treatment_index = {name: idx for idx, name in enumerate(schema.treatment_features)}
        for i, record in enumerate(baseline_records):
            row, patient_id = record["row"], record["patient_id"]
            arrays["static"][i] = [
                _number(before.cell_value(row, static_columns[name])) if _number(before.cell_value(row, static_columns[name])) is not None else np.nan
                for name in schema.static_features
            ]
            baseline_tbr = _number(before.cell_value(row, tbr_col))
            baseline_cac = _number(before.cell_value(row, cac_col))
            arrays["baseline"][i] = [
                baseline_tbr if baseline_tbr is not None else np.nan,
                baseline_cac if baseline_cac is not None else np.nan,
            ]
            after_row = after_by_id[patient_id]
            endpoint_tbr = _number(after.cell_value(after_row, after_header["第二次胸主动脉TBR"]))
            endpoint_cac = _number(after.cell_value(after_row, after_header["第二次Agatston评分"]))
            arrays["targets"][i] = [
                endpoint_tbr if endpoint_tbr is not None else np.nan,
                endpoint_cac if endpoint_cac is not None else np.nan,
            ]
            endpoint_time = _number(after.cell_value(after_row, after_header["影像时间"]))
            patient_events = events.get(patient_id, {})
            if not patient_events:
                continue
            dates = sorted(patient_events)
            start = record["baseline_time"] if record["baseline_time"] is not None else dates[0]
            end = endpoint_time if endpoint_time is not None and endpoint_time > start else dates[-1]
            denominator = max(end - start, 1.0)
            last_seen = {name: start for name in schema.longitudinal_features}
            time_sum = np.zeros(p, dtype=float)
            time_count = np.zeros(p, dtype=float)
            buckets: dict[tuple[int, str], list[float]] = {}
            for visit_index, date in enumerate(dates):
                relative = float(np.clip((date - start) / denominator, 0.0, 1.0))
                patch = min(p - 1, int(relative * p))
                time_sum[patch] += relative
                time_count[patch] += 1
                event = patient_events[date]
                for name, value in event.items():
                    if name in feature_index:
                        if visit_index < event_length:
                            arrays["irregular_values"][i, visit_index, feature_index[name]] = float(value)
                            arrays["irregular_mask"][i, visit_index, feature_index[name]] = 1.0
                        buckets.setdefault((patch, name), []).append(float(value))
                        arrays["delta"][i, patch, feature_index[name]] = max(0.0, date - last_seen[name])
                        last_seen[name] = date
                    elif name in treatment_index:
                        if visit_index < event_length:
                            arrays["irregular_treatments"][i, visit_index, treatment_index[name]] = 1.0
                        arrays["treatments"][i, patch, treatment_index[name]] = 1.0
                if visit_index < event_length:
                    arrays["irregular_times"][i, visit_index] = relative
            arrays["times"][i] = np.where(time_count > 0, time_sum / np.maximum(time_count, 1), arrays["times"][i])
            arrays["times"][i] = np.maximum.accumulate(arrays["times"][i])
            for (patch, name), values in buckets.items():
                col = feature_index[name]
                arrays["values"][i, patch, col] = float(np.median(values))
                arrays["mask"][i, patch, col] = 1.0
        return arrays

    @staticmethod
    def assert_locked_artifacts(checkpoint: str | Path, preprocessor: str | Path) -> None:
        for path in (checkpoint, preprocessor):
            if not Path(path).is_file():
                raise ExternalValidationGuardrailError(f"Locked training-center artifact missing: {path}")
