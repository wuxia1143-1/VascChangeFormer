from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np

from .schema import (
    DEFAULT_LONGITUDINAL_FEATURES,
    DEFAULT_TREATMENTS,
    FeatureSchema,
)


BASE_STATIC_FEATURES = (
    "age",
    "sex_code",
    "weight",
    "smoking_years",
    "drinking_years",
    "pathology_type",
)
PATHOLOGY_LEVELS = tuple(range(8))
DAYS_PER_MONTH = 365.25 / 12
DEVELOPMENT_STATIC_FEATURES = (
    "age",
    "sex_code",
    "weight",
    "smoking_years",
    "drinking_years",
    *(f"pathology_type_{level}" for level in PATHOLOGY_LEVELS),
    *(f"{name}_missing" for name in BASE_STATIC_FEATURES),
)


def qingyi_development_schema() -> FeatureSchema:
    """Schema used for this real internal-development cohort.

    Pathology is one-hot encoded because its source values are categories, not
    an ordinal measurement. Explicit missing indicators preserve informative
    missingness while fold-fitted preprocessing handles the numeric values.
    """

    longitudinal = tuple(
        "systolic_blood_pressure" if name == "blood_pressure" else name
        for name in DEFAULT_LONGITUDINAL_FEATURES
    )
    treatments = tuple(
        "surgery_or_procedure" if name == "surgery" else name
        for name in DEFAULT_TREATMENTS
    )
    return FeatureSchema(
        static_features=DEVELOPMENT_STATIC_FEATURES,
        longitudinal_features=longitudinal,
        treatment_features=treatments,
    )


@dataclass(frozen=True)
class DevelopmentWorkbookAudit:
    source_patient_rows: int
    matched_patient_rows: int
    modeled_patients: int
    patients_with_longitudinal_measurements: int
    patients_without_longitudinal_measurements: int
    longitudinal_sheets: int
    longitudinal_sheets_mapped: int
    invalid_date_cells: int
    date_cells_before_baseline_excluded: int
    date_cells_after_endpoint_excluded: int
    date_cells_after_prediction_cutoff_excluded: int
    date_cells_with_unresolved_window_excluded: int
    patients_with_derived_baseline_date: int
    patients_with_derived_endpoint_date: int
    followup_interval_unit: str
    days_per_interval_unit: float
    patients_with_unresolved_date_window: int
    truncated_event_sequences: int
    invalid_static_values_by_feature: dict[str, int]
    complete_baseline_and_endpoints: int
    prediction_lead_months: float
    patients_without_visible_longitudinal_events: int
    treatment_patient_counts: dict[str, int]
    feature_patient_coverage: dict[str, int]
    followup_filter_min_months: float | None
    followup_filter_max_months: float | None
    followup_filter_excluded_missing: int
    followup_filter_excluded_below_minimum: int
    followup_filter_excluded_above_maximum: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", _text(value).replace(",", ""))
    return float(match.group()) if match else None


def _bounded_number(
    value: Any,
    lower: float,
    upper: float,
) -> float | None:
    parsed = _number(value)
    return parsed if parsed is not None and lower <= parsed <= upper else None


def _excel_date(value: Any, datemode: int) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        numeric = float(value)
        return numeric if 30000 <= numeric <= 60000 else None
    text = _text(value)
    if not text:
        return None
    for format_string in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
    ):
        try:
            parsed = datetime.strptime(text, format_string)
            import xlrd

            return float(
                xlrd.xldate.xldate_from_datetime_tuple(
                    (parsed.year, parsed.month, parsed.day, 0, 0, 0),
                    datemode,
                )
            )
        except ValueError:
            continue
    return None


def _treatment_present(value: Any) -> bool:
    text = _text(value)
    if not text:
        return False
    numeric = _number(text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text.replace(",", "")):
        return numeric is not None and numeric > 0
    compact = re.sub(r"\s+", "", text).lower()
    return compact not in {
        "无",
        "否",
        "未",
        "未行",
        "未治疗",
        "未予治疗",
        "none",
        "no",
        "n/a",
        "na",
    }


class DevelopmentWorkbookReader:
    """Strict, read-only adapter for real internal development and nested CV."""

    REQUIRED_SHEETS = {"备注", "治疗前", "治疗后"}

    def __init__(self, schema: FeatureSchema | None = None):
        self.schema = schema or qingyi_development_schema()

    def prepare_arrays(
        self,
        path: str | Path,
        prediction_lead_months: float = 0.0,
        min_followup_months: float | None = None,
        max_followup_months: float | None = None,
    ) -> tuple[dict[str, np.ndarray], DevelopmentWorkbookAudit]:
        import xlrd

        prediction_lead_months = float(prediction_lead_months)
        if not np.isfinite(prediction_lead_months) or prediction_lead_months < 0:
            raise ValueError("prediction_lead_months must be finite and nonnegative")
        if min_followup_months is not None:
            min_followup_months = float(min_followup_months)
            if not np.isfinite(min_followup_months) or min_followup_months <= 0:
                raise ValueError("min_followup_months must be finite and positive")
        if max_followup_months is not None:
            max_followup_months = float(max_followup_months)
            if not np.isfinite(max_followup_months) or max_followup_months <= 0:
                raise ValueError("max_followup_months must be finite and positive")
        if (
            min_followup_months is not None
            and max_followup_months is not None
            and min_followup_months > max_followup_months
        ):
            raise ValueError(
                "min_followup_months must not exceed max_followup_months"
            )

        workbook = xlrd.open_workbook(str(path), on_demand=True)
        missing = self.REQUIRED_SHEETS - set(workbook.sheet_names())
        if missing:
            raise ValueError(f"Missing required sheets: {sorted(missing)}")
        before = workbook.sheet_by_name("治疗前")
        after = workbook.sheet_by_name("治疗后")
        header = [_text(before.cell_value(1, col)) for col in range(before.ncols)]
        groups = [_text(before.cell_value(0, col)) for col in range(before.ncols)]
        header_index = {name: idx for idx, name in enumerate(header) if name}
        tbr_col = next(
            index
            for index, (group, name) in enumerate(zip(groups, header))
            if group == "TBR值" and name == "胸主动脉"
        )
        cac_col = next(
            index
            for index, (group, name) in enumerate(zip(groups, header))
            if group == "Agatston评分" and name == "胸主动脉"
        )
        static_columns = {
            "age": header_index["年龄"],
            "sex_code": header_index["性别"],
            "weight": header_index["体重（kg）"],
            "smoking_years": header_index["烟龄"],
            "drinking_years": header_index["酒龄"],
            "pathology_type": header_index["肿瘤病理类型"],
        }
        after_header = {
            _text(after.cell_value(0, col)): col for col in range(after.ncols)
        }
        required_followup = {
            "第二次胸主动脉TBR",
            "第二次Agatston评分",
            "影像时间",
            "治疗前后间隔时间",
        }
        missing_followup = required_followup - set(after_header)
        if missing_followup:
            raise ValueError(
                f"Missing required follow-up columns: {sorted(missing_followup)}"
            )
        after_by_id = {
            _text(after.cell_value(row, 1)).lower(): row
            for row in range(1, after.nrows)
            if _text(after.cell_value(row, 1))
        }

        records: list[dict[str, Any]] = []
        psn_to_id: dict[str, str] = {}
        derived_start = 0
        derived_end = 0
        for row in range(2, before.nrows):
            patient_id = _text(before.cell_value(row, 2)).lower()
            psn = _text(before.cell_value(row, 0)).upper().replace(" ", "")
            if not patient_id or patient_id not in after_by_id:
                continue
            followup_row = after_by_id[patient_id]
            start = _excel_date(
                before.cell_value(row, header_index["检验时间"]),
                workbook.datemode,
            )
            end = _excel_date(
                after.cell_value(followup_row, after_header["影像时间"]),
                workbook.datemode,
            )
            interval = _number(
                after.cell_value(
                    followup_row, after_header["治疗前后间隔时间"]
                )
            )
            interval = interval if interval is not None and interval > 0 else None
            if start is None and end is not None and interval is not None:
                start = end - interval * DAYS_PER_MONTH
                derived_start += 1
            if end is None and start is not None and interval is not None:
                end = start + interval * DAYS_PER_MONTH
                derived_end += 1
            psn_to_id[psn] = patient_id
            records.append(
                {
                    "patient_id": patient_id,
                    "psn": psn,
                    "row": row,
                    "followup_row": followup_row,
                    "start": start,
                    "end": end,
                    "interval": interval,
                }
            )

        matched_patient_rows = len(records)
        excluded_missing_followup = 0
        excluded_below_followup = 0
        excluded_above_followup = 0
        if min_followup_months is not None or max_followup_months is not None:
            eligible_records: list[dict[str, Any]] = []
            for record in records:
                interval = record["interval"]
                if interval is None or not np.isfinite(interval):
                    excluded_missing_followup += 1
                elif (
                    min_followup_months is not None
                    and interval < min_followup_months
                ):
                    excluded_below_followup += 1
                elif (
                    max_followup_months is not None
                    and interval > max_followup_months
                ):
                    excluded_above_followup += 1
                else:
                    eligible_records.append(record)
            records = eligible_records

        longitudinal_map = {
            "BMI": "BMI",
            "血压": "systolic_blood_pressure",
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
        # Added laboratory channels for the 2026-09-22 full-input experiment.
        longitudinal_map.update({'神经元特异性烯醇化酶': 'nse', '鳞状细胞癌相关抗原': 'scc_antigen', '胃泌素释放肽前体': 'progrp', '细胞角蛋白19片段': 'cyfra21_1', '降钙素原': 'procalcitonin', 'BNP': 'bnp', '高敏肌钙蛋白T': 'hs_troponin_t', '脂蛋白a': 'lipoprotein_a', '游离三碘甲状腺原氨酸': 'ft3', '促甲状腺激素': 'tsh', '游离甲状腺素': 'ft4'})
        treatment_map = {
            "化疗": "chemotherapy",
            "放疗": "radiotherapy",
            "手术治疗": "surgery_or_procedure",
            "免疫治疗": "immunotherapy",
            "靶向治疗": "targeted_therapy",
        }
        events: dict[str, dict[float, dict[str, float]]] = {}
        invalid_date_cells = 0
        mapped_sheets = 0
        for sheet_name in workbook.sheet_names():
            if sheet_name in self.REQUIRED_SHEETS:
                continue
            sheet = workbook.sheet_by_name(sheet_name)
            raw_id = (
                _text(sheet.cell_value(0, 0)).lower()
                if sheet.nrows and sheet.ncols
                else ""
            )
            if not re.fullmatch(r"[0-9a-f]{32}", raw_id):
                digits = re.findall(r"\d+", sheet_name)
                raw_id = psn_to_id.get(
                    f"PSN{digits[-1]}" if digits else "",
                    "",
                )
            if not raw_id:
                continue
            mapped_sheets += 1
            patient_events = events.setdefault(raw_id, {})
            for col in range(1, sheet.ncols):
                raw_date = sheet.cell_value(0, col)
                date = _excel_date(raw_date, workbook.datemode)
                if date is None:
                    invalid_date_cells += int(bool(_text(raw_date)))
                    continue
                event = patient_events.setdefault(date, {})
                for row_index in range(1, sheet.nrows):
                    label = _text(sheet.cell_value(row_index, 0))
                    value = sheet.cell_value(row_index, col)
                    if label in longitudinal_map:
                        numeric = _number(value)
                        if numeric is not None and np.isfinite(numeric):
                            event[longitudinal_map[label]] = float(numeric)
                    elif label in treatment_map and _treatment_present(value):
                        event[treatment_map[label]] = 1.0

        n = len(records)
        p = self.schema.time_patches
        event_length = self.schema.max_events
        d = len(self.schema.longitudinal_features)
        a = len(self.schema.treatment_features)
        arrays = {
            "static": np.full(
                (n, len(self.schema.static_features)),
                np.nan,
                dtype=np.float32,
            ),
            "baseline": np.full((n, 2), np.nan, dtype=np.float32),
            "values": np.zeros((n, p, d), dtype=np.float32),
            "mask": np.zeros((n, p, d), dtype=np.float32),
            "delta": np.zeros((n, p, d), dtype=np.float32),
            "times": np.tile(
                np.linspace(1 / (2 * p), 1 - 1 / (2 * p), p),
                (n, 1),
            ).astype(np.float32),
            "treatments": np.zeros((n, p, a), dtype=np.float32),
            "irregular_values": np.zeros(
                (n, event_length, d), dtype=np.float32
            ),
            "irregular_mask": np.zeros(
                (n, event_length, d), dtype=np.float32
            ),
            "irregular_times": np.zeros(
                (n, event_length), dtype=np.float32
            ),
            "irregular_treatments": np.zeros(
                (n, event_length, a), dtype=np.float32
            ),
            "targets": np.full((n, 2), np.nan, dtype=np.float32),
            "followup_months": np.asarray(
                [
                    (
                        np.nan
                        if record["interval"] is None
                        else float(record["interval"])
                    )
                    for record in records
                ],
                dtype=np.float32,
            ),
            "endpoint_elapsed_years": np.full(n, np.nan, dtype=np.float32),
            "prediction_cutoff_elapsed_years": np.full(
                n, np.nan, dtype=np.float32
            ),
            "patient_ids": np.asarray(
                [record["patient_id"] for record in records]
            ),
        }
        feature_index = {
            name: index
            for index, name in enumerate(self.schema.longitudinal_features)
        }
        treatment_index = {
            name: index
            for index, name in enumerate(self.schema.treatment_features)
        }
        invalid_static = {name: 0 for name in BASE_STATIC_FEATURES}
        before_excluded = 0
        after_excluded = 0
        after_prediction_cutoff_excluded = 0
        unresolved_excluded = 0
        unresolved_patients = 0
        truncated = 0
        for patient_index, record in enumerate(records):
            row = record["row"]
            followup_row = record["followup_row"]
            raw_static = {
                "age": _bounded_number(
                    before.cell_value(row, static_columns["age"]), 18, 100
                ),
                "sex_code": _bounded_number(
                    before.cell_value(row, static_columns["sex_code"]), 0, 1
                ),
                "weight": _bounded_number(
                    before.cell_value(row, static_columns["weight"]), 25, 250
                ),
                "smoking_years": _bounded_number(
                    before.cell_value(row, static_columns["smoking_years"]),
                    0,
                    80,
                ),
                "drinking_years": _bounded_number(
                    before.cell_value(row, static_columns["drinking_years"]),
                    0,
                    80,
                ),
                "pathology_type": _bounded_number(
                    before.cell_value(
                        row, static_columns["pathology_type"]
                    ),
                    min(PATHOLOGY_LEVELS),
                    max(PATHOLOGY_LEVELS),
                ),
            }
            for name, value in raw_static.items():
                invalid_static[name] += int(value is None)
            pathology = raw_static["pathology_type"]
            static_values = [
                raw_static["age"],
                raw_static["sex_code"],
                raw_static["weight"],
                raw_static["smoking_years"],
                raw_static["drinking_years"],
                *[
                    (
                        1.0
                        if pathology is not None
                        and int(pathology) == level
                        else 0.0
                    )
                    for level in PATHOLOGY_LEVELS
                ],
                *[
                    float(raw_static[name] is None)
                    for name in BASE_STATIC_FEATURES
                ],
            ]
            arrays["static"][patient_index] = np.asarray(
                [
                    np.nan if value is None else value
                    for value in static_values
                ],
                dtype=np.float32,
            )
            baseline_tbr = _number(before.cell_value(row, tbr_col))
            baseline_cac = _number(before.cell_value(row, cac_col))
            endpoint_tbr = _number(
                after.cell_value(
                    followup_row, after_header["第二次胸主动脉TBR"]
                )
            )
            endpoint_cac = _number(
                after.cell_value(
                    followup_row, after_header["第二次Agatston评分"]
                )
            )
            arrays["baseline"][patient_index] = [
                np.nan if baseline_tbr is None else baseline_tbr,
                np.nan if baseline_cac is None else baseline_cac,
            ]
            arrays["targets"][patient_index] = [
                np.nan if endpoint_tbr is None else endpoint_tbr,
                np.nan if endpoint_cac is None else endpoint_cac,
            ]

            patient_events = events.get(record["patient_id"], {})
            all_dates = sorted(patient_events)
            start = record["start"]
            end = record["end"]
            if start is None and end is not None:
                candidates = [date for date in all_dates if date <= end]
                if candidates:
                    start = min(candidates)
                    derived_start += 1
            if end is None and start is not None:
                candidates = [date for date in all_dates if date >= start]
                if candidates:
                    end = max(candidates)
                    derived_end += 1
            if start is None or end is None or end <= start:
                unresolved_patients += 1
                unresolved_excluded += len(all_dates)
                interval_years = (
                    float(record["interval"]) / 12
                    if record["interval"] is not None
                    else 1.0
                )
                arrays["times"][patient_index] *= interval_years
                continue
            dates = []
            prediction_cutoff = end - prediction_lead_months * DAYS_PER_MONTH
            arrays["endpoint_elapsed_years"][patient_index] = (
                end - start
            ) / 365.25
            arrays["prediction_cutoff_elapsed_years"][patient_index] = (
                prediction_cutoff - start
            ) / 365.25
            for date in all_dates:
                if date < start:
                    before_excluded += 1
                elif date > end:
                    after_excluded += 1
                elif date > prediction_cutoff:
                    after_prediction_cutoff_excluded += 1
                else:
                    dates.append(date)
            if len(dates) > event_length:
                truncated += 1
            denominator = end - start
            arrays["times"][patient_index] *= denominator / 365.25
            last_seen = {
                name: start for name in self.schema.longitudinal_features
            }
            time_sum = np.zeros(p, dtype=float)
            time_count = np.zeros(p, dtype=float)
            buckets: dict[tuple[int, str], list[float]] = {}
            for visit_index, date in enumerate(dates):
                relative = float(np.clip((date - start) / denominator, 0, 1))
                elapsed_years = max(0.0, (date - start) / 365.25)
                patch = min(p - 1, int(relative * p))
                time_sum[patch] += elapsed_years
                time_count[patch] += 1
                event = patient_events[date]
                for name, value in event.items():
                    if name in feature_index:
                        feature = feature_index[name]
                        if visit_index < event_length:
                            arrays["irregular_values"][
                                patient_index, visit_index, feature
                            ] = value
                            arrays["irregular_mask"][
                                patient_index, visit_index, feature
                            ] = 1
                        buckets.setdefault((patch, name), []).append(value)
                        arrays["delta"][
                            patient_index, patch, feature
                        ] = max(0, date - last_seen[name])
                        last_seen[name] = date
                    elif name in treatment_index:
                        treatment = treatment_index[name]
                        arrays["treatments"][
                            patient_index, patch, treatment
                        ] = 1
                        if visit_index < event_length:
                            arrays["irregular_treatments"][
                                patient_index, visit_index, treatment
                            ] = 1
                if visit_index < event_length:
                    arrays["irregular_times"][
                        patient_index, visit_index
                    ] = elapsed_years
            arrays["times"][patient_index] = np.where(
                time_count > 0,
                time_sum / np.maximum(time_count, 1),
                arrays["times"][patient_index],
            )
            arrays["times"][patient_index] = np.maximum.accumulate(
                arrays["times"][patient_index]
            )
            for (patch, name), values in buckets.items():
                feature = feature_index[name]
                arrays["values"][
                    patient_index, patch, feature
                ] = float(np.median(values))
                arrays["mask"][patient_index, patch, feature] = 1

        complete = (
            np.isfinite(arrays["baseline"]).all(axis=1)
            & np.isfinite(arrays["targets"]).all(axis=1)
            & (arrays["baseline"][:, 0] > 0)
            & (arrays["targets"][:, 0] > 0)
            & (arrays["baseline"][:, 1] >= 0)
            & (arrays["targets"][:, 1] >= 0)
        )
        if not complete.all():
            arrays = {
                key: (
                    np.asarray(value)[complete]
                    if np.asarray(value).ndim
                    and len(np.asarray(value)) == len(complete)
                    else np.asarray(value)
                )
                for key, value in arrays.items()
            }
        ids = [str(value) for value in arrays["patient_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Patient identifiers must be unique")
        if not np.isfinite(arrays["baseline"]).all():
            raise ValueError("Baseline outcomes must be finite")
        if not np.isfinite(arrays["targets"]).all():
            raise ValueError("Endpoint outcomes must be finite")
        if not np.isfinite(arrays["followup_months"]).all():
            raise ValueError("Follow-up interval in months must be finite")
        if np.any(arrays["followup_months"] <= 0):
            raise ValueError("Follow-up interval in months must be positive")
        if np.any(np.diff(arrays["times"], axis=1) < -1e-8):
            raise ValueError("Prepared time patches must be monotonic")

        audit = DevelopmentWorkbookAudit(
            source_patient_rows=max(0, before.nrows - 2),
            matched_patient_rows=matched_patient_rows,
            modeled_patients=len(ids),
            patients_with_longitudinal_measurements=int(
                arrays["mask"].astype(bool).any(axis=(1, 2)).sum()
            ),
            patients_without_longitudinal_measurements=int(
                (~arrays["mask"].astype(bool).any(axis=(1, 2))).sum()
            ),
            longitudinal_sheets=len(workbook.sheet_names())
            - len(self.REQUIRED_SHEETS),
            longitudinal_sheets_mapped=mapped_sheets,
            invalid_date_cells=invalid_date_cells,
            date_cells_before_baseline_excluded=before_excluded,
            date_cells_after_endpoint_excluded=after_excluded,
            date_cells_after_prediction_cutoff_excluded=(
                after_prediction_cutoff_excluded
            ),
            date_cells_with_unresolved_window_excluded=unresolved_excluded,
            patients_with_derived_baseline_date=derived_start,
            patients_with_derived_endpoint_date=derived_end,
            followup_interval_unit="months",
            days_per_interval_unit=DAYS_PER_MONTH,
            patients_with_unresolved_date_window=unresolved_patients,
            truncated_event_sequences=truncated,
            invalid_static_values_by_feature=invalid_static,
            complete_baseline_and_endpoints=int(complete.sum()),
            prediction_lead_months=prediction_lead_months,
            patients_without_visible_longitudinal_events=int(
                (~arrays["mask"].astype(bool).any(axis=(1, 2))).sum()
            ),
            treatment_patient_counts={
                name: int(
                    (
                        arrays["treatments"][:, :, index] > 0
                    ).any(axis=1).sum()
                )
                for index, name in enumerate(self.schema.treatment_features)
            },
            feature_patient_coverage={
                name: int(
                    arrays["mask"][:, :, index].astype(bool).any(axis=1).sum()
                )
                for index, name in enumerate(
                    self.schema.longitudinal_features
                )
            },
            followup_filter_min_months=min_followup_months,
            followup_filter_max_months=max_followup_months,
            followup_filter_excluded_missing=excluded_missing_followup,
            followup_filter_excluded_below_minimum=excluded_below_followup,
            followup_filter_excluded_above_maximum=excluded_above_followup,
        )
        return arrays, audit
