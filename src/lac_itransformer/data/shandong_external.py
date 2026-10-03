from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .schema import FeatureSchema


DAYS_PER_MONTH = 365.25 / 12.0
EXCEL_EPOCH = datetime(1899, 12, 30)
BASE_STATIC_FEATURES = (
    "age",
    "sex_code",
    "weight",
    "smoking_years",
    "drinking_years",
    "pathology_type",
)
PATHOLOGY_LEVELS = tuple(range(8))


@dataclass(frozen=True)
class ShandongExternalAudit:
    source_records: int
    modeled_patients: int
    complete_baseline_and_endpoints: int
    patients_with_longitudinal_measurements: int
    patients_without_longitudinal_measurements: int
    invalid_event_date_cells: int
    date_cells_before_baseline_excluded: int
    date_cells_after_endpoint_excluded: int
    patients_with_derived_baseline_date: int
    patients_with_derived_endpoint_date: int
    patients_with_unresolved_date_window: int
    truncated_event_sequences: int
    invalid_static_values_by_feature: dict[str, int]
    treatment_patient_counts: dict[str, int]
    feature_patient_coverage: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", _text(value).replace(",", ""))
    if not match:
        return None
    parsed = float(match.group())
    return parsed if math.isfinite(parsed) else None


def _bounded_number(value: Any, lower: float, upper: float) -> float | None:
    parsed = _number(value)
    return parsed if parsed is not None and lower <= parsed <= upper else None


def _excel_day(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        parsed = float(value)
        return parsed if 30000 <= parsed <= 70000 else None
    text = _text(value)
    if not text or text.upper() in {"NULL", "NA", "N/A", "NONE"}:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return float((parsed - EXCEL_EPOCH).days)
    except ValueError:
        pass
    for format_string in ("%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            parsed = datetime.strptime(text, format_string)
            return float((parsed - EXCEL_EPOCH).days)
        except ValueError:
            continue
    return None


class ShandongExternalJSONReader:
    """Read the deidentified Shandong center export without fitting anything."""

    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def prepare_arrays(
        self, path: str | Path
    ) -> tuple[dict[str, np.ndarray], ShandongExternalAudit]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.prepare_payload(payload)

    def prepare_payload(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], ShandongExternalAudit]:
        """Convert an already parsed deidentified payload without a temp file."""
        if payload.get("schema_version") != "shandong_external_v37_deidentified_v1":
            raise ValueError("Unexpected Shandong external JSON schema version")
        records = list(payload.get("records", []))
        patient_ids = [str(record.get("patient_id", "")).strip() for record in records]
        if not records or any(not value for value in patient_ids):
            raise ValueError("External records require nonempty pseudonymous patient IDs")
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("External patient IDs must be unique")

        n = len(records)
        p = self.schema.time_patches
        d = len(self.schema.longitudinal_features)
        a = len(self.schema.treatment_features)
        event_length = self.schema.max_events
        arrays = {
            "static": np.full((n, len(self.schema.static_features)), np.nan, dtype=np.float32),
            "baseline": np.full((n, 2), np.nan, dtype=np.float32),
            "values": np.zeros((n, p, d), dtype=np.float32),
            "mask": np.zeros((n, p, d), dtype=np.float32),
            "delta": np.zeros((n, p, d), dtype=np.float32),
            "times": np.tile(
                np.linspace(1 / (2 * p), 1 - 1 / (2 * p), p), (n, 1)
            ).astype(np.float32),
            "treatments": np.zeros((n, p, a), dtype=np.float32),
            "irregular_values": np.zeros((n, event_length, d), dtype=np.float32),
            "irregular_mask": np.zeros((n, event_length, d), dtype=np.float32),
            "irregular_times": np.zeros((n, event_length), dtype=np.float32),
            "irregular_treatments": np.zeros((n, event_length, a), dtype=np.float32),
            "targets": np.full((n, 2), np.nan, dtype=np.float32),
            "followup_months": np.full(n, np.nan, dtype=np.float32),
            "patient_ids": np.asarray(patient_ids),
        }
        feature_source = {
            "BMI": "bmi",
            "systolic_blood_pressure": "systolic_bp",
            "glucose": "glucose",
            "cholesterol": "total_cholesterol",
            "ldl": "ldl",
            "triglyceride": "triglycerides",
            "hdl": "hdl",
            "d_dimer": "d_dimer",
            "platelet": "platelets",
            "neutrophil": "neutrophils",
            "lymphocyte": "lymphocytes",
            "nlr": "nlr",
            "crp": "crp",
            "il6": "il6",
            "egfr": "egfr",
            "creatinine": "creatinine",
        }
        missing_features = set(self.schema.longitudinal_features) - set(feature_source)
        if missing_features:
            raise ValueError(f"No Shandong source mapping for features: {sorted(missing_features)}")
        treatment_source = {
            "chemotherapy": "chemotherapy",
            "radiotherapy": "radiotherapy",
            "surgery_or_procedure": "surgery",
            "immunotherapy": "immunotherapy",
            "targeted_therapy": "targeted_therapy",
        }
        missing_treatments = set(self.schema.treatment_features) - set(treatment_source)
        if missing_treatments:
            raise ValueError(f"No Shandong treatment mapping for: {sorted(missing_treatments)}")
        feature_index = {
            name: index for index, name in enumerate(self.schema.longitudinal_features)
        }
        treatment_index = {
            name: index for index, name in enumerate(self.schema.treatment_features)
        }
        invalid_static = {name: 0 for name in BASE_STATIC_FEATURES}
        feature_patients = {name: 0 for name in self.schema.longitudinal_features}
        treatment_patients = {name: 0 for name in self.schema.treatment_features}
        invalid_event_dates = 0
        before_excluded = 0
        after_excluded = 0
        derived_start = 0
        derived_end = 0
        unresolved = 0
        truncated = 0
        with_measurements = 0

        for patient_index, record in enumerate(records):
            raw_static = {
                "age": _bounded_number(record.get("age"), 18, 100),
                "sex_code": _bounded_number(record.get("sex_code"), 0, 1),
                "weight": _bounded_number(record.get("weight_kg"), 25, 250),
                "smoking_years": _bounded_number(record.get("smoking_years"), 0, 80),
                "drinking_years": _bounded_number(record.get("drinking_years"), 0, 80),
                "pathology_type": _bounded_number(
                    record.get("pathology_type"), min(PATHOLOGY_LEVELS), max(PATHOLOGY_LEVELS)
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
                    1.0 if pathology is not None and int(pathology) == level else 0.0
                    for level in PATHOLOGY_LEVELS
                ],
                *[float(raw_static[name] is None) for name in BASE_STATIC_FEATURES],
            ]
            if len(static_values) != len(self.schema.static_features):
                raise ValueError("Shandong static encoding differs from the locked schema")
            arrays["static"][patient_index] = np.asarray(
                [np.nan if value is None else value for value in static_values], dtype=np.float32
            )

            baseline_tbr = _number(record.get("baseline_tbr"))
            endpoint_tbr = _number(record.get("endpoint_tbr"))
            baseline_cac = _number(record.get("baseline_cac"))
            endpoint_cac = _number(record.get("endpoint_cac"))
            if None in (baseline_tbr, endpoint_tbr, baseline_cac, endpoint_cac):
                raise ValueError(f"External primary endpoint missing for {patient_ids[patient_index]}")
            if baseline_cac < 0 or endpoint_cac < 0:
                raise ValueError("CAC scores must be nonnegative")
            arrays["baseline"][patient_index] = [baseline_tbr, baseline_cac]
            arrays["targets"][patient_index] = [endpoint_tbr, endpoint_cac]

            interval = _number(record.get("followup_interval_months"))
            interval = interval if interval is not None and interval > 0 else None
            start = _excel_day(record.get("first_petct_time"))
            if start is None:
                start = _excel_day(record.get("baseline_time"))
            end = _excel_day(record.get("second_petct_time"))
            if start is None and end is not None and interval is not None:
                start = end - interval * DAYS_PER_MONTH
                derived_start += 1
            if end is None and start is not None and interval is not None:
                end = start + interval * DAYS_PER_MONTH
                derived_end += 1
            arrays["followup_months"][patient_index] = (
                np.nan if interval is None else float(interval)
            )

            events_by_date: dict[float, dict[str, float]] = {}
            for raw_event in record.get("events", []):
                date = _excel_day(raw_event.get("date"))
                if date is None:
                    invalid_event_dates += 1
                    continue
                event = events_by_date.setdefault(date, {})
                for feature_name, source_name in feature_source.items():
                    value = _number(raw_event.get(source_name))
                    if value is not None:
                        event[feature_name] = value
                raw_treatment = raw_event.get("treatment", {}) or {}
                for treatment_name, source_name in treatment_source.items():
                    value = _number(raw_treatment.get(source_name))
                    if value is not None and value > 0:
                        event[treatment_name] = 1.0
            all_dates = sorted(events_by_date)
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
            if interval is None and start is not None and end is not None and end > start:
                interval = (end - start) / DAYS_PER_MONTH
                arrays["followup_months"][patient_index] = float(interval)
            if start is None or end is None or end <= start:
                unresolved += 1
                interval_years = float(interval) / 12.0 if interval is not None else 1.0
                arrays["times"][patient_index] *= interval_years
                continue
            dates = []
            for date in all_dates:
                if date < start:
                    before_excluded += 1
                elif date > end:
                    after_excluded += 1
                else:
                    dates.append(date)
            if len(dates) > event_length:
                truncated += 1
            denominator = end - start
            arrays["times"][patient_index] *= denominator / 365.25
            last_seen = {name: start for name in self.schema.longitudinal_features}
            time_sum = np.zeros(p, dtype=float)
            time_count = np.zeros(p, dtype=float)
            buckets: dict[tuple[int, str], list[float]] = {}
            patient_feature_seen: set[str] = set()
            patient_treatment_seen: set[str] = set()
            for visit_index, date in enumerate(dates):
                relative = float(np.clip((date - start) / denominator, 0.0, 1.0))
                elapsed_years = max(0.0, (date - start) / 365.25)
                patch = min(p - 1, int(relative * p))
                time_sum[patch] += elapsed_years
                time_count[patch] += 1
                event = events_by_date[date]
                for name, value in event.items():
                    if name in feature_index:
                        patient_feature_seen.add(name)
                        column = feature_index[name]
                        buckets.setdefault((patch, name), []).append(float(value))
                        arrays["delta"][patient_index, patch, column] = max(
                            0.0, date - last_seen[name]
                        )
                        last_seen[name] = date
                        if visit_index < event_length:
                            arrays["irregular_values"][patient_index, visit_index, column] = float(value)
                            arrays["irregular_mask"][patient_index, visit_index, column] = 1.0
                    elif name in treatment_index:
                        patient_treatment_seen.add(name)
                        column = treatment_index[name]
                        arrays["treatments"][patient_index, patch, column] = 1.0
                        if visit_index < event_length:
                            arrays["irregular_treatments"][patient_index, visit_index, column] = 1.0
                if visit_index < event_length:
                    arrays["irregular_times"][patient_index, visit_index] = elapsed_years
            arrays["times"][patient_index] = np.where(
                time_count > 0,
                time_sum / np.maximum(time_count, 1.0),
                arrays["times"][patient_index],
            )
            arrays["times"][patient_index] = np.maximum.accumulate(arrays["times"][patient_index])
            for (patch, name), bucket_values in buckets.items():
                column = feature_index[name]
                arrays["values"][patient_index, patch, column] = float(np.median(bucket_values))
                arrays["mask"][patient_index, patch, column] = 1.0
            with_measurements += int(bool(patient_feature_seen))
            for name in patient_feature_seen:
                feature_patients[name] += 1
            for name in patient_treatment_seen:
                treatment_patients[name] += 1

        audit = ShandongExternalAudit(
            source_records=n,
            modeled_patients=n,
            complete_baseline_and_endpoints=int(
                (
                    np.isfinite(arrays["baseline"]).all(axis=1)
                    & np.isfinite(arrays["targets"]).all(axis=1)
                ).sum()
            ),
            patients_with_longitudinal_measurements=with_measurements,
            patients_without_longitudinal_measurements=n - with_measurements,
            invalid_event_date_cells=invalid_event_dates,
            date_cells_before_baseline_excluded=before_excluded,
            date_cells_after_endpoint_excluded=after_excluded,
            patients_with_derived_baseline_date=derived_start,
            patients_with_derived_endpoint_date=derived_end,
            patients_with_unresolved_date_window=unresolved,
            truncated_event_sequences=truncated,
            invalid_static_values_by_feature=invalid_static,
            treatment_patient_counts=treatment_patients,
            feature_patient_coverage=feature_patients,
        )
        return arrays, audit
