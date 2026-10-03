from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


DEFAULT_LONGITUDINAL_FEATURES = (
    "BMI",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "ldl",
    "triglyceride",
    "hdl",
    "d_dimer",
    "platelet",
    "neutrophil",
    "lymphocyte",
    "nlr",
    "crp",
    "il6",
    "egfr",
    "creatinine",
)
DEFAULT_TREATMENTS = ("chemotherapy", "radiotherapy", "surgery", "immunotherapy", "targeted_therapy")
DEFAULT_STATIC_FEATURES = ("age", "sex", "weight", "smoking_years", "drinking_years", "pathology_type")


@dataclass(frozen=True)
class FeatureSchema:
    static_features: tuple[str, ...] = DEFAULT_STATIC_FEATURES
    longitudinal_features: tuple[str, ...] = DEFAULT_LONGITUDINAL_FEATURES
    treatment_features: tuple[str, ...] = DEFAULT_TREATMENTS
    target_features: tuple[str, ...] = ("endpoint_tbr", "endpoint_cac")
    time_patches: int = 3
    max_events: int = 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureSchema":
        return cls(
            static_features=tuple(value["static_features"]),
            longitudinal_features=tuple(value["longitudinal_features"]),
            treatment_features=tuple(value["treatment_features"]),
            target_features=tuple(value.get("target_features", ("endpoint_tbr", "endpoint_cac"))),
            time_patches=int(value.get("time_patches", 3)),
            max_events=int(value.get("max_events", 48)),
        )


@dataclass
class LongitudinalBatch:
    static: torch.Tensor
    baseline: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    delta: torch.Tensor
    times: torch.Tensor
    treatments: torch.Tensor
    targets: torch.Tensor
    irregular_values: torch.Tensor
    irregular_mask: torch.Tensor
    irregular_times: torch.Tensor
    irregular_treatments: torch.Tensor
    patient_ids: list[str]

    def to(self, device: torch.device | str) -> "LongitudinalBatch":
        return LongitudinalBatch(
            static=self.static.to(device),
            baseline=self.baseline.to(device),
            values=self.values.to(device),
            mask=self.mask.to(device),
            delta=self.delta.to(device),
            times=self.times.to(device),
            treatments=self.treatments.to(device),
            targets=self.targets.to(device),
            irregular_values=self.irregular_values.to(device),
            irregular_mask=self.irregular_mask.to(device),
            irregular_times=self.irregular_times.to(device),
            irregular_treatments=self.irregular_treatments.to(device),
            patient_ids=self.patient_ids,
        )

    @property
    def size(self) -> int:
        return int(self.static.shape[0])


def save_npz(path: str, arrays: dict[str, np.ndarray], schema: FeatureSchema) -> None:
    np.savez_compressed(path, **arrays, schema_json=np.asarray([str(schema.to_dict())], dtype=object))
