from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    scale = q75 - q25
    std = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    median = np.where(np.isfinite(median), median, 0.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


@dataclass
class FoldPreprocessor:
    """Train-fold-only robust preprocessing state.

    Never call ``fit`` on external validation data. The serialized state is part
    of the locked model artifact and is reused unchanged at external evaluation.
    """

    static_median: list[float]
    static_scale: list[float]
    value_median: list[float]
    value_scale: list[float]
    fitted_patient_ids_hash: str = ""

    @classmethod
    def fit(cls, arrays: dict[str, np.ndarray], patient_ids_hash: str = "") -> "FoldPreprocessor":
        static = arrays["static"].astype(float)
        values = arrays["values"].astype(float).copy()
        mask = arrays["mask"].astype(bool)
        values[~mask] = np.nan
        sm, ss = _robust_location_scale(static)
        vm, vs = _robust_location_scale(values.reshape(-1, values.shape[-1]))
        return cls(sm.tolist(), ss.tolist(), vm.tolist(), vs.tolist(), patient_ids_hash)

    def transform(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {key: np.asarray(value).copy() for key, value in arrays.items()}
        sm, ss = np.asarray(self.static_median), np.asarray(self.static_scale)
        vm, vs = np.asarray(self.value_median), np.asarray(self.value_scale)
        out["static"] = np.nan_to_num((out["static"] - sm) / ss).astype(np.float32)
        scaled = (out["values"] - vm) / vs
        scaled = np.where(out["mask"].astype(bool), scaled, 0.0)
        out["values"] = np.nan_to_num(scaled, nan=0.0, posinf=8.0, neginf=-8.0).clip(-8, 8).astype(np.float32)
        if "irregular_values" in out:
            irregular_scaled = (out["irregular_values"] - vm) / vs
            irregular_scaled = np.where(out["irregular_mask"].astype(bool), irregular_scaled, 0.0)
            out["irregular_values"] = np.nan_to_num(
                irregular_scaled, nan=0.0, posinf=8.0, neginf=-8.0
            ).clip(-8, 8).astype(np.float32)
        out["delta"] = np.nan_to_num(out["delta"] / 365.0, nan=1.0).clip(0, 10).astype(np.float32)
        for key in (
            "mask",
            "times",
            "treatments",
            "baseline",
            "targets",
            "followup_months",
        ):
            if key not in out:
                continue
            out[key] = out[key].astype(np.float32)
        for key in ("irregular_mask", "irregular_times", "irregular_treatments"):
            if key in out:
                out[key] = out[key].astype(np.float32)
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FoldPreprocessor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def subset_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, Any] = {}
    n = len(arrays["static"])
    for key, value in arrays.items():
        array = np.asarray(value)
        result[key] = array[indices] if array.ndim and len(array) == n else array
    return result
