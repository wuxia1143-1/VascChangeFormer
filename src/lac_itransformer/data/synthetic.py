from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import FeatureSchema


PROFILE_KEYS = {
    "BMI": "BMI",
    "blood_pressure": "血压",
    "glucose": "葡萄糖",
    "cholesterol": "胆固醇",
    "ldl": "低密度脂蛋白胆固醇",
    "triglyceride": "甘油三酯",
    "hdl": "高密度脂蛋白胆固醇",
    "d_dimer": "D-二聚体",
    "platelet": "血小板",
    "neutrophil": "中性粒细胞",
    "lymphocyte": "淋巴细胞",
    "nlr": "NLR",
    "crp": "CRP",
    "il6": "IL-6",
    "egfr": "肾小球滤过率",
    "creatinine": "肌酐",
}


class SyntheticCohortGenerator:
    """Generate unit-test data from aggregate external-cohort distributions.

    The mechanism intentionally contains stronger lagged inflammation-to-
    calcification coupling and weaker calcification-to-inflammation coupling.
    This is a simulation property, not a clinical conclusion.
    """

    def __init__(self, profile: dict[str, Any], schema: FeatureSchema | None = None, seed: int = 2026):
        if profile.get("privacy", {}).get("contains_patient_identifiers", True):
            raise ValueError("Synthetic generation requires an aggregate, identifier-free profile")
        self.profile = profile
        self.schema = schema or FeatureSchema()
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    @classmethod
    def from_json(cls, path: str | Path, seed: int = 2026) -> "SyntheticCohortGenerator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), seed=seed)

    def _truncated(self, median: float, p05: float, p95: float, size: tuple[int, ...]) -> np.ndarray:
        sd = max((p95 - p05) / 3.29, 1e-4)
        return self.rng.normal(median, sd, size=size).clip(p05, p95)

    def generate(self, n_patients: int = 443) -> dict[str, np.ndarray]:
        n, p = n_patients, self.schema.time_patches
        d, a = len(self.schema.longitudinal_features), len(self.schema.treatment_features)
        age = self._truncated(63.0, 42.0, 76.2, (n,))
        sex = self.rng.binomial(1, 0.686, n)
        weight = self._truncated(68.0, 52.0, 89.0, (n,))
        smoking = np.where(self.rng.random(n) < 0.5, self._truncated(30, 0, 50, (n,)), 0)
        drinking = np.where(self.rng.random(n) < 0.35, self._truncated(30, 0, 45, (n,)), 0)
        pathology = self.rng.choice(np.arange(8), n, p=[0.08, 0.48, 0.28, 0.04, 0.07, 0.02, 0.01, 0.02])
        static = np.column_stack([age, sex, weight, smoking, drinking, pathology]).astype(np.float32)

        bt = self.profile["baseline_targets"]
        baseline_tbr = self._truncated(bt["baseline_tbr"]["median"], bt["baseline_tbr"]["p05"], bt["baseline_tbr"]["p95"], (n,))
        cac_positive = self.rng.lognormal(np.log(180.0), 1.55, n).clip(0, bt["baseline_cac"]["p95"])
        baseline_cac = np.where(self.rng.random(n) < 0.34, 0.0, cac_positive)
        baseline = np.column_stack([baseline_tbr, baseline_cac]).astype(np.float32)

        base_times = np.linspace(0.12, 0.88, p)
        times = np.sort((base_times + self.rng.normal(0, 0.035, (n, p))).clip(0, 1), axis=1).astype(np.float32)
        rates = self.profile["treatment_exposure_observed_rates"]
        rate_order = [rates[k]["observed_rate"] for k in ("化疗", "放疗", "手术治疗", "免疫治疗", "靶向治疗")]
        treatments = np.zeros((n, p, a), dtype=np.float32)
        for j, rate in enumerate(rate_order):
            patient_probability = min(0.85, max(0.08, rate * 2.8))
            active = self.rng.random(n) < patient_probability
            start = self.rng.integers(0, p, n)
            for i in range(n):
                if active[i]:
                    treatments[i, start[i] :, j] = 1.0
        treatments[:, 1:, 2] = 0.0  # surgery is treated as a discrete early exposure

        values = np.zeros((n, p, d), dtype=np.float32)
        mask = np.zeros_like(values)
        delta = np.zeros_like(values)
        latent_i = np.zeros((n, p), dtype=np.float32)
        latent_c = np.zeros((n, p), dtype=np.float32)
        i_prev = 0.75 * (baseline_tbr - 1.6) / 0.35 + self.rng.normal(0, 0.35, n)
        c_prev = np.log1p(baseline_cac) / 7.0

        long_profile = self.profile.get("longitudinal_numeric", {})
        treatment_effect_i = np.asarray([0.20, 0.16, 0.05, 0.24, 0.08])
        treatment_effect_c = np.asarray([0.07, 0.12, 0.02, 0.04, 0.04])
        for step in range(p):
            tx = treatments[:, step]
            reverse = 0.035 * np.tanh(c_prev - 0.5)
            i_state = 0.62 * i_prev + tx @ treatment_effect_i + reverse + self.rng.normal(0, 0.32, n)
            c_state = np.maximum(0, c_prev + 0.18 * np.maximum(i_prev, 0) + tx @ treatment_effect_c + 0.015 * (age - 60) / 10 + self.rng.normal(0, 0.09, n))
            latent_i[:, step], latent_c[:, step] = i_state, c_state

            for j, feature in enumerate(self.schema.longitudinal_features):
                source = long_profile.get(PROFILE_KEYS[feature], {})
                median = float(source.get("median", 1.0))
                p05, p95 = float(source.get("p05", median * 0.6)), float(source.get("p95", median * 1.6 + 1e-3))
                base = self._truncated(median, p05, p95, (n,))
                if feature in {"crp", "il6", "nlr", "neutrophil", "d_dimer"}:
                    base *= np.exp(0.13 * i_state).clip(0.55, 2.5)
                elif feature in {"creatinine"}:
                    base *= (1 + 0.04 * tx[:, 0])
                elif feature == "egfr":
                    base *= (1 - 0.04 * tx[:, 0])
                elif feature in {"glucose", "triglyceride", "ldl"}:
                    base *= (1 + 0.025 * c_state)
                values[:, step, j] = base
                missing_rate = float(source.get("missing_rate", 0.45))
                observed = self.rng.random(n) > np.clip(missing_rate, 0.05, 0.93)
                mask[:, step, j] = observed
                if step == 0:
                    delta[:, step, j] = times[:, step] * 365
                else:
                    delta[:, step, j] = (times[:, step] - times[:, step - 1]) * 365
            i_prev, c_prev = i_state, c_state

        # Preserve the mathematical NLR relationship when its components exist.
        idx = {name: i for i, name in enumerate(self.schema.longitudinal_features)}
        values[:, :, idx["nlr"]] = values[:, :, idx["neutrophil"]] / np.maximum(values[:, :, idx["lymphocyte"]], 0.05)
        mask[:, :, idx["nlr"]] *= mask[:, :, idx["neutrophil"]] * mask[:, :, idx["lymphocyte"]]

        tbr_delta = 0.05 + 0.16 * latent_i[:, -1] + 0.035 * latent_c[:, -1] + self.rng.normal(0, 0.16, n)
        log_cac_delta = np.maximum(-0.12, 0.055 + 0.20 * np.maximum(latent_i[:, -2], 0) + 0.12 * latent_c[:, -1] + self.rng.normal(0, 0.10, n))
        endpoint_tbr = np.maximum(0.8, baseline_tbr + tbr_delta)
        endpoint_cac = np.maximum(0.0, np.expm1(np.log1p(baseline_cac) + log_cac_delta))
        targets = np.column_stack([endpoint_tbr, endpoint_cac]).astype(np.float32)

        # Padded event-level representation for irregular-time comparators such
        # as APN.  LAC-iTransformer continues to consume the prespecified patches.
        event_length = self.schema.max_events
        irregular_values = np.zeros((n, event_length, d), dtype=np.float32)
        irregular_mask = np.zeros_like(irregular_values)
        irregular_times = np.zeros((n, event_length), dtype=np.float32)
        irregular_treatments = np.zeros((n, event_length, a), dtype=np.float32)
        upper_events = min(event_length, max(p + 1, 4 * p))
        for patient in range(n):
            count = int(self.rng.integers(p + 1, upper_events + 1))
            event_times = np.sort(self.rng.uniform(0.02, 0.98, count)).astype(np.float32)
            patch_index = np.minimum(p - 1, (event_times * p).astype(int))
            irregular_times[patient, :count] = event_times
            for visit, patch in enumerate(patch_index):
                noise = self.rng.normal(0, 0.025, d).astype(np.float32)
                irregular_values[patient, visit] = values[patient, patch] * (1.0 + noise)
                irregular_mask[patient, visit] = mask[patient, patch]
                irregular_treatments[patient, visit] = treatments[patient, patch]
        irregular_values *= irregular_mask

        return {
            "static": static,
            "baseline": baseline,
            "values": values,
            "mask": mask,
            "delta": delta.astype(np.float32),
            "times": times,
            "treatments": treatments,
            "targets": targets,
            "irregular_values": irregular_values,
            "irregular_mask": irregular_mask,
            "irregular_times": irregular_times,
            "irregular_treatments": irregular_treatments,
            "latent_inflammation": latent_i,
            "latent_calcification": latent_c,
            "patient_ids": np.asarray([f"SYN-{self.seed}-{i:05d}" for i in range(n)]),
            "schema_json": np.asarray([json.dumps(self.schema.to_dict())]),
        }

    def save(self, path: str | Path, n_patients: int = 443) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **self.generate(n_patients))
        return output


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], FeatureSchema]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files if key != "schema_json"}
        schema = FeatureSchema.from_dict(json.loads(str(loaded["schema_json"][0])))
    return arrays, schema
