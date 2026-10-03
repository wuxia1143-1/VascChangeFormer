from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


FEATURE_NAMES = (
    "baseline_tbr",
    "baseline_log_cac",
    "pred_delta_tbr",
    "pred_delta_log_cac",
    "followup_years",
    "observation_density",
    "adapter_gate_inflammation",
    "adapter_gate_calcification",
    "reliability_gate",
    "decision_gate_cac",
)


def _aligned_raw_indices(
    frame: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> np.ndarray:
    lookup = {
        str(patient_id): index
        for index, patient_id in enumerate(arrays["patient_ids"])
    }
    identifiers = frame["patient_id"].astype(str).tolist()
    missing = [value for value in identifiers if value not in lookup]
    if missing:
        raise ValueError(f"Prediction patients missing from arrays: {missing[:3]}")
    return np.asarray([lookup[value] for value in identifiers], dtype=int)


def uncertainty_features(
    frame: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> np.ndarray:
    indices = _aligned_raw_indices(frame, arrays)
    baseline = np.asarray(arrays["baseline"])[indices].astype(float)
    followup = np.asarray(arrays["followup_months"])[indices].astype(float) / 12.0
    density = np.asarray(arrays["mask"])[indices].astype(float).mean(axis=(1, 2))

    def column(*names: str) -> np.ndarray:
        for name in names:
            if name in frame:
                return frame[name].to_numpy(float)
        return np.zeros(len(frame), dtype=float)

    values = np.column_stack(
        [
            baseline[:, 0],
            np.log1p(np.maximum(baseline[:, 1], 0.0)),
            column("pred_delta_tbr"),
            column("pred_delta_log_cac"),
            np.nan_to_num(followup, nan=float(np.nanmedian(followup))),
            density,
            column("adapter_gate_inflammation"),
            column("adapter_gate_calcification"),
            column("reliability_gate", "coupling_gate"),
            column("decision_gate_cac"),
        ]
    )
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0)


def _scale_regressor(options: dict[str, Any], seed: int):
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        n_estimators=int(options.get("n_estimators", 100)),
        max_depth=int(options.get("max_depth", 2)),
        learning_rate=float(options.get("learning_rate", 0.03)),
        min_samples_leaf=int(options.get("min_samples_leaf", 20)),
        random_state=int(seed),
    )


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, float))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Conformal alpha must be strictly between zero and one")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be non-empty and finite")
    rank = min(len(values), int(math.ceil((len(values) + 1) * (1.0 - alpha))))
    return float(values[max(0, rank - 1)])


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DetachedConformalSidecar:
    """Cross-fitted heteroscedastic scale model that never changes the mean."""

    alpha: float
    scale_floor: float
    feature_names: tuple[str, ...] = FEATURE_NAMES
    models: dict[str, GradientBoostingRegressor] = field(default_factory=dict)
    conformal_quantiles: dict[str, float] = field(default_factory=dict)
    scale_clip: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _predict_scale(self, task: str, features: np.ndarray) -> np.ndarray:
        if task not in self.models:
            raise RuntimeError(f"Uncertainty sidecar is not fitted for {task}")
        raw = np.exp(self.models[task].predict(features))
        low, high = self.scale_clip[task]
        return np.clip(raw, low, high)

    def apply(
        self, frame: pd.DataFrame, arrays: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        result = frame.copy()
        before = result[["pred_delta_tbr", "pred_delta_log_cac"]].to_numpy(copy=True)
        features = uncertainty_features(result, arrays)
        baseline = result[["baseline_tbr", "baseline_cac"]].to_numpy(float)
        for task, prediction_column in (
            ("tbr", "pred_delta_tbr"),
            ("cac", "pred_delta_log_cac"),
        ):
            scale = self._predict_scale(task, features)
            half_width = self.conformal_quantiles[task] * scale
            prediction = result[prediction_column].to_numpy(float)
            lower = prediction - half_width
            upper = prediction + half_width
            result[f"{task}_laplace_scale"] = scale
            result[f"{task}_pi_lower_delta"] = lower
            result[f"{task}_pi_upper_delta"] = upper
            if task == "tbr":
                result["tbr_pi_lower"] = baseline[:, 0] + lower
                result["tbr_pi_upper"] = baseline[:, 0] + upper
            else:
                base_log = np.log1p(np.maximum(baseline[:, 1], 0.0))
                result["cac_pi_lower"] = np.maximum(0.0, np.expm1(base_log + lower))
                result["cac_pi_upper"] = np.maximum(0.0, np.expm1(base_log + upper))
        after = result[["pred_delta_tbr", "pred_delta_log_cac"]].to_numpy()
        if not np.array_equal(before, after):
            raise RuntimeError("Detached uncertainty sidecar changed a point prediction")
        return result


def save_detached_conformal_sidecar(
    sidecar: DetachedConformalSidecar,
    path: str | Path,
    source_oof_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the post-fit sidecar and return a hash-locked manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(sidecar, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    manifest: dict[str, Any] = {
        "artifact": destination.name,
        "artifact_sha256": _sha256(destination),
        "serialization": "python_pickle_trusted_local_artifact",
        "class": "DetachedConformalSidecar",
        "alpha": float(sidecar.alpha),
        "scale_floor": float(sidecar.scale_floor),
        "feature_names": list(sidecar.feature_names),
        "tasks": sorted(sidecar.models),
        "point_prediction_mutation_allowed": False,
    }
    if source_oof_path is not None:
        manifest["source_oof"] = Path(source_oof_path).name
        manifest["source_oof_sha256"] = _sha256(source_oof_path)
    manifest_path = destination.with_name(
        destination.stem + "_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def fit_detached_conformal_sidecar(
    oof: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    options: dict[str, Any] | None = None,
    seed: int = 2026,
) -> tuple[DetachedConformalSidecar, pd.DataFrame, dict[str, Any]]:
    options = dict(options or {})
    required = {
        "patient_id",
        "outer_fold",
        "true_delta_tbr",
        "pred_delta_tbr",
        "true_delta_log_cac",
        "pred_delta_log_cac",
    }
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF frame missing uncertainty columns: {sorted(missing)}")
    if len(oof) != oof["patient_id"].nunique():
        raise ValueError("Uncertainty fitting requires one OOF row per patient")
    folds = oof["outer_fold"].to_numpy(int)
    if sorted(np.unique(folds).tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("Uncertainty fitting requires five complete outer folds")
    features = uncertainty_features(oof, arrays)
    alpha = float(options.get("alpha", 0.05))
    floor = float(options.get("scale_floor", 1e-3))
    if not 0.0 < alpha < 1.0:
        raise ValueError("Uncertainty alpha must be strictly between zero and one")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("Uncertainty scale_floor must be positive and finite")
    sidecar = DetachedConformalSidecar(alpha=alpha, scale_floor=floor)
    cross_fitted_scales: dict[str, np.ndarray] = {}
    task_records: dict[str, Any] = {}
    for task, truth_column, prediction_column in (
        ("tbr", "true_delta_tbr", "pred_delta_tbr"),
        ("cac", "true_delta_log_cac", "pred_delta_log_cac"),
    ):
        error = np.abs(
            oof[truth_column].to_numpy(float)
            - oof[prediction_column].to_numpy(float)
        )
        log_scale_target = np.log(np.maximum(error, floor))
        scale = np.empty(len(oof), dtype=float)
        for fold in range(1, 6):
            train = folds != fold
            validation = folds == fold
            model = _scale_regressor(options, seed + 100 * fold + (0 if task == "tbr" else 50))
            model.fit(features[train], log_scale_target[train])
            scale[validation] = np.exp(model.predict(features[validation]))
        clip = (
            max(floor, float(np.quantile(scale, 0.01))),
            max(floor * 2, float(np.quantile(scale, 0.99))),
        )
        scale = np.clip(scale, *clip)
        scores = error / np.maximum(scale, floor)
        quantile = _finite_sample_quantile(scores, alpha)
        final = _scale_regressor(options, seed + (0 if task == "tbr" else 50))
        final.fit(features, log_scale_target)
        sidecar.models[task] = final
        sidecar.conformal_quantiles[task] = quantile
        sidecar.scale_clip[task] = clip
        cross_fitted_scales[task] = scale
        lower = oof[prediction_column].to_numpy(float) - quantile * scale
        upper = oof[prediction_column].to_numpy(float) + quantile * scale
        covered = (
            (oof[truth_column].to_numpy(float) >= lower)
            & (oof[truth_column].to_numpy(float) <= upper)
        )
        association = float(
            pd.Series(scale).corr(pd.Series(error), method="spearman")
        )
        task_records[task] = {
            "coverage": float(covered.mean()),
            "mean_delta_interval_width": float(np.mean(upper - lower)),
            "median_delta_interval_width": float(np.median(upper - lower)),
            "conformal_quantile": quantile,
            "scale_error_spearman": (
                association if np.isfinite(association) else None
            ),
            "scale_clip": list(clip),
        }

    internal = oof.copy()
    point_before = internal[["pred_delta_tbr", "pred_delta_log_cac"]].to_numpy(copy=True)
    for task, truth_column, prediction_column in (
        ("tbr", "true_delta_tbr", "pred_delta_tbr"),
        ("cac", "true_delta_log_cac", "pred_delta_log_cac"),
    ):
        scale = cross_fitted_scales[task]
        half = sidecar.conformal_quantiles[task] * scale
        internal[f"{task}_laplace_scale"] = scale
        internal[f"{task}_pi_lower_delta"] = internal[prediction_column] - half
        internal[f"{task}_pi_upper_delta"] = internal[prediction_column] + half
        if task == "tbr" and "baseline_tbr" in internal:
            internal["tbr_pi_lower"] = (
                internal["baseline_tbr"] + internal["tbr_pi_lower_delta"]
            )
            internal["tbr_pi_upper"] = (
                internal["baseline_tbr"] + internal["tbr_pi_upper_delta"]
            )
        elif task == "cac" and "baseline_cac" in internal:
            base_log = np.log1p(
                np.maximum(internal["baseline_cac"].to_numpy(float), 0.0)
            )
            internal["cac_pi_lower"] = np.maximum(
                0.0,
                np.expm1(base_log + internal["cac_pi_lower_delta"]),
            )
            internal["cac_pi_upper"] = np.maximum(
                0.0,
                np.expm1(base_log + internal["cac_pi_upper_delta"]),
            )
    point_after = internal[["pred_delta_tbr", "pred_delta_log_cac"]].to_numpy()
    identity = bool(np.array_equal(point_before, point_after))
    if not identity:
        raise RuntimeError("Uncertainty fitting changed OOF point predictions")
    summary = {
        "method": "detached_cross_fitted_laplace_scale_normalized_conformal",
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "patient_count": int(len(oof)),
        "feature_names": list(FEATURE_NAMES),
        "point_predictions_bitwise_identical": identity,
        "external_labels_used": False,
        "tasks": task_records,
    }
    return sidecar, internal, summary
