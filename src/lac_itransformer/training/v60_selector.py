from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .v40_final_selector import V40FinalSelector


class V60PrunedTailSelector(V40FinalSelector):
    """Selected V6-A decision layer with no reachable direct tail addition."""

    def predict(self, heads, features, modes=None):
        del modes
        return super().predict(heads, features, {"tbr": "identity", "cac": "no_tail"})

    def audit(self):
        result = super().audit()
        result["architecture_version"] = "V6.0-selected"
        result["explicit_i_to_c_or_tail_residual_in_final_prediction"] = False
        result["diagnostic_features_retained_for_robust_calibration"] = True
        result["unique_fixed_configuration"]["cac"]["tail_weight"] = 0.0
        return result


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / max(denominator, 1e-12)),
    }


def _tail_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    cutoff = float(np.quantile(np.abs(target), 0.90))
    subset = np.abs(target) >= cutoff
    return float(np.sqrt(np.square(prediction[subset] - target[subset]).mean()))


def mechanism_free_features(
    features: np.ndarray,
    generic_feature_count: int,
) -> np.ndarray:
    """Keep ordinary covariates and optional task-agnostic temporal context.

    The audited feature builder places six mechanism/tail-derived fields just
    before the calcification adapter gate.  Generic V6-C features are appended
    after that gate, so the base block is separated before applying the locked
    V4.2 exclusion rule.
    """

    values = np.asarray(features, float)
    generic_feature_count = int(generic_feature_count)
    if generic_feature_count < 0 or generic_feature_count >= values.shape[1]:
        raise ValueError("Invalid V6 generic temporal feature count")
    if generic_feature_count:
        base = values[:, :-generic_feature_count]
        generic = values[:, -generic_feature_count:]
    else:
        base = values
        generic = np.empty((len(values), 0), dtype=float)
    if base.ndim != 2 or base.shape[1] < 10:
        raise ValueError("V6 CAC decision features are incomplete")
    safe = np.column_stack([base[:, 0], base[:, 3:-6], base[:, -1]])
    return np.column_stack([safe, generic])


def _regressor(seed: int, min_samples_leaf: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
    )


class V60GenericDecisionLayer:
    """OOF-only robust CAC calibration using no mechanism-derived feature."""

    configured_generic_feature_count = 0
    configured_min_samples_leaf = 20

    def __init__(
        self,
        candidate_weights=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
        ridge_alphas=None,
        degradation_tolerance=0.005,
        required_consistent_folds=3,
        seed=2026,
    ):
        del ridge_alphas
        self.candidate_weights = tuple(float(value) for value in candidate_weights)
        self.degradation_tolerance = float(degradation_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.seed = int(seed)
        self.generic_feature_count = int(self.configured_generic_feature_count)
        self.min_samples_leaf = int(self.configured_min_samples_leaf)
        self.calibration_weight = 0.0
        self.expert: GradientBoostingRegressor | None = None
        self.record: dict[str, Any] = {}

    def fit(self, heads, targets, features, meta_fold):
        central = np.asarray(heads["cac"][:, 0], float)
        target = np.asarray(targets["cac"], float)
        inputs = mechanism_free_features(
            features["cac"], self.generic_feature_count
        )
        folds = np.asarray(meta_fold, int)
        if sorted(np.unique(folds).tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError("V6 calibration requires five complete inner OOF folds")

        expert_oof = np.empty_like(target)
        for fold in range(1, 6):
            train = folds != fold
            validation = folds == fold
            expert = _regressor(self.seed + fold, self.min_samples_leaf)
            expert.fit(inputs[train], target[train])
            expert_oof[validation] = expert.predict(inputs[validation])

        reference = _metrics(target, central)
        reference_tail = _tail_rmse(target, central)
        tolerance = 1.0 + self.degradation_tolerance
        candidates = []
        for weight in self.candidate_weights:
            prediction = (1.0 - weight) * central + weight * expert_oof
            metrics = _metrics(target, prediction)
            tail = _tail_rmse(target, prediction)
            consistent = 0
            for fold in range(1, 6):
                subset = folds == fold
                fold_reference = _metrics(target[subset], central[subset])
                fold_candidate = _metrics(target[subset], prediction[subset])
                fold_score = (
                    fold_candidate["mae"] / max(fold_reference["mae"], 1e-8)
                    + fold_candidate["rmse"]
                    / max(fold_reference["rmse"], 1e-8)
                )
                consistent += int(fold_score <= 2.0)
            admissible = bool(
                metrics["mae"] <= reference["mae"] * tolerance
                and metrics["rmse"] <= reference["rmse"] * tolerance
                and tail <= reference_tail * tolerance
                and consistent >= self.required_consistent_folds
            )
            candidates.append(
                {
                    "calibration_weight": float(weight),
                    "metrics": metrics,
                    "tail_rmse": tail,
                    "consistent_folds": int(consistent),
                    "admissible": admissible,
                    "score": float(
                        metrics["mae"] / max(reference["mae"], 1e-8)
                        + metrics["rmse"] / max(reference["rmse"], 1e-8)
                        + 0.25 * tail / max(reference_tail, 1e-8)
                    ),
                }
            )
        selected = min(
            [item for item in candidates if item["admissible"]]
            or [candidates[0]],
            key=lambda item: (item["score"], item["calibration_weight"]),
        )
        self.calibration_weight = float(selected["calibration_weight"])
        self.expert = _regressor(self.seed + 99, self.min_samples_leaf)
        self.expert.fit(inputs, target)
        self.record = {
            "selected_method": "mechanism_free_oof_robust_calibration",
            "generic_temporal_feature_count": self.generic_feature_count,
            "safe_feature_count": int(inputs.shape[1]),
            "reference_metrics": reference,
            "reference_tail_rmse": reference_tail,
            "selected": selected,
            "candidates": candidates,
        }
        return self

    def predict(self, heads, features, modes=None):
        if self.expert is None:
            raise RuntimeError("V6 decision layer has not been fitted")
        modes = dict(modes or {})
        tbr = np.asarray(heads["tbr"][:, 0], float)
        central = np.asarray(heads["cac"][:, 0], float)
        inputs = mechanism_free_features(
            features["cac"], self.generic_feature_count
        )
        mode = modes.get("cac", "full")
        weight = 0.0 if mode in {"central", "identity", "raw"} else self.calibration_weight
        cac = (1.0 - weight) * central + weight * self.expert.predict(inputs)
        return np.column_stack([tbr, cac]), {
            "tbr": np.zeros_like(tbr),
            "cac": np.full_like(cac, weight),
        }

    def audit(self):
        return {
            "selection_scope": "five complete inner OOF folds only",
            "feature_policy": {
                "included": [
                    "central_cac_prediction",
                    "baseline_log_cac",
                    "baseline_tbr",
                    "followup_duration_and_window",
                    "baseline_cac_positive",
                    "observation_density",
                    "treatment",
                    "calcification_soft_adapter_gate",
                    "task_agnostic_temporal_statistics",
                ],
                "excluded": [
                    "corrected_tail_head",
                    "head_disagreement",
                    "progression_probability",
                    "progression_magnitude",
                    "coupling_gates",
                    "lag_diagnostics",
                ],
            },
            "generic_temporal_feature_count": self.generic_feature_count,
            "candidate_weights": list(self.candidate_weights),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "unique_fixed_configuration": {
                "tbr": {"method": "identity", "weight": 0.0},
                "cac": {
                    "method": "mechanism_free_oof_robust_calibration",
                    "calibration_weight": self.calibration_weight,
                },
            },
            "tasks": {
                "tbr": {"selected_method": "identity"},
                "cac": self.record,
            },
            "all_constraints_passed": bool(
                self.record.get("selected", {}).get("admissible", False)
            ),
            "outer_test_labels_used": False,
            "external_labels_used": False,
        }
