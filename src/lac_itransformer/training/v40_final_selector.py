from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


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
    threshold = float(np.quantile(np.abs(target), 0.90))
    subset = np.abs(target) >= threshold
    return float(np.sqrt(np.square(prediction[subset] - target[subset]).mean()))


def _calibration_expert(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=15,
        random_state=int(seed),
    )


class V40FinalSelector:
    """Inner-OOF-only joint calibration and protected-tail selector.

    A cross-fitted robust expert calibrates the direct central prediction.  A
    separate neural hurdle supplies only a residual correction.  The two
    weights are selected jointly under per-metric and tail non-inferiority, so
    either module can be skipped without changing the base neural prediction.
    """

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
        self.calibration_weight = 0.0
        self.tail_weight = 0.0
        self.expert: GradientBoostingRegressor | None = None
        self.record: dict[str, Any] = {}

    def fit(self, heads, targets, features, meta_fold):
        central = np.asarray(heads["cac"][:, 0], float)
        corrected = np.asarray(heads["cac"][:, 1], float)
        target = np.asarray(targets["cac"], float)
        inputs = np.asarray(features["cac"], float)
        folds = np.asarray(meta_fold, int)
        if sorted(np.unique(folds).tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError("V4.0-final selection requires five complete inner OOF folds")

        expert_oof = np.empty_like(target)
        for fold in range(1, 6):
            train = folds != fold
            validation = folds == fold
            expert = _calibration_expert(self.seed + fold)
            expert.fit(inputs[train], target[train])
            expert_oof[validation] = expert.predict(inputs[validation])

        reference = _metrics(target, central)
        reference_tail = _tail_rmse(target, central)
        candidates = []
        tolerance = 1.0 + self.degradation_tolerance
        for calibration_weight in self.candidate_weights:
            calibrated = (
                (1.0 - calibration_weight) * central
                + calibration_weight * expert_oof
            )
            for tail_weight in self.candidate_weights:
                prediction = calibrated + tail_weight * (corrected - central)
                metrics = _metrics(target, prediction)
                tail = _tail_rmse(target, prediction)
                consistent = 0
                for fold in range(1, 6):
                    subset = folds == fold
                    fold_reference = _metrics(target[subset], central[subset])
                    fold_candidate = _metrics(target[subset], prediction[subset])
                    fold_score = (
                        fold_candidate["mae"] / max(fold_reference["mae"], 1e-8)
                        + fold_candidate["rmse"] / max(fold_reference["rmse"], 1e-8)
                    )
                    consistent += int(fold_score <= 2.0)
                score = (
                    metrics["mae"] / max(reference["mae"], 1e-8)
                    + metrics["rmse"] / max(reference["rmse"], 1e-8)
                    + 0.25 * tail / max(reference_tail, 1e-8)
                )
                admissible = bool(
                    metrics["mae"] <= reference["mae"] * tolerance
                    and metrics["rmse"] <= reference["rmse"] * tolerance
                    and tail <= reference_tail * tolerance
                    and consistent >= self.required_consistent_folds
                )
                candidates.append(
                    {
                        "calibration_weight": float(calibration_weight),
                        "tail_weight": float(tail_weight),
                        "metrics": metrics,
                        "tail_rmse": float(tail),
                        "consistent_folds": int(consistent),
                        "score": float(score),
                        "admissible": admissible,
                    }
                )

        admissible = [candidate for candidate in candidates if candidate["admissible"]]
        selected = min(
            admissible or [candidates[0]],
            key=lambda candidate: (
                candidate["score"],
                candidate["tail_weight"],
                candidate["calibration_weight"],
            ),
        )
        self.calibration_weight = float(selected["calibration_weight"])
        self.tail_weight = float(selected["tail_weight"])
        self.expert = _calibration_expert(self.seed + 99)
        self.expert.fit(inputs, target)
        self.record = {
            "selected_method": "cross_fitted_calibration_plus_protected_tail",
            "selected_calibration_weight": self.calibration_weight,
            "selected_tail_weight": self.tail_weight,
            "reference_metrics": reference,
            "reference_tail_rmse": reference_tail,
            "selected_metrics": selected["metrics"],
            "selected_tail_rmse": selected["tail_rmse"],
            "selected_consistent_folds": selected["consistent_folds"],
            "selected_admissible": bool(selected["admissible"]),
            "candidates": candidates,
        }
        return self

    def predict(self, heads, features, modes=None):
        if self.expert is None:
            raise RuntimeError("V4.0-final selector has not been fitted")
        modes = dict(modes or {})
        tbr = np.asarray(heads["tbr"][:, 0], float)
        central = np.asarray(heads["cac"][:, 0], float)
        corrected = np.asarray(heads["cac"][:, 1], float)
        expert = self.expert.predict(np.asarray(features["cac"], float))
        mode = modes.get("cac", "full")
        if mode in {"central", "identity", "raw"}:
            calibration_weight, tail_weight = 0.0, 0.0
        elif mode == "no_calibration":
            calibration_weight, tail_weight = 0.0, self.tail_weight
        elif mode in {"no_tail", "calibration_only"}:
            calibration_weight, tail_weight = self.calibration_weight, 0.0
        elif mode == "expert":
            calibration_weight, tail_weight = 1.0, 0.0
        elif mode == "tail_only":
            calibration_weight, tail_weight = 0.0, 1.0
        else:
            calibration_weight = self.calibration_weight
            tail_weight = self.tail_weight
        calibrated = (
            (1.0 - calibration_weight) * central
            + calibration_weight * expert
        )
        cac = calibrated + tail_weight * (corrected - central)
        return np.column_stack([tbr, cac]), {
            "tbr": np.zeros_like(tbr),
            "cac": np.full_like(cac, tail_weight),
        }

    def audit(self):
        return {
            "selection_objective": (
                "inner-OOF normalized CAC MAE+RMSE plus 0.25*tail RMSE, "
                "with per-metric and tail non-inferiority"
            ),
            "candidate_weights": list(self.candidate_weights),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "unique_fixed_configuration": {
                "tbr": {"method": "identity", "weight": 0.0},
                "cac": {
                    "method": "cross_fitted_calibration_plus_protected_tail",
                    "calibration_weight": self.calibration_weight,
                    "tail_weight": self.tail_weight,
                },
            },
            "tasks": {
                "tbr": {"selected_method": "identity"},
                "cac": self.record,
            },
            "all_constraints_passed": bool(
                self.record.get("selected_admissible", False)
            ),
            "outer_test_labels_used": False,
        }
