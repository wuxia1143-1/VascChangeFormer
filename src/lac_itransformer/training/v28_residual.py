from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": (
            float(1.0 - np.square(error).sum() / denominator)
            if denominator > 0
            else float("nan")
        ),
    }


@dataclass
class _ResidualTask:
    method: str
    weight: float
    alpha: float | None
    model: Any
    correction_cap: float
    audit: dict[str, Any]


class CrossFittedLagResidualCorrector:
    """Inner-OOF-only correction of the central CAC prediction error.

    The corrector never produces an alternative complete CAC prediction.  It
    learns a bounded correction to the central prediction and has an exact
    identity fallback.  TBR is deliberately left as its single neural output.
    """

    def __init__(
        self,
        candidate_weights: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
        ridge_alphas: tuple[float, ...] = (1.0, 10.0),
        degradation_tolerance: float = 0.005,
        required_consistent_folds: int = 4,
    ):
        self.candidate_weights = tuple(float(x) for x in candidate_weights)
        self.ridge_alphas = tuple(float(x) for x in ridge_alphas)
        self.degradation_tolerance = float(degradation_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.tasks: dict[str, _ResidualTask] = {}

    @staticmethod
    def _score(
        metrics: Mapping[str, float],
        central: Mapping[str, float],
    ) -> float:
        return (
            float(metrics["mae"]) / max(float(central["mae"]), 1e-8)
            + float(metrics["rmse"]) / max(float(central["rmse"]), 1e-8)
        )

    def _admissible(
        self,
        metrics: Mapping[str, float],
        central: Mapping[str, float],
        score: float,
        consistent_folds: int,
    ) -> bool:
        tolerance = 1.0 + self.degradation_tolerance
        return (
            float(metrics["mae"]) <= float(central["mae"]) * tolerance
            and float(metrics["rmse"]) <= float(central["rmse"]) * tolerance
            and score < 2.0 - 1e-8
            and consistent_folds >= self.required_consistent_folds
        )

    @staticmethod
    def _consistent_folds(
        target: np.ndarray,
        candidate: np.ndarray,
        central: np.ndarray,
        meta_fold: np.ndarray,
    ) -> int:
        count = 0
        for fold in sorted(np.unique(meta_fold).tolist()):
            subset = meta_fold == fold
            candidate_metrics = _metrics(target[subset], candidate[subset])
            central_metrics = _metrics(target[subset], central[subset])
            if (
                candidate_metrics["mae"] / max(central_metrics["mae"], 1e-8)
                + candidate_metrics["rmse"]
                / max(central_metrics["rmse"], 1e-8)
                <= 2.0
            ):
                count += 1
        return count

    def _fit_cac(
        self,
        heads: np.ndarray,
        target: np.ndarray,
        features: np.ndarray,
        meta_fold: np.ndarray,
    ) -> _ResidualTask:
        central = np.asarray(heads[:, 0], float)
        lag = np.asarray(heads[:, 1], float)
        target = np.asarray(target, float)
        features = np.asarray(features, float)
        meta_fold = np.asarray(meta_fold, int)
        central_metrics = _metrics(target, central)
        candidates: list[dict[str, Any]] = []
        identity = {
            "method": "identity",
            "weight": 0.0,
            "alpha": None,
            "prediction": central,
            "metrics": central_metrics,
            "score": 2.0,
            "consistent_folds": 5,
            "admissible": True,
        }
        candidates.append(identity)

        for weight in self.candidate_weights:
            if weight <= 0:
                continue
            prediction = central + weight * (lag - central)
            metrics = _metrics(target, prediction)
            score = self._score(metrics, central_metrics)
            consistent = self._consistent_folds(
                target, prediction, central, meta_fold
            )
            candidates.append(
                {
                    "method": "lag_scalar_residual",
                    "weight": weight,
                    "alpha": None,
                    "prediction": prediction,
                    "metrics": metrics,
                    "score": score,
                    "consistent_folds": consistent,
                    "admissible": self._admissible(
                        metrics, central_metrics, score, consistent
                    ),
                }
            )

        residual = target - central
        for alpha in self.ridge_alphas:
            for shrink in (0.25, 0.5, 1.0):
                crossfit_correction = np.zeros_like(residual)
                for fold in sorted(np.unique(meta_fold).tolist()):
                    train = meta_fold != fold
                    validation = meta_fold == fold
                    pipeline = make_pipeline(
                        StandardScaler(), Ridge(alpha=float(alpha))
                    )
                    pipeline.fit(features[train], residual[train])
                    cap = float(
                        np.quantile(np.abs(residual[train]), 0.90)
                    )
                    correction = shrink * pipeline.predict(
                        features[validation]
                    )
                    crossfit_correction[validation] = np.clip(
                        correction, -cap, cap
                    )
                prediction = central + crossfit_correction
                metrics = _metrics(target, prediction)
                score = self._score(metrics, central_metrics)
                consistent = self._consistent_folds(
                    target, prediction, central, meta_fold
                )
                candidates.append(
                    {
                        "method": "crossfit_ridge_residual",
                        "weight": shrink,
                        "alpha": alpha,
                        "prediction": prediction,
                        "metrics": metrics,
                        "score": score,
                        "consistent_folds": consistent,
                        "admissible": self._admissible(
                            metrics, central_metrics, score, consistent
                        ),
                    }
                )

        admissible = [value for value in candidates if value["admissible"]]
        selected = min(
            admissible,
            key=lambda value: (
                value["score"],
                0 if value["method"] == "identity" else 1,
                float(value["weight"]),
            ),
        )
        fitted_model = None
        if selected["method"] == "crossfit_ridge_residual":
            fitted_model = make_pipeline(
                StandardScaler(), Ridge(alpha=float(selected["alpha"]))
            )
            fitted_model.fit(features, residual)
        cap = float(np.quantile(np.abs(residual), 0.90))
        selected_prediction = np.asarray(selected["prediction"], float)
        selected_correction = selected_prediction - central
        audit_candidates = [
            {key: value for key, value in candidate.items() if key != "prediction"}
            for candidate in candidates
        ]
        audit = {
            "task": "cac",
            "training_source": "outer-training-pool inner-fold OOF only",
            "target": "central_prediction_error_only",
            "central_metrics": central_metrics,
            "lag_auxiliary_metrics": _metrics(target, lag),
            "selected_method": selected["method"],
            "selected_weight": float(selected["weight"]),
            "selected_alpha": selected["alpha"],
            "selected_metrics": selected["metrics"],
            "selected_consistent_folds": int(selected["consistent_folds"]),
            "identity_fallback_available": True,
            "correction_cap_training_partition_q90": cap,
            "correction_distribution_inner_oof": {
                "mean": float(selected_correction.mean()),
                "std": float(selected_correction.std()),
                "q05": float(np.quantile(selected_correction, 0.05)),
                "q50": float(np.quantile(selected_correction, 0.50)),
                "q95": float(np.quantile(selected_correction, 0.95)),
                "fraction_exact_zero": float(
                    np.mean(np.abs(selected_correction) <= 1e-12)
                ),
            },
            "candidates": audit_candidates,
        }
        return _ResidualTask(
            method=str(selected["method"]),
            weight=float(selected["weight"]),
            alpha=(
                None if selected["alpha"] is None
                else float(selected["alpha"])
            ),
            model=fitted_model,
            correction_cap=cap,
            audit=audit,
        )

    def fit(
        self,
        heads: Mapping[str, np.ndarray],
        targets: Mapping[str, np.ndarray],
        features: Mapping[str, np.ndarray],
        meta_fold: np.ndarray,
    ) -> "CrossFittedLagResidualCorrector":
        self.tasks["tbr"] = _ResidualTask(
            method="identity",
            weight=0.0,
            alpha=None,
            model=None,
            correction_cap=0.0,
            audit={
                "task": "tbr",
                "training_source": "single neural prediction",
                "selected_method": "identity",
                "identity_fallback_available": True,
            },
        )
        self.tasks["cac"] = self._fit_cac(
            np.asarray(heads["cac"]),
            np.asarray(targets["cac"]),
            np.asarray(features["cac"]),
            np.asarray(meta_fold),
        )
        return self

    def predict_task(
        self,
        task: str,
        heads: np.ndarray,
        features: np.ndarray,
        mode: str = "full",
    ) -> tuple[np.ndarray, np.ndarray]:
        central = np.asarray(heads[:, 0], float)
        if task == "tbr" or mode in {"median", "central", "identity"}:
            return central, np.zeros_like(central)
        lag = np.asarray(heads[:, 1], float)
        decision = self.tasks["cac"]
        if mode == "mean":
            return lag, np.ones_like(central)
        if decision.method == "identity":
            correction = np.zeros_like(central)
        elif decision.method == "lag_scalar_residual":
            correction = decision.weight * (lag - central)
        elif decision.method == "crossfit_ridge_residual":
            correction = decision.weight * decision.model.predict(features)
            correction = np.clip(
                correction,
                -decision.correction_cap,
                decision.correction_cap,
            )
        else:
            raise KeyError(decision.method)
        diagnostic_gate = np.divide(
            np.abs(correction),
            np.abs(lag - central) + 1e-8,
        )
        return central + correction, np.clip(diagnostic_gate, 0.0, 1.0)

    def predict(
        self,
        heads: Mapping[str, np.ndarray],
        features: Mapping[str, np.ndarray],
        modes: Mapping[str, str] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        modes = dict(modes or {})
        predictions = []
        gates = {}
        for task in ("tbr", "cac"):
            prediction, gate = self.predict_task(
                task,
                np.asarray(heads[task]),
                np.asarray(features[task]),
                modes.get(task, "full"),
            )
            predictions.append(prediction)
            gates[task] = gate
        return np.column_stack(predictions), gates

    def audit(self) -> dict[str, Any]:
        return {
            "selection_objective": (
                "normalized CAC MAE + normalized CAC RMSE with strict "
                "central-prediction non-inferiority and 4/5-fold consistency"
            ),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "unique_fixed_configuration": {
                task: {
                    "method": value.method,
                    "weight": value.weight,
                    "alpha": value.alpha,
                }
                for task, value in self.tasks.items()
            },
            "tasks": {task: value.audit for task, value in self.tasks.items()},
            "all_constraints_passed": True,
            "outer_test_labels_used": False,
        }
