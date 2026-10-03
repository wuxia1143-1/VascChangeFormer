from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, float) - np.asarray(target, float)
    denominator = float(
        np.square(target - np.mean(target)).sum()
    )
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": (
            float(1.0 - np.square(error).sum() / denominator)
            if denominator > 0
            else float("nan")
        ),
    }


def _oracle_gate(
    target: np.ndarray,
    robust: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    difference = mean - robust
    gate = np.zeros_like(target, dtype=float)
    usable = np.abs(difference) > 1e-8
    gate[usable] = (
        target[usable] - robust[usable]
    ) / difference[usable]
    return np.clip(gate, 0.0, 1.0)


@dataclass
class _TaskDecision:
    method: str
    weight: float | None
    alpha: float | None
    model: Any
    rule: dict[str, Any] | None
    audit: dict[str, Any]


class SafeOOFDecisionLayer:
    """One-prediction decision layer fitted exclusively from inner OOF data."""

    def __init__(
        self,
        candidate_weights: tuple[float, ...] = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ),
        ridge_alphas: tuple[float, ...] = (1.0, 10.0),
        degradation_tolerance: float = 0.02,
        required_consistent_folds: int = 4,
    ):
        self.candidate_weights = tuple(float(x) for x in candidate_weights)
        self.ridge_alphas = tuple(float(x) for x in ridge_alphas)
        self.degradation_tolerance = float(degradation_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.tasks: dict[str, _TaskDecision] = {}

    @staticmethod
    def _score(
        metrics: Mapping[str, float],
        reference_mae: float,
        reference_rmse: float,
    ) -> float:
        return (
            float(metrics["mae"]) / max(reference_mae, 1e-8)
            + float(metrics["rmse"]) / max(reference_rmse, 1e-8)
        )

    def _constraints(
        self,
        metrics: Mapping[str, float],
        best_mae: float,
        best_rmse: float,
    ) -> bool:
        tolerance = 1.0 + self.degradation_tolerance
        return (
            metrics["mae"] <= best_mae * tolerance
            and metrics["rmse"] <= best_rmse * tolerance
        )

    def _fit_task(
        self,
        task: str,
        heads: np.ndarray,
        target: np.ndarray,
        features: np.ndarray,
        meta_fold: np.ndarray,
    ) -> _TaskDecision:
        robust = np.asarray(heads[:, 0], float)
        mean = np.asarray(heads[:, 1], float)
        target = np.asarray(target, float)
        robust_metrics = _metrics(target, robust)
        mean_metrics = _metrics(target, mean)
        best_mae = min(robust_metrics["mae"], mean_metrics["mae"])
        best_rmse = min(robust_metrics["rmse"], mean_metrics["rmse"])
        candidates: list[dict[str, Any]] = []
        for weight in self.candidate_weights:
            prediction = robust + weight * (mean - robust)
            metrics = _metrics(target, prediction)
            candidates.append(
                {
                    "method": "global_convex",
                    "weight": weight,
                    "alpha": None,
                    "metrics": metrics,
                    "score": self._score(metrics, best_mae, best_rmse),
                    "constraints_passed": self._constraints(
                        metrics,
                        best_mae,
                        best_rmse,
                    ),
                    "prediction": prediction,
                }
            )
        admissible_global = [
            value for value in candidates if value["constraints_passed"]
        ]
        selected = min(
            admissible_global or candidates,
            key=lambda value: (value["score"], value["weight"]),
        )

        # Pre-registered, low-dimensional tail gates.  These use only
        # disagreement/uncertainty and clinical state available at inference.
        risk_indices = (
            ((2, "head_disagreement", False), (1, "absolute_mean_head", True))
            if task == "tbr"
            else (
                (2, "head_disagreement", False),
                (1, "absolute_mean_head", True),
                (5, "followup_years", False),
                (19, "magnitude_probe", False),
                (20, "coupling_gate", False),
            )
        )
        for risk_index, risk_name, use_absolute in risk_indices:
            if risk_index >= features.shape[1]:
                continue
            risk = np.asarray(features[:, risk_index], float)
            if use_absolute:
                risk = np.abs(risk)
            for quantile in (0.75, 0.85, 0.90):
                threshold = float(np.quantile(risk, quantile))
                active = risk >= threshold
                for weight in (0.25, 0.50, 0.75, 1.0):
                    gate = active.astype(float) * weight
                    prediction = robust + gate * (mean - robust)
                    metrics = _metrics(target, prediction)
                    consistent = 0
                    for fold in sorted(np.unique(meta_fold).tolist()):
                        subset = meta_fold == fold
                        candidate_fold = _metrics(
                            target[subset],
                            prediction[subset],
                        )
                        robust_fold = _metrics(
                            target[subset],
                            robust[subset],
                        )
                        if (
                            candidate_fold["mae"]
                            + candidate_fold["rmse"]
                            <= robust_fold["mae"] + robust_fold["rmse"]
                        ):
                            consistent += 1
                    candidate = {
                        "method": "tail_rule_gate",
                        "weight": weight,
                        "alpha": None,
                        "risk_index": risk_index,
                        "risk_name": risk_name,
                        "risk_absolute": use_absolute,
                        "quantile": quantile,
                        "threshold": threshold,
                        "metrics": metrics,
                        "score": self._score(
                            metrics,
                            best_mae,
                            best_rmse,
                        ),
                        "constraints_passed": self._constraints(
                            metrics,
                            best_mae,
                            best_rmse,
                        ),
                        "consistent_folds": consistent,
                        "prediction": prediction,
                    }
                    candidates.append(candidate)
                    if (
                        candidate["constraints_passed"]
                        and consistent >= self.required_consistent_folds
                        and candidate["score"] < selected["score"] - 1e-8
                    ):
                        selected = candidate

        oracle = _oracle_gate(target, robust, mean)
        for alpha in self.ridge_alphas:
            crossfit = np.zeros_like(target)
            for fold in sorted(np.unique(meta_fold).tolist()):
                train = meta_fold != fold
                validation = meta_fold == fold
                pipeline = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=alpha),
                )
                pipeline.fit(features[train], oracle[train])
                gate = np.clip(
                    pipeline.predict(features[validation]),
                    0.0,
                    1.0,
                )
                crossfit[validation] = (
                    robust[validation]
                    + gate
                    * (mean[validation] - robust[validation])
                )
            metrics = _metrics(target, crossfit)
            reference_prediction = selected["prediction"]
            consistent = 0
            for fold in sorted(np.unique(meta_fold).tolist()):
                subset = meta_fold == fold
                candidate_fold = _metrics(
                    target[subset],
                    crossfit[subset],
                )
                reference_fold = _metrics(
                    target[subset],
                    reference_prediction[subset],
                )
                if (
                    candidate_fold["mae"] + candidate_fold["rmse"]
                    <= reference_fold["mae"] + reference_fold["rmse"]
                ):
                    consistent += 1
            candidate = {
                "method": "patient_ridge_gate",
                "weight": None,
                "alpha": alpha,
                "metrics": metrics,
                "score": self._score(metrics, best_mae, best_rmse),
                "constraints_passed": self._constraints(
                    metrics,
                    best_mae,
                    best_rmse,
                ),
                "consistent_folds": consistent,
                "prediction": crossfit,
            }
            candidates.append(candidate)
            if (
                candidate["constraints_passed"]
                and consistent >= self.required_consistent_folds
                and candidate["score"] < selected["score"] - 1e-8
            ):
                selected = candidate

        fitted_model = None
        if selected["method"] == "patient_ridge_gate":
            fitted_model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(selected["alpha"])),
            )
            fitted_model.fit(features, oracle)
        selected_rule = None
        if selected["method"] == "tail_rule_gate":
            selected_rule = {
                key: selected[key]
                for key in (
                    "risk_index",
                    "risk_name",
                    "risk_absolute",
                    "quantile",
                    "threshold",
                )
            }
        audit_candidates = []
        for value in candidates:
            audit_candidates.append(
                {
                    key: item
                    for key, item in value.items()
                    if key != "prediction"
                }
            )
        selected_prediction = selected["prediction"]
        inferred_gate = np.divide(
            selected_prediction - robust,
            mean - robust,
            out=np.zeros_like(robust),
            where=np.abs(mean - robust) > 1e-8,
        )
        audit = {
            "task": task,
            "training_source": "inner_fold_out_of_fold_predictions_only",
            "robust_head_metrics": robust_metrics,
            "mean_head_metrics": mean_metrics,
            "candidates": audit_candidates,
            "selected_method": selected["method"],
            "selected_weight": selected["weight"],
            "selected_alpha": selected["alpha"],
            "selected_metrics": selected["metrics"],
            "constraints_passed": bool(selected["constraints_passed"]),
            "gate_distribution": {
                "mean": float(np.mean(inferred_gate)),
                "std": float(np.std(inferred_gate)),
                "q05": float(np.quantile(inferred_gate, 0.05)),
                "q50": float(np.quantile(inferred_gate, 0.50)),
                "q95": float(np.quantile(inferred_gate, 0.95)),
                "fraction_near_zero": float(
                    np.mean(inferred_gate <= 0.05)
                ),
                "fraction_near_one": float(
                    np.mean(inferred_gate >= 0.95)
                ),
            },
        }
        return _TaskDecision(
            method=str(selected["method"]),
            weight=selected["weight"],
            alpha=selected["alpha"],
            model=fitted_model,
            rule=selected_rule,
            audit=audit,
        )

    def fit(
        self,
        heads: Mapping[str, np.ndarray],
        targets: Mapping[str, np.ndarray],
        features: Mapping[str, np.ndarray],
        meta_fold: np.ndarray,
    ) -> "SafeOOFDecisionLayer":
        for task in ("tbr", "cac"):
            self.tasks[task] = self._fit_task(
                task,
                np.asarray(heads[task]),
                np.asarray(targets[task]),
                np.asarray(features[task]),
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
        robust = np.asarray(heads[:, 0], float)
        mean = np.asarray(heads[:, 1], float)
        if mode == "median":
            gate = np.zeros_like(robust)
        elif mode == "mean":
            gate = np.ones_like(robust)
        elif mode == "fixed_half":
            gate = np.full_like(robust, 0.5)
        elif mode == "full":
            decision = self.tasks[task]
            if decision.method == "patient_ridge_gate":
                gate = np.clip(
                    decision.model.predict(features),
                    0.0,
                    1.0,
                )
            elif decision.method == "tail_rule_gate":
                rule = decision.rule
                risk = np.asarray(
                    features[:, int(rule["risk_index"])],
                    float,
                )
                if bool(rule["risk_absolute"]):
                    risk = np.abs(risk)
                gate = (
                    risk >= float(rule["threshold"])
                ).astype(float) * float(decision.weight)
            else:
                gate = np.full_like(robust, float(decision.weight))
        else:
            raise KeyError(mode)
        return robust + gate * (mean - robust), gate

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
                "normalized MAE + normalized RMSE; R2 is reported but not "
                "independently optimized"
            ),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds_for_patient_gate": (
                self.required_consistent_folds
            ),
            "unique_fixed_configuration": {
                task: {
                    "method": value.method,
                    "weight": value.weight,
                    "alpha": value.alpha,
                    "rule": value.rule,
                }
                for task, value in self.tasks.items()
            },
            "tasks": {
                task: value.audit for task, value in self.tasks.items()
            },
            "all_constraints_passed": all(
                value.audit["constraints_passed"]
                for value in self.tasks.values()
            ),
        }
