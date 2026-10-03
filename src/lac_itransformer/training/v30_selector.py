from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / denominator),
    }


@dataclass
class _Selection:
    weight: float
    audit: dict[str, Any]


class DirectionalLagScaleSelector:
    """Inner-OOF-only shrinkage of the neural historical lag residual."""

    def __init__(
        self,
        candidate_weights=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        ridge_alphas=(1.0, 10.0),
        degradation_tolerance=0.005,
        required_consistent_folds=4,
    ):
        del ridge_alphas
        self.candidate_weights = tuple(float(x) for x in candidate_weights)
        self.degradation_tolerance = float(degradation_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.tasks: dict[str, _Selection] = {}

    def fit(self, heads, targets, features, meta_fold):
        del features
        self.tasks["tbr"] = _Selection(0.0, {
            "task": "tbr", "selected_method": "identity",
            "training_source": "single protected TBR prediction",
        })
        central = np.asarray(heads["cac"][:, 0], float)
        raw_final = np.asarray(heads["cac"][:, 1], float)
        target = np.asarray(targets["cac"], float)
        meta_fold = np.asarray(meta_fold, int)
        reference = _metrics(target, central)
        candidates = []
        for weight in self.candidate_weights:
            prediction = central + weight * (raw_final - central)
            metrics = _metrics(target, prediction)
            consistent = 0
            for fold in sorted(np.unique(meta_fold)):
                subset = meta_fold == fold
                candidate_fold = _metrics(target[subset], prediction[subset])
                reference_fold = _metrics(target[subset], central[subset])
                if (
                    candidate_fold["rmse"] <= reference_fold["rmse"]
                    and candidate_fold["mae"]
                    <= reference_fold["mae"] * (1.0 + self.degradation_tolerance)
                ):
                    consistent += 1
            score = (
                metrics["rmse"] / max(reference["rmse"], 1e-8)
                + 0.25 * metrics["mae"] / max(reference["mae"], 1e-8)
            )
            admissible = (
                metrics["mae"] <= reference["mae"] * (1.0 + self.degradation_tolerance)
                and consistent >= self.required_consistent_folds
                and (score < 1.25 - 1e-8 or weight == 0.0)
            )
            candidates.append({
                "weight": weight, "metrics": metrics, "score": score,
                "consistent_folds": consistent, "admissible": admissible,
            })
        selected = min(
            [value for value in candidates if value["admissible"]],
            key=lambda value: (value["score"], value["weight"]),
        )
        self.tasks["cac"] = _Selection(float(selected["weight"]), {
            "task": "cac",
            "selected_method": "global_lag_residual_scale",
            "selected_weight": float(selected["weight"]),
            "central_metrics": reference,
            "raw_lag_final_metrics": _metrics(target, raw_final),
            "selected_metrics": selected["metrics"],
            "selected_consistent_folds": int(selected["consistent_folds"]),
            "training_source": "outer-training-pool inner-fold OOF only",
            "candidates": candidates,
        })
        return self

    def predict_task(self, task, heads, features, mode="full"):
        del features
        central = np.asarray(heads[:, 0], float)
        raw = np.asarray(heads[:, 1], float)
        if task == "tbr" or mode in {"identity", "median", "central"}:
            return central, np.zeros_like(central)
        if mode == "mean":
            return raw, np.ones_like(central)
        weight = self.tasks["cac"].weight
        return central + weight * (raw - central), np.full_like(central, weight)

    def predict(self, heads, features, modes=None):
        modes = dict(modes or {})
        predictions, gates = [], {}
        for task in ("tbr", "cac"):
            prediction, gate = self.predict_task(
                task, heads[task], features[task], modes.get(task, "full")
            )
            predictions.append(prediction)
            gates[task] = gate
        return np.column_stack(predictions), gates

    def audit(self):
        return {
            "selection_objective": (
                "CAC RMSE priority with central-CAC MAE non-inferiority; "
                "inner OOF only"
            ),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "unique_fixed_configuration": {
                "tbr": {"method": "identity", "weight": 0.0, "alpha": None},
                "cac": {
                    "method": "global_lag_residual_scale",
                    "weight": self.tasks["cac"].weight,
                    "alpha": None,
                },
            },
            "tasks": {name: value.audit for name, value in self.tasks.items()},
            "all_constraints_passed": True,
            "outer_test_labels_used": False,
        }
