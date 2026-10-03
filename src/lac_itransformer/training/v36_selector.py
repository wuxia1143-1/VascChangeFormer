from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / denominator),
    }


def _expert(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=15,
        random_state=int(seed),
    )


class ParetoSafeClinicalSelector:
    """Inner-OOF-only robust CAC expert with one shared final prediction."""

    def __init__(
        self,
        candidate_weights=(0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
        ridge_alphas=(1.0, 10.0),
        degradation_tolerance=0.005,
        required_consistent_folds=3,
        seed=2026,
    ):
        del ridge_alphas
        self.candidate_weights = tuple(float(value) for value in candidate_weights)
        self.degradation_tolerance = float(degradation_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.seed = int(seed)
        self.expert = None
        self.weight = 0.0
        self.record: dict[str, Any] = {}

    def fit(self, heads, targets, features, meta_fold):
        raw = np.asarray(heads["cac"][:, 0], float)
        target = np.asarray(targets["cac"], float)
        inputs = np.asarray(features["cac"], float)
        meta_fold = np.asarray(meta_fold, int)
        folds = sorted(np.unique(meta_fold).tolist())
        if folds != [1, 2, 3, 4, 5]:
            raise ValueError("V3.6 calibration requires five complete inner OOF folds")
        expert_oof = np.empty_like(target)
        for fold in folds:
            train, validation = meta_fold != fold, meta_fold == fold
            model = _expert(self.seed + fold)
            model.fit(inputs[train], target[train])
            expert_oof[validation] = model.predict(inputs[validation])
        reference = _metrics(target, raw)
        candidates = []
        for weight in self.candidate_weights:
            prediction = (1.0 - weight) * raw + weight * expert_oof
            metrics = _metrics(target, prediction)
            consistent = 0
            for fold in folds:
                subset = meta_fold == fold
                fold_metrics = _metrics(target[subset], prediction[subset])
                fold_reference = _metrics(target[subset], raw[subset])
                fold_score = (
                    fold_metrics["mae"] / max(fold_reference["mae"], 1e-8)
                    + fold_metrics["rmse"] / max(fold_reference["rmse"], 1e-8)
                )
                if fold_score <= 2.0:
                    consistent += 1
            score = (
                metrics["mae"] / max(reference["mae"], 1e-8)
                + metrics["rmse"] / max(reference["rmse"], 1e-8)
            )
            admissible = bool(
                metrics["mae"]
                <= reference["mae"] * (1.0 + self.degradation_tolerance)
                and metrics["rmse"]
                <= reference["rmse"] * (1.0 + self.degradation_tolerance)
                and consistent >= self.required_consistent_folds
            )
            candidates.append(
                {
                    "weight": weight,
                    "metrics": metrics,
                    "normalized_mae_plus_rmse": float(score),
                    "consistent_folds": int(consistent),
                    "admissible": admissible,
                }
            )
        admissible = [candidate for candidate in candidates if candidate["admissible"]]
        selected = min(
            admissible or [candidates[0]],
            key=lambda candidate: (
                candidate["normalized_mae_plus_rmse"],
                candidate["weight"],
            ),
        )
        self.weight = float(selected["weight"])
        self.expert = _expert(self.seed + 99)
        self.expert.fit(inputs, target)
        self.record = {
            "selected_method": "inner_oof_pareto_safe_robust_calibration",
            "selected_weight": self.weight,
            "raw_metrics": reference,
            "selected_metrics": selected["metrics"],
            "selected_consistent_folds": int(selected["consistent_folds"]),
            "candidates": candidates,
        }
        return self

    def predict(self, heads, features, modes=None):
        modes = dict(modes or {})
        raw_tbr = np.asarray(heads["tbr"][:, 0], float)
        raw_cac = np.asarray(heads["cac"][:, 0], float)
        expert_cac = self.expert.predict(np.asarray(features["cac"], float))
        mode = modes.get("cac", "full")
        if mode in {"identity", "raw", "central"}:
            weight = 0.0
        elif mode in {"expert", "calibration_only"}:
            weight = 1.0
        else:
            weight = self.weight
        cac = (1.0 - weight) * raw_cac + weight * expert_cac
        return (
            np.column_stack([raw_tbr, cac]),
            {
                "tbr": np.zeros_like(raw_tbr),
                "cac": np.full_like(raw_cac, weight),
            },
        )

    def audit(self):
        return {
            "selection_objective": (
                "equal normalized CAC MAE and RMSE with 0.5% per-metric "
                "non-inferiority and at least three of five inner-fold directions"
            ),
            "candidate_weights": list(self.candidate_weights),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "unique_fixed_configuration": {
                "tbr": {"method": "identity", "weight": 0.0},
                "cac": {
                    "method": "inner_oof_pareto_safe_robust_calibration",
                    "weight": self.weight,
                },
            },
            "tasks": {
                "tbr": {"selected_method": "identity"},
                "cac": self.record,
            },
            "all_constraints_passed": True,
            "outer_test_labels_used": False,
        }
