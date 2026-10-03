from __future__ import annotations

from typing import Any

import numpy as np


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
    threshold = float(np.quantile(target, 0.90))
    subset = target >= threshold
    return float(np.sqrt(np.square(prediction[subset] - target[subset]).mean()))


class ProgressionHurdleSelector:
    """Inner-OOF-only fail-safe selection of one hurdle correction weight."""

    def __init__(
        self,
        candidate_weights=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
        ridge_alphas=None,
        degradation_tolerance=None,
        mae_tolerance=0.0025,
        rmse_tolerance=0.0025,
        tail_tolerance=0.01,
        required_consistent_folds=3,
        seed=2026,
    ):
        self.candidate_weights = tuple(float(value) for value in candidate_weights)
        self.mae_tolerance = float(
            mae_tolerance if degradation_tolerance is None else degradation_tolerance
        )
        self.rmse_tolerance = float(rmse_tolerance)
        self.tail_tolerance = float(tail_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.seed = int(seed)
        self.weight = 0.0
        self.record: dict[str, Any] = {}

    def fit(self, heads, targets, features, meta_fold):
        central = np.asarray(heads["cac"][:, 0], float)
        corrected = np.asarray(heads["cac"][:, 1], float)
        target = np.asarray(targets["cac"], float)
        folds = np.asarray(meta_fold, int)
        if sorted(np.unique(folds).tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError("V4.1 selection requires five complete inner OOF folds")
        reference = _metrics(target, central)
        reference_tail = _tail_rmse(target, central)
        candidates = []
        for weight in self.candidate_weights:
            prediction = central + weight * (corrected - central)
            metrics = _metrics(target, prediction)
            tail = _tail_rmse(target, prediction)
            improved_folds = 0
            for fold in range(1, 6):
                subset = folds == fold
                fold_reference = _metrics(target[subset], central[subset])
                fold_candidate = _metrics(target[subset], prediction[subset])
                improved_folds += int(
                    fold_candidate["mae"] + fold_candidate["rmse"]
                    <= fold_reference["mae"] + fold_reference["rmse"]
                )
            score = (
                metrics["mae"] / max(reference["mae"], 1e-8)
                + metrics["rmse"] / max(reference["rmse"], 1e-8)
            )
            admissible = bool(
                metrics["mae"] <= reference["mae"] * (1.0 + self.mae_tolerance)
                and metrics["rmse"] <= reference["rmse"] * (1.0 + self.rmse_tolerance)
                and tail <= reference_tail * (1.0 + self.tail_tolerance)
                and improved_folds >= self.required_consistent_folds
            )
            candidates.append({
                "weight": weight,
                "metrics": metrics,
                "tail_rmse": tail,
                "improved_folds": int(improved_folds),
                "score": float(score),
                "admissible": admissible,
            })
        admissible = [item for item in candidates if item["admissible"]]
        selected = min(admissible, key=lambda item: (item["score"], item["weight"])) if admissible else candidates[0]
        self.weight = float(selected["weight"])
        self.record = {
            "selected_method": "inner_oof_progression_hurdle_weight",
            "selected_weight": self.weight,
            "selected_parameters": {"weight": self.weight},
            "reference_metrics": reference,
            "reference_tail_rmse": reference_tail,
            "selected_metrics": selected["metrics"],
            "selected_tail_rmse": selected["tail_rmse"],
            "selected_improved_folds": selected["improved_folds"],
            "selected_admissible": bool(selected["admissible"]),
            "candidates": candidates,
        }
        return self

    def predict(self, heads, features, modes=None):
        modes = dict(modes or {})
        tbr = np.asarray(heads["tbr"][:, 0], float)
        central = np.asarray(heads["cac"][:, 0], float)
        corrected = np.asarray(heads["cac"][:, 1], float)
        mode = modes.get("cac", "full")
        if mode in {"central", "base", "identity", "raw"}:
            weight = 0.0
        elif mode in {"corrected", "mean", "mechanism", "expert"}:
            weight = 1.0
        else:
            weight = self.weight
        cac = central + weight * (corrected - central)
        return np.column_stack([tbr, cac]), {
            "tbr": np.zeros_like(tbr),
            "cac": np.full_like(cac, weight),
        }

    def audit(self):
        return {
            "selection_objective": (
                "inner-OOF CAC MAE+RMSE with central, tail and >=3/5-fold "
                "non-inferiority; fallback is exactly the V3.7 central head"
            ),
            "candidate_hierarchy": ["global_hurdle_weight_only"],
            "candidate_weights": list(self.candidate_weights),
            "mae_tolerance": self.mae_tolerance,
            "rmse_tolerance": self.rmse_tolerance,
            "tail_tolerance": self.tail_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "unique_fixed_configuration": {
                "tbr": {"method": "identity", "weight": 0.0},
                "cac": {"method": "progression_hurdle", "weight": self.weight},
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
