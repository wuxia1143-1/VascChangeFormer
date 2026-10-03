from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CalibrationSelection:
    task: str
    expert: str
    selected_expert_weight: float
    raw_mae: float
    raw_rmse: float
    calibrated_mae: float
    calibrated_rmse: float
    candidate_scores: list[dict[str, float]]
    meta_fold_patient_counts: dict[str, int]
    selection_used_outer_test: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(truth - prediction)))


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _expert(task: str, seed: int):
    if task == "tbr":
        return make_pipeline(
            StandardScaler(),
            HuberRegressor(
                epsilon=1.35,
                alpha=0.1,
                max_iter=1000,
            ),
        )
    if task == "cac":
        return GradientBoostingRegressor(
            loss="huber",
            alpha=0.85,
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            min_samples_leaf=15,
            random_state=seed,
        )
    raise KeyError(task)


class CrossFittedClinicalCalibrator:
    """Task-specific robust experts fitted from inner OOF predictions only."""

    def __init__(
        self,
        seed: int = 2026,
        candidate_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        rmse_tolerance: float = 0.02,
    ):
        self.seed = int(seed)
        self.candidate_weights = tuple(float(value) for value in candidate_weights)
        self.rmse_tolerance = float(rmse_tolerance)
        self.models: dict[str, Any] = {}
        self.weights: dict[str, float] = {}
        self.selection: dict[str, CalibrationSelection] = {}

    def fit(
        self,
        features: dict[str, np.ndarray],
        raw_predictions: np.ndarray,
        residual_targets: np.ndarray,
        meta_folds: np.ndarray,
    ) -> "CrossFittedClinicalCalibrator":
        unique_folds = sorted(np.unique(meta_folds).tolist())
        if unique_folds != [1, 2, 3, 4, 5]:
            raise ValueError("Calibration requires five complete inner OOF folds")
        for task_index, task in enumerate(("tbr", "cac")):
            task_features = np.asarray(features[task], dtype=float)
            truth = residual_targets[:, task_index].astype(float)
            raw = raw_predictions[:, task_index].astype(float)
            expert_oof = np.empty_like(truth)
            counts = {}
            for meta_fold in unique_folds:
                train = meta_folds != meta_fold
                validation = meta_folds == meta_fold
                counts[str(meta_fold)] = int(validation.sum())
                model = _expert(task, self.seed + task_index * 100 + meta_fold)
                model.fit(task_features[train], truth[train])
                expert_oof[validation] = model.predict(task_features[validation])
            raw_mae = _mae(truth, raw)
            raw_rmse = _rmse(truth, raw)
            candidates = []
            for weight in self.candidate_weights:
                prediction = weight * expert_oof + (1.0 - weight) * raw
                mae = _mae(truth, prediction)
                rmse = _rmse(truth, prediction)
                feasible = rmse <= raw_rmse * (1.0 + self.rmse_tolerance)
                candidates.append(
                    {
                        "expert_weight": weight,
                        "mae": mae,
                        "rmse": rmse,
                        "rmse_feasible": float(feasible),
                    }
                )
            feasible = [
                record
                for record in candidates
                if bool(record["rmse_feasible"])
            ]
            selected = min(
                feasible or candidates,
                key=lambda record: (
                    record["mae"],
                    record["rmse"],
                    record["expert_weight"],
                ),
            )
            final_model = _expert(task, self.seed + task_index * 100 + 99)
            final_model.fit(task_features, truth)
            self.models[task] = final_model
            self.weights[task] = float(selected["expert_weight"])
            self.selection[task] = CalibrationSelection(
                task=task,
                expert=type(
                    final_model.steps[-1][1]
                    if task == "tbr"
                    else final_model
                ).__name__,
                selected_expert_weight=float(selected["expert_weight"]),
                raw_mae=raw_mae,
                raw_rmse=raw_rmse,
                calibrated_mae=float(selected["mae"]),
                calibrated_rmse=float(selected["rmse"]),
                candidate_scores=candidates,
                meta_fold_patient_counts=counts,
            )
        return self

    def predict(
        self,
        features: dict[str, np.ndarray],
        raw_predictions: np.ndarray,
    ) -> np.ndarray:
        result = np.asarray(raw_predictions, dtype=float).copy()
        for task_index, task in enumerate(("tbr", "cac")):
            expert = self.models[task].predict(
                np.asarray(features[task], dtype=float)
            )
            weight = self.weights[task]
            result[:, task_index] = (
                weight * expert + (1.0 - weight) * result[:, task_index]
            )
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "protocol": "fivefold cross-fitted inner-OOF clinical calibration",
            "candidate_weights": list(self.candidate_weights),
            "rmse_tolerance": self.rmse_tolerance,
            "selection": {
                task: record.to_dict()
                for task, record in self.selection.items()
            },
            "outer_test_used_for_fit_or_selection": False,
        }
