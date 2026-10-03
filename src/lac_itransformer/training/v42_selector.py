from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


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


def safe_cac_features(features: np.ndarray) -> np.ndarray:
    """Remove every head-disagreement and tail-derived decision feature.

    The shared V2.7 feature builder emits central head, corrected head, head
    disagreement, ordinary clinical/meta features, five tail diagnostics, and
    the calcification adapter gate.  V4.2 retains central + ordinary features +
    adapter gate and excludes everything derived from the tail computation.
    """

    values = np.asarray(features, float)
    if values.ndim != 2 or values.shape[1] < 10:
        raise ValueError("V4.2 CAC decision features are incomplete")
    return np.column_stack([values[:, 0], values[:, 3:-6], values[:, -1]])


def _regressor(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.85,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=15,
        random_state=int(seed),
    )


def _classifier(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=15,
        random_state=int(seed),
    )


def _consistent_folds(
    target: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    folds: np.ndarray,
) -> int:
    consistent = 0
    for fold in range(1, 6):
        subset = folds == fold
        candidate_metrics = _metrics(target[subset], prediction[subset])
        reference_metrics = _metrics(target[subset], reference[subset])
        score = (
            candidate_metrics["mae"] / max(reference_metrics["mae"], 1e-8)
            + candidate_metrics["rmse"] / max(reference_metrics["rmse"], 1e-8)
        )
        consistent += int(score <= 2.0)
    return consistent


def _refinement_admissible(
    metrics: dict[str, float],
    tail_rmse: float,
    reference_metrics: dict[str, float],
    reference_tail_rmse: float,
    mae_tolerance: float,
    consistent_folds: int,
    required_consistent_folds: int,
) -> bool:
    """Apply the locked variance/two-stage refinement constraints."""

    return bool(
        metrics["mae"] <= reference_metrics["mae"] * mae_tolerance
        and metrics["rmse"] <= reference_metrics["rmse"] + 1e-12
        and tail_rmse <= reference_tail_rmse + 1e-12
        and consistent_folds >= required_consistent_folds
    )


def _fit_two_stage_models(
    features: np.ndarray,
    target: np.ndarray,
    threshold: float,
    seed: int,
):
    positive = target > float(threshold)
    classifier: GradientBoostingClassifier | None
    constant_probability: float | None
    if np.unique(positive).size == 2:
        classifier = _classifier(seed)
        classifier.fit(features, positive.astype(int))
        constant_probability = None
    else:
        classifier = None
        constant_probability = float(positive.mean())

    regressors: list[GradientBoostingRegressor] = []
    for group, offset in ((~positive, 11), (positive, 29)):
        # The locked cohort contains both groups in every inner training pool.
        # This fallback keeps the estimator defined for synthetic/small tests.
        subset = group if int(group.sum()) >= 5 else np.ones(len(target), dtype=bool)
        model = _regressor(seed + offset)
        model.fit(features[subset], target[subset])
        regressors.append(model)
    return classifier, constant_probability, regressors[0], regressors[1]


def _predict_two_stage(models, features: np.ndarray) -> np.ndarray:
    classifier, constant_probability, negative_model, positive_model = models
    if classifier is None:
        probability = np.full(len(features), float(constant_probability))
    else:
        probability = classifier.predict_proba(features)[:, 1]
    negative = negative_model.predict(features)
    positive = positive_model.predict(features)
    return (1.0 - probability) * negative + probability * positive


class V42DecisionLayer:
    """Inner-OOF-only strict calibration, variance calibration and two-stage CAC."""

    def __init__(
        self,
        candidate_weights=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
        ridge_alphas=None,
        degradation_tolerance=0.005,
        required_consistent_folds=3,
        variance_scales=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
        progression_threshold=0.25,
        cac_mode="strict",
        seed=2026,
    ):
        del ridge_alphas
        self.candidate_weights = tuple(float(value) for value in candidate_weights)
        self.variance_scales = tuple(float(value) for value in variance_scales)
        self.degradation_tolerance = float(degradation_tolerance)
        self.required_consistent_folds = int(required_consistent_folds)
        self.progression_threshold = float(progression_threshold)
        self.cac_mode = str(cac_mode)
        self.seed = int(seed)
        self.strict_weight = 0.0
        self.variance_scale = 1.0
        self.variance_intercept = 0.0
        self.two_stage_weight = 0.0
        self.strict_expert: GradientBoostingRegressor | None = None
        self.two_stage_models = None
        self.record: dict[str, Any] = {}

    def fit(self, heads, targets, features, meta_fold):
        central = np.asarray(heads["cac"][:, 0], float)
        target = np.asarray(targets["cac"], float)
        safe = safe_cac_features(features["cac"])
        folds = np.asarray(meta_fold, int)
        if sorted(np.unique(folds).tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError("V4.2 selection requires five complete inner OOF folds")

        expert_oof = np.empty_like(target)
        for fold in range(1, 6):
            train = folds != fold
            validation = folds == fold
            expert = _regressor(self.seed + fold)
            expert.fit(safe[train], target[train])
            expert_oof[validation] = expert.predict(safe[validation])
        self.strict_expert = _regressor(self.seed + 99)
        self.strict_expert.fit(safe, target)

        tolerance = 1.0 + self.degradation_tolerance
        central_metrics = _metrics(target, central)
        central_tail = _tail_rmse(target, central)
        strict_candidates = []
        for weight in self.candidate_weights:
            prediction = (1.0 - weight) * central + weight * expert_oof
            metrics = _metrics(target, prediction)
            tail = _tail_rmse(target, prediction)
            consistent = _consistent_folds(target, prediction, central, folds)
            admissible = bool(
                metrics["mae"] <= central_metrics["mae"] * tolerance
                and metrics["rmse"] <= central_metrics["rmse"] * tolerance
                and tail <= central_tail * tolerance
                and consistent >= self.required_consistent_folds
            )
            strict_candidates.append(
                {
                    "weight": weight,
                    "metrics": metrics,
                    "tail_rmse": tail,
                    "consistent_folds": consistent,
                    "admissible": admissible,
                    "score": (
                        metrics["mae"] / max(central_metrics["mae"], 1e-8)
                        + metrics["rmse"] / max(central_metrics["rmse"], 1e-8)
                        + 0.25 * tail / max(central_tail, 1e-8)
                    ),
                }
            )
        strict_selected = min(
            [item for item in strict_candidates if item["admissible"]]
            or [strict_candidates[0]],
            key=lambda item: (item["score"], item["weight"]),
        )
        self.strict_weight = float(strict_selected["weight"])
        strict_oof = (
            (1.0 - self.strict_weight) * central
            + self.strict_weight * expert_oof
        )
        strict_metrics = _metrics(target, strict_oof)
        strict_tail = _tail_rmse(target, strict_oof)

        variance_candidates = [
            {
                "scale": 1.0,
                "intercept": 0.0,
                "metrics": strict_metrics,
                "tail_rmse": strict_tail,
                "consistent_folds": 5,
                "admissible": True,
                "score": 1.5,
            }
        ]
        prediction_mean = float(strict_oof.mean())
        target_mean = float(target.mean())
        for scale in self.variance_scales:
            intercept = target_mean - scale * prediction_mean
            prediction = intercept + scale * strict_oof
            metrics = _metrics(target, prediction)
            tail = _tail_rmse(target, prediction)
            consistent = _consistent_folds(target, prediction, strict_oof, folds)
            admissible = _refinement_admissible(
                metrics,
                tail,
                strict_metrics,
                strict_tail,
                tolerance,
                consistent,
                self.required_consistent_folds,
            )
            variance_candidates.append(
                {
                    "scale": scale,
                    "intercept": intercept,
                    "metrics": metrics,
                    "tail_rmse": tail,
                    "consistent_folds": consistent,
                    "admissible": admissible,
                    "score": (
                        metrics["rmse"] / max(strict_metrics["rmse"], 1e-8)
                        + 0.5 * tail / max(strict_tail, 1e-8)
                    ),
                }
            )
        variance_selected = min(
            [item for item in variance_candidates if item["admissible"]],
            key=lambda item: (item["score"], abs(item["scale"] - 1.0)),
        )
        self.variance_scale = float(variance_selected["scale"])
        self.variance_intercept = float(variance_selected["intercept"])

        mixture_oof = np.empty_like(target)
        for fold in range(1, 6):
            train = folds != fold
            validation = folds == fold
            models = _fit_two_stage_models(
                safe[train],
                target[train],
                self.progression_threshold,
                self.seed + 1_000 + fold * 100,
            )
            mixture_oof[validation] = _predict_two_stage(models, safe[validation])
        self.two_stage_models = _fit_two_stage_models(
            safe,
            target,
            self.progression_threshold,
            self.seed + 9_000,
        )
        two_stage_candidates = []
        for weight in self.candidate_weights:
            prediction = (1.0 - weight) * strict_oof + weight * mixture_oof
            metrics = _metrics(target, prediction)
            tail = _tail_rmse(target, prediction)
            consistent = _consistent_folds(target, prediction, strict_oof, folds)
            admissible = _refinement_admissible(
                metrics,
                tail,
                strict_metrics,
                strict_tail,
                tolerance,
                consistent,
                self.required_consistent_folds,
            )
            two_stage_candidates.append(
                {
                    "weight": weight,
                    "metrics": metrics,
                    "tail_rmse": tail,
                    "consistent_folds": consistent,
                    "admissible": admissible,
                    "score": (
                        metrics["rmse"] / max(strict_metrics["rmse"], 1e-8)
                        + 0.5 * tail / max(strict_tail, 1e-8)
                    ),
                }
            )
        two_stage_selected = min(
            [item for item in two_stage_candidates if item["admissible"]]
            or [two_stage_candidates[0]],
            key=lambda item: (item["score"], item["weight"]),
        )
        self.two_stage_weight = float(two_stage_selected["weight"])
        self.record = {
            "safe_feature_count": int(safe.shape[1]),
            "strict": {
                "reference_metrics": central_metrics,
                "reference_tail_rmse": central_tail,
                "selected": strict_selected,
                "candidates": strict_candidates,
            },
            "variance": {
                "reference_metrics": strict_metrics,
                "reference_tail_rmse": strict_tail,
                "selected": variance_selected,
                "candidates": variance_candidates,
            },
            "two_stage": {
                "progression_threshold": self.progression_threshold,
                "reference_metrics": strict_metrics,
                "reference_tail_rmse": strict_tail,
                "selected": two_stage_selected,
                "candidates": two_stage_candidates,
            },
        }
        return self

    def _strict_prediction(self, heads, features) -> np.ndarray:
        if self.strict_expert is None:
            raise RuntimeError("V4.2 decision layer has not been fitted")
        central = np.asarray(heads["cac"][:, 0], float)
        expert = self.strict_expert.predict(safe_cac_features(features["cac"]))
        return (1.0 - self.strict_weight) * central + self.strict_weight * expert

    def predict(self, heads, features, modes=None):
        modes = dict(modes or {})
        tbr = np.asarray(heads["tbr"][:, 0], float)
        strict = self._strict_prediction(heads, features)
        mode = modes.get("cac", "strict")
        if mode in {"strict", "no_tail", "identity"}:
            cac = strict
            gate_value = self.strict_weight
        elif mode in {"variance", "varcal"}:
            cac = self.variance_intercept + self.variance_scale * strict
            gate_value = self.variance_scale
        elif mode in {"two_stage", "two-stage"}:
            if self.two_stage_models is None:
                raise RuntimeError("V4.2 two-stage models have not been fitted")
            mixture = _predict_two_stage(
                self.two_stage_models,
                safe_cac_features(features["cac"]),
            )
            cac = (
                (1.0 - self.two_stage_weight) * strict
                + self.two_stage_weight * mixture
            )
            gate_value = self.two_stage_weight
        else:
            raise KeyError(mode)
        return np.column_stack([tbr, cac]), {
            "tbr": np.zeros_like(tbr),
            "cac": np.full_like(cac, gate_value),
        }

    def audit(self):
        selected = {
            "strict": self.record["strict"]["selected"],
            "variance": self.record["variance"]["selected"],
            "two_stage": self.record["two_stage"]["selected"],
        }[self.cac_mode]
        return {
            "selection_scope": "five complete inner OOF folds only",
            "safe_feature_policy": {
                "included": [
                    "central_cac_prediction",
                    "baseline_log_cac",
                    "baseline_tbr",
                    "followup_duration_and_window",
                    "baseline_cac_positive",
                    "observation_density",
                    "treatment",
                    "calcification_soft_adapter_gate",
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
            "candidate_weights": list(self.candidate_weights),
            "variance_scales": list(self.variance_scales),
            "degradation_tolerance": self.degradation_tolerance,
            "required_consistent_folds": self.required_consistent_folds,
            "selected": {
                "strict_weight": self.strict_weight,
                "variance_scale": self.variance_scale,
                "variance_intercept": self.variance_intercept,
                "two_stage_weight": self.two_stage_weight,
            },
            "unique_fixed_configuration": {
                "tbr": {
                    "method": "identity",
                    "weight": 0.0,
                    "alpha": None,
                    "rule": "protected central TBR head",
                },
                "cac": {
                    "method": f"v42_{self.cac_mode}",
                    "weight": selected.get("weight"),
                    "alpha": selected.get("scale"),
                    "rule": "preregistered inner-OOF-only V4.2 decision",
                },
            },
            "all_constraints_passed": bool(selected["admissible"]),
            "tasks": {"tbr": {"selected_method": "identity"}, "cac": self.record},
            "outer_test_labels_used": False,
            "external_labels_used": False,
        }
