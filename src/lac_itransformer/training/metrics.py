from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else float("nan")


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    true_tbr, pred_tbr = targets[:, 0], predictions[:, 0]
    true_log_cac, pred_log_cac = np.log1p(np.maximum(targets[:, 1], 0)), np.log1p(np.maximum(predictions[:, 1], 0))
    return {
        "tbr_mae": float(mean_absolute_error(true_tbr, pred_tbr)),
        "tbr_rmse": float(mean_squared_error(true_tbr, pred_tbr) ** 0.5),
        "tbr_r2": _safe_r2(true_tbr, pred_tbr),
        "log_cac_mae": float(mean_absolute_error(true_log_cac, pred_log_cac)),
        "log_cac_rmse": float(mean_squared_error(true_log_cac, pred_log_cac) ** 0.5),
        "log_cac_r2": _safe_r2(true_log_cac, pred_log_cac),
        "cac_median_ae": float(np.median(np.abs(targets[:, 1] - predictions[:, 1]))),
    }


def change_space_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    """Evaluate the two prespecified residual targets without endpoint mixing."""
    true_delta_tbr = targets[:, 0] - baseline[:, 0]
    pred_delta_tbr = predictions[:, 0] - baseline[:, 0]
    true_delta_log_cac = np.log1p(np.maximum(targets[:, 1], 0)) - np.log1p(
        np.maximum(baseline[:, 1], 0)
    )
    pred_delta_log_cac = np.log1p(
        np.maximum(predictions[:, 1], 0)
    ) - np.log1p(np.maximum(baseline[:, 1], 0))
    return {
        "delta_tbr_mae": float(
            mean_absolute_error(true_delta_tbr, pred_delta_tbr)
        ),
        "delta_tbr_rmse": float(
            mean_squared_error(true_delta_tbr, pred_delta_tbr) ** 0.5
        ),
        "delta_tbr_r2": _safe_r2(true_delta_tbr, pred_delta_tbr),
        "delta_log_cac_mae": float(
            mean_absolute_error(true_delta_log_cac, pred_delta_log_cac)
        ),
        "delta_log_cac_rmse": float(
            mean_squared_error(true_delta_log_cac, pred_delta_log_cac) ** 0.5
        ),
        "delta_log_cac_r2": _safe_r2(
            true_delta_log_cac, pred_delta_log_cac
        ),
    }


def bootstrap_change_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    point = change_space_metrics(targets, predictions, baseline)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(targets), len(targets))
        result = change_space_metrics(
            targets[index], predictions[index], baseline[index]
        )
        for key, value in result.items():
            if np.isfinite(value):
                samples[key].append(value)
    return {
        key: {
            "estimate": value,
            "ci_low": (
                float(np.quantile(samples[key], 0.025))
                if samples[key]
                else float("nan")
            ),
            "ci_high": (
                float(np.quantile(samples[key], 0.975))
                if samples[key]
                else float("nan")
            ),
        }
        for key, value in point.items()
    }


def bootstrap_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    point = regression_metrics(targets, predictions)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(targets), len(targets))
        result = regression_metrics(targets[index], predictions[index])
        for key, value in result.items():
            if np.isfinite(value):
                samples[key].append(value)
    return {
        key: {
            "estimate": value,
            "ci_low": float(np.quantile(samples[key], 0.025)) if samples[key] else float("nan"),
            "ci_high": float(np.quantile(samples[key], 0.975)) if samples[key] else float("nan"),
        }
        for key, value in point.items()
    }
