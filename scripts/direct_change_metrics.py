"""Performance in native change coordinates, before endpoint reconstruction.

Column 0 is delta TBR and column 1 is delta log(1 + TAC). Negative
predicted changes are valid and are never clipped or exponentiated here.
"""
from __future__ import annotations

import numpy as np

TARGETS = ("delta_tbr", "delta_log_tac")
METRICS = ("mae", "rmse", "r2")


def direct_metrics(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != prediction.shape:
        raise ValueError("Expected nonempty matching one-dimensional change arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Changes must be finite")
    error = prediction - truth
    denominator = np.square(truth - truth.mean()).sum()
    return np.array([
        np.abs(error).mean(),
        np.sqrt(np.square(error).mean()),
        1 - np.square(error).sum() / denominator if denominator > 0 else np.nan,
    ])


def patient_draws(n: int, centre: str, experiment: str,
                  replicates: int = 2000, seed: int = 2026) -> np.ndarray:
    if n < 1 or replicates < 1:
        raise ValueError("Patient and bootstrap counts must be positive")
    # Preserve the index convention of the archived October 22 evaluation.
    dtype = np.int32 if centre == "C" and experiment == "primary" else np.int64
    return np.random.default_rng(seed).integers(0, n, size=(replicates, n), dtype=dtype)


def bootstrap_samples(truth: np.ndarray, prediction: np.ndarray,
                      draws: np.ndarray) -> np.ndarray:
    direct_metrics(truth, prediction)  # Validate before indexing.
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    draws = np.asarray(draws)
    if (draws.ndim != 2 or draws.shape[1] != len(truth)
            or not np.issubdtype(draws.dtype, np.integer)
            or not draws.size or draws.min() < 0 or draws.max() >= len(truth)):
        raise ValueError("Invalid patient bootstrap indices")
    target = truth[draws]
    error = prediction[draws] - target
    squared_error = np.square(error).sum(axis=1)
    denominator = np.square(target - target.mean(axis=1, keepdims=True)).sum(axis=1)
    r2 = 1 - np.divide(squared_error, denominator,
                       out=np.full(len(draws), np.nan), where=denominator > 0)
    return np.column_stack([
        np.abs(error).mean(axis=1), np.sqrt(np.square(error).mean(axis=1)), r2,
    ])
