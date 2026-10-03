from __future__ import annotations

import numpy as np


PATTERN_NAMES = np.asarray(["stable", "inflammation_dominant", "calcification_dominant", "dual_active"])


def fit_pattern_thresholds(inflammation_score: np.ndarray, calcification_score: np.ndarray) -> dict[str, float]:
    """Fit once on the development cohort; reuse unchanged externally."""
    return {
        "inflammation": float(np.median(inflammation_score)),
        "calcification": float(np.median(calcification_score)),
    }


def assign_patterns(inflammation_score: np.ndarray, calcification_score: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    high_i = inflammation_score >= thresholds["inflammation"]
    high_c = calcification_score >= thresholds["calcification"]
    code = high_i.astype(int) + 2 * high_c.astype(int)
    mapping = np.asarray([0, 1, 2, 3])
    return PATTERN_NAMES[mapping[code]]


def summarize_patterns(patterns: np.ndarray, targets: np.ndarray) -> list[dict]:
    result = []
    for name in PATTERN_NAMES:
        selected = patterns == name
        result.append({
            "pattern": str(name),
            "n": int(selected.sum()),
            "endpoint_tbr_mean": float(targets[selected, 0].mean()) if selected.any() else None,
            "endpoint_cac_median": float(np.median(targets[selected, 1])) if selected.any() else None,
        })
    return result
