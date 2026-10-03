from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def build_patient_fold_plan(
    patient_ids: Sequence[str], n_folds: int = 5, seed: int = 2026
) -> dict[str, int]:
    """Create a deterministic, row-order-independent patient-level fold plan."""
    ids = np.asarray([str(value) for value in patient_ids])
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} patients, received {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("Patient identifiers must be unique before fold assignment")
    ordered = np.sort(ids)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    plan: dict[str, int] = {}
    for fold, (_, test_position) in enumerate(splitter.split(ordered), start=1):
        for position in test_position:
            plan[str(ordered[position])] = fold
    validate_patient_fold_plan(ids, plan, n_folds)
    return plan


def validate_patient_fold_plan(
    patient_ids: Sequence[str], plan: Mapping[str, int], n_folds: int = 5
) -> None:
    ids = [str(value) for value in patient_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Patient identifiers are not unique")
    missing = set(ids) - set(plan)
    extra = set(plan) - set(ids)
    if missing or extra:
        raise ValueError(f"Fold plan mismatch: missing={len(missing)}, extra={len(extra)}")
    observed = {int(plan[patient_id]) for patient_id in ids}
    expected = set(range(1, n_folds + 1))
    if observed != expected:
        raise ValueError(f"Expected fold labels {sorted(expected)}, received {sorted(observed)}")
    if any(int(plan[patient_id]) not in expected for patient_id in ids):
        raise ValueError("Fold plan contains an invalid label")


def fold_plan_checksum(plan: Mapping[str, int]) -> str:
    lines = "\n".join(f"{patient_id}\t{int(plan[patient_id])}" for patient_id in sorted(plan))
    return hashlib.sha256(lines.encode()).hexdigest()


def save_patient_fold_plan(
    plan: Mapping[str, int], output_csv: str | Path, seed: int, n_folds: int = 5
) -> dict[str, object]:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"patient_id": patient_id, "test_fold": int(plan[patient_id])} for patient_id in sorted(plan)]
    )
    frame.to_csv(output, index=False)
    counts = {str(key): int(value) for key, value in frame["test_fold"].value_counts().sort_index().items()}
    manifest = {
        "protocol": "patient_level_outer_kfold",
        "n_folds": n_folds,
        "seed": seed,
        "n_patients": len(frame),
        "fold_counts": counts,
        "checksum_sha256": fold_plan_checksum(plan),
        "each_patient_is_test_once": True,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def indices_for_outer_fold(
    patient_ids: Sequence[str], plan: Mapping[str, int], test_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(plan[str(value)]) for value in patient_ids])
    test = np.flatnonzero(labels == test_fold)
    train = np.flatnonzero(labels != test_fold)
    if len(test) == 0 or len(train) == 0:
        raise ValueError(f"Fold {test_fold} has an empty train or test partition")
    return train, test


def inner_calibration_indices(
    patient_ids: Sequence[str], fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split only the four-fold training pool for epoch selection."""
    if not 0 < fraction < 0.5:
        raise ValueError("inner_validation_fraction must be between 0 and 0.5")
    ids = np.asarray([str(value) for value in patient_ids])
    order = np.argsort(ids)
    rng = np.random.default_rng(seed)
    shuffled = order[rng.permutation(len(order))]
    n_validation = max(1, int(round(len(ids) * fraction)))
    n_validation = min(n_validation, len(ids) - 1)
    validation = np.sort(shuffled[:n_validation])
    training = np.sort(shuffled[n_validation:])
    return training, validation
