from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASKS = {
    "tbr": ("true_delta_tbr", "pred_delta_tbr"),
    "cac": ("true_delta_log_cac", "pred_delta_log_cac"),
}
REFERENCE_MODEL = "lac_v40_final_no_tail"
CANDIDATE_MODEL = "lac_v50_full"


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


def _load_oof(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "outer_fold"}
    for truth, prediction in TASKS.values():
        required.update((truth, prediction))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF file missing columns: {missing}")
    if len(frame) != 443 or frame["patient_id"].nunique() != 443:
        raise ValueError("Locked V5.0 assessment requires 443 unique patients")
    if sorted(frame["outer_fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("OOF file must contain all five locked outer folds")
    numeric = ["outer_fold"] + [
        value for pair in TASKS.values() for value in pair
    ]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("OOF assessment columns must all be finite")
    return frame.sort_values("patient_id").reset_index(drop=True)


def _validate_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if not reference["patient_id"].equals(candidate["patient_id"]):
        raise ValueError("Candidate and reference patient sets differ")
    if not np.array_equal(reference["outer_fold"], candidate["outer_fold"]):
        raise ValueError("Candidate and reference outer folds differ")
    for truth, _ in TASKS.values():
        if not np.allclose(
            reference[truth].to_numpy(float),
            candidate[truth].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Candidate target changed: {truth}")


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for task, (truth, prediction) in TASKS.items():
        values = _metrics(frame[truth], frame[prediction])
        result.update({f"{task}_{name}": value for name, value in values.items()})
    return result


def _decision_diagnostics(candidate: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Evaluate preregistered decision ablations from identical neural OOF heads."""

    variants = {
        "lac_v50_central": "median_pred_delta_log_cac",
        "lac_v50_no_oof_calibration": "mean_pred_delta_log_cac",
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for name, cac_prediction in variants.items():
        if cac_prediction not in candidate:
            continue
        derived = candidate.copy()
        derived["pred_delta_log_cac"] = derived[cac_prediction]
        diagnostics[name] = _summary(derived)
    return diagnostics


def _paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if int(replicates) <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    observed_reference = _summary(reference)
    observed_candidate = _summary(candidate)
    names = tuple(observed_reference)
    differences = {
        name: np.empty(int(replicates), dtype=float) for name in names
    }
    rng = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        indices = rng.integers(0, len(reference), size=len(reference))
        reference_metrics = _summary(reference.iloc[indices])
        candidate_metrics = _summary(candidate.iloc[indices])
        for name in names:
            differences[name][replicate] = (
                candidate_metrics[name] - reference_metrics[name]
            )
    return {
        name: {
            "observed_difference_candidate_minus_reference": float(
                observed_candidate[name] - observed_reference[name]
            ),
            "bootstrap_mean_difference": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in differences.items()
    }


def evaluate_v50(
    reference_path: str | Path,
    candidate_path: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
    absolute_mae_tolerance: float = 0.01,
) -> dict[str, Any]:
    tolerance = float(absolute_mae_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Absolute MAE tolerance must be finite and non-negative")
    reference = _load_oof(reference_path)
    candidate = _load_oof(candidate_path)
    _validate_alignment(reference, candidate)
    metrics = {
        REFERENCE_MODEL: _summary(reference),
        CANDIDATE_MODEL: _summary(candidate),
    }
    bootstrap = _paired_bootstrap(
        reference, candidate, int(replicates), int(seed)
    )
    degradation = {
        task: float(
            metrics[CANDIDATE_MODEL][f"{task}_mae"]
            - metrics[REFERENCE_MODEL][f"{task}_mae"]
        )
        for task in TASKS
    }
    criteria = {
        f"{task}_mae_absolute_degradation_le_{tolerance:g}": bool(
            value <= tolerance
        )
        for task, value in degradation.items()
    }
    passed = bool(all(criteria.values()))
    return {
        "scope": "locked 443-patient internal structural non-inferiority",
        "reference_model": REFERENCE_MODEL,
        "candidate_model": CANDIDATE_MODEL,
        "external_labels_or_predictions_read": False,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "absolute_mae_tolerance": tolerance,
        "metrics": metrics,
        "decision_diagnostics": _decision_diagnostics(candidate),
        "mae_degradation_candidate_minus_reference": degradation,
        "paired_bootstrap": bootstrap,
        "acceptance": {
            "criteria": criteria,
            "passed": passed,
            "bootstrap_is_additional_gate": False,
        },
        "eligible_for_locked_full_refit": passed,
        "external_evaluation_authorized": False,
        "external_authorization_pending": (
            ["hash-locked full-refit model/decision/sidecar/preprocessing manifest"]
            if passed
            else ["internal structural non-inferiority did not pass"]
        ),
        "final_internal_decision": (
            "freeze_v50_and_build_locked_full_refit_manifest"
            if passed
            else "retain_v40_final_no_tail"
        ),
    }


def _write_outputs(result: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame.from_dict(result["metrics"], orient="index")
    metrics.index.name = "model"
    metrics.reset_index().to_csv(
        destination / "v50_internal_metrics.csv", index=False
    )
    rows = [
        {"metric": metric} | values
        for metric, values in result["paired_bootstrap"].items()
    ]
    pd.DataFrame(rows).to_csv(
        destination / "v50_paired_bootstrap_95ci.csv", index=False
    )
    diagnostics = result.get("decision_diagnostics", {})
    if diagnostics:
        diagnostic_frame = pd.DataFrame.from_dict(diagnostics, orient="index")
        diagnostic_frame.index.name = "model"
        diagnostic_frame.reset_index().to_csv(
            destination / "v50_decision_diagnostics.csv", index=False
        )
    (destination / "v50_internal_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--absolute-mae-tolerance", type=float, default=0.01)
    arguments = parser.parse_args()
    result = evaluate_v50(
        arguments.reference,
        arguments.candidate,
        replicates=arguments.bootstrap,
        seed=arguments.seed,
        absolute_mae_tolerance=arguments.absolute_mae_tolerance,
    )
    _write_outputs(result, arguments.output)
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
