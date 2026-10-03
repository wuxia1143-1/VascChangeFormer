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
MODEL_ORDER = (
    "lac_v40_final_no_tail",
    "lac_v42_strict_no_tail",
    "lac_v42_strict_no_tail_varcal",
    "lac_v42_strict_no_tail_two_stage",
)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, float) - np.asarray(target, float)
    target = np.asarray(target, float)
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / max(denominator, 1e-12)),
    }


def _load_oof(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "outer_fold"}
    for true_name, prediction_name in TASKS.values():
        required.update((true_name, prediction_name))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF file missing columns: {missing}")
    if frame["patient_id"].duplicated().any():
        raise ValueError("OOF file contains duplicate patients")
    if sorted(frame["outer_fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("OOF file must contain all five outer folds")
    return frame.sort_values("patient_id").reset_index(drop=True)


def _validate_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if not reference["patient_id"].equals(candidate["patient_id"]):
        raise ValueError("Candidate and reference patient sets differ")
    if not np.array_equal(reference["outer_fold"], candidate["outer_fold"]):
        raise ValueError("Candidate and reference outer folds differ")
    for target_name, _ in TASKS.values():
        if not np.allclose(
            reference[target_name].to_numpy(float),
            candidate[target_name].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Candidate target changed: {target_name}")


def _summary(frame: pd.DataFrame, tail_mask: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for task, (target_name, prediction_name) in TASKS.items():
        metrics = _metrics(
            frame[target_name].to_numpy(float),
            frame[prediction_name].to_numpy(float),
        )
        result.update({f"{task}_{key}": value for key, value in metrics.items()})
    cac_target = frame[TASKS["cac"][0]].to_numpy(float)
    cac_prediction = frame[TASKS["cac"][1]].to_numpy(float)
    result["cac_top_decile_rmse"] = _metrics(
        cac_target[tail_mask], cac_prediction[tail_mask]
    )["rmse"]
    return result


def _paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    tail_threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(reference)
    names = (
        "tbr_mae",
        "tbr_rmse",
        "tbr_r2",
        "cac_mae",
        "cac_rmse",
        "cac_r2",
        "cac_top_decile_rmse",
    )
    differences = {name: np.empty(replicates, float) for name in names}
    locked_tail = (
        np.abs(reference[TASKS["cac"][0]].to_numpy(float)) >= tail_threshold
    )
    observed_reference = _summary(reference, locked_tail)
    observed_candidate = _summary(candidate, locked_tail)
    for replicate in range(replicates):
        index = rng.integers(0, n, size=n)
        reference_sample = reference.iloc[index].reset_index(drop=True)
        candidate_sample = candidate.iloc[index].reset_index(drop=True)
        tail = (
            np.abs(reference_sample[TASKS["cac"][0]].to_numpy(float))
            >= tail_threshold
        )
        # A bootstrap sample can theoretically miss the locked tail subset.
        if not tail.any():
            tail[np.argmax(
                np.abs(reference_sample[TASKS["cac"][0]].to_numpy(float))
            )] = True
        reference_metrics = _summary(reference_sample, tail)
        candidate_metrics = _summary(candidate_sample, tail)
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


def _fold_cac_rmse_improvements(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[int, list[dict[str, float | int | bool]]]:
    records = []
    for fold in range(1, 6):
        ref = reference[reference["outer_fold"] == fold]
        cand = candidate[candidate["outer_fold"] == fold]
        target_name, prediction_name = TASKS["cac"]
        ref_rmse = _metrics(
            ref[target_name].to_numpy(float), ref[prediction_name].to_numpy(float)
        )["rmse"]
        cand_rmse = _metrics(
            cand[target_name].to_numpy(float), cand[prediction_name].to_numpy(float)
        )["rmse"]
        records.append(
            {
                "outer_fold": fold,
                "reference_rmse": ref_rmse,
                "candidate_rmse": cand_rmse,
                "improved": bool(cand_rmse < ref_rmse),
            }
        )
    return sum(int(item["improved"]) for item in records), records


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    error_names = ("tbr_mae", "tbr_rmse", "cac_mae", "cac_rmse")
    score_names = ("tbr_r2", "cac_r2")
    non_worse = all(left[name] <= right[name] for name in error_names) and all(
        left[name] >= right[name] for name in score_names
    )
    strictly_better = any(left[name] < right[name] for name in error_names) or any(
        left[name] > right[name] for name in score_names
    )
    return bool(non_worse and strictly_better)


def evaluate_v42(
    reference_path: str | Path,
    candidate_paths: dict[str, str | Path],
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    reference = _load_oof(reference_path)
    if len(reference) != 443:
        raise ValueError("Locked V4.2 assessment requires exactly 443 patients")
    candidates = {name: _load_oof(path) for name, path in candidate_paths.items()}
    if tuple(candidates) != MODEL_ORDER[1:]:
        raise ValueError("V4.2 candidates must be supplied in preregistered order")
    for candidate in candidates.values():
        _validate_alignment(reference, candidate)

    cac_target = reference[TASKS["cac"][0]].to_numpy(float)
    tail_threshold = float(np.quantile(np.abs(cac_target), 0.90))
    tail_mask = np.abs(cac_target) >= tail_threshold
    frames = {MODEL_ORDER[0]: reference} | candidates
    summaries = {name: _summary(frame, tail_mask) for name, frame in frames.items()}
    bootstrap = {
        name: _paired_bootstrap(
            reference,
            candidate,
            tail_threshold,
            int(replicates),
            int(seed),
        )
        for name, candidate in candidates.items()
    }

    acceptance: dict[str, Any] = {}
    reference_metrics = summaries[MODEL_ORDER[0]]
    simpler: list[str] = [MODEL_ORDER[0]]
    for name in MODEL_ORDER[1:]:
        current = summaries[name]
        ci = bootstrap[name]
        improved_folds, fold_records = _fold_cac_rmse_improvements(
            reference, candidates[name]
        )
        cac_ci_favors = ci["cac_rmse"]["ci95_high"] < 0.0
        cac_ci_noninferior = ci["cac_rmse"]["ci95_high"] <= (
            0.005 * reference_metrics["cac_rmse"]
        )
        criteria = {
            "cac_mae_within_0_5pct": current["cac_mae"]
            <= 1.005 * reference_metrics["cac_mae"],
            "cac_rmse_point_improves_and_ci_acceptable": (
                current["cac_rmse"] < reference_metrics["cac_rmse"]
                and (cac_ci_favors or cac_ci_noninferior)
            ),
            "cac_r2_point_improves": current["cac_r2"]
            > reference_metrics["cac_r2"],
            "cac_top_decile_rmse_improves": current["cac_top_decile_rmse"]
            < reference_metrics["cac_top_decile_rmse"],
            "tbr_noninferior": (
                current["tbr_mae"] <= 1.005 * reference_metrics["tbr_mae"]
                and current["tbr_rmse"] <= 1.005 * reference_metrics["tbr_rmse"]
                and current["tbr_r2"]
                >= reference_metrics["tbr_r2"]
                - 0.005 * abs(reference_metrics["tbr_r2"])
            ),
            "outer_fold_consistency": (
                improved_folds >= 4 or (improved_folds >= 3 and cac_ci_favors)
            ),
        }
        dominators = [
            prior for prior in simpler if _dominates(summaries[prior], current)
        ]
        criteria["not_pareto_dominated"] = not dominators
        acceptance[name] = {
            "passed": bool(all(criteria.values())),
            "criteria": {key: bool(value) for key, value in criteria.items()},
            "cac_rmse_improved_outer_folds": improved_folds,
            "outer_fold_records": fold_records,
            "pareto_dominators": dominators,
        }
        simpler.append(name)

    passing = [name for name in MODEL_ORDER[1:] if acceptance[name]["passed"]]
    return {
        "scope": "locked 443-patient internal development only",
        "external_labels_or_predictions_read": False,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "tail_definition": {
            "absolute_true_delta_log_cac_quantile": 0.90,
            "threshold": tail_threshold,
            "patient_count": int(tail_mask.sum()),
        },
        "metrics": summaries,
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
        "selected_v42_candidate": passing[0] if passing else None,
        "final_internal_decision": (
            "freeze_selected_v42_for_new_external_confirmation"
            if passing
            else "retain_v40_final"
        ),
    }


def _write_outputs(result: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame.from_dict(result["metrics"], orient="index")
    metrics.index.name = "model"
    metrics.reset_index().to_csv(destination / "v42_internal_metrics.csv", index=False)
    bootstrap_rows = []
    for model, records in result["paired_bootstrap"].items():
        for metric, values in records.items():
            bootstrap_rows.append({"model": model, "metric": metric} | values)
    pd.DataFrame(bootstrap_rows).to_csv(
        destination / "v42_paired_bootstrap_95ci.csv", index=False
    )
    (destination / "v42_internal_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    root = Path(arguments.candidate_root)
    candidates = {
        name: root / name / "out_of_fold_predictions.csv"
        for name in MODEL_ORDER[1:]
    }
    result = evaluate_v42(
        arguments.reference,
        candidates,
        arguments.bootstrap,
        arguments.seed,
    )
    _write_outputs(result, arguments.output)
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
