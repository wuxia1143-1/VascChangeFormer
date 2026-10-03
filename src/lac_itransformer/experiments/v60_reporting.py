from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_MODEL = "lac_v40_final_no_tail"
STAGE_MODELS = (
    "lac_v60_a_no_i2c_tail",
    "lac_v60_b_no_mechanism_features",
    "lac_v60_c_generic_temporal",
)
CORE_MODELS = (
    "lac_v60_no_task_adapters",
    "lac_v60_no_baseline_anchoring",
    "lac_v60_no_oof_calibration",
    "lac_v60_shared_only_mtl",
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


def _load_oof(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str}).sort_values(
        "patient_id"
    ).reset_index(drop=True)
    required = {
        "patient_id",
        "outer_fold",
        "true_delta_tbr",
        "pred_delta_tbr",
        "true_delta_log_cac",
        "pred_delta_log_cac",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF file is missing columns: {sorted(missing)}")
    if frame["patient_id"].duplicated().any():
        raise ValueError("OOF file contains duplicate patients")
    return frame


def _validate_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if not reference["patient_id"].equals(candidate["patient_id"]):
        raise ValueError("Reference and candidate patient IDs are not aligned")
    if not reference["outer_fold"].equals(candidate["outer_fold"]):
        raise ValueError("Reference and candidate outer folds differ")
    for column in ("true_delta_tbr", "true_delta_log_cac"):
        if not np.allclose(
            reference[column].to_numpy(float),
            candidate[column].to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"Reference and candidate targets differ: {column}")


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    result = {}
    for task in ("tbr", "log_cac"):
        values = _metrics(
            frame[f"true_delta_{task}"].to_numpy(float),
            frame[f"pred_delta_{task}"].to_numpy(float),
        )
        result.update({f"delta_{task}_{key}": value for key, value in values.items()})
    return result


def _paired_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    candidate_name: str,
    reference_name: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(reference)
    rows = []
    for task in ("tbr", "log_cac"):
        target = reference[f"true_delta_{task}"].to_numpy(float)
        candidate_prediction = candidate[f"pred_delta_{task}"].to_numpy(float)
        reference_prediction = reference[f"pred_delta_{task}"].to_numpy(float)
        for metric in ("mae", "rmse", "r2"):
            observed = (
                _metrics(target, candidate_prediction)[metric]
                - _metrics(target, reference_prediction)[metric]
            )
            values = np.empty(int(replicates), dtype=float)
            for index in range(int(replicates)):
                sample = rng.integers(0, n, size=n)
                values[index] = (
                    _metrics(target[sample], candidate_prediction[sample])[metric]
                    - _metrics(target[sample], reference_prediction[sample])[metric]
                )
            rows.append(
                {
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "metric": f"delta_{task}_{metric}",
                    "difference_definition": "candidate_minus_reference",
                    "estimate": float(observed),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                }
            )
    return rows


def _select_stage(stage: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    a = stage.get(STAGE_MODELS[0])
    b = stage.get(STAGE_MODELS[1])
    c = stage.get(STAGE_MODELS[2])
    if a and a["passes_both_mae"]:
        if b and b["passes_both_mae"]:
            return STAGE_MODELS[1], "V6-A and V6-B pass; select simpler V6-B and do not use V6-C"
        return STAGE_MODELS[0], "V6-A passes and V6-B fails; retain V6-A and do not use V6-C"
    if b and b["passes_both_mae"]:
        return STAGE_MODELS[1], "V6-A fails and V6-B passes; select V6-B"
    if c and c["passes_both_mae"]:
        return STAGE_MODELS[2], "V6-A and V6-B fail; V6-C passes after generic temporal context"
    return None, "No V6 stage passes both locked MAE margins; retain V4.0-final"


def evaluate_v60(
    reference_oof: str | Path,
    candidate_root: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    reference = _load_oof(reference_oof)
    root = Path(candidate_root)
    frames = {REFERENCE_MODEL: reference}
    for name in STAGE_MODELS + CORE_MODELS:
        path = root / name / "out_of_fold_predictions.csv"
        if path.is_file():
            frame = _load_oof(path)
            _validate_alignment(reference, frame)
            frames[name] = frame

    metrics = {name: _summary(frame) for name, frame in frames.items()}
    reference_metrics = metrics[REFERENCE_MODEL]
    stage = {}
    for name in STAGE_MODELS:
        if name not in metrics:
            continue
        delta_tbr = (
            metrics[name]["delta_tbr_mae"]
            - reference_metrics["delta_tbr_mae"]
        )
        delta_cac = (
            metrics[name]["delta_log_cac_mae"]
            - reference_metrics["delta_log_cac_mae"]
        )
        stage[name] = {
            "delta_tbr_mae_candidate_minus_v40": float(delta_tbr),
            "delta_log_cac_mae_candidate_minus_v40": float(delta_cac),
            "passes_tbr_mae": bool(delta_tbr <= 0.01 + 1e-12),
            "passes_cac_mae": bool(delta_cac <= 0.01 + 1e-12),
            "passes_both_mae": bool(
                delta_tbr <= 0.01 + 1e-12 and delta_cac <= 0.01 + 1e-12
            ),
        }
    selected, reason = _select_stage(stage)

    bootstrap = []
    for index, name in enumerate(frames):
        if name == REFERENCE_MODEL:
            continue
        bootstrap.extend(
            _paired_bootstrap(
                frames[name],
                reference,
                name,
                REFERENCE_MODEL,
                replicates,
                seed + index * 10_000,
            )
        )
    core = {}
    if selected is not None:
        selected_frame = frames[selected]
        selected_metrics = metrics[selected]
        for index, name in enumerate(CORE_MODELS):
            if name not in frames:
                continue
            core[name] = {
                metric: float(metrics[name][metric] - selected_metrics[metric])
                for metric in selected_metrics
            }
            bootstrap.extend(
                _paired_bootstrap(
                    frames[name],
                    selected_frame,
                    name,
                    selected,
                    replicates,
                    seed + 100_000 + index * 10_000,
                )
            )

    return {
        "reference_model": REFERENCE_MODEL,
        "patient_count": int(len(reference)),
        "outer_folds": sorted(reference["outer_fold"].unique().tolist()),
        "difference_definition": "candidate_minus_reference",
        "mae_noninferiority_margin": 0.01,
        "metrics": metrics,
        "stage_acceptance": stage,
        "selected_v60_full": selected,
        "selection_reason": reason,
        "retain_v40_final": selected is None,
        "external_evaluation_authorized": False,
        "core_ablation_difference_from_selected_full": core,
        "bootstrap": bootstrap,
    }


def _write_outputs(result: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = [
        {"model": name, **values}
        for name, values in result["metrics"].items()
    ]
    pd.DataFrame(metric_rows).to_csv(output / "v60_internal_metrics.csv", index=False)
    pd.DataFrame(result["bootstrap"]).to_csv(
        output / "v60_paired_bootstrap_95ci.csv", index=False
    )
    acceptance = {key: value for key, value in result.items() if key != "bootstrap"}
    (output / "v60_internal_acceptance.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-oof", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = evaluate_v60(
        args.reference_oof,
        args.candidate_root,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    _write_outputs(result, args.output)
    print(
        json.dumps(
            {
                "selected_v60_full": result["selected_v60_full"],
                "selection_reason": result["selection_reason"],
                "stage_acceptance": result["stage_acceptance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
