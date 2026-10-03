from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from ..training.metrics import (
    bootstrap_change_metrics,
    bootstrap_metrics,
    change_space_metrics,
    regression_metrics,
)
from ..training.v27_nested import _v27_diagnostics


DERIVED_MODES = {
    "lac_v40_final_no_calibration": "no_calibration",
    "lac_v40_final_no_tail": "no_tail",
    "lac_v40_final_central_only": "central",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_for_mode(frame: pd.DataFrame, mode: str) -> np.ndarray:
    central = frame["median_pred_delta_log_cac"].to_numpy(float)
    corrected = frame["mean_pred_delta_log_cac"].to_numpy(float)
    tail_weight = frame["decision_gate_cac"].to_numpy(float)
    final = frame["pred_delta_log_cac"].to_numpy(float)
    if mode == "no_calibration":
        return central + tail_weight * (corrected - central)
    if mode == "no_tail":
        return final - tail_weight * (corrected - central)
    if mode == "central":
        return central
    raise KeyError(mode)


def _update_predictions(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    result = frame.copy()
    residual = _prediction_for_mode(result, mode)
    result["pred_delta_log_cac"] = residual
    result["pred_cac"] = np.maximum(
        0.0,
        np.expm1(
            np.log1p(np.maximum(result["baseline_cac"].to_numpy(float), 0.0))
            + residual
        ),
    )
    if mode in {"no_tail", "central"}:
        result["decision_gate_cac"] = 0.0
    return result


def _metrics(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    targets = frame[["true_tbr", "true_cac"]].to_numpy(float)
    predictions = frame[["pred_tbr", "pred_cac"]].to_numpy(float)
    baseline = frame[["baseline_tbr", "baseline_cac"]].to_numpy(float)
    return (
        regression_metrics(targets, predictions),
        change_space_metrics(targets, predictions, baseline),
    )


def derive_decision_mode(
    source_directory: str | Path,
    output_directory: str | Path,
    model_name: str,
    mode: str,
    copy_fold_artifacts: bool = False,
) -> dict[str, Any]:
    if mode not in {"no_calibration", "no_tail", "central"}:
        raise KeyError(mode)
    source = Path(source_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    required = (
        "summary.json",
        "out_of_fold_predictions.csv",
        "outer_fold_metrics.csv",
        "patient_outer_fold_plan.csv",
        "patient_outer_fold_plan.json",
        "training_config.json",
    )
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    summary = _read_json(source / "summary.json")
    source_model_name = str(summary["model_name"])
    source_frame = pd.read_csv(source / "out_of_fold_predictions.csv")
    if source_frame["patient_id"].duplicated().any():
        raise RuntimeError("Full-model OOF predictions contain duplicate patients")
    frame = _update_predictions(source_frame, mode)
    endpoint, change = _metrics(frame)

    output.mkdir(parents=True)
    for name in required[3:]:
        shutil.copy2(source / name, output / name)
    if copy_fold_artifacts:
        for fold in range(1, 6):
            shutil.copytree(source / f"fold_{fold}", output / f"fold_{fold}")
    frame.to_csv(output / "out_of_fold_predictions.csv", index=False)

    source_fold_metrics = pd.read_csv(source / "outer_fold_metrics.csv")
    source_records = {
        int(record["outer_fold"]): record for record in summary["fold_records"]
    }
    fold_rows = []
    fold_records = []
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        subset = frame.loc[frame["outer_fold"].astype(int) == fold]
        fold_endpoint, fold_change = _metrics(subset)
        record = dict(source_records[fold])
        for key in list(record):
            if key in fold_endpoint or key in fold_change:
                record.pop(key)
        record.update(fold_endpoint)
        record.update(fold_change)
        record["derived_from_full_neural_state"] = True
        record["derived_decision_mode"] = mode
        fold_records.append(record)
        row = dict(source_fold_metrics.loc[
            source_fold_metrics["outer_fold"].astype(int) == fold
        ].iloc[0])
        row.update(fold_endpoint)
        row.update(fold_change)
        row["derived_from_full_neural_state"] = True
        row["derived_decision_mode"] = mode
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(output / "outer_fold_metrics.csv", index=False)

    targets = frame[["true_tbr", "true_cac"]].to_numpy(float)
    predictions = frame[["pred_tbr", "pred_cac"]].to_numpy(float)
    baseline = frame[["baseline_tbr", "baseline_cac"]].to_numpy(float)
    config = _read_json(output / "training_config.json")
    replicates = int(config.get("bootstrap_replicates", 2000))
    seed = int(config.get("seed", 2026))
    derived = dict(summary)
    derived.update(
        {
            "model_name": model_name,
            "fold_records": fold_records,
            "pooled_oof_metrics": endpoint,
            "pooled_change_metrics": change,
            "diagnostics": _v27_diagnostics(frame),
            "bootstrap_95_ci": bootstrap_metrics(
                targets, predictions, n_bootstrap=replicates, seed=seed
            ),
            "change_bootstrap_95_ci": bootstrap_change_metrics(
                targets,
                predictions,
                baseline,
                n_bootstrap=replicates,
                seed=seed,
            ),
            "derived_ablation_audit": {
                "source_model": source_model_name,
                "source_summary": str(source / "summary.json"),
                "source_summary_sha256": _sha256(source / "summary.json"),
                "source_oof_sha256": _sha256(
                    source / "out_of_fold_predictions.csv"
                ),
                "neural_training_repeated": False,
                "neural_state": "exactly_shared_with_full_model",
                "changed_component": mode,
                "fold_artifacts_copied": bool(copy_fold_artifacts),
                "outer_test_labels_used_for_derivation": False,
            },
        }
    )
    derived["cv_protocol"] = dict(derived["cv_protocol"])
    derived["cv_protocol"]["decision_mode"] = {
        "tbr": "identity",
        "cac": mode,
    }
    (output / "summary.json").write_text(
        json.dumps(derived, indent=2), encoding="utf-8"
    )
    return derived


def derive_decision_ablation(
    full_directory: str | Path,
    output_directory: str | Path,
    model_name: str,
) -> dict[str, Any]:
    if model_name not in DERIVED_MODES:
        raise KeyError(model_name)
    source = Path(full_directory).resolve()
    summary = _read_json(source / "summary.json")
    if summary["model_name"] != "lac_v40_final_full":
        raise ValueError("Decision ablations must be derived from V4.0-final full")
    return derive_decision_mode(
        source,
        output_directory,
        model_name,
        DERIVED_MODES[model_name],
        copy_fold_artifacts=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--models", nargs="+", default=list(DERIVED_MODES)
    )
    args = parser.parse_args()
    result = {}
    for model_name in args.models:
        result[model_name] = derive_decision_ablation(
            args.full, Path(args.output_root) / model_name, model_name
        )["pooled_change_metrics"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
