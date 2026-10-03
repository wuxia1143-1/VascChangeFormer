from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..training.folds import (
    build_patient_fold_plan,
    fold_plan_checksum,
    save_patient_fold_plan,
)
from ..training.metrics import change_space_metrics, regression_metrics
from ..training.v22_nested import V22_MODELS, run_v22_nested_cross_validation
from .real_internal import load_prepared_real_cohort
from .v21_refinement import (
    _normalized_plan_checksum,
    _paired_bootstrap_metric_difference,
)


DISPLAY = {
    "persistence": "Persistence",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "apn_dr": "APN-DR",
    "itransformer_mtl": "iTransformer-MTL",
    "first_icu_mtl": "FIRST-ICU-MTL",
    "learning_to_route": "Learning to Route",
    "lac_itransformer": "LAC-iTransformer V1",
    "lac_v2": "LAC-iTransformer V2",
    "lac_v21": "LAC-iTransformer V2.1",
    "lac_v22_full": "LAC-iTransformer V2.2 Full",
    "lac_v22_no_adapters": "V2.2 w/o phenotype adapters",
    "lac_v22_no_coupling": "V2.2 w/o I-to-C coupling",
    "lac_v22_no_treatment": "V2.2 w/o treatment conditioning",
    "lac_v22_no_calibration": "V2.2 w/o clinical calibration",
}
COMPARISON_MODELS = (
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
    "lac_itransformer",
    "lac_v2",
    "lac_v21",
    "lac_v22_full",
)
ABLATION_MODELS = (
    "lac_v22_full",
    "lac_v22_no_adapters",
    "lac_v22_no_coupling",
    "lac_v22_no_treatment",
    "lac_v22_no_calibration",
)


def run_v22_model(
    prepared_path: str | Path,
    config: dict[str, Any],
    output_root: str | Path,
    model_name: str,
) -> dict[str, Any]:
    if model_name not in V22_MODELS:
        raise KeyError(model_name)
    arrays, schema = load_prepared_real_cohort(prepared_path)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    plan = build_patient_fold_plan(arrays["patient_ids"], 5, seed)
    locked_path = root / "locked_patient_outer_fold_plan.csv"
    if locked_path.is_file():
        frame = pd.read_csv(locked_path)
        existing = {
            str(row.patient_id): int(row.test_fold)
            for row in frame.itertuples(index=False)
        }
        if fold_plan_checksum(existing) != fold_plan_checksum(plan):
            raise RuntimeError("Locked V2.2 outer plan is incompatible")
    else:
        save_patient_fold_plan(
            plan,
            locked_path,
            seed=seed,
            n_folds=5,
        )
    output = root / model_name
    summary_path = output / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial output will not be overwritten: {output}")
    return run_v22_nested_cross_validation(
        arrays,
        schema,
        config,
        output,
        model_name,
        outer_fold_assignments=plan,
    )


def _summary_row(
    model: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = summary["pooled_oof_metrics"] | summary[
        "pooled_change_metrics"
    ]
    parameter_count = summary.get("parameter_count")
    if parameter_count is None:
        counts = [
            record.get("parameter_count")
            for record in summary.get("fold_records", [])
            if record.get("parameter_count") is not None
        ]
        parameter_count = int(np.median(counts)) if counts else None
    return {
        "model": model,
        "display_name": DISPLAY[model],
        "parameter_count": parameter_count,
        **metrics,
    }


def _no_calibration_artifact(
    full_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(full_dir / "out_of_fold_predictions.csv")
    frame["pred_tbr"] = frame["raw_pred_tbr"]
    frame["pred_cac"] = frame["raw_pred_cac"]
    frame["pred_delta_tbr"] = frame["raw_pred_delta_tbr"]
    frame["pred_delta_log_cac"] = frame["raw_pred_delta_log_cac"]
    frame.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    targets = frame[["true_tbr", "true_cac"]].to_numpy()
    prediction = frame[["pred_tbr", "pred_cac"]].to_numpy()
    baseline = frame[["baseline_tbr", "baseline_cac"]].to_numpy()
    summary = {
        "model_name": "lac_v22_no_calibration",
        "patient_count": len(frame),
        "parameter_count": json.loads(
            (full_dir / "summary.json").read_text(encoding="utf-8")
        )["parameter_count"],
        "pooled_oof_metrics": regression_metrics(targets, prediction),
        "pooled_change_metrics": change_space_metrics(
            targets,
            prediction,
            baseline,
        ),
        "derived_from_same_full_neural_core": True,
        "outer_test_used_for_derivation": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def aggregate_v22_results(
    v22_root: str | Path,
    previous_root: str | Path,
    v21_root: str | Path,
    output_dir: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    v22_root = Path(v22_root)
    previous_root = Path(previous_root)
    v21_root = Path(v21_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    no_calibration = _no_calibration_artifact(
        v22_root / "lac_v22_full",
        v22_root / "lac_v22_no_calibration",
    )
    directories = {
        model: (
            v22_root / model
            if model.startswith("lac_v22")
            else (
                v21_root / "lac_v21"
                if model == "lac_v21"
                else previous_root / model
            )
        )
        for model in set(COMPARISON_MODELS + ABLATION_MODELS)
    }
    summaries = {}
    rows = []
    for model, directory in directories.items():
        if model == "lac_v22_no_calibration":
            summary = no_calibration
        else:
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
        summaries[model] = summary
        rows.append(_summary_row(model, summary))
    results = pd.DataFrame(rows)
    comparison = (
        results.set_index("model")
        .loc[list(COMPARISON_MODELS)]
        .reset_index()
    )
    ablation = (
        results.set_index("model")
        .loc[list(ABLATION_MODELS)]
        .reset_index()
    )
    comparison.to_csv(output / "v22_comparison_results.csv", index=False)
    ablation.to_csv(output / "v22_ablation_results.csv", index=False)

    v2_summary = summaries["lac_v2"]
    outer_checksum = summaries["lac_v22_full"]["cv_protocol"][
        "outer_fold_plan_checksum"
    ]
    split_records = []
    for model in V22_MODELS:
        model_summary = summaries[model]
        model_checksum = model_summary["cv_protocol"][
            "outer_fold_plan_checksum"
        ]
        for fold in range(1, 6):
            v2_plan = (
                previous_root
                / "lac_v2"
                / f"fold_{fold}"
                / "inner_patient_fold_plan.csv"
            )
            candidate_plan = (
                v22_root
                / model
                / f"fold_{fold}"
                / "inner_patient_fold_plan.csv"
            )
            split_records.append(
                {
                    "model": model,
                    "outer_fold": fold,
                    "outer_plan_identical": (
                        model_checksum
                        == v2_summary["cv_protocol"][
                            "outer_fold_plan_checksum"
                        ]
                    ),
                    "inner_plan_identical": (
                        _normalized_plan_checksum(v2_plan)
                        == _normalized_plan_checksum(candidate_plan)
                    ),
                }
            )
    split_frame = pd.DataFrame(split_records)
    split_frame.to_csv(output / "v22_split_audit.csv", index=False)
    if not split_frame[
        ["outer_plan_identical", "inner_plan_identical"]
    ].to_numpy().all():
        raise RuntimeError("V2.2 did not reuse the prior patient fold plans")

    paired_rows = []
    full_path = directories["lac_v22_full"] / "out_of_fold_predictions.csv"
    for reference in (
        "persistence",
        "xgboost",
        "itransformer_mtl",
        "lac_v2",
        "lac_v21",
        "lac_v22_no_adapters",
        "lac_v22_no_coupling",
        "lac_v22_no_treatment",
        "lac_v22_no_calibration",
    ):
        reference_path = (
            directories[reference] / "out_of_fold_predictions.csv"
        )
        for metric in (
            "tbr_mae",
            "delta_tbr_rmse",
            "delta_tbr_r2",
            "delta_log_cac_mae",
            "delta_log_cac_rmse",
            "delta_log_cac_r2",
        ):
            paired_rows.append(
                {
                    "candidate": "lac_v22_full",
                    "reference": reference,
                }
                | _paired_bootstrap_metric_difference(
                    full_path,
                    reference_path,
                    metric,
                    bootstrap_replicates,
                    seed,
                )
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v22_paired_bootstrap.csv", index=False)

    ablation_index = ablation.set_index("model")
    full_tbr = float(ablation_index.loc["lac_v22_full", "tbr_mae"])
    full_cac = float(
        ablation_index.loc["lac_v22_full", "delta_log_cac_mae"]
    )
    full_best_tbr = bool(
        full_tbr
        <= ablation_index.drop("lac_v22_full")["tbr_mae"].min()
    )
    full_best_cac = bool(
        full_cac
        <= ablation_index.drop("lac_v22_full")[
            "delta_log_cac_mae"
        ].min()
    )
    comparison_index = comparison.set_index("model")
    full_best_comparison_tbr = bool(
        float(comparison_index.loc["lac_v22_full", "tbr_mae"])
        <= comparison_index.drop("lac_v22_full")["tbr_mae"].min()
    )
    full_best_comparison_cac = bool(
        float(
            comparison_index.loc[
                "lac_v22_full",
                "delta_log_cac_mae",
            ]
        )
        <= comparison_index.drop("lac_v22_full")[
            "delta_log_cac_mae"
        ].min()
    )
    result = {
        "status": "v22_optimization_complete",
        "outer_fold_plan_checksum": outer_checksum,
        "all_outer_and_inner_plans_identical_to_v2": True,
        "comparison_full_best_tbr_mae": full_best_comparison_tbr,
        "comparison_full_best_delta_log_cac_mae": (
            full_best_comparison_cac
        ),
        "ablation_full_best_tbr_mae": full_best_tbr,
        "ablation_full_best_delta_log_cac_mae": full_best_cac,
        "bootstrap_replicates": bootstrap_replicates,
        "development_interpretation": (
            "iterative internal development; not independent confirmation"
        ),
        "outer_test_used_for_current_fold_training_or_calibration": False,
        "github_push_performed": False,
    }
    (output / "v22_optimization_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
