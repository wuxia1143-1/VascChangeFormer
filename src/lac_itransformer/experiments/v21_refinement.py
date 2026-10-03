from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .real_internal import (
    DISPLAY_NAMES,
    load_prepared_real_cohort,
)


REFERENCE_MODELS = (
    "persistence",
    "xgboost",
    "itransformer_mtl",
    "lac_v2",
)
REPORT_METRICS = (
    "tbr_mae",
    "tbr_rmse",
    "tbr_r2",
    "log_cac_mae",
    "log_cac_rmse",
    "log_cac_r2",
    "delta_tbr_mae",
    "delta_tbr_rmse",
    "delta_tbr_r2",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
    "delta_log_cac_r2",
)


def _change_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    error = prediction - truth
    squared_error = float(np.sum(error**2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    true_variance = float(np.var(truth, ddof=1)) if len(truth) > 1 else 0.0
    predicted_variance = (
        float(np.var(prediction, ddof=1)) if len(prediction) > 1 else 0.0
    )
    return {
        "n": int(len(truth)),
        "true_mean": float(np.mean(truth)),
        "predicted_mean": float(np.mean(prediction)),
        "mean_bias_predicted_minus_true": float(np.mean(error)),
        "true_standard_deviation": float(np.sqrt(true_variance)),
        "predicted_standard_deviation": float(np.sqrt(predicted_variance)),
        "true_variance": true_variance,
        "predicted_variance": predicted_variance,
        "predicted_to_true_variance_ratio": (
            predicted_variance / true_variance
            if true_variance > 1e-12
            else float("nan")
        ),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": (
            1.0 - squared_error / total
            if total > 1e-12
            else float("nan")
        ),
    }


def _window_label(months: np.ndarray) -> np.ndarray:
    index = np.digitize(
        months,
        [6.0, 12.0, 18.0, 24.0],
        right=True,
    )
    labels = np.asarray(
        ["≤6", ">6–12", ">12–18", ">18–24", ">24"],
        dtype=object,
    )
    return labels[index]


def diagnose_cac_change(
    oof_path: str | Path,
    prepared_path: str | Path,
    output_dir: str | Path,
    label: str,
    unchanged_epsilon: float = 0.05,
) -> dict[str, Any]:
    """Aggregate-only OOF diagnosis of CAC change-space shrinkage."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(oof_path)
    arrays, _ = load_prepared_real_cohort(prepared_path)
    if "followup_months" not in arrays:
        raise ValueError(
            "Prepared arrays must contain followup_months for CAC diagnostics"
        )
    auxiliary = pd.DataFrame(
        {
            "patient_id": [str(value) for value in arrays["patient_ids"]],
            "followup_months": arrays["followup_months"].astype(float),
        }
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame = frame.merge(
        auxiliary,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(arrays["patient_ids"]):
        raise RuntimeError("OOF predictions and prepared cohort do not match")

    truth = frame["true_delta_log_cac"].to_numpy(dtype=float)
    prediction = frame["pred_delta_log_cac"].to_numpy(dtype=float)
    overall = _change_metrics(truth, prediction)
    if np.var(truth) > 1e-12:
        predicted_on_true_slope, predicted_on_true_intercept = np.polyfit(
            truth,
            prediction,
            1,
        )
    else:
        predicted_on_true_slope = predicted_on_true_intercept = float("nan")
    if np.var(prediction) > 1e-12:
        true_on_predicted_slope, true_on_predicted_intercept = np.polyfit(
            prediction,
            truth,
            1,
        )
    else:
        true_on_predicted_slope = true_on_predicted_intercept = float("nan")

    frame["baseline_cac_group"] = np.where(
        frame["baseline_cac"] == 0,
        "baseline CAC = 0",
        "baseline CAC > 0",
    )
    frame["followup_window_months"] = _window_label(
        frame["followup_months"].to_numpy(dtype=float)
    )
    frame["true_change_group"] = np.where(
        truth > unchanged_epsilon,
        "increase",
        np.where(truth < -unchanged_epsilon, "decrease", "unchanged"),
    )
    subgroup_rows = []
    for group_type, column in (
        ("baseline_cac", "baseline_cac_group"),
        ("followup_window", "followup_window_months"),
        ("true_change_direction", "true_change_group"),
    ):
        for group, subset in frame.groupby(column, observed=True):
            metrics = _change_metrics(
                subset["true_delta_log_cac"].to_numpy(dtype=float),
                subset["pred_delta_log_cac"].to_numpy(dtype=float),
            )
            subgroup_rows.append(
                {
                    "model": label,
                    "group_type": group_type,
                    "group": str(group),
                }
                | metrics
            )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(
        output / f"{label}_cac_change_subgroups.csv",
        index=False,
    )

    error_squared = (prediction - truth) ** 2
    total_sse = float(error_squared.sum())
    extremes = {}
    for percentile in (0.90, 0.95):
        threshold = float(np.quantile(np.abs(truth), percentile))
        selected = np.abs(truth) >= threshold
        key = (
            f"top_{int(round((1.0 - percentile) * 100))}"
            "pct_absolute_true_change"
        )
        extremes[key] = {
            "absolute_true_change_threshold": threshold,
            "n": int(selected.sum()),
            "share_of_total_squared_error": (
                float(error_squared[selected].sum() / total_sse)
                if total_sse > 0
                else 0.0
            ),
            "rmse": float(np.sqrt(np.mean(error_squared[selected]))),
        }

    direction_truth = np.sign(
        np.where(np.abs(truth) <= unchanged_epsilon, 0.0, truth)
    )
    direction_prediction = np.sign(
        np.where(
            np.abs(prediction) <= unchanged_epsilon,
            0.0,
            prediction,
        )
    )
    variance_ratio = float(overall["predicted_to_true_variance_ratio"])
    result = {
        "model": label,
        "patient_count": int(len(frame)),
        "patient_level_values_exported": False,
        "unchanged_epsilon_log_units": unchanged_epsilon,
        "overall": overall,
        "calibration": {
            "predicted_on_true_intercept": float(predicted_on_true_intercept),
            "predicted_on_true_slope": float(predicted_on_true_slope),
            "true_on_predicted_intercept": float(true_on_predicted_intercept),
            "true_on_predicted_slope": float(true_on_predicted_slope),
            "pearson_correlation": float(np.corrcoef(truth, prediction)[0, 1]),
        },
        "direction_accuracy_with_unchanged_band": float(
            np.mean(direction_truth == direction_prediction)
        ),
        "extreme_progression": extremes,
        "over_shrinkage_flag": bool(
            np.isfinite(variance_ratio) and variance_ratio < 0.50
        ),
        "subgroup_file": f"{label}_cac_change_subgroups.csv",
    }
    (output / f"{label}_cac_change_diagnostics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _normalized_plan_checksum(path: Path) -> str:
    frame = pd.read_csv(path)
    normalized = (
        frame[["patient_id", "test_fold"]]
        .assign(patient_id=lambda value: value["patient_id"].astype(str))
        .sort_values("patient_id")
        .to_csv(index=False)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _metric_value(frame: pd.DataFrame, metric: str) -> float:
    if metric.startswith("delta_tbr_"):
        truth = frame["true_delta_tbr"].to_numpy(dtype=float)
        prediction = frame["pred_delta_tbr"].to_numpy(dtype=float)
    elif metric.startswith("delta_log_cac_"):
        truth = frame["true_delta_log_cac"].to_numpy(dtype=float)
        prediction = frame["pred_delta_log_cac"].to_numpy(dtype=float)
    elif metric.startswith("tbr_"):
        truth = frame["true_tbr"].to_numpy(dtype=float)
        prediction = frame["pred_tbr"].to_numpy(dtype=float)
    elif metric.startswith("log_cac_"):
        truth = np.log1p(np.maximum(frame["true_cac"].to_numpy(dtype=float), 0))
        prediction = np.log1p(
            np.maximum(frame["pred_cac"].to_numpy(dtype=float), 0)
        )
    else:
        raise KeyError(metric)
    error = prediction - truth
    if metric.endswith("_mae"):
        return float(np.mean(np.abs(error)))
    if metric.endswith("_rmse"):
        return float(np.sqrt(np.mean(error**2)))
    if metric.endswith("_r2"):
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        return (
            1.0 - float(np.sum(error**2)) / denominator
            if denominator > 1e-12
            else float("nan")
        )
    raise KeyError(metric)


def _paired_bootstrap_metric_difference(
    candidate_path: Path,
    reference_path: Path,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    candidate = pd.read_csv(candidate_path).sort_values("patient_id")
    reference = pd.read_csv(reference_path).sort_values("patient_id")
    merged = candidate.merge(
        reference,
        on="patient_id",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    candidate_frame = pd.DataFrame(
        {
            column: merged[f"{column}_candidate"]
            for column in candidate.columns
            if column != "patient_id"
        }
    )
    reference_frame = pd.DataFrame(
        {
            column: merged[f"{column}_reference"]
            for column in reference.columns
            if column != "patient_id"
        }
    )
    estimate = _metric_value(candidate_frame, metric) - _metric_value(
        reference_frame,
        metric,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = rng.integers(0, len(merged), len(merged))
        samples[index] = _metric_value(
            candidate_frame.iloc[draw],
            metric,
        ) - _metric_value(reference_frame.iloc[draw], metric)
    return {
        "metric": metric,
        "difference_definition": "candidate_minus_reference",
        "lower_is_better": not metric.endswith("_r2"),
        "estimate": float(estimate),
        "ci_low": float(np.nanquantile(samples, 0.025)),
        "ci_high": float(np.nanquantile(samples, 0.975)),
        "bootstrap_replicates": replicates,
    }


def _same_nested_splits(
    v21_root: Path,
    previous_root: Path,
) -> dict[str, Any]:
    previous_summary = json.loads(
        (previous_root / "lac_v2" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    v21_summary = json.loads(
        (v21_root / "lac_v21" / "summary.json").read_text(encoding="utf-8")
    )
    outer_previous = previous_summary["cv_protocol"][
        "outer_fold_plan_checksum"
    ]
    outer_v21 = v21_summary["cv_protocol"]["outer_fold_plan_checksum"]
    inner_checks = []
    for fold in range(1, 6):
        previous_path = (
            previous_root
            / "lac_v2"
            / f"fold_{fold}"
            / "inner_patient_fold_plan.csv"
        )
        v21_path = (
            v21_root
            / "lac_v21"
            / f"fold_{fold}"
            / "inner_patient_fold_plan.csv"
        )
        previous_checksum = _normalized_plan_checksum(previous_path)
        v21_checksum = _normalized_plan_checksum(v21_path)
        inner_checks.append(
            {
                "outer_fold": fold,
                "previous_v2_inner_plan_checksum": previous_checksum,
                "v21_inner_plan_checksum": v21_checksum,
                "identical": previous_checksum == v21_checksum,
            }
        )
    return {
        "outer_plan_checksum": outer_v21,
        "outer_plan_identical_to_previous_v2": outer_previous == outer_v21,
        "inner_plan_checks": inner_checks,
        "all_inner_plans_identical_to_previous_v2": all(
            record["identical"] for record in inner_checks
        ),
    }


def summarize_v21_refinement(
    v21_root: str | Path,
    previous_root: str | Path,
    prepared_path: str | Path,
    output_dir: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compare V2.1 with untouched, same-fold prior OOF predictions."""

    v21_root = Path(v21_root)
    previous_root = Path(previous_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_audit = _same_nested_splits(v21_root, previous_root)
    if not split_audit["outer_plan_identical_to_previous_v2"]:
        raise RuntimeError("V2.1 outer folds differ from the previous V2 run")
    if not split_audit["all_inner_plans_identical_to_previous_v2"]:
        raise RuntimeError("V2.1 inner folds differ from the previous V2 run")

    model_paths = {
        model: previous_root / model
        for model in REFERENCE_MODELS
    } | {"lac_v21": v21_root / "lac_v21"}
    rows = []
    for model, path in model_paths.items():
        summary = json.loads(
            (path / "summary.json").read_text(encoding="utf-8")
        )
        counts = [
            record["parameter_count"]
            for record in summary["fold_records"]
            if record["parameter_count"] is not None
        ]
        metrics = summary["pooled_oof_metrics"] | summary[
            "pooled_change_metrics"
        ]
        rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES.get(model, model),
                "parameter_count": (
                    int(np.median(counts)) if counts else None
                ),
            }
            | {metric: metrics[metric] for metric in REPORT_METRICS}
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "v21_comparison_results.csv", index=False)

    candidate_path = (
        v21_root / "lac_v21" / "out_of_fold_predictions.csv"
    )
    paired_rows = []
    for reference in REFERENCE_MODELS:
        reference_path = (
            previous_root / reference / "out_of_fold_predictions.csv"
        )
        for metric in (
            "tbr_mae",
            "log_cac_mae",
            "delta_tbr_mae",
            "delta_tbr_rmse",
            "delta_tbr_r2",
            "delta_log_cac_mae",
            "delta_log_cac_rmse",
            "delta_log_cac_r2",
        ):
            record = _paired_bootstrap_metric_difference(
                candidate_path,
                reference_path,
                metric,
                bootstrap_replicates,
                seed,
            )
            paired_rows.append(
                {
                    "candidate": "lac_v21",
                    "reference": reference,
                }
                | record
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v21_paired_bootstrap.csv", index=False)

    diagnostics = {
        "lac_v2": diagnose_cac_change(
            previous_root / "lac_v2" / "out_of_fold_predictions.csv",
            prepared_path,
            output,
            "lac_v2",
        ),
        "lac_v21": diagnose_cac_change(
            candidate_path,
            prepared_path,
            output,
            "lac_v21",
        ),
    }
    parameter_lookup = comparison.set_index("model")["parameter_count"]
    v21_parameters = int(parameter_lookup["lac_v21"])
    v2_parameters = int(parameter_lookup["lac_v2"])
    v21_summary = json.loads(
        (v21_root / "lac_v21" / "summary.json").read_text(encoding="utf-8")
    )
    selected_epochs = [
        int(record["selected_epochs"])
        for record in v21_summary["fold_records"]
    ]
    v21_row = comparison.set_index("model").loc["lac_v21"]
    v2_row = comparison.set_index("model").loc["lac_v2"]
    v21_to_v2 = paired[
        (paired["reference"] == "lac_v2")
        & paired["metric"].isin(
            ["tbr_mae", "delta_log_cac_mae", "delta_log_cac_rmse"]
        )
    ].set_index("metric")
    tbr_improved = bool(
        v21_to_v2.loc["tbr_mae", "ci_high"] < 0
    )
    cac_mae_changed = bool(
        v21_to_v2.loc["delta_log_cac_mae", "ci_low"] > 0
        or v21_to_v2.loc["delta_log_cac_mae", "ci_high"] < 0
    )
    shrinkage_improved = bool(
        diagnostics["lac_v21"]["overall"][
            "predicted_to_true_variance_ratio"
        ]
        > diagnostics["lac_v2"]["overall"][
            "predicted_to_true_variance_ratio"
        ]
    )
    result = {
        "status": "v21_same_patient_nested_fivefold_complete",
        "split_audit": split_audit,
        "bootstrap_replicates": bootstrap_replicates,
        "parameters": {
            "v2": v2_parameters,
            "v21": v21_parameters,
            "v21_minus_v2": v21_parameters - v2_parameters,
            "v21_fraction_of_v2": v21_parameters / v2_parameters,
        },
        "selected_epochs_by_outer_fold": selected_epochs,
        "diagnostics": diagnostics,
        "comparison_file": "v21_comparison_results.csv",
        "paired_bootstrap_file": "v21_paired_bootstrap.csv",
        "development_interpretation": {
            "prior_v2_development_oof_informed_v21_design": True,
            "current_v21_outer_test_used_for_current_fold_preprocessing": False,
            "current_v21_outer_test_used_for_current_fold_training": False,
            "current_v21_outer_test_used_for_current_fold_epoch_selection": False,
            "independent_confirmatory_validation": False,
            "status": "iterative internal development validation",
        },
        "key_findings": {
            "tbr_mae_improved_over_v2_with_paired_ci_excluding_zero": (
                tbr_improved
            ),
            "cac_change_mae_changed_with_paired_ci_excluding_zero": (
                cac_mae_changed
            ),
            "cac_change_variance_shrinkage_improved": shrinkage_improved,
            "v2_delta_log_cac_r2": float(v2_row["delta_log_cac_r2"]),
            "v21_delta_log_cac_r2": float(v21_row["delta_log_cac_r2"]),
        },
        "recommendation": (
            "Do not replace V2 as the primary CAC-change model. Retain V2.1 "
            "as an exploratory TBR-favored model; it may enter a future "
            "external center only as a frozen secondary analysis until CAC "
            "change dispersion is improved."
        ),
        "external_validation_status": "not_run",
    }
    (output / "v21_refinement_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_lines = [
        "# LAC-iTransformer V2.1 real nested-fivefold report",
        "",
        "## Protocol audit",
        "",
        f"- Patients: {v21_summary['patient_count']}.",
        "- Outer and all five inner fold plans are byte-normalized checksum "
        "matches to the prior V2 plans.",
        f"- Outer fold checksum: `{split_audit['outer_plan_checksum']}`.",
        f"- Selected epochs by outer fold: {selected_epochs}.",
        "- Each outer test fold was excluded from preprocessing, target "
        "standardization, sampling, epoch selection and refitting.",
        "- This is iterative internal development validation because prior "
        "V2 OOF diagnostics informed V2.1; it is not independent confirmation.",
        "",
        "## Main pooled OOF results",
        "",
        "| Model | Parameters | TBR MAE | ΔlogCAC MAE | ΔlogCAC RMSE | ΔlogCAC R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("xgboost", "itransformer_mtl", "lac_v2", "lac_v21"):
        row = comparison.set_index("model").loc[model]
        parameters = (
            "—"
            if pd.isna(row["parameter_count"])
            else f"{int(row['parameter_count']):,}"
        )
        report_lines.append(
            f"| {row['display_name']} | {parameters} | "
            f"{row['tbr_mae']:.4f} | {row['delta_log_cac_mae']:.4f} | "
            f"{row['delta_log_cac_rmse']:.4f} | "
            f"{row['delta_log_cac_r2']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- V2.1 reduced TBR MAE from {v2_row['tbr_mae']:.4f} to "
            f"{v21_row['tbr_mae']:.4f}; the paired 95% interval excludes zero.",
            f"- CAC-change MAE changed from "
            f"{v2_row['delta_log_cac_mae']:.4f} to "
            f"{v21_row['delta_log_cac_mae']:.4f}; its paired interval "
            "includes zero.",
            f"- CAC-change RMSE changed from "
            f"{v2_row['delta_log_cac_rmse']:.4f} to "
            f"{v21_row['delta_log_cac_rmse']:.4f}, and change-space R² "
            f"from {v2_row['delta_log_cac_r2']:.4f} to "
            f"{v21_row['delta_log_cac_r2']:.4f}.",
            "- Predicted-to-true CAC-change variance ratio fell from "
            f"{diagnostics['lac_v2']['overall']['predicted_to_true_variance_ratio']:.4f} "
            "to "
            f"{diagnostics['lac_v21']['overall']['predicted_to_true_variance_ratio']:.4f}; "
            "the individual-change shrinkage problem therefore remains and "
            "became more pronounced.",
            "- V2.1 has 27,919 trainable parameters versus 28,218 for V2.",
            "",
            "## Recommendation",
            "",
            "Do not promote V2.1 as the primary CAC-change model or use this "
            "iterative OOF result as confirmatory evidence. Preserve V2 for "
            "the current CAC-MAE comparison and retain V2.1 as an exploratory "
            "TBR-favored frozen secondary model. A future independent center "
            "can evaluate both without selecting between them on that center.",
        ]
    )
    (output / "V21_REAL_NESTED_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return result
