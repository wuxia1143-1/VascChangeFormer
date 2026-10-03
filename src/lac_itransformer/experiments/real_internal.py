from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.development_workbook import DevelopmentWorkbookReader
from ..data.schema import FeatureSchema
from ..training.folds import (
    build_patient_fold_plan,
    fold_plan_checksum,
    save_patient_fold_plan,
)
from ..training.nested import run_nested_cross_validation


REAL_COMPARISON_MODELS = (
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
    "lac_itransformer",
    "lac_v2",
)
REAL_ABLATION_MODELS = (
    "lac_itransformer",
    "lac_v1_no_competitive_routing",
    "lac_v2",
    "lac_v2_no_adapters",
    "lac_v2_no_coupling",
    "lac_v2_forward_only",
    "lac_v2_symmetric",
    "lac_v2_no_treatment",
)
ALL_REAL_MODELS = tuple(
    dict.fromkeys(REAL_COMPARISON_MODELS + REAL_ABLATION_MODELS)
)
REAL_REFINEMENT_MODELS = ("lac_v21",)

DISPLAY_NAMES = {
    "persistence": "Persistence",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "apn_dr": "APN-DR",
    "itransformer_mtl": "iTransformer-MTL",
    "first_icu_mtl": "FIRST-ICU-MTL",
    "learning_to_route": "Learning to Route",
    "lac_itransformer": "LAC-iTransformer V1",
    "lac_v1_no_competitive_routing": "V1 w/o competitive routing",
    "lac_v2": "LAC-iTransformer V2",
    "lac_v2_no_adapters": "V2 w/o phenotype adapters",
    "lac_v2_no_coupling": "V2 w/o coupling",
    "lac_v2_forward_only": "V2 forward-only",
    "lac_v2_symmetric": "V2 symmetric coupling",
    "lac_v2_no_treatment": "V2 w/o treatment conditioning",
    "lac_v21": "LAC-iTransformer V2.1",
}

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_real_development_cohort(
    workbook_path: str | Path,
    output_dir: str | Path,
    prediction_lead_months: float = 0.0,
    min_followup_months: float | None = None,
    max_followup_months: float | None = None,
) -> dict[str, Any]:
    source = Path(workbook_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reader = DevelopmentWorkbookReader()
    arrays, audit = reader.prepare_arrays(
        source,
        prediction_lead_months=prediction_lead_months,
        min_followup_months=min_followup_months,
        max_followup_months=max_followup_months,
    )
    schema = reader.schema
    prepared_path = output / "prepared_arrays.npz"
    np.savez_compressed(
        prepared_path,
        **arrays,
        schema_json=np.asarray(
            [json.dumps(schema.to_dict(), ensure_ascii=False)]
        ),
    )
    manifest = {
        "status": "real_internal_development_cohort_prepared",
        "source_sha256": _sha256(source),
        "source_filename": source.name,
        "source_file_modified": False,
        "patient_identifiers_are_source_hashes": True,
        "prepared_patient_count": len(arrays["patient_ids"]),
        "prediction_lead_months": float(prediction_lead_months),
        "followup_inclusion": {
            "minimum_months_inclusive": min_followup_months,
            "maximum_months_inclusive": max_followup_months,
            "missing_policy": (
                "exclude"
                if min_followup_months is not None
                or max_followup_months is not None
                else "require_finite"
            ),
            "included_patient_count": len(arrays["patient_ids"]),
            "excluded_missing": audit.followup_filter_excluded_missing,
            "excluded_below_minimum": (
                audit.followup_filter_excluded_below_minimum
            ),
            "excluded_above_maximum": (
                audit.followup_filter_excluded_above_maximum
            ),
        },
        "prediction_target": (
            "Predict the observed endpoint TBR and CAC state at the patient's "
            "next follow-up from information available at least the configured "
            "lead time before that endpoint."
        ),
        "schema": schema.to_dict(),
        "audit": audit.to_dict(),
        "date_window_rule": (
            "Only longitudinal events on or after baseline imaging and on or "
            f"before endpoint imaging minus {float(prediction_lead_months):g} "
            "months are retained. Later events are excluded before any fold "
            "is built. The endpoint labels and full observed follow-up interval "
            "are retained unchanged. The source follow-up "
            "interval is in months (confirmed against 386 paired imaging-date "
            "records) and is converted with 365.25/12 days per month only "
            "when an imaging date must be derived."
        ),
        "followup_inclusion_rule": (
            "Patients are retained only when the observed follow-up interval "
            f"is within [{min_followup_months}, {max_followup_months}] months, "
            "with both bounds inclusive; missing intervals are excluded."
            if min_followup_months is not None
            or max_followup_months is not None
            else "No additional follow-up interval range filter is applied."
        ),
        "static_encoding": (
            "Pathology codes are one-hot encoded; invalid/missing static "
            "values have explicit missing indicators and are imputed only "
            "inside each training fold."
        ),
        "treatment_encoding": (
            "A treatment is positive only for a nonempty, nonzero, "
            "nonnegative source entry; the source treatment-row categories "
            "are retained, with surgery interpreted as surgery_or_procedure."
        ),
        "time_encoding": (
            "Model times and irregular times are elapsed years since baseline "
            "imaging; patch membership is determined by relative position in "
            "the patient-specific interval. APN privately rescales its copy "
            "to [0, 1] because its published adaptive patches use relative "
            "coordinates."
        ),
        "followup_interval_array": (
            "followup_months stores the source treatment-to-follow-up interval "
            "for prespecified training-stratum balancing, patch-horizon "
            "definition and aggregate CAC change diagnostics. It is never "
            "passed to a prediction head and is not shortened by censoring."
        ),
        "clinical_interpretation": (
            "This cohort is used for model development/internal validation "
            "and is no longer an independent external-validation cohort."
        ),
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "schema.json").write_text(
        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_prepared_real_cohort(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], FeatureSchema]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {
            key: loaded[key] for key in loaded.files if key != "schema_json"
        }
        schema = FeatureSchema.from_dict(
            json.loads(str(loaded["schema_json"][0]))
        )
    return arrays, schema


def run_real_nested_model(
    prepared_path: str | Path,
    config: dict[str, Any],
    output_root: str | Path,
    model_name: str,
) -> dict[str, Any]:
    if model_name not in ALL_REAL_MODELS + REAL_REFINEMENT_MODELS:
        raise KeyError(f"Model is not in the locked real-data suite: {model_name}")
    if str(config.get("result_scope")) != "real_internal_nested_fivefold":
        raise ValueError("Real nested run requires the locked real-data result scope")
    arrays, schema = load_prepared_real_cohort(prepared_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    outer_plan = build_patient_fold_plan(
        arrays["patient_ids"],
        n_folds=5,
        seed=seed,
    )
    shared_plan_path = output / "locked_patient_outer_fold_plan.csv"
    shared_manifest_path = shared_plan_path.with_suffix(".json")
    if shared_plan_path.is_file() and shared_manifest_path.is_file():
        existing = pd.read_csv(shared_plan_path)
        existing_plan = {
            str(row.patient_id): int(row.test_fold)
            for row in existing.itertuples(index=False)
        }
        if fold_plan_checksum(existing_plan) != fold_plan_checksum(outer_plan):
            raise RuntimeError("Existing locked outer fold plan is incompatible")
        shared_manifest = json.loads(
            shared_manifest_path.read_text(encoding="utf-8")
        )
    else:
        shared_manifest = save_patient_fold_plan(
            outer_plan,
            shared_plan_path,
            seed=seed,
            n_folds=5,
        )
    model_output = output / model_name
    summary_path = model_output / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if model_output.exists() and any(model_output.iterdir()):
        raise RuntimeError(
            f"Partial output exists and will not be overwritten: {model_output}"
        )
    model_config = copy.deepcopy(config)
    model_config["model_name"] = model_name
    summary = run_nested_cross_validation(
        arrays,
        schema,
        model_config,
        model_output,
        outer_fold_assignments=outer_plan,
    )
    if (
        summary["cv_protocol"]["outer_fold_plan_checksum"]
        != shared_manifest["checksum_sha256"]
    ):
        raise RuntimeError("Model did not use the locked shared outer fold plan")
    return summary


def run_real_nested_suite(
    prepared_path: str | Path,
    config: dict[str, Any],
    output_root: str | Path,
    models: Iterable[str] = ALL_REAL_MODELS,
) -> dict[str, Any]:
    summaries = {}
    for model_name in tuple(dict.fromkeys(models)):
        summaries[model_name] = run_real_nested_model(
            prepared_path,
            config,
            output_root,
            model_name,
        )
    return summaries


def _model_row(
    model_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    counts = [
        record["parameter_count"]
        for record in summary["fold_records"]
        if record["parameter_count"] is not None
    ]
    metrics = (
        summary["pooled_oof_metrics"]
        | summary["pooled_change_metrics"]
    )
    metric_ci = summary["bootstrap_95_ci"]
    change_ci = summary["change_bootstrap_95_ci"]
    fold_mean = summary["mean_outer_fold_metrics"]
    fold_sd = summary["outer_fold_standard_deviation"]
    return {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "parameter_count": (
            int(np.median(counts)) if counts else None
        ),
        **{
            metric: metrics[metric]
            for metric in REPORT_METRICS
        },
        "tbr_mae_fold_mean": fold_mean["tbr_mae"],
        "tbr_mae_fold_sd": fold_sd["tbr_mae"],
        "tbr_mae_ci_low": metric_ci["tbr_mae"]["ci_low"],
        "tbr_mae_ci_high": metric_ci["tbr_mae"]["ci_high"],
        "log_cac_mae_fold_mean": fold_mean["log_cac_mae"],
        "log_cac_mae_fold_sd": fold_sd["log_cac_mae"],
        "log_cac_mae_ci_low": metric_ci["log_cac_mae"]["ci_low"],
        "log_cac_mae_ci_high": metric_ci["log_cac_mae"]["ci_high"],
        "delta_tbr_mae_ci_low": change_ci["delta_tbr_mae"]["ci_low"],
        "delta_tbr_mae_ci_high": change_ci["delta_tbr_mae"]["ci_high"],
        "delta_log_cac_mae_ci_low": change_ci[
            "delta_log_cac_mae"
        ]["ci_low"],
        "delta_log_cac_mae_ci_high": change_ci[
            "delta_log_cac_mae"
        ]["ci_high"],
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(format_value(value) for value in row) + " |"
        )
    return "\n".join(lines)


def _per_patient_error(frame: pd.DataFrame, metric: str) -> np.ndarray:
    if metric == "tbr_mae":
        return np.abs(frame["true_tbr"] - frame["pred_tbr"]).to_numpy()
    if metric == "log_cac_mae":
        return np.abs(
            np.log1p(np.maximum(frame["true_cac"], 0))
            - np.log1p(np.maximum(frame["pred_cac"], 0))
        )
    if metric == "delta_tbr_mae":
        return np.abs(
            frame["true_delta_tbr"] - frame["pred_delta_tbr"]
        ).to_numpy()
    if metric == "delta_log_cac_mae":
        return np.abs(
            frame["true_delta_log_cac"]
            - frame["pred_delta_log_cac"]
        ).to_numpy()
    raise KeyError(metric)


def _paired_bootstrap_difference(
    candidate_path: Path,
    reference_path: Path,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    candidate = pd.read_csv(candidate_path)
    reference = pd.read_csv(reference_path)
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
    differences = _per_patient_error(
        candidate_frame, metric
    ) - _per_patient_error(reference_frame, metric)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = rng.integers(0, len(differences), len(differences))
        samples[index] = np.mean(differences[draw])
    return {
        "metric": metric,
        "difference_definition": "candidate_minus_reference",
        "lower_is_better": True,
        "estimate": float(np.mean(differences)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_replicates": replicates,
    }


def audit_real_nested_results(
    output_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    errors = []
    revisions = set()
    fold_checksums = set()
    model_records = {}
    for model in ALL_REAL_MODELS:
        directory = root / model
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            errors.append(f"{model}: missing summary")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        revisions.add(str(summary["git_revision"]))
        fold_checksums.add(
            str(summary["cv_protocol"]["outer_fold_plan_checksum"])
        )
        oof = pd.read_csv(directory / "out_of_fold_predictions.csv")
        prediction_columns = [
            "pred_tbr",
            "pred_cac",
            "pred_delta_tbr",
            "pred_delta_log_cac",
        ]
        record_errors = []
        if len(oof) != 443 or oof["patient_id"].nunique() != 443:
            record_errors.append("OOF does not contain 443 unique patients")
        if set(oof["outer_fold"]) != {1, 2, 3, 4, 5}:
            record_errors.append("OOF outer fold labels are incomplete")
        if not np.isfinite(oof[prediction_columns].to_numpy()).all():
            record_errors.append("OOF contains nonfinite predictions")
        inner_patient_counts = []
        selected_epochs = []
        for fold in range(1, 6):
            fold_dir = directory / f"fold_{fold}"
            split = json.loads(
                (fold_dir / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for flag in (
                "test_fold_used_for_preprocessing",
                "test_fold_used_for_inner_selection",
                "test_fold_used_for_training",
            ):
                if split[flag]:
                    record_errors.append(f"fold {fold}: leakage flag {flag}")
            inner = pd.read_csv(fold_dir / "inner_patient_fold_plan.csv")
            inner_patient_counts.append(len(inner))
            if len(inner) != int(split["n_outer_train"]):
                record_errors.append(
                    f"fold {fold}: inner plan size differs from outer train"
                )
            if set(inner["test_fold"]) != {1, 2, 3, 4, 5}:
                record_errors.append(
                    f"fold {fold}: inner fold labels are incomplete"
                )
            outer_test_ids = set(
                oof.loc[oof["outer_fold"] == fold, "patient_id"].astype(str)
            )
            if outer_test_ids & set(inner["patient_id"].astype(str)):
                record_errors.append(
                    f"fold {fold}: outer test patient appears in inner plan"
                )
            if split["selected_epochs"] is not None:
                selected_epochs.append(int(split["selected_epochs"]))
        if int(summary["bootstrap_replicates"]) != 2000:
            record_errors.append("formal bootstrap count is not 2000")
        errors.extend(f"{model}: {message}" for message in record_errors)
        model_records[model] = {
            "oof_rows": len(oof),
            "unique_oof_patients": int(oof["patient_id"].nunique()),
            "outer_fold_counts": {
                str(key): int(value)
                for key, value in oof["outer_fold"]
                .value_counts()
                .sort_index()
                .items()
            },
            "inner_patient_counts": inner_patient_counts,
            "selected_epochs": selected_epochs,
            "passed": not record_errors,
        }
    if len(revisions) != 1:
        errors.append(f"multiple training revisions: {sorted(revisions)}")
    if len(fold_checksums) != 1:
        errors.append(
            f"multiple outer fold checksums: {sorted(fold_checksums)}"
        )
    result = {
        "status": "passed" if not errors else "failed",
        "expected_model_count": len(ALL_REAL_MODELS),
        "audited_model_count": len(model_records),
        "training_git_revisions": sorted(revisions),
        "outer_fold_checksums": sorted(fold_checksums),
        "all_models_have_443_unique_oof_patients": all(
            record["unique_oof_patients"] == 443
            for record in model_records.values()
        ),
        "all_leakage_flags_false": not any(
            "leakage flag" in error for error in errors
        ),
        "errors": errors,
        "models": model_records,
    }
    Path(output_path).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(
            "Formal real nested-CV result audit failed: "
            + "; ".join(errors[:5])
        )
    return result


def aggregate_real_nested_results(
    output_root: str | Path,
    output_dir: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    root = Path(output_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {
        model: json.loads(
            (root / model / "summary.json").read_text(encoding="utf-8")
        )
        for model in ALL_REAL_MODELS
    }
    checksums = {
        summary["cv_protocol"]["outer_fold_plan_checksum"]
        for summary in summaries.values()
    }
    if len(checksums) != 1:
        raise RuntimeError("Models do not share one locked outer fold plan")
    comparison = pd.DataFrame(
        [
            _model_row(model, summaries[model])
            for model in REAL_COMPARISON_MODELS
        ]
    )
    ablation = pd.DataFrame(
        [
            _model_row(model, summaries[model])
            for model in REAL_ABLATION_MODELS
        ]
    )
    comparison.to_csv(output / "comparison_results.csv", index=False)
    ablation.to_csv(output / "ablation_results.csv", index=False)

    paired_rows = []
    for model in REAL_COMPARISON_MODELS:
        if model == "lac_v2":
            continue
        for metric in (
            "tbr_mae",
            "log_cac_mae",
            "delta_tbr_mae",
            "delta_log_cac_mae",
        ):
            paired_rows.append(
                {
                    "experiment": "comparison_vs_v2",
                    "candidate": model,
                    "reference": "lac_v2",
                }
                | _paired_bootstrap_difference(
                    root / model / "out_of_fold_predictions.csv",
                    root / "lac_v2" / "out_of_fold_predictions.csv",
                    metric,
                    bootstrap_replicates,
                    seed,
                )
            )
    ablation_references = {
        "lac_v1_no_competitive_routing": "lac_itransformer",
        "lac_v2_no_adapters": "lac_v2",
        "lac_v2_no_coupling": "lac_v2",
        "lac_v2_forward_only": "lac_v2",
        "lac_v2_symmetric": "lac_v2",
        "lac_v2_no_treatment": "lac_v2",
    }
    for candidate, reference in ablation_references.items():
        for metric in (
            "tbr_mae",
            "log_cac_mae",
            "delta_tbr_mae",
            "delta_log_cac_mae",
        ):
            paired_rows.append(
                {
                    "experiment": "ablation",
                    "candidate": candidate,
                    "reference": reference,
                }
                | _paired_bootstrap_difference(
                    root / candidate / "out_of_fold_predictions.csv",
                    root / reference / "out_of_fold_predictions.csv",
                    metric,
                    bootstrap_replicates,
                    seed + 1,
                )
            )
    paired = pd.DataFrame(paired_rows)
    for metric in (
        "tbr_mae",
        "log_cac_mae",
        "delta_tbr_mae",
        "delta_log_cac_mae",
    ):
        paired_rows.append(
            {
                "experiment": "mechanism_i_to_c",
                "candidate": "lac_v2_no_coupling",
                "reference": "lac_v2_forward_only",
            }
            | _paired_bootstrap_difference(
                root
                / "lac_v2_no_coupling"
                / "out_of_fold_predictions.csv",
                root
                / "lac_v2_forward_only"
                / "out_of_fold_predictions.csv",
                metric,
                bootstrap_replicates,
                seed + 2,
            )
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired_bootstrap_differences.csv", index=False)
    quality_audit = audit_real_nested_results(
        root,
        output / "quality_audit.json",
    )

    tbr_best = comparison.sort_values("tbr_mae").iloc[0]
    cac_best = comparison.sort_values("log_cac_mae").iloc[0]
    comparison_report = comparison[
        [
            "display_name",
            "parameter_count",
            "tbr_mae",
            "tbr_mae_ci_low",
            "tbr_mae_ci_high",
            "tbr_r2",
            "log_cac_mae",
            "log_cac_mae_ci_low",
            "log_cac_mae_ci_high",
            "log_cac_r2",
            "delta_tbr_r2",
            "delta_log_cac_r2",
        ]
    ]
    ablation_report = ablation[
        [
            "display_name",
            "parameter_count",
            "tbr_mae",
            "log_cac_mae",
            "delta_tbr_r2",
            "delta_log_cac_r2",
        ]
    ]
    report_lines = [
        "# Real internal nested fivefold results",
        "",
        "This report uses one real development cohort with patient-level outer "
        "fivefold evaluation and a separate patient-level inner fivefold "
        "inside every outer training pool.",
        "",
        "The cohort was reclassified for development/internal validation; it "
        "must not also be reported as an independent external validation.",
        "",
        "## Comparison",
        "",
        _markdown_table(comparison_report),
        "",
        "## Ablation",
        "",
        _markdown_table(ablation_report),
        "",
        "## Direct observations",
        "",
        f"- Lowest pooled OOF TBR MAE: {tbr_best['display_name']} "
        f"({tbr_best['tbr_mae']:.4f}).",
        f"- Lowest pooled OOF log-CAC MAE: {cac_best['display_name']} "
        f"({cac_best['log_cac_mae']:.4f}).",
        "- Paired bootstrap differences are stored separately; a negative "
        "candidate-minus-reference MAE favors the candidate.",
        "- These are internal-validation estimates, not external-validation "
        "or deployment claims.",
        f"- Automated quality audit: {quality_audit['status']}; "
        f"{quality_audit['audited_model_count']} models checked.",
    ]
    (output / "REAL_INTERNAL_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "real_internal_nested_fivefold_complete",
        "patient_level_outer_folds": 5,
        "patient_level_inner_folds": 5,
        "shared_outer_fold_checksum": next(iter(checksums)),
        "models": list(ALL_REAL_MODELS),
        "comparison_models": list(REAL_COMPARISON_MODELS),
        "ablation_models": list(REAL_ABLATION_MODELS),
        "bootstrap_replicates": bootstrap_replicates,
        "best_tbr_model": str(tbr_best["model"]),
        "best_log_cac_model": str(cac_best["model"]),
        "external_validation_status": (
            "not_run; this cohort is development/internal only"
        ),
        "quality_audit_status": quality_audit["status"],
    }
    (output / "aggregate_manifest.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
