from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v21_refinement import _normalized_plan_checksum


PRIMARY_METRICS = (
    "tbr_mae",
    "tbr_rmse",
    "tbr_r2",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
    "delta_log_cac_r2",
)
LOWER_IS_BETTER = {
    "tbr_mae",
    "tbr_rmse",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
}
V27_ABLATIONS = (
    "lac_v27_full",
    "lac_v27_median_only",
    "lac_v27_mean_only",
    "lac_v27_no_adapters",
    "lac_v27_no_coupling",
    "lac_v27_no_treatment",
    "lac_v27_no_decision_gate",
    "lac_v27_cac_central_only",
    "lac_v27_cac_lag_mean_only",
    "lac_v27_no_stop_gradient",
)


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _row(model: str, path: Path) -> dict[str, Any]:
    summary = _read_summary(path)
    metrics = summary["pooled_oof_metrics"] | summary[
        "pooled_change_metrics"
    ]
    return {
        "model": model,
        "parameter_count": summary.get("parameter_count"),
        **{metric: metrics[metric] for metric in PRIMARY_METRICS},
    }


def _dominates(candidate: pd.Series, reference: pd.Series) -> bool:
    no_worse = []
    strictly_better = []
    for metric in PRIMARY_METRICS:
        if metric in LOWER_IS_BETTER:
            no_worse.append(candidate[metric] <= reference[metric])
            strictly_better.append(candidate[metric] < reference[metric])
        else:
            no_worse.append(candidate[metric] >= reference[metric])
            strictly_better.append(candidate[metric] > reference[metric])
    return bool(all(no_worse) and any(strictly_better))


def _metric_value(frame: pd.DataFrame, metric: str) -> float:
    if metric.startswith("tbr_"):
        target = frame["true_tbr"].to_numpy(float)
        prediction = frame["pred_tbr"].to_numpy(float)
        statistic = metric.removeprefix("tbr_")
    elif metric.startswith("delta_log_cac_"):
        target = frame["true_delta_log_cac"].to_numpy(float)
        prediction = frame["pred_delta_log_cac"].to_numpy(float)
        statistic = metric.removeprefix("delta_log_cac_")
    else:
        raise KeyError(metric)
    error = prediction - target
    if statistic == "mae":
        return float(np.abs(error).mean())
    if statistic == "rmse":
        return float(np.sqrt(np.square(error).mean()))
    if statistic == "r2":
        denominator = np.square(target - target.mean()).sum()
        return float(1.0 - np.square(error).sum() / denominator)
    raise KeyError(metric)


def _paired_bootstrap_metric_difference(
    candidate_path: Path,
    reference_path: Path,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    columns = [
        "patient_id",
        "true_tbr",
        "pred_tbr",
        "true_delta_log_cac",
        "pred_delta_log_cac",
    ]
    candidate = pd.read_csv(candidate_path)[columns]
    reference = pd.read_csv(reference_path)[columns]
    merged = candidate.merge(
        reference,
        on="patient_id",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    candidate_frame = pd.DataFrame(
        {
            column: merged[f"{column}_candidate"]
            for column in columns
            if column != "patient_id"
        }
    )
    reference_frame = pd.DataFrame(
        {
            column: merged[f"{column}_reference"]
            for column in columns
            if column != "patient_id"
        }
    )
    estimate = _metric_value(candidate_frame, metric) - _metric_value(
        reference_frame,
        metric,
    )
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, float)
    for index in range(replicates):
        draw = rng.integers(0, len(merged), len(merged))
        values[index] = _metric_value(
            candidate_frame.iloc[draw],
            metric,
        ) - _metric_value(reference_frame.iloc[draw], metric)
    return {
        "metric": metric,
        "difference_definition": "candidate_minus_reference",
        "lower_is_better": metric in LOWER_IS_BETTER,
        "estimate": float(estimate),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
        "bootstrap_replicates": int(replicates),
    }


def aggregate_v27_results(
    v27_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    v27_root = Path(v27_root)
    formal_root = Path(formal_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    formal_comparators = (
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
    paths: dict[str, Path] = {
        model: formal_root / model
        for model in formal_comparators
    }
    paths.update(
        {
            model: Path(path)
            for model, path in version_roots.items()
        }
    )
    paths.update({model: v27_root / model for model in V27_ABLATIONS})
    missing = [
        str(path / "summary.json")
        for path in paths.values()
        if not (path / "summary.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing completed model artifacts:\n" + "\n".join(missing)
        )

    table = pd.DataFrame(
        [_row(model, path) for model, path in paths.items()]
    )
    table.to_csv(output / "all_comparison_and_ablation_results.csv", index=False)
    comparison_names = ["lac_v27_full", *formal_comparators]
    comparison = (
        table.set_index("model").loc[comparison_names].reset_index()
    )
    historical_names = [
        "lac_v27_full",
        *version_roots.keys(),
    ]
    historical = (
        table.set_index("model").loc[historical_names].reset_index()
    )
    ablation = (
        table.set_index("model").loc[list(V27_ABLATIONS)].reset_index()
    )
    comparison.to_csv(output / "v27_comparison_results.csv", index=False)
    historical.to_csv(
        output / "v27_historical_version_reference.csv",
        index=False,
    )
    ablation.to_csv(output / "v27_ablation_results.csv", index=False)

    full = table.set_index("model").loc["lac_v27_full"]
    rank_records = []
    acceptance_names = list(dict.fromkeys([
        *formal_comparators,
        *V27_ABLATIONS,
    ]))
    acceptance_table = (
        table.set_index("model").loc[acceptance_names].reset_index()
    )
    for metric in PRIMARY_METRICS:
        ascending = metric in LOWER_IS_BETTER
        ordered = acceptance_table.sort_values(
            metric,
            ascending=ascending,
        )
        rank = int(
            np.flatnonzero(
                ordered["model"].to_numpy() == "lac_v27_full"
            )[0]
            + 1
        )
        rank_records.append(
            {
                "metric": metric,
                "v27_full_value": float(full[metric]),
                "rank_among_acceptance_models": rank,
                "best_model": str(ordered.iloc[0]["model"]),
                "best_value": float(ordered.iloc[0][metric]),
            }
        )
    rank_frame = pd.DataFrame(rank_records)
    rank_frame.to_csv(output / "v27_primary_metric_ranks.csv", index=False)

    ablation_index = ablation.set_index("model")
    dominating_ablations = [
        model
        for model in V27_ABLATIONS[1:]
        if _dominates(ablation_index.loc[model], full)
    ]
    paired_rows = []
    full_prediction = paths[
        "lac_v27_full"
    ] / "out_of_fold_predictions.csv"
    for reference, path in paths.items():
        if reference == "lac_v27_full":
            continue
        reference_prediction = path / "out_of_fold_predictions.csv"
        if not reference_prediction.is_file():
            continue
        for metric in PRIMARY_METRICS:
            paired_rows.append(
                {
                    "candidate": "lac_v27_full",
                    "reference": reference,
                }
                | _paired_bootstrap_metric_difference(
                    full_prediction,
                    reference_prediction,
                    metric,
                    bootstrap_replicates,
                    seed,
                )
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v27_paired_bootstrap_95ci.csv", index=False)

    v27_summary = _read_summary(paths["lac_v27_full"])
    split_checks = []
    reference_plan = version_roots.get("lac_v22_full")
    if reference_plan is not None:
        reference_plan = Path(reference_plan)
        for model in V27_ABLATIONS:
            for fold in range(1, 6):
                split_checks.append(
                    {
                        "model": model,
                        "outer_fold": fold,
                        "inner_plan_identical_to_v22": (
                            _normalized_plan_checksum(
                                paths[model]
                                / f"fold_{fold}"
                                / "inner_patient_fold_plan.csv"
                            )
                            == _normalized_plan_checksum(
                                reference_plan
                                / f"fold_{fold}"
                                / "inner_patient_fold_plan.csv"
                            )
                        ),
                    }
                )
    split_frame = pd.DataFrame(split_checks)
    split_frame.to_csv(output / "v27_split_audit.csv", index=False)
    summary = {
        "v27_full_six_metric_rank_first": bool(
            (rank_frame["rank_among_acceptance_models"] == 1).all()
        ),
        "acceptance_competitors": (
            "prespecified comparator models and independently retrained "
            "V2.7 ablations; V2.1-V2.6 are historical references only"
        ),
        "v27_full_not_pareto_dominated_by_any_ablation": not bool(
            dominating_ablations
        ),
        "dominating_ablations": dominating_ablations,
        "split_audit_all_identical": (
            bool(split_frame["inner_plan_identical_to_v22"].all())
            if len(split_frame)
            else None
        ),
        "v27_full_diagnostics": v27_summary.get("diagnostics"),
        "development_status": (
            "iterative internal development validation; outer-fold results "
            "have been inspected across multiple prior versions and are not "
            "independent confirmation"
        ),
        "external_validation_status": "not_run",
    }
    (output / "v27_result_assessment.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v27-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    version_map = json.loads(
        Path(arguments.version_map).read_text(encoding="utf-8")
    )
    result = aggregate_v27_results(
        arguments.v27_root,
        arguments.formal_root,
        version_map,
        arguments.output,
        bootstrap_replicates=arguments.bootstrap,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
