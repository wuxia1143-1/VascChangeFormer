from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v21_refinement import _normalized_plan_checksum


PRIMARY_METRICS = (
    "delta_tbr_mae",
    "delta_tbr_rmse",
    "delta_tbr_r2",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
    "delta_log_cac_r2",
)
LOWER_IS_BETTER = {
    "delta_tbr_mae",
    "delta_tbr_rmse",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
}
FORMAL_COMPARATORS = (
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
V28_MODELS = (
    "lac_v28_full",
    "lac_v28_no_residual",
    "lac_v28_no_coupling",
    "lac_v28_no_treatment",
    "lac_v28_no_adapters",
    "lac_v28_naive_gradients",
    "lac_v28_no_multitask_sharing",
    "lac_v28_no_stop_gradient",
    "lac_v28_no_real_time",
)


def _summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _result_row(model: str, path: Path, category: str) -> dict[str, Any]:
    summary = _summary(path)
    change = summary["pooled_change_metrics"]
    endpoint = summary["pooled_oof_metrics"]
    return {
        "model": model,
        "category": category,
        "patient_count": summary.get("patient_count", 443),
        "parameter_count": summary.get("parameter_count"),
        **{metric: float(change[metric]) for metric in PRIMARY_METRICS},
        **{
            f"endpoint_{metric}": float(value)
            for metric, value in endpoint.items()
        },
    }


def _dominates(candidate: pd.Series, reference: pd.Series) -> bool:
    no_worse = []
    better = []
    for metric in PRIMARY_METRICS:
        if metric in LOWER_IS_BETTER:
            no_worse.append(candidate[metric] <= reference[metric])
            better.append(candidate[metric] < reference[metric])
        else:
            no_worse.append(candidate[metric] >= reference[metric])
            better.append(candidate[metric] > reference[metric])
    return bool(all(no_worse) and any(better))


def _metric(target: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    error = prediction - target
    if kind == "mae":
        return float(np.mean(np.abs(error)))
    if kind == "rmse":
        return float(np.sqrt(np.mean(np.square(error))))
    if kind == "r2":
        denominator = float(np.square(target - target.mean()).sum())
        return float(1.0 - np.square(error).sum() / denominator)
    raise KeyError(kind)


def _arrays(frame: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric.startswith("delta_tbr_"):
        return (
            frame["true_delta_tbr"].to_numpy(float),
            frame["pred_delta_tbr"].to_numpy(float),
        )
    if metric.startswith("delta_log_cac_"):
        return (
            frame["true_delta_log_cac"].to_numpy(float),
            frame["pred_delta_log_cac"].to_numpy(float),
        )
    raise KeyError(metric)


def _paired_difference(
    full_path: Path,
    reference_path: Path,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    full = pd.read_csv(full_path).sort_values("patient_id").reset_index(drop=True)
    reference = (
        pd.read_csv(reference_path)
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if not full["patient_id"].equals(reference["patient_id"]):
        raise RuntimeError("Paired bootstrap patient identifiers differ")
    target, full_prediction = _arrays(full, metric)
    reference_target, reference_prediction = _arrays(reference, metric)
    maximum_target_difference = float(
        np.max(np.abs(target - reference_target))
    )
    if not np.allclose(target, reference_target, rtol=0, atol=1e-6):
        raise RuntimeError(
            "Paired bootstrap target values differ beyond CSV floating-point "
            f"tolerance: max_abs_difference={maximum_target_difference}"
        )
    kind = metric.rsplit("_", 1)[1]
    estimate = _metric(target, full_prediction, kind) - _metric(
        target, reference_prediction, kind
    )
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, float)
    for index in range(replicates):
        draw = rng.integers(0, len(target), len(target))
        values[index] = _metric(
            target[draw], full_prediction[draw], kind
        ) - _metric(target[draw], reference_prediction[draw], kind)
    low = float(np.nanquantile(values, 0.025))
    high = float(np.nanquantile(values, 0.975))
    lower = metric in LOWER_IS_BETTER
    return {
        "metric": metric,
        "difference_definition": "V2.8_Full_minus_reference",
        "lower_is_better": lower,
        "estimate": float(estimate),
        "ci_low": low,
        "ci_high": high,
        "full_significantly_favored": bool(high < 0 if lower else low > 0),
        "reference_significantly_favored": bool(low > 0 if lower else high < 0),
        "bootstrap_replicates": int(replicates),
        "max_abs_target_serialization_difference": maximum_target_difference,
    }


def _split_audit(paths: dict[str, Path], reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(
        reference_path / "out_of_fold_predictions.csv"
    ).sort_values("patient_id").reset_index(drop=True)
    rows = []
    for model, path in paths.items():
        frame = pd.read_csv(path / "out_of_fold_predictions.csv").sort_values(
            "patient_id"
        ).reset_index(drop=True)
        summary = _summary(path)
        rows.append(
            {
                "model": model,
                "patient_count": len(frame),
                "unique_patient_count": frame["patient_id"].nunique(),
                "patient_ids_identical": frame["patient_id"].equals(
                    reference["patient_id"]
                ),
                "outer_folds_identical": (
                    frame["outer_fold"].equals(reference["outer_fold"])
                    if "outer_fold" in frame and "outer_fold" in reference
                    else True
                ),
                "outer_checksum": summary.get("cv_protocol", {}).get(
                    "outer_fold_plan_checksum",
                    summary.get("cv_protocol", {}).get("fold_plan_checksum"),
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_v28_results(
    v28_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    v28_root = Path(v28_root)
    formal_root = Path(formal_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: formal_root / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name] = Path(value)
        categories[name] = "historical_version"
    for name in V28_MODELS:
        paths[name] = v28_root / name
        categories[name] = "v28_full" if name == "lac_v28_full" else "v28_ablation"
    missing = [str(path / "summary.json") for path in paths.values() if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing))
    table = pd.DataFrame(
        [_result_row(name, path, categories[name]) for name, path in paths.items()]
    )
    table.to_csv(output / "v28_all_results.csv", index=False)
    comparison_names = ["lac_v28_full", *FORMAL_COMPARATORS]
    comparison = table.set_index("model").loc[comparison_names].reset_index()
    comparison.to_csv(output / "v28_comparison_results.csv", index=False)
    ablation = table.set_index("model").loc[list(V28_MODELS)].reset_index()
    ablation.to_csv(output / "v28_ablation_results.csv", index=False)
    historical_names = ["lac_v28_full", *version_roots.keys()]
    historical = table.set_index("model").loc[historical_names].reset_index()
    historical.to_csv(output / "v28_historical_version_reference.csv", index=False)

    full = table.set_index("model").loc["lac_v28_full"]
    acceptance = table.set_index("model").loc[
        [*FORMAL_COMPARATORS, *V28_MODELS]
    ].reset_index()
    ranks = []
    for metric in PRIMARY_METRICS:
        ordered = acceptance.sort_values(
            metric, ascending=metric in LOWER_IS_BETTER, kind="mergesort"
        ).reset_index(drop=True)
        full_value = float(full[metric])
        if metric in LOWER_IS_BETTER:
            rank = int(1 + np.sum(ordered[metric].to_numpy(float) < full_value - 1e-12))
        else:
            rank = int(1 + np.sum(ordered[metric].to_numpy(float) > full_value + 1e-12))
        ranks.append(
            {
                "metric": metric,
                "full_value": full_value,
                "rank_with_ties": rank,
                "best_value": float(ordered.iloc[0][metric]),
                "best_models": ";".join(
                    ordered.loc[
                        np.isclose(ordered[metric], ordered.iloc[0][metric], rtol=0, atol=1e-12),
                        "model",
                    ].astype(str)
                ),
            }
        )
    rank_frame = pd.DataFrame(ranks)
    rank_frame.to_csv(output / "v28_primary_metric_ranks.csv", index=False)

    full_prediction = paths["lac_v28_full"] / "out_of_fold_predictions.csv"
    paired_rows = []
    for reference, path in paths.items():
        if reference == "lac_v28_full":
            continue
        prediction_path = path / "out_of_fold_predictions.csv"
        if not prediction_path.is_file():
            continue
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            paired_rows.append(
                {"candidate": "lac_v28_full", "reference": reference}
                | _paired_difference(
                    full_prediction,
                    prediction_path,
                    metric,
                    replicates,
                    seed + metric_index,
                )
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v28_paired_bootstrap_95ci.csv", index=False)

    split = _split_audit(paths, paths["lac_v22_full"])
    for model in V28_MODELS:
        for fold in range(1, 6):
            current = paths[model] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            if current.is_file() and reference.is_file():
                split.loc[split["model"] == model, f"inner_fold_{fold}_identical"] = (
                    _normalized_plan_checksum(current)
                    == _normalized_plan_checksum(reference)
                )
    split.to_csv(output / "v28_split_audit.csv", index=False)

    ablation_index = ablation.set_index("model")
    dominating = [
        name for name in V28_MODELS[1:]
        if _dominates(ablation_index.loc[name], full)
    ]
    formal_index = comparison.set_index("model")
    formal_best = {
        metric: (
            float(full[metric]) <= float(formal_index.loc[list(FORMAL_COMPARATORS), metric].min())
            if metric in LOWER_IS_BETTER
            else float(full[metric]) >= float(formal_index.loc[list(FORMAL_COMPARATORS), metric].max())
        )
        for metric in PRIMARY_METRICS
    }
    full_summary = _summary(paths["lac_v28_full"])
    assessment = {
        "frozen_commit": full_summary.get("git_revision"),
        "patient_count": int(full_summary.get("patient_count", 443)),
        "outer_fold_checksum": full_summary["cv_protocol"]["outer_fold_plan_checksum"],
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(all(formal_best.values())),
        "formal_comparator_metric_acceptance": formal_best,
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "full_metric_ranks": ranks,
        "split_and_patient_audit_all_passed": bool(
            (split["patient_count"] == 443).all()
            and (split["unique_patient_count"] == 443).all()
            and split["patient_ids_identical"].all()
            and split["outer_folds_identical"].all()
            and all(
                split[column].fillna(True).all()
                for column in split.columns if column.startswith("inner_fold_")
            )
        ),
        "full_diagnostics": full_summary.get("diagnostics"),
        "interpretation": (
            "frozen deterministic internal validation on the reused 443-patient "
            "development cohort; not independent external confirmation"
        ),
        "github_push": False,
    }
    (output / "v28_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v28-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    version_map = json.loads(Path(arguments.version_map).read_text(encoding="utf-8"))
    result = aggregate_v28_results(
        arguments.v28_root,
        arguments.formal_root,
        version_map,
        arguments.output,
        arguments.bootstrap,
        arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
