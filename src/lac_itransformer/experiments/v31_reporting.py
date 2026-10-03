from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v21_refinement import _normalized_plan_checksum
from .v28_reporting import (
    FORMAL_COMPARATORS,
    LOWER_IS_BETTER,
    PRIMARY_METRICS,
    _dominates,
    _paired_difference,
    _result_row,
    _split_audit,
    _summary,
)


V31_MODELS = (
    "lac_v31_full",
    "lac_v31_dual_independent",
    "lac_v31_no_lag",
    "lac_v31_history_permuted",
    "lac_v31_history_shifted",
    "lac_v31_no_treatment",
    "lac_v31_no_real_time",
    "lac_v31_no_patient_gate",
    "lac_v31_no_adapters",
)
ROBUSTNESS_MODELS = (
    "lac_v31_full",
    "lac_v31_no_lag",
    "lac_v31_dual_independent",
)
SEEDS = tuple(range(5))


def _paired_v31(
    candidate: Path,
    reference: Path,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result = _paired_difference(candidate, reference, metric, replicates, seed)
    result["difference_definition"] = "V3.1_Full_minus_reference"
    return result


def _selected_lag_scale(model_path: Path, fold: int) -> float:
    audit_path = model_path / f"fold_{fold}" / "decision_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return float(audit["unique_fixed_configuration"]["cac"]["weight"])


def _rank_rows(full: pd.Series, candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric in PRIMARY_METRICS:
        values = candidates[metric].to_numpy(float)
        full_value = float(full[metric])
        better_mask = (
            values < full_value - 1e-12
            if metric in LOWER_IS_BETTER
            else values > full_value + 1e-12
        )
        best = float(
            np.min(values) if metric in LOWER_IS_BETTER else np.max(values)
        )
        rows.append({
            "metric": metric,
            "full_value": full_value,
            "rank_with_ties": int(1 + better_mask.sum()),
            "best_value": best,
            "best_models": ";".join(
                candidates.loc[
                    np.isclose(candidates[metric], best, rtol=0, atol=1e-12),
                    "model",
                ].astype(str)
            ),
        })
    return rows


def _seed_tables(multiseed_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model in ROBUSTNESS_MODELS:
            path = multiseed_root / f"seed{seed}" / model
            if not (path / "summary.json").is_file():
                raise FileNotFoundError(path / "summary.json")
            row = _result_row(model, path, "v31_seed_robustness")
            metric_rows.append({"training_seed_offset": seed, **row})
    metrics = pd.DataFrame(metric_rows)
    summary_rows = []
    for model, group in metrics.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "seed_count": len(group)}
        for metric in PRIMARY_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summaries = pd.DataFrame(summary_rows)

    effect_rows: list[dict[str, Any]] = []
    indexed = metrics.set_index(["training_seed_offset", "model"])
    for seed in SEEDS:
        full = indexed.loc[(seed, "lac_v31_full")]
        no_lag = indexed.loc[(seed, "lac_v31_no_lag")]
        independent = indexed.loc[(seed, "lac_v31_dual_independent")]
        effect_rows.extend([
            {
                "training_seed_offset": seed,
                "contrast": "lag_full_vs_no_lag",
                "tbr_mae_difference": float(full.delta_tbr_mae - no_lag.delta_tbr_mae),
                "cac_mae_difference": float(full.delta_log_cac_mae - no_lag.delta_log_cac_mae),
                "cac_rmse_difference": float(full.delta_log_cac_rmse - no_lag.delta_log_cac_rmse),
                "cac_r2_difference": float(full.delta_log_cac_r2 - no_lag.delta_log_cac_r2),
                "tbr_mae_noninferior": bool(full.delta_tbr_mae <= no_lag.delta_tbr_mae * 1.005),
                "cac_mae_noninferior": bool(full.delta_log_cac_mae <= no_lag.delta_log_cac_mae * 1.005),
                "cac_rmse_better": bool(full.delta_log_cac_rmse < no_lag.delta_log_cac_rmse),
                "cac_r2_better": bool(full.delta_log_cac_r2 > no_lag.delta_log_cac_r2),
            },
            {
                "training_seed_offset": seed,
                "contrast": "shared_dual_no_lag_vs_independent",
                "tbr_mae_difference": float(no_lag.delta_tbr_mae - independent.delta_tbr_mae),
                "cac_mae_difference": float(no_lag.delta_log_cac_mae - independent.delta_log_cac_mae),
                "cac_rmse_difference": float(no_lag.delta_log_cac_rmse - independent.delta_log_cac_rmse),
                "cac_r2_difference": float(no_lag.delta_log_cac_r2 - independent.delta_log_cac_r2),
                "tbr_mae_noninferior": bool(no_lag.delta_tbr_mae <= independent.delta_tbr_mae * 1.005),
                "cac_mae_noninferior": bool(no_lag.delta_log_cac_mae <= independent.delta_log_cac_mae * 1.005),
                "cac_rmse_better": bool(no_lag.delta_log_cac_rmse < independent.delta_log_cac_rmse),
                "cac_r2_better": bool(no_lag.delta_log_cac_r2 > independent.delta_log_cac_r2),
            },
        ])
    effects = pd.DataFrame(effect_rows)
    effects["joint_acceptance"] = (
        effects["tbr_mae_noninferior"]
        & effects["cac_mae_noninferior"]
        & effects["cac_rmse_better"]
        & effects["cac_r2_better"]
    )
    return metrics, summaries, effects


def aggregate_v31_results(
    v31_root: str | Path,
    multiseed_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    v31_root = Path(v31_root)
    multiseed_root = Path(multiseed_root)
    formal_root = Path(formal_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {name: formal_root / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name] = Path(value)
        categories[name] = "historical_version"
    for name in V31_MODELS:
        paths[name] = v31_root / name
        categories[name] = "v31_full" if name == "lac_v31_full" else "v31_ablation"
    missing = [
        str(path / "summary.json")
        for path in paths.values()
        if not (path / "summary.json").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing))

    table = pd.DataFrame([
        _result_row(name, path, categories[name]) for name, path in paths.items()
    ])
    table.to_csv(output / "v31_all_results.csv", index=False)
    indexed = table.set_index("model")
    comparison = indexed.loc[["lac_v31_full", *FORMAL_COMPARATORS]].reset_index()
    comparison.to_csv(output / "v31_comparison_results.csv", index=False)
    ablation = indexed.loc[list(V31_MODELS)].reset_index()
    ablation.to_csv(output / "v31_ablation_results.csv", index=False)
    historical = indexed.loc[["lac_v31_full", *version_roots.keys()]].reset_index()
    historical.to_csv(output / "v31_historical_reference.csv", index=False)

    full = indexed.loc["lac_v31_full"]
    rank_candidates = indexed.loc[["lac_v31_full", *FORMAL_COMPARATORS]].reset_index()
    ranks = _rank_rows(full, rank_candidates)
    pd.DataFrame(ranks).to_csv(output / "v31_primary_metric_ranks.csv", index=False)

    full_prediction = paths["lac_v31_full"] / "out_of_fold_predictions.csv"
    paired_rows: list[dict[str, Any]] = []
    for reference, path in paths.items():
        if reference == "lac_v31_full":
            continue
        prediction = path / "out_of_fold_predictions.csv"
        if not prediction.is_file():
            continue
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            paired_rows.append(
                {"candidate": "lac_v31_full", "reference": reference}
                | _paired_v31(
                    full_prediction,
                    prediction,
                    metric,
                    replicates,
                    seed + metric_index,
                )
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v31_paired_bootstrap_95ci.csv", index=False)

    split_paths = dict(paths)
    for seed_offset in SEEDS:
        for model in ROBUSTNESS_MODELS:
            split_paths[f"seed{seed_offset}_{model}"] = (
                multiseed_root / f"seed{seed_offset}" / model
            )
    split = _split_audit(split_paths, paths["lac_v22_full"])
    for model, model_path in split_paths.items():
        if not model.startswith("lac_v31") and not model.startswith("seed"):
            continue
        for fold in range(1, 6):
            current = model_path / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split["model"] == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current) == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v31_split_audit.csv", index=False)

    seed_metrics, seed_summary, seed_effects = _seed_tables(multiseed_root)
    seed_metrics.to_csv(output / "v31_seed_metrics.csv", index=False)
    seed_summary.to_csv(output / "v31_seed_summary.csv", index=False)
    seed_effects.to_csv(output / "v31_seed_effects.csv", index=False)

    lag_scale_rows = []
    for seed_offset in SEEDS:
        model_path = multiseed_root / f"seed{seed_offset}" / "lac_v31_full"
        for fold in range(1, 6):
            lag_scale_rows.append({
                "training_seed_offset": seed_offset,
                "outer_fold": fold,
                "selected_lag_scale": _selected_lag_scale(model_path, fold),
            })
    lag_scales = pd.DataFrame(lag_scale_rows)
    lag_scales.to_csv(output / "v31_selected_lag_scales.csv", index=False)

    ablation_index = ablation.set_index("model")
    dominating = [
        name for name in V31_MODELS[1:]
        if _dominates(ablation_index.loc[name], full)
    ]
    formal_index = comparison.set_index("model")
    formal_best = {
        metric: bool(
            float(full[metric])
            <= float(formal_index.loc[list(FORMAL_COMPARATORS), metric].min())
            if metric in LOWER_IS_BETTER
            else float(full[metric])
            >= float(formal_index.loc[list(FORMAL_COMPARATORS), metric].max())
        )
        for metric in PRIMARY_METRICS
    }

    seed_acceptance: dict[str, Any] = {}
    for contrast in ("lag_full_vs_no_lag", "shared_dual_no_lag_vs_independent"):
        current = seed_effects[seed_effects["contrast"] == contrast]
        seed_acceptance[contrast] = {
            "tbr_mae_noninferior_seeds": int(current["tbr_mae_noninferior"].sum()),
            "cac_mae_noninferior_seeds": int(current["cac_mae_noninferior"].sum()),
            "cac_rmse_better_seeds": int(current["cac_rmse_better"].sum()),
            "cac_r2_better_seeds": int(current["cac_r2_better"].sum()),
            "joint_acceptance_seeds": int(current["joint_acceptance"].sum()),
        }

    full_row = ablation_index.loc["lac_v31_full"]
    negative_control_results = {}
    for control_name in ("lac_v31_history_permuted", "lac_v31_history_shifted"):
        control = ablation_index.loc[control_name]
        negative_control_results[control_name] = {
            "cac_mae_noninferior": bool(
                float(full_row.delta_log_cac_mae)
                <= float(control.delta_log_cac_mae) * 1.005
            ),
            "full_cac_rmse_better": bool(
                float(full_row.delta_log_cac_rmse)
                < float(control.delta_log_cac_rmse)
            ),
            "full_cac_r2_better": bool(
                float(full_row.delta_log_cac_r2)
                > float(control.delta_log_cac_r2)
            ),
        }
    negative_controls_pass = all(
        item["cac_mae_noninferior"]
        and item["full_cac_rmse_better"]
        and item["full_cac_r2_better"]
        for item in negative_control_results.values()
    )

    inner_columns = [column for column in split if column.startswith("inner_fold_")]
    split_pass = bool(
        (split["patient_count"] == 443).all()
        and (split["unique_patient_count"] == 443).all()
        and split["patient_ids_identical"].all()
        and split["outer_folds_identical"].all()
        and all(split[column].fillna(True).all() for column in inner_columns)
    )
    full_summary = _summary(paths["lac_v31_full"])
    lag_active_outer_folds_seed0 = int(
        np.count_nonzero(
            lag_scales.loc[
                lag_scales.training_seed_offset == 0, "selected_lag_scale"
            ].to_numpy(float) > 0
        )
    )
    innovation_pass = bool(
        seed_acceptance["shared_dual_no_lag_vs_independent"]["joint_acceptance_seeds"] >= 4
        and seed_acceptance["lag_full_vs_no_lag"]["joint_acceptance_seeds"] >= 4
        and lag_active_outer_folds_seed0 >= 4
        and negative_controls_pass
        and not dominating
    )
    assessment = {
        "frozen_commit": full_summary.get("git_revision"),
        "patient_count": int(full_summary.get("patient_count", 443)),
        "outer_fold_checksum": full_summary["cv_protocol"]["outer_fold_plan_checksum"],
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(all(formal_best.values())),
        "formal_comparator_metric_acceptance": formal_best,
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "full_metric_ranks_vs_formal_comparators": ranks,
        "split_and_patient_audit_all_passed": split_pass,
        "selected_lag_scale_by_outer_fold_seed0": lag_scales.loc[
            lag_scales.training_seed_offset == 0, "selected_lag_scale"
        ].tolist(),
        "lag_active_outer_folds_seed0": lag_active_outer_folds_seed0,
        "five_seed_directional_acceptance": seed_acceptance,
        "negative_control_results_seed0": negative_control_results,
        "negative_controls_passed_seed0": negative_controls_pass,
        "innovation_acceptance_passed": innovation_pass,
        "full_diagnostics_seed0": full_summary.get("diagnostics"),
        "interpretation": (
            "frozen exploratory internal validation on the repeatedly inspected "
            "443-patient development cohort; not independent confirmation"
        ),
        "outer_results_used_for_model_selection": False,
        "github_push": False,
    }
    (output / "v31_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31-root", required=True)
    parser.add_argument("--multiseed-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    version_map = json.loads(Path(arguments.version_map).read_text(encoding="utf-8"))
    result = aggregate_v31_results(
        arguments.v31_root,
        arguments.multiseed_root,
        arguments.formal_root,
        version_map,
        arguments.output,
        arguments.bootstrap,
        arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
