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
from ..training.v33_nested import V33_MODELS


FULL = "lac_v33_full"


def _selected_history_scale(model_path: Path, fold: int) -> float:
    audit = json.loads(
        (model_path / f"fold_{fold}" / "decision_audit.json").read_text(
            encoding="utf-8"
        )
    )
    return float(audit["unique_fixed_configuration"]["cac"]["weight"])


def _rank_rows(full: pd.Series, candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric in PRIMARY_METRICS:
        values = candidates[metric].to_numpy(float)
        full_value = float(full[metric])
        better = (
            values < full_value - 1e-12
            if metric in LOWER_IS_BETTER
            else values > full_value + 1e-12
        )
        best = float(np.min(values) if metric in LOWER_IS_BETTER else np.max(values))
        rows.append(
            {
                "metric": metric,
                "full_value": full_value,
                "rank_with_ties": int(1 + better.sum()),
                "best_value": best,
                "best_models": ";".join(
                    candidates.loc[
                        np.isclose(candidates[metric], best, rtol=0, atol=1e-12),
                        "model",
                    ].astype(str)
                ),
            }
        )
    return rows


def _better(candidate: pd.Series, reference: pd.Series, metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return bool(float(candidate[metric]) < float(reference[metric]))
    return bool(float(candidate[metric]) > float(reference[metric]))


def _contrast_row(candidate_name: str, reference_name: str, indexed: pd.DataFrame):
    candidate, reference = indexed.loc[candidate_name], indexed.loc[reference_name]
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        **{
            f"candidate_minus_reference_{metric}": float(
                candidate[metric] - reference[metric]
            )
            for metric in PRIMARY_METRICS
        },
        "candidate_pareto_dominates_reference": _dominates(candidate, reference),
        "reference_pareto_dominates_candidate": _dominates(reference, candidate),
        "candidate_metrics_better": int(
            sum(_better(candidate, reference, metric) for metric in PRIMARY_METRICS)
        ),
    }


def aggregate_v33_results(
    v33_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    v33_root, formal_root, output = Path(v33_root), Path(formal_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: formal_root / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name], categories[name] = Path(value), "historical_version"
    for name in V33_MODELS:
        paths[name] = v33_root / name
        categories[name] = "v33_full" if name == FULL else "v33_ablation"
    missing = [str(path / "summary.json") for path in paths.values() if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing))

    table = pd.DataFrame(
        [_result_row(name, path, categories[name]) for name, path in paths.items()]
    )
    table.to_csv(output / "v33_all_results.csv", index=False)
    indexed = table.set_index("model")
    comparison = indexed.loc[[FULL, *FORMAL_COMPARATORS]].reset_index()
    ablation = indexed.loc[list(V33_MODELS)].reset_index()
    historical = indexed.loc[[FULL, *version_roots.keys()]].reset_index()
    comparison.to_csv(output / "v33_comparison_results.csv", index=False)
    ablation.to_csv(output / "v33_ablation_results.csv", index=False)
    historical.to_csv(output / "v33_historical_reference.csv", index=False)

    full = indexed.loc[FULL]
    formal_ranks = _rank_rows(full, comparison)
    ablation_ranks = _rank_rows(full, ablation)
    pd.DataFrame(formal_ranks).to_csv(output / "v33_formal_metric_ranks.csv", index=False)
    pd.DataFrame(ablation_ranks).to_csv(output / "v33_ablation_metric_ranks.csv", index=False)

    paired_rows: list[dict[str, Any]] = []
    full_prediction = paths[FULL] / "out_of_fold_predictions.csv"
    for reference, path in paths.items():
        if reference == FULL or not (path / "out_of_fold_predictions.csv").is_file():
            continue
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            result = _paired_difference(
                full_prediction,
                path / "out_of_fold_predictions.csv",
                metric,
                replicates,
                seed + metric_index,
            )
            result["difference_definition"] = "V3.3_Full_minus_reference"
            paired_rows.append({"candidate": FULL, "reference": reference} | result)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v33_paired_bootstrap_95ci.csv", index=False)

    innovation_pairs = (
        ("lac_v33_no_hurdle", "lac_v33_dual_independent", "protected_shared_vs_independent"),
        (FULL, "lac_v33_no_history", "full_vs_no_history"),
        (FULL, "lac_v33_current_only", "full_vs_recent_history_only"),
        (FULL, "lac_v33_history_permuted", "full_vs_permuted_history"),
        (FULL, "lac_v33_no_hurdle", "full_vs_no_progression_hurdle"),
        (FULL, "lac_v33_naive_shared_gradients", "protected_vs_naive_shared_gradients"),
        (FULL, "lac_v33_with_adapters", "without_vs_with_adapters"),
        (FULL, "lac_v33_no_treatment", "full_vs_no_treatment"),
        (FULL, "lac_v33_no_time_decay", "full_vs_no_time_decay"),
    )
    innovation_rows: list[dict[str, Any]] = []
    for candidate, reference, contrast in innovation_pairs:
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            result = _paired_difference(
                paths[candidate] / "out_of_fold_predictions.csv",
                paths[reference] / "out_of_fold_predictions.csv",
                metric,
                replicates,
                seed + 100 + metric_index,
            )
            result["difference_definition"] = "candidate_minus_reference"
            innovation_rows.append(
                {
                    "contrast": contrast,
                    "candidate": candidate,
                    "reference": reference,
                }
                | result
            )
    pd.DataFrame(innovation_rows).to_csv(
        output / "v33_innovation_paired_bootstrap_95ci.csv", index=False
    )

    split_paths = {name: paths[name] for name in V33_MODELS}
    split = _split_audit(split_paths, paths["lac_v22_full"])
    for model, model_path in split_paths.items():
        for fold in range(1, 6):
            current = model_path / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split["model"] == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current) == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v33_split_audit.csv", index=False)

    scale_rows = []
    for model in V33_MODELS:
        for fold in range(1, 6):
            scale_rows.append(
                {
                    "model": model,
                    "outer_fold": fold,
                    "selected_history_hurdle_scale": _selected_history_scale(paths[model], fold),
                }
            )
    scales = pd.DataFrame(scale_rows)
    scales.to_csv(output / "v33_selected_history_scales.csv", index=False)

    contrasts = [
        _contrast_row(FULL, reference, indexed) for reference in V33_MODELS[1:]
    ]
    # The no-hurdle protected shared encoder versus a separate CAC encoder is
    # the clean dual-task-sharing contrast, uncontaminated by the history head.
    contrasts.append(
        _contrast_row("lac_v33_no_hurdle", "lac_v33_dual_independent", indexed)
    )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output / "v33_core_contrasts.csv", index=False)

    dominating_ablations = [
        name for name in V33_MODELS[1:] if _dominates(indexed.loc[name], full)
    ]
    formal_best = {
        metric: bool(
            float(full[metric]) <= float(indexed.loc[list(FORMAL_COMPARATORS), metric].min())
            if metric in LOWER_IS_BETTER
            else float(full[metric]) >= float(indexed.loc[list(FORMAL_COMPARATORS), metric].max())
        )
        for metric in PRIMARY_METRICS
    }
    history_active = int(
        np.count_nonzero(
            scales.loc[scales.model == FULL, "selected_history_hurdle_scale"].to_numpy(float) > 0
        )
    )
    full_vs_no_history = _contrast_row(FULL, "lac_v33_no_history", indexed)
    full_vs_current = _contrast_row(FULL, "lac_v33_current_only", indexed)
    full_vs_permuted = _contrast_row(FULL, "lac_v33_history_permuted", indexed)
    full_vs_no_hurdle = _contrast_row(FULL, "lac_v33_no_hurdle", indexed)
    shared_vs_independent = _contrast_row(
        "lac_v33_no_hurdle", "lac_v33_dual_independent", indexed
    )

    def cac_support(contrast: dict[str, Any]) -> dict[str, bool]:
        return {
            "mae_no_worse_1_percent": bool(
                float(indexed.loc[contrast["candidate"], "delta_log_cac_mae"])
                <= float(indexed.loc[contrast["reference"], "delta_log_cac_mae"]) * 1.01
            ),
            "rmse_better": _better(
                indexed.loc[contrast["candidate"]], indexed.loc[contrast["reference"]],
                "delta_log_cac_rmse",
            ),
            "r2_better": _better(
                indexed.loc[contrast["candidate"]], indexed.loc[contrast["reference"]],
                "delta_log_cac_r2",
            ),
        }

    history_support = cac_support(full_vs_no_history)
    current_support = cac_support(full_vs_current)
    negative_control_support = cac_support(full_vs_permuted)
    hurdle_support = cac_support(full_vs_no_hurdle)
    split_columns = [column for column in split if column.startswith("inner_fold_")]
    split_pass = bool(
        (split["patient_count"] == 443).all()
        and (split["unique_patient_count"] == 443).all()
        and split["patient_ids_identical"].all()
        and split["outer_folds_identical"].all()
        and all(split[column].fillna(True).all() for column in split_columns)
    )
    dual_task_support = {
        "tbr_mae_no_worse_0_5_percent": bool(
            float(indexed.loc["lac_v33_no_hurdle", "delta_tbr_mae"])
            <= float(indexed.loc["lac_v33_dual_independent", "delta_tbr_mae"]) * 1.005
        ),
        "shared_metrics_better_count": shared_vs_independent["candidate_metrics_better"],
        "shared_pareto_dominates_independent": shared_vs_independent[
            "candidate_pareto_dominates_reference"
        ],
    }
    full_summary = _summary(paths[FULL])
    assessment = {
        "frozen_commit": full_summary.get("git_revision"),
        "patient_count": int(full_summary["patient_count"]),
        "outer_fold_checksum": full_summary["cv_protocol"]["outer_fold_plan_checksum"],
        "split_and_patient_audit_all_passed": split_pass,
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(all(formal_best.values())),
        "formal_comparator_metric_acceptance": formal_best,
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating_ablations),
        "dominating_ablations": dominating_ablations,
        "full_metric_ranks_vs_formal_comparators": formal_ranks,
        "full_metric_ranks_vs_ablations": ablation_ranks,
        "dual_task_sharing_point_support": dual_task_support,
        "historical_I_to_C_point_support": history_support,
        "historical_I_to_C_active_outer_folds": history_active,
        "full_vs_current_only_support": current_support,
        "permuted_history_negative_control_support": negative_control_support,
        "progression_hurdle_point_support": hurdle_support,
        "historical_I_to_C_acceptance": bool(
            all(history_support.values())
            and history_active >= 3
            and all(current_support.values())
            and all(negative_control_support.values())
        ),
        "full_diagnostics": full_summary.get("diagnostics"),
        "interpretation": (
            "frozen post-hoc exploratory internal validation on the repeatedly "
            "inspected 443-patient development cohort; requires independent confirmation"
        ),
        "outer_results_used_for_current_model_selection": False,
        "github_push": False,
    }
    (output / "v33_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v33-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    version_map = json.loads(Path(arguments.version_map).read_text(encoding="utf-8"))
    result = aggregate_v33_results(
        arguments.v33_root,
        arguments.formal_root,
        version_map,
        arguments.output,
        arguments.bootstrap,
        arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
