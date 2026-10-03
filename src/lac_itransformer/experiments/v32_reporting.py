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
from ..training.v32_nested import V32_MODELS


def _selected_lag_scale(model_path: Path, fold: int) -> float:
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
        best = float(
            np.min(values) if metric in LOWER_IS_BETTER else np.max(values)
        )
        rows.append({
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
        })
    return rows


def aggregate_v32_results(
    v32_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    v32_root = Path(v32_root)
    formal_root = Path(formal_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {name: formal_root / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name] = Path(value)
        categories[name] = "historical_version"
    for name in V32_MODELS:
        paths[name] = v32_root / name
        categories[name] = "v32_full" if name == "lac_v32_full" else "v32_ablation"
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
    table.to_csv(output / "v32_all_results.csv", index=False)
    indexed = table.set_index("model")
    comparison = indexed.loc[["lac_v32_full", *FORMAL_COMPARATORS]].reset_index()
    comparison.to_csv(output / "v32_comparison_results.csv", index=False)
    ablation = indexed.loc[list(V32_MODELS)].reset_index()
    ablation.to_csv(output / "v32_ablation_results.csv", index=False)
    historical = indexed.loc[["lac_v32_full", *version_roots.keys()]].reset_index()
    historical.to_csv(output / "v32_historical_reference.csv", index=False)

    full = indexed.loc["lac_v32_full"]
    formal_candidates = indexed.loc[["lac_v32_full", *FORMAL_COMPARATORS]].reset_index()
    ablation_candidates = indexed.loc[list(V32_MODELS)].reset_index()
    formal_ranks = _rank_rows(full, formal_candidates)
    ablation_ranks = _rank_rows(full, ablation_candidates)
    pd.DataFrame(formal_ranks).to_csv(output / "v32_formal_metric_ranks.csv", index=False)
    pd.DataFrame(ablation_ranks).to_csv(output / "v32_ablation_metric_ranks.csv", index=False)

    full_prediction = paths["lac_v32_full"] / "out_of_fold_predictions.csv"
    paired_rows: list[dict[str, Any]] = []
    for reference, path in paths.items():
        if reference == "lac_v32_full":
            continue
        prediction = path / "out_of_fold_predictions.csv"
        if not prediction.is_file():
            continue
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            result = _paired_difference(
                full_prediction,
                prediction,
                metric,
                replicates,
                seed + metric_index,
            )
            result["difference_definition"] = "V3.2_Full_minus_reference"
            paired_rows.append({
                "candidate": "lac_v32_full",
                "reference": reference,
            } | result)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v32_paired_bootstrap_95ci.csv", index=False)

    split_paths = {name: paths[name] for name in V32_MODELS}
    split = _split_audit(split_paths, paths["lac_v22_full"])
    for model, model_path in split_paths.items():
        for fold in range(1, 6):
            current = model_path / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split["model"] == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current)
                == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v32_split_audit.csv", index=False)

    lag_rows = []
    for model in V32_MODELS:
        for fold in range(1, 6):
            lag_rows.append({
                "model": model,
                "outer_fold": fold,
                "selected_lag_scale": _selected_lag_scale(paths[model], fold),
            })
    lag_scales = pd.DataFrame(lag_rows)
    lag_scales.to_csv(output / "v32_selected_lag_scales.csv", index=False)

    ablation_index = ablation.set_index("model")
    dominating = [
        name for name in V32_MODELS[1:]
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

    contrast_rows = []
    for reference in V32_MODELS[1:]:
        row = ablation_index.loc[reference]
        contrast_rows.append({
            "reference": reference,
            **{
                f"full_minus_reference_{metric}": float(full[metric] - row[metric])
                for metric in PRIMARY_METRICS
            },
            "reference_pareto_dominates_full": bool(_dominates(row, full)),
            "full_pareto_dominates_reference": bool(_dominates(full, row)),
        })
    pd.DataFrame(contrast_rows).to_csv(output / "v32_ablation_contrasts.csv", index=False)

    no_lag = ablation_index.loc["lac_v32_no_lag"]
    dual_independent = ablation_index.loc["lac_v32_dual_independent"]
    lag_active_folds = int(np.count_nonzero(
        lag_scales.loc[lag_scales.model == "lac_v32_full", "selected_lag_scale"]
        .to_numpy(float) > 0
    ))
    lag_point_support = {
        "cac_mae_within_2_percent_of_no_lag": bool(
            float(full.delta_log_cac_mae)
            <= float(no_lag.delta_log_cac_mae) * 1.02
        ),
        "cac_rmse_better_than_no_lag": bool(
            float(full.delta_log_cac_rmse) < float(no_lag.delta_log_cac_rmse)
        ),
        "cac_r2_better_than_no_lag": bool(
            float(full.delta_log_cac_r2) > float(no_lag.delta_log_cac_r2)
        ),
        "lag_active_outer_folds": lag_active_folds,
    }
    negative_controls = {}
    for name in ("lac_v32_history_permuted", "lac_v32_history_shifted"):
        row = ablation_index.loc[name]
        negative_controls[name] = {
            "full_cac_mae_no_worse_2_percent": bool(
                float(full.delta_log_cac_mae) <= float(row.delta_log_cac_mae) * 1.02
            ),
            "full_cac_rmse_better": bool(
                float(full.delta_log_cac_rmse) < float(row.delta_log_cac_rmse)
            ),
            "full_cac_r2_better": bool(
                float(full.delta_log_cac_r2) > float(row.delta_log_cac_r2)
            ),
        }
    negative_controls_pass = all(all(value.values()) for value in negative_controls.values())
    split_columns = [column for column in split if column.startswith("inner_fold_")]
    split_pass = bool(
        (split["patient_count"] == 443).all()
        and (split["unique_patient_count"] == 443).all()
        and split["patient_ids_identical"].all()
        and split["outer_folds_identical"].all()
        and all(split[column].fillna(True).all() for column in split_columns)
    )
    dual_task_point_support = {
        "tbr_mae_no_worse_0_5_percent": bool(
            float(full.delta_tbr_mae) <= float(dual_independent.delta_tbr_mae) * 1.005
        ),
        "cac_mae_no_worse_2_percent": bool(
            float(full.delta_log_cac_mae)
            <= float(dual_independent.delta_log_cac_mae) * 1.02
        ),
        "cac_rmse_better": bool(
            float(full.delta_log_cac_rmse) < float(dual_independent.delta_log_cac_rmse)
        ),
        "cac_r2_better": bool(
            float(full.delta_log_cac_r2) > float(dual_independent.delta_log_cac_r2)
        ),
    }
    assessment = {
        "frozen_commit": _summary(paths["lac_v32_full"]).get("git_revision"),
        "patient_count": int(_summary(paths["lac_v32_full"])["patient_count"]),
        "outer_fold_checksum": _summary(paths["lac_v32_full"])["cv_protocol"]["outer_fold_plan_checksum"],
        "split_and_patient_audit_all_passed": split_pass,
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(all(formal_best.values())),
        "formal_comparator_metric_acceptance": formal_best,
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "full_metric_ranks_vs_formal_comparators": formal_ranks,
        "full_metric_ranks_vs_ablations": ablation_ranks,
        "dual_task_point_support": dual_task_point_support,
        "lag_point_support": lag_point_support,
        "negative_control_results": negative_controls,
        "negative_controls_passed": negative_controls_pass,
        "lag_innovation_point_acceptance": bool(
            all(value for key, value in lag_point_support.items() if key != "lag_active_outer_folds")
            and lag_active_folds >= 3
            and negative_controls_pass
        ),
        "full_diagnostics": _summary(paths["lac_v32_full"]).get("diagnostics"),
        "interpretation": (
            "frozen exploratory internal validation on the repeatedly inspected "
            "443-patient development cohort; not independent confirmation"
        ),
        "outer_results_used_for_model_selection": False,
        "github_push": False,
    }
    (output / "v32_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v32-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    version_map = json.loads(Path(arguments.version_map).read_text(encoding="utf-8"))
    result = aggregate_v32_results(
        arguments.v32_root,
        arguments.formal_root,
        version_map,
        arguments.output,
        arguments.bootstrap,
        arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
