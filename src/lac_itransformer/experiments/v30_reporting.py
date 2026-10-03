from __future__ import annotations

import argparse
import json
from pathlib import Path

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


V30_MODELS = (
    "lac_v30_full",
    "lac_v30_dual_independent",
    "lac_v30_no_lag",
    "lac_v30_no_tbr_teacher",
    "lac_v30_no_shared_encoder",
    "lac_v30_no_treatment",
    "lac_v30_no_real_time",
    "lac_v30_no_adapters",
)


def aggregate_v30_results(
    v30_root,
    formal_root,
    version_roots,
    output_dir,
    replicates=2000,
    seed=2026,
):
    v30_root = Path(v30_root)
    formal_root = Path(formal_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: formal_root / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name] = Path(value)
        categories[name] = "historical_version"
    for name in V30_MODELS:
        paths[name] = v30_root / name
        categories[name] = "v30_full" if name == "lac_v30_full" else "v30_ablation"
    missing = [str(path / "summary.json") for path in paths.values() if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing))

    table = pd.DataFrame(
        [_result_row(name, path, categories[name]) for name, path in paths.items()]
    )
    table.to_csv(output / "v30_all_results.csv", index=False)
    comparison = table.set_index("model").loc[
        ["lac_v30_full", *FORMAL_COMPARATORS]
    ].reset_index()
    comparison.to_csv(output / "v30_comparison_results.csv", index=False)
    ablation = table.set_index("model").loc[list(V30_MODELS)].reset_index()
    ablation.to_csv(output / "v30_ablation_results.csv", index=False)
    historical = table.set_index("model").loc[
        ["lac_v30_full", *version_roots.keys()]
    ].reset_index()
    historical.to_csv(output / "v30_historical_version_reference.csv", index=False)

    full = table.set_index("model").loc["lac_v30_full"]
    acceptance = table.set_index("model").loc[
        [*FORMAL_COMPARATORS, *V30_MODELS]
    ].reset_index()
    ranks = []
    for metric in PRIMARY_METRICS:
        values = acceptance[metric].to_numpy(float)
        full_value = float(full[metric])
        rank = int(
            1 + np.sum(
                values < full_value - 1e-12
                if metric in LOWER_IS_BETTER
                else values > full_value + 1e-12
            )
        )
        best = float(np.min(values) if metric in LOWER_IS_BETTER else np.max(values))
        ranks.append({
            "metric": metric,
            "full_value": full_value,
            "rank_with_ties": rank,
            "best_value": best,
        })
    pd.DataFrame(ranks).to_csv(output / "v30_primary_metric_ranks.csv", index=False)

    paired_rows = []
    full_prediction = paths["lac_v30_full"] / "out_of_fold_predictions.csv"
    for reference, path in paths.items():
        if reference == "lac_v30_full":
            continue
        prediction = path / "out_of_fold_predictions.csv"
        if not prediction.is_file():
            continue
        for index, metric in enumerate(PRIMARY_METRICS):
            difference = _paired_difference(
                full_prediction, prediction, metric, replicates, seed + index
            )
            difference["difference_definition"] = "V3.0_Full_minus_reference"
            paired_rows.append(
                {"candidate": "lac_v30_full", "reference": reference}
                | difference
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v30_paired_bootstrap_95ci.csv", index=False)

    split = _split_audit(paths, paths["lac_v22_full"])
    for model in V30_MODELS:
        for fold in range(1, 6):
            current = paths[model] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split["model"] == model, f"inner_fold_{fold}_identical"] = (
                _normalized_plan_checksum(current) == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v30_split_audit.csv", index=False)

    ablation_index = ablation.set_index("model")
    dominating = [
        name for name in V30_MODELS[1:]
        if _dominates(ablation_index.loc[name], full)
    ]
    independent = ablation_index.loc["lac_v30_dual_independent"]
    no_lag = ablation_index.loc["lac_v30_no_lag"]
    no_teacher = ablation_index.loc["lac_v30_no_tbr_teacher"]
    tbr_noninferior = (
        float(full["delta_tbr_mae"])
        <= float(independent["delta_tbr_mae"]) * 1.005
    )
    cac_mae_noninferior = (
        float(full["delta_log_cac_mae"])
        <= float(no_lag["delta_log_cac_mae"]) * 1.005
    )
    lag_point_gain = (
        float(full["delta_log_cac_rmse"]) < float(no_lag["delta_log_cac_rmse"])
        or float(full["delta_log_cac_r2"]) > float(no_lag["delta_log_cac_r2"])
    )
    teacher_point_gain = any(
        float(full[metric]) < float(no_teacher[metric])
        if metric in LOWER_IS_BETTER
        else float(full[metric]) > float(no_teacher[metric])
        for metric in (
            "delta_log_cac_mae", "delta_log_cac_rmse", "delta_log_cac_r2"
        )
    )
    selected_lag_weights = []
    for fold in range(1, 6):
        audit = json.loads(
            (paths["lac_v30_full"] / f"fold_{fold}" / "decision_audit.json").read_text()
        )
        selected_lag_weights.append(
            float(audit["unique_fixed_configuration"]["cac"]["weight"])
        )
    assessment = {
        "patient_count": int(_summary(paths["lac_v30_full"])["patient_count"]),
        "full_tbr_mae_noninferior_to_dual_independent": bool(tbr_noninferior),
        "full_cac_mae_noninferior_to_no_lag": bool(cac_mae_noninferior),
        "full_cac_rmse_or_r2_point_gain_over_no_lag": bool(lag_point_gain),
        "full_has_cac_point_gain_over_no_tbr_teacher": bool(teacher_point_gain),
        "selected_lag_scale_by_outer_fold": selected_lag_weights,
        "lag_active_in_at_least_four_outer_folds": bool(
            np.count_nonzero(np.asarray(selected_lag_weights) > 0) >= 4
        ),
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "innovation_acceptance_passed": bool(
            tbr_noninferior
            and cac_mae_noninferior
            and lag_point_gain
            and teacher_point_gain
            and np.count_nonzero(np.asarray(selected_lag_weights) > 0) >= 4
            and not dominating
        ),
        "interpretation": (
            "exploratory internal evidence because the same 443-patient "
            "development cohort has informed prior architecture revisions"
        ),
        "github_push": False,
    }
    (output / "v30_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    return assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v30-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    version_roots = json.loads(Path(args.version_map).read_text())
    result = aggregate_v30_results(
        args.v30_root, args.formal_root, version_roots, args.output,
        args.bootstrap, args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
