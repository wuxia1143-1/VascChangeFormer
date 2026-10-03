from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
from ..training.v34_nested import V34_MODELS


FULL = "lac_v34_full"


def _better(candidate: pd.Series, reference: pd.Series, metric: str) -> bool:
    return bool(
        float(candidate[metric]) < float(reference[metric])
        if metric in LOWER_IS_BETTER
        else float(candidate[metric]) > float(reference[metric])
    )


def _rank_rows(full: pd.Series, candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric in PRIMARY_METRICS:
        values = candidates[metric].to_numpy(float)
        value = float(full[metric])
        better = values < value - 1e-12 if metric in LOWER_IS_BETTER else values > value + 1e-12
        best = float(np.min(values) if metric in LOWER_IS_BETTER else np.max(values))
        rows.append(
            {
                "metric": metric,
                "full_value": value,
                "rank_with_ties": int(1 + better.sum()),
                "best_value": best,
                "best_models": ";".join(
                    candidates.loc[
                        np.isclose(candidates[metric], best, rtol=0, atol=1e-12), "model"
                    ].astype(str)
                ),
            }
        )
    return rows


def _contrast(candidate: str, reference: str, table: pd.DataFrame) -> dict[str, Any]:
    indexed = table.set_index("model")
    left, right = indexed.loc[candidate], indexed.loc[reference]
    return {
        "candidate": candidate,
        "reference": reference,
        **{
            f"candidate_minus_reference_{metric}": float(left[metric] - right[metric])
            for metric in PRIMARY_METRICS
        },
        "candidate_metrics_better": int(
            sum(_better(left, right, metric) for metric in PRIMARY_METRICS)
        ),
        "candidate_pareto_dominates_reference": _dominates(left, right),
        "reference_pareto_dominates_candidate": _dominates(right, left),
    }


def _mechanism_diagnostics(model: str, path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path / "out_of_fold_predictions.csv")
    target = (frame["true_delta_log_cac"].to_numpy(float) > 0.05).astype(int)
    probability = frame["coupling_gate"].to_numpy(float)
    if model == "lac_v34_no_mechanism_aux":
        probability = np.full_like(probability, 0.5)
    return {
        "model": model,
        "patient_count": int(len(frame)),
        "progression_prevalence": float(target.mean()),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "probability_mean": float(probability.mean()),
        "probability_sd": float(probability.std(ddof=0)),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "collapsed": bool(probability.std(ddof=0) < 1e-6),
    }


def _coefficient_rows(model: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for fold in range(1, 6):
        checkpoint = torch.load(path / f"fold_{fold}" / "model.pt", map_location="cpu", weights_only=False)
        raw = checkpoint["state_dict"]["mechanism_burden_coefficient_raw"].detach().numpy()
        monotonic = bool(checkpoint["model_config"]["mechanism_monotonic"])
        coefficient = np.logaddexp(0.0, raw) if monotonic else raw
        rows.append(
            {
                "model": model,
                "outer_fold": fold,
                "baseline_cac_zero_coefficient": float(coefficient[0]),
                "baseline_cac_positive_coefficient": float(coefficient[1]),
                "monotonic_constraint": monotonic,
                "negative_coefficient_count": int((coefficient < 0).sum()),
            }
        )
    return rows


def aggregate_v34_results(
    v34_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    root, formal, output = Path(v34_root), Path(formal_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: formal / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name], categories[name] = Path(value), "historical_version"
    for name in V34_MODELS:
        paths[name] = root / name
        categories[name] = "v34_full" if name == FULL else "v34_ablation"
    missing = [str(path / "summary.json") for path in paths.values() if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing))

    table = pd.DataFrame([_result_row(name, path, categories[name]) for name, path in paths.items()])
    table.to_csv(output / "v34_all_results.csv", index=False)
    indexed = table.set_index("model")
    comparison = indexed.loc[[FULL, *FORMAL_COMPARATORS]].reset_index()
    ablation = indexed.loc[list(V34_MODELS)].reset_index()
    historical = indexed.loc[[FULL, *version_roots.keys()]].reset_index()
    comparison.to_csv(output / "v34_comparison_results.csv", index=False)
    ablation.to_csv(output / "v34_ablation_results.csv", index=False)
    historical.to_csv(output / "v34_historical_reference.csv", index=False)
    formal_ranks = _rank_rows(indexed.loc[FULL], comparison)
    ablation_ranks = _rank_rows(indexed.loc[FULL], ablation)
    pd.DataFrame(formal_ranks).to_csv(output / "v34_formal_metric_ranks.csv", index=False)
    pd.DataFrame(ablation_ranks).to_csv(output / "v34_ablation_metric_ranks.csv", index=False)

    paired_rows = []
    full_prediction = paths[FULL] / "out_of_fold_predictions.csv"
    for reference, path in paths.items():
        if reference == FULL:
            continue
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            result = _paired_difference(
                full_prediction,
                path / "out_of_fold_predictions.csv",
                metric,
                replicates,
                seed + metric_index,
            )
            result["difference_definition"] = "V3.4_Full_minus_reference"
            paired_rows.append({"candidate": FULL, "reference": reference} | result)
    pd.DataFrame(paired_rows).to_csv(output / "v34_paired_bootstrap_95ci.csv", index=False)

    contrasts = [_contrast(FULL, name, table) for name in V34_MODELS[1:]]
    pd.DataFrame(contrasts).to_csv(output / "v34_core_contrasts.csv", index=False)

    mechanism = pd.DataFrame([_mechanism_diagnostics(name, paths[name]) for name in V34_MODELS])
    mechanism.to_csv(output / "v34_mechanism_diagnostics.csv", index=False)
    coefficients = pd.DataFrame(
        row for name in V34_MODELS for row in _coefficient_rows(name, paths[name])
    )
    coefficients.to_csv(output / "v34_mechanism_coefficients.csv", index=False)

    split = _split_audit({name: paths[name] for name in V34_MODELS}, paths["lac_v22_full"])
    for model in V34_MODELS:
        for fold in range(1, 6):
            current = paths[model] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split.model == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current) == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v34_split_audit.csv", index=False)

    full = indexed.loc[FULL]
    formal_best = {
        metric: bool(
            float(full[metric]) <= float(indexed.loc[list(FORMAL_COMPARATORS), metric].min())
            if metric in LOWER_IS_BETTER
            else float(full[metric]) >= float(indexed.loc[list(FORMAL_COMPARATORS), metric].max())
        )
        for metric in PRIMARY_METRICS
    }
    dominating = [name for name in V34_MODELS[1:] if _dominates(indexed.loc[name], full)]
    full_mechanism = mechanism.set_index("model").loc[FULL]
    random_mechanism = mechanism.set_index("model").loc["lac_v34_random_mechanism_view"]
    split_columns = [column for column in split if column.startswith("inner_fold_")]
    split_pass = bool(
        (split.patient_count == 443).all()
        and (split.unique_patient_count == 443).all()
        and split.patient_ids_identical.all()
        and split.outer_folds_identical.all()
        and all(split[column].fillna(True).all() for column in split_columns)
    )
    assessment = {
        "frozen_commit": _summary(paths[FULL]).get("git_revision"),
        "patient_count": int(_summary(paths[FULL])["patient_count"]),
        "split_and_patient_audit_all_passed": split_pass,
        "formal_comparator_metric_acceptance": formal_best,
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(all(formal_best.values())),
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "full_metric_ranks_vs_formal_comparators": formal_ranks,
        "full_metric_ranks_vs_ablations": ablation_ranks,
        "dual_task_shared_vs_independent": _contrast(FULL, "lac_v34_dual_independent", table),
        "gradient_protection": _contrast(FULL, "lac_v34_no_gradient_protection", table),
        "baseline_anchoring": _contrast(FULL, "lac_v34_no_baseline_anchoring", table),
        "mechanism_auxiliary": _contrast(FULL, "lac_v34_no_mechanism_aux", table),
        "hard_phenotype_views": _contrast(FULL, "lac_v34_no_hard_views", table),
        "mechanism_progression_auc": float(full_mechanism.roc_auc),
        "random_view_progression_auc": float(random_mechanism.roc_auc),
        "mechanism_auc_above_chance": bool(full_mechanism.roc_auc > 0.5),
        "medical_view_auc_better_than_random_view": bool(full_mechanism.roc_auc > random_mechanism.roc_auc),
        "all_monotonic_coefficients_nonnegative": bool(
            (coefficients.loc[coefficients.model == FULL, "negative_coefficient_count"] == 0).all()
        ),
        "interpretation": "post-hoc exploratory internal validation on a repeatedly inspected 443-patient cohort; requires independent confirmation",
        "outer_results_used_for_current_model_selection": False,
        "github_push": False,
    }
    (output / "v34_result_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v34-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    mapping = json.loads(Path(args.version_map).read_text(encoding="utf-8"))
    result = aggregate_v34_results(args.v34_root, args.formal_root, mapping, args.output, args.bootstrap, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
