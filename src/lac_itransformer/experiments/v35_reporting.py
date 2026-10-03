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
from .v34_reporting import _contrast, _rank_rows
from ..training.v35_nested import V35_MODELS


FULL = "lac_v35_full"


def _mechanism_diagnostics(model: str, path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path / "out_of_fold_predictions.csv")
    target = (frame["true_delta_log_cac"].to_numpy(float) > 0.05).astype(int)
    probability = frame["coupling_gate"].to_numpy(float)
    if np.unique(target).size < 2:
        auc = average_precision = float("nan")
    else:
        auc = float(roc_auc_score(target, probability))
        average_precision = float(average_precision_score(target, probability))
    return {
        "model": model,
        "patient_count": int(len(frame)),
        "progression_prevalence": float(target.mean()),
        "roc_auc": auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(target, probability)),
        "probability_mean": float(probability.mean()),
        "probability_sd": float(probability.std(ddof=0)),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "collapsed": bool(probability.std(ddof=0) < 1e-6),
    }


def _training_diagnostics(model: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for fold in range(1, 6):
        checkpoint = torch.load(
            path / f"fold_{fold}" / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["state_dict"]
        raw = state["mechanism_burden_coefficient_raw"].detach().numpy()
        monotonic = bool(checkpoint["model_config"]["mechanism_monotonic"])
        coefficient = np.logaddexp(0.0, raw) if monotonic else raw
        transfer_enabled = bool(
            checkpoint["model_config"].get("shared_transfer_enabled", False)
        )
        transfer = (
            float(torch.sigmoid(state["shared_transfer_logit"]).item())
            if transfer_enabled
            else 0.0
        )
        history = json.loads(
            (path / f"fold_{fold}" / "refit_history.json").read_text(
                encoding="utf-8"
            )
        )
        cac_history = next(
            stage for stage in history["stages"] if stage["stage"] == "cac"
        )
        conflict = np.asarray(
            cac_history.get("mechanism_gradient_conflict_fraction", []), float
        )
        rows.append(
            {
                "model": model,
                "outer_fold": fold,
                "shared_transfer_alpha": transfer,
                "baseline_cac_zero_mechanism_coefficient": float(coefficient[0]),
                "baseline_cac_positive_mechanism_coefficient": float(coefficient[1]),
                "monotonic_constraint": monotonic,
                "negative_coefficient_count": int((coefficient < 0).sum()),
                "mechanism_conflict_epoch_mean": (
                    float(conflict.mean()) if conflict.size else 0.0
                ),
                "mechanism_conflict_epoch_max": (
                    float(conflict.max()) if conflict.size else 0.0
                ),
                "selected_tbr_epochs": int(history["stage_epochs"]["tbr"]),
                "selected_cac_epochs": int(history["stage_epochs"]["cac"]),
            }
        )
    return rows


def _gate_diagnostics(model: str, path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path / "out_of_fold_predictions.csv")
    result: dict[str, Any] = {"model": model}
    for column in ("adapter_gate_calcification", "coupling_gate"):
        values = frame[column].to_numpy(float)
        result |= {
            f"{column}_mean": float(values.mean()),
            f"{column}_sd": float(values.std(ddof=0)),
            f"{column}_q05": float(np.quantile(values, 0.05)),
            f"{column}_q50": float(np.quantile(values, 0.50)),
            f"{column}_q95": float(np.quantile(values, 0.95)),
            f"{column}_collapsed": bool(values.std(ddof=0) < 1e-6),
        }
    return result


def _paired_auc_difference(
    full_path: Path,
    reference_path: Path,
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    full = pd.read_csv(full_path).sort_values("patient_id").reset_index(drop=True)
    other = pd.read_csv(reference_path).sort_values("patient_id").reset_index(drop=True)
    if not full["patient_id"].equals(other["patient_id"]):
        raise RuntimeError("Mechanism bootstrap patient identifiers differ")
    target = (full["true_delta_log_cac"].to_numpy(float) > 0.05).astype(int)
    other_target = (other["true_delta_log_cac"].to_numpy(float) > 0.05).astype(int)
    if not np.array_equal(target, other_target):
        raise RuntimeError("Mechanism bootstrap targets differ")
    full_probability = full["coupling_gate"].to_numpy(float)
    other_probability = other["coupling_gate"].to_numpy(float)
    estimate = float(
        roc_auc_score(target, full_probability)
        - roc_auc_score(target, other_probability)
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        draw = rng.integers(0, len(target), len(target))
        if np.unique(target[draw]).size < 2:
            continue
        values.append(
            roc_auc_score(target[draw], full_probability[draw])
            - roc_auc_score(target[draw], other_probability[draw])
        )
    low, high = np.quantile(np.asarray(values, float), (0.025, 0.975))
    return {
        "candidate": FULL,
        "reference": reference,
        "difference_definition": "V3.5_Full_AUC_minus_reference_AUC",
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "full_significantly_favored": bool(low > 0),
        "reference_significantly_favored": bool(high < 0),
        "bootstrap_replicates": int(replicates),
        "valid_replicates": int(len(values)),
    }


def aggregate_v35_results(
    v35_root: str | Path,
    formal_root: str | Path,
    version_roots: dict[str, str | Path],
    output_dir: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    root, formal, output = Path(v35_root), Path(formal_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: formal / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in version_roots.items():
        paths[name], categories[name] = Path(value), "historical_version"
    for name in V35_MODELS:
        paths[name] = root / name
        categories[name] = "v35_full" if name == FULL else "v35_ablation"
    missing = [
        str(path / "summary.json")
        for path in paths.values()
        if not (path / "summary.json").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing))

    table = pd.DataFrame(
        [_result_row(name, path, categories[name]) for name, path in paths.items()]
    )
    table.to_csv(output / "v35_all_results.csv", index=False)
    indexed = table.set_index("model")
    comparison = indexed.loc[[FULL, *FORMAL_COMPARATORS]].reset_index()
    ablation = indexed.loc[list(V35_MODELS)].reset_index()
    historical = indexed.loc[[FULL, *version_roots.keys()]].reset_index()
    comparison.to_csv(output / "v35_comparison_results.csv", index=False)
    ablation.to_csv(output / "v35_ablation_results.csv", index=False)
    historical.to_csv(output / "v35_historical_reference.csv", index=False)
    formal_ranks = _rank_rows(indexed.loc[FULL], comparison)
    ablation_ranks = _rank_rows(indexed.loc[FULL], ablation)
    pd.DataFrame(formal_ranks).to_csv(output / "v35_formal_metric_ranks.csv", index=False)
    pd.DataFrame(ablation_ranks).to_csv(output / "v35_ablation_metric_ranks.csv", index=False)

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
            result["difference_definition"] = "V3.5_Full_minus_reference"
            paired_rows.append({"candidate": FULL, "reference": reference} | result)
    pd.DataFrame(paired_rows).to_csv(output / "v35_paired_bootstrap_95ci.csv", index=False)

    contrasts = [_contrast(FULL, name, table) for name in V35_MODELS[1:]]
    pd.DataFrame(contrasts).to_csv(output / "v35_core_contrasts.csv", index=False)
    mechanism = pd.DataFrame(
        [_mechanism_diagnostics(name, paths[name]) for name in V35_MODELS]
    )
    mechanism.to_csv(output / "v35_mechanism_diagnostics.csv", index=False)
    mechanism_bootstrap = pd.DataFrame(
        [
            _paired_auc_difference(
                paths[FULL] / "out_of_fold_predictions.csv",
                paths[name] / "out_of_fold_predictions.csv",
                name,
                replicates,
                seed + 35,
            )
            for name in V35_MODELS[1:]
        ]
    )
    mechanism_bootstrap.to_csv(
        output / "v35_mechanism_auc_paired_bootstrap_95ci.csv", index=False
    )
    training = pd.DataFrame(
        row
        for name in V35_MODELS
        for row in _training_diagnostics(name, paths[name])
    )
    training.to_csv(output / "v35_training_diagnostics.csv", index=False)
    gates = pd.DataFrame(
        [_gate_diagnostics(name, paths[name]) for name in V35_MODELS]
    )
    gates.to_csv(output / "v35_gate_diagnostics.csv", index=False)

    split = _split_audit({name: paths[name] for name in V35_MODELS}, paths["lac_v22_full"])
    for model in V35_MODELS:
        for fold in range(1, 6):
            current = paths[model] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split.model == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current)
                == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v35_split_audit.csv", index=False)

    full = indexed.loc[FULL]
    formal_best = {
        metric: bool(
            float(full[metric])
            <= float(indexed.loc[list(FORMAL_COMPARATORS), metric].min())
            if metric in LOWER_IS_BETTER
            else float(full[metric])
            >= float(indexed.loc[list(FORMAL_COMPARATORS), metric].max())
        )
        for metric in PRIMARY_METRICS
    }
    dominating = [
        name for name in V35_MODELS[1:] if _dominates(indexed.loc[name], full)
    ]
    mechanism_index = mechanism.set_index("model")
    full_training = training.loc[training.model == FULL]
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
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(
            all(formal_best.values())
        ),
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "full_metric_ranks_vs_formal_comparators": formal_ranks,
        "full_metric_ranks_vs_ablations": ablation_ranks,
        "shared_transfer": _contrast(FULL, "lac_v35_no_shared_transfer", table),
        "private_cac_encoder": _contrast(FULL, "lac_v35_no_private_cac_encoder", table),
        "cac_only_adapter": _contrast(FULL, "lac_v35_no_cac_adapter", table),
        "directional_gradient_protection": _contrast(FULL, "lac_v35_no_gradient_protection", table),
        "mechanism_conflict_protection": _contrast(FULL, "lac_v35_no_conflict_protection", table),
        "mechanism_auxiliary": _contrast(FULL, "lac_v35_no_mechanism_aux", table),
        "historical_burden": _contrast(FULL, "lac_v35_no_historical_burden", table),
        "hard_phenotype_views": _contrast(FULL, "lac_v35_no_hard_views", table),
        "baseline_anchoring": _contrast(FULL, "lac_v35_no_baseline_anchoring", table),
        "mechanism_progression_auc": float(mechanism_index.loc[FULL, "roc_auc"]),
        "no_history_progression_auc": float(
            mechanism_index.loc["lac_v35_no_historical_burden", "roc_auc"]
        ),
        "random_view_progression_auc": float(
            mechanism_index.loc["lac_v35_random_mechanism_view", "roc_auc"]
        ),
        "medical_history_auc_better_than_no_history": bool(
            mechanism_index.loc[FULL, "roc_auc"]
            > mechanism_index.loc["lac_v35_no_historical_burden", "roc_auc"]
        ),
        "medical_history_auc_better_than_random_view": bool(
            mechanism_index.loc[FULL, "roc_auc"]
            > mechanism_index.loc["lac_v35_random_mechanism_view", "roc_auc"]
        ),
        "medical_history_auc_vs_no_history_95ci": mechanism_bootstrap.set_index(
            "reference"
        ).loc["lac_v35_no_historical_burden"].to_dict(),
        "medical_history_auc_vs_random_view_95ci": mechanism_bootstrap.set_index(
            "reference"
        ).loc["lac_v35_random_mechanism_view"].to_dict(),
        "all_full_monotonic_coefficients_nonnegative": bool(
            (full_training.negative_coefficient_count == 0).all()
        ),
        "mean_full_shared_transfer_alpha": float(
            full_training.shared_transfer_alpha.mean()
        ),
        "interpretation": (
            "post-hoc exploratory internal validation on a repeatedly inspected "
            "443-patient development cohort; requires independent confirmation"
        ),
        "outer_results_used_for_current_model_selection": False,
        "github_push": False,
    }
    (output / "v35_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v35-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    mapping = json.loads(Path(arguments.version_map).read_text(encoding="utf-8"))
    result = aggregate_v35_results(
        arguments.v35_root,
        arguments.formal_root,
        mapping,
        arguments.output,
        arguments.bootstrap,
        arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
