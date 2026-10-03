from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
from .v35_reporting import _gate_diagnostics, _mechanism_diagnostics
from ..training.v36_nested import V36_MODELS


FULL = "lac_v36_full"


def _paired_mechanism_auc(
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
    other_target = (
        other["true_delta_log_cac"].to_numpy(float) > 0.05
    ).astype(int)
    if not np.array_equal(target, other_target):
        raise RuntimeError("Mechanism bootstrap targets differ")
    candidate = full["coupling_gate"].to_numpy(float)
    comparator = other["coupling_gate"].to_numpy(float)
    estimate = float(
        roc_auc_score(target, candidate) - roc_auc_score(target, comparator)
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        draw = rng.integers(0, len(target), len(target))
        if np.unique(target[draw]).size < 2:
            continue
        values.append(
            roc_auc_score(target[draw], candidate[draw])
            - roc_auc_score(target[draw], comparator[draw])
        )
    low, high = np.quantile(np.asarray(values, float), (0.025, 0.975))
    return {
        "candidate": FULL,
        "reference": reference,
        "difference_definition": "V3.6_Full_AUC_minus_reference_AUC",
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "full_significantly_favored": bool(low > 0),
        "reference_significantly_favored": bool(high < 0),
        "bootstrap_replicates": int(replicates),
        "valid_replicates": int(len(values)),
    }


def _decision_rows(model: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    mode = (
        "identity"
        if model == "lac_v36_no_calibration"
        else "expert"
        if model == "lac_v36_calibration_only"
        else "selected"
    )
    for fold in range(1, 6):
        audit = json.loads(
            (path / f"fold_{fold}" / "decision_audit.json").read_text(
                encoding="utf-8"
            )
        )
        task = audit["tasks"]["cac"]
        rows.append(
            {
                "model": model,
                "outer_fold": fold,
                "applied_mode": mode,
                "inner_selected_weight": float(task["selected_weight"]),
                "inner_raw_mae": float(task["raw_metrics"]["mae"]),
                "inner_raw_rmse": float(task["raw_metrics"]["rmse"]),
                "inner_selected_mae": float(task["selected_metrics"]["mae"]),
                "inner_selected_rmse": float(task["selected_metrics"]["rmse"]),
                "consistent_inner_folds": int(task["selected_consistent_folds"]),
                "outer_test_labels_used": bool(audit["outer_test_labels_used"]),
            }
        )
    return rows


def _metric_best(full: pd.Series, others: pd.DataFrame) -> dict[str, bool]:
    return {
        metric: bool(
            float(full[metric]) <= float(others[metric].min()) + 1e-12
            if metric in LOWER_IS_BETTER
            else float(full[metric]) >= float(others[metric].max()) - 1e-12
        )
        for metric in PRIMARY_METRICS
    }


def _write_report(
    path: Path,
    comparison: pd.DataFrame,
    ablation: pd.DataFrame,
    assessment: dict[str, Any],
) -> None:
    columns = ["model", *PRIMARY_METRICS]

    def markdown(frame: pd.DataFrame) -> str:
        rows = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for record in frame[columns].itertuples(index=False, name=None):
            cells = [str(record[0])] + [f"{float(value):.6f}" for value in record[1:]]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    comparison_text = markdown(comparison)
    ablation_text = markdown(ablation)
    formal_pass = assessment[
        "full_best_point_estimate_for_all_six_vs_formal_comparators"
    ]
    ablation_pass = assessment[
        "full_best_point_estimate_for_all_six_vs_ablations"
    ]
    text = f"""# LAC-iTransformer V3.6 内部开发验证报告

## 结果定位

本结果来自对同一 443 例开发队列反复检查后的探索性内部验证，不是独立确认性证据。结构、候选权重及选择规则已在本轮外层测试前冻结；患者级外层五折仅用于一次性评价，所有校准选择均来自各外层训练池的内层 OOF。

## 正式对比

{comparison_text}

六项点估计均不差于正式对比模型：**{formal_pass}**。

## 核心消融

{ablation_text}

Full 六项点估计均不差于全部预注册消融：**{ablation_pass}**。Pareto 全面支配 Full 的消融：{assessment['dominating_ablations'] or '无'}。

## 解释边界

V3.6 的主要预测创新是同一双任务网络中的硬炎症/钙化表型视图、TBR 所有权式共享编码、受方向性梯度保护的严格历史 I→C 低秩传递、基线锚定，以及仅用内层 OOF 选择且可退化为原始 CAC 输出的稳健临床专家。炎症历史机制辅助任务不直接加到连续 CAC 预测上，因此其作用须同时依据预测消融、机制 AUC和门控诊断解释；若置信区间跨 0，只能报告方向性或探索性结果。
"""
    path.write_text(text, encoding="utf-8")


def aggregate_v36_results(
    v36_root: str | Path,
    formal_root: str | Path,
    output_dir: str | Path,
    historical_roots: dict[str, str | Path] | None = None,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    root, formal, output = Path(v36_root), Path(formal_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    historical_roots = dict(historical_roots or {})
    paths = {name: formal / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in historical_roots.items():
        paths[name], categories[name] = Path(value), "historical_version"
    for name in V36_MODELS:
        paths[name] = root / name
        categories[name] = "v36_full" if name == FULL else "v36_ablation"
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
    table.to_csv(output / "v36_all_results.csv", index=False)
    indexed = table.set_index("model")
    comparison = indexed.loc[[FULL, *FORMAL_COMPARATORS]].reset_index()
    ablation = indexed.loc[list(V36_MODELS)].reset_index()
    comparison.to_csv(output / "v36_comparison_results.csv", index=False)
    ablation.to_csv(output / "v36_ablation_results.csv", index=False)
    if historical_roots:
        indexed.loc[[FULL, *historical_roots]].reset_index().to_csv(
            output / "v36_historical_reference.csv", index=False
        )

    formal_ranks = _rank_rows(indexed.loc[FULL], comparison)
    ablation_ranks = _rank_rows(indexed.loc[FULL], ablation)
    pd.DataFrame(formal_ranks).to_csv(
        output / "v36_formal_metric_ranks.csv", index=False
    )
    pd.DataFrame(ablation_ranks).to_csv(
        output / "v36_ablation_metric_ranks.csv", index=False
    )

    paired = []
    for reference, reference_path in paths.items():
        if reference == FULL:
            continue
        for index, metric in enumerate(PRIMARY_METRICS):
            result = _paired_difference(
                paths[FULL] / "out_of_fold_predictions.csv",
                reference_path / "out_of_fold_predictions.csv",
                metric,
                replicates,
                seed + index,
            )
            result["difference_definition"] = "V3.6_Full_minus_reference"
            paired.append({"candidate": FULL, "reference": reference} | result)
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / "v36_paired_bootstrap_95ci.csv", index=False)

    contrasts = pd.DataFrame(
        [_contrast(FULL, name, table) for name in V36_MODELS[1:]]
    )
    contrasts.to_csv(output / "v36_core_contrasts.csv", index=False)
    decisions = pd.DataFrame(
        row
        for name in V36_MODELS
        for row in _decision_rows(name, paths[name])
    )
    decisions.to_csv(output / "v36_inner_decision_audit.csv", index=False)
    gates = pd.DataFrame(
        [_gate_diagnostics(name, paths[name]) for name in V36_MODELS]
    )
    gates.to_csv(output / "v36_gate_diagnostics.csv", index=False)
    mechanism = pd.DataFrame(
        [_mechanism_diagnostics(name, paths[name]) for name in V36_MODELS]
    )
    mechanism.to_csv(output / "v36_mechanism_diagnostics.csv", index=False)
    mechanism_auc = pd.DataFrame(
        [
            _paired_mechanism_auc(
                paths[FULL] / "out_of_fold_predictions.csv",
                paths[name] / "out_of_fold_predictions.csv",
                name,
                replicates,
                seed + 36,
            )
            for name in (
                "lac_v36_no_historical_burden",
                "lac_v36_no_mechanism_aux",
                "lac_v36_no_hard_views",
            )
        ]
    )
    mechanism_auc.to_csv(
        output / "v36_mechanism_auc_paired_bootstrap_95ci.csv", index=False
    )

    split = _split_audit(
        {name: paths[name] for name in V36_MODELS}, paths["lac_v22_full"]
    )
    for model in V36_MODELS:
        for fold in range(1, 6):
            current = paths[model] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = (
                paths["lac_v22_full"]
                / f"fold_{fold}"
                / "inner_patient_fold_plan.csv"
            )
            split.loc[split.model == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current)
                == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v36_split_audit.csv", index=False)

    full = indexed.loc[FULL]
    formal_best = _metric_best(full, indexed.loc[list(FORMAL_COMPARATORS)])
    ablation_best = _metric_best(full, indexed.loc[list(V36_MODELS[1:])])
    dominating = [
        name for name in V36_MODELS[1:] if _dominates(indexed.loc[name], full)
    ]
    split_columns = [column for column in split if column.startswith("inner_fold_")]
    split_pass = bool(
        (split.patient_count == 443).all()
        and (split.unique_patient_count == 443).all()
        and split.patient_ids_identical.all()
        and split.outer_folds_identical.all()
        and all(split[column].fillna(True).all() for column in split_columns)
    )
    full_decision = decisions.loc[decisions.model == FULL]
    assessment = {
        "frozen_commit": _summary(paths[FULL]).get("git_revision"),
        "patient_count": int(_summary(paths[FULL])["patient_count"]),
        "split_and_patient_audit_all_passed": split_pass,
        "formal_comparator_metric_acceptance": formal_best,
        "ablation_metric_acceptance": ablation_best,
        "full_best_point_estimate_for_all_six_vs_formal_comparators": bool(
            all(formal_best.values())
        ),
        "full_best_point_estimate_for_all_six_vs_ablations": bool(
            all(ablation_best.values())
        ),
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "full_metric_ranks_vs_formal_comparators": formal_ranks,
        "full_metric_ranks_vs_ablations": ablation_ranks,
        "inner_selected_cac_weights": full_decision[
            "inner_selected_weight"
        ].tolist(),
        "all_inner_decisions_exclude_outer_labels": bool(
            (~decisions.outer_test_labels_used).all()
        ),
        "calibration": _contrast(FULL, "lac_v36_no_calibration", table),
        "strict_historical_shared_transfer": _contrast(
            FULL, "lac_v36_no_shared_transfer", table
        ),
        "cac_residual_adapter": _contrast(FULL, "lac_v36_no_cac_adapter", table),
        "directional_gradient_protection": _contrast(
            FULL, "lac_v36_no_gradient_protection", table
        ),
        "shared_dual_task_core": _contrast(FULL, "lac_v36_dual_independent", table),
        "hard_phenotype_views": _contrast(FULL, "lac_v36_no_hard_views", table),
        "baseline_anchoring": _contrast(
            FULL, "lac_v36_no_baseline_anchoring", table
        ),
        "mechanism_auxiliary": _contrast(FULL, "lac_v36_no_mechanism_aux", table),
        "historical_burden": _contrast(
            FULL, "lac_v36_no_historical_burden", table
        ),
        "mechanism_auc_paired_bootstrap": mechanism_auc.to_dict("records"),
        "interpretation": (
            "post-hoc exploratory internal validation on a repeatedly inspected "
            "443-patient development cohort; requires independent confirmation"
        ),
        "outer_results_used_for_current_model_selection": False,
        "github_push": False,
    }
    (output / "v36_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    _write_report(output / "V36_INTERNAL_REPORT_CN.md", comparison, ablation, assessment)
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v36-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-map")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    mapping = (
        json.loads(Path(args.historical_map).read_text(encoding="utf-8"))
        if args.historical_map
        else {}
    )
    result = aggregate_v36_results(
        args.v36_root,
        args.formal_root,
        args.output,
        historical_roots=mapping,
        replicates=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
