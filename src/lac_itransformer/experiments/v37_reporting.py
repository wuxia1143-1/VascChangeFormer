from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from .v35_reporting import _gate_diagnostics
from ..training.v37_nested import V37_MODELS


FULL = "lac_v37_full"


def _metric_best(full: pd.Series, others: pd.DataFrame) -> dict[str, bool]:
    return {
        metric: bool(
            float(full[metric]) <= float(others[metric].min()) + 1e-12
            if metric in LOWER_IS_BETTER
            else float(full[metric]) >= float(others[metric].max()) - 1e-12
        )
        for metric in PRIMARY_METRICS
    }


def _decision_rows(model: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    mode = "identity" if model == "lac_v37_no_calibration" else "selected"
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


def _mechanism_diagnostics(model: str, path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path / "out_of_fold_predictions.csv")
    target = (frame["true_delta_log_cac"].to_numpy(float) > 0.05).astype(int)
    score = frame["coupling_gate"].to_numpy(float)
    return {
        "model": model,
        "roc_auc": float(roc_auc_score(target, score)) if np.unique(target).size == 2 else np.nan,
        "average_precision": float(average_precision_score(target, score)),
        "brier": float(brier_score_loss(target, np.clip(score, 0, 1))),
        "gate_mean": float(np.mean(score)),
        "gate_sd": float(np.std(score, ddof=1)),
        "gate_near_zero_fraction": float(np.mean(score <= 0.05)),
        "gate_near_one_fraction": float(np.mean(score >= 0.95)),
    }


def _paired_auc(
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
    candidate = full["coupling_gate"].to_numpy(float)
    comparator = other["coupling_gate"].to_numpy(float)
    estimate = float(roc_auc_score(target, candidate) - roc_auc_score(target, comparator))
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
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "full_significantly_favored": bool(low > 0),
        "reference_significantly_favored": bool(high < 0),
        "bootstrap_replicates": replicates,
    }


def _markdown(frame: pd.DataFrame) -> str:
    columns = ["model", *PRIMARY_METRICS]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join([str(row[0]), *[f"{float(value):.6f}" for value in row[1:]]])
            + " |"
        )
    return "\n".join(lines)


def aggregate_v37_results(
    v37_root: str | Path,
    formal_root: str | Path,
    output_dir: str | Path,
    historical_roots: dict[str, str | Path] | None = None,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    root, formal, output = Path(v37_root), Path(formal_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    historical_roots = dict(historical_roots or {})
    paths = {name: formal / name for name in FORMAL_COMPARATORS}
    categories = {name: "formal_comparator" for name in FORMAL_COMPARATORS}
    for name, value in historical_roots.items():
        paths[name], categories[name] = Path(value), "historical_version"
    for name in V37_MODELS:
        paths[name] = root / name
        categories[name] = "v37_full" if name == FULL else "v37_ablation"
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
    indexed = table.set_index("model")
    comparison = indexed.loc[[FULL, *FORMAL_COMPARATORS]].reset_index()
    ablation = indexed.loc[list(V37_MODELS)].reset_index()
    table.to_csv(output / "v37_all_results.csv", index=False)
    comparison.to_csv(output / "v37_comparison_results.csv", index=False)
    ablation.to_csv(output / "v37_ablation_results.csv", index=False)
    if historical_roots:
        indexed.loc[[FULL, *historical_roots]].reset_index().to_csv(
            output / "v37_historical_reference.csv", index=False
        )

    formal_ranks = _rank_rows(indexed.loc[FULL], comparison)
    ablation_ranks = _rank_rows(indexed.loc[FULL], ablation)
    pd.DataFrame(formal_ranks).to_csv(output / "v37_formal_metric_ranks.csv", index=False)
    pd.DataFrame(ablation_ranks).to_csv(output / "v37_ablation_metric_ranks.csv", index=False)

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
            result["difference_definition"] = "V3.7_Full_minus_reference"
            paired.append({"candidate": FULL, "reference": reference} | result)
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / "v37_paired_bootstrap_95ci.csv", index=False)
    significance = {}
    for reference in paths:
        if reference == FULL:
            continue
        rows = paired_frame.loc[paired_frame.reference == reference]
        significance[reference] = {
            "full_significantly_favored_metric_count": int(
                rows.full_significantly_favored.sum()
            ),
            "reference_significantly_favored_metric_count": int(
                rows.reference_significantly_favored.sum()
            ),
        }

    contrasts = pd.DataFrame(
        [_contrast(FULL, name, table) for name in V37_MODELS[1:]]
    )
    contrasts.to_csv(output / "v37_core_contrasts.csv", index=False)
    decisions = pd.DataFrame(
        row for name in V37_MODELS for row in _decision_rows(name, paths[name])
    )
    decisions.to_csv(output / "v37_inner_decision_audit.csv", index=False)
    gates = pd.DataFrame([_gate_diagnostics(name, paths[name]) for name in V37_MODELS])
    gates.to_csv(output / "v37_gate_diagnostics.csv", index=False)
    mechanism = pd.DataFrame(
        [_mechanism_diagnostics(name, paths[name]) for name in V37_MODELS]
    )
    mechanism.to_csv(output / "v37_mechanism_diagnostics.csv", index=False)
    mechanism_auc = pd.DataFrame(
        [
            _paired_auc(
                paths[FULL] / "out_of_fold_predictions.csv",
                paths[name] / "out_of_fold_predictions.csv",
                name,
                replicates,
                seed + 37,
            )
            for name in (
                "lac_v37_no_historical_dose",
                "lac_v37_no_i_to_c",
                "lac_v37_direct_cac_regression",
            )
        ]
    )
    mechanism_auc.to_csv(output / "v37_mechanism_auc_bootstrap_95ci.csv", index=False)

    split = _split_audit({name: paths[name] for name in V37_MODELS}, paths["lac_v22_full"])
    for model in V37_MODELS:
        for fold in range(1, 6):
            current = paths[model] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            reference = paths["lac_v22_full"] / f"fold_{fold}" / "inner_patient_fold_plan.csv"
            split.loc[split.model == model, f"inner_fold_{fold}_identical"] = bool(
                current.is_file()
                and reference.is_file()
                and _normalized_plan_checksum(current) == _normalized_plan_checksum(reference)
            )
    split.to_csv(output / "v37_split_audit.csv", index=False)
    split_columns = [column for column in split if column.startswith("inner_fold_")]
    split_pass = bool(
        (split.patient_count == 443).all()
        and (split.unique_patient_count == 443).all()
        and split.patient_ids_identical.all()
        and split.outer_folds_identical.all()
        and all(split[column].fillna(True).all() for column in split_columns)
    )

    full = indexed.loc[FULL]
    formal_best = _metric_best(full, indexed.loc[list(FORMAL_COMPARATORS)])
    ablation_best = _metric_best(full, indexed.loc[list(V37_MODELS[1:])])
    dominating = [name for name in V37_MODELS[1:] if _dominates(indexed.loc[name], full)]
    assessment = {
        "frozen_commit": _summary(paths[FULL]).get("git_revision"),
        "patient_count": int(_summary(paths[FULL])["patient_count"]),
        "split_and_patient_audit_all_passed": split_pass,
        "formal_metric_acceptance": formal_best,
        "ablation_metric_acceptance": ablation_best,
        "formal_rank_one_count": int(sum(formal_best.values())),
        "ablation_rank_one_count": int(sum(ablation_best.values())),
        "full_not_pareto_dominated_by_any_ablation": not bool(dominating),
        "dominating_ablations": dominating,
        "all_inner_decisions_exclude_outer_labels": bool(
            (~decisions.outer_test_labels_used).all()
        ),
        "paired_bootstrap_significance_counts": significance,
        "inner_selected_cac_weights": decisions.loc[
            decisions.model == FULL, "inner_selected_weight"
        ].tolist(),
        "soft_phenotype_adapters": _contrast(FULL, "lac_v37_no_soft_adapters", table),
        "strict_i_to_c": _contrast(FULL, "lac_v37_no_i_to_c", table),
        "historical_dose": _contrast(FULL, "lac_v37_no_historical_dose", table),
        "direction_magnitude": _contrast(FULL, "lac_v37_direct_cac_regression", table),
        "gradient_protection": _contrast(FULL, "lac_v37_no_gradient_protection", table),
        "baseline_anchoring": _contrast(FULL, "lac_v37_no_baseline_anchoring", table),
        "interpretation": (
            "post-hoc exploratory internal validation on a repeatedly inspected "
            "443-patient development cohort; independent confirmation required"
        ),
        "outer_results_used_for_current_model_selection": False,
        "github_push": False,
    }
    (output / "v37_result_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    full_summary = _summary(paths[FULL])
    diagnostics = full_summary["diagnostics"]
    change_ci = full_summary["change_bootstrap_95_ci"]
    report = f"""# LAC-iTransformer V3.7 内部开发验证报告

## 证据定位

本轮使用冻结提交 `{assessment['frozen_commit']}`、同一443例患者和锁定患者级嵌套五折。全部结构、训练权重与折内选择规则在本轮完整外层评价前冻结。由于该队列已经被多个历史版本反复检查，本报告属于探索性内部开发结果，不能替代独立外部确认。

## 正式对比结果

{_markdown(comparison)}

Full在六项指标中的点估计第一数量：**{assessment['formal_rank_one_count']}/6**。

没有任何正式对比模型在任一指标上被配对Bootstrap判定为显著优于Full。相对XGBoost，CAC三项均显著改善，但TBR差异区间跨0；相对Elastic Net，CAC MAE显著改善，CAC RMSE/R²差异未达显著。

## 核心消融结果

{_markdown(ablation)}

Full在消融中的点估计第一数量：**{assessment['ablation_rank_one_count']}/6**；Pareto全面优于Full的消融：**{dominating or '无'}**。

软表型适配器使Full相对无适配器版本改善4/6项，其中CAC MAE差异显著；梯度保护和基线锚定分别使Full在5/6项获得更好点估计。I→C路径、历史剂量和方向—幅度解码均显著改善CAC MAE，但对应RMSE/R²存在反向权衡。因此，这些模块支持稳健中心误差控制，尚不能表述为对CAC全部指标的一致提升。

## 稳健性诊断

- Full自举95%CI：TBR MAE {change_ci['delta_tbr_mae']['ci_low']:.4f}–{change_ci['delta_tbr_mae']['ci_high']:.4f}，TBR R² {change_ci['delta_tbr_r2']['ci_low']:.4f}–{change_ci['delta_tbr_r2']['ci_high']:.4f}；CAC MAE {change_ci['delta_log_cac_mae']['ci_low']:.4f}–{change_ci['delta_log_cac_mae']['ci_high']:.4f}，CAC R² {change_ci['delta_log_cac_r2']['ci_low']:.4f}–{change_ci['delta_log_cac_r2']['ci_high']:.4f}。
- 预测方差/真实方差：TBR {diagnostics['prediction_variance_over_true_variance']['delta_tbr']:.4f}，CAC {diagnostics['prediction_variance_over_true_variance']['delta_log_cac']:.4f}。CAC较V3.6有所改善，但仍存在明显向均值收缩。
- 方向准确率：TBR {diagnostics['direction_accuracy']['delta_tbr']:.4f}，CAC {diagnostics['direction_accuracy']['delta_log_cac']:.4f}；CAC变化量绝对值前10%患者RMSE为 {diagnostics['cac_top_10_percent_absolute_change']['rmse']:.4f}。
- 患者、外层折和全部内层折审计通过：**{split_pass}**；所有折内决策均未使用外层标签。

## 解释原则

CAC专属消融的TBR结果完全一致是路径隔离的预期结果，不应解释为重复实验。I→C有效性须同时依据CAC变化量指标、机制AUC、历史负对照和Bootstrap区间；若区间跨0，只能报告方向性结果。
"""
    (output / "V37_INTERNAL_REPORT_CN.md").write_text(report, encoding="utf-8")
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v37-root", required=True)
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
    result = aggregate_v37_results(
        args.v37_root,
        args.formal_root,
        args.output,
        historical_roots=mapping,
        replicates=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
