from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .v28_reporting import (
    FORMAL_COMPARATORS,
    LOWER_IS_BETTER,
    PRIMARY_METRICS,
    _dominates,
    _paired_difference,
    _result_row,
)
from ..training.v40_final_external import V40_FROZEN_CANDIDATE
from ..training.v40_final_nested import V40_FINAL_MODELS


STRUCTURAL_ABLATIONS = (
    "lac_v40_final_no_i_to_c",
    "lac_v40_final_no_history",
    "lac_v40_final_no_treatment",
    "lac_v40_final_no_risk_gate",
    "lac_v40_final_no_soft_adapters",
    "lac_v40_final_no_baseline_anchoring",
)


FINAL_POLICY_SELECTION_PROVENANCE = {
    # Every outer fold remained untouched by its own training, epoch selection,
    # and decision-layer fitting.  However, the final policy exported for the
    # external cohort was chosen after comparing aggregate 443-patient outer-OOF
    # results.  Keep these two scopes explicit in every generated assessment.
    "internal_outer_oof_used_for_final_policy_selection": True,
    "outer_test_labels_used_for_fold_training_or_decision_fit": False,
    "external_labels_used_for_selection": False,
}


def _summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _v40_paths(root: Path) -> dict[str, Path]:
    paths = {
        "lac_v40_final_full": root / "full" / "lac_v40_final_full",
        "lac_v40_final_no_calibration": root / "ablations" / "lac_v40_final_no_calibration",
        "lac_v40_final_no_tail": root / "ablations" / "lac_v40_final_no_tail",
        "lac_v40_final_central_only": root / "ablations" / "lac_v40_final_central_only",
    }
    paths.update({name: root / "selected_policy" / name for name in STRUCTURAL_ABLATIONS})
    return paths


def _markdown(frame: pd.DataFrame) -> str:
    columns = ["model", *PRIMARY_METRICS]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                [str(row[0]), *[f"{float(value):.6f}" for value in row[1:]]]
            )
            + " |"
        )
    return "\n".join(lines)


def _contrast(candidate: pd.Series, ablation: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate": str(candidate.name),
        "reference": str(ablation.name),
    }
    improved = 0
    for metric in PRIMARY_METRICS:
        difference = float(candidate[metric] - ablation[metric])
        record[f"{metric}_candidate_minus_reference"] = difference
        candidate_better = (
            difference < 0 if metric in LOWER_IS_BETTER else difference > 0
        )
        record[f"{metric}_candidate_better"] = bool(candidate_better)
        improved += int(candidate_better)
    record["candidate_better_metric_count"] = improved
    return record


def _candidate_rank(table: pd.DataFrame, candidate: str) -> list[dict[str, Any]]:
    indexed = table.set_index("model")
    rows = []
    for metric in PRIMARY_METRICS:
        ascending = metric in LOWER_IS_BETTER
        ranks = indexed[metric].rank(ascending=ascending, method="min")
        rows.append(
            {
                "metric": metric,
                "candidate": candidate,
                "rank": int(ranks.loc[candidate]),
                "model_count": len(indexed),
                "candidate_value": float(indexed.loc[candidate, metric]),
                "best_value": float(
                    indexed[metric].min() if ascending else indexed[metric].max()
                ),
            }
        )
    return rows


def aggregate_v40_final_results(
    internal_root: str | Path,
    formal_root: str | Path,
    v37_full: str | Path,
    output_dir: str | Path,
    external_root: str | Path | None = None,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    internal = Path(internal_root)
    formal = Path(formal_root)
    v37 = Path(v37_full)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = _v40_paths(internal)
    missing = [str(path / "summary.json") for path in paths.values() if not (path / "summary.json").is_file()]
    if missing:
        raise FileNotFoundError("Missing V4.0 results:\n" + "\n".join(missing))
    if not (v37 / "summary.json").is_file():
        raise FileNotFoundError(v37 / "summary.json")

    v40_table = pd.DataFrame(
        [
            _result_row(
                name,
                path,
                "v40_selected" if name == V40_FROZEN_CANDIDATE else "v40_ablation",
            )
            for name, path in paths.items()
        ]
    )
    v40_table.to_csv(output / "v40_internal_ablation_results.csv", index=False)

    comparison_paths = {name: formal / name for name in FORMAL_COMPARATORS}
    comparison_paths["lac_v37_full"] = v37
    missing_comparison = [
        str(path / "summary.json")
        for path in comparison_paths.values()
        if not (path / "summary.json").is_file()
    ]
    if missing_comparison:
        raise FileNotFoundError(
            "Missing internal comparators:\n" + "\n".join(missing_comparison)
        )
    comparison = pd.DataFrame(
        [
            _result_row(V40_FROZEN_CANDIDATE, paths[V40_FROZEN_CANDIDATE], "v40_selected")
        ]
        + [
            _result_row(name, path, "internal_comparator")
            for name, path in comparison_paths.items()
        ]
    )
    comparison.to_csv(output / "v40_internal_comparison_results.csv", index=False)
    pd.DataFrame(_candidate_rank(comparison, V40_FROZEN_CANDIDATE)).to_csv(
        output / "v40_internal_candidate_ranks.csv", index=False
    )

    paired_rows = []
    paired_paths = comparison_paths | {
        name: path for name, path in paths.items() if name != V40_FROZEN_CANDIDATE
    }
    for reference_index, (reference, reference_path) in enumerate(paired_paths.items()):
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            record = _paired_difference(
                paths[V40_FROZEN_CANDIDATE] / "out_of_fold_predictions.csv",
                reference_path / "out_of_fold_predictions.csv",
                metric,
                replicates,
                seed + reference_index * 31 + metric_index,
            )
            record["difference_definition"] = "V4_selected_minus_reference"
            paired_rows.append(
                {"candidate": V40_FROZEN_CANDIDATE, "reference": reference} | record
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "v40_internal_paired_bootstrap_95ci.csv", index=False)

    indexed = v40_table.set_index("model")
    candidate = indexed.loc[V40_FROZEN_CANDIDATE]
    contrasts = pd.DataFrame(
        [
            _contrast(candidate, indexed.loc[name])
            for name in V40_FINAL_MODELS
            if name != V40_FROZEN_CANDIDATE
        ]
    )
    contrasts.to_csv(output / "v40_internal_module_contrasts.csv", index=False)
    dominators = [
        name
        for name in indexed.index
        if name != V40_FROZEN_CANDIDATE and _dominates(indexed.loc[name], candidate)
    ]

    tail_full = indexed.loc["lac_v40_final_full"]
    central = indexed.loc["lac_v40_final_central_only"]
    tail_mae_degradation = float(
        (tail_full["delta_log_cac_mae"] - candidate["delta_log_cac_mae"])
        / candidate["delta_log_cac_mae"]
    )
    calibration_mae_improvement = float(
        (central["delta_log_cac_mae"] - candidate["delta_log_cac_mae"])
        / central["delta_log_cac_mae"]
    )
    module_assessment = {
        "frozen_candidate": V40_FROZEN_CANDIDATE,
        "selection_reason": (
            "Tail correction failed the preregistered CAC MAE non-inferiority "
            "constraint; calibration-only policy was selected before external use."
        ),
        "tail_correction_cac_mae_relative_degradation": tail_mae_degradation,
        "tail_correction_passes_mae_noninferiority_0_5_percent": bool(
            tail_mae_degradation <= 0.005
        ),
        "calibration_cac_mae_relative_improvement": calibration_mae_improvement,
        "calibration_improves_cac_mae_rmse_r2": bool(
            candidate["delta_log_cac_mae"] < central["delta_log_cac_mae"]
            and candidate["delta_log_cac_rmse"] < central["delta_log_cac_rmse"]
            and candidate["delta_log_cac_r2"] > central["delta_log_cac_r2"]
        ),
        "candidate_pareto_dominators": dominators,
        "candidate_not_pareto_dominated": not bool(dominators),
        **FINAL_POLICY_SELECTION_PROVENANCE,
        "internal_evidence_interpretation": (
            "development and final-policy selection on the repeatedly inspected "
            "443-patient cohort; not an independent confirmation of the selected policy"
        ),
    }

    external_table = None
    if external_root is not None:
        external = Path(external_root)
        external_table = pd.read_csv(external / "all_external_metrics.csv")
        external_table.to_csv(output / "v40_external_all_results.csv", index=False)
        external_table.loc[
            external_table["model"].isin([*FORMAL_COMPARATORS[:7], "lac_v37_full", V40_FROZEN_CANDIDATE])
        ].to_csv(output / "v40_external_comparison_results.csv", index=False)
        external_table.loc[external_table["model"].isin(V40_FINAL_MODELS)].to_csv(
            output / "v40_external_ablation_results.csv", index=False
        )
        shutil_source = external / "paired_bootstrap_v40_candidate_vs_all.csv"
        if shutil_source.is_file():
            pd.read_csv(shutil_source).to_csv(
                output / "v40_external_paired_bootstrap_95ci.csv", index=False
            )
        external_indexed = external_table.set_index("model")
        external_candidate = external_indexed.loc[V40_FROZEN_CANDIDATE]
        external_assessment = {
            "candidate_metrics": {
                metric: float(external_candidate[metric]) for metric in PRIMARY_METRICS
            },
            "candidate_vs_v37": _contrast(
                external_candidate, external_indexed.loc["lac_v37_full"]
            ),
            "candidate_comparison_ranks": _candidate_rank(
                external_table.loc[
                    external_table["model"].isin(
                        [*FORMAL_COMPARATORS[:7], "lac_v37_full", V40_FROZEN_CANDIDATE]
                    )
                ],
                V40_FROZEN_CANDIDATE,
            ),
        }
        module_assessment["external_assessment"] = external_assessment

    (output / "v40_final_assessment.json").write_text(
        json.dumps(module_assessment, indent=2), encoding="utf-8"
    )
    candidate_ci = _summary(paths[V40_FROZEN_CANDIDATE])["change_bootstrap_95_ci"]
    report = f"""# LAC-iTransformer V4.0-final 内部开发验证与独立外部验证报告

## 冻结候选

预注册 Full 的尾部修正使内部 CAC MAE 相对去尾部版本恶化 {tail_mae_degradation:.2%}，超过 0.5% 非劣界，因此在读取外部标签前锁定 `{V40_FROZEN_CANDIDATE}`（保留稳健校准、关闭尾部修正）作为 V4.0-final。

协议审计：每个外层折的预处理、轮数和决策层仅使用该折训练池的内层 OOF，未使用该折测试标签；但最终从 Full 切换到 `no_tail` 使用了 443 例完整外层 OOF 指标。因此，443 例结果应解释为内部开发与最终策略选择证据，不是对所选 `no_tail` 策略的完全独立内部确认。该选择未使用任何外部标签。

## 内部443例开发/选择性验证对比

{_markdown(comparison)}

候选内部95%CI：TBR MAE {candidate_ci['delta_tbr_mae']['ci_low']:.4f}–{candidate_ci['delta_tbr_mae']['ci_high']:.4f}；CAC MAE {candidate_ci['delta_log_cac_mae']['ci_low']:.4f}–{candidate_ci['delta_log_cac_mae']['ci_high']:.4f}；CAC R² {candidate_ci['delta_log_cac_r2']['ci_low']:.4f}–{candidate_ci['delta_log_cac_r2']['ci_high']:.4f}。

## 内部消融

{_markdown(v40_table)}

校准使 CAC MAE 相对中央预测改善 {calibration_mae_improvement:.2%}，且 MAE、RMSE、R²三项点估计同向改善。尾部修正未通过预注册门槛，因此不得宣称所有最初设计模块都不可缺少。候选的 Pareto 支配者：{dominators or '无'}。
"""
    if external_table is not None:
        external_comparison = external_table.loc[
            external_table["model"].isin(
                [*FORMAL_COMPARATORS[:7], "lac_v37_full", V40_FROZEN_CANDIDATE]
            )
        ]
        external_ablation = external_table.loc[
            external_table["model"].isin(V40_FINAL_MODELS)
        ]
        report += f"""

## 山东外部710例一次性验证

{_markdown(external_comparison)}

## 外部消融

{_markdown(external_ablation)}

外部结果只用于确认冻结模型的泛化，不参与任何结构、权重、阈值、轮数或校准选择。
"""
    (output / "V40_FINAL_REPORT_CN.md").write_text(report, encoding="utf-8")
    return module_assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--v37-full", required=True)
    parser.add_argument("--external-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = aggregate_v40_final_results(
        args.internal_root,
        args.formal_root,
        args.v37_full,
        args.output,
        external_root=args.external_root,
        replicates=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
