from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lac_itransformer.experiments.a_center_paired_bootstrap import (
    METRICS,
    TASKS,
    _load_oof,
    _validate_alignment,
    simultaneous_paired_bootstrap,
)


MODEL_ORDER = (
    "vascmtl",
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
)
DISPLAY_NAMES = {
    "vascmtl": "VascMTL",
    "persistence": "Persistence",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "apn_dr": "APN-DR",
    "itransformer_mtl": "iTransformer-MTL",
    "first_icu_mtl": "FIRST-ICU-MTL",
    "learning_to_route": "Learning to Route",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vascmtl_baseline_oof(
    vascmtl_oof: str | Path,
    baseline_root: str | Path,
    expected_patient_count: int = 443,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {
        MODEL_ORDER[0]: Path(vascmtl_oof),
        **{
            model: Path(baseline_root) / model / "out_of_fold_predictions.csv"
            for model in MODEL_ORDER[1:]
        },
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing OOF files: {missing}")
    frames = {model: _load_oof(path) for model, path in paths.items()}
    reference = frames[MODEL_ORDER[0]]
    if len(reference) != int(expected_patient_count):
        raise ValueError(
            f"Expected {expected_patient_count} patients, found {len(reference)}"
        )
    for model in MODEL_ORDER[1:]:
        _validate_alignment(reference, frames[model])
    return frames, paths


def _label_results(
    model_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_metrics = model_metrics.copy()
    pairwise = pairwise.copy()
    model_metrics["display_name"] = model_metrics["model"].map(DISPLAY_NAMES)
    pairwise["left_display_name"] = pairwise["left_model"].map(DISPLAY_NAMES)
    pairwise["right_display_name"] = pairwise["right_model"].map(DISPLAY_NAMES)
    pairwise["comparison"] = (
        pairwise["left_display_name"] + " vs " + pairwise["right_display_name"]
    )
    pairwise["favored_model"] = pairwise.apply(
        lambda row: (
            row["left_display_name"]
            if row["ci_high"] < 0.0
            else row["right_display_name"]
            if row["ci_low"] > 0.0
            else "inconclusive"
        ),
        axis=1,
    )
    return model_metrics, pairwise


def _difference_ci_text(row: pd.Series) -> str:
    return (
        f"{row['difference_estimate']:.4f} "
        f"({row['ci_low']:.4f}, {row['ci_high']:.4f})"
    )


def _metric_ci_text(row: pd.Series) -> str:
    return f"{row['estimate']:.4f} ({row['ci_low']:.4f}, {row['ci_high']:.4f})"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_comparison_table(primary_comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for baseline in MODEL_ORDER[1:]:
        subset = primary_comparisons[
            primary_comparisons["right_model"] == baseline
        ]
        row = {"Comparison": f"VascMTL vs {DISPLAY_NAMES[baseline]}"}
        for task in TASKS:
            for metric in METRICS:
                record = subset[
                    (subset["task"] == task) & (subset["metric"] == metric)
                ].iloc[0]
                row[f"Δ{task} {metric} (95% CI)"] = _difference_ci_text(record)
        rows.append(row)
    return pd.DataFrame(rows)


def build_metric_table(model_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        subset = model_metrics[model_metrics["model"] == model]
        row = {"Model": DISPLAY_NAMES[model]}
        for task in TASKS:
            for metric in METRICS:
                record = subset[
                    (subset["task"] == task) & (subset["metric"] == metric)
                ].iloc[0]
                row[f"{task} {metric} (95% CI)"] = _metric_ci_text(record)
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    model_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    replicate_metrics: pd.DataFrame,
    audit: dict[str, Any],
    source_paths: dict[str, Path],
    output: str | Path,
    prediction_lead_months: float = 3.0,
    followup_min_months: float | None = None,
    followup_max_months: float | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_metrics, pairwise = _label_results(model_metrics, pairwise)
    primary_comparisons = pairwise[
        pairwise["left_model"] == MODEL_ORDER[0]
    ].copy()
    comparison_table = build_comparison_table(primary_comparisons)
    metric_table = build_metric_table(model_metrics)

    metrics_path = output / "vascmtl_baseline_model_metrics_95ci.csv"
    differences_path = output / "vascmtl_vs_baselines_95ci.csv"
    table_path = output / "vascmtl_vs_baselines_table_ready.csv"
    all_pairs_path = output / "vascmtl_baseline_all_pairwise_95ci.csv"
    replicates_path = output / "vascmtl_baseline_metric_replicates.csv"
    audit_path = output / "vascmtl_baseline_audit.json"
    lead_label = f"{prediction_lead_months:g}"
    report_path = output / (
        f"A_CENTER_LEAD{lead_label}_VASCMTL_VS_BASELINES_REPORT_CN.md"
    )
    model_metrics.to_csv(metrics_path, index=False)
    primary_comparisons.to_csv(differences_path, index=False)
    comparison_table.to_csv(table_path, index=False)
    pairwise.to_csv(all_pairs_path, index=False)
    replicate_metrics.to_csv(replicates_path, index=False)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = "\n".join(
        [
            f"# A中心提前{lead_label}个月预测：VascMTL与基线的配对Bootstrap",
            "",
            "## 统计口径",
            "",
            f"- 使用最新VascMTL合并OOF预测，共{audit['patient_count']}例患者。",
            (
                f"- 纳入条件：真实随访间隔{followup_min_months:g}–"
                f"{followup_max_months:g}个月（含边界）；输入仅保留真实随访终点"
                f"前至少{lead_label}个月可获得的实验室与治疗记录。"
                if followup_min_months is not None
                and followup_max_months is not None
                else f"- 输入仅保留真实随访终点前至少{lead_label}个月可获得的实验室与治疗记录。"
            ),
            f"- 患者级同步配对Bootstrap {audit['bootstrap_replicates']}次（seed={audit['bootstrap_seed']}）。",
            "- 所有模型使用完全相同的患者重采样索引；患者ID、外折及真实目标均已对齐。",
            "- 差值定义为VascMTL减Baseline；MAE/RMSE差值为负表示VascMTL误差更低。",
            "- 95% CI为未作多重比较校正的percentile区间。",
            "",
            "## 待填表格",
            "",
            _markdown_table(comparison_table),
            "",
            "## 各模型指标",
            "",
            _markdown_table(metric_table),
            "",
            "## 审计",
            "",
            f"- 同步重采样索引矩阵SHA256：`{audit['bootstrap_index_matrix_sha256']}`。",
            "- CI仅反映锁定OOF预测条件下的患者抽样不确定性，不包含重新训练随机性。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    outputs = {
        "model_metrics": metrics_path,
        "vascmtl_vs_baselines": differences_path,
        "table_ready": table_path,
        "all_pairwise": all_pairs_path,
        "bootstrap_metric_replicates": replicates_path,
        "audit": audit_path,
        "report": report_path,
    }
    manifest = {
        **audit,
        "prediction_lead_months": float(prediction_lead_months),
        "followup_min_months_inclusive": followup_min_months,
        "followup_max_months_inclusive": followup_max_months,
        "model_order": list(MODEL_ORDER),
        "display_names": DISPLAY_NAMES,
        "source_oof": {
            model: {"path": str(path), "sha256": _sha256(path)}
            for model, path in source_paths.items()
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = output / "vascmtl_baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vascmtl-oof", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-patient-count", type=int, default=443)
    parser.add_argument("--prediction-lead-months", type=float, default=3.0)
    parser.add_argument("--followup-min-months", type=float)
    parser.add_argument("--followup-max-months", type=float)
    args = parser.parse_args()
    frames, paths = load_vascmtl_baseline_oof(
        args.vascmtl_oof,
        args.baseline_root,
        expected_patient_count=args.expected_patient_count,
    )
    model_metrics, pairwise, replicate_metrics, audit = simultaneous_paired_bootstrap(
        frames,
        model_order=MODEL_ORDER,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    manifest = write_outputs(
        model_metrics,
        pairwise,
        replicate_metrics,
        audit,
        paths,
        args.output,
        prediction_lead_months=args.prediction_lead_months,
        followup_min_months=args.followup_min_months,
        followup_max_months=args.followup_max_months,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "patient_count": manifest["patient_count"],
                "bootstrap_replicates": manifest["bootstrap_replicates"],
                "bootstrap_seed": manifest["bootstrap_seed"],
                "bootstrap_index_matrix_sha256": manifest[
                    "bootstrap_index_matrix_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
