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
    "lac_v60_a_no_i2c_tail",
    "lac_v60_shared_only_mtl",
    "lac_v60_no_baseline_anchoring",
    "lac_v60_no_oof_calibration",
)
DISPLAY_NAMES = {
    "lac_v60_a_no_i2c_tail": "Full LAC",
    "lac_v60_shared_only_mtl": "V6 shared-only MTL control",
    "lac_v60_no_baseline_anchoring": "w/o baseline anchoring",
    "lac_v60_no_oof_calibration": "w/o CAC calibration",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_core_ablation_oof(
    full_oof: str | Path,
    ablation_root: str | Path,
    expected_patient_count: int = 443,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {
        MODEL_ORDER[0]: Path(full_oof),
        **{
            model: Path(ablation_root) / model / "out_of_fold_predictions.csv"
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


def _wide_metrics(model_metrics: pd.DataFrame, patient_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        row: dict[str, Any] = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "n": int(patient_count),
        }
        for task in TASKS:
            for metric in METRICS:
                record = model_metrics[
                    (model_metrics["model"] == model)
                    & (model_metrics["task"] == task)
                    & (model_metrics["metric"] == metric)
                ].iloc[0]
                prefix = f"{task.lower()}_{metric.lower()}"
                row[prefix] = float(record["estimate"])
                row[f"{prefix}_ci_low"] = float(record["ci_low"])
                row[f"{prefix}_ci_high"] = float(record["ci_high"])
        rows.append(row)
    return pd.DataFrame(rows)


def _wide_differences(full_comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ablation in MODEL_ORDER[1:]:
        row: dict[str, Any] = {
            "left_model": MODEL_ORDER[0],
            "right_model": ablation,
            "comparison": f"Full LAC vs {DISPLAY_NAMES[ablation]}",
            "difference_definition": "Full LAC minus ablation",
        }
        for task in TASKS:
            for metric in METRICS:
                record = full_comparisons[
                    (full_comparisons["right_model"] == ablation)
                    & (full_comparisons["task"] == task)
                    & (full_comparisons["metric"] == metric)
                ].iloc[0]
                prefix = f"delta_{task.lower()}_{metric.lower()}"
                row[prefix] = float(record["difference_estimate"])
                row[f"{prefix}_ci_low"] = float(record["ci_low"])
                row[f"{prefix}_ci_high"] = float(record["ci_high"])
                row[f"{prefix}_ci_excludes_zero"] = bool(
                    record["ci_excludes_zero"]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _ci_text(estimate: float, ci_low: float, ci_high: float) -> str:
    return f"{estimate:.4f} ({ci_low:.4f}, {ci_high:.4f})"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_outputs(
    model_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    replicate_metrics: pd.DataFrame,
    audit: dict[str, Any],
    source_paths: dict[str, Path],
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    named_metrics = model_metrics.copy()
    named_metrics["display_name"] = named_metrics["model"].map(DISPLAY_NAMES)
    named_pairwise = pairwise.copy()
    named_pairwise["left_display_name"] = named_pairwise["left_model"].map(
        DISPLAY_NAMES
    )
    named_pairwise["right_display_name"] = named_pairwise["right_model"].map(
        DISPLAY_NAMES
    )
    named_pairwise["comparison"] = (
        named_pairwise["left_display_name"]
        + " vs "
        + named_pairwise["right_display_name"]
    )
    named_pairwise["favored_model"] = named_pairwise["favored_model"].map(
        lambda value: DISPLAY_NAMES.get(value, value)
    )
    full_comparisons = named_pairwise[
        named_pairwise["left_model"] == MODEL_ORDER[0]
    ].copy()

    wide_metrics = _wide_metrics(named_metrics, int(audit["patient_count"]))
    wide_differences = _wide_differences(full_comparisons)
    metrics_path = output / "core_ablation_metrics_95ci.csv"
    differences_path = output / "core_ablation_full_vs_ablations_95ci.csv"
    all_pairs_path = output / "core_ablation_all_pairwise_95ci.csv"
    replicates_path = output / "core_ablation_metric_replicates.csv"
    audit_path = output / "core_ablation_audit.json"
    report_path = output / "A_CENTER_LEAD3_CORE_ABLATION_REPORT_CN.md"
    wide_metrics.to_csv(metrics_path, index=False)
    wide_differences.to_csv(differences_path, index=False)
    named_pairwise.to_csv(all_pairs_path, index=False)
    replicate_metrics.to_csv(replicates_path, index=False)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_table_rows = []
    for row in wide_metrics.to_dict(orient="records"):
        metric_table_rows.append(
            {
                "Model": row["display_name"],
                "n": row["n"],
                "TBR MAE (95% CI)": _ci_text(
                    row["tbr_mae"], row["tbr_mae_ci_low"], row["tbr_mae_ci_high"]
                ),
                "TBR RMSE (95% CI)": _ci_text(
                    row["tbr_rmse"],
                    row["tbr_rmse_ci_low"],
                    row["tbr_rmse_ci_high"],
                ),
                "CAC MAE (95% CI)": _ci_text(
                    row["cac_mae"], row["cac_mae_ci_low"], row["cac_mae_ci_high"]
                ),
                "CAC RMSE (95% CI)": _ci_text(
                    row["cac_rmse"],
                    row["cac_rmse_ci_low"],
                    row["cac_rmse_ci_high"],
                ),
            }
        )

    difference_table_rows = []
    for row in wide_differences.to_dict(orient="records"):
        difference_table_rows.append(
            {
                "Comparison": row["comparison"],
                "ΔTBR MAE (95% CI)": _ci_text(
                    row["delta_tbr_mae"],
                    row["delta_tbr_mae_ci_low"],
                    row["delta_tbr_mae_ci_high"],
                ),
                "ΔTBR RMSE (95% CI)": _ci_text(
                    row["delta_tbr_rmse"],
                    row["delta_tbr_rmse_ci_low"],
                    row["delta_tbr_rmse_ci_high"],
                ),
                "ΔCAC MAE (95% CI)": _ci_text(
                    row["delta_cac_mae"],
                    row["delta_cac_mae_ci_low"],
                    row["delta_cac_mae_ci_high"],
                ),
                "ΔCAC RMSE (95% CI)": _ci_text(
                    row["delta_cac_rmse"],
                    row["delta_cac_rmse_ci_low"],
                    row["delta_cac_rmse_ci_high"],
                ),
            }
        )

    report = "\n".join(
        [
            "# A中心提前3个月预测：核心消融实验",
            "",
            "## 实验口径",
            "",
            "- 使用最新提前3个月数据切片：每名患者的纵向输入仅保留至真实随访终点至少前3个月。",
            "- 四个模型使用同一A中心患者级内部五折划分与合并OOF评估。",
            f"- 基于{audit['patient_count']}例患者进行{audit['bootstrap_replicates']}次patient-level paired bootstrap（seed={audit['bootstrap_seed']}）。",
            "- 每次抽取同一批患者索引，并同时用于所有模型、任务和指标。95% CI为未校正percentile区间。",
            "- TBR对应ΔTBR；CAC对应Δlog(1+CAC)。",
            "- 差值定义为Full LAC − ablation；MAE/RMSE差值为负表示Full LAC误差更低。",
            "- `lac_v60_shared_only_mtl`是V6框架内的shared-only MTL消融对照，不等同于独立baseline中的标准`itransformer_mtl`。",
            "",
            "## 各模型性能",
            "",
            _markdown_table(pd.DataFrame(metric_table_rows)),
            "",
            "## Full LAC与各消融模型的患者级配对差值",
            "",
            _markdown_table(pd.DataFrame(difference_table_rows)),
            "",
            "## 统计说明与审计",
            "",
            f"- 同步重采样索引矩阵SHA256：`{audit['bootstrap_index_matrix_sha256']}`。",
            "- 患者ID、外折归属、真实目标及数值有限性检查均通过。",
            "- CI反映锁定OOF预测条件下的患者抽样不确定性，不包含重新训练带来的随机性。",
            "- 未进行多重比较校正，结果应按预设核心消融的探索性证据解释。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    output_files = {
        "metrics": metrics_path,
        "full_vs_ablations": differences_path,
        "all_pairwise": all_pairs_path,
        "bootstrap_metric_replicates": replicates_path,
        "audit": audit_path,
        "report": report_path,
    }
    manifest = {
        **audit,
        "model_order": list(MODEL_ORDER),
        "display_names": DISPLAY_NAMES,
        "source_oof": {
            model: {"path": str(path), "sha256": _sha256(path)}
            for model, path in source_paths.items()
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in output_files.items()
        },
    }
    manifest_path = output / "core_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-oof", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-patient-count", type=int, default=443)
    args = parser.parse_args()

    frames, paths = load_core_ablation_oof(
        args.full_oof,
        args.ablation_root,
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
