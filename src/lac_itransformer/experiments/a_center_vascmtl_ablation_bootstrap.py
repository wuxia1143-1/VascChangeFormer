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
    "vascmtl_no_baseline_anchoring",
    "vascmtl_no_cac_calibration",
    "vascmtl_tbr_single",
    "vascmtl_cac_single",
)
DISPLAY_NAMES = {
    "vascmtl": "VascMTL",
    "vascmtl_no_baseline_anchoring": "w/o baseline anchoring",
    "vascmtl_no_cac_calibration": "w/o CAC calibration",
    "vascmtl_tbr_single": "VascMTL-TBR-single",
    "vascmtl_cac_single": "VascMTL-CAC-single",
}
AVAILABLE_TASKS = {
    "vascmtl": ("TBR", "CAC"),
    "vascmtl_no_baseline_anchoring": ("TBR", "CAC"),
    "vascmtl_no_cac_calibration": ("TBR", "CAC"),
    "vascmtl_tbr_single": ("TBR",),
    "vascmtl_cac_single": ("CAC",),
}
COMPARISON_TASKS = {
    "vascmtl_no_baseline_anchoring": ("TBR", "CAC"),
    "vascmtl_no_cac_calibration": ("TBR", "CAC"),
    "vascmtl_tbr_single": ("TBR",),
    "vascmtl_cac_single": ("CAC",),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ablation_oof(
    experiment_root: str | Path,
    expected_patient_count: int = 443,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {
        model: Path(experiment_root) / model / "out_of_fold_predictions.csv"
        for model in MODEL_ORDER
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


def _available_metrics(model_metrics: pd.DataFrame) -> pd.DataFrame:
    result = model_metrics[
        model_metrics.apply(
            lambda row: row["task"] in AVAILABLE_TASKS[row["model"]], axis=1
        )
    ].copy()
    result["display_name"] = result["model"].map(DISPLAY_NAMES)
    return result


def _full_differences(pairwise: pd.DataFrame) -> pd.DataFrame:
    result = pairwise[pairwise["left_model"] == MODEL_ORDER[0]].copy()
    result = result[
        result.apply(
            lambda row: row["task"] in COMPARISON_TASKS[row["right_model"]],
            axis=1,
        )
    ].copy()
    result["left_display_name"] = result["left_model"].map(DISPLAY_NAMES)
    result["right_display_name"] = result["right_model"].map(DISPLAY_NAMES)
    result["comparison"] = "VascMTL vs " + result["right_display_name"]
    result["favored_model"] = result.apply(
        lambda row: (
            row["left_display_name"]
            if row["ci_high"] < 0.0
            else row["right_display_name"]
            if row["ci_low"] > 0.0
            else "inconclusive"
        ),
        axis=1,
    )
    return result


def _ci_text(
    row: pd.Series, estimate: str = "estimate", digits: int = 4
) -> str:
    return (
        f"{row[estimate]:.{digits}f} "
        f"({row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f})"
    )


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
    paths: dict[str, Path],
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    available = _available_metrics(model_metrics)
    full_differences = _full_differences(pairwise)
    all_pairs = pairwise.copy()
    all_pairs["left_display_name"] = all_pairs["left_model"].map(DISPLAY_NAMES)
    all_pairs["right_display_name"] = all_pairs["right_model"].map(DISPLAY_NAMES)

    metrics_path = output / "vascmtl_ablation_metrics_95ci.csv"
    differences_path = output / "vascmtl_vs_ablations_95ci.csv"
    all_pairs_path = output / "vascmtl_ablation_all_pairwise_95ci.csv"
    replicates_path = output / "vascmtl_ablation_metric_replicates.csv"
    audit_path = output / "vascmtl_ablation_audit.json"
    report_path = output / "A_CENTER_LEAD3_VASCMTL_ABLATION_REPORT_CN.md"
    available.to_csv(metrics_path, index=False)
    full_differences.to_csv(differences_path, index=False)
    all_pairs.to_csv(all_pairs_path, index=False)
    replicate_metrics.to_csv(replicates_path, index=False)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_rows = []
    for model in MODEL_ORDER:
        row: dict[str, Any] = {
            "Model": DISPLAY_NAMES[model],
            "n": audit["patient_count"],
        }
        for task in TASKS:
            for metric in METRICS:
                key = f"{task} {metric} (95% CI)"
                if task not in AVAILABLE_TASKS[model]:
                    row[key] = "—"
                    continue
                record = available[
                    (available["model"] == model)
                    & (available["task"] == task)
                    & (available["metric"] == metric)
                ].iloc[0]
                row[key] = _ci_text(record)
        metric_rows.append(row)

    difference_rows = []
    for ablation in MODEL_ORDER[1:]:
        for task in COMPARISON_TASKS[ablation]:
            group = full_differences[
                (full_differences["right_model"] == ablation)
                & (full_differences["task"] == task)
            ]
            row = {
                "Comparison": f"VascMTL vs {DISPLAY_NAMES[ablation]}",
                "Task": task,
            }
            for metric in METRICS:
                record = group[group["metric"] == metric].iloc[0]
                row[f"Δ{metric} (95% CI)"] = _ci_text(
                    record, estimate="difference_estimate", digits=5
                )
            difference_rows.append(row)

    report = "\n".join(
        [
            "# A中心提前3个月预测：VascMTL核心消融实验",
            "",
            "## 实验口径",
            "",
            "- VascMTL为原`joint_hps_mtl`结构：单一共享iTransformer编码器、独立TBR/CAC任务头、同时优化双任务损失。",
            "- 使用最新提前3个月数据切片及同一锁定A中心患者级内部五折。",
            f"- 合并OOF患者数为{audit['patient_count']}；patient-level paired bootstrap {audit['bootstrap_replicates']}次（seed={audit['bootstrap_seed']}）。",
            "- baseline与calibration消融只关闭指定模块；两个single-task保持VascMTL骨架，仅删除另一任务损失。",
            "- 每次bootstrap使用同一批患者索引同步重采样全部模型、任务和指标。",
            "- 差值定义为VascMTL减消融模型；MAE/RMSE差值为负表示VascMTL误差更低。",
            "",
            "## 性能汇总",
            "",
            _markdown_table(pd.DataFrame(metric_rows)),
            "",
            "## VascMTL与消融模型的患者级配对差值",
            "",
            _markdown_table(pd.DataFrame(difference_rows)),
            "",
            "## 审计说明",
            "",
            f"- 同步重采样索引矩阵SHA256：`{audit['bootstrap_index_matrix_sha256']}`。",
            "- 95% CI为未作多重比较校正的percentile区间。",
            "- CI仅反映锁定OOF预测条件下的患者抽样不确定性，不包含重新训练随机性。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    outputs = {
        "metrics": metrics_path,
        "vascmtl_vs_ablations": differences_path,
        "all_pairwise": all_pairs_path,
        "bootstrap_metric_replicates": replicates_path,
        "audit": audit_path,
        "report": report_path,
    }
    manifest = {
        **audit,
        "model_order": list(MODEL_ORDER),
        "display_names": DISPLAY_NAMES,
        "available_tasks": AVAILABLE_TASKS,
        "source_oof": {
            model: {"path": str(path), "sha256": _sha256(path)}
            for model, path in paths.items()
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = output / "vascmtl_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-patient-count", type=int, default=443)
    args = parser.parse_args()
    frames, paths = load_ablation_oof(
        args.experiment_root, expected_patient_count=args.expected_patient_count
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
