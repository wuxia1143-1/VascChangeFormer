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
    "lac_v60_joint_hps_mtl",
    "lac_v60_joint_shared_adapter_mtl",
    "lac_v60_single_task_tbr",
    "lac_v60_single_task_cac",
)
DISPLAY_NAMES = {
    "lac_v60_joint_hps_mtl": "joint_hps_mtl",
    "lac_v60_joint_shared_adapter_mtl": "joint_shared_adapter_mtl",
    "lac_v60_single_task_tbr": "single_task_tbr",
    "lac_v60_single_task_cac": "single_task_cac",
}
AVAILABLE_TASKS = {
    "lac_v60_joint_hps_mtl": ("TBR", "CAC"),
    "lac_v60_joint_shared_adapter_mtl": ("TBR", "CAC"),
    "lac_v60_single_task_tbr": ("TBR",),
    "lac_v60_single_task_cac": ("CAC",),
}
PRIMARY_COMPARISONS = (
    ("lac_v60_joint_shared_adapter_mtl", "lac_v60_joint_hps_mtl", "TBR"),
    ("lac_v60_joint_shared_adapter_mtl", "lac_v60_joint_hps_mtl", "CAC"),
    ("lac_v60_joint_hps_mtl", "lac_v60_single_task_tbr", "TBR"),
    ("lac_v60_joint_hps_mtl", "lac_v60_single_task_cac", "CAC"),
    ("lac_v60_joint_shared_adapter_mtl", "lac_v60_single_task_tbr", "TBR"),
    ("lac_v60_joint_shared_adapter_mtl", "lac_v60_single_task_cac", "CAC"),
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_control_oof(
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


def _named_available_metrics(model_metrics: pd.DataFrame) -> pd.DataFrame:
    available = model_metrics[
        model_metrics.apply(
            lambda row: row["task"] in AVAILABLE_TASKS[row["model"]], axis=1
        )
    ].copy()
    available["display_name"] = available["model"].map(DISPLAY_NAMES)
    return available


def _primary_differences(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right, task in PRIMARY_COMPARISONS:
        selected = pairwise[
            (pairwise["left_model"] == left)
            & (pairwise["right_model"] == right)
            & (pairwise["task"] == task)
        ]
        if selected.empty:
            selected = pairwise[
                (pairwise["left_model"] == right)
                & (pairwise["right_model"] == left)
                & (pairwise["task"] == task)
            ].copy()
            if selected.empty:
                raise RuntimeError(f"Missing comparison: {left}, {right}, {task}")
            selected[["left_model", "right_model"]] = selected[
                ["right_model", "left_model"]
            ].to_numpy()
            selected[["left_estimate", "right_estimate"]] = selected[
                ["right_estimate", "left_estimate"]
            ].to_numpy()
            selected["difference_estimate"] = -selected["difference_estimate"]
            old_low = selected["ci_low"].copy()
            selected["ci_low"] = -selected["ci_high"]
            selected["ci_high"] = -old_low
            selected["bootstrap_probability_left_lower"] = (
                1.0 - selected["bootstrap_probability_left_lower"]
            )
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    result["left_display_name"] = result["left_model"].map(DISPLAY_NAMES)
    result["right_display_name"] = result["right_model"].map(DISPLAY_NAMES)
    result["comparison"] = (
        result["left_display_name"] + " vs " + result["right_display_name"]
    )
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
    available = _named_available_metrics(model_metrics)
    primary = _primary_differences(pairwise)
    all_pairs = pairwise.copy()
    all_pairs["left_display_name"] = all_pairs["left_model"].map(DISPLAY_NAMES)
    all_pairs["right_display_name"] = all_pairs["right_model"].map(DISPLAY_NAMES)

    metrics_path = output / "dual_task_controls_metrics_95ci.csv"
    differences_path = output / "dual_task_controls_primary_differences_95ci.csv"
    all_pairs_path = output / "dual_task_controls_all_pairwise_95ci.csv"
    replicates_path = output / "dual_task_controls_metric_replicates.csv"
    audit_path = output / "dual_task_controls_audit.json"
    report_path = output / "A_CENTER_LEAD3_DUAL_TASK_CONTROLS_REPORT_CN.md"
    available.to_csv(metrics_path, index=False)
    primary.to_csv(differences_path, index=False)
    all_pairs.to_csv(all_pairs_path, index=False)
    replicate_metrics.to_csv(replicates_path, index=False)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_rows = []
    for model in MODEL_ORDER:
        row: dict[str, Any] = {"Model": DISPLAY_NAMES[model], "n": audit["patient_count"]}
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
    for (left, right, task), group in primary.groupby(
        ["left_model", "right_model", "task"], sort=False
    ):
        row = {
            "Comparison": f"{DISPLAY_NAMES[left]} vs {DISPLAY_NAMES[right]}",
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
            "# A中心提前3个月预测：双任务与单任务对照",
            "",
            "## 统计口径",
            "",
            "- 使用最新提前3个月数据切片及同一锁定A中心患者级内部五折。",
            f"- 合并OOF患者数为{audit['patient_count']}；patient-level paired bootstrap {audit['bootstrap_replicates']}次（seed={audit['bootstrap_seed']}）。",
            "- joint模型同时优化标准化TBR与CAC损失；single-task模型仅报告实际训练的对应任务。",
            "- 每次bootstrap使用同一批患者索引同步重采样全部模型、任务和指标。",
            "- 差值定义为左侧模型减右侧模型；MAE/RMSE差值为负表示左侧模型误差更低。",
            "",
            "## 性能汇总",
            "",
            _markdown_table(pd.DataFrame(metric_rows)),
            "",
            "## 预设核心配对比较",
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
        "primary_differences": differences_path,
        "all_pairwise": all_pairs_path,
        "bootstrap_metric_replicates": replicates_path,
        "audit": audit_path,
        "report": report_path,
    }
    manifest = {
        **audit,
        "model_order": list(MODEL_ORDER),
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
    manifest_path = output / "dual_task_controls_manifest.json"
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
    frames, paths = load_control_oof(
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
