from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .a_center_paired_bootstrap import _load_oof, _sha256


HORIZONS = (6, 12, 18, 24)
INTERVAL_LABELS = {
    6: "[3, 9]",
    12: "(9, 15]",
    18: "(15, 21]",
    24: "(21, 27]",
}
TASKS = {
    "TBR": ("true_delta_tbr", "pred_delta_tbr"),
    "CAC": ("true_delta_log_cac", "pred_delta_log_cac"),
}
METRICS = ("MAE", "RMSE")


def assign_nominal_horizon(followup_months: np.ndarray) -> np.ndarray:
    months = np.asarray(followup_months, dtype=float)
    if not np.isfinite(months).all():
        raise ValueError("followup_months contains non-finite values")
    horizon = np.full(months.shape, -1, dtype=int)
    horizon[(months >= 3.0) & (months <= 9.0)] = 6
    horizon[(months > 9.0) & (months <= 15.0)] = 12
    horizon[(months > 15.0) & (months <= 21.0)] = 18
    horizon[(months > 21.0) & (months <= 27.0)] = 24
    if np.any(horizon < 0):
        raise ValueError(
            "followup_months contains values outside the prespecified [3, 27] month bins"
        )
    return horizon


def load_stage_analysis_frame(
    oof_path: str | Path,
    prepared_data_path: str | Path,
    expected_patient_count: int = 443,
) -> pd.DataFrame:
    oof = _load_oof(oof_path)
    with np.load(prepared_data_path, allow_pickle=False) as prepared:
        required = {"patient_ids", "followup_months"}
        missing = required - set(prepared.files)
        if missing:
            raise ValueError(f"Prepared data is missing arrays: {sorted(missing)}")
        patient_ids = prepared["patient_ids"].astype(str)
        followup_months = prepared["followup_months"].astype(float)
    if patient_ids.ndim != 1 or followup_months.ndim != 1:
        raise ValueError("patient_ids and followup_months must be one-dimensional")
    if len(patient_ids) != len(followup_months):
        raise ValueError("patient_ids and followup_months lengths differ")
    metadata = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "actual_followup_months": followup_months,
        }
    )
    if metadata["patient_id"].duplicated().any():
        raise ValueError("Prepared data contains duplicate patient IDs")
    if len(oof) != int(expected_patient_count) or len(metadata) != int(
        expected_patient_count
    ):
        raise ValueError(
            "OOF and prepared data must both contain the expected patient count"
        )
    merged = oof.merge(
        metadata,
        on="patient_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"OOF and prepared data have {unmatched} unmatched patient IDs")
    merged = merged.drop(columns="_merge")
    merged["nominal_horizon_months"] = assign_nominal_horizon(
        merged["actual_followup_months"].to_numpy(float)
    )
    if len(merged) != int(expected_patient_count):
        raise ValueError("Merged stage-analysis frame has an unexpected patient count")
    return merged.sort_values("patient_id").reset_index(drop=True)


def bootstrap_stage_performance(
    frame: pd.DataFrame,
    replicates: int = 2000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")
    observed_horizons = set(frame["nominal_horizon_months"].astype(int).unique())
    if observed_horizons != set(HORIZONS):
        raise ValueError(
            f"Expected non-empty horizons {HORIZONS}, found {sorted(observed_horizons)}"
        )
    required = {
        "patient_id",
        "actual_followup_months",
        "nominal_horizon_months",
        *(column for columns in TASKS.values() for column in columns),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stage-analysis frame is missing columns: {sorted(missing)}")
    numeric = [
        "actual_followup_months",
        "nominal_horizon_months",
        *(column for columns in TASKS.values() for column in columns),
    ]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Stage-analysis frame contains non-finite values")

    rng = np.random.default_rng(int(seed))
    stage_rows: list[dict[str, Any]] = []
    samples: dict[tuple[int, str, str], np.ndarray] = {}
    replicate_data: dict[str, np.ndarray] = {
        "bootstrap_replicate": np.arange(1, int(replicates) + 1, dtype=int)
    }
    group_audit: dict[str, Any] = {}
    for horizon in HORIZONS:
        group = frame[frame["nominal_horizon_months"] == horizon].reset_index(drop=True)
        n = len(group)
        sample_indices = rng.integers(
            0,
            n,
            size=(int(replicates), n),
            dtype=np.int64,
        )
        index_hash = hashlib.sha256(sample_indices.tobytes()).hexdigest()
        row: dict[str, Any] = {
            "nominal_horizon_months": horizon,
            "actual_followup_interval": INTERVAL_LABELS[horizon],
            "n": int(n),
            "actual_followup_min": float(group["actual_followup_months"].min()),
            "actual_followup_median": float(group["actual_followup_months"].median()),
            "actual_followup_max": float(group["actual_followup_months"].max()),
        }
        for task, (target_column, prediction_column) in TASKS.items():
            error = (
                group[prediction_column].to_numpy(float)
                - group[target_column].to_numpy(float)
            )
            absolute_error = np.abs(error)
            squared_error = np.square(error)
            mae_samples = absolute_error[sample_indices].mean(axis=1)
            rmse_samples = np.sqrt(squared_error[sample_indices].mean(axis=1))
            samples[(horizon, task, "MAE")] = mae_samples
            samples[(horizon, task, "RMSE")] = rmse_samples
            replicate_data[f"h{horizon}__{task.lower()}__mae"] = mae_samples
            replicate_data[f"h{horizon}__{task.lower()}__rmse"] = rmse_samples
            prefix = task.lower()
            row[f"{prefix}_mae"] = float(absolute_error.mean())
            row[f"{prefix}_mae_ci_low"] = float(np.quantile(mae_samples, 0.025))
            row[f"{prefix}_mae_ci_high"] = float(np.quantile(mae_samples, 0.975))
            row[f"{prefix}_rmse"] = float(np.sqrt(squared_error.mean()))
            row[f"{prefix}_rmse_ci_low"] = float(np.quantile(rmse_samples, 0.025))
            row[f"{prefix}_rmse_ci_high"] = float(np.quantile(rmse_samples, 0.975))
        stage_rows.append(row)
        group_audit[str(horizon)] = {
            "actual_followup_interval": INTERVAL_LABELS[horizon],
            "n": int(n),
            "bootstrap_index_matrix_shape": [int(replicates), int(n)],
            "bootstrap_index_matrix_sha256": index_hash,
            "same_indices_reused_for_all_tasks_and_metrics_within_stage": True,
        }

    pair_rows: list[dict[str, Any]] = []
    for left_horizon, right_horizon in itertools.combinations(HORIZONS, 2):
        for task in TASKS:
            for metric in METRICS:
                left_values = samples[(left_horizon, task, metric)]
                right_values = samples[(right_horizon, task, metric)]
                differences = left_values - right_values
                left_record = stage_rows[HORIZONS.index(left_horizon)]
                right_record = stage_rows[HORIZONS.index(right_horizon)]
                field = f"{task.lower()}_{metric.lower()}"
                estimate = float(left_record[field] - right_record[field])
                ci_low = float(np.quantile(differences, 0.025))
                ci_high = float(np.quantile(differences, 0.975))
                pair_rows.append(
                    {
                        "left_horizon_months": left_horizon,
                        "right_horizon_months": right_horizon,
                        "comparison": f"{left_horizon}m minus {right_horizon}m",
                        "task": task,
                        "metric": metric,
                        "difference_definition": "left_stage_minus_right_stage",
                        "difference_estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "ci_excludes_zero": bool(ci_high < 0.0 or ci_low > 0.0),
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_seed": int(seed),
                    }
                )

    audit = {
        "status": "passed",
        "analysis": "A-center 3-month-ahead performance by actual follow-up stage",
        "patient_count": int(len(frame)),
        "bootstrap_replicates_per_stage": int(replicates),
        "bootstrap_seed": int(seed),
        "ci_method": "unadjusted_percentile_2.5_97.5",
        "metric_scales": {"TBR": "delta_TBR", "CAC": "delta_log1p_CAC"},
        "horizon_assignment": {
            "6": "3 <= actual_followup_months <= 9",
            "12": "9 < actual_followup_months <= 15",
            "18": "15 < actual_followup_months <= 21",
            "24": "21 < actual_followup_months <= 27",
        },
        "actual_second_pet_ct_value_used_as_target": True,
        "three_month_lead_slice_used": True,
        "groups": group_audit,
        "patient_ids_unique_and_joined_one_to_one": True,
        "all_required_values_finite": True,
        "stage_pairwise_differences_are_unpaired_independent_group_bootstraps": True,
    }
    return (
        pd.DataFrame(stage_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(replicate_data),
        audit,
    )


def _ci_text(row: pd.Series, field: str) -> str:
    return (
        f"{row[field]:.4f} "
        f"({row[f'{field}_ci_low']:.4f}, {row[f'{field}_ci_high']:.4f})"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def write_stage_outputs(
    stage_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    replicate_metrics: pd.DataFrame,
    audit: dict[str, Any],
    oof_path: str | Path,
    prepared_data_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "stage_performance_metrics_95ci.csv"
    pairwise_path = output / "stage_pairwise_differences_95ci.csv"
    replicates_path = output / "stage_bootstrap_metric_replicates.csv"
    audit_path = output / "stage_stability_audit.json"
    report_path = output / "A_CENTER_LEAD3_STAGE_STABILITY_REPORT_CN.md"
    stage_metrics.to_csv(metrics_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    replicate_metrics.to_csv(replicates_path, index=False)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    table_rows = []
    for _, row in stage_metrics.iterrows():
        table_rows.append(
            {
                "名义随访阶段": f"{int(row['nominal_horizon_months'])}个月",
                "真实周期范围": row["actual_followup_interval"],
                "n": int(row["n"]),
                "TBR MAE (95% CI)": _ci_text(row, "tbr_mae"),
                "TBR RMSE (95% CI)": _ci_text(row, "tbr_rmse"),
                "CAC MAE (95% CI)": _ci_text(row, "cac_mae"),
                "CAC RMSE (95% CI)": _ci_text(row, "cac_rmse"),
            }
        )
    significant = pairwise[pairwise["ci_excludes_zero"]]
    if significant.empty:
        stability_text = "四项指标的全部阶段间差值95% CI均包含0。"
    else:
        summaries = [
            f"{int(row.left_horizon_months)}m−{int(row.right_horizon_months)}m "
            f"{row.task} {row.metric}: {row.difference_estimate:.4f} "
            f"({row.ci_low:.4f}, {row.ci_high:.4f})"
            for row in significant.itertuples(index=False)
        ]
        stability_text = "差值95% CI排除0的探索性比较：" + "；".join(summaries) + "。"
    report = "\n".join(
        [
            "# A中心不同随访阶段的提前3个月预测稳定性",
            "",
            "## 实验口径",
            "",
            "- 当前模型：LAC-iTransformer V6-A；评价使用锁定患者级五折的合并OOF预测。",
            "- 输入仍采用真实第二次PET/CT终点前至少3个月的数据切片。",
            "- 分组规则：[3,9]→6个月，(9,15]→12个月，(15,21]→18个月，(21,27]→24个月。",
            "- 目标为患者真实第二次PET/CT的实测终点值；名义月份仅用于随访阶段分层。",
            f"- 每组独立进行{audit['bootstrap_replicates_per_stage']}次patient-level bootstrap；组内同一索引同时用于四项指标。",
            "- 95% CI为未作多重比较校正的percentile区间；TBR为ΔTBR，CAC为Δlog(1+CAC)。",
            "",
            "## 分阶段性能",
            "",
            _markdown_table(pd.DataFrame(table_rows)),
            "",
            "## 稳定性探索",
            "",
            stability_text,
            "",
            "阶段间比较来自互斥患者组的独立组bootstrap，并非配对比较；完整结果见 `stage_pairwise_differences_95ci.csv`。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    output_paths = {
        "stage_metrics": metrics_path,
        "stage_pairwise_differences": pairwise_path,
        "bootstrap_metric_replicates": replicates_path,
        "audit": audit_path,
        "report": report_path,
    }
    manifest = {
        **audit,
        "sources": {
            "oof_predictions": {
                "path": str(oof_path),
                "sha256": _sha256(oof_path),
            },
            "prepared_data": {
                "path": str(prepared_data_path),
                "sha256": _sha256(prepared_data_path),
            },
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in output_paths.items()
        },
    }
    manifest_path = output / "stage_stability_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-patient-count", type=int, default=443)
    args = parser.parse_args()

    frame = load_stage_analysis_frame(
        args.oof,
        args.prepared_data,
        expected_patient_count=args.expected_patient_count,
    )
    stage_metrics, pairwise, replicate_metrics, audit = bootstrap_stage_performance(
        frame,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    manifest = write_stage_outputs(
        stage_metrics,
        pairwise,
        replicate_metrics,
        audit,
        args.oof,
        args.prepared_data,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "patient_count": manifest["patient_count"],
                "group_counts": {
                    key: value["n"] for key, value in manifest["groups"].items()
                },
                "bootstrap_replicates_per_stage": manifest[
                    "bootstrap_replicates_per_stage"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
