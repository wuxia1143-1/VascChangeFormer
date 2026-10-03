from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_ORDER = (
    "lac_v60_a_no_i2c_tail",
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
)
DISPLAY_NAMES = {
    "lac_v60_a_no_i2c_tail": "LAC-iTransformer V6-A",
    "persistence": "Persistence",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "apn_dr": "APN-DR",
    "itransformer_mtl": "iTransformer-MTL",
    "first_icu_mtl": "FIRST-ICU-MTL",
    "learning_to_route": "Learning to Route",
}
TASKS = {
    "TBR": ("true_delta_tbr", "pred_delta_tbr"),
    "CAC": ("true_delta_log_cac", "pred_delta_log_cac"),
}
METRICS = ("MAE", "RMSE")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_oof(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str}).sort_values(
        "patient_id"
    ).reset_index(drop=True)
    required = {
        "patient_id",
        "outer_fold",
        *(column for columns in TASKS.values() for column in columns),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF file is missing columns: {sorted(missing)}")
    if frame["patient_id"].duplicated().any():
        raise ValueError("OOF file contains duplicate patient IDs")
    numeric = ["outer_fold", *(column for columns in TASKS.values() for column in columns)]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("OOF file contains non-finite folds, targets, or predictions")
    return frame


def _validate_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if not reference["patient_id"].equals(candidate["patient_id"]):
        raise ValueError("OOF patient IDs are not aligned")
    if not reference["outer_fold"].equals(candidate["outer_fold"]):
        raise ValueError("OOF patient-to-fold mappings differ")
    for target_column, _ in TASKS.values():
        if not np.allclose(
            reference[target_column].to_numpy(float),
            candidate[target_column].to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"OOF targets differ: {target_column}")


def load_experiment_oof(
    proposed_oof: str | Path,
    baseline_root: str | Path,
    expected_patient_count: int = 443,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {
        MODEL_ORDER[0]: Path(proposed_oof),
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


def simultaneous_paired_bootstrap(
    frames: dict[str, pd.DataFrame],
    model_order: tuple[str, ...] = MODEL_ORDER,
    replicates: int = 2000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")
    if not model_order:
        raise ValueError("model_order must not be empty")
    reference = frames[model_order[0]]
    for model in model_order[1:]:
        _validate_alignment(reference, frames[model])
    n_patients = len(reference)
    if n_patients < 1:
        raise ValueError("OOF frames must contain at least one patient")

    rng = np.random.default_rng(int(seed))
    # This single matrix is deliberately reused for every model, task, and metric.
    sample_indices = rng.integers(
        0,
        n_patients,
        size=(int(replicates), n_patients),
        dtype=np.int64,
    )
    sample_index_sha256 = hashlib.sha256(sample_indices.tobytes()).hexdigest()

    observed: dict[tuple[str, str, str], float] = {}
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    replicate_data: dict[str, np.ndarray] = {
        "bootstrap_replicate": np.arange(1, int(replicates) + 1, dtype=int)
    }
    for task, (target_column, prediction_column) in TASKS.items():
        target = reference[target_column].to_numpy(float)
        for model in model_order:
            prediction = frames[model][prediction_column].to_numpy(float)
            error = prediction - target
            absolute_error = np.abs(error)
            squared_error = np.square(error)
            mae_samples = absolute_error[sample_indices].mean(axis=1)
            rmse_samples = np.sqrt(squared_error[sample_indices].mean(axis=1))
            observed[(model, task, "MAE")] = float(absolute_error.mean())
            observed[(model, task, "RMSE")] = float(np.sqrt(squared_error.mean()))
            samples[(model, task, "MAE")] = mae_samples
            samples[(model, task, "RMSE")] = rmse_samples
            replicate_data[f"{model}__{task.lower()}__mae"] = mae_samples
            replicate_data[f"{model}__{task.lower()}__rmse"] = rmse_samples

    model_rows = []
    for model in model_order:
        for task in TASKS:
            for metric in METRICS:
                values = samples[(model, task, metric)]
                model_rows.append(
                    {
                        "model": model,
                        "display_name": DISPLAY_NAMES.get(model, model),
                        "task": task,
                        "metric": metric,
                        "estimate": observed[(model, task, metric)],
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_seed": int(seed),
                    }
                )

    pair_rows = []
    for left_model, right_model in itertools.combinations(model_order, 2):
        for task in TASKS:
            for metric in METRICS:
                difference_samples = (
                    samples[(left_model, task, metric)]
                    - samples[(right_model, task, metric)]
                )
                estimate = (
                    observed[(left_model, task, metric)]
                    - observed[(right_model, task, metric)]
                )
                ci_low = float(np.quantile(difference_samples, 0.025))
                ci_high = float(np.quantile(difference_samples, 0.975))
                if ci_high < 0.0:
                    favored_model = left_model
                elif ci_low > 0.0:
                    favored_model = right_model
                else:
                    favored_model = "inconclusive"
                pair_rows.append(
                    {
                        "left_model": left_model,
                        "left_display_name": DISPLAY_NAMES.get(left_model, left_model),
                        "right_model": right_model,
                        "right_display_name": DISPLAY_NAMES.get(right_model, right_model),
                        "comparison": (
                            f"{DISPLAY_NAMES.get(left_model, left_model)} vs "
                            f"{DISPLAY_NAMES.get(right_model, right_model)}"
                        ),
                        "task": task,
                        "metric": metric,
                        "difference_definition": "left_minus_right",
                        "left_estimate": observed[(left_model, task, metric)],
                        "right_estimate": observed[(right_model, task, metric)],
                        "difference_estimate": float(estimate),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "ci_excludes_zero": bool(ci_high < 0.0 or ci_low > 0.0),
                        "favored_model": favored_model,
                        "bootstrap_probability_left_lower": float(
                            np.mean(difference_samples < 0.0)
                        ),
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_seed": int(seed),
                    }
                )

    audit = {
        "status": "passed",
        "method": "patient_level_simultaneous_paired_bootstrap",
        "patient_count": int(n_patients),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "same_resample_indices_across_all_models_tasks_metrics": True,
        "bootstrap_index_matrix_shape": [int(replicates), int(n_patients)],
        "bootstrap_index_matrix_sha256": sample_index_sha256,
        "ci_method": "unadjusted_percentile_2.5_97.5",
        "difference_definition": "left_minus_right; negative favors left for MAE/RMSE",
        "aligned_patient_ids": True,
        "aligned_outer_folds": True,
        "aligned_targets": True,
        "all_required_values_finite": True,
    }
    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(replicate_data),
        audit,
    )


def _metric_ci_text(row: pd.Series) -> str:
    return f"{row['estimate']:.4f} ({row['ci_low']:.4f}, {row['ci_high']:.4f})"


def _difference_ci_text(row: pd.Series) -> str:
    return (
        f"{row['difference_estimate']:.4f} "
        f"({row['ci_low']:.4f}, {row['ci_high']:.4f})"
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
    source_paths: dict[str, Path],
    output: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    primary_model = MODEL_ORDER[0]
    lac_comparisons = pairwise[pairwise["left_model"] == primary_model].copy()

    model_path = output / "paired_bootstrap_model_metrics_95ci.csv"
    lac_path = output / "paired_bootstrap_lac_vs_baselines_95ci.csv"
    all_pairs_path = output / "paired_bootstrap_all_pairwise_95ci.csv"
    replicate_path = output / "paired_bootstrap_metric_replicates.csv"
    audit_path = output / "paired_bootstrap_audit.json"
    report_path = output / "A_CENTER_LEAD3_PAIRED_BOOTSTRAP_REPORT_CN.md"
    model_metrics.to_csv(model_path, index=False)
    lac_comparisons.to_csv(lac_path, index=False)
    pairwise.to_csv(all_pairs_path, index=False)
    replicate_metrics.to_csv(replicate_path, index=False)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    main_rows = []
    for model in MODEL_ORDER:
        row: dict[str, str] = {"Model": DISPLAY_NAMES[model]}
        for task in TASKS:
            for metric in METRICS:
                record = model_metrics[
                    (model_metrics["model"] == model)
                    & (model_metrics["task"] == task)
                    & (model_metrics["metric"] == metric)
                ].iloc[0]
                row[f"{task} {metric} (95% CI)"] = _metric_ci_text(record)
        main_rows.append(row)

    comparison_rows = []
    for baseline in MODEL_ORDER[1:]:
        row = {
            "Comparison": f"LAC vs {DISPLAY_NAMES[baseline]}",
        }
        for task in TASKS:
            for metric in METRICS:
                record = lac_comparisons[
                    (lac_comparisons["right_model"] == baseline)
                    & (lac_comparisons["task"] == task)
                    & (lac_comparisons["metric"] == metric)
                ].iloc[0]
                row[f"Δ{task} {metric} (95% CI)"] = _difference_ci_text(record)
        comparison_rows.append(row)

    report = "\n".join(
        [
            "# A中心提前3个月预测：Patient-level Paired Bootstrap",
            "",
            "## 统计口径",
            "",
            f"- 基于{audit['patient_count']}例患者的合并OOF预测，bootstrap {audit['bootstrap_replicates']}次。",
            f"- 随机种子：{audit['bootstrap_seed']}；每次生成一组患者索引，并将完全相同的索引同时用于全部模型、任务和指标。",
            "- 95% CI采用未作多重比较校正的percentile bootstrap（2.5%与97.5%分位数）。",
            "- TBR对应ΔTBR；CAC对应Δlog(1+CAC)。",
            "- 差值定义为LAC − baseline；MAE/RMSE差值为负表示LAC误差更小。",
            "",
            "## 各模型指标",
            "",
            _markdown_table(pd.DataFrame(main_rows)),
            "",
            "## LAC与核心Baseline的配对差值",
            "",
            _markdown_table(pd.DataFrame(comparison_rows)),
            "",
            "## 审计",
            "",
            f"- 同步重采样索引矩阵SHA256：`{audit['bootstrap_index_matrix_sha256']}`。",
            "- 患者ID、外折、真实目标和数值有限性检查均通过。",
            "- 全部模型两两差值见 `paired_bootstrap_all_pairwise_95ci.csv`。",
            "- CI为探索性区间；未进行多重比较校正。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    output_files = {
        "model_metrics": model_path,
        "lac_vs_baselines": lac_path,
        "all_pairwise": all_pairs_path,
        "bootstrap_metric_replicates": replicate_path,
        "audit": audit_path,
        "report": report_path,
    }
    manifest = {
        **audit,
        "source_oof": {
            model: {"path": str(path), "sha256": _sha256(path)}
            for model, path in source_paths.items()
        },
        "outputs": {
            name: {
                "file": path.name,
                "sha256": _sha256(path),
            }
            for name, path in output_files.items()
        },
    }
    manifest_path = output / "paired_bootstrap_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-oof", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-patient-count", type=int, default=443)
    args = parser.parse_args()

    frames, paths = load_experiment_oof(
        args.proposed_oof,
        args.baseline_root,
        expected_patient_count=args.expected_patient_count,
    )
    model_metrics, pairwise, replicate_metrics, audit = simultaneous_paired_bootstrap(
        frames,
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
