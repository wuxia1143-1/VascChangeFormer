from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.schema import FeatureSchema
from ..training.folds import build_patient_fold_plan, save_patient_fold_plan
from ..training.metrics import bootstrap_metrics, regression_metrics
from ..training.trainer import run_cross_validation
from .comparison import PRESPECIFIED_MODELS
from .prediction import model_registry


SINGLE_TASK_COMPONENTS = ("single_tbr", "single_cac")
INTERNAL_FIVEFOLD_MODELS = PRESPECIFIED_MODELS + SINGLE_TASK_COMPONENTS


def _config_checksum(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean_and_standard_deviation(
    fold_metrics: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    keys = tuple(fold_metrics[0])
    mean = {
        key: float(np.mean([row[key] for row in fold_metrics]))
        for key in keys
    }
    standard_deviation = {
        key: float(np.std([row[key] for row in fold_metrics], ddof=1))
        for key in keys
    }
    return mean, standard_deviation


def combine_single_task_oof_predictions(
    suite_dir: str | Path,
    seed: int = 2026,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Pair the valid TBR and CAC heads from two independently fitted models."""
    root = Path(suite_dir)
    tbr = pd.read_csv(root / "single_tbr" / "out_of_fold_predictions.csv")
    cac = pd.read_csv(root / "single_cac" / "out_of_fold_predictions.csv")
    required = {"patient_id", "fold", "true_tbr", "pred_tbr", "true_cac", "pred_cac"}
    if not required.issubset(tbr) or not required.issubset(cac):
        raise ValueError("Single-task OOF files do not satisfy the prediction contract")
    if tbr["patient_id"].duplicated().any() or cac["patient_id"].duplicated().any():
        raise ValueError("Single-task OOF files contain duplicate patients")
    merged = tbr.merge(cac, on="patient_id", suffixes=("_tbr", "_cac"), validate="one_to_one")
    if len(merged) != len(tbr) or len(merged) != len(cac):
        raise ValueError("Single-task OOF patient sets differ")
    if not np.array_equal(merged["fold_tbr"].to_numpy(), merged["fold_cac"].to_numpy()):
        raise ValueError("Single-task models were not evaluated with identical folds")
    for endpoint in ("tbr", "cac"):
        if not np.allclose(
            merged[f"true_{endpoint}_tbr"], merged[f"true_{endpoint}_cac"], equal_nan=True
        ):
            raise ValueError(f"Single-task OOF truth mismatch for {endpoint}")
    predictions = pd.DataFrame({
        "patient_id": merged["patient_id"],
        "fold": merged["fold_tbr"].astype(int),
        "true_tbr": merged["true_tbr_tbr"],
        "pred_tbr": merged["pred_tbr_tbr"],
        "true_cac": merged["true_cac_tbr"],
        "pred_cac": merged["pred_cac_cac"],
    }).sort_values("patient_id").reset_index(drop=True)
    output = root / "single_task_pair"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)
    fold_metrics = []
    for fold in sorted(predictions["fold"].unique()):
        frame = predictions[predictions["fold"] == fold]
        metrics = regression_metrics(
            frame[["true_tbr", "true_cac"]].to_numpy(),
            frame[["pred_tbr", "pred_cac"]].to_numpy(),
        )
        fold_metrics.append({"fold": int(fold)} | metrics)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    means, standard_deviation = _mean_and_standard_deviation(
        [{key: value for key, value in row.items() if key != "fold"} for row in fold_metrics]
    )
    targets = predictions[["true_tbr", "true_cac"]].to_numpy()
    estimates = predictions[["pred_tbr", "pred_cac"]].to_numpy()
    summary = {
        "model_name": "single_task_pair",
        "components": {"tbr": "single_tbr", "cac": "single_cac"},
        "metric_scope": "both_endpoints",
        "fold_metrics": fold_metrics,
        "mean_metrics": means,
        "fold_standard_deviation": standard_deviation,
        "pooled_oof_metrics": regression_metrics(targets, estimates),
        "bootstrap_95_ci": bootstrap_metrics(
            targets, estimates, n_bootstrap=bootstrap_replicates, seed=seed
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "external_validation_status": "not_run",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_internal_fivefold_suite(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    base_config: dict[str, Any],
    output_dir: str | Path,
    models: Iterable[str] = INTERNAL_FIVEFOLD_MODELS,
) -> dict[str, Any]:
    """Run one locked patient-level 5-fold plan for every requested model."""
    if int(base_config.get("num_folds", 5)) != 5:
        raise ValueError("Internal validation is locked to exactly five patient-level folds")
    selected = tuple(dict.fromkeys(models))
    allowed_models = set(model_registry()) | set(SINGLE_TASK_COMPONENTS)
    unknown = set(selected) - allowed_models
    if unknown:
        raise KeyError(f"Unknown internal-validation models: {sorted(unknown)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(base_config.get("seed", 2026))
    bootstrap_replicates = int(base_config.get("bootstrap_replicates", 2000))
    fold_plan = build_patient_fold_plan(arrays["patient_ids"], n_folds=5, seed=seed)
    fold_manifest = save_patient_fold_plan(
        fold_plan, output / "patient_fold_plan.csv", seed=seed, n_folds=5
    )
    (output / "training_config.json").write_text(
        json.dumps(base_config, indent=2), encoding="utf-8"
    )
    summaries: dict[str, Any] = {}
    for model_name in selected:
        config = copy.deepcopy(base_config)
        config["model_name"] = model_name
        summaries[model_name] = run_cross_validation(
            arrays,
            schema,
            config,
            output / model_name,
            fold_assignments=fold_plan,
        )
    if set(SINGLE_TASK_COMPONENTS).issubset(selected):
        summaries["single_task_pair"] = combine_single_task_oof_predictions(
            output, seed=seed, bootstrap_replicates=bootstrap_replicates
        )
    rows = []
    for model_name, summary in summaries.items():
        role = "single_task_component" if model_name in SINGLE_TASK_COMPONENTS else "complete_model"
        rows.append(
            {
                "model": model_name,
                "evaluation_role": role,
                "metric_scope": summary["metric_scope"],
            }
            | summary["pooled_oof_metrics"]
        )
    pd.DataFrame(rows).to_csv(output / "internal_fivefold_metrics.csv", index=False)
    checksums = {
        summary["cv_protocol"]["fold_plan_checksum"]
        for name, summary in summaries.items()
        if name != "single_task_pair"
    }
    if checksums != {fold_manifest["checksum_sha256"]}:
        raise RuntimeError("Models did not use the same locked patient fold plan")
    result_scope = str(
        base_config.get("result_scope", "development_internal_fivefold")
    )
    if result_scope.startswith("synthetic_smoke"):
        status = "synthetic_smoke_only"
    elif result_scope.startswith("synthetic"):
        status = "synthetic_diagnostic_only"
    else:
        status = "development_internal_validation_completed"
    manifest = {
        "status": status,
        "formal_conclusions_generated": False,
        "result_scope": result_scope,
        "protocol": "patient-level five-fold cross-validation",
        "outer_loop": "four folds train, one fold untouched test",
        "deep_model_selection": "inner split of the four-fold training pool followed by full four-fold refit",
        "same_patient_folds_for_all_models": True,
        "test_fold_used_only_for_final_evaluation": True,
        "external_center_used": False,
        "training_config_sha256": _config_checksum(base_config),
        "fold_plan": fold_manifest,
        "models": list(selected),
        "derived_models": ["single_task_pair"] if "single_task_pair" in summaries else [],
        "summaries": summaries,
    }
    (output / "internal_fivefold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
