from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from ..data.preprocessing import FoldPreprocessor, subset_arrays
from ..data.schema import FeatureSchema
from ..experiments.prediction import build_classical_model, build_torch_model
from .folds import (
    build_patient_fold_plan,
    fold_plan_checksum,
    indices_for_outer_fold,
    save_patient_fold_plan,
    validate_patient_fold_plan,
)
from .metrics import (
    bootstrap_change_metrics,
    bootstrap_metrics,
    change_space_metrics,
    regression_metrics,
)
from .trainer import (
    _git_revision,
    _loader,
    _model_config,
    _parameter_count,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    predict,
    resolve_device,
    seed_everything,
    train_model,
)


CLASSICAL_MODELS = {"elastic_net", "xgboost"}


def _parameter_grid(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return [{}]
    if isinstance(value, list):
        if not value or not all(isinstance(item, dict) for item in value):
            raise ValueError("A list-valued grid must contain parameter dictionaries")
        return [dict(item) for item in value]
    if not isinstance(value, dict):
        raise ValueError("Classical grid must be a dictionary or list of dictionaries")
    keys = list(value)
    choices = [
        candidate if isinstance(candidate, list) else [candidate]
        for candidate in value.values()
    ]
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*choices)
    ]


def _candidate_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _residual_targets(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            arrays["targets"][:, 0] - arrays["baseline"][:, 0],
            np.log1p(np.maximum(arrays["targets"][:, 1], 0))
            - np.log1p(np.maximum(arrays["baseline"][:, 1], 0)),
        ]
    )


def _robust_task_scales(train_arrays: dict[str, np.ndarray]) -> np.ndarray:
    residuals = _residual_targets(train_arrays)
    q25, q75 = np.quantile(residuals, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    standard_deviation = np.std(residuals, axis=0, ddof=1)
    return np.where(
        iqr > 1e-6,
        iqr,
        np.where(standard_deviation > 1e-6, standard_deviation, 1.0),
    )


def _joint_selection_score(
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    predictions: np.ndarray,
) -> tuple[float, dict[str, float]]:
    truth = _residual_targets(validation_arrays)
    predicted = np.column_stack(
        [
            predictions[:, 0] - validation_arrays["baseline"][:, 0],
            np.log1p(np.maximum(predictions[:, 1], 0))
            - np.log1p(np.maximum(validation_arrays["baseline"][:, 1], 0)),
        ]
    )
    scales = _robust_task_scales(train_arrays)
    task_mae = np.mean(np.abs(truth - predicted), axis=0)
    normalized = task_mae / scales
    return float(np.mean(normalized)), {
        "delta_tbr_mae": float(task_mae[0]),
        "delta_log_cac_mae": float(task_mae[1]),
        "delta_tbr_training_iqr_scale": float(scales[0]),
        "delta_log_cac_training_iqr_scale": float(scales[1]),
        "joint_normalized_mae": float(np.mean(normalized)),
    }


def _save_inner_plan(
    train_pool_raw: dict[str, np.ndarray],
    inner_plan: Mapping[str, int],
    fold_dir: Path,
    seed: int,
    inner_folds: int,
) -> dict[str, Any]:
    return save_patient_fold_plan(
        inner_plan,
        fold_dir / "inner_patient_fold_plan.csv",
        seed=seed,
        n_folds=inner_folds,
    )


def _select_classical_parameters(
    model_name: str,
    train_pool_raw: dict[str, np.ndarray],
    inner_plan: Mapping[str, int],
    inner_folds: int,
    config: dict[str, Any],
    outer_fold: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nested = config.get("nested", {})
    configured_grid = nested.get("classical_grids", {}).get(model_name)
    if configured_grid is None:
        configured_grid = config.get("classical", {}).get(model_name, {})
    candidates = _parameter_grid(configured_grid)
    patient_ids = train_pool_raw["patient_ids"]
    prepared_folds = []
    for inner_fold in range(1, inner_folds + 1):
        train_index, validation_index = indices_for_outer_fold(
            patient_ids, inner_plan, inner_fold
        )
        inner_train_raw = subset_arrays(train_pool_raw, train_index)
        inner_validation_raw = subset_arrays(train_pool_raw, validation_index)
        preprocessor = FoldPreprocessor.fit(
            inner_train_raw,
            patient_id_hash(inner_train_raw["patient_ids"]),
        )
        prepared_folds.append(
            (
                inner_fold,
                inner_train_raw,
                preprocessor.transform(inner_train_raw),
                preprocessor.transform(inner_validation_raw),
                inner_validation_raw,
            )
        )
    candidate_records = []
    for candidate_index, parameters in enumerate(candidates):
        fold_scores = []
        fold_details = []
        for (
            inner_fold,
            inner_train_raw,
            inner_train,
            inner_validation,
            inner_validation_raw,
        ) in prepared_folds:
            candidate_seed = (
                seed
                + outer_fold * 10_000
                + candidate_index * 100
                + inner_fold
            )
            model = build_classical_model(
                model_name,
                seed=candidate_seed,
                **parameters,
            ).fit(inner_train)
            predictions = model.predict(inner_validation)
            score, details = _joint_selection_score(
                inner_train_raw,
                inner_validation_raw,
                predictions,
            )
            if not np.isfinite(score):
                score = float("inf")
            fold_scores.append(score)
            fold_details.append(
                {
                    "inner_fold": inner_fold,
                    "n_train": len(inner_train["patient_ids"]),
                    "n_validation": len(inner_validation["patient_ids"]),
                    "score": score,
                }
                | details
            )
        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "candidate_sha256_12": _candidate_hash(parameters),
                "parameters": parameters,
                "mean_joint_normalized_mae": float(np.mean(fold_scores)),
                "inner_fold_scores": fold_details,
            }
        )
    selected = min(
        candidate_records,
        key=lambda record: (
            record["mean_joint_normalized_mae"],
            record["candidate_index"],
        ),
    )
    return dict(selected["parameters"]), candidate_records


def _select_deep_epochs(
    model_name: str,
    train_pool_raw: dict[str, np.ndarray],
    schema: FeatureSchema,
    inner_plan: Mapping[str, int],
    inner_folds: int,
    config: dict[str, Any],
    outer_fold: int,
    seed: int,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], float]:
    patient_ids = train_pool_raw["patient_ids"]
    lac_config = _model_config(
        schema,
        config.get("model", {}),
        model_name,
        config.get("model_v2", {}),
        config.get("model_v21", {}),
    )
    records = []
    started = time.perf_counter()
    for inner_fold in range(1, inner_folds + 1):
        train_index, validation_index = indices_for_outer_fold(
            patient_ids, inner_plan, inner_fold
        )
        inner_train_raw = subset_arrays(train_pool_raw, train_index)
        inner_validation_raw = subset_arrays(train_pool_raw, validation_index)
        preprocessor = FoldPreprocessor.fit(
            inner_train_raw,
            patient_id_hash(inner_train_raw["patient_ids"]),
        )
        inner_train = preprocessor.transform(inner_train_raw)
        inner_validation = preprocessor.transform(inner_validation_raw)
        fit_seed = seed + outer_fold * 10_000 + inner_fold
        seed_everything(fit_seed)
        model = build_torch_model(model_name, lac_config).to(device)
        model, history = train_model(
            model,
            inner_train,
            inner_validation,
            config,
            device,
            fit_seed,
        )
        predictions, _, _ = predict(
            model,
            _loader(
                inner_validation,
                int(config.get("batch_size", 32)),
                False,
            ),
            device,
        )
        score, details = _joint_selection_score(
            inner_train_raw,
            inner_validation_raw,
            predictions,
        )
        records.append(
            {
                "inner_fold": inner_fold,
                "seed": fit_seed,
                "n_train": len(train_index),
                "n_validation": len(validation_index),
                "train_patient_hash": patient_id_hash(
                    inner_train_raw["patient_ids"]
                ),
                "validation_patient_hash": patient_id_hash(
                    inner_validation_raw["patient_ids"]
                ),
                "best_epoch": max(1, int(history["best_epoch"])),
                "epochs_ran": int(history["epochs_ran"]),
                "best_validation_loss": float(
                    np.min(history["validation"])
                ),
                "selection_score": score,
                "target_scaler": history.get("target_scaler"),
                "sampling": history.get("sampling"),
            }
            | details
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected_epochs = max(
        1,
        int(np.rint(np.median([record["best_epoch"] for record in records]))),
    )
    return selected_epochs, records, time.perf_counter() - started


def run_nested_cross_validation(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    outer_fold_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Patient-level outer fivefold with a patient-level inner fivefold."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    outer_folds = int(config.get("num_folds", 5))
    inner_folds = int(config.get("nested", {}).get("inner_folds", 5))
    if outer_folds != 5 or inner_folds != 5:
        raise ValueError("Formal real-data protocol requires outer 5 × inner 5 folds")
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    patient_ids = np.asarray([str(value) for value in arrays["patient_ids"]])
    outer_plan = dict(
        outer_fold_assignments
        or build_patient_fold_plan(patient_ids, outer_folds, seed)
    )
    validate_patient_fold_plan(patient_ids, outer_plan, outer_folds)
    outer_checksum = fold_plan_checksum(outer_plan)
    outer_manifest = save_patient_fold_plan(
        outer_plan,
        output / "patient_outer_fold_plan.csv",
        seed=seed,
        n_folds=outer_folds,
    )
    (output / "training_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    model_name = str(config["model_name"])
    all_predictions = []
    fold_records = []
    fold_metrics = []
    fold_change_metrics = []
    for outer_fold in range(1, outer_folds + 1):
        outer_train_index, outer_test_index = indices_for_outer_fold(
            patient_ids, outer_plan, outer_fold
        )
        train_pool_raw = subset_arrays(arrays, outer_train_index)
        test_raw = subset_arrays(arrays, outer_test_index)
        train_ids = set(str(value) for value in train_pool_raw["patient_ids"])
        test_ids = set(str(value) for value in test_raw["patient_ids"])
        if train_ids & test_ids:
            raise RuntimeError("Patient leakage across outer folds")
        fold_dir = output / f"fold_{outer_fold}"
        fold_dir.mkdir(exist_ok=True)
        inner_seed = seed + outer_fold * 1_000
        inner_plan = build_patient_fold_plan(
            train_pool_raw["patient_ids"],
            n_folds=inner_folds,
            seed=inner_seed,
        )
        inner_manifest = _save_inner_plan(
            train_pool_raw,
            inner_plan,
            fold_dir,
            inner_seed,
            inner_folds,
        )
        outer_preprocessor = FoldPreprocessor.fit(
            train_pool_raw,
            patient_id_hash(train_pool_raw["patient_ids"]),
        )
        outer_train = outer_preprocessor.transform(train_pool_raw)

        selected_epochs = None
        selected_parameters: dict[str, Any] | None = None
        inner_records: list[dict[str, Any]] = []
        selection_seconds = 0.0
        if model_name == "persistence":
            model = None
            refit_history: dict[str, Any] = {"not_required": True}
            refit_started = time.perf_counter()
        elif model_name in CLASSICAL_MODELS:
            selection_started = time.perf_counter()
            selected_parameters, candidate_records = (
                _select_classical_parameters(
                    model_name,
                    train_pool_raw,
                    inner_plan,
                    inner_folds,
                    config,
                    outer_fold,
                    seed,
                )
            )
            selection_seconds = time.perf_counter() - selection_started
            inner_records = candidate_records
            refit_started = time.perf_counter()
            model = build_classical_model(
                model_name,
                seed=seed + outer_fold * 10_000 + 999,
                **selected_parameters,
            ).fit(outer_train)
            refit_history = {
                "fit_on_complete_outer_training_pool": True,
                "selected_parameters": selected_parameters,
            }
        else:
            selected_epochs, inner_records, selection_seconds = (
                _select_deep_epochs(
                    model_name,
                    train_pool_raw,
                    schema,
                    inner_plan,
                    inner_folds,
                    config,
                    outer_fold,
                    seed,
                    device,
                )
            )
            lac_config = _model_config(
                schema,
                config.get("model", {}),
                model_name,
                config.get("model_v2", {}),
                config.get("model_v21", {}),
            )
            seed_everything(seed + outer_fold * 10_000 + 999)
            model = build_torch_model(model_name, lac_config).to(device)
            refit_started = time.perf_counter()
            model, refit_history = fit_model_fixed_epochs(
                model,
                outer_train,
                config,
                device,
                selected_epochs,
                seed + outer_fold * 10_000 + 999,
            )
        refit_seconds = time.perf_counter() - refit_started
        test = outer_preprocessor.transform(test_raw)
        inference_started = time.perf_counter()
        if model_name == "persistence":
            predictions = test["baseline"].copy()
            targets = test["targets"]
            prediction_ids = [
                str(value) for value in test["patient_ids"]
            ]
        elif model_name in CLASSICAL_MODELS:
            predictions = model.predict(test)
            targets = test["targets"]
            prediction_ids = [
                str(value) for value in test["patient_ids"]
            ]
        else:
            predictions, targets, prediction_ids = predict(
                model,
                _loader(
                    test,
                    int(config.get("batch_size", 32)),
                    False,
                ),
                device,
            )
        inference_seconds = time.perf_counter() - inference_started
        metrics = regression_metrics(targets, predictions)
        change_metrics = change_space_metrics(
            targets,
            predictions,
            test["baseline"],
        )
        fold_metrics.append({"fold": outer_fold} | metrics)
        fold_change_metrics.append({"fold": outer_fold} | change_metrics)
        frame = pd.DataFrame(
            {
                "patient_id": prediction_ids,
                "outer_fold": outer_fold,
                "baseline_tbr": test["baseline"][:, 0],
                "baseline_cac": test["baseline"][:, 1],
                "true_tbr": targets[:, 0],
                "pred_tbr": predictions[:, 0],
                "true_cac": targets[:, 1],
                "pred_cac": predictions[:, 1],
            }
        )
        frame["true_delta_tbr"] = (
            frame["true_tbr"] - frame["baseline_tbr"]
        )
        frame["pred_delta_tbr"] = (
            frame["pred_tbr"] - frame["baseline_tbr"]
        )
        frame["true_delta_log_cac"] = np.log1p(
            np.maximum(frame["true_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        frame["pred_delta_log_cac"] = np.log1p(
            np.maximum(frame["pred_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        all_predictions.append(frame)
        outer_preprocessor.save(fold_dir / "preprocessor.json")
        split_manifest = {
            "outer_test_fold": outer_fold,
            "n_outer_train": len(outer_train_index),
            "n_outer_test": len(outer_test_index),
            "outer_train_patient_hash": patient_id_hash(
                train_pool_raw["patient_ids"]
            ),
            "outer_test_patient_hash": patient_id_hash(
                test_raw["patient_ids"]
            ),
            "outer_fold_plan_checksum": outer_checksum,
            "inner_folds": inner_folds,
            "inner_fold_plan": inner_manifest,
            "test_fold_used_for_preprocessing": False,
            "test_fold_used_for_inner_selection": False,
            "test_fold_used_for_training": False,
            "selected_epochs": selected_epochs,
            "selected_classical_parameters": selected_parameters,
        }
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2),
            encoding="utf-8",
        )
        (fold_dir / "inner_selection.json").write_text(
            json.dumps(inner_records, indent=2),
            encoding="utf-8",
        )
        (fold_dir / "refit_history.json").write_text(
            json.dumps(refit_history, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "model_name": model_name,
            "model_config": (
                model.config.to_dict()
                if model is not None
                and hasattr(model, "config")
                and hasattr(model.config, "to_dict")
                else None
            ),
            "schema": schema.to_dict(),
            "training_config": config,
            "seed": seed,
            "outer_fold": outer_fold,
            "outer_fold_plan_checksum": outer_checksum,
            "selected_epochs": selected_epochs,
            "selected_classical_parameters": selected_parameters,
            "git_revision": _git_revision(),
            "runtime_versions": _runtime_versions(),
            "cv_protocol": "patient-level outer 5 x inner 5 nested CV",
        }
        if model_name in CLASSICAL_MODELS:
            with (fold_dir / "model.pkl").open("wb") as handle:
                pickle.dump(model, handle)
            (fold_dir / "model_meta.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        elif model_name != "persistence":
            torch.save(
                {"state_dict": model.state_dict()} | metadata,
                fold_dir / "model.pt",
            )
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "n_train": len(outer_train_index),
                "n_test": len(outer_test_index),
                "selected_epochs": selected_epochs,
                "selected_classical_parameters": (
                    json.dumps(selected_parameters, sort_keys=True)
                    if selected_parameters is not None
                    else None
                ),
                "parameter_count": _parameter_count(model),
                "selection_seconds": selection_seconds,
                "refit_seconds": refit_seconds,
                "inference_seconds": inference_seconds,
            }
            | metrics
            | change_metrics
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(all_predictions, ignore_index=True)
    if len(oof) != len(patient_ids) or oof["patient_id"].nunique() != len(
        patient_ids
    ):
        raise RuntimeError("Nested OOF predictions must contain each patient once")
    if set(oof["patient_id"]) != set(patient_ids):
        raise RuntimeError("Nested OOF patient identifiers do not match the cohort")
    oof = oof.sort_values("patient_id").reset_index(drop=True)
    oof.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(
        output / "outer_fold_metrics.csv",
        index=False,
    )
    pooled_targets = oof[["true_tbr", "true_cac"]].to_numpy()
    pooled_predictions = oof[["pred_tbr", "pred_cac"]].to_numpy()
    pooled_baseline = oof[["baseline_tbr", "baseline_cac"]].to_numpy()
    pooled_metrics = regression_metrics(pooled_targets, pooled_predictions)
    pooled_change = change_space_metrics(
        pooled_targets,
        pooled_predictions,
        pooled_baseline,
    )
    bootstrap_replicates = int(config.get("bootstrap_replicates", 2000))
    metric_ci = bootstrap_metrics(
        pooled_targets,
        pooled_predictions,
        n_bootstrap=bootstrap_replicates,
        seed=seed,
    )
    change_ci = bootstrap_change_metrics(
        pooled_targets,
        pooled_predictions,
        pooled_baseline,
        n_bootstrap=bootstrap_replicates,
        seed=seed,
    )
    metric_names = list(pooled_metrics)
    change_names = list(pooled_change)
    summary = {
        "model_name": model_name,
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
        "patient_count": len(patient_ids),
        "cv_protocol": {
            "level": "patient",
            "outer_folds": outer_folds,
            "inner_folds_within_each_outer_training_pool": inner_folds,
            "outer_training_fraction": "4/5",
            "outer_test_fraction": "1/5",
            "deep_hyperparameters": "prespecified before real-data evaluation",
            "deep_epoch_selection": (
                "median best epoch from five inner folds, followed by refit "
                "on all four outer training folds"
            ),
            "classical_hyperparameter_selection": (
                "minimum mean joint normalized residual MAE across five inner folds"
            ),
            "outer_test_used_only_once_for_final_evaluation": True,
            "outer_fold_plan_checksum": outer_checksum,
            "outer_fold_plan": outer_manifest,
        },
        "fold_records": fold_records,
        "mean_outer_fold_metrics": {
            name: float(
                np.nanmean([record[name] for record in fold_records])
            )
            for name in metric_names + change_names
        },
        "outer_fold_standard_deviation": {
            name: float(
                np.nanstd(
                    [record[name] for record in fold_records],
                    ddof=1,
                )
            )
            for name in metric_names + change_names
        },
        "pooled_oof_metrics": pooled_metrics,
        "pooled_change_metrics": pooled_change,
        "bootstrap_95_ci": metric_ci,
        "change_bootstrap_95_ci": change_ci,
        "bootstrap_replicates": bootstrap_replicates,
        "external_validation_status": "not_applicable_cohort_used_for_development",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
