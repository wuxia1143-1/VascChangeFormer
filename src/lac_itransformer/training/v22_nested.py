from __future__ import annotations

import hashlib
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
from ..experiments.prediction import build_torch_model
from ..models.lac_v22 import LACV22Config
from .dataset import model_inputs
from .folds import (
    build_patient_fold_plan,
    fold_plan_checksum,
    indices_for_outer_fold,
    inner_calibration_indices,
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
    _parameter_count,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    resolve_device,
    seed_everything,
    train_model,
)
from .v22_calibration import CrossFittedClinicalCalibrator


V22_MODELS = (
    "lac_v22_full",
    "lac_v22_no_adapters",
    "lac_v22_no_coupling",
    "lac_v22_no_treatment",
)


def _config_for_model(
    schema: FeatureSchema,
    config: dict[str, Any],
    model_name: str,
) -> LACV22Config:
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
    } | dict(config.get("model_v22", {}))
    if model_name == "lac_v22_no_adapters":
        options["phenotype_adapters"] = False
    elif model_name == "lac_v22_no_coupling":
        options["coupling_enabled"] = False
    elif model_name == "lac_v22_no_treatment":
        options["treatment_conditioning"] = False
    elif model_name != "lac_v22_full":
        raise KeyError(model_name)
    return LACV22Config(**options)


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v22_no_adapters",
        "coupling": model_name != "lac_v22_no_coupling",
        "treatment": model_name
        not in {"lac_v22_no_coupling", "lac_v22_no_treatment"},
    }


def _residual_to_endpoint(
    baseline: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            baseline[:, 0] + residual[:, 0],
            np.maximum(
                0,
                np.expm1(
                    np.log1p(np.maximum(baseline[:, 1], 0))
                    + residual[:, 1]
                ),
            ),
        ]
    )


@torch.no_grad()
def _predict_base_with_features(
    model: torch.nn.Module,
    transformed: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    model_name: str,
) -> dict[str, Any]:
    model.eval()
    raw_index = {
        str(patient_id): index
        for index, patient_id in enumerate(raw["patient_ids"])
    }
    residual_predictions = []
    targets = []
    baselines = []
    ids: list[str] = []
    tbr_features = []
    cac_features = []
    flags = _feature_flags(model_name)
    for batch in _loader(transformed, batch_size, False):
        batch = batch.to(device)
        output = model(**model_inputs(batch))
        batch_ids = [str(value) for value in batch.patient_ids]
        indices = np.asarray([raw_index[value] for value in batch_ids])
        raw_baseline = raw["baseline"][indices].astype(float)
        followup = raw["followup_months"][indices].astype(float) / 12.0
        observation_density = raw["mask"][indices].mean(axis=(1, 2))
        raw_treatment = raw["treatments"][indices].max(axis=1).astype(float)
        if not flags["treatment"]:
            raw_treatment = np.zeros_like(raw_treatment)
        window = np.digitize(
            raw["followup_months"][indices].astype(float),
            [6, 12, 18, 24],
            right=True,
        )
        window_one_hot = np.eye(5)[window]
        residual = torch.stack(
            [output["delta_tbr"], output["delta_log_cac"]],
            dim=-1,
        ).cpu().numpy()
        progression = torch.sigmoid(
            output["cac_progression_logit"]
        ).cpu().numpy()
        magnitude = output["cac_change_magnitude"].cpu().numpy()
        coupling_gate = output[
            "gate_inflammation_to_calcification"
        ].mean(dim=1).cpu().numpy()
        coupling_gate_max = output[
            "gate_inflammation_to_calcification"
        ].amax(dim=1).cpu().numpy()
        lag_elapsed = (
            output["lag_attention"] * output["lag_time_deltas"]
        ).sum(dim=-1).mean(dim=-1).cpu().numpy()
        adapter_c = output["phenotype_gate_calcification"].mean(
            dim=(1, 2, 3)
        ).cpu().numpy()
        if not flags["coupling"]:
            coupling_gate = np.zeros_like(coupling_gate)
            coupling_gate_max = np.zeros_like(coupling_gate_max)
            lag_elapsed = np.zeros_like(lag_elapsed)
        if not flags["adapter"]:
            adapter_c = np.zeros_like(adapter_c)
        tbr_features.append(
            np.column_stack(
                [
                    residual[:, 0],
                    raw_baseline[:, 0],
                    followup,
                    observation_density,
                ]
            )
        )
        cac_features.append(
            np.column_stack(
                [
                    residual[:, 1],
                    np.log1p(np.maximum(raw_baseline[:, 1], 0)),
                    raw_baseline[:, 0],
                    followup,
                    (raw_baseline[:, 1] > 0).astype(float),
                    window_one_hot,
                    observation_density,
                    raw_treatment,
                    progression,
                    magnitude,
                    coupling_gate,
                    coupling_gate_max,
                    lag_elapsed,
                    adapter_c,
                ]
            )
        )
        residual_predictions.append(residual)
        targets.append(raw["targets"][indices].astype(float))
        baselines.append(raw_baseline)
        ids.extend(batch_ids)
    baseline = np.concatenate(baselines)
    endpoint_targets = np.concatenate(targets)
    residual_targets = np.column_stack(
        [
            endpoint_targets[:, 0] - baseline[:, 0],
            np.log1p(np.maximum(endpoint_targets[:, 1], 0))
            - np.log1p(np.maximum(baseline[:, 1], 0)),
        ]
    )
    return {
        "patient_ids": ids,
        "baseline": baseline,
        "endpoint_targets": endpoint_targets,
        "residual_targets": residual_targets,
        "raw_residual_predictions": np.concatenate(residual_predictions),
        "features": {
            "tbr": np.concatenate(tbr_features),
            "cac": np.concatenate(cac_features),
        },
    }


def _inner_oof_predictions(
    train_pool_raw: dict[str, np.ndarray],
    schema: FeatureSchema,
    inner_plan: Mapping[str, int],
    config: dict[str, Any],
    model_name: str,
    outer_fold: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    patient_ids = train_pool_raw["patient_ids"]
    records = []
    bundles = []
    selected_epochs = []
    model_config = _config_for_model(schema, config, model_name)
    for inner_fold in range(1, 6):
        train_index, validation_index = indices_for_outer_fold(
            patient_ids,
            inner_plan,
            inner_fold,
        )
        inner_train_raw = subset_arrays(train_pool_raw, train_index)
        inner_validation_raw = subset_arrays(
            train_pool_raw,
            validation_index,
        )
        fit_seed = seed + outer_fold * 100_000 + inner_fold * 1_000
        early_train_index, early_validation_index = (
            inner_calibration_indices(
                inner_train_raw["patient_ids"],
                float(config.get("inner_validation_fraction", 0.15)),
                fit_seed,
            )
        )
        early_train_raw = subset_arrays(
            inner_train_raw,
            early_train_index,
        )
        early_validation_raw = subset_arrays(
            inner_train_raw,
            early_validation_index,
        )
        early_preprocessor = FoldPreprocessor.fit(
            early_train_raw,
            patient_id_hash(early_train_raw["patient_ids"]),
        )
        early_train = early_preprocessor.transform(early_train_raw)
        early_validation = early_preprocessor.transform(
            early_validation_raw
        )
        seed_everything(fit_seed)
        selection_model = build_torch_model(
            model_name,
            model_config,
        ).to(device)
        selection_model, selection_history = train_model(
            selection_model,
            early_train,
            early_validation,
            config,
            device,
            fit_seed,
        )
        epochs = max(1, int(selection_history["best_epoch"]))
        selected_epochs.append(epochs)
        del selection_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        inner_preprocessor = FoldPreprocessor.fit(
            inner_train_raw,
            patient_id_hash(inner_train_raw["patient_ids"]),
        )
        inner_train = inner_preprocessor.transform(inner_train_raw)
        inner_validation = inner_preprocessor.transform(
            inner_validation_raw
        )
        refit_seed = fit_seed + 99
        seed_everything(refit_seed)
        refit_model = build_torch_model(model_name, model_config).to(device)
        refit_model, refit_history = fit_model_fixed_epochs(
            refit_model,
            inner_train,
            config,
            device,
            epochs,
            refit_seed,
        )
        bundle = _predict_base_with_features(
            refit_model,
            inner_validation,
            inner_validation_raw,
            int(config.get("batch_size", 64)),
            device,
            model_name,
        )
        bundle["meta_fold"] = np.full(
            len(bundle["patient_ids"]),
            inner_fold,
            dtype=int,
        )
        bundles.append(bundle)
        records.append(
            {
                "inner_fold": inner_fold,
                "selection_seed": fit_seed,
                "refit_seed": refit_seed,
                "n_early_train": len(early_train_index),
                "n_early_validation": len(early_validation_index),
                "n_inner_refit": len(train_index),
                "n_inner_oof": len(validation_index),
                "selected_epochs": epochs,
                "early_train_patient_hash": patient_id_hash(
                    early_train_raw["patient_ids"]
                ),
                "early_validation_patient_hash": patient_id_hash(
                    early_validation_raw["patient_ids"]
                ),
                "inner_oof_patient_hash": patient_id_hash(
                    inner_validation_raw["patient_ids"]
                ),
                "inner_oof_used_for_early_stopping": False,
                "target_scaler": refit_history.get("target_scaler"),
                "sampling": refit_history.get("sampling"),
            }
        )
        del refit_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = {
        "patient_ids": sum(
            [bundle["patient_ids"] for bundle in bundles],
            [],
        ),
        "baseline": np.concatenate(
            [bundle["baseline"] for bundle in bundles]
        ),
        "endpoint_targets": np.concatenate(
            [bundle["endpoint_targets"] for bundle in bundles]
        ),
        "residual_targets": np.concatenate(
            [bundle["residual_targets"] for bundle in bundles]
        ),
        "raw_residual_predictions": np.concatenate(
            [bundle["raw_residual_predictions"] for bundle in bundles]
        ),
        "features": {
            task: np.concatenate(
                [bundle["features"][task] for bundle in bundles]
            )
            for task in ("tbr", "cac")
        },
        "meta_fold": np.concatenate(
            [bundle["meta_fold"] for bundle in bundles]
        ),
    }
    if set(combined["patient_ids"]) != set(
        str(value) for value in patient_ids
    ):
        raise RuntimeError("Inner OOF calibration predictions are incomplete")
    if len(combined["patient_ids"]) != len(set(combined["patient_ids"])):
        raise RuntimeError("Inner OOF calibration patients must be unique")
    return (
        combined,
        records,
        max(1, int(np.rint(np.median(selected_epochs)))),
    )


def run_v22_nested_cross_validation(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if model_name not in V22_MODELS:
        raise KeyError(model_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    device = resolve_device(str(config.get("device", "auto")))
    patient_ids = np.asarray([str(value) for value in arrays["patient_ids"]])
    outer_plan = dict(
        outer_fold_assignments
        or build_patient_fold_plan(patient_ids, 5, seed)
    )
    validate_patient_fold_plan(patient_ids, outer_plan, 5)
    outer_checksum = fold_plan_checksum(outer_plan)
    outer_manifest = save_patient_fold_plan(
        outer_plan,
        output / "patient_outer_fold_plan.csv",
        seed=seed,
        n_folds=5,
    )
    model_training_config = dict(config) | {"model_name": model_name}
    (output / "training_config.json").write_text(
        json.dumps(model_training_config, indent=2),
        encoding="utf-8",
    )
    all_predictions = []
    fold_records = []
    model_config = _config_for_model(schema, config, model_name)
    for outer_fold in range(1, 6):
        fold_started = time.perf_counter()
        train_index, test_index = indices_for_outer_fold(
            patient_ids,
            outer_plan,
            outer_fold,
        )
        train_pool_raw = subset_arrays(arrays, train_index)
        test_raw = subset_arrays(arrays, test_index)
        if set(train_pool_raw["patient_ids"]) & set(test_raw["patient_ids"]):
            raise RuntimeError("Outer patient leakage")
        fold_dir = output / f"fold_{outer_fold}"
        fold_dir.mkdir(exist_ok=True)
        inner_seed = seed + outer_fold * 1_000
        inner_plan = build_patient_fold_plan(
            train_pool_raw["patient_ids"],
            5,
            inner_seed,
        )
        inner_manifest = save_patient_fold_plan(
            inner_plan,
            fold_dir / "inner_patient_fold_plan.csv",
            seed=inner_seed,
            n_folds=5,
        )
        inner_started = time.perf_counter()
        inner_oof, inner_records, selected_epochs = (
            _inner_oof_predictions(
                train_pool_raw,
                schema,
                inner_plan,
                config,
                model_name,
                outer_fold,
                seed,
                device,
            )
        )
        inner_seconds = time.perf_counter() - inner_started
        calibrator = CrossFittedClinicalCalibrator(
            seed=seed + outer_fold * 10_000,
            candidate_weights=tuple(
                config.get("calibration", {}).get(
                    "candidate_weights",
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                )
            ),
            rmse_tolerance=float(
                config.get("calibration", {}).get(
                    "rmse_tolerance",
                    0.02,
                )
            ),
        ).fit(
            inner_oof["features"],
            inner_oof["raw_residual_predictions"],
            inner_oof["residual_targets"],
            inner_oof["meta_fold"],
        )
        (fold_dir / "calibration_audit.json").write_text(
            json.dumps(calibrator.audit(), indent=2),
            encoding="utf-8",
        )
        with (fold_dir / "calibrator.pkl").open("wb") as handle:
            pickle.dump(calibrator, handle)

        outer_preprocessor = FoldPreprocessor.fit(
            train_pool_raw,
            patient_id_hash(train_pool_raw["patient_ids"]),
        )
        outer_train = outer_preprocessor.transform(train_pool_raw)
        outer_test = outer_preprocessor.transform(test_raw)
        final_seed = seed + outer_fold * 100_000 + 99_999
        seed_everything(final_seed)
        model = build_torch_model(model_name, model_config).to(device)
        model, refit_history = fit_model_fixed_epochs(
            model,
            outer_train,
            config,
            device,
            selected_epochs,
            final_seed,
        )
        test_bundle = _predict_base_with_features(
            model,
            outer_test,
            test_raw,
            int(config.get("batch_size", 64)),
            device,
            model_name,
        )
        calibrated_residual = calibrator.predict(
            test_bundle["features"],
            test_bundle["raw_residual_predictions"],
        )
        calibrated_endpoint = _residual_to_endpoint(
            test_bundle["baseline"],
            calibrated_residual,
        )
        raw_endpoint = _residual_to_endpoint(
            test_bundle["baseline"],
            test_bundle["raw_residual_predictions"],
        )
        targets = test_bundle["endpoint_targets"]
        metrics = regression_metrics(targets, calibrated_endpoint)
        change = change_space_metrics(
            targets,
            calibrated_endpoint,
            test_bundle["baseline"],
        )
        raw_metrics = regression_metrics(targets, raw_endpoint)
        raw_change = change_space_metrics(
            targets,
            raw_endpoint,
            test_bundle["baseline"],
        )
        frame = pd.DataFrame(
            {
                "patient_id": test_bundle["patient_ids"],
                "outer_fold": outer_fold,
                "baseline_tbr": test_bundle["baseline"][:, 0],
                "baseline_cac": test_bundle["baseline"][:, 1],
                "true_tbr": targets[:, 0],
                "pred_tbr": calibrated_endpoint[:, 0],
                "true_cac": targets[:, 1],
                "pred_cac": calibrated_endpoint[:, 1],
                "raw_pred_tbr": raw_endpoint[:, 0],
                "raw_pred_cac": raw_endpoint[:, 1],
            }
        )
        frame["true_delta_tbr"] = (
            frame["true_tbr"] - frame["baseline_tbr"]
        )
        frame["pred_delta_tbr"] = (
            frame["pred_tbr"] - frame["baseline_tbr"]
        )
        frame["raw_pred_delta_tbr"] = (
            frame["raw_pred_tbr"] - frame["baseline_tbr"]
        )
        frame["true_delta_log_cac"] = np.log1p(
            np.maximum(frame["true_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        frame["pred_delta_log_cac"] = np.log1p(
            np.maximum(frame["pred_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        frame["raw_pred_delta_log_cac"] = np.log1p(
            np.maximum(frame["raw_pred_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        all_predictions.append(frame)
        outer_preprocessor.save(fold_dir / "preprocessor.json")
        (fold_dir / "inner_selection.json").write_text(
            json.dumps(inner_records, indent=2),
            encoding="utf-8",
        )
        (fold_dir / "refit_history.json").write_text(
            json.dumps(refit_history, indent=2),
            encoding="utf-8",
        )
        split_manifest = {
            "outer_test_fold": outer_fold,
            "n_outer_train": len(train_index),
            "n_outer_test": len(test_index),
            "outer_fold_plan_checksum": outer_checksum,
            "inner_fold_plan": inner_manifest,
            "selected_epochs": selected_epochs,
            "test_fold_used_for_preprocessing": False,
            "test_fold_used_for_inner_base_training": False,
            "test_fold_used_for_calibrator_fit": False,
            "test_fold_used_for_calibrator_weight_selection": False,
            "test_fold_used_for_outer_refit": False,
        }
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2),
            encoding="utf-8",
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "model_config": model.config.to_dict(),
                "schema": schema.to_dict(),
                "training_config": model_training_config,
                "git_revision": _git_revision(),
                "outer_fold": outer_fold,
            },
            fold_dir / "model.pt",
        )
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "selected_epochs": selected_epochs,
                "parameter_count": _parameter_count(model),
                "inner_oof_and_selection_seconds": inner_seconds,
                "total_fold_seconds": time.perf_counter() - fold_started,
                "calibration_weight_tbr": calibrator.weights["tbr"],
                "calibration_weight_cac": calibrator.weights["cac"],
            }
            | metrics
            | change
            | {
                f"raw_{key}": value
                for key, value in (raw_metrics | raw_change).items()
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(all_predictions, ignore_index=True)
    if len(oof) != len(patient_ids) or oof["patient_id"].nunique() != len(
        patient_ids
    ):
        raise RuntimeError("V2.2 OOF must contain every patient exactly once")
    oof = oof.sort_values("patient_id").reset_index(drop=True)
    oof.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(
        output / "outer_fold_metrics.csv",
        index=False,
    )
    pooled_targets = oof[["true_tbr", "true_cac"]].to_numpy()
    pooled_prediction = oof[["pred_tbr", "pred_cac"]].to_numpy()
    pooled_baseline = oof[["baseline_tbr", "baseline_cac"]].to_numpy()
    pooled_metrics = regression_metrics(pooled_targets, pooled_prediction)
    pooled_change = change_space_metrics(
        pooled_targets,
        pooled_prediction,
        pooled_baseline,
    )
    bootstrap_replicates = int(config.get("bootstrap_replicates", 2000))
    summary = {
        "model_name": model_name,
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
        "patient_count": len(patient_ids),
        "parameter_count": int(
            np.median(
                [record["parameter_count"] for record in fold_records]
            )
        ),
        "cv_protocol": {
            "level": "patient",
            "outer_folds": 5,
            "inner_folds": 5,
            "calibration": (
                "cross-fitted from untouched inner OOF predictions; "
                "expert blend selected by grouped fivefold meta-CV"
            ),
            "outer_test_used_only_for_final_evaluation": True,
            "outer_fold_plan_checksum": outer_checksum,
            "outer_fold_plan": outer_manifest,
        },
        "fold_records": fold_records,
        "pooled_oof_metrics": pooled_metrics,
        "pooled_change_metrics": pooled_change,
        "bootstrap_95_ci": bootstrap_metrics(
            pooled_targets,
            pooled_prediction,
            n_bootstrap=bootstrap_replicates,
            seed=seed,
        ),
        "change_bootstrap_95_ci": bootstrap_change_metrics(
            pooled_targets,
            pooled_prediction,
            pooled_baseline,
            n_bootstrap=bootstrap_replicates,
            seed=seed,
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "development_status": "iterative internal development validation",
        "external_validation_status": "not_run",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
