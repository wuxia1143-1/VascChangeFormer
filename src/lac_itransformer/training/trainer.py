from __future__ import annotations

import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import pickle
import random
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.preprocessing import FoldPreprocessor, subset_arrays
from ..data.schema import FeatureSchema
from ..models.lac import LACConfig, LACiTransformer
from ..models.lac_v2 import LACV2Config, LACiTransformerV2
from ..models.lac_v21 import LACV21Config, LACiTransformerV21
from ..models.lac_v22 import LACV22Config, LACiTransformerV22
from ..experiments.prediction import build_classical_model, build_torch_model
from .dataset import ArrayDataset, collate_batch, model_inputs
from .losses import LACLoss, masked_reconstruction_loss
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def patient_id_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in ids)).encode()).hexdigest()


def _git_revision() -> str:
    try:
        repository_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "uncommitted"


def _runtime_versions() -> dict[str, str]:
    result = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "xgboost"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _balanced_sampling_weights(
    arrays: dict[str, np.ndarray],
    sampling_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if "followup_months" not in arrays:
        raise ValueError(
            "CAC-stratified sampling requires followup_months in prepared arrays"
        )
    baseline_positive = (arrays["baseline"][:, 1] > 0).astype(int)
    window = np.digitize(
        arrays["followup_months"].astype(float),
        [6.0, 12.0, 18.0, 24.0],
        right=True,
    )
    strata = baseline_positive * 5 + window
    unique, counts = np.unique(strata, return_counts=True)
    count_by_stratum = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray(
        [1.0 / np.sqrt(count_by_stratum[int(value)]) for value in strata],
        dtype=np.float64,
    )
    cap_ratio = float(sampling_config.get("max_weight_ratio", 5.0))
    minimum = float(weights.min())
    weights = np.minimum(weights, minimum * cap_ratio)
    return weights, {
        "method": "joint baseline-CAC and follow-up-window square-root inverse frequency",
        "followup_window_boundaries_months": [6, 12, 18, 24],
        "max_weight_ratio": cap_ratio,
        "stratum_counts": {
            str(key): int(value) for key, value in count_by_stratum.items()
        },
    }


def _loader(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    sampling_config: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> DataLoader:
    dataset = ArrayDataset(arrays)
    options = dict(sampling_config or {})
    sampler = None
    if shuffle and bool(options.get("balance_cac_strata", False)):
        weights, _ = _balanced_sampling_weights(arrays, options)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_batch,
    )


def _fit_target_scaler(
    model: torch.nn.Module,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(config.get("loss", {}).get("target_standardization", False)):
        return None
    setter = getattr(model, "set_target_scaler", None)
    if setter is None:
        raise ValueError(
            "Target standardization was requested for a model without a "
            "set_target_scaler method"
        )
    residuals = np.column_stack(
        [
            train_arrays["targets"][:, 0] - train_arrays["baseline"][:, 0],
            np.log1p(np.maximum(train_arrays["targets"][:, 1], 0))
            - np.log1p(np.maximum(train_arrays["baseline"][:, 1], 0)),
        ]
    )
    center = residuals.mean(axis=0)
    scale = residuals.std(axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    setter(center, scale)
    return {
        "fitted_on_training_partition_only": True,
        "target_order": ["delta_tbr", "delta_log_cac"],
        "center": center.tolist(),
        "scale": scale.tolist(),
        "n_training_patients": int(len(residuals)),
    }


def pretrain_masked(
    model: LACiTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    mask_ratio: float = 0.15,
) -> None:
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            selected = (torch.rand_like(batch.mask) < mask_ratio) & batch.mask.bool()
            masked_values = batch.values.masked_fill(selected, 0.0)
            masked_indicator = batch.mask.masked_fill(selected, 0.0)
            inputs = model_inputs(batch) | {"values": masked_values, "mask": masked_indicator}
            output = model(**inputs)
            loss = masked_reconstruction_loss(output, batch.values, selected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    predictions, targets, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(**model_inputs(batch))
        predictions.append(torch.stack([output["endpoint_tbr"], output["endpoint_cac"]], dim=-1).cpu().numpy())
        targets.append(batch.targets.cpu().numpy())
        ids.extend(batch.patient_ids)
    return np.concatenate(predictions), np.concatenate(targets), ids


def train_model(
    model: torch.nn.Module,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    fit_seed = int(config.get("seed", 2026) if seed is None else seed)
    target_scaler = _fit_target_scaler(model, train_arrays, config)
    loss_config = config.get("loss", {})
    criterion = LACLoss(**loss_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    sampling_config = config.get("sampling", {})
    train_loader = _loader(
        train_arrays,
        int(config.get("batch_size", 32)),
        True,
        sampling_config,
        fit_seed,
    )
    val_loader = _loader(val_arrays, int(config.get("batch_size", 32)), False)
    if int(config.get("pretrain_epochs", 0)) and isinstance(
        model, (LACiTransformer, LACiTransformerV2, LACiTransformerV21)
    ):
        pretrain_masked(model, train_loader, optimizer, device, int(config["pretrain_epochs"]))
    history: dict[str, Any] = {
        "train": [],
        "validation": [],
        "best_epoch": 0,
        "epochs_ran": 0,
        "target_scaler": target_scaler,
        "sampling": (
            _balanced_sampling_weights(train_arrays, sampling_config)[1]
            if bool(sampling_config.get("balance_cac_strata", False))
            else None
        ),
    }
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, int(config.get("epochs", 100)) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            losses = criterion(model, model(**model_inputs(batch)), batch.baseline, batch.targets)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            train_losses.append(float(losses["total"].detach()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                losses = criterion(model, model(**model_inputs(batch)), batch.baseline, batch.targets)
                val_losses.append(float(losses["total"]))
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train"].append(train_loss)
        history["validation"].append(val_loss)
        history["epochs_ran"] = epoch
        if val_loss < best_loss - 1e-7:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= int(config.get("patience", 20)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def fit_model_fixed_epochs(
    model: torch.nn.Module,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Refit on the complete outer training pool without touching test data."""
    fit_seed = int(config.get("seed", 2026) if seed is None else seed)
    target_scaler = _fit_target_scaler(model, train_arrays, config)
    criterion = LACLoss(**config.get("loss", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    sampling_config = config.get("sampling", {})
    loader = _loader(
        train_arrays,
        int(config.get("batch_size", 32)),
        True,
        sampling_config,
        fit_seed,
    )
    if int(config.get("pretrain_epochs", 0)) and isinstance(
        model, (LACiTransformer, LACiTransformerV2, LACiTransformerV21)
    ):
        pretrain_masked(model, loader, optimizer, device, int(config["pretrain_epochs"]))
    history: dict[str, Any] = {
        "train": [],
        "epochs": int(epochs),
        "target_scaler": target_scaler,
        "sampling": (
            _balanced_sampling_weights(train_arrays, sampling_config)[1]
            if bool(sampling_config.get("balance_cac_strata", False))
            else None
        ),
    }
    for _ in range(int(epochs)):
        model.train()
        losses_for_epoch = []
        for batch in loader:
            batch = batch.to(device)
            losses = criterion(model, model(**model_inputs(batch)), batch.baseline, batch.targets)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            losses_for_epoch.append(float(losses["total"].detach()))
        history["train"].append(float(np.mean(losses_for_epoch)))
    return model, history


V2_MODEL_NAMES = {
    "lac_v2",
    "lac_v2_no_adapters",
    "lac_v2_no_coupling",
    "lac_v2_forward_only",
    "lac_v2_symmetric",
    "lac_v2_no_treatment",
}
V21_MODEL_NAMES = {"lac_v21"}
V22_MODEL_NAMES = {
    "lac_v22_full",
    "lac_v22_no_adapters",
    "lac_v22_no_coupling",
    "lac_v22_no_treatment",
}


def _model_config(
    schema: FeatureSchema,
    model_options: dict[str, Any],
    model_name: str,
    v2_options: dict[str, Any] | None = None,
    v21_options: dict[str, Any] | None = None,
    v22_options: dict[str, Any] | None = None,
) -> LACConfig | LACV2Config | LACV21Config | LACV22Config:
    common = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
    }
    if model_name in V2_MODEL_NAMES:
        return LACV2Config(**(common | dict(v2_options or {})))
    if model_name in V21_MODEL_NAMES:
        return LACV21Config(**(common | dict(v21_options or {})))
    if model_name in V22_MODEL_NAMES:
        return LACV22Config(**(common | dict(v22_options or {})))
    return LACConfig(
        **(common | model_options),
    )


def _parameter_count(model: Any) -> int | None:
    if isinstance(model, torch.nn.Module):
        return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return None


def _metric_scope(model_name: str) -> tuple[str, ...] | None:
    if model_name == "single_tbr":
        return ("tbr_",)
    if model_name == "single_cac":
        return ("log_cac_", "cac_")
    return None


def _scoped_metrics(
    targets: np.ndarray, predictions: np.ndarray, model_name: str
) -> dict[str, float]:
    metrics = regression_metrics(targets, predictions)
    prefixes = _metric_scope(model_name)
    if prefixes is None:
        return metrics
    return {key: value for key, value in metrics.items() if key.startswith(prefixes)}


def _scoped_change_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    metrics = change_space_metrics(targets, predictions, baseline)
    if model_name == "single_tbr":
        return {
            key: value
            for key, value in metrics.items()
            if key.startswith("delta_tbr_")
        }
    if model_name == "single_cac":
        return {
            key: value
            for key, value in metrics.items()
            if key.startswith("delta_log_cac_")
        }
    return metrics


def run_cross_validation(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    fold_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    n_folds = int(config.get("num_folds", 5))
    patient_ids_array = np.asarray([str(value) for value in arrays["patient_ids"]])
    plan = dict(fold_assignments or build_patient_fold_plan(patient_ids_array, n_folds, seed))
    validate_patient_fold_plan(patient_ids_array, plan, n_folds)
    plan_checksum = fold_plan_checksum(plan)
    plan_manifest = save_patient_fold_plan(
        plan, output / "patient_fold_plan.csv", seed=seed, n_folds=n_folds
    )
    (output / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []
    fold_change_metrics: list[dict[str, float]] = []
    model_options = config.get("model", {})
    v2_model_options = config.get("model_v2", {})
    v21_model_options = config.get("model_v21", {})
    model_name = str(config.get("model_name", "lac_itransformer"))
    classical_names = {"persistence", "elastic_net", "xgboost"}
    fold_records: list[dict[str, Any]] = []
    inner_fraction = float(config.get("inner_validation_fraction", 0.15))
    for fold in range(1, n_folds + 1):
        train_idx, test_idx = indices_for_outer_fold(patient_ids_array, plan, fold)
        train_pool_raw, test_raw = subset_arrays(arrays, train_idx), subset_arrays(arrays, test_idx)
        train_ids = set(str(x) for x in train_pool_raw["patient_ids"])
        test_ids = set(str(x) for x in test_raw["patient_ids"])
        if train_ids & test_ids:
            raise RuntimeError("Patient leakage across folds")
        outer_train_folds = sorted(set(range(1, n_folds + 1)) - {fold})
        lac_config = _model_config(
            schema,
            model_options,
            model_name,
            v2_model_options,
            v21_model_options,
        )
        selection_history: dict[str, Any] = {"not_required": True}
        selected_epochs: int | None = None
        inner_selection: dict[str, Any] | None = None
        selection_seconds = 0.0
        preprocessor = FoldPreprocessor.fit(
            train_pool_raw, patient_id_hash(train_pool_raw["patient_ids"])
        )
        train_data = preprocessor.transform(train_pool_raw)
        if model_name == "persistence":
            refit_started = time.perf_counter()
            model = None
            refit_history = {"not_required": True}
        elif model_name in classical_names:
            refit_started = time.perf_counter()
            classical_options = config.get("classical", {}).get(model_name, {})
            model = build_classical_model(
                model_name, seed=seed + 100 * fold, **classical_options
            ).fit(train_data)
            refit_history = {"fit_on_complete_outer_training_pool": True}
        else:
            calibration_train_idx, calibration_val_idx = inner_calibration_indices(
                train_pool_raw["patient_ids"], inner_fraction, seed + 100 * fold
            )
            calibration_train_raw = subset_arrays(train_pool_raw, calibration_train_idx)
            calibration_val_raw = subset_arrays(train_pool_raw, calibration_val_idx)
            inner_seed = seed + 100 * fold
            inner_selection = {
                "seed": inner_seed,
                "n_train": len(calibration_train_idx),
                "n_validation": len(calibration_val_idx),
                "train_patient_hash": patient_id_hash(calibration_train_raw["patient_ids"]),
                "validation_patient_hash": patient_id_hash(calibration_val_raw["patient_ids"]),
            }
            calibration_preprocessor = FoldPreprocessor.fit(
                calibration_train_raw, patient_id_hash(calibration_train_raw["patient_ids"])
            )
            calibration_train = calibration_preprocessor.transform(calibration_train_raw)
            calibration_val = calibration_preprocessor.transform(calibration_val_raw)
            seed_everything(seed + 100 * fold)
            selection_model = build_torch_model(model_name, lac_config).to(device)
            selection_started = time.perf_counter()
            _, selection_history = train_model(
                selection_model,
                calibration_train,
                calibration_val,
                config,
                device,
                seed + 100 * fold,
            )
            selection_seconds = time.perf_counter() - selection_started
            selected_epochs = max(1, int(selection_history["best_epoch"]))
            del selection_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            seed_everything(seed + 100 * fold + 1)
            model = build_torch_model(model_name, lac_config).to(device)
            refit_started = time.perf_counter()
            model, refit_history = fit_model_fixed_epochs(
                model,
                train_data,
                config,
                device,
                selected_epochs,
                seed + 100 * fold + 1,
            )
        refit_seconds = time.perf_counter() - refit_started
        # The held-out fold is transformed only after model selection and the
        # complete four-fold refit have finished.
        test_data = preprocessor.transform(test_raw)
        inference_started = time.perf_counter()
        if model_name == "persistence":
            prediction = test_data["baseline"].copy()
            target = test_data["targets"]
            patient_ids = [str(value) for value in test_data["patient_ids"]]
        elif model_name in classical_names:
            prediction = model.predict(test_data)
            target = test_data["targets"]
            patient_ids = [str(value) for value in test_data["patient_ids"]]
        else:
            prediction, target, patient_ids = predict(
                model, _loader(test_data, int(config.get("batch_size", 32)), False), device
            )
        inference_seconds = time.perf_counter() - inference_started
        metrics = _scoped_metrics(target, prediction, model_name)
        baseline_test = test_data["baseline"]
        change_metrics = _scoped_change_metrics(
            target, prediction, baseline_test, model_name
        )
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)
        fold_change_metrics.append(
            {"fold": float(fold)} | change_metrics
        )
        true_delta_tbr = target[:, 0] - baseline_test[:, 0]
        pred_delta_tbr = prediction[:, 0] - baseline_test[:, 0]
        true_delta_log_cac = np.log1p(np.maximum(target[:, 1], 0)) - np.log1p(
            np.maximum(baseline_test[:, 1], 0)
        )
        pred_delta_log_cac = np.log1p(
            np.maximum(prediction[:, 1], 0)
        ) - np.log1p(np.maximum(baseline_test[:, 1], 0))
        frame = pd.DataFrame({
            "patient_id": patient_ids,
            "fold": fold,
            "baseline_tbr": baseline_test[:, 0],
            "baseline_cac": baseline_test[:, 1],
            "true_tbr": target[:, 0],
            "pred_tbr": prediction[:, 0],
            "true_cac": target[:, 1],
            "pred_cac": prediction[:, 1],
            "true_delta_tbr": true_delta_tbr,
            "pred_delta_tbr": pred_delta_tbr,
            "true_delta_log_cac": true_delta_log_cac,
            "pred_delta_log_cac": pred_delta_log_cac,
        })
        all_predictions.append(frame)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        preprocessor.save(fold_dir / "preprocessor.json")
        split_manifest = {
            "outer_test_fold": fold,
            "outer_training_folds": outer_train_folds,
            "n_outer_train": len(train_idx),
            "n_outer_test": len(test_idx),
            "outer_train_patient_hash": patient_id_hash(train_pool_raw["patient_ids"]),
            "outer_test_patient_hash": patient_id_hash(test_raw["patient_ids"]),
            "fold_plan_checksum": plan_checksum,
            "test_fold_used_for_preprocessing": False,
            "test_fold_used_for_epoch_selection": False,
            "test_fold_used_for_training": False,
            "selected_epochs": selected_epochs,
            "inner_validation_fraction": inner_fraction if selected_epochs is not None else None,
            "inner_epoch_selection": inner_selection,
        }
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )
        effective_config = getattr(model, "config", lac_config)
        metadata = {
            "model_name": model_name,
            "model_config": effective_config.to_dict(),
            "schema": schema.to_dict(),
            "seed": seed,
            "fold": fold,
            "fold_plan_checksum": plan_checksum,
            "outer_training_folds": outer_train_folds,
            "outer_test_fold": fold,
            "selected_epochs": selected_epochs,
            "git_revision": _git_revision(),
            "runtime_versions": _runtime_versions(),
            "external_validation_locked": model_name in {
                "lac_itransformer",
                "lac_v2",
            },
        }
        if model_name in classical_names:
            if model is not None:
                with (fold_dir / "model.pkl").open("wb") as handle:
                    pickle.dump(model, handle)
            (fold_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            torch.save({"state_dict": model.state_dict()} | metadata, fold_dir / "model.pt")
        (fold_dir / "selection_history.json").write_text(
            json.dumps(selection_history, indent=2), encoding="utf-8"
        )
        (fold_dir / "refit_history.json").write_text(
            json.dumps(refit_history, indent=2), encoding="utf-8"
        )
        fold_record = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "selected_epochs": selected_epochs,
            "parameter_count": _parameter_count(model),
            "selection_seconds": selection_seconds,
            "refit_seconds": refit_seconds,
            "inference_seconds": inference_seconds,
        } | {key: value for key, value in metrics.items() if key != "fold"} | change_metrics
        fold_records.append(fold_record)
    predictions = pd.concat(all_predictions, ignore_index=True)
    if len(predictions) != len(patient_ids_array) or predictions["patient_id"].nunique() != len(patient_ids_array):
        raise RuntimeError("OOF predictions must contain every patient exactly once")
    expected_ids = set(patient_ids_array)
    if set(predictions["patient_id"]) != expected_ids:
        raise RuntimeError("OOF patient identifiers do not match the input cohort")
    predictions = predictions.sort_values("patient_id").reset_index(drop=True)
    predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / "fold_metrics.csv", index=False)
    pooled_targets = predictions[["true_tbr", "true_cac"]].to_numpy()
    pooled_predictions = predictions[["pred_tbr", "pred_cac"]].to_numpy()
    pooled_baseline = predictions[["baseline_tbr", "baseline_cac"]].to_numpy()
    pooled_metrics = _scoped_metrics(pooled_targets, pooled_predictions, model_name)
    pooled_change_metrics = _scoped_change_metrics(
        pooled_targets, pooled_predictions, pooled_baseline, model_name
    )
    bootstrap_replicates = int(config.get("bootstrap_replicates", 2000))
    confidence_intervals = bootstrap_metrics(
        pooled_targets, pooled_predictions, n_bootstrap=bootstrap_replicates, seed=seed
    )
    change_confidence_intervals = bootstrap_change_metrics(
        pooled_targets,
        pooled_predictions,
        pooled_baseline,
        n_bootstrap=bootstrap_replicates,
        seed=seed,
    )
    metric_prefixes = _metric_scope(model_name)
    if metric_prefixes is not None:
        confidence_intervals = {
            key: value
            for key, value in confidence_intervals.items()
            if key.startswith(metric_prefixes)
        }
        if model_name == "single_tbr":
            change_confidence_intervals = {
                key: value
                for key, value in change_confidence_intervals.items()
                if key.startswith("delta_tbr_")
            }
        else:
            change_confidence_intervals = {
                key: value
                for key, value in change_confidence_intervals.items()
                if key.startswith("delta_log_cac_")
            }
    mean_fold_metrics = {
        key: float(np.nanmean([metric[key] for metric in fold_metrics]))
        for key in fold_metrics[0] if key != "fold"
    }
    fold_standard_deviation = {
        key: float(np.nanstd([metric[key] for metric in fold_metrics], ddof=1))
        for key in fold_metrics[0] if key != "fold"
    }
    summary = {
        "model_name": model_name,
        "metric_scope": (
            "tbr_only" if model_name == "single_tbr" else
            "cac_only" if model_name == "single_cac" else "both_endpoints"
        ),
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
        "cv_protocol": {
            "level": "patient",
            "outer_folds": n_folds,
            "outer_training_folds_per_iteration": n_folds - 1,
            "outer_test_folds_per_iteration": 1,
            "test_fold_used_only_for_final_evaluation": True,
            "deep_model_epoch_selection": "inner split within outer training pool, then refit on the complete outer training pool",
            "fold_plan_checksum": plan_checksum,
            "fold_plan_manifest": plan_manifest,
        },
        "fold_metrics": fold_metrics,
        "fold_change_metrics": fold_change_metrics,
        "fold_records": fold_records,
        "mean_metrics": mean_fold_metrics,
        "fold_standard_deviation": fold_standard_deviation,
        "pooled_oof_metrics": pooled_metrics,
        "pooled_change_metrics": pooled_change_metrics,
        "bootstrap_95_ci": confidence_intervals,
        "change_bootstrap_95_ci": change_confidence_intervals,
        "bootstrap_replicates": bootstrap_replicates,
        "external_validation_status": "not_run",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
