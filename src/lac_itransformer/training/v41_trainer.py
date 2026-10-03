from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.lac_v41 import LACiTransformerV41
from . import v31_trainer as base
from . import v37_trainer as central
from .dataset import model_inputs
from .v32_trainer import decode_stage_epochs, encode_stage_epochs


_HURDLE_PREFIXES = ("v41_",)


def _hurdle_allowed(model: LACiTransformerV41, name: str) -> bool:
    if not model.config.progression_hurdle_enabled:
        return False
    if name.startswith("v41_i2c_adapter") and not model.config.hurdle_i_to_c_enabled:
        return False
    if name.startswith("v41_risk_head") and not model.config.hurdle_risk_gate_enabled:
        return False
    if name.startswith("v41_history_projection") and not (
        model.config.hurdle_history_enabled
    ):
        return False
    return True


def _set_hurdle_stage(model: LACiTransformerV41) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith(_HURDLE_PREFIXES) and _hurdle_allowed(model, name)
        )


def _restore_declared_trainable(model: LACiTransformerV41) -> None:
    central._restore_declared_trainable(model)
    for name, parameter in model.named_parameters():
        if name.startswith(_HURDLE_PREFIXES):
            parameter.requires_grad_(_hurdle_allowed(model, name))


def _hurdle_loss(
    model: LACiTransformerV41,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
    tail_threshold: float,
) -> torch.Tensor:
    options = dict(config.get("loss_v41", {}))
    _, target_cac = base._targets(model, baseline, targets)
    central_prediction = base._prediction(
        model, output["delta_log_cac_central"], 1
    ).detach()
    corrected_prediction = base._prediction(
        model, output["delta_log_cac_mean"], 1
    )
    raw_target = (
        torch.log1p(targets[:, 1].clamp_min(0))
        - torch.log1p(baseline[:, 1].clamp_min(0))
    )
    progression_threshold = float(options.get("progression_threshold", 0.25))
    progression_target = (raw_target > progression_threshold).to(targets.dtype)
    positive_weight = torch.as_tensor(
        float(options.get("progression_positive_weight", 2.0)),
        dtype=targets.dtype,
        device=targets.device,
    )
    risk_loss = F.binary_cross_entropy_with_logits(
        output["v41_progression_logit"],
        progression_target,
        pos_weight=positive_weight,
    )

    raw_central = output["delta_log_cac_central"].detach()
    positive_residual = (raw_target - raw_central).clamp_min(0)
    progression = progression_target > 0
    if bool(progression.any()):
        magnitude_loss = F.smooth_l1_loss(
            output["v41_positive_residual"][progression],
            positive_residual[progression],
            beta=float(options.get("residual_huber_delta", 0.5)),
        )
    else:
        magnitude_loss = raw_target.sum() * 0.0

    tail = (raw_target >= float(tail_threshold)).to(targets.dtype)
    weight = 1.0 + float(options.get("tail_weight", 1.0)) * tail
    error = corrected_prediction - target_cac
    huber = F.huber_loss(
        corrected_prediction,
        target_cac,
        delta=float(options.get("huber_delta", 1.0)),
        reduction="none",
    )
    mse_weight = float(options.get("final_mse_weight", 0.50))
    regression = torch.mean(
        weight * ((1.0 - mse_weight) * huber + mse_weight * error.square())
    )
    gate_sparsity = output["v41_progression_probability"].mean()
    correction_l2 = output["v41_hurdle_correction"].square().mean()
    central_guard = F.relu(
        torch.abs(corrected_prediction - target_cac)
        - torch.abs(central_prediction - target_cac)
    ).mean()
    return (
        regression
        + float(options.get("progression_weight", 0.25)) * risk_loss
        + float(options.get("magnitude_weight", 0.25)) * magnitude_loss
        + float(options.get("gate_sparsity_weight", 0.002)) * gate_sparsity
        + float(options.get("correction_l2_weight", 0.001)) * correction_l2
        + float(options.get("central_guard_weight", 0.05)) * central_guard
    )


def _fit_hurdle_stage(
    model: LACiTransformerV41,
    train_loader,
    val_loader,
    config: dict[str, Any],
    device: torch.device,
    max_epochs: int,
    fixed_epochs: int | None,
    tail_threshold: float,
):
    _set_hurdle_stage(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or max_epochs <= 0:
        return 0, {"stage": "hurdle", "epochs_ran": 0, "best_epoch": 0}
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("hurdle_learning_rate", config.get("learning_rate", 3e-4))),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    limit = int(fixed_epochs if fixed_epochs is not None else max_epochs)
    history = {
        "stage": "hurdle",
        "train": [],
        "validation": [],
        "best_epoch": limit,
        "epochs_ran": 0,
    }
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, limit + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            loss = _hurdle_loss(
                model,
                model(**model_inputs(batch)),
                batch.baseline,
                batch.targets,
                config,
                tail_threshold,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config.get("grad_clip", 1.0))
            )
            optimizer.step()
            train_losses.append(float(loss.detach()))
        history["train"].append(float(np.mean(train_losses)))
        history["epochs_ran"] = epoch
        if val_loader is None:
            continue
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                validation_losses.append(float(_hurdle_loss(
                    model,
                    model(**model_inputs(batch)),
                    batch.baseline,
                    batch.targets,
                    config,
                    tail_threshold,
                )))
        validation = float(np.mean(validation_losses))
        history["validation"].append(validation)
        if validation < best_loss - 1e-7:
            best_loss = validation
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= int(config.get("stage_patience", 10)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return int(history["best_epoch"]), history


def _fit(
    model: LACiTransformerV41,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None,
    epoch_code: int | None,
):
    scaler = base._fit_target_scaler(
        model, train_arrays, {"loss": {"target_standardization": True}}
    )
    sampling = dict(config.get("sampling", {}))
    train_loader = base._loader(
        train_arrays, int(config.get("batch_size", 64)), True, sampling, seed
    )
    val_loader = (
        base._loader(val_arrays, int(config.get("batch_size", 64)), False)
        if val_arrays is not None else None
    )
    raw_cac = np.log1p(np.maximum(train_arrays["targets"][:, 1], 0)) - np.log1p(
        np.maximum(train_arrays["baseline"][:, 1], 0)
    )
    tail_threshold = float(np.quantile(raw_cac, 0.90))
    fixed = (
        decode_stage_epochs(epoch_code)
        if epoch_code is not None else (None, None, None)
    )
    maxima = (
        int(config.get("stage_tbr_epochs", 120)),
        int(config.get("stage_cac_epochs", 120)),
        int(config.get("stage_lag_epochs", 100)),
    )
    stages = []
    selected_epochs = []
    for stage, maximum, fixed_epoch in zip(
        ("tbr", "cac", "hurdle"), maxima, fixed
    ):
        if stage in {"tbr", "cac"}:
            selected, history = central._phase_fit(
                model,
                stage,
                train_loader,
                val_loader,
                config,
                device,
                maximum,
                fixed_epoch,
                tail_threshold,
            )
        else:
            selected, history = _fit_hurdle_stage(
                model,
                train_loader,
                val_loader,
                config,
                device,
                maximum,
                fixed_epoch,
                tail_threshold,
            )
        selected_epochs.append(selected)
        stages.append(history)
    code = encode_stage_epochs(*selected_epochs)
    _restore_declared_trainable(model)
    return model, {
        "best_epoch": code,
        "stage_epochs": dict(zip(("tbr", "cac", "hurdle"), selected_epochs)),
        "stages": stages,
        "target_scaler": scaler,
        "tail_threshold_training_partition": tail_threshold,
        "sampling": (
            base._balanced_sampling_weights(train_arrays, sampling)[1]
            if bool(sampling.get("balance_cac_strata", False)) else None
        ),
    }


def train_v41_model(
    model: LACiTransformerV41,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int | None = None,
):
    base_seed = int(config.get("seed", 2026) if seed is None else seed)
    return _fit(
        model,
        train_arrays,
        config,
        device,
        base_seed + int(config.get("training_seed_offset", 0)),
        val_arrays,
        None,
    )


def fit_v41_fixed_epochs(
    model: LACiTransformerV41,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int | None = None,
):
    base_seed = int(config.get("seed", 2026) if seed is None else seed)
    return _fit(
        model,
        train_arrays,
        config,
        device,
        base_seed + int(config.get("training_seed_offset", 0)),
        None,
        int(epochs),
    )
