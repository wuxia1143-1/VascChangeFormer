from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.lac_v30 import LACiTransformerV30
from .dataset import model_inputs
from .trainer import (
    _balanced_sampling_weights,
    _fit_target_scaler,
    _loader,
    pretrain_masked,
)


_TBR_PREFIXES = (
    "token_projection", "variable_embedding", "patch_embedding", "encoder",
    "encoder_norm", "static_context", "baseline_context_i", "adapter_i",
    "patient_adapter_gate_i", "update_gate_i", "update_candidate_i", "init_i",
    "pool_i", "head_i", "reconstruction_head",
)
_CAC_PREFIXES = (
    "cac_token_projection", "cac_variable_embedding", "cac_patch_embedding",
    "cac_encoder", "cac_encoder_norm", "cac_static_context",
    "baseline_context_c", "adapter_c", "patient_adapter_gate_c",
    "update_gate_c", "update_candidate_c", "init_c", "pool_c", "head_c",
    "cac_progression_head", "cac_magnitude_head",
)
_LAG_PREFIXES = (
    "lag_adapter_i", "ic_query", "ic_key", "ic_value_down", "ic_value_up",
    "ic_time_score", "ic_log_time_decay", "ic_gate", "pool_c_lag",
    "lag_residual_head", "lag_residual_gate", "interval_treatment_score",
    "interval_treatment_encoder", "current_treatment_encoder",
)


def encode_stage_epochs(tbr: int, cac: int, lag: int) -> int:
    if not (0 <= tbr < 100 and 0 <= cac < 100 and 0 <= lag < 100):
        raise ValueError("Each staged epoch count must be in [0, 99]")
    return int(tbr * 10000 + cac * 100 + lag)


def decode_stage_epochs(code: int) -> tuple[int, int, int]:
    code = int(code)
    return code // 10000, (code // 100) % 100, code % 100


def _set_stage(model: LACiTransformerV30, stage: str) -> None:
    prefixes = {
        "tbr": _TBR_PREFIXES,
        "cac": _CAC_PREFIXES,
        "lag": _LAG_PREFIXES,
    }[stage]
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        if not model.config.phenotype_adapters and (
            name.startswith("adapter_") or name.startswith("patient_adapter_gate_")
        ):
            trainable = False
        if not model.config.treatment_conditioning and name.startswith(
            ("interval_treatment", "current_treatment")
        ):
            trainable = False
        if stage == "lag" and (
            not model.config.coupling_enabled
            or not model.config.lag_residual_enabled
        ):
            trainable = False
        if stage == "cac" and not model.config.separate_cac_encoder and name.startswith(
            ("cac_token", "cac_variable", "cac_patch", "cac_encoder", "cac_static")
        ):
            trainable = False
        parameter.requires_grad_(trainable)


def _restore_declared_trainable(model: LACiTransformerV30) -> None:
    prefixes = _TBR_PREFIXES + _CAC_PREFIXES + _LAG_PREFIXES
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        if not model.config.phenotype_adapters and (
            name.startswith("adapter_") or name.startswith("patient_adapter_gate_")
        ):
            trainable = False
        if not model.config.treatment_conditioning and name.startswith(
            ("interval_treatment", "current_treatment")
        ):
            trainable = False
        if (not model.config.coupling_enabled or not model.config.lag_residual_enabled) and name.startswith(
            _LAG_PREFIXES
        ):
            trainable = False
        parameter.requires_grad_(trainable)


def _targets(
    model: LACiTransformerV30,
    baseline: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_tbr = targets[:, 0] - baseline[:, 0]
    target_cac = torch.log1p(targets[:, 1].clamp_min(0)) - torch.log1p(
        baseline[:, 1].clamp_min(0)
    )
    center = model.target_center
    scale = model.target_scale.clamp_min(1e-6)
    return (
        (target_tbr - center[0]) / scale[0],
        (target_cac - center[1]) / scale[1],
    )


def _prediction(
    model: LACiTransformerV30,
    value: torch.Tensor,
    task: int,
) -> torch.Tensor:
    return (value - model.target_center[task]) / model.target_scale[task].clamp_min(1e-6)


def _phase_loss(
    model: LACiTransformerV30,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    stage: str,
    config: dict[str, Any],
    tail_threshold: float,
) -> torch.Tensor:
    options = dict(config.get("loss_v30", {}))
    target_tbr, target_cac = _targets(model, baseline, targets)
    if stage == "tbr":
        prediction = _prediction(model, output["delta_tbr"], 0)
        mse_weight = float(options.get("tbr_mse_weight", 0.05))
        return (1.0 - mse_weight) * F.l1_loss(
            prediction, target_tbr
        ) + mse_weight * F.mse_loss(prediction, target_tbr)
    central = _prediction(model, output["delta_log_cac_central"], 1)
    huber_delta = float(options.get("huber_delta", 1.0))
    if stage == "cac":
        mse_weight = float(options.get("cac_mse_weight", 0.10))
        central_loss = (
            (1.0 - mse_weight)
            * F.huber_loss(central, target_cac, delta=huber_delta)
            + mse_weight * F.mse_loss(central, target_cac)
        )
        raw_target = (
            torch.log1p(targets[:, 1].clamp_min(0))
            - torch.log1p(baseline[:, 1].clamp_min(0))
        )
        progression = F.binary_cross_entropy_with_logits(
            output["cac_progression_logit"],
            (raw_target > float(options.get("progression_threshold", 0.05))).to(
                targets.dtype
            ),
        )
        magnitude = F.smooth_l1_loss(
            output["cac_change_magnitude"], raw_target.abs()
        )
        return (
            central_loss
            + float(options.get("progression_weight", 0.05)) * progression
            + float(options.get("magnitude_weight", 0.025)) * magnitude
        )
    final = _prediction(model, output["delta_log_cac_raw_final"], 1)
    raw_target = (
        torch.log1p(targets[:, 1].clamp_min(0))
        - torch.log1p(baseline[:, 1].clamp_min(0))
    )
    tail = (raw_target.abs() >= tail_threshold).to(targets.dtype)
    weight = 1.0 + float(options.get("tail_weight", 0.50)) * tail
    huber = F.huber_loss(
        final, target_cac, delta=huber_delta, reduction="none"
    )
    mse = torch.square(final - target_cac)
    mse_weight = float(options.get("lag_mse_weight", 0.15))
    regression = torch.mean(weight * ((1.0 - mse_weight) * huber + mse_weight * mse))
    direction = F.binary_cross_entropy_with_logits(
        final, (target_cac > 0).to(targets.dtype)
    )
    sparsity = output["lag_residual_gate"].mean()
    correction = torch.square(
        _prediction(model, output["lag_correction"], 1)
        - _prediction(model, torch.zeros_like(output["lag_correction"]), 1)
    ).mean()
    return (
        regression
        + float(options.get("direction_weight", 0.025)) * direction
        + float(options.get("gate_sparsity_weight", 0.002)) * sparsity
        + float(options.get("correction_l2_weight", 0.001)) * correction
    )


def _phase_fit(
    model: LACiTransformerV30,
    stage: str,
    train_loader,
    val_loader,
    config: dict[str, Any],
    device: torch.device,
    max_epochs: int,
    fixed_epochs: int | None,
    tail_threshold: float,
) -> tuple[int, dict[str, Any]]:
    _set_stage(model, stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or max_epochs <= 0:
        return 0, {"stage": stage, "epochs_ran": 0, "best_epoch": 0}
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    limit = int(fixed_epochs if fixed_epochs is not None else max_epochs)
    history = {"stage": stage, "train": [], "validation": [], "best_epoch": limit, "epochs_ran": 0}
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, limit + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            output = model(**model_inputs(batch))
            loss = _phase_loss(
                model, output, batch.baseline, batch.targets, stage, config, tail_threshold
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config.get("grad_clip", 1.0)))
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
                loss = _phase_loss(
                    model,
                    model(**model_inputs(batch)),
                    batch.baseline,
                    batch.targets,
                    stage,
                    config,
                    tail_threshold,
                )
                validation_losses.append(float(loss))
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
    model: LACiTransformerV30,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None,
    epoch_code: int | None,
) -> tuple[LACiTransformerV30, dict[str, Any]]:
    scaler = _fit_target_scaler(
        model, train_arrays, {"loss": {"target_standardization": True}}
    )
    sampling = dict(config.get("sampling", {}))
    train_loader = _loader(
        train_arrays, int(config.get("batch_size", 64)), True, sampling, seed
    )
    val_loader = (
        _loader(val_arrays, int(config.get("batch_size", 64)), False)
        if val_arrays is not None else None
    )
    if int(config.get("pretrain_epochs", 0)):
        _set_stage(model, "tbr")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config.get("learning_rate", 3e-4)),
        )
        pretrain_masked(
            model, train_loader, optimizer, device, int(config["pretrain_epochs"])
        )
    raw_cac = np.log1p(np.maximum(train_arrays["targets"][:, 1], 0)) - np.log1p(
        np.maximum(train_arrays["baseline"][:, 1], 0)
    )
    tail_threshold = float(np.quantile(np.abs(raw_cac), 0.90))
    fixed = decode_stage_epochs(epoch_code) if epoch_code is not None else (None, None, None)
    maxima = (
        int(config.get("stage_tbr_epochs", 80)),
        int(config.get("stage_cac_epochs", 80)),
        int(config.get("stage_lag_epochs", 60)),
    )
    stage_epochs = []
    histories = []
    for stage, maximum, fixed_epoch in zip(("tbr", "cac", "lag"), maxima, fixed):
        if stage == "lag" and (
            not model.config.coupling_enabled or not model.config.lag_residual_enabled
        ):
            maximum = 0
            fixed_epoch = 0
        selected, history = _phase_fit(
            model, stage, train_loader, val_loader, config, device,
            maximum, fixed_epoch, tail_threshold,
        )
        stage_epochs.append(selected)
        histories.append(history)
    code = encode_stage_epochs(*stage_epochs)
    _restore_declared_trainable(model)
    return model, {
        "best_epoch": code,
        "stage_epochs": {
            name: value for name, value in zip(("tbr", "cac", "lag"), stage_epochs)
        },
        "stages": histories,
        "target_scaler": scaler,
        "tail_threshold_training_partition": tail_threshold,
        "sampling": (
            _balanced_sampling_weights(train_arrays, sampling)[1]
            if bool(sampling.get("balance_cac_strata", False)) else None
        ),
    }


def train_v30_model(
    model: LACiTransformerV30,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int | None = None,
):
    return _fit(
        model, train_arrays, config, device,
        int(config.get("seed", 2026) if seed is None else seed),
        val_arrays, None,
    )


def fit_v30_fixed_epochs(
    model: LACiTransformerV30,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int | None = None,
):
    return _fit(
        model, train_arrays, config, device,
        int(config.get("seed", 2026) if seed is None else seed),
        None, int(epochs),
    )
