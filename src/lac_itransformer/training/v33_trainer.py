from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.lac_v33 import LACiTransformerV33
from . import v31_trainer as base
from .v32_trainer import decode_stage_epochs, encode_stage_epochs


_BASE_PHASE_LOSS = base._phase_loss
_TBR_PREFIXES = base._TBR_PREFIXES
_CAC_PREFIXES = base._CAC_PREFIXES
_SHARED_PREFIXES = (
    "token_projection", "variable_embedding", "patch_embedding", "encoder",
    "encoder_norm", "static_context",
)
_LAG_PREFIXES = (
    "history_feature_logits", "history_decay_raw", "history_projection",
    "progression_hurdle", "progression_magnitude",
)


def _allowed(model: LACiTransformerV33, name: str, stage: str) -> bool:
    if not model.config.phenotype_adapters and name.startswith(
        ("adapter_", "patient_adapter_gate_")
    ):
        return False
    if stage == "lag" and model.config.history_mode == "none" and name.startswith(
        ("history_feature_logits", "history_decay_raw", "history_projection")
    ):
        return False
    if not model.config.use_time_decay_kernel and name.startswith("history_decay_raw"):
        return False
    if stage == "lag" and not model.config.hurdle_enabled:
        return False
    return True


def _set_stage(model: LACiTransformerV33, stage: str) -> None:
    if stage == "tbr":
        prefixes = _TBR_PREFIXES
    elif stage == "cac":
        prefixes = _CAC_PREFIXES
        if not model.config.isolate_cac_shared_gradient:
            prefixes = prefixes + _SHARED_PREFIXES
    elif stage == "lag":
        prefixes = _LAG_PREFIXES
    else:
        raise KeyError(stage)
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes) and _allowed(model, name, stage)
        if stage == "cac" and not model.config.separate_cac_encoder and name.startswith(
            ("cac_token", "cac_variable", "cac_patch", "cac_encoder", "cac_static")
        ):
            trainable = False
        parameter.requires_grad_(trainable)


def _restore_declared_trainable(model: LACiTransformerV33) -> None:
    prefixes = _TBR_PREFIXES + _CAC_PREFIXES + _LAG_PREFIXES
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        if not model.config.phenotype_adapters and name.startswith(
            ("adapter_", "patient_adapter_gate_")
        ):
            trainable = False
        if model.config.history_mode == "none" and name.startswith(
            ("history_feature_logits", "history_decay_raw", "history_projection")
        ):
            trainable = False
        if not model.config.use_time_decay_kernel and name.startswith(
            "history_decay_raw"
        ):
            trainable = False
        if not model.config.hurdle_enabled and name.startswith(_LAG_PREFIXES):
            trainable = False
        parameter.requires_grad_(trainable)


def _phase_loss(
    model: LACiTransformerV33,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    stage: str,
    config: dict[str, Any],
    tail_threshold: float,
) -> torch.Tensor:
    if stage in {"tbr", "cac"}:
        return _BASE_PHASE_LOSS(
            model, output, baseline, targets, stage, config, tail_threshold
        )
    options = dict(config.get("loss_v33", {}))
    _, target_cac = base._targets(model, baseline, targets)
    final = base._prediction(model, output["delta_log_cac_raw_final"], 1)
    raw_target = (
        torch.log1p(targets[:, 1].clamp_min(0))
        - torch.log1p(baseline[:, 1].clamp_min(0))
    )
    tail = (raw_target.abs() >= tail_threshold).to(targets.dtype)
    weight = 1.0 + float(options.get("tail_weight", 0.50)) * tail
    huber = F.huber_loss(
        final, target_cac, delta=float(options.get("huber_delta", 1.0)),
        reduction="none",
    )
    mse = torch.square(final - target_cac)
    mse_weight = float(options.get("final_mse_weight", 0.20))
    regression = torch.mean(weight * ((1.0 - mse_weight) * huber + mse_weight * mse))

    threshold = float(options.get("progression_threshold", 0.05))
    progression_target = (raw_target > threshold).to(targets.dtype)
    classification = F.binary_cross_entropy_with_logits(
        output["cac_progression_logit"], progression_target
    )
    positive = progression_target > 0
    if torch.any(positive):
        magnitude = F.smooth_l1_loss(
            output["cac_change_magnitude"][positive],
            raw_target[positive].clamp_min(0),
        )
    else:
        magnitude = regression.new_zeros(())
    sparsity = output["lag_residual_gate"].mean()
    return (
        regression
        + float(options.get("progression_weight", 0.15)) * classification
        + float(options.get("magnitude_weight", 0.05)) * magnitude
        + float(options.get("gate_sparsity_weight", 0.0005)) * sparsity
    )


def _fit(
    model: LACiTransformerV33,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None,
    epoch_code: int | None,
):
    names = (
        "_set_stage", "_restore_declared_trainable", "_phase_loss",
        "encode_stage_epochs", "decode_stage_epochs",
    )
    original = {name: getattr(base, name) for name in names}
    try:
        base._set_stage = _set_stage
        base._restore_declared_trainable = _restore_declared_trainable
        base._phase_loss = _phase_loss
        base.encode_stage_epochs = encode_stage_epochs
        base.decode_stage_epochs = decode_stage_epochs
        return base._fit(
            model, train_arrays, config, device, seed, val_arrays, epoch_code
        )
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def train_v33_model(
    model: LACiTransformerV33,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int | None = None,
):
    base_seed = int(config.get("seed", 2026) if seed is None else seed)
    return _fit(
        model, train_arrays, config, device,
        base_seed + int(config.get("training_seed_offset", 0)),
        val_arrays, None,
    )


def fit_v33_fixed_epochs(
    model: LACiTransformerV33,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int | None = None,
):
    base_seed = int(config.get("seed", 2026) if seed is None else seed)
    return _fit(
        model, train_arrays, config, device,
        base_seed + int(config.get("training_seed_offset", 0)),
        None, int(epochs),
    )
