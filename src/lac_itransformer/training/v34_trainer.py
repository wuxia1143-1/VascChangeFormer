from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.lac_v34 import LACiTransformerV34
from . import v31_trainer as base
from .v32_trainer import decode_stage_epochs, encode_stage_epochs


_BASE_PHASE_LOSS = base._phase_loss
_TBR_PREFIXES = base._TBR_PREFIXES
_CAC_PREFIXES = base._CAC_PREFIXES + (
    "mechanism_susceptibility",
    "mechanism_burden_coefficient_raw",
)
_SHARED_PREFIXES = (
    "token_projection", "variable_embedding", "patch_embedding", "encoder",
    "encoder_norm", "static_context",
)


def _allowed(model: LACiTransformerV34, name: str) -> bool:
    # V3.4 deliberately uses the V3.3 central CAC regressor only.  The two
    # hurdle heads remain attributes for checkpoint/API compatibility, but
    # must never be counted or optimized as part of this candidate.
    if name.startswith(("cac_progression_head", "cac_magnitude_head")):
        return False
    if not model.config.phenotype_adapters and name.startswith(
        ("adapter_", "patient_adapter_gate_")
    ):
        return False
    if not model.config.mechanism_aux_enabled and name.startswith("mechanism_"):
        return False
    return True


def _set_stage(model: LACiTransformerV34, stage: str) -> None:
    if stage == "tbr":
        prefixes = _TBR_PREFIXES
    elif stage == "cac":
        prefixes = _CAC_PREFIXES
        if not model.config.isolate_cac_shared_gradient:
            prefixes = prefixes + _SHARED_PREFIXES
    elif stage == "lag":
        prefixes = ()
    else:
        raise KeyError(stage)
    for name, parameter in model.named_parameters():
        trainable = bool(prefixes) and name.startswith(prefixes) and _allowed(model, name)
        if stage == "cac" and not model.config.separate_cac_encoder and name.startswith(
            ("cac_token", "cac_variable", "cac_patch", "cac_encoder", "cac_static")
        ):
            trainable = False
        parameter.requires_grad_(trainable)


def _restore_declared_trainable(model: LACiTransformerV34) -> None:
    prefixes = _TBR_PREFIXES + _CAC_PREFIXES
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes) and _allowed(model, name))


def _phase_loss(
    model: LACiTransformerV34,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    stage: str,
    config: dict[str, Any],
    tail_threshold: float,
) -> torch.Tensor:
    if stage == "tbr":
        return _BASE_PHASE_LOSS(
            model, output, baseline, targets, stage, config, tail_threshold
        )
    if stage == "lag":
        return output["delta_tbr"].sum() * 0.0
    options = dict(config.get("loss_v34", {}))
    _, target_cac = base._targets(model, baseline, targets)
    central = base._prediction(model, output["delta_log_cac_central"], 1)
    mse_weight = float(options.get("cac_mse_weight", 0.10))
    regression = (
        (1.0 - mse_weight)
        * F.huber_loss(
            central,
            target_cac,
            delta=float(options.get("huber_delta", 1.0)),
        )
        + mse_weight * F.mse_loss(central, target_cac)
    )
    if not model.config.mechanism_aux_enabled:
        return regression
    raw_target = (
        torch.log1p(targets[:, 1].clamp_min(0))
        - torch.log1p(baseline[:, 1].clamp_min(0))
    )
    progression = (raw_target > float(options.get("progression_threshold", 0.05))).to(
        targets.dtype
    )
    mechanism = F.binary_cross_entropy_with_logits(
        output["cac_progression_logit"], progression
    )
    return regression + float(options.get("mechanism_weight", 0.10)) * mechanism


def _fit(
    model: LACiTransformerV34,
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
        return base._fit(model, train_arrays, config, device, seed, val_arrays, epoch_code)
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def train_v34_model(
    model: LACiTransformerV34,
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


def fit_v34_fixed_epochs(
    model: LACiTransformerV34,
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
