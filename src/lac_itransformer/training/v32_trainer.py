from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..models.lac_v32 import LACiTransformerV32
from . import v31_trainer as base


_TBR_PREFIXES = base._TBR_PREFIXES
_CAC_PREFIXES = base._CAC_PREFIXES
_LAG_PREFIXES = base._LAG_PREFIXES + (
    "direct_inflammation_encoder",
    "teacher_mix_logit",
)


def encode_stage_epochs(tbr: int, cac: int, lag: int) -> int:
    """Base-1000 encoding supports the preregistered 100-200 epoch stages."""
    if not (0 <= tbr < 1000 and 0 <= cac < 1000 and 0 <= lag < 1000):
        raise ValueError("Each V3.2 staged epoch count must be in [0, 999]")
    return int(tbr * 1_000_000 + cac * 1_000 + lag)


def decode_stage_epochs(code: int) -> tuple[int, int, int]:
    code = int(code)
    return code // 1_000_000, (code // 1_000) % 1000, code % 1000


def _set_stage(model: LACiTransformerV32, stage: str) -> None:
    prefixes = {"tbr": _TBR_PREFIXES, "cac": _CAC_PREFIXES, "lag": _LAG_PREFIXES}[stage]
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        if not model.config.phenotype_adapters and name.startswith(
            ("adapter_", "patient_adapter_gate_")
        ):
            trainable = False
        if not model.config.treatment_conditioning and name.startswith(
            ("interval_treatment", "current_treatment")
        ):
            trainable = False
        if not model.config.patient_lag_gate and name.startswith("patient_lag_gate"):
            trainable = False
        if not model.config.direct_observed_lag and name.startswith(
            "direct_inflammation_encoder"
        ):
            trainable = False
        if not model.config.tbr_teacher_in_lag and name.startswith(
            ("lag_adapter_i", "teacher_mix_logit")
        ):
            trainable = False
        if stage == "lag" and (
            not model.config.coupling_enabled or not model.config.lag_residual_enabled
        ):
            trainable = False
        if stage == "cac" and not model.config.separate_cac_encoder and name.startswith(
            ("cac_token", "cac_variable", "cac_patch", "cac_encoder", "cac_static")
        ):
            trainable = False
        parameter.requires_grad_(trainable)


def _restore_declared_trainable(model: LACiTransformerV32) -> None:
    prefixes = _TBR_PREFIXES + _CAC_PREFIXES + _LAG_PREFIXES
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        if not model.config.phenotype_adapters and name.startswith(
            ("adapter_", "patient_adapter_gate_")
        ):
            trainable = False
        if not model.config.treatment_conditioning and name.startswith(
            ("interval_treatment", "current_treatment")
        ):
            trainable = False
        if not model.config.patient_lag_gate and name.startswith("patient_lag_gate"):
            trainable = False
        if not model.config.direct_observed_lag and name.startswith(
            "direct_inflammation_encoder"
        ):
            trainable = False
        if not model.config.tbr_teacher_in_lag and name.startswith(
            ("lag_adapter_i", "teacher_mix_logit")
        ):
            trainable = False
        if (
            not model.config.coupling_enabled or not model.config.lag_residual_enabled
        ) and name.startswith(_LAG_PREFIXES):
            trainable = False
        parameter.requires_grad_(trainable)


def _fit(
    model: LACiTransformerV32,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None,
    epoch_code: int | None,
):
    names = ("_set_stage", "_restore_declared_trainable", "encode_stage_epochs", "decode_stage_epochs")
    original = {name: getattr(base, name) for name in names}
    try:
        base._set_stage = _set_stage
        base._restore_declared_trainable = _restore_declared_trainable
        base.encode_stage_epochs = encode_stage_epochs
        base.decode_stage_epochs = decode_stage_epochs
        return base._fit(
            model, train_arrays, config, device, seed, val_arrays, epoch_code
        )
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def train_v32_model(
    model: LACiTransformerV32,
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


def fit_v32_fixed_epochs(
    model: LACiTransformerV32,
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
