from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.lac_v37 import LACiTransformerV37
from . import v31_trainer as base
from .dataset import model_inputs
from .v32_trainer import decode_stage_epochs, encode_stage_epochs


_BASE_PHASE_LOSS = base._phase_loss
_TBR_PREFIXES = base._TBR_PREFIXES + (
    "soft_adapter_i", "soft_adapter_gate_i",
)
_CAC_PREFIXES = base._CAC_PREFIXES + (
    "mechanism_susceptibility",
    "mechanism_burden_coefficient_raw",
    "shared_transfer_adapter",
    "shared_transfer_logit",
    "cac_only_adapter",
    "cac_only_adapter_gate",
    "soft_adapter_c",
    "soft_adapter_gate_c",
    "history_dose_projection",
    "cac_direction_head",
    "cac_magnitude_head",
)
_SHARED_PREFIXES = (
    "token_projection", "variable_embedding", "patch_embedding", "encoder",
    "encoder_norm", "static_context",
)


def _allowed(model: LACiTransformerV37, name: str) -> bool:
    if name.startswith((
        "cac_progression_head", "cac_magnitude_head", "adapter_i", "adapter_c",
        "patient_adapter_gate_i", "patient_adapter_gate_c",
    )):
        return False
    if not model.config.use_private_cac_encoder and name.startswith(
        ("cac_token", "cac_variable", "cac_patch", "cac_encoder", "cac_static")
    ):
        return False
    if not model.config.shared_transfer_enabled and name.startswith("shared_transfer"):
        return False
    if not model.config.cac_residual_adapter and name.startswith("cac_only_adapter"):
        return False
    if not model.config.soft_phenotype_adapters and name.startswith("soft_adapter"):
        return False
    if not model.config.direction_magnitude_enabled and name.startswith(
        ("history_dose_projection", "cac_direction_head", "cac_magnitude_head")
    ):
        return False
    if not model.config.mechanism_aux_enabled and name.startswith("mechanism_"):
        return False
    if (
        not model.config.mechanism_history_enabled
        and name.startswith("mechanism_burden_coefficient_raw")
    ):
        return False
    return True


def _set_stage(model: LACiTransformerV37, stage: str) -> None:
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
        parameter.requires_grad_(
            bool(prefixes) and name.startswith(prefixes) and _allowed(model, name)
        )


def _restore_declared_trainable(model: LACiTransformerV37) -> None:
    prefixes = _TBR_PREFIXES + _CAC_PREFIXES
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes) and _allowed(model, name))


def _loss_components(
    model: LACiTransformerV37,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    stage: str,
    config: dict[str, Any],
    tail_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if stage == "tbr":
        return (
            _BASE_PHASE_LOSS(
                model, output, baseline, targets, stage, config, tail_threshold
            ),
            None,
        )
    if stage == "lag":
        return output["delta_tbr"].sum() * 0.0, None
    options = dict(config.get("loss_v35", {}))
    _, target_cac = base._targets(model, baseline, targets)
    prediction = base._prediction(model, output["delta_log_cac_central"], 1)
    mse_weight = float(options.get("cac_mse_weight", 0.10))
    primary = (
        (1.0 - mse_weight)
        * F.huber_loss(
            prediction,
            target_cac,
            delta=float(options.get("huber_delta", 1.0)),
        )
        + mse_weight * F.mse_loss(prediction, target_cac)
    )
    if model.config.shared_transfer_enabled:
        primary = primary + float(options.get("shared_gate_penalty", 1e-4)) * output[
            "shared_transfer_alpha"
        ].mean()
    if model.config.cac_residual_adapter:
        primary = primary + float(options.get("adapter_gate_penalty", 1e-4)) * output[
            "cac_adapter_gate"
        ].mean()
    raw_target = (
        torch.log1p(targets[:, 1].clamp_min(0))
        - torch.log1p(baseline[:, 1].clamp_min(0))
    )
    if not model.config.direction_magnitude_enabled:
        return primary, None
    threshold = float(options.get("direction_threshold", 0.05))
    direction_target = torch.ones_like(raw_target, dtype=torch.long)
    direction_target = torch.where(
        raw_target < -threshold,
        torch.zeros_like(direction_target),
        direction_target,
    )
    direction_target = torch.where(
        raw_target > threshold,
        torch.full_like(direction_target, 2),
        direction_target,
    )
    direction_loss = F.cross_entropy(
        output["cac_direction_logits"],
        direction_target,
        label_smoothing=float(options.get("direction_label_smoothing", 0.02)),
    )
    nonstable = direction_target != 1
    if bool(nonstable.any()):
        magnitude_index = (direction_target[nonstable] == 2).to(torch.long)
        magnitude_prediction = output["cac_direction_magnitudes"][nonstable].gather(
            1, magnitude_index[:, None]
        ).squeeze(1)
        magnitude_loss = F.smooth_l1_loss(
            magnitude_prediction,
            raw_target[nonstable].abs(),
        )
    else:
        magnitude_loss = raw_target.sum() * 0.0
    distribution_prediction = base._prediction(
        model, output["delta_log_cac_mean"], 1
    )
    distribution_mse_weight = float(
        options.get("distribution_mse_weight", 0.20)
    )
    distribution_loss = (
        (1.0 - distribution_mse_weight)
        * F.huber_loss(
            distribution_prediction,
            target_cac,
            delta=float(options.get("huber_delta", 1.0)),
        )
        + distribution_mse_weight
        * F.mse_loss(distribution_prediction, target_cac)
    )
    auxiliary = (
        float(options.get("direction_weight", 0.40)) * direction_loss
        + float(options.get("magnitude_weight", 0.20)) * magnitude_loss
        + float(options.get("distribution_weight", 0.40)) * distribution_loss
    )
    return primary, auxiliary


def _phase_loss(
    model: LACiTransformerV37,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    stage: str,
    config: dict[str, Any],
    tail_threshold: float,
) -> torch.Tensor:
    primary, auxiliary = _loss_components(
        model, output, baseline, targets, stage, config, tail_threshold
    )
    if auxiliary is None:
        return primary
    return primary + float(model.config.mechanism_gradient_scale) * auxiliary


def _pcgrad_step(
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    scale: float,
) -> bool:
    primary_gradient = torch.autograd.grad(
        primary, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_gradient = torch.autograd.grad(
        auxiliary, parameters, allow_unused=True
    )
    overlap = [
        (left, right)
        for left, right in zip(primary_gradient, auxiliary_gradient)
        if left is not None and right is not None
    ]
    if overlap:
        dot = sum((left * right).sum() for left, right in overlap)
        norm = sum(left.square().sum() for left, _ in overlap).clamp_min(1e-12)
        conflicting = bool(dot.detach().item() < 0)
        projection = dot / norm if conflicting else dot.new_zeros(())
    else:
        conflicting = False
        projection = primary.new_zeros(())
    for parameter, left, right in zip(
        parameters, primary_gradient, auxiliary_gradient
    ):
        if left is None and right is None:
            parameter.grad = None
        elif left is None:
            parameter.grad = float(scale) * right
        elif right is None:
            parameter.grad = left
        else:
            corrected = right - projection * left if conflicting else right
            parameter.grad = left + float(scale) * corrected
    return conflicting


def _phase_fit(
    model: LACiTransformerV37,
    stage: str,
    train_loader,
    val_loader,
    config: dict[str, Any],
    device: torch.device,
    max_epochs: int,
    fixed_epochs: int | None,
    tail_threshold: float,
):
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
    history = {
        "stage": stage,
        "train": [],
        "validation": [],
        "best_epoch": limit,
        "epochs_ran": 0,
        "mechanism_gradient_conflict_fraction": [],
    }
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, limit + 1):
        model.train()
        train_losses = []
        conflicts = []
        for batch in train_loader:
            batch = batch.to(device)
            output = model(**model_inputs(batch))
            primary, auxiliary = _loss_components(
                model, output, batch.baseline, batch.targets, stage, config,
                tail_threshold,
            )
            optimizer.zero_grad(set_to_none=True)
            use_pcgrad = bool(
                stage == "cac"
                and auxiliary is not None
                and model.config.mechanism_gradient_conflict_protection
            )
            if use_pcgrad:
                conflicts.append(
                    _pcgrad_step(
                        primary,
                        auxiliary,
                        parameters,
                        float(model.config.mechanism_gradient_scale),
                    )
                )
                logged = primary.detach() + float(
                    model.config.mechanism_gradient_scale
                ) * auxiliary.detach()
            else:
                loss = primary
                if auxiliary is not None:
                    loss = loss + float(model.config.mechanism_gradient_scale) * auxiliary
                loss.backward()
                logged = loss.detach()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config.get("grad_clip", 1.0))
            )
            optimizer.step()
            train_losses.append(float(logged))
        history["train"].append(float(np.mean(train_losses)))
        history["epochs_ran"] = epoch
        history["mechanism_gradient_conflict_fraction"].append(
            float(np.mean(conflicts)) if conflicts else 0.0
        )
        if val_loader is None:
            continue
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                primary, _ = _loss_components(
                    model,
                    model(**model_inputs(batch)),
                    batch.baseline,
                    batch.targets,
                    stage,
                    config,
                    tail_threshold,
                )
                validation_losses.append(float(primary))
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
    model: LACiTransformerV37,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None,
    epoch_code: int | None,
):
    names = (
        "_set_stage", "_restore_declared_trainable", "_phase_loss", "_phase_fit",
        "encode_stage_epochs", "decode_stage_epochs",
    )
    original = {name: getattr(base, name) for name in names}
    try:
        base._set_stage = _set_stage
        base._restore_declared_trainable = _restore_declared_trainable
        base._phase_loss = _phase_loss
        base._phase_fit = _phase_fit
        base.encode_stage_epochs = encode_stage_epochs
        base.decode_stage_epochs = decode_stage_epochs
        return base._fit(model, train_arrays, config, device, seed, val_arrays, epoch_code)
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def train_v37_model(
    model: LACiTransformerV37,
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


def fit_v37_fixed_epochs(
    model: LACiTransformerV37,
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
