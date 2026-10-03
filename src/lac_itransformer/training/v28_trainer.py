from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from ..experiments.prediction import build_torch_model
from ..models.lac_v28 import LACiTransformerV28
from .dataset import model_inputs
from .trainer import (
    _balanced_sampling_weights,
    _fit_target_scaler,
    _loader,
    pretrain_masked,
)
from .v28_loss import V28Loss


_SHARED_PREFIXES = (
    "token_projection",
    "variable_embedding",
    "patch_embedding",
    "encoder",
    "encoder_norm",
    "static_context",
)


def _shared_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(_SHARED_PREFIXES)
    ]


def _pcgrad_step(
    model: torch.nn.Module,
    losses: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> bool:
    """Coordinate the two task gradients only on the shared encoder.

    Task-specific gradients are left untouched.  A symmetric projection is
    applied only when the flattened shared gradients conflict.
    """

    shared = _shared_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    (0.5 * losses["tbr"]).backward(retain_graph=True)
    tbr_grad = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in shared
    ]
    for parameter in shared:
        parameter.grad = None
    (0.5 * losses["cac"]).backward()
    cac_grad = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in shared
    ]
    paired = [
        (left, right)
        for left, right in zip(tbr_grad, cac_grad)
        if left is not None and right is not None
    ]
    conflict = False
    if paired:
        dot = sum(torch.sum(left * right) for left, right in paired)
        norm_t = sum(torch.sum(left.square()) for left, _ in paired)
        norm_c = sum(torch.sum(right.square()) for _, right in paired)
        conflict = bool(dot.item() < 0)
        coefficient_t = (
            dot / norm_c.clamp_min(1e-12) if conflict else dot.new_zeros(())
        )
        coefficient_c = (
            dot / norm_t.clamp_min(1e-12) if conflict else dot.new_zeros(())
        )
    else:
        coefficient_t = None
        coefficient_c = None
    for parameter, left, right in zip(shared, tbr_grad, cac_grad):
        if left is None:
            parameter.grad = right
        elif right is None:
            parameter.grad = left
        elif conflict:
            parameter.grad = (
                left - coefficient_t * right
                + right - coefficient_c * left
            )
        else:
            parameter.grad = left + right
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return conflict


def _ordinary_step(
    model: torch.nn.Module,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


def _criterion(config: dict[str, Any]) -> V28Loss:
    return V28Loss(**dict(config.get("loss_v28", {})))


def _fit(
    model: LACiTransformerV28,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[LACiTransformerV28, dict[str, Any]]:
    target_scaler = _fit_target_scaler(
        model,
        train_arrays,
        {"loss": {"target_standardization": bool(
            config.get("loss_v28", {}).get("target_standardization", True)
        )}},
    )
    criterion = _criterion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    sampling_config = dict(config.get("sampling", {}))
    train_loader = _loader(
        train_arrays,
        int(config.get("batch_size", 64)),
        True,
        sampling_config,
        seed,
    )
    if int(config.get("pretrain_epochs", 0)):
        pretrain_masked(
            model,
            train_loader,
            optimizer,
            device,
            int(config["pretrain_epochs"]),
        )
    val_loader = (
        _loader(val_arrays, int(config.get("batch_size", 64)), False)
        if val_arrays is not None
        else None
    )
    history: dict[str, Any] = {
        "train": [],
        "validation": [],
        "best_epoch": int(epochs) if val_loader is None else 0,
        "epochs_ran": 0,
        "target_scaler": target_scaler,
        "sampling": (
            _balanced_sampling_weights(train_arrays, sampling_config)[1]
            if bool(sampling_config.get("balance_cac_strata", False))
            else None
        ),
        "gradient_coordination": bool(model.config.gradient_coordination),
        "pcgrad_conflict_batches": 0,
        "pcgrad_total_batches": 0,
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            output = model(**model_inputs(batch))
            losses = criterion(model, output, batch.baseline, batch.targets)
            if model.config.gradient_coordination:
                conflict = _pcgrad_step(
                    model,
                    losses,
                    optimizer,
                    float(config.get("grad_clip", 1.0)),
                )
                history["pcgrad_total_batches"] += 1
                history["pcgrad_conflict_batches"] += int(conflict)
            else:
                _ordinary_step(
                    model,
                    losses["total"],
                    optimizer,
                    float(config.get("grad_clip", 1.0)),
                )
            epoch_losses.append(float(losses["total"].detach()))
        train_loss = float(np.mean(epoch_losses))
        history["train"].append(train_loss)
        history["epochs_ran"] = epoch
        if val_loader is None:
            continue
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                losses = criterion(
                    model,
                    model(**model_inputs(batch)),
                    batch.baseline,
                    batch.targets,
                )
                validation_losses.append(float(losses["total"]))
        validation_loss = float(np.mean(validation_losses))
        history["validation"].append(validation_loss)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= int(config.get("patience", 15)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["pcgrad_conflict_fraction"] = (
        history["pcgrad_conflict_batches"]
        / max(history["pcgrad_total_batches"], 1)
    )
    return model, history


def train_v28_model(
    model: LACiTransformerV28,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int | None = None,
) -> tuple[LACiTransformerV28, dict[str, Any]]:
    fit_seed = int(config.get("seed", 2026) if seed is None else seed)
    return _fit(
        model,
        train_arrays,
        config,
        device,
        int(config.get("epochs", 150)),
        fit_seed,
        val_arrays,
    )


def fit_v28_fixed_epochs(
    model: LACiTransformerV28,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int | None = None,
) -> tuple[LACiTransformerV28, dict[str, Any]]:
    fit_seed = int(config.get("seed", 2026) if seed is None else seed)
    return _fit(
        model,
        train_arrays,
        config,
        device,
        int(epochs),
        fit_seed,
        None,
    )


def build_v28_model(name: str, config: Any) -> LACiTransformerV28:
    return build_torch_model(name, config)
