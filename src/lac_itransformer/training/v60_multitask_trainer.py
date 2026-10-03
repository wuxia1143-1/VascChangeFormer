from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from ..models.lac_v60 import LACiTransformerV60
from . import v31_trainer as base
from . import v37_trainer as central
from .dataset import model_inputs
from .v32_trainer import decode_stage_epochs, encode_stage_epochs


JOINT_OBJECTIVES = {"joint_hps_mtl", "joint_shared_adapter_mtl"}
SINGLE_OBJECTIVES = {"single_task_tbr", "single_task_cac"}
SUPPORTED_OBJECTIVES = JOINT_OBJECTIVES | SINGLE_OBJECTIVES


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if finite.any() else float("nan")


def _set_objective_trainable(model: LACiTransformerV60, objective: str) -> None:
    if objective in JOINT_OBJECTIVES:
        prefixes = central._TBR_PREFIXES + central._CAC_PREFIXES
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(prefixes) and central._allowed(model, name)
            )
    elif objective == "single_task_tbr":
        central._set_stage(model, "tbr")
    elif objective == "single_task_cac":
        central._set_stage(model, "cac")
    else:
        raise KeyError(objective)


def _objective_loss(
    model: LACiTransformerV60,
    output: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
    tail_threshold: float,
    objective: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if objective == "single_task_tbr":
        tbr, _ = central._loss_components(
            model, output, baseline, targets, "tbr", config, tail_threshold
        )
        return tbr, {"tbr": float(tbr.detach()), "cac": float("nan")}
    if objective == "single_task_cac":
        cac, auxiliary = central._loss_components(
            model, output, baseline, targets, "cac", config, tail_threshold
        )
        if auxiliary is not None:
            raise RuntimeError("V6 single-task CAC unexpectedly produced an auxiliary loss")
        return cac, {"tbr": float("nan"), "cac": float(cac.detach())}

    tbr, _ = central._loss_components(
        model, output, baseline, targets, "tbr", config, tail_threshold
    )
    cac, auxiliary = central._loss_components(
        model, output, baseline, targets, "cac", config, tail_threshold
    )
    if auxiliary is not None:
        raise RuntimeError("V6 joint MTL unexpectedly produced an auxiliary loss")
    options = dict(config.get("joint_mtl", {}))
    tbr_weight = float(options.get("tbr_weight", 0.5))
    cac_weight = float(options.get("cac_weight", 0.5))
    if tbr_weight < 0.0 or cac_weight < 0.0 or tbr_weight + cac_weight <= 0.0:
        raise ValueError("Joint MTL loss weights must be nonnegative with positive sum")
    normalizer = tbr_weight + cac_weight
    loss = (tbr_weight * tbr + cac_weight * cac) / normalizer
    return loss, {"tbr": float(tbr.detach()), "cac": float(cac.detach())}


def _encoded_epoch(objective: str, selected_epoch: int) -> int:
    if not 0 <= int(selected_epoch) < 100:
        raise ValueError("Selected epoch must be in [0, 99]")
    if objective == "single_task_cac":
        return encode_stage_epochs(0, int(selected_epoch), 0)
    return encode_stage_epochs(int(selected_epoch), 0, 0)


def _fixed_epoch(objective: str, epoch_code: int | None) -> int | None:
    if epoch_code is None:
        return None
    tbr, cac, _ = decode_stage_epochs(int(epoch_code))
    return int(cac if objective == "single_task_cac" else tbr)


def _fit(
    model: LACiTransformerV60,
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    val_arrays: dict[str, np.ndarray] | None,
    epoch_code: int | None,
):
    objective = str(model.config.training_objective)
    if objective not in SUPPORTED_OBJECTIVES:
        raise KeyError(objective)
    scaler = base._fit_target_scaler(
        model, train_arrays, {"loss": {"target_standardization": True}}
    )
    sampling = dict(config.get("sampling", {}))
    train_loader = base._loader(
        train_arrays, int(config.get("batch_size", 64)), True, sampling, seed
    )
    val_loader = (
        base._loader(val_arrays, int(config.get("batch_size", 64)), False)
        if val_arrays is not None
        else None
    )
    raw_cac = np.log1p(np.maximum(train_arrays["targets"][:, 1], 0)) - np.log1p(
        np.maximum(train_arrays["baseline"][:, 1], 0)
    )
    tail_threshold = float(np.quantile(np.abs(raw_cac), 0.90))
    _set_objective_trainable(model, objective)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters for {objective}")

    configured_maximum = {
        "single_task_tbr": int(config.get("stage_tbr_epochs", 99)),
        "single_task_cac": int(config.get("stage_cac_epochs", 99)),
    }.get(objective, int(config.get("joint_mtl_epochs", 99)))
    maximum = min(configured_maximum, 99)
    fixed_epoch = _fixed_epoch(objective, epoch_code)
    limit = int(fixed_epoch if fixed_epoch is not None else maximum)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    history: dict[str, Any] = {
        "stage": objective,
        "train": [],
        "validation": [],
        "train_tbr": [],
        "train_cac": [],
        "validation_tbr": [],
        "validation_cac": [],
        "best_epoch": limit,
        "epochs_ran": 0,
    }
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, limit + 1):
        model.train()
        train_loss: list[float] = []
        train_tbr: list[float] = []
        train_cac: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            loss, components = _objective_loss(
                model,
                model(**model_inputs(batch)),
                batch.baseline,
                batch.targets,
                config,
                tail_threshold,
                objective,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config.get("grad_clip", 1.0))
            )
            optimizer.step()
            train_loss.append(float(loss.detach()))
            train_tbr.append(components["tbr"])
            train_cac.append(components["cac"])
        history["train"].append(float(np.mean(train_loss)))
        history["train_tbr"].append(_finite_mean(train_tbr))
        history["train_cac"].append(_finite_mean(train_cac))
        history["epochs_ran"] = epoch
        if val_loader is None:
            continue

        model.eval()
        validation_loss: list[float] = []
        validation_tbr: list[float] = []
        validation_cac: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss, components = _objective_loss(
                    model,
                    model(**model_inputs(batch)),
                    batch.baseline,
                    batch.targets,
                    config,
                    tail_threshold,
                    objective,
                )
                validation_loss.append(float(loss))
                validation_tbr.append(components["tbr"])
                validation_cac.append(components["cac"])
        validation = float(np.mean(validation_loss))
        history["validation"].append(validation)
        history["validation_tbr"].append(_finite_mean(validation_tbr))
        history["validation_cac"].append(_finite_mean(validation_cac))
        if validation < best_loss - 1e-7:
            best_loss = validation
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= int(config.get("stage_patience", 25)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    central._restore_declared_trainable(model)
    selected_epoch = int(history["best_epoch"])
    stage_epochs = {
        "joint": selected_epoch if objective in JOINT_OBJECTIVES else 0,
        "tbr": selected_epoch if objective == "single_task_tbr" else 0,
        "cac": selected_epoch if objective == "single_task_cac" else 0,
    }
    return model, {
        "best_epoch": _encoded_epoch(objective, selected_epoch),
        "stage_epochs": stage_epochs,
        "stages": [history],
        "training_objective": objective,
        "target_scaler": scaler,
        "tail_threshold_training_partition": tail_threshold,
        "sampling": (
            base._balanced_sampling_weights(train_arrays, sampling)[1]
            if bool(sampling.get("balance_cac_strata", False))
            else None
        ),
    }


def train_v60_multitask_model(
    model: LACiTransformerV60,
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


def fit_v60_multitask_fixed_epochs(
    model: LACiTransformerV60,
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
