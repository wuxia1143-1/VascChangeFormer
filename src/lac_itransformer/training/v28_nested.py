from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v28 import LACV28Config
from . import v27_nested as audited_runner
from .v28_residual import CrossFittedLagResidualCorrector
from .v28_trainer import fit_v28_fixed_epochs, train_v28_model


V28_MODELS = (
    "lac_v28_full",
    "lac_v28_no_residual",
    "lac_v28_no_coupling",
    "lac_v28_no_treatment",
    "lac_v28_no_adapters",
    "lac_v28_naive_gradients",
    "lac_v28_no_multitask_sharing",
    "lac_v28_no_stop_gradient",
    "lac_v28_no_real_time",
)


def _seed_everything_v28(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def _config_for_model(
    schema: FeatureSchema,
    config: dict[str, Any],
    model_name: str,
) -> LACV28Config:
    if model_name not in V28_MODELS:
        raise KeyError(model_name)
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
    } | dict(config.get("model_v28", {}))
    overrides = {
        "lac_v28_full": {},
        "lac_v28_no_residual": {},
        "lac_v28_no_coupling": {"coupling_enabled": False},
        "lac_v28_no_treatment": {"treatment_conditioning": False},
        "lac_v28_no_adapters": {"phenotype_adapters": False},
        "lac_v28_naive_gradients": {"gradient_coordination": False},
        "lac_v28_no_multitask_sharing": {"cac_shared_gradient": False},
        "lac_v28_no_stop_gradient": {"stop_gradient_lag": False},
        "lac_v28_no_real_time": {"use_real_time_intervals": False},
    }
    return LACV28Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v28_no_adapters",
        "coupling": model_name != "lac_v28_no_coupling",
        "treatment": model_name not in {
            "lac_v28_no_coupling", "lac_v28_no_treatment"
        },
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    return {
        "tbr": "identity",
        "cac": (
            "identity"
            if model_name in {"lac_v28_no_residual", "lac_v28_no_coupling"}
            else "full"
        ),
    }


def run_v28_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Reuse the previously audited split/persistence runner with V2.8 hooks."""
    if model_name not in V28_MODELS:
        raise KeyError(model_name)
    names = (
        "V27_MODELS",
        "_config_for_model",
        "_feature_flags",
        "_decision_modes",
        "SafeOOFDecisionLayer",
        "train_model",
        "fit_model_fixed_epochs",
        "seed_everything",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V28_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = CrossFittedLagResidualCorrector
        audited_runner.train_model = train_v28_model
        audited_runner.fit_model_fixed_epochs = fit_v28_fixed_epochs
        audited_runner.seed_everything = _seed_everything_v28
        summary = audited_runner.run_v27_nested_cross_validation(
            arrays,
            schema,
            config,
            output_dir,
            model_name,
            outer_fold_assignments=outer_fold_assignments,
            outer_folds=outer_folds,
        )
    finally:
        for name, value in original.items():
            setattr(audited_runner, name, value)
    summary["architecture_version"] = "V2.8"
    summary["development_status"] = (
        "frozen deterministic internal validation on the 443-patient cohort"
    )
    summary["cv_protocol"]["decision_layer"] = (
        "bounded central-CAC residual correction fitted exclusively from "
        "the outer-training pool's five-fold inner OOF errors; exact identity "
        "fallback; one final prediction per patient"
    )
    summary["cv_protocol"]["gradient_coordination"] = (
        "PCGrad on shared encoder only; task-specific gradients unchanged"
    )
    summary["cv_protocol"]["outer_results_used_for_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = {
        "seed": int(config.get("seed", 2026)),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "flash_attention": False,
        "memory_efficient_attention": False,
        "math_attention": True,
    }
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
