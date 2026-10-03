from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v30 import LACV30Config
from . import v27_nested as audited_runner
from .v30_selector import DirectionalLagScaleSelector
from .v30_trainer import fit_v30_fixed_epochs, train_v30_model


V30_MODELS = (
    "lac_v30_full",
    "lac_v30_dual_independent",
    "lac_v30_no_lag",
    "lac_v30_no_tbr_teacher",
    "lac_v30_no_shared_encoder",
    "lac_v30_no_treatment",
    "lac_v30_no_real_time",
    "lac_v30_no_adapters",
)


def _seed_everything_v30(seed: int) -> None:
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
) -> LACV30Config:
    if model_name not in V30_MODELS:
        raise KeyError(model_name)
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
    } | dict(config.get("model_v30", {}))
    overrides = {
        "lac_v30_full": {},
        "lac_v30_dual_independent": {
            "separate_cac_encoder": True,
            "coupling_enabled": False,
            "lag_residual_enabled": False,
            "tbr_supervised_lag_teacher": False,
        },
        "lac_v30_no_lag": {
            "coupling_enabled": False,
            "lag_residual_enabled": False,
        },
        "lac_v30_no_tbr_teacher": {
            "tbr_supervised_lag_teacher": False,
        },
        "lac_v30_no_shared_encoder": {"separate_cac_encoder": True},
        "lac_v30_no_treatment": {"treatment_conditioning": False},
        "lac_v30_no_real_time": {"use_real_time_intervals": False},
        "lac_v30_no_adapters": {"phenotype_adapters": False},
    }
    return LACV30Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v30_no_adapters",
        "coupling": model_name not in {
            "lac_v30_no_lag", "lac_v30_dual_independent"
        },
        "treatment": model_name not in {
            "lac_v30_no_lag", "lac_v30_dual_independent",
            "lac_v30_no_treatment",
        },
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    return {
        "tbr": "identity",
        "cac": (
            "identity"
            if model_name in {"lac_v30_no_lag", "lac_v30_dual_independent"}
            else "full"
        ),
    }


def run_v30_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V30_MODELS:
        raise KeyError(model_name)
    names = (
        "V27_MODELS", "_config_for_model", "_feature_flags",
        "_decision_modes", "SafeOOFDecisionLayer", "train_model",
        "fit_model_fixed_epochs", "seed_everything",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V30_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = DirectionalLagScaleSelector
        audited_runner.train_model = train_v30_model
        audited_runner.fit_model_fixed_epochs = fit_v30_fixed_epochs
        audited_runner.seed_everything = _seed_everything_v30
        summary = audited_runner.run_v27_nested_cross_validation(
            arrays, schema, config, output_dir, model_name,
            outer_fold_assignments=outer_fold_assignments,
            outer_folds=outer_folds,
        )
    finally:
        for name, value in original.items():
            setattr(audited_runner, name, value)
    summary["architecture_version"] = "V3.0"
    summary["development_status"] = (
        "exploratory internal validation on a repeatedly inspected "
        "443-patient development cohort"
    )
    summary["cv_protocol"]["directional_training"] = (
        "TBR-supervised inflammation teacher, frozen CAC central training, "
        "then historical lag-residual training"
    )
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
