from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v33 import LACV33Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v32_selector import ObservedLagSelector
from .v33_trainer import fit_v33_fixed_epochs, train_v33_model


V33_MODELS = (
    "lac_v33_full",
    "lac_v33_dual_independent",
    "lac_v33_no_history",
    "lac_v33_current_only",
    "lac_v33_history_permuted",
    "lac_v33_no_hurdle",
    "lac_v33_naive_shared_gradients",
    "lac_v33_with_adapters",
    "lac_v33_no_treatment",
    "lac_v33_no_time_decay",
    "lac_v33_no_baseline_anchoring",
)


def _seed_everything_v33(seed: int) -> None:
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
) -> LACV33Config:
    if model_name not in V33_MODELS:
        raise KeyError(model_name)
    feature_index = {
        name: index for index, name in enumerate(schema.longitudinal_features)
    }
    missing = [
        name for name in INFLAMMATION_FEATURES + CALCIFICATION_FEATURES
        if name not in feature_index
    ]
    if missing:
        raise ValueError(f"Missing preregistered phenotype features: {missing}")
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
        "inflammation_feature_indices": tuple(
            feature_index[name] for name in INFLAMMATION_FEATURES
        ),
        "calcification_feature_indices": tuple(
            feature_index[name] for name in CALCIFICATION_FEATURES
        ),
    } | dict(config.get("model_v33", {}))
    overrides = {
        "lac_v33_full": {},
        "lac_v33_dual_independent": {
            "separate_cac_encoder": True,
            "history_mode": "none",
            "hurdle_enabled": False,
            "coupling_enabled": False,
            "lag_residual_enabled": False,
        },
        "lac_v33_no_history": {"history_mode": "none"},
        "lac_v33_current_only": {"history_mode": "current"},
        "lac_v33_history_permuted": {"history_mode": "permuted"},
        "lac_v33_no_hurdle": {
            "hurdle_enabled": False,
            "coupling_enabled": False,
            "lag_residual_enabled": False,
        },
        "lac_v33_naive_shared_gradients": {
            "isolate_cac_shared_gradient": False,
        },
        "lac_v33_with_adapters": {"phenotype_adapters": True},
        "lac_v33_no_treatment": {"treatment_conditioning": False},
        "lac_v33_no_time_decay": {"use_time_decay_kernel": False},
        "lac_v33_no_baseline_anchoring": {"baseline_anchoring": False},
    }
    return LACV33Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    history_off = model_name in {
        "lac_v33_dual_independent", "lac_v33_no_history", "lac_v33_no_hurdle"
    }
    return {
        "adapter": model_name == "lac_v33_with_adapters",
        "coupling": not history_off,
        "treatment": not history_off and model_name != "lac_v33_no_treatment",
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    return {
        "tbr": "identity",
        "cac": (
            "identity"
            if model_name in {"lac_v33_dual_independent", "lac_v33_no_hurdle"}
            else "full"
        ),
    }


def run_v33_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V33_MODELS:
        raise KeyError(model_name)
    names = (
        "V27_MODELS", "_config_for_model", "_feature_flags",
        "_decision_modes", "SafeOOFDecisionLayer", "train_model",
        "fit_model_fixed_epochs", "seed_everything", "_inner_oof_predictions",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V33_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = ObservedLagSelector
        audited_runner.train_model = train_v33_model
        audited_runner.fit_model_fixed_epochs = fit_v33_fixed_epochs
        original_inner_oof = original["_inner_oof_predictions"]

        def componentwise_inner_oof(*args, **kwargs):
            combined, records, _ = original_inner_oof(*args, **kwargs)
            return (
                combined,
                records,
                _componentwise_epoch_median(
                    [int(record["selected_epochs"]) for record in records]
                ),
            )

        audited_runner._inner_oof_predictions = componentwise_inner_oof
        offset = int(config.get("training_seed_offset", 0))
        audited_runner.seed_everything = (
            lambda value: _seed_everything_v33(int(value) + offset)
        )
        summary = audited_runner.run_v27_nested_cross_validation(
            arrays, schema, config, output_dir, model_name,
            outer_fold_assignments=outer_fold_assignments,
            outer_folds=outer_folds,
        )
    finally:
        for name, value in original.items():
            setattr(audited_runner, name, value)
    summary["architecture_version"] = "V3.3"
    summary["development_status"] = (
        "post-hoc exploratory internal development on a repeatedly inspected "
        "443-patient cohort; requires independent confirmation"
    )
    summary["cv_protocol"]["directional_training"] = (
        "TBR-protected shared encoder, calcification central predictor, and "
        "direct observed-inflammation causal-kernel progression hurdle"
    )
    summary["cv_protocol"]["phenotype_partition"] = {
        "inflammation_only": list(INFLAMMATION_FEATURES),
        "cac_central_only": list(CALCIFICATION_FEATURES),
        "overlap": [],
    }
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
