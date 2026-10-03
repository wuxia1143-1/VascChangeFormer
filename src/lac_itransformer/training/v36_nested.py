from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v36 import LACV36Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v35_trainer import fit_v35_fixed_epochs, train_v35_model
from .v36_selector import ParetoSafeClinicalSelector


V36_MODELS = (
    "lac_v36_full",
    "lac_v36_no_calibration",
    "lac_v36_calibration_only",
    "lac_v36_no_shared_transfer",
    "lac_v36_no_cac_adapter",
    "lac_v36_no_gradient_protection",
    "lac_v36_dual_independent",
    "lac_v36_no_hard_views",
    "lac_v36_no_baseline_anchoring",
    "lac_v36_no_mechanism_aux",
    "lac_v36_no_historical_burden",
)


def _seed_everything_v36(seed: int) -> None:
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
) -> LACV36Config:
    if model_name not in V36_MODELS:
        raise KeyError(model_name)
    feature_index = {
        name: index for index, name in enumerate(schema.longitudinal_features)
    }
    missing = [
        name
        for name in INFLAMMATION_FEATURES + CALCIFICATION_FEATURES
        if name not in feature_index
    ]
    if missing:
        raise ValueError(f"Missing preregistered phenotype features: {missing}")
    inflammation_indices = tuple(
        feature_index[name] for name in INFLAMMATION_FEATURES
    )
    calcification_indices = tuple(
        feature_index[name] for name in CALCIFICATION_FEATURES
    )
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
        "inflammation_feature_indices": inflammation_indices,
        "calcification_feature_indices": calcification_indices,
        "mechanism_feature_indices": inflammation_indices,
    } | dict(config.get("model_v36", {}))
    overrides = {
        "lac_v36_full": {},
        "lac_v36_no_calibration": {},
        "lac_v36_calibration_only": {},
        "lac_v36_no_shared_transfer": {"shared_transfer_enabled": False},
        "lac_v36_no_cac_adapter": {"cac_residual_adapter": False},
        "lac_v36_no_gradient_protection": {
            "isolate_cac_shared_gradient": False,
        },
        "lac_v36_dual_independent": {
            "use_private_cac_encoder": True,
            "shared_transfer_enabled": False,
        },
        "lac_v36_no_hard_views": {"hard_phenotype_views": False},
        "lac_v36_no_baseline_anchoring": {"baseline_anchoring": False},
        "lac_v36_no_mechanism_aux": {"mechanism_aux_enabled": False},
        "lac_v36_no_historical_burden": {"mechanism_history_enabled": False},
    }
    return LACV36Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v36_no_cac_adapter",
        "coupling": model_name not in {
            "lac_v36_no_shared_transfer",
            "lac_v36_dual_independent",
            "lac_v36_no_mechanism_aux",
        },
        "treatment": False,
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    if model_name == "lac_v36_no_calibration":
        return {"tbr": "identity", "cac": "identity"}
    if model_name == "lac_v36_calibration_only":
        return {"tbr": "identity", "cac": "expert"}
    return {"tbr": "identity", "cac": "full"}


def run_v36_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V36_MODELS:
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
        "_inner_oof_predictions",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V36_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = ParetoSafeClinicalSelector
        audited_runner.train_model = train_v35_model
        audited_runner.fit_model_fixed_epochs = fit_v35_fixed_epochs
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
            lambda value: _seed_everything_v36(int(value) + offset)
        )
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
    summary["architecture_version"] = "V3.6"
    summary["development_status"] = (
        "post-hoc exploratory internal development after repeated inspection "
        "of the 443-patient cohort; requires independent confirmation"
    )
    summary["cv_protocol"]["frozen_design"] = (
        "protected shared dual-task core, exact RNG-isolated TBR stage, "
        "and inner-OOF-only Pareto-safe CAC calibration"
    )
    summary["cv_protocol"]["mechanism_role"] = (
        "strict historical inflammation burden is an auxiliary explanatory "
        "endpoint and never a direct continuous-CAC residual"
    )
    summary["cv_protocol"]["decision_mode"] = _decision_modes(model_name)
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
