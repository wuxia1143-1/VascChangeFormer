from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v42 import LACV42Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v41_trainer import fit_v41_fixed_epochs, train_v41_model
from .v42_selector import V42DecisionLayer


V42_MODELS = (
    "lac_v42_strict_no_tail",
    "lac_v42_strict_no_tail_varcal",
    "lac_v42_strict_no_tail_two_stage",
)


class ConfiguredV42DecisionLayer(V42DecisionLayer):
    """Pickle-safe runner adapter for preregistered YAML decision options."""

    configured_variance_scales = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
    configured_progression_threshold = 0.25
    configured_cac_mode = "strict"
    configured_seed = 2026

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            variance_scales=tuple(self.configured_variance_scales),
            progression_threshold=float(self.configured_progression_threshold),
            cac_mode=str(self.configured_cac_mode),
            seed=int(self.configured_seed),
            **kwargs,
        )


def _seed_everything_v42(seed: int) -> None:
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
) -> LACV42Config:
    if model_name not in V42_MODELS:
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
        "mechanism_feature_indices": tuple(
            feature_index[name] for name in INFLAMMATION_FEATURES
        ),
    } | dict(config.get("model_v42", {}))
    return LACV42Config(**options)


def _feature_flags(model_name: str) -> dict[str, bool]:
    if model_name not in V42_MODELS:
        raise KeyError(model_name)
    return {"adapter": True, "coupling": False, "treatment": True}


def _decision_modes(model_name: str) -> dict[str, str]:
    modes = {
        "lac_v42_strict_no_tail": "strict",
        "lac_v42_strict_no_tail_varcal": "variance",
        "lac_v42_strict_no_tail_two_stage": "two_stage",
    }
    if model_name not in modes:
        raise KeyError(model_name)
    return {"tbr": "identity", "cac": modes[model_name]}


def run_v42_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V42_MODELS:
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
    configured = (
        ConfiguredV42DecisionLayer.configured_variance_scales,
        ConfiguredV42DecisionLayer.configured_progression_threshold,
        ConfiguredV42DecisionLayer.configured_cac_mode,
        ConfiguredV42DecisionLayer.configured_seed,
    )
    try:
        decision_options = dict(config.get("decision", {}))
        ConfiguredV42DecisionLayer.configured_variance_scales = tuple(
            decision_options.get(
                "variance_scales",
                (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
            )
        )
        ConfiguredV42DecisionLayer.configured_progression_threshold = float(
            decision_options.get("progression_threshold", 0.25)
        )
        ConfiguredV42DecisionLayer.configured_cac_mode = _decision_modes(
            model_name
        )["cac"]
        ConfiguredV42DecisionLayer.configured_seed = int(config.get("seed", 2026))

        audited_runner.V27_MODELS = V42_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = ConfiguredV42DecisionLayer
        audited_runner.train_model = train_v41_model
        audited_runner.fit_model_fixed_epochs = fit_v41_fixed_epochs
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
            lambda value: _seed_everything_v42(int(value) + offset)
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
        (
            ConfiguredV42DecisionLayer.configured_variance_scales,
            ConfiguredV42DecisionLayer.configured_progression_threshold,
            ConfiguredV42DecisionLayer.configured_cac_mode,
            ConfiguredV42DecisionLayer.configured_seed,
        ) = configured
    summary["architecture_version"] = "V4.2"
    summary["development_status"] = (
        "preregistered internal development on the repeatedly inspected "
        "443-patient cohort; independent external confirmation required"
    )
    summary["cv_protocol"]["frozen_design"] = (
        "protected V3.7 TBR path, private direct CAC central path, no active "
        "tail stage, and tail-feature-free inner-OOF decision layer"
    )
    summary["cv_protocol"]["decision_mode"] = _decision_modes(model_name)
    summary["cv_protocol"]["safe_cac_decision_features"] = True
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["external_labels_used"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
