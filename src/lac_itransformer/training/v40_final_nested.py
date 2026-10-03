from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v40_final import LACV40FinalConfig
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v40_final_selector import V40FinalSelector
from .v41_trainer import fit_v41_fixed_epochs, train_v41_model


V40_FINAL_MODELS = (
    "lac_v40_final_full",
    "lac_v40_final_no_calibration",
    "lac_v40_final_no_tail",
    "lac_v40_final_central_only",
    "lac_v40_final_no_i_to_c",
    "lac_v40_final_no_history",
    "lac_v40_final_no_treatment",
    "lac_v40_final_no_risk_gate",
    "lac_v40_final_no_soft_adapters",
    "lac_v40_final_no_baseline_anchoring",
)


def _seed_everything_v40_final(seed: int) -> None:
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
) -> LACV40FinalConfig:
    if model_name not in V40_FINAL_MODELS:
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
        "mechanism_feature_indices": tuple(
            feature_index[name] for name in INFLAMMATION_FEATURES
        ),
    } | dict(config.get("model_v40_final", {}))
    overrides = {
        "lac_v40_final_full": {},
        "lac_v40_final_no_calibration": {},
        "lac_v40_final_no_tail": {"progression_hurdle_enabled": False},
        "lac_v40_final_central_only": {"progression_hurdle_enabled": False},
        "lac_v40_final_no_i_to_c": {"hurdle_i_to_c_enabled": False},
        "lac_v40_final_no_history": {
            "hurdle_i_to_c_enabled": False,
            "hurdle_history_enabled": False,
        },
        "lac_v40_final_no_treatment": {
            "hurdle_treatment_enabled": False,
            "treatment_conditioning": False,
        },
        "lac_v40_final_no_risk_gate": {"hurdle_risk_gate_enabled": False},
        "lac_v40_final_no_soft_adapters": {"soft_phenotype_adapters": False},
        "lac_v40_final_no_baseline_anchoring": {"baseline_anchoring": False},
    }
    return LACV40FinalConfig(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v40_final_no_soft_adapters",
        "coupling": model_name not in {
            "lac_v40_final_no_tail",
            "lac_v40_final_central_only",
            "lac_v40_final_no_i_to_c",
            "lac_v40_final_no_history",
        },
        "treatment": model_name != "lac_v40_final_no_treatment",
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    if model_name == "lac_v40_final_no_calibration":
        return {"tbr": "identity", "cac": "no_calibration"}
    if model_name == "lac_v40_final_no_tail":
        return {"tbr": "identity", "cac": "no_tail"}
    if model_name == "lac_v40_final_central_only":
        return {"tbr": "identity", "cac": "central"}
    return {"tbr": "identity", "cac": "full"}


def run_v40_final_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V40_FINAL_MODELS:
        raise KeyError(model_name)
    names = (
        "V27_MODELS", "_config_for_model", "_feature_flags", "_decision_modes",
        "SafeOOFDecisionLayer", "train_model", "fit_model_fixed_epochs",
        "seed_everything", "_inner_oof_predictions",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V40_FINAL_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = V40FinalSelector
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
            lambda value: _seed_everything_v40_final(int(value) + offset)
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
    summary["architecture_version"] = "V4.0-final"
    summary["development_status"] = (
        "post-hoc development on the repeatedly inspected 443-patient cohort; "
        "the frozen candidate requires one-pass external evaluation"
    )
    summary["cv_protocol"]["frozen_design"] = (
        "unchanged protected V3.7 TBR path, private direct CAC central path, "
        "cross-fitted robust calibration and a detached skippable tail hurdle"
    )
    summary["cv_protocol"]["mechanism_role"] = (
        "strictly earlier inflammation and treatment can affect only the detached "
        "CAC tail correction; the central CAC path has no shared I-to-C transfer"
    )
    summary["cv_protocol"]["decision_mode"] = _decision_modes(model_name)
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
