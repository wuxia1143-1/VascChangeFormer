from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v34 import LACV34Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v32_selector import ObservedLagSelector
from .v34_trainer import fit_v34_fixed_epochs, train_v34_model


V34_MODELS = (
    "lac_v34_full",
    "lac_v34_dual_independent",
    "lac_v34_no_gradient_protection",
    "lac_v34_no_baseline_anchoring",
    "lac_v34_no_mechanism_aux",
    "lac_v34_no_monotonic_constraint",
    "lac_v34_no_hard_views",
    "lac_v34_random_mechanism_view",
    "lac_v34_with_adapters",
)


def _seed_everything_v34(seed: int) -> None:
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
) -> LACV34Config:
    if model_name not in V34_MODELS:
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
    # Fixed before fitting: same-size negative-control view selected from the
    # non-inflammation variables using a preregistered seed.
    candidates = np.asarray(
        [index for index in range(len(schema.longitudinal_features)) if index not in inflammation_indices]
    )
    rng = np.random.default_rng(3404)
    random_indices = tuple(
        int(value)
        for value in rng.choice(
            candidates,
            size=len(inflammation_indices),
            replace=len(candidates) < len(inflammation_indices),
        )
    )
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
        "inflammation_feature_indices": inflammation_indices,
        "calcification_feature_indices": calcification_indices,
        "mechanism_feature_indices": inflammation_indices,
    } | dict(config.get("model_v34", {}))
    overrides = {
        "lac_v34_full": {},
        "lac_v34_dual_independent": {"separate_cac_encoder": True},
        "lac_v34_no_gradient_protection": {"isolate_cac_shared_gradient": False},
        "lac_v34_no_baseline_anchoring": {"baseline_anchoring": False},
        "lac_v34_no_mechanism_aux": {"mechanism_aux_enabled": False},
        "lac_v34_no_monotonic_constraint": {"mechanism_monotonic": False},
        "lac_v34_no_hard_views": {"hard_phenotype_views": False},
        "lac_v34_random_mechanism_view": {
            "mechanism_feature_indices": random_indices,
        },
        "lac_v34_with_adapters": {"phenotype_adapters": True},
    }
    return LACV34Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name == "lac_v34_with_adapters",
        "coupling": model_name != "lac_v34_no_mechanism_aux",
        "treatment": False,
    }


def _decision_modes(_: str) -> dict[str, str]:
    return {"tbr": "identity", "cac": "identity"}


def run_v34_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V34_MODELS:
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
        audited_runner.V27_MODELS = V34_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = ObservedLagSelector
        audited_runner.train_model = train_v34_model
        audited_runner.fit_model_fixed_epochs = fit_v34_fixed_epochs
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
            lambda value: _seed_everything_v34(int(value) + offset)
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
    summary["architecture_version"] = "V3.4"
    summary["development_status"] = (
        "post-hoc exploratory internal development on a repeatedly inspected "
        "443-patient cohort; requires independent confirmation"
    )
    summary["cv_protocol"]["directional_training"] = (
        "hard phenotype views, TBR-trained protected shared encoder, CAC-private "
        "central regression, and monotonic inflammation-burden progression auxiliary task"
    )
    summary["cv_protocol"]["mechanism_not_added_to_primary_prediction"] = True
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
