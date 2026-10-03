from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v35 import LACV35Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v35_trainer import fit_v35_fixed_epochs, train_v35_model


V35_MODELS = (
    "lac_v35_full",
    "lac_v35_no_shared_transfer",
    "lac_v35_no_private_cac_encoder",
    "lac_v35_no_cac_adapter",
    "lac_v35_no_gradient_protection",
    "lac_v35_no_conflict_protection",
    "lac_v35_no_mechanism_aux",
    "lac_v35_no_historical_burden",
    "lac_v35_random_mechanism_view",
    "lac_v35_no_monotonic_constraint",
    "lac_v35_no_hard_views",
    "lac_v35_no_baseline_anchoring",
)


class IdentityOOFSelector:
    """API-compatible no-op selector for one frozen prediction per task."""

    def __init__(self, **_: Any):
        self.patient_count = 0

    def fit(self, heads, targets, features, meta_fold):
        del targets, features, meta_fold
        self.patient_count = int(len(heads["tbr"]))
        return self

    def predict(self, heads, features, modes=None):
        del features, modes
        tbr = np.asarray(heads["tbr"][:, 0], float)
        cac = np.asarray(heads["cac"][:, 0], float)
        return (
            np.column_stack([tbr, cac]),
            {"tbr": np.zeros_like(tbr), "cac": np.zeros_like(cac)},
        )

    def audit(self) -> dict[str, Any]:
        return {
            "selection_objective": "none: one frozen primary prediction per task",
            "unique_fixed_configuration": {
                "tbr": {"method": "identity", "weight": 0.0, "alpha": None},
                "cac": {"method": "identity", "weight": 0.0, "alpha": None},
            },
            "tasks": {
                task: {
                    "task": task,
                    "selected_method": "identity",
                    "training_source": "frozen model primary head",
                }
                for task in ("tbr", "cac")
            },
            "inner_oof_patient_count": self.patient_count,
            "all_constraints_passed": True,
            "outer_test_labels_used": False,
        }


def _seed_everything_v35(seed: int) -> None:
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
) -> LACV35Config:
    if model_name not in V35_MODELS:
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
    inflammation_indices = tuple(
        feature_index[name] for name in INFLAMMATION_FEATURES
    )
    calcification_indices = tuple(
        feature_index[name] for name in CALCIFICATION_FEATURES
    )
    candidates = np.asarray(
        [
            index for index in range(len(schema.longitudinal_features))
            if index not in inflammation_indices
        ]
    )
    rng = np.random.default_rng(3505)
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
    } | dict(config.get("model_v35", {}))
    overrides = {
        "lac_v35_full": {},
        "lac_v35_no_shared_transfer": {"shared_transfer_enabled": False},
        "lac_v35_no_private_cac_encoder": {"use_private_cac_encoder": False},
        "lac_v35_no_cac_adapter": {"cac_residual_adapter": False},
        "lac_v35_no_gradient_protection": {
            "isolate_cac_shared_gradient": False,
        },
        "lac_v35_no_conflict_protection": {
            "mechanism_gradient_conflict_protection": False,
        },
        "lac_v35_no_mechanism_aux": {"mechanism_aux_enabled": False},
        "lac_v35_no_historical_burden": {"mechanism_history_enabled": False},
        "lac_v35_random_mechanism_view": {
            "mechanism_feature_indices": random_indices,
        },
        "lac_v35_no_monotonic_constraint": {"mechanism_monotonic": False},
        "lac_v35_no_hard_views": {"hard_phenotype_views": False},
        "lac_v35_no_baseline_anchoring": {"baseline_anchoring": False},
    }
    return LACV35Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v35_no_cac_adapter",
        "coupling": model_name != "lac_v35_no_mechanism_aux",
        "treatment": False,
    }


def _decision_modes(_: str) -> dict[str, str]:
    return {"tbr": "identity", "cac": "identity"}


def run_v35_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V35_MODELS:
        raise KeyError(model_name)
    names = (
        "V27_MODELS", "_config_for_model", "_feature_flags", "_decision_modes",
        "SafeOOFDecisionLayer", "train_model", "fit_model_fixed_epochs",
        "seed_everything", "_inner_oof_predictions",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V35_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = IdentityOOFSelector
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
            lambda value: _seed_everything_v35(int(value) + offset)
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
    summary["architecture_version"] = "V3.5"
    summary["development_status"] = (
        "post-hoc exploratory internal development after repeated inspection "
        "of the 443-patient cohort; requires independent confirmation"
    )
    summary["cv_protocol"]["pareto_safe_design"] = (
        "protected TBR shared encoder, private CAC encoder, skip-safe shared "
        "transfer, CAC-only all-view adapter, and conflict-protected mechanism auxiliary"
    )
    summary["cv_protocol"]["mechanism_not_added_to_primary_prediction"] = True
    summary["cv_protocol"]["decision_layer"] = (
        "identity: one frozen primary prediction per task; no post-hoc blending"
    )
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
