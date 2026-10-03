from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v32 import LACV32Config
from . import v27_nested as audited_runner
from .v32_selector import ObservedLagSelector
from .v32_trainer import (
    decode_stage_epochs,
    encode_stage_epochs,
    fit_v32_fixed_epochs,
    train_v32_model,
)


V32_MODELS = (
    "lac_v32_full",
    "lac_v32_dual_independent",
    "lac_v32_no_lag",
    "lac_v32_direct_only_lag",
    "lac_v32_teacher_only_lag",
    "lac_v32_no_persistence",
    "lac_v32_history_permuted",
    "lac_v32_history_shifted",
    "lac_v32_no_treatment",
    "lac_v32_no_real_time",
    "lac_v32_no_patient_gate",
    "lac_v32_no_adapters",
    "lac_v32_no_baseline_anchoring",
)

INFLAMMATION_FEATURES = (
    "d_dimer", "platelet", "neutrophil", "lymphocyte", "nlr", "crp", "il6",
)
CALCIFICATION_FEATURES = (
    "BMI", "systolic_blood_pressure", "glucose", "cholesterol", "ldl",
    "triglyceride", "hdl", "egfr", "creatinine",
)


def _seed_everything_v32(seed: int) -> None:
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
) -> LACV32Config:
    if model_name not in V32_MODELS:
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
    } | dict(config.get("model_v32", {}))
    overrides = {
        "lac_v32_full": {},
        "lac_v32_dual_independent": {
            "separate_cac_encoder": True,
            "coupling_enabled": False,
            "lag_residual_enabled": False,
        },
        "lac_v32_no_lag": {
            "coupling_enabled": False,
            "lag_residual_enabled": False,
        },
        "lac_v32_direct_only_lag": {"tbr_teacher_in_lag": False},
        "lac_v32_teacher_only_lag": {
            "direct_observed_lag": False,
            "persistence_gate_features": False,
        },
        "lac_v32_no_persistence": {"persistence_gate_features": False},
        "lac_v32_history_permuted": {"permute_patient_history": True},
        "lac_v32_history_shifted": {"shift_history_to_past": True},
        "lac_v32_no_treatment": {"treatment_conditioning": False},
        "lac_v32_no_real_time": {"use_real_time_intervals": False},
        "lac_v32_no_patient_gate": {"patient_lag_gate": False},
        "lac_v32_no_adapters": {"phenotype_adapters": False},
        "lac_v32_no_baseline_anchoring": {"baseline_anchoring": False},
    }
    return LACV32Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    coupling_off = model_name in {
        "lac_v32_no_lag", "lac_v32_dual_independent"
    }
    return {
        "adapter": model_name != "lac_v32_no_adapters",
        "coupling": not coupling_off,
        "treatment": not coupling_off and model_name != "lac_v32_no_treatment",
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    return {
        "tbr": "identity",
        "cac": (
            "identity"
            if model_name in {"lac_v32_no_lag", "lac_v32_dual_independent"}
            else "full"
        ),
    }


def _componentwise_epoch_median(codes: list[int]) -> int:
    """Aggregate early-stopped stages independently, not lexicographically."""
    if not codes:
        raise ValueError("At least one inner-fold epoch code is required")
    decoded = np.asarray([decode_stage_epochs(code) for code in codes], dtype=int)
    selected = np.rint(np.median(decoded, axis=0)).astype(int)
    selected[0] = max(1, int(selected[0]))
    selected[1] = max(1, int(selected[1]))
    selected[2] = max(0, int(selected[2]))
    return encode_stage_epochs(*selected.tolist())


def run_v32_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V32_MODELS:
        raise KeyError(model_name)
    names = (
        "V27_MODELS", "_config_for_model", "_feature_flags",
        "_decision_modes", "SafeOOFDecisionLayer", "train_model",
        "fit_model_fixed_epochs", "seed_everything", "_inner_oof_predictions",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    try:
        audited_runner.V27_MODELS = V32_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        audited_runner.SafeOOFDecisionLayer = ObservedLagSelector
        audited_runner.train_model = train_v32_model
        audited_runner.fit_model_fixed_epochs = fit_v32_fixed_epochs
        original_inner_oof = original["_inner_oof_predictions"]

        def componentwise_inner_oof(*args, **kwargs):
            combined, records, _ = original_inner_oof(*args, **kwargs)
            epoch_code = _componentwise_epoch_median(
                [int(record["selected_epochs"]) for record in records]
            )
            return combined, records, epoch_code

        audited_runner._inner_oof_predictions = componentwise_inner_oof
        offset = int(config.get("training_seed_offset", 0))
        audited_runner.seed_everything = (
            lambda value: _seed_everything_v32(int(value) + offset)
        )
        summary = audited_runner.run_v27_nested_cross_validation(
            arrays, schema, config, output_dir, model_name,
            outer_fold_assignments=outer_fold_assignments,
            outer_folds=outer_folds,
        )
    finally:
        for name, value in original.items():
            setattr(audited_runner, name, value)
    summary["architecture_version"] = "V3.2"
    summary["development_status"] = (
        "exploratory internal validation on a repeatedly inspected "
        "443-patient development cohort"
    )
    summary["cv_protocol"]["directional_training"] = (
        "protected inflammation-only TBR branch; CAC-specific observed "
        "inflammation history plus protected TBR teacher through a strictly "
        "causal patient-gated residual route"
    )
    summary["cv_protocol"]["phenotype_partition"] = {
        "inflammation_only": list(INFLAMMATION_FEATURES),
        "cac_central_only": list(CALCIFICATION_FEATURES),
        "overlap": [],
    }
    summary["cv_protocol"]["training_seed_offset"] = int(
        config.get("training_seed_offset", 0)
    )
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
