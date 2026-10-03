from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from ..data.schema import FeatureSchema
from ..models.lac_v50 import LACV50Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v40_final_selector import V40FinalSelector
from .v41_trainer import fit_v41_fixed_epochs, train_v41_model


V50_MODELS = (
    "lac_v50_full",
    "lac_v50_no_task_adapter",
    "lac_v50_no_baseline_anchoring",
    "lac_v50_no_reliability_gate",
    "lac_v50_no_oof_calibration",
)


def _seed_everything_v50(seed: int) -> None:
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
) -> LACV50Config:
    if model_name not in V50_MODELS:
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
    } | dict(config.get("model_v50", {}))
    overrides = {
        "lac_v50_full": {},
        "lac_v50_no_task_adapter": {"soft_phenotype_adapters": False},
        "lac_v50_no_baseline_anchoring": {"baseline_anchoring": False},
        "lac_v50_no_reliability_gate": {"reliability_gate_enabled": False},
        "lac_v50_no_oof_calibration": {},
    }
    return LACV50Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    if model_name not in V50_MODELS:
        raise KeyError(model_name)
    return {
        "adapter": model_name != "lac_v50_no_task_adapter",
        # The compatibility field carries the V5 statistical reliability gate;
        # no inflammation-to-calcification representation is present.
        "coupling": model_name != "lac_v50_no_reliability_gate",
        "treatment": False,
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    if model_name not in V50_MODELS:
        raise KeyError(model_name)
    if model_name == "lac_v50_no_oof_calibration":
        # Keep the neural reliability residual at its native weight of one,
        # but bypass both the cross-fitted calibration expert and OOF-selected
        # residual weight.
        return {"tbr": "identity", "cac": "tail_only"}
    return {"tbr": "identity", "cac": "full"}


def run_v50_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V50_MODELS:
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
        audited_runner.V27_MODELS = V50_MODELS
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
            lambda value: _seed_everything_v50(int(value) + offset)
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
    summary["architecture_version"] = "V5.0"
    summary["development_status"] = (
        "preregistered internal development on the repeatedly inspected "
        "443-patient cohort; frozen confirmation follows only after the locked "
        "absolute-MAE non-inferiority rule"
    )
    summary["cv_protocol"]["frozen_design"] = (
        "protected baseline-anchored TBR path, private CAC path, task-specific "
        "low-rank adapters, mechanism-free reliability gate and inner-OOF "
        "calibration; detached uncertainty is fitted after point prediction"
    )
    summary["cv_protocol"]["explicit_i_to_c_path"] = False
    summary["cv_protocol"]["history_or_treatment_in_reliability_gate"] = False
    summary["cv_protocol"]["uncertainty_can_modify_point_prediction"] = False
    summary["cv_protocol"]["decision_mode"] = _decision_modes(model_name)
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["external_labels_used"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    oof_path = Path(output_dir, "out_of_fold_predictions.csv")
    if oof_path.is_file():
        oof = pd.read_csv(oof_path, dtype={"patient_id": str})
        if "coupling_gate" in oof:
            oof = oof.rename(columns={"coupling_gate": "reliability_gate"})
            oof.to_csv(oof_path, index=False)
        gate_distributions = summary.get("diagnostics", {}).get(
            "gate_distributions", {}
        )
        if "coupling_gate" in gate_distributions:
            gate_distributions["reliability_gate"] = gate_distributions.pop(
                "coupling_gate"
            )
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
