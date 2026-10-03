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
from ..models.lac_v60 import LACV60Config
from . import v27_nested as audited_runner
from .v32_nested import (
    CALCIFICATION_FEATURES,
    INFLAMMATION_FEATURES,
    _componentwise_epoch_median,
)
from .v40_final_selector import V40FinalSelector
from .v41_trainer import fit_v41_fixed_epochs, train_v41_model
from .v42_selector import V42DecisionLayer
from .v60_multitask_trainer import (
    fit_v60_multitask_fixed_epochs,
    train_v60_multitask_model,
)
from .v60_selector import V60GenericDecisionLayer


V60_MODELS = (
    "lac_v60_a_no_i2c_tail",
    "lac_v60_b_no_mechanism_features",
    "lac_v60_c_generic_temporal",
    "lac_v60_no_task_adapters",
    "lac_v60_no_baseline_anchoring",
    "lac_v60_no_oof_calibration",
    "lac_v60_shared_only_mtl",
    "lac_v60_joint_hps_mtl",
    "lac_v60_joint_shared_adapter_mtl",
    "lac_v60_single_task_tbr",
    "lac_v60_single_task_cac",
    "vascmtl",
    "vascmtl_no_baseline_anchoring",
    "vascmtl_no_cac_calibration",
    "vascmtl_tbr_single",
    "vascmtl_cac_single",
)

V60_JOINT_AND_SINGLE_TASK_MODELS = {
    "lac_v60_joint_hps_mtl",
    "lac_v60_joint_shared_adapter_mtl",
    "lac_v60_single_task_tbr",
    "lac_v60_single_task_cac",
    "vascmtl",
    "vascmtl_no_baseline_anchoring",
    "vascmtl_no_cac_calibration",
    "vascmtl_tbr_single",
    "vascmtl_cac_single",
}

VASCMTL_MODELS = {
    "vascmtl",
    "vascmtl_no_baseline_anchoring",
    "vascmtl_no_cac_calibration",
    "vascmtl_tbr_single",
    "vascmtl_cac_single",
}

GENERIC_CONTEXT_MODELS = {
    "lac_v60_c_generic_temporal",
    "lac_v60_no_task_adapters",
    "lac_v60_no_baseline_anchoring",
    "lac_v60_no_oof_calibration",
    "lac_v60_shared_only_mtl",
    *V60_JOINT_AND_SINGLE_TASK_MODELS,
}


class V60SafeDecisionLayer(V42DecisionLayer):
    """V6-B adapter around the locked strict V4.2 selector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, cac_mode="strict", seed=2026, **kwargs)


def _seed_everything_v60(seed: int) -> None:
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
) -> LACV60Config:
    if model_name not in V60_MODELS:
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
    } | dict(config.get("model_v60", {}))
    overrides = {
        "lac_v60_a_no_i2c_tail": {},
        "lac_v60_b_no_mechanism_features": {},
        "lac_v60_c_generic_temporal": {},
        "lac_v60_no_task_adapters": {"soft_phenotype_adapters": False},
        "lac_v60_no_baseline_anchoring": {"baseline_anchoring": False},
        "lac_v60_no_oof_calibration": {},
        "lac_v60_shared_only_mtl": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
        },
        "lac_v60_joint_hps_mtl": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
            "training_objective": "joint_hps_mtl",
        },
        "lac_v60_joint_shared_adapter_mtl": {
            "shared_backbone_mtl": True,
            "soft_phenotype_adapters": True,
            "isolate_cac_shared_gradient": False,
            "training_objective": "joint_shared_adapter_mtl",
        },
        "lac_v60_single_task_tbr": {
            "training_objective": "single_task_tbr",
        },
        "lac_v60_single_task_cac": {
            "training_objective": "single_task_cac",
        },
        "vascmtl": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
            "training_objective": "joint_hps_mtl",
        },
        "vascmtl_no_baseline_anchoring": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
            "baseline_anchoring": False,
            "training_objective": "joint_hps_mtl",
        },
        "vascmtl_no_cac_calibration": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
            "training_objective": "joint_hps_mtl",
        },
        "vascmtl_tbr_single": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
            "training_objective": "single_task_tbr",
        },
        "vascmtl_cac_single": {
            "shared_only_mtl": True,
            "soft_phenotype_adapters": False,
            "isolate_cac_shared_gradient": False,
            "training_objective": "single_task_cac",
        },
    }
    return LACV60Config(**(options | overrides[model_name]))


def _feature_flags(model_name: str) -> dict[str, bool]:
    if model_name not in V60_MODELS:
        raise KeyError(model_name)
    return {
        "adapter": model_name not in {
            "lac_v60_no_task_adapters",
            "lac_v60_shared_only_mtl",
            "lac_v60_joint_hps_mtl",
            *VASCMTL_MODELS,
        },
        "coupling": False,
        "treatment": True,
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    if model_name not in V60_MODELS:
        raise KeyError(model_name)
    if model_name == "lac_v60_a_no_i2c_tail":
        return {"tbr": "identity", "cac": "no_tail"}
    if model_name == "lac_v60_b_no_mechanism_features":
        return {"tbr": "identity", "cac": "strict"}
    if model_name in {
        "lac_v60_no_oof_calibration",
        "vascmtl_no_cac_calibration",
    }:
        return {"tbr": "identity", "cac": "central"}
    return {"tbr": "identity", "cac": "full"}


def _generic_temporal_context(
    transformed: dict[str, np.ndarray],
) -> np.ndarray:
    """Create fixed, task-agnostic summaries without outcome information."""

    values = np.nan_to_num(
        np.asarray(transformed["values"], float),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )
    mask = np.asarray(transformed["mask"], float)
    delta = np.nan_to_num(
        np.asarray(transformed["delta"], float),
        nan=0.0,
        posinf=8.0,
        neginf=0.0,
    )
    count = mask.sum(axis=1)
    denominator = np.maximum(count, 1.0)
    mean = (values * mask).sum(axis=1) / denominator
    centered = (values - mean[:, None, :]) * mask
    standard_deviation = np.sqrt(
        np.square(centered).sum(axis=1) / denominator
    )
    first_index = mask.argmax(axis=1)
    last_index = values.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    first = np.take_along_axis(
        values, first_index[:, None, :], axis=1
    ).squeeze(1)
    last = np.take_along_axis(
        values, last_index[:, None, :], axis=1
    ).squeeze(1)
    observed = count > 0
    change = np.where(observed, last - first, 0.0)
    coverage = mask.mean(axis=1)
    mean_delta = (delta * mask).sum(axis=1) / denominator

    patch_count = mask.sum(axis=2)
    patch_denominator = np.maximum(patch_count, 1.0)
    patch_mean = (values * mask).sum(axis=2) / patch_denominator
    patch_centered = (values - patch_mean[:, :, None]) * mask
    patch_standard_deviation = np.sqrt(
        np.square(patch_centered).sum(axis=2) / patch_denominator
    )
    patch_coverage = mask.mean(axis=2)
    times = np.nan_to_num(
        np.asarray(transformed["times"], float),
        nan=0.0,
        posinf=8.0,
        neginf=0.0,
    )
    context = np.column_stack(
        [
            mean,
            standard_deviation,
            change,
            coverage,
            mean_delta,
            patch_mean,
            patch_standard_deviation,
            patch_coverage,
            times,
        ]
    )
    return np.nan_to_num(context, nan=0.0, posinf=8.0, neginf=-8.0)


def run_v60_nested_cross_validation(
    arrays: dict[str, Any],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V60_MODELS:
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
        "_predict_base_with_features",
    )
    original = {name: getattr(audited_runner, name) for name in names}
    generic_count = int(
        5 * arrays["values"].shape[-1] + 4 * arrays["values"].shape[1]
    )
    configured_generic = V60GenericDecisionLayer.configured_generic_feature_count
    configured_leaf = V60GenericDecisionLayer.configured_min_samples_leaf
    try:
        audited_runner.V27_MODELS = V60_MODELS
        audited_runner._config_for_model = _config_for_model
        audited_runner._feature_flags = _feature_flags
        audited_runner._decision_modes = _decision_modes
        if model_name == "lac_v60_a_no_i2c_tail":
            audited_runner.SafeOOFDecisionLayer = V40FinalSelector
        elif model_name == "lac_v60_b_no_mechanism_features":
            audited_runner.SafeOOFDecisionLayer = V60SafeDecisionLayer
        else:
            V60GenericDecisionLayer.configured_generic_feature_count = generic_count
            V60GenericDecisionLayer.configured_min_samples_leaf = int(
                config.get("decision", {}).get("generic_min_samples_leaf", 20)
            )
            audited_runner.SafeOOFDecisionLayer = V60GenericDecisionLayer
        if model_name in V60_JOINT_AND_SINGLE_TASK_MODELS:
            audited_runner.train_model = train_v60_multitask_model
            audited_runner.fit_model_fixed_epochs = fit_v60_multitask_fixed_epochs
        else:
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
        original_predict = original["_predict_base_with_features"]

        def predict_with_locked_context(*args, **kwargs):
            bundle = original_predict(*args, **kwargs)
            if model_name not in GENERIC_CONTEXT_MODELS:
                return bundle
            transformed = args[1] if len(args) > 1 else kwargs["transformed"]
            context = _generic_temporal_context(transformed)
            transformed_ids = [str(value) for value in transformed["patient_ids"]]
            index = {patient_id: row for row, patient_id in enumerate(transformed_ids)}
            order = np.asarray([index[str(value)] for value in bundle["patient_ids"]])
            ordered_context = context[order]
            if ordered_context.shape[1] != generic_count:
                raise RuntimeError("V6 generic temporal feature contract changed")
            bundle["features"]["cac"] = np.column_stack(
                [bundle["features"]["cac"], ordered_context]
            )
            return bundle

        audited_runner._predict_base_with_features = predict_with_locked_context
        offset = int(config.get("training_seed_offset", 0))
        audited_runner.seed_everything = (
            lambda value: _seed_everything_v60(int(value) + offset)
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
        V60GenericDecisionLayer.configured_generic_feature_count = configured_generic
        V60GenericDecisionLayer.configured_min_samples_leaf = configured_leaf
    summary["architecture_version"] = (
        "VascMTL-1.0" if model_name in VASCMTL_MODELS else "V6.0"
    )
    summary["development_status"] = (
        "locked sequential simplification on the repeatedly inspected internal "
        "443-patient cohort; not independent confirmation"
    )
    summary["cv_protocol"]["frozen_design"] = (
        "protected TBR, direct CAC, no explicit I-to-C/tail route, and an "
        "inner-OOF-only decision layer"
    )
    summary["cv_protocol"]["explicit_i_to_c_path"] = False
    summary["cv_protocol"]["tail_residual_path"] = False
    summary["cv_protocol"]["mechanism_derived_decision_features"] = bool(
        model_name == "lac_v60_a_no_i2c_tail"
    )
    summary["cv_protocol"]["generic_temporal_context"] = bool(
        model_name in GENERIC_CONTEXT_MODELS
    )
    summary["cv_protocol"]["generic_temporal_feature_count"] = (
        generic_count if model_name in GENERIC_CONTEXT_MODELS else 0
    )
    summary["cv_protocol"]["decision_mode"] = _decision_modes(model_name)
    summary["cv_protocol"]["training_objective"] = _config_for_model(
        schema, config, model_name
    ).training_objective
    summary["cv_protocol"]["outer_results_used_for_current_model_selection"] = False
    summary["cv_protocol"]["external_labels_used"] = False
    summary["cv_protocol"]["deterministic_execution"] = True
    oof_path = Path(output_dir, "out_of_fold_predictions.csv")
    if oof_path.is_file():
        oof = pd.read_csv(oof_path, dtype={"patient_id": str})
        for column in ("coupling_gate", "lag_elapsed_years"):
            if column in oof and np.any(oof[column].to_numpy(float) != 0.0):
                raise RuntimeError(f"Forbidden V6 mechanism diagnostic is nonzero: {column}")
    Path(output_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
