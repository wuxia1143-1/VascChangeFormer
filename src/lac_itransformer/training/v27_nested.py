from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from ..data.preprocessing import FoldPreprocessor, subset_arrays
from ..data.schema import FeatureSchema
from ..experiments.prediction import build_torch_model
from ..models.lac_v27 import LACV27Config
from .dataset import model_inputs
from .folds import (
    build_patient_fold_plan,
    fold_plan_checksum,
    indices_for_outer_fold,
    inner_calibration_indices,
    save_patient_fold_plan,
    validate_patient_fold_plan,
)
from .metrics import (
    bootstrap_change_metrics,
    bootstrap_metrics,
    change_space_metrics,
    regression_metrics,
)
from .trainer import (
    _git_revision,
    _loader,
    _parameter_count,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    resolve_device,
    seed_everything,
    train_model,
)
from .v27_decision import SafeOOFDecisionLayer


V27_MODELS = (
    "lac_v27_full",
    "lac_v27_median_only",
    "lac_v27_mean_only",
    "lac_v27_no_adapters",
    "lac_v27_no_coupling",
    "lac_v27_no_treatment",
    "lac_v27_no_decision_gate",
    "lac_v27_cac_central_only",
    "lac_v27_cac_lag_mean_only",
    "lac_v27_no_stop_gradient",
)


def _config_for_model(
    schema: FeatureSchema,
    config: dict[str, Any],
    model_name: str,
) -> LACV27Config:
    options = {
        "static_dim": len(schema.static_features),
        "num_variables": len(schema.longitudinal_features),
        "treatment_dim": len(schema.treatment_features),
        "num_patches": schema.time_patches,
    } | dict(config.get("model_v27", {}))
    if model_name == "lac_v27_no_adapters":
        options["phenotype_adapters"] = False
    elif model_name == "lac_v27_no_coupling":
        options["coupling_enabled"] = False
    elif model_name == "lac_v27_no_treatment":
        options["treatment_conditioning"] = False
    elif model_name == "lac_v27_no_stop_gradient":
        options["stop_gradient_lag_source"] = False
        options["isolate_cac_shared_gradient"] = False
    elif model_name not in V27_MODELS:
        raise KeyError(model_name)
    return LACV27Config(**options)


def _feature_flags(model_name: str) -> dict[str, bool]:
    return {
        "adapter": model_name != "lac_v27_no_adapters",
        "coupling": model_name != "lac_v27_no_coupling",
        "treatment": model_name
        not in {"lac_v27_no_coupling", "lac_v27_no_treatment"},
    }


def _decision_modes(model_name: str) -> dict[str, str]:
    if model_name == "lac_v27_median_only":
        return {"tbr": "median", "cac": "median"}
    if model_name == "lac_v27_mean_only":
        return {"tbr": "mean", "cac": "mean"}
    if model_name == "lac_v27_no_decision_gate":
        return {"tbr": "fixed_half", "cac": "fixed_half"}
    if model_name == "lac_v27_cac_central_only":
        return {"tbr": "full", "cac": "median"}
    if model_name == "lac_v27_cac_lag_mean_only":
        return {"tbr": "full", "cac": "mean"}
    return {"tbr": "full", "cac": "full"}


def _residual_to_endpoint(
    baseline: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            baseline[:, 0] + residual[:, 0],
            np.maximum(
                0,
                np.expm1(
                    np.log1p(np.maximum(baseline[:, 1], 0))
                    + residual[:, 1]
                ),
            ),
        ]
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "fraction_near_zero": float(np.mean(values <= 0.05)),
        "fraction_near_one": float(np.mean(values >= 0.95)),
    }


def _v27_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    true_tbr = frame["true_delta_tbr"].to_numpy(float)
    pred_tbr = frame["pred_delta_tbr"].to_numpy(float)
    true_cac = frame["true_delta_log_cac"].to_numpy(float)
    pred_cac = frame["pred_delta_log_cac"].to_numpy(float)
    tail_cutoff = float(np.quantile(np.abs(true_cac), 0.90))
    tail = np.abs(true_cac) >= tail_cutoff
    return {
        "prediction_variance_over_true_variance": {
            "delta_tbr": float(
                np.var(pred_tbr, ddof=1)
                / max(np.var(true_tbr, ddof=1), 1e-12)
            ),
            "delta_log_cac": float(
                np.var(pred_cac, ddof=1)
                / max(np.var(true_cac, ddof=1), 1e-12)
            ),
        },
        "direction_accuracy": {
            "delta_tbr": float(
                np.mean((pred_tbr > 0) == (true_tbr > 0))
            ),
            "delta_log_cac": float(
                np.mean((pred_cac > 0) == (true_cac > 0))
            ),
        },
        "cac_top_10_percent_absolute_change": {
            "threshold": tail_cutoff,
            "patient_count": int(tail.sum()),
            "mae": float(np.mean(np.abs(pred_cac[tail] - true_cac[tail]))),
            "rmse": float(
                np.sqrt(np.mean(np.square(pred_cac[tail] - true_cac[tail])))
            ),
        },
        "gate_distributions": {
            column: _distribution(frame[column].to_numpy(float))
            for column in (
                "decision_gate_tbr",
                "decision_gate_cac",
                "adapter_gate_inflammation",
                "adapter_gate_calcification",
                "coupling_gate",
            )
        },
    }


@torch.no_grad()
def _predict_base_with_features(
    model: torch.nn.Module,
    transformed: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    model_name: str,
) -> dict[str, Any]:
    model.eval()
    raw_index = {
        str(patient_id): index
        for index, patient_id in enumerate(raw["patient_ids"])
    }
    head_predictions = {"tbr": [], "cac": []}
    targets = []
    baselines = []
    ids: list[str] = []
    tbr_features = []
    cac_features = []
    neural_diagnostics = {
        "adapter_i": [],
        "adapter_c": [],
        "coupling_gate": [],
        "lag_elapsed": [],
    }
    flags = _feature_flags(model_name)
    for batch in _loader(transformed, batch_size, False):
        batch = batch.to(device)
        output = model(**model_inputs(batch))
        batch_ids = [str(value) for value in batch.patient_ids]
        indices = np.asarray([raw_index[value] for value in batch_ids])
        raw_baseline = raw["baseline"][indices].astype(float)
        followup = raw["followup_months"][indices].astype(float) / 12.0
        observation_density = raw["mask"][indices].mean(axis=(1, 2))
        raw_treatment = raw["treatments"][indices].max(axis=1).astype(float)
        if not flags["treatment"]:
            raw_treatment = np.zeros_like(raw_treatment)
        window = np.digitize(
            raw["followup_months"][indices].astype(float),
            [6, 12, 18, 24],
            right=True,
        )
        window_one_hot = np.eye(5)[window]
        tbr_heads = torch.stack(
            [output["delta_tbr_median"], output["delta_tbr_mean"]],
            dim=-1,
        ).cpu().numpy()
        cac_heads = torch.stack(
            [
                output["delta_log_cac_median"],
                output["delta_log_cac_mean"],
            ],
            dim=-1,
        ).cpu().numpy()
        progression = torch.sigmoid(
            output["cac_progression_logit"]
        ).cpu().numpy()
        magnitude = output["cac_change_magnitude"].cpu().numpy()
        if "reliability_gate" in output:
            # V5.0 deliberately has no I-to-C or lag pathway.  Its detached
            # CAC residual is controlled by a scalar statistical reliability
            # gate, which occupies the legacy decision-feature slots only for
            # compatibility with the audited nested-CV runner.
            reliability_gate = output["reliability_gate"].cpu().numpy()
            coupling_gate = reliability_gate
            coupling_gate_max = reliability_gate
            lag_elapsed = np.zeros_like(reliability_gate)
        else:
            coupling_gate = output[
                "gate_inflammation_to_calcification"
            ].mean(dim=1).cpu().numpy()
            coupling_gate_max = output[
                "gate_inflammation_to_calcification"
            ].amax(dim=1).cpu().numpy()
            lag_elapsed = (
                output["lag_attention"] * output["lag_time_deltas"]
            ).sum(dim=-1).mean(dim=-1).cpu().numpy()
        adapter_c = output["phenotype_gate_calcification"].mean(
            dim=(1, 2, 3)
        ).cpu().numpy()
        adapter_i = output["phenotype_gate_inflammation"].mean(
            dim=(1, 2, 3)
        ).cpu().numpy()
        if not flags["coupling"]:
            coupling_gate = np.zeros_like(coupling_gate)
            coupling_gate_max = np.zeros_like(coupling_gate_max)
            lag_elapsed = np.zeros_like(lag_elapsed)
        if not flags["adapter"]:
            adapter_c = np.zeros_like(adapter_c)
        tbr_features.append(
            np.column_stack(
                [
                    tbr_heads[:, 0],
                    tbr_heads[:, 1],
                    np.abs(tbr_heads[:, 1] - tbr_heads[:, 0]),
                    raw_baseline[:, 0],
                    followup,
                    observation_density,
                ]
            )
        )
        cac_features.append(
            np.column_stack(
                [
                    cac_heads[:, 0],
                    cac_heads[:, 1],
                    np.abs(cac_heads[:, 1] - cac_heads[:, 0]),
                    np.log1p(np.maximum(raw_baseline[:, 1], 0)),
                    raw_baseline[:, 0],
                    followup,
                    (raw_baseline[:, 1] > 0).astype(float),
                    window_one_hot,
                    observation_density,
                    raw_treatment,
                    progression,
                    magnitude,
                    coupling_gate,
                    coupling_gate_max,
                    lag_elapsed,
                    adapter_c,
                ]
            )
        )
        head_predictions["tbr"].append(tbr_heads)
        head_predictions["cac"].append(cac_heads)
        neural_diagnostics["adapter_i"].append(adapter_i)
        neural_diagnostics["adapter_c"].append(adapter_c)
        neural_diagnostics["coupling_gate"].append(coupling_gate)
        neural_diagnostics["lag_elapsed"].append(lag_elapsed)
        targets.append(raw["targets"][indices].astype(float))
        baselines.append(raw_baseline)
        ids.extend(batch_ids)
    baseline = np.concatenate(baselines)
    endpoint_targets = np.concatenate(targets)
    residual_targets = np.column_stack(
        [
            endpoint_targets[:, 0] - baseline[:, 0],
            np.log1p(np.maximum(endpoint_targets[:, 1], 0))
            - np.log1p(np.maximum(baseline[:, 1], 0)),
        ]
    )
    return {
        "patient_ids": ids,
        "baseline": baseline,
        "endpoint_targets": endpoint_targets,
        "residual_targets": residual_targets,
        "heads": {
            task: np.concatenate(values)
            for task, values in head_predictions.items()
        },
        "features": {
            "tbr": np.concatenate(tbr_features),
            "cac": np.concatenate(cac_features),
        },
        "neural_diagnostics": {
            name: np.concatenate(values)
            for name, values in neural_diagnostics.items()
        },
    }


def _inner_oof_predictions(
    train_pool_raw: dict[str, np.ndarray],
    schema: FeatureSchema,
    inner_plan: Mapping[str, int],
    config: dict[str, Any],
    model_name: str,
    outer_fold: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    patient_ids = train_pool_raw["patient_ids"]
    records = []
    bundles = []
    selected_epochs = []
    model_config = _config_for_model(schema, config, model_name)
    for inner_fold in range(1, 6):
        train_index, validation_index = indices_for_outer_fold(
            patient_ids,
            inner_plan,
            inner_fold,
        )
        inner_train_raw = subset_arrays(train_pool_raw, train_index)
        inner_validation_raw = subset_arrays(
            train_pool_raw,
            validation_index,
        )
        fit_seed = seed + outer_fold * 100_000 + inner_fold * 1_000
        early_train_index, early_validation_index = (
            inner_calibration_indices(
                inner_train_raw["patient_ids"],
                float(config.get("inner_validation_fraction", 0.15)),
                fit_seed,
            )
        )
        early_train_raw = subset_arrays(
            inner_train_raw,
            early_train_index,
        )
        early_validation_raw = subset_arrays(
            inner_train_raw,
            early_validation_index,
        )
        early_preprocessor = FoldPreprocessor.fit(
            early_train_raw,
            patient_id_hash(early_train_raw["patient_ids"]),
        )
        early_train = early_preprocessor.transform(early_train_raw)
        early_validation = early_preprocessor.transform(
            early_validation_raw
        )
        seed_everything(fit_seed)
        selection_model = build_torch_model(
            model_name,
            model_config,
        ).to(device)
        selection_model, selection_history = train_model(
            selection_model,
            early_train,
            early_validation,
            config,
            device,
            fit_seed,
        )
        epochs = max(1, int(selection_history["best_epoch"]))
        selected_epochs.append(epochs)
        del selection_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        inner_preprocessor = FoldPreprocessor.fit(
            inner_train_raw,
            patient_id_hash(inner_train_raw["patient_ids"]),
        )
        inner_train = inner_preprocessor.transform(inner_train_raw)
        inner_validation = inner_preprocessor.transform(
            inner_validation_raw
        )
        refit_seed = fit_seed + 99
        seed_everything(refit_seed)
        refit_model = build_torch_model(model_name, model_config).to(device)
        refit_model, refit_history = fit_model_fixed_epochs(
            refit_model,
            inner_train,
            config,
            device,
            epochs,
            refit_seed,
        )
        bundle = _predict_base_with_features(
            refit_model,
            inner_validation,
            inner_validation_raw,
            int(config.get("batch_size", 64)),
            device,
            model_name,
        )
        bundle["meta_fold"] = np.full(
            len(bundle["patient_ids"]),
            inner_fold,
            dtype=int,
        )
        bundles.append(bundle)
        records.append(
            {
                "inner_fold": inner_fold,
                "selection_seed": fit_seed,
                "refit_seed": refit_seed,
                "n_early_train": len(early_train_index),
                "n_early_validation": len(early_validation_index),
                "n_inner_refit": len(train_index),
                "n_inner_oof": len(validation_index),
                "selected_epochs": epochs,
                "early_train_patient_hash": patient_id_hash(
                    early_train_raw["patient_ids"]
                ),
                "early_validation_patient_hash": patient_id_hash(
                    early_validation_raw["patient_ids"]
                ),
                "inner_oof_patient_hash": patient_id_hash(
                    inner_validation_raw["patient_ids"]
                ),
                "inner_oof_used_for_early_stopping": False,
                "target_scaler": refit_history.get("target_scaler"),
                "sampling": refit_history.get("sampling"),
            }
        )
        del refit_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = {
        "patient_ids": sum(
            [bundle["patient_ids"] for bundle in bundles],
            [],
        ),
        "baseline": np.concatenate(
            [bundle["baseline"] for bundle in bundles]
        ),
        "endpoint_targets": np.concatenate(
            [bundle["endpoint_targets"] for bundle in bundles]
        ),
        "residual_targets": np.concatenate(
            [bundle["residual_targets"] for bundle in bundles]
        ),
        "heads": {
            task: np.concatenate(
                [bundle["heads"][task] for bundle in bundles]
            )
            for task in ("tbr", "cac")
        },
        "features": {
            task: np.concatenate(
                [bundle["features"][task] for bundle in bundles]
            )
            for task in ("tbr", "cac")
        },
        "meta_fold": np.concatenate(
            [bundle["meta_fold"] for bundle in bundles]
        ),
        "neural_diagnostics": {
            name: np.concatenate(
                [bundle["neural_diagnostics"][name] for bundle in bundles]
            )
            for name in (
                "adapter_i",
                "adapter_c",
                "coupling_gate",
                "lag_elapsed",
            )
        },
    }
    if set(combined["patient_ids"]) != set(
        str(value) for value in patient_ids
    ):
        raise RuntimeError("Inner OOF decision predictions are incomplete")
    if len(combined["patient_ids"]) != len(set(combined["patient_ids"])):
        raise RuntimeError("Inner OOF decision patients must be unique")
    return (
        combined,
        records,
        max(1, int(np.rint(np.median(selected_epochs)))),
    )


def run_v27_nested_cross_validation(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    model_name: str,
    outer_fold_assignments: Mapping[str, int] | None = None,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V27_MODELS:
        raise KeyError(model_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    device = resolve_device(str(config.get("device", "auto")))
    patient_ids = np.asarray([str(value) for value in arrays["patient_ids"]])
    outer_plan = dict(
        outer_fold_assignments
        or build_patient_fold_plan(patient_ids, 5, seed)
    )
    validate_patient_fold_plan(patient_ids, outer_plan, 5)
    outer_checksum = fold_plan_checksum(outer_plan)
    outer_manifest = save_patient_fold_plan(
        outer_plan,
        output / "patient_outer_fold_plan.csv",
        seed=seed,
        n_folds=5,
    )
    model_training_config = dict(config) | {"model_name": model_name}
    (output / "training_config.json").write_text(
        json.dumps(model_training_config, indent=2),
        encoding="utf-8",
    )
    all_predictions = []
    fold_records = []
    model_config = _config_for_model(schema, config, model_name)
    selected_outer_folds = tuple(int(value) for value in outer_folds)
    if not selected_outer_folds or not set(selected_outer_folds) <= set(
        range(1, 6)
    ):
        raise ValueError("outer_folds must be a nonempty subset of 1..5")
    for outer_fold in selected_outer_folds:
        fold_started = time.perf_counter()
        train_index, test_index = indices_for_outer_fold(
            patient_ids,
            outer_plan,
            outer_fold,
        )
        train_pool_raw = subset_arrays(arrays, train_index)
        test_raw = subset_arrays(arrays, test_index)
        if set(train_pool_raw["patient_ids"]) & set(test_raw["patient_ids"]):
            raise RuntimeError("Outer patient leakage")
        fold_dir = output / f"fold_{outer_fold}"
        fold_dir.mkdir(exist_ok=True)
        inner_seed = seed + outer_fold * 1_000
        inner_plan = build_patient_fold_plan(
            train_pool_raw["patient_ids"],
            5,
            inner_seed,
        )
        inner_manifest = save_patient_fold_plan(
            inner_plan,
            fold_dir / "inner_patient_fold_plan.csv",
            seed=inner_seed,
            n_folds=5,
        )
        inner_started = time.perf_counter()
        inner_oof, inner_records, selected_epochs = (
            _inner_oof_predictions(
                train_pool_raw,
                schema,
                inner_plan,
                config,
                model_name,
                outer_fold,
                seed,
                device,
            )
        )
        inner_seconds = time.perf_counter() - inner_started
        decision = SafeOOFDecisionLayer(
            candidate_weights=tuple(
                config.get("decision", {}).get(
                    "candidate_weights",
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1.0],
                )
            ),
            ridge_alphas=tuple(
                config.get("decision", {}).get(
                    "ridge_alphas",
                    [1.0, 10.0],
                )
            ),
            degradation_tolerance=float(
                config.get("decision", {}).get(
                    "degradation_tolerance",
                    0.02,
                )
            ),
            required_consistent_folds=int(
                config.get("decision", {}).get(
                    "required_consistent_folds",
                    4,
                )
            ),
        ).fit(
            heads=inner_oof["heads"],
            targets={
                "tbr": inner_oof["residual_targets"][:, 0],
                "cac": inner_oof["residual_targets"][:, 1],
            },
            features=inner_oof["features"],
            meta_fold=inner_oof["meta_fold"],
        )
        decision_audit = decision.audit()
        (fold_dir / "decision_audit.json").write_text(
            json.dumps(decision_audit, indent=2),
            encoding="utf-8",
        )
        with (fold_dir / "decision_layer.pkl").open("wb") as handle:
            pickle.dump(decision, handle)

        outer_preprocessor = FoldPreprocessor.fit(
            train_pool_raw,
            patient_id_hash(train_pool_raw["patient_ids"]),
        )
        outer_train = outer_preprocessor.transform(train_pool_raw)
        outer_test = outer_preprocessor.transform(test_raw)
        final_seed = seed + outer_fold * 100_000 + 99_999
        seed_everything(final_seed)
        model = build_torch_model(model_name, model_config).to(device)
        model, refit_history = fit_model_fixed_epochs(
            model,
            outer_train,
            config,
            device,
            selected_epochs,
            final_seed,
        )
        test_bundle = _predict_base_with_features(
            model,
            outer_test,
            test_raw,
            int(config.get("batch_size", 64)),
            device,
            model_name,
        )
        decided_residual, decision_gates = decision.predict(
            test_bundle["heads"],
            test_bundle["features"],
            _decision_modes(model_name),
        )
        decided_endpoint = _residual_to_endpoint(
            test_bundle["baseline"],
            decided_residual,
        )
        robust_residual = np.column_stack(
            [
                test_bundle["heads"]["tbr"][:, 0],
                test_bundle["heads"]["cac"][:, 0],
            ]
        )
        robust_endpoint = _residual_to_endpoint(
            test_bundle["baseline"],
            robust_residual,
        )
        targets = test_bundle["endpoint_targets"]
        metrics = regression_metrics(targets, decided_endpoint)
        change = change_space_metrics(
            targets,
            decided_endpoint,
            test_bundle["baseline"],
        )
        robust_metrics = regression_metrics(targets, robust_endpoint)
        robust_change = change_space_metrics(
            targets,
            robust_endpoint,
            test_bundle["baseline"],
        )
        frame = pd.DataFrame(
            {
                "patient_id": test_bundle["patient_ids"],
                "outer_fold": outer_fold,
                "baseline_tbr": test_bundle["baseline"][:, 0],
                "baseline_cac": test_bundle["baseline"][:, 1],
                "true_tbr": targets[:, 0],
                "pred_tbr": decided_endpoint[:, 0],
                "true_cac": targets[:, 1],
                "pred_cac": decided_endpoint[:, 1],
                "robust_pred_tbr": robust_endpoint[:, 0],
                "robust_pred_cac": robust_endpoint[:, 1],
                "median_pred_delta_tbr": test_bundle["heads"]["tbr"][:, 0],
                "mean_pred_delta_tbr": test_bundle["heads"]["tbr"][:, 1],
                "median_pred_delta_log_cac": test_bundle["heads"]["cac"][:, 0],
                "mean_pred_delta_log_cac": test_bundle["heads"]["cac"][:, 1],
                "decision_gate_tbr": decision_gates["tbr"],
                "decision_gate_cac": decision_gates["cac"],
                "adapter_gate_inflammation": test_bundle[
                    "neural_diagnostics"
                ]["adapter_i"],
                "adapter_gate_calcification": test_bundle[
                    "neural_diagnostics"
                ]["adapter_c"],
                "coupling_gate": test_bundle["neural_diagnostics"][
                    "coupling_gate"
                ],
                "lag_elapsed_years": test_bundle["neural_diagnostics"][
                    "lag_elapsed"
                ],
            }
        )
        frame["true_delta_tbr"] = (
            frame["true_tbr"] - frame["baseline_tbr"]
        )
        frame["pred_delta_tbr"] = (
            frame["pred_tbr"] - frame["baseline_tbr"]
        )
        frame["true_delta_log_cac"] = np.log1p(
            np.maximum(frame["true_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        frame["pred_delta_log_cac"] = np.log1p(
            np.maximum(frame["pred_cac"], 0)
        ) - np.log1p(np.maximum(frame["baseline_cac"], 0))
        all_predictions.append(frame)
        outer_preprocessor.save(fold_dir / "preprocessor.json")
        (fold_dir / "inner_selection.json").write_text(
            json.dumps(inner_records, indent=2),
            encoding="utf-8",
        )
        (fold_dir / "refit_history.json").write_text(
            json.dumps(refit_history, indent=2),
            encoding="utf-8",
        )
        split_manifest = {
            "outer_test_fold": outer_fold,
            "n_outer_train": len(train_index),
            "n_outer_test": len(test_index),
            "outer_fold_plan_checksum": outer_checksum,
            "inner_fold_plan": inner_manifest,
            "selected_epochs": selected_epochs,
            "test_fold_used_for_preprocessing": False,
            "test_fold_used_for_inner_base_training": False,
            "test_fold_used_for_decision_fit": False,
            "test_fold_used_for_decision_selection": False,
            "test_fold_used_for_outer_refit": False,
        }
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2),
            encoding="utf-8",
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "model_config": model.config.to_dict(),
                "schema": schema.to_dict(),
                "training_config": model_training_config,
                "git_revision": _git_revision(),
                "outer_fold": outer_fold,
            },
            fold_dir / "model.pt",
        )
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "selected_epochs": selected_epochs,
                "parameter_count": _parameter_count(model),
                "inner_oof_and_selection_seconds": inner_seconds,
                "total_fold_seconds": time.perf_counter() - fold_started,
                "decision_method_tbr": decision_audit[
                    "unique_fixed_configuration"
                ]["tbr"]["method"],
                "decision_method_cac": decision_audit[
                    "unique_fixed_configuration"
                ]["cac"]["method"],
                "decision_constraints_passed": decision_audit[
                    "all_constraints_passed"
                ],
            }
            | metrics
            | change
            | {
                f"robust_{key}": value
                for key, value in (
                    robust_metrics | robust_change
                ).items()
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(all_predictions, ignore_index=True)
    plan_values = np.asarray(list(outer_plan.values()))
    expected = int(
        sum(
            np.count_nonzero(plan_values == fold)
            for fold in selected_outer_folds
        )
    )
    if len(oof) != expected or oof["patient_id"].nunique() != expected:
        raise RuntimeError(
            "V2.7 OOF must contain every selected-fold patient exactly once"
        )
    oof = oof.sort_values("patient_id").reset_index(drop=True)
    oof.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(
        output / "outer_fold_metrics.csv",
        index=False,
    )
    pooled_targets = oof[["true_tbr", "true_cac"]].to_numpy()
    pooled_prediction = oof[["pred_tbr", "pred_cac"]].to_numpy()
    pooled_baseline = oof[["baseline_tbr", "baseline_cac"]].to_numpy()
    pooled_metrics = regression_metrics(pooled_targets, pooled_prediction)
    pooled_change = change_space_metrics(
        pooled_targets,
        pooled_prediction,
        pooled_baseline,
    )
    bootstrap_replicates = int(config.get("bootstrap_replicates", 2000))
    summary = {
        "model_name": model_name,
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
        "patient_count": len(oof),
        "parameter_count": int(
            np.median(
                [record["parameter_count"] for record in fold_records]
            )
        ),
        "cv_protocol": {
            "level": "patient",
            "outer_folds": 5,
            "outer_folds_evaluated": list(selected_outer_folds),
            "inner_folds": 5,
            "decision_layer": (
                "single final prediction selected from robust and mean heads "
                "using untouched inner OOF predictions only"
            ),
            "outer_test_used_only_for_final_evaluation": True,
            "outer_fold_plan_checksum": outer_checksum,
            "outer_fold_plan": outer_manifest,
        },
        "fold_records": fold_records,
        "pooled_oof_metrics": pooled_metrics,
        "pooled_change_metrics": pooled_change,
        "diagnostics": _v27_diagnostics(oof),
        "bootstrap_95_ci": bootstrap_metrics(
            pooled_targets,
            pooled_prediction,
            n_bootstrap=bootstrap_replicates,
            seed=seed,
        ),
        "change_bootstrap_95_ci": bootstrap_change_metrics(
            pooled_targets,
            pooled_prediction,
            pooled_baseline,
            n_bootstrap=bootstrap_replicates,
            seed=seed,
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "development_status": "iterative internal development validation",
        "external_validation_status": "not_run",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
