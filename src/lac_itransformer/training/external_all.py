from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from ..data.external_validation import ExternalValidationGuardrailError, ExternalWorkbookReader
from ..data.preprocessing import FoldPreprocessor
from ..data.schema import FeatureSchema
from ..experiments.comparison import PRESPECIFIED_MODELS
from ..experiments.prediction import build_classical_model, build_torch_model
from ..models.lac import LACConfig
from .metrics import bootstrap_metrics, regression_metrics
from .folds import fold_plan_checksum
from .trainer import (
    _git_revision,
    _loader,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    predict,
    resolve_device,
    seed_everything,
)


CLASSICAL_MODELS = {"persistence", "elastic_net", "xgboost"}
DEEP_MODELS = set(PRESPECIFIED_MODELS) - CLASSICAL_MODELS


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty locked output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_arrays(
    arrays: dict[str, np.ndarray], schema: FeatureSchema, cohort_role: str
) -> None:
    required = {
        "static", "baseline", "values", "mask", "delta", "times",
        "treatments", "targets", "patient_ids",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"{cohort_role} arrays are missing keys: {sorted(missing)}")
    n = len(arrays["patient_ids"])
    if n < 1:
        raise ValueError(f"{cohort_role} cohort is empty")
    ids = [str(value) for value in arrays["patient_ids"]]
    if len(set(ids)) != n:
        raise ValueError(f"{cohort_role} patient identifiers must be unique")
    for key in required - {"patient_ids"}:
        if len(arrays[key]) != n:
            raise ValueError(f"{cohort_role} key {key} is not patient-aligned")
    expected = {
        "static": (n, len(schema.static_features)),
        "baseline": (n, 2),
        "values": (n, schema.time_patches, len(schema.longitudinal_features)),
        "mask": (n, schema.time_patches, len(schema.longitudinal_features)),
        "delta": (n, schema.time_patches, len(schema.longitudinal_features)),
        "times": (n, schema.time_patches),
        "treatments": (n, schema.time_patches, len(schema.treatment_features)),
        "targets": (n, 2),
    }
    for key, shape in expected.items():
        if tuple(np.asarray(arrays[key]).shape) != shape:
            raise ValueError(
                f"{cohort_role} key {key} expected shape {shape}, got {np.asarray(arrays[key]).shape}"
            )
    if not np.isfinite(arrays["baseline"]).all():
        raise ValueError(f"{cohort_role} baseline endpoints contain missing/non-finite values")
    if not np.isfinite(arrays["targets"]).all():
        raise ValueError(f"{cohort_role} outcome endpoints contain missing/non-finite values")
    if (np.asarray(arrays["baseline"])[:, 0] <= 0).any() or (
        np.asarray(arrays["targets"])[:, 0] <= 0
    ).any():
        raise ValueError(f"{cohort_role} TBR endpoints must be positive")
    if (np.asarray(arrays["baseline"])[:, 1] < 0).any() or (
        np.asarray(arrays["targets"])[:, 1] < 0
    ).any():
        raise ValueError(f"{cohort_role} CAC endpoints must be non-negative")


def _locked_epoch_count(fold_records: list[dict[str, Any]]) -> tuple[int, list[int]]:
    epochs = [int(record["selected_epochs"]) for record in fold_records]
    if len(epochs) != 5 or any(value < 1 for value in epochs):
        raise ValueError("Deep-model internal validation must contain five positive selected epochs")
    return int(np.median(epochs)), epochs


def _internal_fold_metadata(model_name: str, fold_dir: Path) -> dict[str, Any]:
    if model_name in CLASSICAL_MODELS:
        path = fold_dir / "model_meta.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing internal model metadata: {path}")
        return _read_json(path)
    path = fold_dir / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing internal checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def freeze_internal_selection(
    internal_cv_dir: str | Path,
    development_arrays: dict[str, np.ndarray],
    development_schema: FeatureSchema,
    models: Iterable[str] = PRESPECIFIED_MODELS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate center-B OOF artifacts and freeze choices before full-data refit."""
    root = Path(internal_cv_dir)
    selected = tuple(dict.fromkeys(models))
    unknown = set(selected) - set(PRESPECIFIED_MODELS)
    if unknown:
        raise KeyError(f"Only the eight prespecified models can be finalized: {sorted(unknown)}")
    suite_manifest = _read_json(root / "internal_fivefold_manifest.json")
    training_config = _read_json(root / "training_config.json")
    if int(training_config.get("num_folds", 0)) != 5:
        raise ValueError("Center-B internal validation must use exactly five folds")
    if suite_manifest.get("external_center_used") is not False:
        raise ExternalValidationGuardrailError(
            "Internal model selection is not certified as independent of the external center"
        )
    if suite_manifest.get("training_config_sha256") != _canonical_hash(training_config):
        raise ExternalValidationGuardrailError(
            "Internal training configuration is missing its lock or its checksum changed"
        )
    if not set(selected).issubset(set(suite_manifest.get("models", []))):
        raise ValueError("Requested final models are absent from the internal five-fold suite")
    plan = pd.read_csv(root / "patient_fold_plan.csv", dtype={"patient_id": str})
    development_ids = {str(value) for value in development_arrays["patient_ids"]}
    if set(plan["patient_id"]) != development_ids:
        raise ValueError("Center-B full-refit cohort does not match the internal five-fold cohort")
    expected_checksum = suite_manifest["fold_plan"]["checksum_sha256"]
    observed_checksum = fold_plan_checksum(
        dict(zip(plan["patient_id"], plan["test_fold"]))
    )
    if observed_checksum != expected_checksum:
        raise ExternalValidationGuardrailError("Internal patient fold table checksum changed")
    if suite_manifest["fold_plan"]["n_folds"] != 5:
        raise ValueError("Internal fold manifest is not five-fold")
    frozen: dict[str, Any] = {}
    source_revisions = set()
    for model_name in selected:
        model_dir = root / model_name
        summary = _read_json(model_dir / "summary.json")
        protocol = summary["cv_protocol"]
        if protocol["outer_folds"] != 5 or protocol["fold_plan_checksum"] != expected_checksum:
            raise ValueError(f"{model_name} does not use the locked center-B five-fold plan")
        if summary.get("external_validation_status") != "not_run":
            raise ExternalValidationGuardrailError(
                f"{model_name} was not frozen before external validation"
            )
        expected_model_training_config = dict(training_config)
        expected_model_training_config["model_name"] = model_name
        if _canonical_hash(_read_json(model_dir / "training_config.json")) != _canonical_hash(
            expected_model_training_config
        ):
            raise ExternalValidationGuardrailError(
                f"Internal training configuration mismatch for {model_name}"
            )
        fold_dirs = sorted(path for path in model_dir.glob("fold_*") if path.is_dir())
        if len(fold_dirs) != 5:
            raise ValueError(f"{model_name} must have exactly five internal fold artifacts")
        metadata = [_internal_fold_metadata(model_name, path) for path in fold_dirs]
        if any(item["model_name"] != model_name for item in metadata):
            raise ValueError(f"Internal artifact identity mismatch for {model_name}")
        schema_hashes = {_canonical_hash(item["schema"]) for item in metadata}
        config_hashes = {_canonical_hash(item["model_config"]) for item in metadata}
        if schema_hashes != {_canonical_hash(development_schema.to_dict())}:
            raise ValueError(f"Center-B schema mismatch for {model_name}")
        if len(config_hashes) != 1:
            raise ValueError(f"Architecture changed across internal folds for {model_name}")
        fold_record_by_number = {
            int(record["fold"]): record for record in summary["fold_records"]
        }
        for fold_dir in fold_dirs:
            fold_number = int(fold_dir.name.split("_")[-1])
            split = _read_json(fold_dir / "split_manifest.json")
            if split["fold_plan_checksum"] != expected_checksum:
                raise ExternalValidationGuardrailError(
                    f"Internal fold checksum mismatch for {model_name} fold {fold_number}"
                )
            if any(
                split[key]
                for key in (
                    "test_fold_used_for_preprocessing",
                    "test_fold_used_for_epoch_selection",
                    "test_fold_used_for_training",
                )
            ):
                raise ExternalValidationGuardrailError(
                    f"Internal leakage flag set for {model_name} fold {fold_number}"
                )
            if model_name in DEEP_MODELS and int(split["selected_epochs"]) != int(
                fold_record_by_number[fold_number]["selected_epochs"]
            ):
                raise ExternalValidationGuardrailError(
                    f"Selected epoch mismatch for {model_name} fold {fold_number}"
                )
        source_revisions.add(summary["git_revision"])
        locked_epochs = None
        fold_epochs: list[int] = []
        if model_name in DEEP_MODELS:
            locked_epochs, fold_epochs = _locked_epoch_count(summary["fold_records"])
        frozen[model_name] = {
            "model_name": model_name,
            "model_config": metadata[0]["model_config"],
            "architecture_sha256": next(iter(config_hashes)),
            "internal_fold_selected_epochs": fold_epochs,
            "locked_final_training_epochs": locked_epochs,
            "epoch_lock_rule": "median_of_five_outer_fold_selected_epochs" if locked_epochs else None,
            "internal_oof_metrics": summary["pooled_oof_metrics"],
            "internal_summary_sha256": file_sha256(model_dir / "summary.json"),
            "internal_git_revision": summary["git_revision"],
        }
    if len(source_revisions) != 1:
        raise ExternalValidationGuardrailError(
            "The eight internal models were not produced by one frozen code revision"
        )
    return training_config, frozen, {
        "internal_suite_manifest_sha256": file_sha256(root / "internal_fivefold_manifest.json"),
        "internal_fold_plan_checksum": expected_checksum,
        "internal_source_git_revisions": sorted(source_revisions),
        "internal_result_scope": suite_manifest.get("result_scope", ""),
        "development_patient_count": len(development_ids),
        "development_patient_hash": patient_id_hash(np.asarray(sorted(development_ids))),
    }


def finalize_models_for_external_validation(
    development_arrays: dict[str, np.ndarray],
    development_schema: FeatureSchema,
    internal_cv_dir: str | Path,
    output_dir: str | Path,
    models: Iterable[str] = PRESPECIFIED_MODELS,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Refit eight locked models on all center-B patients, never center A."""
    _validate_arrays(development_arrays, development_schema, "center_B_development")
    selected = tuple(dict.fromkeys(models))
    training_config, frozen, internal_source = freeze_internal_selection(
        internal_cv_dir, development_arrays, development_schema, selected
    )
    output = _prepare_output(output_dir)
    device = resolve_device(device_name)
    development_hash = patient_id_hash(development_arrays["patient_ids"])
    preprocessor = FoldPreprocessor.fit(development_arrays, development_hash)
    transformed = preprocessor.transform(development_arrays)
    seed = int(training_config.get("seed", 2026))
    suite_records: dict[str, Any] = {}
    for model_name in selected:
        record = frozen[model_name]
        model_dir = output / model_name
        model_dir.mkdir()
        preprocessing_path = model_dir / "preprocessor.json"
        preprocessor.save(preprocessing_path)
        model_seed = seed + 10_000 + 100 * PRESPECIFIED_MODELS.index(model_name)
        seed_everything(model_seed)
        started = time.perf_counter()
        model_config = LACConfig(**record["model_config"])
        if model_name == "persistence":
            model = None
            fit_history: dict[str, Any] = {"not_required": True}
            artifact_path = model_dir / "model_meta.json"
            artifact_path.write_text("{}", encoding="utf-8")
        elif model_name in CLASSICAL_MODELS:
            options = training_config.get("classical", {}).get(model_name, {})
            model = build_classical_model(model_name, seed=model_seed, **options).fit(transformed)
            fit_history = {"fit_on_all_center_B_patients": True}
            artifact_path = model_dir / "model.pkl"
            with artifact_path.open("wb") as handle:
                pickle.dump(model, handle)
        else:
            model = build_torch_model(model_name, model_config).to(device)
            model, fit_history = fit_model_fixed_epochs(
                model,
                transformed,
                training_config,
                device,
                int(record["locked_final_training_epochs"]),
            )
            artifact_path = model_dir / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_config": record["model_config"],
                    "schema": development_schema.to_dict(),
                },
                artifact_path,
            )
        fit_seconds = time.perf_counter() - started
        final_manifest = {
            **record,
            "status": "locked_full_center_B_refit",
            "external_validation_locked": True,
            "development_center": "center_B",
            "development_patient_count": len(development_arrays["patient_ids"]),
            "development_patient_hash": development_hash,
            "preprocessor_fitted_on_all_center_B": True,
            "model_refitted_on_all_center_B": model_name != "persistence",
            "external_data_seen_during_fit_or_selection": False,
            "model_seed": model_seed,
            "fit_seconds": fit_seconds,
            "fit_history": fit_history,
            "training_config_sha256": _canonical_hash(training_config),
            "finalization_git_revision": _git_revision(),
            "runtime_versions": _runtime_versions(),
            "artifact_file": artifact_path.name,
            "artifact_sha256": file_sha256(artifact_path),
            "preprocessor_sha256": file_sha256(preprocessing_path),
        }
        (model_dir / "finalization_manifest.json").write_text(
            json.dumps(final_manifest, indent=2), encoding="utf-8"
        )
        suite_records[model_name] = final_manifest
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output / "schema.json").write_text(
        json.dumps(development_schema.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "locked_training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )
    result_scope = str(internal_source["internal_result_scope"])
    if result_scope.startswith("synthetic_smoke"):
        status = "synthetic_smoke_locked_bundle"
    elif result_scope.startswith("synthetic"):
        status = "synthetic_diagnostic_locked_bundle"
    else:
        status = "development_center_B_locked_bundle"
    result = {
        "status": status,
        "protocol": "center_B_fivefold_freeze_then_full_center_B_refit",
        "models": list(selected),
        "model_count": len(selected),
        "device": str(device),
        "hyperparameters_frozen_from_internal_validation": True,
        "preprocessor_refitted_on_all_center_B": True,
        "trainable_model_weights_refitted_on_all_center_B": True,
        "external_center_A_used": False,
        "training_config_sha256": _canonical_hash(training_config),
        "schema_sha256": _canonical_hash(development_schema.to_dict()),
        "internal_source": internal_source,
        "model_records": suite_records,
    }
    (output / "locked_bundle_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _validate_locked_bundle(bundle: Path) -> tuple[dict[str, Any], FeatureSchema]:
    manifest = _read_json(bundle / "locked_bundle_manifest.json")
    if manifest.get("external_center_A_used") is not False:
        raise ExternalValidationGuardrailError("Bundle was not frozen before center-A evaluation")
    if not manifest.get("hyperparameters_frozen_from_internal_validation"):
        raise ExternalValidationGuardrailError("Bundle lacks locked internal-selection provenance")
    locked_training_config = _read_json(bundle / "locked_training_config.json")
    if manifest["training_config_sha256"] != _canonical_hash(locked_training_config):
        raise ExternalValidationGuardrailError("Locked training configuration checksum mismatch")
    schema = FeatureSchema.from_dict(_read_json(bundle / "schema.json"))
    if manifest["schema_sha256"] != _canonical_hash(schema.to_dict()):
        raise ExternalValidationGuardrailError("Locked schema checksum mismatch")
    return manifest, schema


def evaluate_external_arrays_all_models(
    external_arrays: dict[str, np.ndarray],
    external_schema: FeatureSchema,
    bundle_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    n_bootstrap: int = 2000,
    center_name: str = "center_A",
    source_type: str = "standardized_npz",
    source_sha256: str | None = None,
    source_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply locked models once to center A; this function has no fitting path."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    bundle = Path(bundle_dir)
    bundle_manifest, locked_schema = _validate_locked_bundle(bundle)
    if external_schema != locked_schema:
        raise ExternalValidationGuardrailError("Center-A schema differs from the locked center-B schema")
    _validate_arrays(external_arrays, external_schema, "center_A_external")
    output = _prepare_output(output_dir)
    device = resolve_device(device_name)
    all_rows = []
    comparison_rows = []
    summaries: dict[str, Any] = {}
    targets = np.asarray(external_arrays["targets"])
    external_ids = [str(value) for value in external_arrays["patient_ids"]]
    for model_index, model_name in enumerate(bundle_manifest["models"]):
        model_dir = bundle / model_name
        lock = _read_json(model_dir / "finalization_manifest.json")
        root_lock = bundle_manifest["model_records"].get(model_name)
        if root_lock is None or _canonical_hash(root_lock) != _canonical_hash(lock):
            raise ExternalValidationGuardrailError(
                f"{model_name} root and per-model lock manifests differ"
            )
        if not lock.get("external_validation_locked"):
            raise ExternalValidationGuardrailError(f"{model_name} is not externally locked")
        preprocessing_path = model_dir / "preprocessor.json"
        artifact_path = model_dir / lock["artifact_file"]
        expected_artifact = (
            "model_meta.json" if model_name == "persistence" else
            "model.pkl" if model_name in CLASSICAL_MODELS else "model.pt"
        )
        if lock["artifact_file"] != expected_artifact:
            raise ExternalValidationGuardrailError(f"{model_name} artifact filename is invalid")
        if file_sha256(preprocessing_path) != lock["preprocessor_sha256"]:
            raise ExternalValidationGuardrailError(f"{model_name} preprocessor checksum mismatch")
        if file_sha256(artifact_path) != lock["artifact_sha256"]:
            raise ExternalValidationGuardrailError(f"{model_name} model artifact checksum mismatch")
        preprocessor = FoldPreprocessor.load(preprocessing_path)
        if preprocessor.fitted_patient_ids_hash != lock["development_patient_hash"]:
            raise ExternalValidationGuardrailError(f"{model_name} preprocessor provenance mismatch")
        transformed = preprocessor.transform(external_arrays)
        started = time.perf_counter()
        if model_name == "persistence":
            prediction = transformed["baseline"].copy()
            ids = external_ids
        elif model_name in CLASSICAL_MODELS:
            with artifact_path.open("rb") as handle:
                model = pickle.load(handle)
            prediction = model.predict(transformed)
            ids = external_ids
        else:
            checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
            if checkpoint.get("model_name") != model_name:
                raise ExternalValidationGuardrailError(f"{model_name} checkpoint identity mismatch")
            if checkpoint.get("schema") != locked_schema.to_dict():
                raise ExternalValidationGuardrailError(f"{model_name} checkpoint schema mismatch")
            model = build_torch_model(
                model_name, LACConfig(**checkpoint["model_config"])
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            prediction, predicted_targets, ids = predict(
                model,
                _loader(transformed, 64, False),
                device,
            )
            if not np.allclose(predicted_targets, targets):
                raise RuntimeError(f"{model_name} prediction target order mismatch")
        inference_seconds = time.perf_counter() - started
        if ids != external_ids:
            raise RuntimeError(f"{model_name} external patient order mismatch")
        if prediction.shape != targets.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"{model_name} produced invalid external predictions")
        metrics = regression_metrics(targets, prediction)
        confidence = bootstrap_metrics(
            targets,
            prediction,
            n_bootstrap=n_bootstrap,
            seed=2026 + model_index,
        )
        model_output = output / model_name
        model_output.mkdir()
        pd.DataFrame({
            "patient_id": ids,
            "true_tbr": targets[:, 0],
            "pred_tbr": prediction[:, 0],
            "true_cac": targets[:, 1],
            "pred_cac": prediction[:, 1],
        }).to_csv(model_output / "external_predictions.csv", index=False)
        internal_metrics = lock["internal_oof_metrics"]
        gap = {
            key: float(metrics[key] - internal_metrics[key])
            for key in metrics if key in internal_metrics
        }
        summary = {
            "model_name": model_name,
            "status": "external_validation_complete",
            "external_center": center_name,
            "n_external_patients": len(ids),
            "metrics": metrics,
            "bootstrap_95_ci": confidence,
            "internal_oof_metrics": internal_metrics,
            "external_minus_internal_metric_gap": gap,
            "inference_seconds": inference_seconds,
            "fit_or_selection_on_external_data": False,
            "model_weights_refitted_on_all_center_B": model_name != "persistence",
            "prediction_rule_locked_after_center_B_finalization": True,
            "locked_model_artifact_sha256": lock["artifact_sha256"],
        }
        (model_output / "external_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summaries[model_name] = summary
        all_rows.append({"model": model_name} | metrics)
        comparison_rows.append(
            {"model": model_name}
            | {f"internal_{key}": value for key, value in internal_metrics.items()}
            | {f"external_{key}": value for key, value in metrics.items()}
            | {f"gap_{key}": value for key, value in gap.items()}
        )
        if model_name != "persistence":
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(all_rows).to_csv(output / "external_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(
        output / "internal_external_comparison.csv", index=False
    )
    bundle_status = str(bundle_manifest["status"])
    formal = not bundle_status.startswith("synthetic")
    if formal:
        status = "external_validation_complete"
    elif bundle_status.startswith("synthetic_smoke"):
        status = "synthetic_smoke_only"
    else:
        status = "synthetic_diagnostic_only"
    result = {
        "status": status,
        "protocol": "single_center_A_evaluation_after_full_center_B_refit",
        "external_center": center_name,
        "external_source_type": source_type,
        "external_source_sha256": source_sha256,
        "external_source_audit": source_audit,
        "external_patient_count": len(external_ids),
        "external_patient_hash": patient_id_hash(np.asarray(external_ids)),
        "models": list(bundle_manifest["models"]),
        "model_count": len(bundle_manifest["models"]),
        "device": str(device),
        "evaluation_git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "locked_bundle_manifest_sha256": file_sha256(
            bundle / "locked_bundle_manifest.json"
        ),
        "bootstrap_replicates": n_bootstrap,
        "external_data_used_for_fit_preprocessing_selection_or_epoch_choice": False,
        "one_locked_model_per_algorithm": True,
        "formal_conclusions_generated": False,
        "summaries": summaries,
    }
    (output / "external_validation_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def evaluate_external_workbook_all_models(
    workbook_path: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    n_bootstrap: int = 2000,
    center_name: str = "center_A",
) -> dict[str, Any]:
    bundle_manifest, schema = _validate_locked_bundle(Path(bundle_dir))
    del bundle_manifest
    reader = ExternalWorkbookReader(purpose="external_validation")
    audit = reader.audit(workbook_path)
    arrays = reader.prepare_locked_arrays(workbook_path, schema)
    return evaluate_external_arrays_all_models(
        arrays,
        schema,
        bundle_dir,
        output_dir,
        device_name=device_name,
        n_bootstrap=n_bootstrap,
        center_name=center_name,
        source_type="legacy_excel_workbook",
        source_sha256=file_sha256(workbook_path),
        source_audit=audit.to_dict(),
    )
