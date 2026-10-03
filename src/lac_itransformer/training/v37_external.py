from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.preprocessing import FoldPreprocessor, subset_arrays
from ..data.schema import FeatureSchema
from ..data.shandong_external import ShandongExternalJSONReader
from ..data.shandong_workbook import ShandongExternalWorkbookReader
from ..experiments.prediction import build_classical_model, build_torch_model
from .metrics import (
    bootstrap_change_metrics,
    change_space_metrics,
    regression_metrics,
)
from .trainer import (
    _git_revision,
    _loader,
    _model_config,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    predict,
    resolve_device,
    seed_everything,
)
from .v27_nested import _predict_base_with_features, _residual_to_endpoint
from .v32_nested import _componentwise_epoch_median
from .v36_selector import ParetoSafeClinicalSelector
from .v37_nested import V37_MODELS, _config_for_model, _decision_modes
from .v37_trainer import fit_v37_fixed_epochs


COMPARISON_MODELS = (
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
    "lac_v37_full",
)
CLASSICAL_MODELS = {"elastic_net", "xgboost"}
DEEP_COMPARATORS = {
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
}
TABLE_A_MODELS = (
    "lac_v37_full",
    "lac_v37_dual_independent",
    "lac_v37_no_soft_adapters",
    "lac_v37_no_gradient_protection",
    "lac_v37_no_baseline_anchoring",
)
TABLE_B_MODELS = (
    "lac_v37_full",
    "lac_v37_no_shared_transfer",
    "lac_v37_no_historical_dose",
    "lac_v37_no_i_to_c",
    "lac_v37_no_treatment",
    "lac_v37_direct_cac_regression",
    "lac_v37_no_calibration",
)
PRIMARY_METRICS = (
    "delta_tbr_mae",
    "delta_tbr_rmse",
    "delta_tbr_r2",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
    "delta_log_cac_r2",
)
LOCKED_DEVELOPMENT_PATIENT_HASH = (
    "eabc11cf936d01147f23ff7775ef8e22a549f9cb8cb7c41b421a6b173628b46c"
)
LOCKED_DEVELOPMENT_FOLLOWUP_MEDIAN_MONTHS = 11.0


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _validate_arrays(arrays: dict[str, np.ndarray], schema: FeatureSchema, role: str) -> None:
    required = {
        "static", "baseline", "values", "mask", "delta", "times",
        "treatments", "irregular_values", "irregular_mask",
        "irregular_times", "irregular_treatments", "targets",
        "followup_months", "patient_ids",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"{role} arrays missing: {sorted(missing)}")
    n = len(arrays["patient_ids"])
    if n < 1 or len(set(str(value) for value in arrays["patient_ids"])) != n:
        raise ValueError(f"{role} patient IDs must be nonempty and unique")
    shapes = {
        "static": (n, len(schema.static_features)),
        "baseline": (n, 2),
        "values": (n, schema.time_patches, len(schema.longitudinal_features)),
        "mask": (n, schema.time_patches, len(schema.longitudinal_features)),
        "delta": (n, schema.time_patches, len(schema.longitudinal_features)),
        "times": (n, schema.time_patches),
        "treatments": (n, schema.time_patches, len(schema.treatment_features)),
        "irregular_values": (n, schema.max_events, len(schema.longitudinal_features)),
        "irregular_mask": (n, schema.max_events, len(schema.longitudinal_features)),
        "irregular_times": (n, schema.max_events),
        "irregular_treatments": (n, schema.max_events, len(schema.treatment_features)),
        "targets": (n, 2),
        "followup_months": (n,),
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{role} {key} shape {arrays[key].shape} != {shape}")
    if not np.isfinite(arrays["baseline"]).all() or not np.isfinite(arrays["targets"]).all():
        raise ValueError(f"{role} baseline and primary endpoints must be complete")
    if np.any(arrays["baseline"][:, 1] < 0) or np.any(arrays["targets"][:, 1] < 0):
        raise ValueError(f"{role} CAC values must be nonnegative")


def prepare_shandong_external(
    input_json: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = _prepare_output(output_dir)
    schema = FeatureSchema.from_dict(_read_json(schema_path))
    arrays, audit = ShandongExternalJSONReader(schema).prepare_arrays(input_json)
    _validate_arrays(arrays, schema, "external_shandong")
    save_npz(output / "prepared_arrays.npz", arrays)
    (output / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2), encoding="utf-8"
    )
    result = {
        "status": "external_shandong_prepared_for_locked_inference_only",
        "source_sha256": _sha256(input_json),
        "source_schema_version": "shandong_external_v37_deidentified_v1",
        "external_patient_count": len(arrays["patient_ids"]),
        "patient_id_hash": patient_id_hash(arrays["patient_ids"]),
        "schema_sha256": _canonical_hash(schema.to_dict()),
        "fit_or_selection_performed": False,
        "audit": audit.to_dict(),
    }
    (output / "external_data_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def prepare_shandong_workbook_external(
    input_workbook: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare the revised 755-patient XLSX for frozen external inference.

    Source quality-control fills are used only to exclude the 97 yellow and
    27 green rows.  They never enter the model arrays as features.  The reader
    enforces the locked 879-source/124-exclusion/755-retained cohort contract.
    """
    output = _prepare_output(output_dir)
    schema = FeatureSchema.from_dict(_read_json(schema_path))
    arrays, audit = ShandongExternalWorkbookReader(schema).prepare_arrays(
        input_workbook
    )
    _validate_arrays(arrays, schema, "external_shandong_755_supplement")
    prepared_path = output / "prepared_arrays.npz"
    save_npz(prepared_path, arrays)
    (output / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2), encoding="utf-8"
    )
    result = {
        "status": "external_shandong_755_supplement_prepared_for_locked_inference_only",
        "source_format": "revised_shandong_xlsx_with_summary_row_qc_fills",
        "source_sha256": _sha256(input_workbook),
        "source_schema_version": "shandong_external_revised_workbook_v1",
        "external_patient_count": len(arrays["patient_ids"]),
        "patient_id_hash": patient_id_hash(arrays["patient_ids"]),
        "schema_sha256": _canonical_hash(schema.to_dict()),
        "prepared_arrays_sha256": _sha256(prepared_path),
        "cohort_role": "supplemental_external_validation_only",
        "final_model_locked_before_inference": "lac_v40_final_no_tail",
        "fit_or_selection_performed": False,
        "audit": audit.to_dict(),
    }
    (output / "external_data_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _fold_epoch_values(summary: dict[str, Any]) -> list[int]:
    values = [
        int(record["selected_epochs"])
        for record in summary.get("fold_records", [])
        if record.get("selected_epochs") is not None
    ]
    if len(values) != 5:
        raise ValueError("Final external refit requires five locked outer-fold epoch choices")
    return values


def _median_epoch(values: list[int]) -> int:
    return max(1, int(np.rint(np.median(values))))


def _modal_classical_parameters(summary: dict[str, Any]) -> dict[str, Any]:
    values = []
    for record in summary.get("fold_records", []):
        value = record.get("selected_classical_parameters")
        if value is None:
            continue
        parsed = json.loads(value) if isinstance(value, str) else dict(value)
        values.append(
            json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        )
    if len(values) != 5:
        raise ValueError(
            "Final external classical refit requires five locked outer-fold "
            "hyperparameter choices"
        )
    counts = Counter(values)
    best_count = max(counts.values())
    modes = [value for value, count in counts.items() if count == best_count]
    if len(modes) != 1:
        raise ValueError(
            "Locked outer-fold classical hyperparameters do not have a unique mode"
        )
    return json.loads(modes[0])


def _source_for_v37(v37_root: Path, model_name: str) -> Path:
    candidates = (
        v37_root / "full" / model_name,
        v37_root / "ablations" / model_name,
    )
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one frozen V3.7 source for {model_name}; "
            f"found {[str(path) for path in existing]}"
        )
    return existing[0]


def _v37_final_refit_seed(config: dict[str, Any]) -> int:
    return int(config.get("seed", 2026)) + 800_000


def _combine_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "patient_ids": sum([bundle["patient_ids"] for bundle in bundles], []),
        "baseline": np.concatenate([bundle["baseline"] for bundle in bundles]),
        "endpoint_targets": np.concatenate([bundle["endpoint_targets"] for bundle in bundles]),
        "residual_targets": np.concatenate([bundle["residual_targets"] for bundle in bundles]),
        "heads": {
            task: np.concatenate([bundle["heads"][task] for bundle in bundles])
            for task in ("tbr", "cac")
        },
        "features": {
            task: np.concatenate([bundle["features"][task] for bundle in bundles])
            for task in ("tbr", "cac")
        },
        "meta_fold": np.concatenate([bundle["meta_fold"] for bundle in bundles]),
    }


def _fit_locked_v37_decision(
    development_arrays: dict[str, np.ndarray],
    source_dir: Path,
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ParetoSafeClinicalSelector, dict[str, Any]]:
    plan_frame = pd.read_csv(source_dir / "patient_outer_fold_plan.csv")
    plan = {
        str(row.patient_id): int(row.test_fold)
        for row in plan_frame.itertuples(index=False)
    }
    raw_index = {
        str(patient_id): index
        for index, patient_id in enumerate(development_arrays["patient_ids"])
    }
    bundles: list[dict[str, Any]] = []
    for fold in range(1, 6):
        ids = [patient_id for patient_id, assigned in plan.items() if assigned == fold]
        indices = np.asarray([raw_index[patient_id] for patient_id in ids], dtype=int)
        raw = subset_arrays(development_arrays, indices)
        fold_dir = source_dir / f"fold_{fold}"
        checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
        model = build_torch_model(
            model_name,
            _config_for_model(
                FeatureSchema.from_dict(checkpoint["schema"]),
                config,
                model_name,
            ),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        preprocessor = FoldPreprocessor.load(fold_dir / "preprocessor.json")
        bundle = _predict_base_with_features(
            model,
            preprocessor.transform(raw),
            raw,
            int(config.get("batch_size", 64)),
            device,
            model_name,
        )
        bundle["meta_fold"] = np.full(len(ids), fold, dtype=int)
        bundles.append(bundle)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = _combine_bundles(bundles)
    if len(combined["patient_ids"]) != len(development_arrays["patient_ids"]):
        raise RuntimeError("V3.7 OOF decision reconstruction is incomplete")
    options = dict(config.get("decision", {}))
    selector = ParetoSafeClinicalSelector(
        candidate_weights=tuple(options.get("candidate_weights", (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0))),
        ridge_alphas=tuple(options.get("ridge_alphas", (1.0, 10.0))),
        degradation_tolerance=float(options.get("degradation_tolerance", 0.005)),
        required_consistent_folds=int(options.get("required_consistent_folds", 3)),
    ).fit(
        heads=combined["heads"],
        targets={
            "tbr": combined["residual_targets"][:, 0],
            "cac": combined["residual_targets"][:, 1],
        },
        features=combined["features"],
        meta_fold=combined["meta_fold"],
    )
    return selector, selector.audit()


def finalize_v37_external_bundle(
    development_npz: str | Path,
    schema_path: str | Path,
    formal_root: str | Path,
    v37_root: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Refit frozen models on all 443 development patients; never reads external data."""
    output = _prepare_output(output_dir)
    schema = FeatureSchema.from_dict(_read_json(schema_path))
    development = load_npz(development_npz)
    _validate_arrays(development, schema, "development_center_B")
    if len(development["patient_ids"]) != 443:
        raise ValueError("Frozen V3.7 external bundle requires the original 443-patient cohort")
    formal_root = Path(formal_root)
    v37_root = Path(v37_root)
    device = resolve_device(device_name)
    preprocessor = FoldPreprocessor.fit(
        development, patient_id_hash(development["patient_ids"])
    )
    transformed = preprocessor.transform(development)
    preprocessor.save(output / "preprocessor.json")
    (output / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2), encoding="utf-8"
    )

    records: dict[str, Any] = {}
    comparator_sources = COMPARISON_MODELS[:-1]
    for model_index, model_name in enumerate(comparator_sources):
        source_dir = formal_root / model_name
        summary = _read_json(source_dir / "summary.json")
        config = _read_json(source_dir / "training_config.json")
        if int(summary["patient_count"]) != 443:
            raise ValueError(f"{model_name} internal cohort differs from 443 patients")
        model_dir = output / model_name
        model_dir.mkdir()
        seed = int(config.get("seed", 2026)) + 700_000 + model_index * 1000
        seed_everything(seed)
        started = time.perf_counter()
        if model_name == "persistence":
            artifact = model_dir / "model_meta.json"
            artifact.write_text("{}", encoding="utf-8")
            locked_epochs = None
            history = {"not_required": True}
        elif model_name in CLASSICAL_MODELS:
            options = _modal_classical_parameters(summary)
            model = build_classical_model(model_name, seed=seed, **options).fit(transformed)
            artifact = model_dir / "model.pkl"
            with artifact.open("wb") as handle:
                pickle.dump(model, handle)
            locked_epochs = None
            history = {
                "fit_on_all_443_development_patients": True,
                "locked_hyperparameters": options,
                "locking_rule": "unique_mode_of_five_outer_fold_choices",
            }
            del model
        elif model_name in DEEP_COMPARATORS:
            epoch_values = _fold_epoch_values(summary)
            locked_epochs = _median_epoch(epoch_values)
            model_config = _model_config(
                schema,
                config.get("model", {}),
                model_name,
                config.get("model_v2", {}),
                config.get("model_v21", {}),
                config.get("model_v22", {}),
            )
            model = build_torch_model(model_name, model_config).to(device)
            model, history = fit_model_fixed_epochs(
                model, transformed, config, device, locked_epochs, seed
            )
            artifact = model_dir / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_config": model_config.to_dict(),
                    "schema": schema.to_dict(),
                },
                artifact,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            raise KeyError(model_name)
        record = {
            "model_name": model_name,
            "source_internal_summary": str(source_dir / "summary.json"),
            "source_internal_summary_sha256": _sha256(source_dir / "summary.json"),
            "source_internal_git_revision": summary["git_revision"],
            "source_outer_fold_checksum": summary["cv_protocol"]["outer_fold_plan_checksum"],
            "source_internal_oof_change_metrics": summary.get("pooled_change_metrics"),
            "locked_final_training_epochs": locked_epochs,
            "artifact_file": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "fit_seconds": time.perf_counter() - started,
            "seed": seed,
            "fit_history": history,
            "external_data_seen": False,
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records[model_name] = record

    for model_name in V37_MODELS:
        source_dir = _source_for_v37(v37_root, model_name)
        summary = _read_json(source_dir / "summary.json")
        config = _read_json(source_dir / "training_config.json")
        if summary["git_revision"] != "7d362511e0b2ef4558ff9edc73ac38b0f9f5dfee":
            raise ValueError(f"{model_name} is not from the frozen V3.7 model commit")
        if summary["cv_protocol"]["outer_fold_plan_checksum"] != "7c18f9bd52a884639839a5a2f8e8664f1db8f016bcf46ab632ebf1db9db8c655":
            raise ValueError(f"{model_name} does not use the locked patient split")
        model_dir = output / model_name
        if model_dir.exists():
            if any(model_dir.iterdir()):
                raise FileExistsError(model_dir)
        else:
            model_dir.mkdir()
        seed = _v37_final_refit_seed(config)
        seed_everything(seed)
        started = time.perf_counter()
        selector, decision_audit = _fit_locked_v37_decision(
            development, source_dir, model_name, config, device
        )
        with (model_dir / "decision_layer.pkl").open("wb") as handle:
            pickle.dump(selector, handle)
        epoch_values = _fold_epoch_values(summary)
        locked_epochs = _componentwise_epoch_median(epoch_values)
        model_config = _config_for_model(schema, config, model_name)
        model = build_torch_model(model_name, model_config).to(device)
        model, history = fit_v37_fixed_epochs(
            model, transformed, config, device, locked_epochs, seed
        )
        artifact = model_dir / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "model_config": model_config.to_dict(),
                "schema": schema.to_dict(),
            },
            artifact,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        record = {
            "model_name": model_name,
            "source_internal_summary": str(source_dir / "summary.json"),
            "source_internal_summary_sha256": _sha256(source_dir / "summary.json"),
            "source_internal_git_revision": summary["git_revision"],
            "source_outer_fold_checksum": summary["cv_protocol"]["outer_fold_plan_checksum"],
            "source_internal_oof_change_metrics": summary.get("pooled_change_metrics"),
            "locked_stage_epoch_codes": epoch_values,
            "locked_final_stage_epoch_code": locked_epochs,
            "decision_modes": _decision_modes(model_name),
            "decision_audit": decision_audit,
            "artifact_file": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "decision_sha256": _sha256(model_dir / "decision_layer.pkl"),
            "fit_seconds": time.perf_counter() - started,
            "seed": seed,
            "fit_history": history,
            "external_data_seen": False,
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records[model_name] = record

    result = {
        "status": "locked_v37_full_center_B_refit_bundle",
        "protocol": "locked_internal_choices_then_full_443_refit_then_one_time_external_evaluation",
        "development_patient_count": 443,
        "development_patient_hash": patient_id_hash(development["patient_ids"]),
        "comparison_models": list(COMPARISON_MODELS),
        "v37_ablation_models": list(V37_MODELS),
        "preprocessor_fitted_on_all_development_patients": True,
        "external_data_used_for_fit_selection_or_epoch_choice": False,
        "schema_sha256": _canonical_hash(schema.to_dict()),
        "preprocessor_sha256": _sha256(output / "preprocessor.json"),
        "bundle_git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
        "model_records": records,
    }
    (output / "locked_bundle_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def correct_v37_ablation_seed_bundle(
    source_bundle_dir: str | Path,
    development_npz: str | Path,
    schema_path: str | Path,
    v37_root: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Copy a bundle and refit only V3.7 ablations with the full-model seed."""
    source_bundle = Path(source_bundle_dir)
    source_manifest_path = source_bundle / "locked_bundle_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    if source_manifest["external_data_used_for_fit_selection_or_epoch_choice"]:
        raise RuntimeError("Source bundle reports external leakage")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")

    schema = FeatureSchema.from_dict(_read_json(schema_path))
    development = load_npz(development_npz)
    _validate_arrays(development, schema, "development_center_B")
    if len(development["patient_ids"]) != 443:
        raise ValueError("Seed correction requires the original 443-patient cohort")
    if patient_id_hash(development["patient_ids"]) != source_manifest[
        "development_patient_hash"
    ]:
        raise ValueError("Development cohort differs from source bundle")
    if _canonical_hash(schema.to_dict()) != source_manifest["schema_sha256"]:
        raise ValueError("Development schema differs from source bundle")

    shutil.copytree(source_bundle, output)
    preprocessor_path = output / "preprocessor.json"
    if _sha256(preprocessor_path) != source_manifest["preprocessor_sha256"]:
        raise RuntimeError("Copied preprocessor checksum mismatch")
    preprocessor = FoldPreprocessor.load(preprocessor_path)
    transformed = preprocessor.transform(development)
    device = resolve_device(device_name)
    v37_root = Path(v37_root)
    records = dict(source_manifest["model_records"])
    full_seed = int(records["lac_v37_full"]["seed"])
    full_checkpoint = torch.load(
        output / "lac_v37_full" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    corrected_models = []

    for model_name in V37_MODELS:
        if model_name == "lac_v37_full":
            continue
        source_dir = _source_for_v37(v37_root, model_name)
        summary = _read_json(source_dir / "summary.json")
        config = _read_json(source_dir / "training_config.json")
        seed = _v37_final_refit_seed(config)
        if seed != full_seed:
            raise ValueError(f"{model_name} does not share the full-model seed")
        epoch_values = _fold_epoch_values(summary)
        locked_epochs = _componentwise_epoch_median(epoch_values)
        seed_everything(seed)
        started = time.perf_counter()
        model_config = _config_for_model(schema, config, model_name)
        model_dir = output / model_name
        artifact = model_dir / "model.pt"
        full_state_comparison = None
        if model_name == "lac_v37_no_calibration":
            checkpoint = dict(full_checkpoint)
            checkpoint["model_name"] = model_name
            checkpoint["model_config"] = model_config.to_dict()
            history = records["lac_v37_full"]["fit_history"]
            full_state_comparison = {
                "exact_tensor_equality": True,
                "maximum_absolute_tensor_difference": 0.0,
                "interpretation": (
                    "no-calibration changes only the post-model decision mode, so "
                    "the full-model neural state is reused exactly"
                ),
            }
        else:
            # The original finalizer reconstructed the locked OOF decision layer
            # after seeding and before model initialization. Replay that read-only
            # path so the torch RNG reaches the same controlled point for every
            # ablation; keep the already locked decision artifact unchanged.
            replayed_decision, _ = _fit_locked_v37_decision(
                development, source_dir, model_name, config, device
            )
            del replayed_decision
            model = build_torch_model(model_name, model_config).to(device)
            model, history = fit_v37_fixed_epochs(
                model, transformed, config, device, locked_epochs, seed
            )
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "model_config": model_config.to_dict(),
                "schema": schema.to_dict(),
            }
        torch.save(checkpoint, artifact)
        if model_name != "lac_v37_no_calibration":
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        original = records[model_name]
        record = dict(original)
        record.update({
            "locked_stage_epoch_codes": epoch_values,
            "locked_final_stage_epoch_code": locked_epochs,
            "artifact_sha256": _sha256(artifact),
            "fit_seconds": time.perf_counter() - started,
            "seed": seed,
            "fit_history": history,
            "external_data_seen": False,
            "seed_control_correction": {
                "previous_seed": int(original["seed"]),
                "corrected_seed": seed,
                "rule": "same_refit_seed_as_lac_v37_full",
                "neural_refit_performed": (
                    model_name != "lac_v37_no_calibration"
                ),
                "source_bundle_manifest_sha256": _sha256(source_manifest_path),
                "full_state_comparison": full_state_comparison,
            },
        })
        (model_dir / "manifest.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records[model_name] = record
        corrected_models.append(model_name)

    result = dict(source_manifest)
    result.update({
        "status": "locked_v37_full_center_B_refit_bundle_seed_control_corrected",
        "bundle_git_revision": _git_revision(),
        "source_bundle_manifest_sha256": _sha256(source_manifest_path),
        "v37_common_refit_seed": full_seed,
        "v37_ablation_seed_control": (
            "all ten ablations use the exact full-model refit seed; "
            "comparators and lac_v37_full were copied without refit"
        ),
        "seed_corrected_models": corrected_models,
        "model_records": records,
    })
    (output / "locked_bundle_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _load_deep_model(model_dir: Path, device: torch.device):
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    schema = FeatureSchema.from_dict(checkpoint["schema"])
    model_name = checkpoint["model_name"]
    if model_name.startswith("lac_v37_"):
        config = _config_for_model(
            schema,
            {"model_v37": checkpoint["model_config"]},
            model_name,
        )
    else:
        config = _model_config(schema, checkpoint["model_config"], model_name)
    model = build_torch_model(model_name, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, schema


def _prediction_frame(
    patient_ids: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
    prediction: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "patient_id": [str(value) for value in patient_ids],
        "baseline_tbr": baseline[:, 0],
        "baseline_cac": baseline[:, 1],
        "true_tbr": targets[:, 0],
        "pred_tbr": prediction[:, 0],
        "true_cac": targets[:, 1],
        "pred_cac": prediction[:, 1],
    })
    frame["true_delta_tbr"] = frame["true_tbr"] - frame["baseline_tbr"]
    frame["pred_delta_tbr"] = frame["pred_tbr"] - frame["baseline_tbr"]
    frame["true_delta_log_cac"] = np.log1p(np.maximum(frame["true_cac"], 0)) - np.log1p(np.maximum(frame["baseline_cac"], 0))
    frame["pred_delta_log_cac"] = np.log1p(np.maximum(frame["pred_cac"], 0)) - np.log1p(np.maximum(frame["baseline_cac"], 0))
    for name, values in (diagnostics or {}).items():
        frame[name] = values
    return frame


def _load_reused_prediction_frame(
    path: str | Path,
    patient_ids: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    expected_ids = [str(value) for value in patient_ids]
    if frame.get("patient_id") is None:
        raise ValueError(f"Reused predictions have no patient_id column: {path}")
    if frame["patient_id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"Reused predictions changed patient order: {path}")
    expected = {
        "baseline_tbr": baseline[:, 0],
        "baseline_cac": baseline[:, 1],
        "true_tbr": targets[:, 0],
        "true_cac": targets[:, 1],
    }
    for column, values in expected.items():
        if column not in frame or not np.allclose(
            frame[column].to_numpy(float), values, rtol=1e-6, atol=1e-6
        ):
            raise ValueError(f"Reused predictions differ in {column}: {path}")
    required_predictions = ("pred_tbr", "pred_cac")
    if any(column not in frame for column in required_predictions):
        raise ValueError(f"Reused predictions are incomplete: {path}")
    prediction = frame.loc[:, list(required_predictions)].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Reused predictions contain nonfinite values: {path}")
    return frame, prediction


def _paired_bootstrap(
    targets: np.ndarray,
    baseline: np.ndarray,
    prediction_full: np.ndarray,
    prediction_other: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    point_full = change_space_metrics(targets, prediction_full, baseline)
    point_other = change_space_metrics(targets, prediction_other, baseline)
    samples = {key: [] for key in PRIMARY_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(targets), len(targets))
        left = change_space_metrics(targets[index], prediction_full[index], baseline[index])
        right = change_space_metrics(targets[index], prediction_other[index], baseline[index])
        for key in PRIMARY_METRICS:
            value = left[key] - right[key]
            if np.isfinite(value):
                samples[key].append(value)
    return [
        {
            "metric": key,
            "difference_full_minus_other": point_full[key] - point_other[key],
            "ci_low": float(np.quantile(samples[key], 0.025)),
            "ci_high": float(np.quantile(samples[key], 0.975)),
            "bootstrap_replicates": n_bootstrap,
        }
        for key in PRIMARY_METRICS
    ]


def evaluate_v37_external_bundle(
    bundle_dir: str | Path,
    external_npz: str | Path,
    external_schema_path: str | Path,
    external_manifest_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    n_bootstrap: int = 2000,
    reuse_predictions_from: str | Path | None = None,
) -> dict[str, Any]:
    """One-time external prediction path. Contains no fit, calibration, or selection."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    output = _prepare_output(output_dir)
    bundle = Path(bundle_dir)
    bundle_manifest = _read_json(bundle / "locked_bundle_manifest.json")
    if bundle_manifest["external_data_used_for_fit_selection_or_epoch_choice"]:
        raise RuntimeError("Locked bundle reports external leakage")
    schema = FeatureSchema.from_dict(_read_json(external_schema_path))
    if _canonical_hash(schema.to_dict()) != bundle_manifest["schema_sha256"]:
        raise ValueError("External schema differs from the locked development schema")
    external = load_npz(external_npz)
    _validate_arrays(external, schema, "external_shandong")
    source_manifest = _read_json(external_manifest_path)
    if len(external["patient_ids"]) != int(source_manifest["external_patient_count"]):
        raise ValueError("External patient count differs from the audited manifest")
    missing_followup = ~np.isfinite(external["followup_months"].astype(float))
    if np.any(missing_followup):
        if (
            bundle_manifest["development_patient_hash"]
            != LOCKED_DEVELOPMENT_PATIENT_HASH
        ):
            raise RuntimeError(
                "The locked development follow-up imputation rule does not apply "
                "to this bundle"
            )
        external = dict(external)
        external["followup_months"] = external["followup_months"].astype(
            float, copy=True
        )
        external["followup_months"][missing_followup] = (
            LOCKED_DEVELOPMENT_FOLLOWUP_MEDIAN_MONTHS
        )
    followup_imputation = {
        "field": "followup_months",
        "missing_external_values": int(missing_followup.sum()),
        "fill_value_months": LOCKED_DEVELOPMENT_FOLLOWUP_MEDIAN_MONTHS,
        "source": "median_of_exact_locked_443_patient_development_cohort",
        "external_values_used_to_fit_imputation": False,
    }
    preprocessor_path = bundle / "preprocessor.json"
    if _sha256(preprocessor_path) != bundle_manifest["preprocessor_sha256"]:
        raise RuntimeError("Locked preprocessor checksum mismatch")
    preprocessor = FoldPreprocessor.load(preprocessor_path)
    transformed = preprocessor.transform(external)
    device = resolve_device(device_name)
    all_models = tuple(dict.fromkeys(COMPARISON_MODELS + V37_MODELS))
    predictions: dict[str, np.ndarray] = {}
    metric_rows = []
    ci_rows = []
    gate_rows = []
    reuse_root = (
        Path(reuse_predictions_from) if reuse_predictions_from is not None else None
    )
    reuse_records: dict[str, Any] = {}
    attempt = {
        "status": "external_validation_started",
        "bundle_manifest_sha256": _sha256(
            bundle / "locked_bundle_manifest.json"
        ),
        "external_prepared_arrays_sha256": _sha256(external_npz),
        "external_data_manifest_sha256": _sha256(external_manifest_path),
        "followup_imputation": followup_imputation,
        "reuse_predictions_from": str(reuse_root) if reuse_root else None,
        "fit_preprocessing_selection_or_epoch_choice_on_external_data": False,
    }
    (output / "evaluation_attempt_manifest.json").write_text(
        json.dumps(attempt, indent=2), encoding="utf-8"
    )

    for model_index, model_name in enumerate(all_models):
        model_dir = bundle / model_name
        lock = _read_json(model_dir / "manifest.json")
        artifact = model_dir / lock["artifact_file"]
        if _sha256(artifact) != lock["artifact_sha256"]:
            raise RuntimeError(f"{model_name} artifact checksum mismatch")
        diagnostics = None
        reused_path = (
            reuse_root / model_name / "external_predictions.csv"
            if reuse_root is not None
            else None
        )
        reused = reused_path is not None and reused_path.is_file()
        if reused:
            frame, prediction = _load_reused_prediction_frame(
                reused_path,
                external["patient_ids"],
                external["baseline"],
                external["targets"],
            )
            reuse_records[model_name] = {
                "source_file": str(reused_path),
                "source_sha256": _sha256(reused_path),
                "inference_repeated": False,
            }
        elif model_name == "persistence":
            prediction = external["baseline"].astype(float).copy()
        elif model_name in CLASSICAL_MODELS:
            with (model_dir / "model.pkl").open("rb") as handle:
                model = pickle.load(handle)
            prediction = model.predict(transformed)
        elif model_name in DEEP_COMPARATORS:
            model, locked_schema = _load_deep_model(model_dir, device)
            if locked_schema != schema:
                raise RuntimeError(f"{model_name} checkpoint schema mismatch")
            prediction, _, predicted_ids = predict(
                model,
                _loader(transformed, 64, False),
                device,
            )
            if predicted_ids != [str(value) for value in external["patient_ids"]]:
                raise RuntimeError(f"{model_name} changed patient order")
            del model
        elif model_name.startswith("lac_v37_"):
            model, locked_schema = _load_deep_model(model_dir, device)
            if locked_schema != schema:
                raise RuntimeError(f"{model_name} checkpoint schema mismatch")
            base_bundle = _predict_base_with_features(
                model,
                transformed,
                external,
                64,
                device,
                model_name,
            )
            with (model_dir / "decision_layer.pkl").open("rb") as handle:
                decision = pickle.load(handle)
            residual, decision_gates = decision.predict(
                base_bundle["heads"],
                base_bundle["features"],
                _decision_modes(model_name),
            )
            prediction = _residual_to_endpoint(base_bundle["baseline"], residual)
            diagnostics = {
                "decision_gate_tbr": decision_gates["tbr"],
                "decision_gate_cac": decision_gates["cac"],
                "adapter_gate_inflammation": base_bundle["neural_diagnostics"]["adapter_i"],
                "adapter_gate_calcification": base_bundle["neural_diagnostics"]["adapter_c"],
                "coupling_gate": base_bundle["neural_diagnostics"]["coupling_gate"],
                "lag_elapsed_years": base_bundle["neural_diagnostics"]["lag_elapsed"],
            }
            for gate_name, values in diagnostics.items():
                values = np.asarray(values, dtype=float)
                gate_rows.append({
                    "model": model_name,
                    "quantity": gate_name,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q3": float(np.quantile(values, 0.75)),
                    "max": float(np.max(values)),
                    "near_zero_fraction": float(np.mean(np.abs(values) < 0.01)),
                    "near_one_fraction": float(np.mean(np.abs(values - 1.0) < 0.01)),
                })
            del model
        else:
            raise KeyError(model_name)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictions[model_name] = np.asarray(prediction, dtype=float)
        model_output = output / model_name
        model_output.mkdir()
        prediction_path = model_output / "external_predictions.csv"
        if reused:
            shutil.copy2(reused_path, prediction_path)
        else:
            frame = _prediction_frame(
                external["patient_ids"],
                external["baseline"],
                external["targets"],
                predictions[model_name],
                diagnostics,
            )
            frame.to_csv(prediction_path, index=False)
        endpoint = regression_metrics(external["targets"], predictions[model_name])
        change = change_space_metrics(
            external["targets"], predictions[model_name], external["baseline"]
        )
        metric_rows.append({"model": model_name} | endpoint | change)
        for metric, record in bootstrap_change_metrics(
            external["targets"],
            predictions[model_name],
            external["baseline"],
            n_bootstrap=n_bootstrap,
            seed=2026 + model_index,
        ).items():
            ci_rows.append({"model": model_name, "metric": metric} | record)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "all_external_metrics.csv", index=False)
    metrics[metrics["model"].isin(COMPARISON_MODELS)].to_csv(
        output / "comparison_metrics.csv", index=False
    )
    metrics[metrics["model"].isin(TABLE_A_MODELS)].to_csv(
        output / "ablation_table_a_dual_task.csv", index=False
    )
    metrics.loc[
        metrics["model"].isin(TABLE_B_MODELS),
        ["model", "delta_log_cac_mae", "delta_log_cac_rmse", "delta_log_cac_r2"],
    ].to_csv(output / "ablation_table_b_cac_mechanism.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output / "metric_bootstrap_95ci.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output / "gate_diagnostics.csv", index=False)

    paired_rows = []
    for model_index, model_name in enumerate(all_models):
        if model_name == "lac_v37_full":
            continue
        for row in _paired_bootstrap(
            external["targets"],
            external["baseline"],
            predictions["lac_v37_full"],
            predictions[model_name],
            n_bootstrap,
            32026 + model_index,
        ):
            paired_rows.append({"other_model": model_name} | row)
    pd.DataFrame(paired_rows).to_csv(output / "paired_bootstrap_full_vs_all.csv", index=False)

    comparison = metrics.set_index("model").loc[list(COMPARISON_MODELS)]
    full = comparison.loc["lac_v37_full"]
    comparison_rank = {
        metric: int(
            comparison[metric].rank(
                ascending=not metric.endswith("_r2"), method="min"
            ).loc["lac_v37_full"]
        )
        for metric in PRIMARY_METRICS
    }
    ablations = metrics.set_index("model").loc[list(V37_MODELS)]
    pareto_dominators = []
    for model_name, row in ablations.iterrows():
        if model_name == "lac_v37_full":
            continue
        no_worse = (
            row["delta_tbr_mae"] <= full["delta_tbr_mae"]
            and row["delta_tbr_rmse"] <= full["delta_tbr_rmse"]
            and row["delta_tbr_r2"] >= full["delta_tbr_r2"]
            and row["delta_log_cac_mae"] <= full["delta_log_cac_mae"]
            and row["delta_log_cac_rmse"] <= full["delta_log_cac_rmse"]
            and row["delta_log_cac_r2"] >= full["delta_log_cac_r2"]
        )
        strictly_better = (
            row["delta_tbr_mae"] < full["delta_tbr_mae"]
            or row["delta_tbr_rmse"] < full["delta_tbr_rmse"]
            or row["delta_tbr_r2"] > full["delta_tbr_r2"]
            or row["delta_log_cac_mae"] < full["delta_log_cac_mae"]
            or row["delta_log_cac_rmse"] < full["delta_log_cac_rmse"]
            or row["delta_log_cac_r2"] > full["delta_log_cac_r2"]
        )
        if no_worse and strictly_better:
            pareto_dominators.append(model_name)

    result = {
        "status": "external_validation_complete",
        "external_center": "Shandong_Provincial_Cancer_Hospital",
        "external_patient_count": len(external["patient_ids"]),
        "models": list(all_models),
        "comparison_models": list(COMPARISON_MODELS),
        "bootstrap_replicates": n_bootstrap,
        "comparison_full_model_ranks": comparison_rank,
        "full_ablation_pareto_dominators": pareto_dominators,
        "followup_imputation": followup_imputation,
        "reused_prediction_models": list(reuse_records),
        "reused_prediction_records": reuse_records,
        "fit_preprocessing_selection_or_epoch_choice_on_external_data": False,
        "bundle_manifest_sha256": _sha256(bundle / "locked_bundle_manifest.json"),
        "external_data_manifest_sha256": _sha256(external_manifest_path),
        "external_prepared_arrays_sha256": _sha256(external_npz),
        "evaluation_git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
    }
    (output / "external_validation_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def verify_v37_external(bundle_dir: str | Path, result_dir: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    result = Path(result_dir)
    lock = _read_json(bundle / "locked_bundle_manifest.json")
    external = _read_json(result / "external_validation_manifest.json")
    if lock["external_data_used_for_fit_selection_or_epoch_choice"]:
        raise RuntimeError("Bundle reports external leakage")
    if external["fit_preprocessing_selection_or_epoch_choice_on_external_data"]:
        raise RuntimeError("External result reports leakage")
    n = int(external["external_patient_count"])
    for model_name in external["models"]:
        model_dir = bundle / model_name
        manifest = _read_json(model_dir / "manifest.json")
        artifact = model_dir / manifest["artifact_file"]
        if _sha256(artifact) != manifest["artifact_sha256"]:
            raise RuntimeError(f"{model_name} artifact checksum mismatch")
        frame = pd.read_csv(result / model_name / "external_predictions.csv")
        if len(frame) != n or frame["patient_id"].nunique() != n:
            raise RuntimeError(f"{model_name} prediction rows are invalid")
    metrics = pd.read_csv(result / "all_external_metrics.csv")
    if set(metrics["model"]) != set(external["models"]):
        raise RuntimeError("External metric model set differs from manifest")
    return {
        "status": "passed",
        "external_patients": n,
        "models_checked": len(external["models"]),
        "full_center_B_refit": True,
        "external_fit_or_selection": False,
    }
