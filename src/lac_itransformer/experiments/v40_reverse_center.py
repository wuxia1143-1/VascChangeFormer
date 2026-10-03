from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
import torch

from ..config import load_yaml
from ..data.preprocessing import FoldPreprocessor, subset_arrays
from ..data.schema import FeatureSchema
from ..data.shandong_external import ShandongExternalJSONReader
from ..models.lac import LACConfig
from ..training.folds import (
    build_patient_fold_plan,
    fold_plan_checksum,
    save_patient_fold_plan,
    validate_patient_fold_plan,
)
from ..training.metrics import (
    bootstrap_change_metrics,
    change_space_metrics,
    regression_metrics,
)
from ..training.trainer import (
    _git_revision,
    _loader,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    predict,
    resolve_device,
    seed_everything,
)
from ..training.v27_nested import _predict_base_with_features, _residual_to_endpoint
from ..training.v32_nested import _componentwise_epoch_median
from ..training.v37_external import (
    _canonical_hash,
    _fold_epoch_values,
    load_npz,
    _paired_bootstrap,
    _prediction_frame,
    _validate_arrays,
)
from ..training.v40_final_external import (
    _final_refit_seed,
    _fit_locked_v40_decision,
    _load_v40_model,
)
from ..training.v40_final_nested import (
    _config_for_model,
    _seed_everything_v40_final,
    run_v40_final_nested_cross_validation,
)
from ..training.v41_trainer import fit_v41_fixed_epochs
from .prediction import build_classical_model, build_torch_model
from .real_internal import load_prepared_real_cohort, run_real_nested_model


V40_FULL = "lac_v40_final_full"
V40_CANDIDATE = "lac_v40_final_no_tail"
REVERSE_BASELINES = (
    "persistence",
    "elastic_net",
    "xgboost",
    "itransformer_mtl",
)
CLASSICAL_BASELINES = {"persistence", "elastic_net", "xgboost"}
PHENOTYPES = (
    "stable",
    "inflammation_dominant",
    "calcification_dominant",
    "dual_active",
)
SUPPORTED_REVERSE_CENTER_B_COUNTS = {710, 754}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
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


def _save_prepared(path: Path, arrays: dict[str, np.ndarray], schema: FeatureSchema) -> None:
    np.savez_compressed(
        path,
        **arrays,
        schema_json=np.asarray([json.dumps(schema.to_dict(), ensure_ascii=False)]),
    )


def prepare_reverse_center_b(
    center_b_json: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    """Prepare B without reading A arrays or labels and lock the shared outer plan."""
    output = _prepare_output(output_dir)
    schema = FeatureSchema.from_dict(_read_json(schema_path))
    center_b_source = Path(center_b_json)
    if center_b_source.suffix.lower() == ".npz":
        arrays = load_npz(center_b_source)
        audit_payload: dict[str, Any] = {
            "source_type": "locked_prepared_npz",
            "source_manifest": str(center_b_source.with_name("external_data_manifest.json")),
        }
    else:
        arrays, audit = ShandongExternalJSONReader(schema).prepare_arrays(center_b_source)
        audit_payload = audit.to_dict()
    patient_count = len(arrays["patient_ids"])
    if patient_count not in SUPPORTED_REVERSE_CENTER_B_COUNTS:
        raise ValueError(
            "Locked reverse protocol requires a supported center-B cohort "
            f"{sorted(SUPPORTED_REVERSE_CENTER_B_COUNTS)}; got {patient_count}"
        )
    followup = np.asarray(arrays["followup_months"], dtype=float)
    observed = np.isfinite(followup)
    if not observed.any():
        raise ValueError("Center B has no observed follow-up interval")
    fill_value = float(np.median(followup[observed]))
    arrays["followup_months"] = followup.copy()
    arrays["followup_months"][~observed] = fill_value
    if not np.isfinite(arrays["followup_months"]).all():
        raise RuntimeError("Center-B-only follow-up imputation is incomplete")
    _validate_arrays(arrays, schema, "reverse_development_center_B")

    prepared = output / "prepared_arrays.npz"
    _save_prepared(prepared, arrays, schema)
    (output / "schema.json").write_text(
        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plan = build_patient_fold_plan(arrays["patient_ids"], n_folds=5, seed=seed)
    plan_manifest = save_patient_fold_plan(
        plan, output / "locked_patient_outer_fold_plan.csv", seed=seed, n_folds=5
    )
    result = {
        "status": "reverse_center_B_prepared_and_locked_before_center_A_access",
        "development_center": f"B_{patient_count}",
        "patient_count": patient_count,
        "patient_hash": patient_id_hash(arrays["patient_ids"]),
        "source_sha256": _sha256(center_b_json),
        "prepared_arrays_sha256": _sha256(prepared),
        "schema_sha256": _canonical_hash(schema.to_dict()),
        "followup_imputation": {
            "missing_count": int((~observed).sum()),
            "fill_value_months": fill_value,
            "source": "median_of_observed_center_B_only",
            "center_A_used": False,
        },
        "fold_plan": plan_manifest,
        "reader_audit": audit_payload,
        "center_A_arrays_or_labels_read": False,
    }
    (output / "center_b_preparation_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _locked_plan(path: str | Path, patient_ids: np.ndarray) -> dict[str, int]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    plan = dict(zip(frame["patient_id"], frame["test_fold"].astype(int)))
    validate_patient_fold_plan(patient_ids, plan, 5)
    return plan


def _derive_no_tail_oof(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    full_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    plan: dict[str, int],
) -> dict[str, Any]:
    output = _prepare_output(output_dir)
    device = resolve_device(str(config.get("device", "auto")))
    raw_index = {str(value): index for index, value in enumerate(arrays["patient_ids"])}
    frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold in range(1, 6):
        identifiers = [value for value, assigned in plan.items() if assigned == fold]
        raw = subset_arrays(arrays, [raw_index[value] for value in identifiers])
        fold_dir = full_dir / f"fold_{fold}"
        checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
        model = build_torch_model(
            V40_FULL, _config_for_model(schema, config, V40_FULL)
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        preprocessor = FoldPreprocessor.load(fold_dir / "preprocessor.json")
        bundle = _predict_base_with_features(
            model,
            preprocessor.transform(raw),
            raw,
            int(config.get("batch_size", 64)),
            device,
            V40_FULL,
        )
        with (fold_dir / "decision_layer.pkl").open("rb") as handle:
            decision = pickle.load(handle)
        residual, gates = decision.predict(
            bundle["heads"], bundle["features"], {"tbr": "identity", "cac": "no_tail"}
        )
        endpoint = _residual_to_endpoint(bundle["baseline"], residual)
        frame = _prediction_frame(
            np.asarray(bundle["patient_ids"]),
            bundle["baseline"],
            bundle["endpoint_targets"],
            endpoint,
            {
                "decision_gate_tbr": gates["tbr"],
                "decision_gate_cac": gates["cac"],
                "adapter_gate_inflammation": bundle["neural_diagnostics"]["adapter_i"],
                "adapter_gate_calcification": bundle["neural_diagnostics"]["adapter_c"],
                "coupling_gate": bundle["neural_diagnostics"]["coupling_gate"],
                "lag_elapsed_years": bundle["neural_diagnostics"]["lag_elapsed"],
            },
        )
        frame["outer_fold"] = fold
        frames.append(frame)
        endpoint_metrics = regression_metrics(bundle["endpoint_targets"], endpoint)
        change_metrics = change_space_metrics(
            bundle["endpoint_targets"], endpoint, bundle["baseline"]
        )
        fold_records.append({"outer_fold": fold, "n_test": len(frame)} | endpoint_metrics | change_metrics)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(frames, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    if len(oof) != len(arrays["patient_ids"]) or oof["patient_id"].nunique() != len(oof):
        raise RuntimeError("Derived no-tail OOF does not contain every B patient exactly once")
    oof.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / "outer_fold_metrics.csv", index=False)
    target = oof[["true_tbr", "true_cac"]].to_numpy(float)
    prediction = oof[["pred_tbr", "pred_cac"]].to_numpy(float)
    baseline = oof[["baseline_tbr", "baseline_cac"]].to_numpy(float)
    summary = {
        "status": "derived_no_tail_from_exact_full_fold_neural_states",
        "model_name": V40_CANDIDATE,
        "patient_count": len(oof),
        "parameter_count": _read_json(full_dir / "summary.json")["parameter_count"],
        "pooled_oof_metrics": regression_metrics(target, prediction),
        "pooled_change_metrics": change_space_metrics(target, prediction, baseline),
        "change_bootstrap_95_ci": bootstrap_change_metrics(
            target,
            prediction,
            baseline,
            n_bootstrap=int(config.get("bootstrap_replicates", 2000)),
            seed=int(config.get("seed", 2026)),
        ),
        "fold_records": fold_records,
        "cv_protocol": {
            "level": "patient",
            "outer_folds": 5,
            "inner_folds": 5,
            "outer_fold_plan_checksum": fold_plan_checksum(plan),
            "decision_mode": {"tbr": "identity", "cac": "no_tail"},
            "derived_from_full_neural_state": True,
            "center_A_used": False,
        },
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_reverse_internal_v40(
    prepared_b: str | Path,
    config_path: str | Path,
    locked_plan_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    arrays, schema = load_prepared_real_cohort(prepared_b)
    plan = _locked_plan(locked_plan_path, arrays["patient_ids"])
    config = load_yaml(config_path)
    config["locked_outer_fold_checksum"] = fold_plan_checksum(plan)
    config["result_scope"] = "reverse_center_B_internal_nested_v40_final"
    config["reverse_protocol"] = {
        "development_center": f"B_{len(arrays['patient_ids'])}",
        "external_center": "A_443_not_read",
        "center_A_used": False,
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    full_dir = root / V40_FULL
    if (full_dir / "summary.json").is_file():
        full = _read_json(full_dir / "summary.json")
    elif full_dir.exists() and any(full_dir.iterdir()):
        raise RuntimeError(f"Partial V4 output will not be overwritten: {full_dir}")
    else:
        full = run_v40_final_nested_cross_validation(
            arrays, schema, config, full_dir, V40_FULL, outer_fold_assignments=plan
        )
    no_tail_dir = root / V40_CANDIDATE
    if (no_tail_dir / "summary.json").is_file():
        no_tail = _read_json(no_tail_dir / "summary.json")
    else:
        no_tail = _derive_no_tail_oof(
            arrays, schema, full_dir, no_tail_dir, config, plan
        )
    result = {
        "status": "reverse_center_B_v40_internal_complete",
        "full": full["pooled_change_metrics"],
        "no_tail": no_tail["pooled_change_metrics"],
        "outer_fold_plan_checksum": fold_plan_checksum(plan),
        "center_A_used": False,
    }
    (root / "v40_internal_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_reverse_internal_baselines(
    prepared_b: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    models: Iterable[str] = REVERSE_BASELINES,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    config["result_scope"] = "real_internal_nested_fivefold"
    selected = tuple(dict.fromkeys(models))
    unknown = set(selected) - set(REVERSE_BASELINES)
    if unknown:
        raise KeyError(f"Unsupported reverse baselines: {sorted(unknown)}")
    result = {
        name: run_real_nested_model(prepared_b, config, output_root, name)
        for name in selected
    }
    root = Path(output_root)
    (root / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plan_source = root / "locked_patient_outer_fold_plan.csv"
    if plan_source.is_file():
        pd.read_csv(plan_source, dtype={"patient_id": str}).to_csv(
            root / "patient_fold_plan.csv", index=False
        )
    manifest = {
        "status": "reverse_center_B_baseline_internal_complete",
        "models": list(selected),
        "center_A_used": False,
        "metrics": {name: value["pooled_change_metrics"] for name, value in result.items()},
    }
    (root / "reverse_baseline_internal_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / len(ordered)


def _build_threshold_lock(oof: pd.DataFrame) -> dict[str, Any]:
    reference_tbr = oof["pred_delta_tbr"].to_numpy(float)
    reference_cac = oof["pred_delta_log_cac"].to_numpy(float)
    risk = 0.5 * (
        _percentile_against(reference_tbr, reference_tbr)
        + _percentile_against(reference_cac, reference_cac)
    )
    return {
        "status": "locked_from_center_B_oof_before_center_A_evaluation",
        "truth_high_progressor": {
            "tbr_q90": float(oof["true_delta_tbr"].quantile(0.90)),
            "cac_q90": float(oof["true_delta_log_cac"].quantile(0.90)),
        },
        "prediction_high_cutoff": {
            "tbr_q90": float(oof["pred_delta_tbr"].quantile(0.90)),
            "cac_q90": float(oof["pred_delta_log_cac"].quantile(0.90)),
        },
        "joint_risk_score_cutoffs": [float(value) for value in np.quantile(risk, [1 / 3, 2 / 3])],
        "joint_phenotype_thresholds": {
            "inflammation_median": float(oof["true_delta_tbr"].median()),
            "calcification_median": float(oof["true_delta_log_cac"].median()),
        },
        "reference_prediction_distributions": {
            "tbr": reference_tbr.tolist(),
            "cac": reference_cac.tolist(),
        },
        "center_A_used": False,
    }


def finalize_reverse_bundle(
    prepared_b: str | Path,
    v40_internal_root: str | Path,
    baseline_internal_root: str | Path,
    baseline_config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Freeze all choices and thresholds on B before any A arrays are supplied."""
    output = _prepare_output(output_dir)
    arrays, schema = load_prepared_real_cohort(prepared_b)
    patient_count = len(arrays["patient_ids"])
    if patient_count not in SUPPORTED_REVERSE_CENTER_B_COUNTS:
        raise ValueError(
            "Reverse finalization requires a supported center-B cohort; "
            f"got {patient_count}"
        )
    _validate_arrays(arrays, schema, "reverse_development_center_B")
    v40_root = Path(v40_internal_root)
    baseline_root = Path(baseline_internal_root)
    device = resolve_device(device_name)
    development_hash = patient_id_hash(arrays["patient_ids"])
    preprocessor = FoldPreprocessor.fit(arrays, development_hash)
    transformed = preprocessor.transform(arrays)
    preprocessor.save(output / "preprocessor.json")
    (output / "schema.json").write_text(json.dumps(schema.to_dict(), indent=2), encoding="utf-8")

    full_source = v40_root / V40_FULL
    no_tail_source = v40_root / V40_CANDIDATE
    full_summary = _read_json(full_source / "summary.json")
    no_tail_summary = _read_json(no_tail_source / "summary.json")
    if (
        int(full_summary["patient_count"]) != patient_count
        or int(no_tail_summary["patient_count"]) != patient_count
    ):
        raise ValueError(
            f"V4 reverse OOF sources do not contain all {patient_count} center-B patients"
        )
    shared_checksum = full_summary["cv_protocol"]["outer_fold_plan_checksum"]
    for name in REVERSE_BASELINES:
        baseline_checksum = _read_json(baseline_root / name / "summary.json")[
            "cv_protocol"
        ]["outer_fold_plan_checksum"]
        if baseline_checksum != shared_checksum:
            raise RuntimeError(
                f"{name} does not use the same center-B outer folds as V4.0-final"
            )
    config = _read_json(full_source / "training_config.json")
    epoch_values = _fold_epoch_values(full_summary)
    locked_epochs = _componentwise_epoch_median(epoch_values)
    seed = _final_refit_seed(config)
    selector, decision_audit = _fit_locked_v40_decision(
        arrays, full_source, V40_FULL, config, device
    )
    _seed_everything_v40_final(seed)
    v40_config = _config_for_model(schema, config, V40_FULL)
    v40_model = build_torch_model(V40_FULL, v40_config).to(device)
    v40_model, history = fit_v41_fixed_epochs(
        v40_model, transformed, config, device, locked_epochs, seed
    )
    v40_dir = output / V40_CANDIDATE
    v40_dir.mkdir()
    torch.save(
        {
            "state_dict": v40_model.state_dict(),
            "model_name": V40_FULL,
            "model_config": v40_config.to_dict(),
            "schema": schema.to_dict(),
        },
        v40_dir / "model.pt",
    )
    with (v40_dir / "decision_layer.pkl").open("wb") as handle:
        pickle.dump(selector, handle)
    v40_record = {
        "model_name": V40_CANDIDATE,
        "neural_model_name": V40_FULL,
        "decision_modes": {"tbr": "identity", "cac": "no_tail"},
        "locked_stage_epoch_codes": epoch_values,
        "locked_final_stage_epoch_code": locked_epochs,
        "decision_audit": decision_audit,
        "fit_history": history,
        "seed": seed,
        "artifact_sha256": _sha256(v40_dir / "model.pt"),
        "decision_sha256": _sha256(v40_dir / "decision_layer.pkl"),
        "internal_oof_metrics": no_tail_summary["pooled_change_metrics"],
        "center_A_used": False,
    }
    (v40_dir / "manifest.json").write_text(json.dumps(v40_record, indent=2), encoding="utf-8")
    del v40_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_config = load_yaml(baseline_config_path)
    baseline_records: dict[str, Any] = {}
    for index, name in enumerate(REVERSE_BASELINES):
        source = baseline_root / name
        summary = _read_json(source / "summary.json")
        model_seed = int(baseline_config.get("seed", 2026)) + 10_000 + index * 100
        seed_everything(model_seed)
        model_dir = output / name
        model_dir.mkdir()
        if name == "persistence":
            artifact = model_dir / "model_meta.json"
            artifact.write_text("{}", encoding="utf-8")
            model = None
            fit_history: dict[str, Any] = {"not_required": True}
            locked_model_epochs = None
        elif name in CLASSICAL_BASELINES:
            options = baseline_config.get("classical", {}).get(name, {})
            model = build_classical_model(name, seed=model_seed, **options).fit(transformed)
            artifact = model_dir / "model.pkl"
            with artifact.open("wb") as handle:
                pickle.dump(model, handle)
            fit_history = {"fit_on_all_center_B_patients": True}
            locked_model_epochs = None
        else:
            checkpoint = torch.load(
                source / "fold_1" / "model.pt", map_location="cpu", weights_only=False
            )
            model_config = LACConfig(**checkpoint["model_config"])
            model = build_torch_model(name, model_config).to(device)
            fold_epochs = [int(record["selected_epochs"]) for record in summary["fold_records"]]
            locked_model_epochs = int(np.median(fold_epochs))
            model, fit_history = fit_model_fixed_epochs(
                model, transformed, baseline_config, device, locked_model_epochs
            )
            artifact = model_dir / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": name,
                    "model_config": checkpoint["model_config"],
                    "schema": schema.to_dict(),
                },
                artifact,
            )
        record = {
            "model_name": name,
            "artifact_file": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "seed": model_seed,
            "locked_final_epochs": locked_model_epochs,
            "fit_history": fit_history,
            "internal_oof_metrics": summary["pooled_change_metrics"],
            "center_A_used": False,
        }
        (model_dir / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        baseline_records[name] = record
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    threshold_lock = _build_threshold_lock(
        pd.read_csv(no_tail_source / "out_of_fold_predictions.csv", dtype={"patient_id": str})
    )
    (output / "analysis_threshold_lock.json").write_text(
        json.dumps(threshold_lock, indent=2), encoding="utf-8"
    )
    result = {
        "status": "reverse_center_B_full_refit_bundle_locked_before_A",
        "development_center": f"B_{patient_count}",
        "development_patient_count": len(arrays["patient_ids"]),
        "development_patient_hash": development_hash,
        "development_prepared_arrays_sha256": _sha256(prepared_b),
        "models": [V40_CANDIDATE, *REVERSE_BASELINES],
        "v40_record": v40_record,
        "baseline_records": baseline_records,
        "preprocessor_sha256": _sha256(output / "preprocessor.json"),
        "schema_sha256": _canonical_hash(schema.to_dict()),
        "threshold_lock_sha256": _sha256(output / "analysis_threshold_lock.json"),
        "external_center_A_used_for_fit_preprocessing_selection_calibration_or_thresholds": False,
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
    }
    (output / "locked_bundle_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def evaluate_reverse_center_a(
    prepared_a: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """The only function in this protocol that accepts center-A arrays."""
    output = _prepare_output(output_dir)
    bundle = Path(bundle_dir)
    lock = _read_json(bundle / "locked_bundle_manifest.json")
    if lock["external_center_A_used_for_fit_preprocessing_selection_calibration_or_thresholds"]:
        raise RuntimeError("Reverse bundle reports center-A leakage")
    if _sha256(bundle / "analysis_threshold_lock.json") != lock["threshold_lock_sha256"]:
        raise RuntimeError("Center-B threshold lock checksum changed")
    arrays, schema = load_prepared_real_cohort(prepared_a)
    if len(arrays["patient_ids"]) != 443:
        raise ValueError(
            f"Locked reverse protocol requires 443 center-A patients; got {len(arrays['patient_ids'])}"
        )
    _validate_arrays(arrays, schema, "reverse_external_center_A")
    if _canonical_hash(schema.to_dict()) != lock["schema_sha256"]:
        raise ValueError("Center-A schema differs from the frozen center-B schema")
    if patient_id_hash(arrays["patient_ids"]) == lock["development_patient_hash"]:
        raise ValueError("Center-A and center-B cohorts unexpectedly have the same identity hash")
    preprocessor = FoldPreprocessor.load(bundle / "preprocessor.json")
    if preprocessor.fitted_patient_ids_hash != lock["development_patient_hash"]:
        raise RuntimeError("Frozen preprocessor provenance differs from center B")
    transformed = preprocessor.transform(arrays)
    device = resolve_device(device_name)
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []

    for model_index, name in enumerate(lock["models"]):
        model_dir = bundle / name
        record = _read_json(model_dir / "manifest.json")
        artifact = model_dir / record.get("artifact_file", "model.pt")
        if _sha256(artifact) != record["artifact_sha256"]:
            raise RuntimeError(f"{name} frozen artifact checksum changed")
        if name == V40_CANDIDATE:
            model, locked_schema, checkpoint_name = _load_v40_model(model_dir, device)
            if locked_schema != schema:
                raise ValueError("V4 checkpoint schema differs from center A")
            base = _predict_base_with_features(
                model, transformed, arrays, 64, device, checkpoint_name
            )
            with (model_dir / "decision_layer.pkl").open("rb") as handle:
                decision = pickle.load(handle)
            residual, gates = decision.predict(
                base["heads"], base["features"], record["decision_modes"]
            )
            prediction = _residual_to_endpoint(base["baseline"], residual)
            diagnostics = {
                "decision_gate_tbr": gates["tbr"],
                "decision_gate_cac": gates["cac"],
            }
        elif name == "persistence":
            prediction = np.asarray(arrays["baseline"], float).copy()
            diagnostics = None
            model = None
        elif name in CLASSICAL_BASELINES:
            with artifact.open("rb") as handle:
                model = pickle.load(handle)
            prediction = model.predict(transformed)
            diagnostics = None
        else:
            checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
            model = build_torch_model(name, LACConfig(**checkpoint["model_config"])).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            prediction, predicted_target, identifiers = predict(
                model, _loader(transformed, 64, False), device
            )
            if identifiers != [str(value) for value in arrays["patient_ids"]]:
                raise RuntimeError(f"{name} changed center-A patient order")
            if not np.allclose(predicted_target, arrays["targets"]):
                raise RuntimeError(f"{name} target order differs from center-A arrays")
            diagnostics = None
        prediction = np.asarray(prediction, float)
        if prediction.shape != arrays["targets"].shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"{name} produced invalid center-A predictions")
        predictions[name] = prediction
        model_output = output / name
        model_output.mkdir()
        _prediction_frame(
            arrays["patient_ids"], arrays["baseline"], arrays["targets"], prediction, diagnostics
        ).to_csv(model_output / "external_predictions.csv", index=False)
        endpoint = regression_metrics(arrays["targets"], prediction)
        change = change_space_metrics(arrays["targets"], prediction, arrays["baseline"])
        metric_rows.append({"model": name} | endpoint | change)
        for metric, values in bootstrap_change_metrics(
            arrays["targets"],
            prediction,
            arrays["baseline"],
            n_bootstrap=n_bootstrap,
            seed=20260818 + model_index,
        ).items():
            ci_rows.append({"model": name, "metric": metric} | values)
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(metric_rows).to_csv(output / "reverse_external_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output / "reverse_metric_bootstrap_95ci.csv", index=False)
    paired_rows = []
    for model_index, name in enumerate(lock["models"]):
        if name == V40_CANDIDATE:
            continue
        for row in _paired_bootstrap(
            arrays["targets"],
            arrays["baseline"],
            predictions[V40_CANDIDATE],
            predictions[name],
            n_bootstrap,
            52026 + model_index,
        ):
            paired_rows.append({"candidate": V40_CANDIDATE, "reference": name} | row)
    pd.DataFrame(paired_rows).to_csv(output / "reverse_paired_v40_vs_baselines.csv", index=False)
    result = {
        "status": "reverse_center_A_single_frozen_evaluation_complete",
        "external_center": "A_443",
        "external_patient_count": len(arrays["patient_ids"]),
        "external_patient_hash": patient_id_hash(arrays["patient_ids"]),
        "external_prepared_arrays_sha256": _sha256(prepared_a),
        "models": lock["models"],
        "bootstrap_replicates": n_bootstrap,
        "fit_preprocessing_selection_calibration_or_thresholding_on_center_A": False,
        "locked_bundle_manifest_sha256": _sha256(bundle / "locked_bundle_manifest.json"),
        "threshold_lock_sha256": lock["threshold_lock_sha256"],
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
    }
    (output / "reverse_external_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _bootstrap_mean(values: np.ndarray, seed: int, replicates: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        samples[index] = values[rng.integers(0, len(values), len(values))].mean()
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def _assign_phenotypes(tbr: np.ndarray, cac: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    code = (
        (np.asarray(tbr, float) >= thresholds["inflammation_median"]).astype(int)
        + 2 * (np.asarray(cac, float) >= thresholds["calcification_median"]).astype(int)
    )
    return np.asarray(PHENOTYPES)[code]


def build_reverse_report(
    v40_internal_root: str | Path,
    baseline_internal_root: str | Path,
    bundle_dir: str | Path,
    external_result_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = _prepare_output(output_dir)
    v40_root = Path(v40_internal_root)
    baseline_root = Path(baseline_internal_root)
    bundle = Path(bundle_dir)
    external = Path(external_result_dir)
    threshold_lock = _read_json(bundle / "analysis_threshold_lock.json")
    lock_manifest = _read_json(bundle / "locked_bundle_manifest.json")
    external_manifest = _read_json(external / "reverse_external_manifest.json")
    if external_manifest["threshold_lock_sha256"] != _sha256(bundle / "analysis_threshold_lock.json"):
        raise RuntimeError("Reported center-A evaluation did not use the current B threshold lock")

    internal_rows = []
    for name in lock_manifest["models"]:
        source = v40_root / name if name == V40_CANDIDATE else baseline_root / name
        summary = _read_json(source / "summary.json")
        internal_rows.append({"model": name} | summary["pooled_change_metrics"])
    internal_metrics = pd.DataFrame(internal_rows)
    external_metrics = pd.read_csv(external / "reverse_external_metrics.csv")
    internal_metrics.insert(0, "evaluation", "B_internal_OOF")
    external_metrics.insert(0, "evaluation", "A_reverse_external")
    performance = pd.concat([internal_metrics, external_metrics], ignore_index=True)
    performance.to_csv(output / "01_reverse_prediction_and_baselines.csv", index=False)
    pd.read_csv(external / "reverse_paired_v40_vs_baselines.csv").to_csv(
        output / "02_reverse_paired_v40_vs_baselines.csv", index=False
    )

    b_oof = pd.read_csv(
        v40_root / V40_CANDIDATE / "out_of_fold_predictions.csv", dtype={"patient_id": str}
    )
    if _canonical_hash(_build_threshold_lock(b_oof)) != _canonical_hash(threshold_lock):
        raise RuntimeError("Frozen clinical thresholds no longer match center-B OOF")
    a_pred = pd.read_csv(
        external / V40_CANDIDATE / "external_predictions.csv", dtype={"patient_id": str}
    )
    high_rows = []
    for task, true_column, prediction_column in (
        ("TBR", "true_delta_tbr", "pred_delta_tbr"),
        ("CAC", "true_delta_log_cac", "pred_delta_log_cac"),
    ):
        key = task.lower()
        truth_threshold = threshold_lock["truth_high_progressor"][f"{key}_q90"]
        prediction_cutoff = threshold_lock["prediction_high_cutoff"][f"{key}_q90"]
        truth = a_pred[true_column].to_numpy(float) >= truth_threshold
        score = a_pred[prediction_column].to_numpy(float)
        predicted = score >= prediction_cutoff
        tp = int(np.sum(truth & predicted))
        fp = int(np.sum(~truth & predicted))
        fn = int(np.sum(truth & ~predicted))
        tn = int(np.sum(~truth & ~predicted))
        top = score >= np.quantile(score, 0.90)
        prevalence = float(truth.mean())
        top_prevalence = float(truth[top].mean())
        high_rows.append(
            {
                "task": task,
                "locked_B_truth_q90": truth_threshold,
                "locked_B_prediction_q90": prediction_cutoff,
                "A_high_progressor_n": int(truth.sum()),
                "A_prevalence": prevalence,
                "A_auroc": float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else np.nan,
                "A_auprc": float(average_precision_score(truth, score)) if truth.any() else np.nan,
                "A_sensitivity": tp / max(tp + fn, 1),
                "A_specificity": tn / max(tn + fp, 1),
                "A_precision": tp / max(tp + fp, 1),
                "A_top_decile_prevalence": top_prevalence,
                "A_top_decile_enrichment": top_prevalence / max(prevalence, 1e-12),
            }
        )
    high = pd.DataFrame(high_rows)
    high.to_csv(output / "03_reverse_high_progressor_enrichment.csv", index=False)

    reference_tbr = np.asarray(threshold_lock["reference_prediction_distributions"]["tbr"])
    reference_cac = np.asarray(threshold_lock["reference_prediction_distributions"]["cac"])
    score = 0.5 * (
        _percentile_against(reference_tbr, a_pred["pred_delta_tbr"])
        + _percentile_against(reference_cac, a_pred["pred_delta_log_cac"])
    )
    groups = np.asarray(["low", "middle", "high"])[
        np.digitize(score, threshold_lock["joint_risk_score_cutoffs"], right=True)
    ]
    patient_risk = a_pred[["patient_id"]].copy()
    patient_risk["locked_B_reference_risk_score"] = score
    patient_risk["risk_group"] = groups
    patient_risk.to_csv(output / "04_reverse_patient_risk_groups.csv", index=False)
    risk_rows = []
    for group_index, group in enumerate(("low", "middle", "high")):
        selected = groups == group
        for task_index, (task, column) in enumerate(
            (("TBR", "true_delta_tbr"), ("CAC", "true_delta_log_cac"))
        ):
            values = a_pred.loc[selected, column].to_numpy(float)
            ci_low, ci_high = _bootstrap_mean(values, 20260818 + 10 * group_index + task_index)
            risk_rows.append(
                {
                    "risk_group": group,
                    "task": task,
                    "n": int(selected.sum()),
                    "A_true_change_mean": float(values.mean()),
                    "A_true_change_median": float(np.median(values)),
                    "mean_ci_low": ci_low,
                    "mean_ci_high": ci_high,
                }
            )
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(output / "04_reverse_risk_group_outcomes.csv", index=False)

    phenotype_thresholds = threshold_lock["joint_phenotype_thresholds"]
    observed = _assign_phenotypes(
        a_pred["true_delta_tbr"], a_pred["true_delta_log_cac"], phenotype_thresholds
    )
    predicted = _assign_phenotypes(
        a_pred["pred_delta_tbr"], a_pred["pred_delta_log_cac"], phenotype_thresholds
    )
    phenotype_patients = a_pred[["patient_id"]].copy()
    phenotype_patients["observed_phenotype"] = observed
    phenotype_patients["predicted_phenotype"] = predicted
    phenotype_patients.to_csv(output / "05_reverse_joint_phenotype_patients.csv", index=False)
    distribution_rows = []
    observed_distribution = []
    predicted_distribution = []
    for pattern in PHENOTYPES:
        observed_fraction = float(np.mean(observed == pattern))
        predicted_fraction = float(np.mean(predicted == pattern))
        observed_distribution.append(observed_fraction)
        predicted_distribution.append(predicted_fraction)
        distribution_rows.append(
            {
                "phenotype": pattern,
                "A_observed_n": int(np.sum(observed == pattern)),
                "A_observed_fraction": observed_fraction,
                "A_predicted_n": int(np.sum(predicted == pattern)),
                "A_predicted_fraction": predicted_fraction,
            }
        )
    pd.DataFrame(distribution_rows).to_csv(
        output / "05_reverse_joint_phenotype_distribution.csv", index=False
    )
    phenotype_performance = {
        "locked_B_thresholds": phenotype_thresholds,
        "A_accuracy": float(accuracy_score(observed, predicted)),
        "A_macro_f1": float(f1_score(observed, predicted, labels=PHENOTYPES, average="macro")),
        "A_cohen_kappa": float(cohen_kappa_score(observed, predicted, labels=PHENOTYPES)),
        "A_observed_vs_predicted_jensen_shannon_distance": float(
            jensenshannon(observed_distribution, predicted_distribution)
        ),
    }
    (output / "05_reverse_joint_phenotype_performance.json").write_text(
        json.dumps(phenotype_performance, indent=2), encoding="utf-8"
    )

    candidate = performance.loc[performance.model == V40_CANDIDATE].set_index("evaluation")
    report = f"""# V4.0-final no-tail 反向跨中心可迁移性分析

## 协议状态

B 中心独立完成 nested 5-fold、全 B 重拟合、OOF 稳健校准和全部临床阈值锁定；A 中心只进行一次冻结推理。该分析是反向可迁移性压力测试，不是新的盲法独立验证。

## 1. 反向预测性能

- B 内部 OOF：TBR MAE {candidate.loc['B_internal_OOF', 'delta_tbr_mae']:.4f}，CAC MAE {candidate.loc['B_internal_OOF', 'delta_log_cac_mae']:.4f}。
- A 反向外部：TBR MAE {candidate.loc['A_reverse_external', 'delta_tbr_mae']:.4f}，CAC MAE {candidate.loc['A_reverse_external', 'delta_log_cac_mae']:.4f}。

完整的 V4 与四个基线、配对 bootstrap、风险分层和联合表型结果分别见同目录 CSV/JSON。任何阈值均来自 B OOF，没有根据 A 结果移动。

## 解释边界

A 标签曾用于既往开发，因此本结果只能支持方向反转后的可迁移性，不能替代前瞻性盲法外部验证；B→A 与 A→B 也不应简单平均。
"""
    (output / "V40_REVERSE_CENTER_REPORT_CN.md").write_text(report, encoding="utf-8")
    result = {
        "status": "reverse_center_report_complete",
        "models": lock_manifest["models"],
        "thresholds_fitted_on_center_B_oof_only": True,
        "center_A_used_for_fit_or_thresholds": False,
        "performance_file": "01_reverse_prediction_and_baselines.csv",
        "high_progressor_file": "03_reverse_high_progressor_enrichment.csv",
        "risk_file": "04_reverse_risk_group_outcomes.csv",
        "joint_phenotype_file": "05_reverse_joint_phenotype_performance.json",
    }
    (output / "reverse_report_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V4.0-final no-tail B-to-A reverse-center protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-b")
    prepare.add_argument("--center-b-json", required=True)
    prepare.add_argument("--schema", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=2026)

    internal_v40 = subparsers.add_parser("internal-v40")
    internal_v40.add_argument("--prepared-b", required=True)
    internal_v40.add_argument("--config", required=True)
    internal_v40.add_argument("--locked-plan", required=True)
    internal_v40.add_argument("--output", required=True)

    internal_baselines = subparsers.add_parser("internal-baselines")
    internal_baselines.add_argument("--prepared-b", required=True)
    internal_baselines.add_argument("--config", required=True)
    internal_baselines.add_argument("--output", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--prepared-b", required=True)
    finalize.add_argument("--v40-internal", required=True)
    finalize.add_argument("--baseline-internal", required=True)
    finalize.add_argument("--baseline-config", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate-a")
    evaluate.add_argument("--prepared-a", required=True)
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--bootstrap", type=int, default=2000)

    report = subparsers.add_parser("report")
    report.add_argument("--v40-internal", required=True)
    report.add_argument("--baseline-internal", required=True)
    report.add_argument("--bundle", required=True)
    report.add_argument("--external-result", required=True)
    report.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "prepare-b":
        result = prepare_reverse_center_b(
            args.center_b_json, args.schema, args.output, seed=args.seed
        )
    elif args.command == "internal-v40":
        result = run_reverse_internal_v40(
            args.prepared_b, args.config, args.locked_plan, args.output
        )
    elif args.command == "internal-baselines":
        result = run_reverse_internal_baselines(args.prepared_b, args.config, args.output)
    elif args.command == "finalize":
        result = finalize_reverse_bundle(
            args.prepared_b,
            args.v40_internal,
            args.baseline_internal,
            args.baseline_config,
            args.output,
            device_name=args.device,
        )
    elif args.command == "evaluate-a":
        result = evaluate_reverse_center_a(
            args.prepared_a,
            args.bundle,
            args.output,
            device_name=args.device,
            n_bootstrap=args.bootstrap,
        )
    else:
        result = build_reverse_report(
            args.v40_internal,
            args.baseline_internal,
            args.bundle,
            args.external_result,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
