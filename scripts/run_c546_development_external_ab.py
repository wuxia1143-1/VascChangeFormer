"""Centre-C546 development, five-fold OOF evaluation, and external A/B validation.

Protocol
--------
1. Exclude the privately specified ineligible patient from the 547-patient Centre-C source before development.
2. Run patient-level nested five-fold VascMTL development entirely within C546.
3. Select the full-refit epoch code and decision layer from C546 OOF artifacts,
   fit one final model on all C546 patients, and freeze it.
4. Predict A323 and B31 once with the single frozen full-C546 model.  External
   outcomes are opened only after blinded prediction files have been hashed.
5. Report six change-scale metrics with 2,000 patient-level percentile
   bootstrap replicates (seed 2026): MAE/RMSE/R2 for delta TBR and delta
   log1p(TAC).

The historical internal field name ``cac`` is retained only where required by
the model implementation.  Public result files use TAC terminology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import run_a323_b31_external_validation as external_shared  # noqa: E402
import run_a323_full_refit as refit_shared  # noqa: E402
from lac_itransformer.data.preprocessing import FoldPreprocessor, subset_arrays  # noqa: E402
from lac_itransformer.data.schema import FeatureSchema  # noqa: E402
from lac_itransformer.training.folds import (  # noqa: E402
    build_patient_fold_plan,
    fold_plan_checksum,
    save_patient_fold_plan,
)
from lac_itransformer.training.trainer import (  # noqa: E402
    _runtime_versions,
    patient_id_hash,
    resolve_device,
)
from lac_itransformer.training.v32_trainer import decode_stage_epochs  # noqa: E402
from lac_itransformer.training.v60_nested import run_v60_nested_cross_validation  # noqa: E402
from lac_itransformer.training.v60_selector import V60GenericDecisionLayer  # noqa: E402


VERSION = "C546_DEVELOPMENT_A323_B31_EXTERNAL_20260928"
MODEL = "vascmtl"
DISPLAY_NAME = "VascMTL"
N_SOURCE_C = 547
N_C = 546
N_A = 323
N_B = 31
OUTER_FOLDS = 5
INNER_FOLDS = 5
SEED = 2026
FULL_REFIT_SEED = 3_546_2026
N_BOOT = 2_000
EXPECTED_DIMS = {"static": 19, "longitudinal": 16, "baseline": 2, "treatment": 5}
METRIC_ORDER = (
    "delta_tbr_mae",
    "delta_tbr_rmse",
    "delta_tbr_r2",
    "delta_log_tac_mae",
    "delta_log_tac_rmse",
    "delta_log_tac_r2",
)
LEGACY_TO_PUBLIC = {
    "delta_tbr_mae": "delta_tbr_mae",
    "delta_tbr_rmse": "delta_tbr_rmse",
    "delta_tbr_r2": "delta_tbr_r2",
    "delta_log_cac_mae": "delta_log_tac_mae",
    "delta_log_cac_rmse": "delta_log_tac_rmse",
    "delta_log_cac_r2": "delta_log_tac_r2",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=json_default),
        encoding="utf-8",
    )


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty path: {path}")
    path.mkdir(parents=True, exist_ok=True)


def dimensions(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        "static": int(arrays["static"].shape[1]),
        "longitudinal": int(arrays["values"].shape[2]),
        "baseline": int(arrays["baseline"].shape[1]),
        "treatment": int(arrays["treatments"].shape[2]),
    }


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        text = text.lstrip("0") or "0"
    return text.upper()


def canonical_set(values: np.ndarray) -> set[str]:
    result = {canonical_id(value) for value in values}
    if len(result) != len(values):
        raise RuntimeError("Canonical patient-ID collision within a cohort")
    return result


def load_plan(path: Path, expected_n: int) -> dict[str, int]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if len(frame) != expected_n or frame.patient_id.nunique() != expected_n:
        raise RuntimeError(f"Incomplete fold plan: {path}")
    return {str(row.patient_id): int(row.test_fold) for row in frame.itertuples(index=False)}


def verify_lock(root: Path, expected_hash: str) -> dict[str, Any]:
    path = root / "protocol_lock" / "protocol_lock.json"
    observed = sha256(path)
    if observed != expected_hash:
        raise RuntimeError(f"Protocol lock changed: {observed} != {expected_hash}")
    lock = read_json(path)
    if lock["version"] != VERSION or lock["model"] != MODEL:
        raise RuntimeError("Protocol lock content changed")
    return lock


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    require_empty(root)
    data_dir = root / "prepared_data"
    lock_dir = root / "protocol_lock"
    config_dir = lock_dir / "locked_configs"
    data_dir.mkdir()
    lock_dir.mkdir()
    config_dir.mkdir()

    excluded_c_id = args.exclude_c_id
    if not excluded_c_id:
        raise ValueError("prepare requires --exclude-c-id from your private cohort manifest")
    source_c = load_npz(args.source_c547)
    c_ids = source_c["patient_ids"].astype(str)
    if len(c_ids) != N_SOURCE_C or len(set(c_ids)) != N_SOURCE_C:
        raise RuntimeError("Centre-C source must contain 547 unique patients")
    excluded = c_ids == excluded_c_id
    if int(excluded.sum()) != 1:
        raise RuntimeError(f"Expected exactly one {excluded_c_id}; found {excluded.sum()}")
    if dimensions(source_c) != EXPECTED_DIMS:
        raise RuntimeError(f"Unexpected Centre-C dimensions: {dimensions(source_c)}")
    c = {key: np.asarray(value)[~excluded] for key, value in source_c.items()}
    if len(c["patient_ids"]) != N_C or "targets" not in c:
        raise RuntimeError("Failed to create C546 development cohort")

    source_a = load_npz(args.source_a323)
    if len(source_a["patient_ids"]) != N_A or dimensions(source_a) != EXPECTED_DIMS:
        raise RuntimeError("Invalid A323 source")
    a_predictors = {key: value for key, value in source_a.items() if key != "targets"}
    a_outcomes = {"patient_ids": source_a["patient_ids"], "targets": source_a["targets"]}

    b_predictors = load_npz(args.source_b31_predictors)
    b_outcomes = load_npz(args.source_b31_outcomes)
    if len(b_predictors["patient_ids"]) != N_B or dimensions(b_predictors) != EXPECTED_DIMS:
        raise RuntimeError("Invalid B31 predictor source")
    if "targets" in b_predictors:
        raise RuntimeError("B31 predictor source contains outcomes")
    if not np.array_equal(b_predictors["patient_ids"].astype(str), b_outcomes["patient_ids"].astype(str)):
        raise RuntimeError("B31 predictor/outcome patient order mismatch")

    sets = {
        "C546": canonical_set(c["patient_ids"]),
        "A323": canonical_set(source_a["patient_ids"]),
        "B31": canonical_set(b_predictors["patient_ids"]),
    }
    overlaps = {
        "C546_A323": len(sets["C546"] & sets["A323"]),
        "C546_B31": len(sets["C546"] & sets["B31"]),
        "A323_B31": len(sets["A323"] & sets["B31"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Cross-centre patient overlap detected: {overlaps}")

    c_path = data_dir / "C546_training_arrays.npz"
    a_predictor_path = data_dir / "A323_predictors_without_outcomes.npz"
    a_outcome_path = data_dir / "A323_sealed_outcomes.npz"
    b_predictor_path = data_dir / "B31_predictors_without_outcomes.npz"
    b_outcome_path = data_dir / "B31_sealed_outcomes.npz"
    save_npz(c_path, c)
    save_npz(a_predictor_path, a_predictors)
    save_npz(a_outcome_path, a_outcomes)
    save_npz(b_predictor_path, b_predictors)
    save_npz(b_outcome_path, b_outcomes)
    shutil.copy2(args.source_schema, data_dir / "schema.json")

    plan = build_patient_fold_plan(c["patient_ids"], n_folds=OUTER_FOLDS, seed=SEED)
    checksum = fold_plan_checksum(plan)
    plan_path = lock_dir / "C546_locked_patient_outer_fold_plan.csv"
    plan_manifest = save_patient_fold_plan(plan, plan_path, seed=SEED, n_folds=OUTER_FOLDS)
    if plan_manifest["checksum_sha256"] != checksum:
        raise RuntimeError("Fold plan checksum changed during serialization")

    source_config = Path(args.source_vascmtl_config)
    config = read_json(source_config)
    config.update(
        {
            "seed": SEED,
            "device": args.device,
            "num_folds": OUTER_FOLDS,
            "bootstrap_replicates": N_BOOT,
            "result_scope": VERSION,
            "locked_outer_fold_checksum": checksum,
            "model_name": MODEL,
        }
    )
    nested = dict(config.get("nested", {}))
    nested["inner_folds"] = INNER_FOLDS
    config["nested"] = nested
    config_path = config_dir / "vascmtl.json"
    write_json(config_path, config)

    lock = {
        "status": "locked_before_C546_model_training",
        "version": VERSION,
        "model": MODEL,
        "input_profile": "static19_longitudinal16_baseline2_treatment5",
        "targets": {
            "delta_tbr": "followup_TBR_minus_baseline_TBR",
            "delta_log_tac": "log1p_followup_TAC_minus_log1p_baseline_TAC",
        },
        "cohorts": {
            "C_development": {
                "source_n": N_SOURCE_C,
                "excluded_patient_id": excluded_c_id,
                "exclusion_reason": "age_below_20",
                "n": N_C,
                "patient_hash": patient_id_hash(c["patient_ids"]),
                "arrays_sha256": sha256(c_path),
            },
            "A_external": {
                "n": N_A,
                "patient_hash": patient_id_hash(a_predictors["patient_ids"]),
                "predictors_sha256": sha256(a_predictor_path),
                "sealed_outcomes_sha256": sha256(a_outcome_path),
            },
            "B_external": {
                "n": N_B,
                "patient_hash": patient_id_hash(b_predictors["patient_ids"]),
                "predictors_sha256": sha256(b_predictor_path),
                "sealed_outcomes_sha256": sha256(b_outcome_path),
            },
            "overlaps": overlaps,
        },
        "cross_validation": {
            "unit": "patient",
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seed": SEED,
            "outer_plan_checksum": checksum,
            "outer_fold_counts": plan_manifest["fold_counts"],
            "preprocessing_fit_inside_outer_training_partitions_only": True,
            "one_complete_OOF_prediction_per_C_patient_required": True,
        },
        "model_config": {
            "source_path": str(source_config),
            "source_sha256": sha256(source_config),
            "effective_sha256": sha256(config_path),
            "A_or_B_results_used_for_C_model_selection": False,
        },
        "confidence_intervals": {
            "method": "patient-level percentile bootstrap",
            "level": 0.95,
            "replicates": N_BOOT,
            "seed": SEED,
            "retraining_inside_bootstrap": False,
        },
        "external_validation_rule": "C546 OOF development then one full-C546 refit, freeze, one-time A323 and B31 prediction",
        "A_or_B_used_in_C_preprocessing_training_early_stopping_or_selection": False,
        "script_sha256": sha256(Path(__file__)),
    }
    lock_path = lock_dir / "protocol_lock.json"
    write_json(lock_path, lock)
    manifest = {
        "status": "C546_A323_B31_prepared_and_protocol_locked",
        "protocol_lock_sha256": sha256(lock_path),
        "C_n": N_C,
        "A_n": N_A,
        "B_n": N_B,
        "overlaps": overlaps,
        "outer_fold_counts": plan_manifest["fold_counts"],
        "outer_fold_checksum": checksum,
    }
    write_json(data_dir / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def run_oof(args: argparse.Namespace) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(args.experiment_root)
    lock = verify_lock(root, args.expected_lock_sha256)
    arrays_path = root / "prepared_data" / "C546_training_arrays.npz"
    if sha256(arrays_path) != lock["cohorts"]["C_development"]["arrays_sha256"]:
        raise RuntimeError("C546 training arrays changed")
    arrays = load_npz(arrays_path)
    schema = FeatureSchema.from_dict(read_json(root / "prepared_data" / "schema.json"))
    plan_path = root / "protocol_lock" / "C546_locked_patient_outer_fold_plan.csv"
    plan = load_plan(plan_path, N_C)
    if fold_plan_checksum(plan) != lock["cross_validation"]["outer_plan_checksum"]:
        raise RuntimeError("C546 outer fold plan changed")
    config_path = root / "protocol_lock" / "locked_configs" / "vascmtl.json"
    if sha256(config_path) != lock["model_config"]["effective_sha256"]:
        raise RuntimeError("Locked VascMTL config changed")
    output = root / "models" / MODEL
    if (output / "execution_manifest.json").exists():
        print("MODEL_ALREADY_COMPLETE vascmtl", flush=True)
        return
    require_empty(output)
    started = time.perf_counter()
    summary = run_v60_nested_cross_validation(
        arrays,
        schema,
        read_json(config_path),
        output,
        MODEL,
        outer_fold_assignments=plan,
    )
    if summary["patient_count"] != N_C:
        raise RuntimeError("Incomplete C546 OOF cohort")
    if summary["cv_protocol"]["outer_fold_plan_checksum"] != lock["cross_validation"]["outer_plan_checksum"]:
        raise RuntimeError("C546 OOF fold checksum mismatch")
    oof_path = output / "out_of_fold_predictions.csv"
    oof = pd.read_csv(oof_path, dtype={"patient_id": str})
    if len(oof) != N_C or oof.patient_id.nunique() != N_C:
        raise RuntimeError("Incomplete C546 OOF predictions")
    manifest = {
        "status": "C546_nested_fivefold_OOF_complete",
        "model": MODEL,
        "patient_count": N_C,
        "fit_seconds": time.perf_counter() - started,
        "protocol_lock_sha256": args.expected_lock_sha256,
        "config_sha256": sha256(config_path),
        "summary_sha256": sha256(output / "summary.json"),
        "oof_sha256": sha256(oof_path),
        "A_or_B_outcomes_read_by_training_stage": False,
        "outer_test_patients_used_for_preprocessing_selection_or_training": False,
    }
    write_json(output / "execution_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def normalized_ci_rows(
    setting: str,
    n: int,
    point: dict[str, float],
    ci: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for legacy_name, public_name in LEGACY_TO_PUBLIC.items():
        values = ci[legacy_name]
        outcome, metric = (
            ("Delta TBR", public_name.removeprefix("delta_tbr_").upper())
            if public_name.startswith("delta_tbr_")
            else ("Delta log-TAC", public_name.removeprefix("delta_log_tac_").upper())
        )
        rows.append(
            {
                "validation_setting": setting,
                "n": n,
                "outcome": outcome,
                "metric": metric,
                "metric_key": public_name,
                "estimate": float(point[legacy_name]),
                "ci_low": float(values["ci_low"]),
                "ci_high": float(values["ci_high"]),
                "bootstrap_replicates": N_BOOT,
                "bootstrap_seed": SEED,
            }
        )
    return rows


def aggregate_oof(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    verify_lock(root, args.expected_lock_sha256)
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    output_path = result_dir / "C546_OOF_change_metrics_95ci.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    arrays = load_npz(root / "prepared_data" / "C546_training_arrays.npz")
    oof_path = root / "models" / MODEL / "out_of_fold_predictions.csv"
    execution = read_json(root / "models" / MODEL / "execution_manifest.json")
    if sha256(oof_path) != execution["oof_sha256"]:
        raise RuntimeError("C546 OOF artifact changed")
    oof = pd.read_csv(oof_path, dtype={"patient_id": str})
    expected = arrays["patient_ids"].astype(str)
    if len(oof) != N_C or set(oof.patient_id) != set(expected):
        raise RuntimeError("C546 OOF patient mismatch")
    oof = oof.set_index("patient_id").loc[expected].reset_index()
    baseline = arrays["baseline"].astype(float)
    target = arrays["targets"].astype(float)
    prediction = oof[["pred_tbr", "pred_cac"]].to_numpy(float)
    point, ci = external_shared.bootstrap_metrics(baseline, target, prediction)
    rows = normalized_ci_rows("C OOF", N_C, point, ci)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "C546_OOF_change_metrics_complete",
        "patient_count": N_C,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "oof_sha256": sha256(oof_path),
        "metrics_sha256": sha256(output_path),
    }
    write_json(result_dir / "C546_OOF_metrics_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def fit_c_decision(root: Path, arrays: dict[str, np.ndarray], schema: FeatureSchema, config: dict[str, Any], device: torch.device) -> V60GenericDecisionLayer:
    plan = load_plan(root / "protocol_lock" / "C546_locked_patient_outer_fold_plan.csv", N_C)
    index = {str(value): i for i, value in enumerate(arrays["patient_ids"])}
    bundles = []
    first_transformed = FoldPreprocessor.load(root / "models" / MODEL / "fold_1" / "preprocessor.json").transform(arrays)
    generic_feature_count = int(refit_shared._generic_temporal_context(first_transformed).shape[1])
    model_config = refit_shared._config_for_model(schema, config, MODEL)
    for fold in range(1, OUTER_FOLDS + 1):
        ids = [patient_id for patient_id, value in plan.items() if value == fold]
        raw = subset_arrays(arrays, np.asarray([index[patient_id] for patient_id in ids], dtype=int))
        fold_dir = root / "models" / MODEL / f"fold_{fold}"
        checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
        model = refit_shared.build_torch_model(MODEL, model_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        transformed = FoldPreprocessor.load(fold_dir / "preprocessor.json").transform(raw)
        bundle = refit_shared.vasc_base(model, transformed, raw, device, int(config.get("batch_size", 64)))
        bundle["meta_fold"] = np.full(len(ids), fold, dtype=int)
        bundles.append(bundle)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    joined = {
        "heads": {key: np.concatenate([bundle["heads"][key] for bundle in bundles]) for key in ("tbr", "cac")},
        "features": {key: np.concatenate([bundle["features"][key] for bundle in bundles]) for key in ("tbr", "cac")},
        "residual_targets": np.concatenate([bundle["residual_targets"] for bundle in bundles]),
        "meta_fold": np.concatenate([bundle["meta_fold"] for bundle in bundles]),
    }
    old_count = V60GenericDecisionLayer.configured_generic_feature_count
    old_leaf = V60GenericDecisionLayer.configured_min_samples_leaf
    try:
        V60GenericDecisionLayer.configured_generic_feature_count = generic_feature_count
        V60GenericDecisionLayer.configured_min_samples_leaf = int(config.get("decision", {}).get("generic_min_samples_leaf", 20))
        options = dict(config.get("decision", {}))
        selector = V60GenericDecisionLayer(
            candidate_weights=tuple(options["candidate_weights"]),
            degradation_tolerance=float(options["degradation_tolerance"]),
            required_consistent_folds=int(options["required_consistent_folds"]),
            seed=SEED,
        ).fit(
            heads=joined["heads"],
            targets={"tbr": joined["residual_targets"][:, 0], "cac": joined["residual_targets"][:, 1]},
            features=joined["features"],
            meta_fold=joined["meta_fold"],
        )
    finally:
        V60GenericDecisionLayer.configured_generic_feature_count = old_count
        V60GenericDecisionLayer.configured_min_samples_leaf = old_leaf
    return selector


def refit_full_c(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    lock = verify_lock(root, args.expected_lock_sha256)
    if not (root / "results" / "C546_OOF_change_metrics_95ci.csv").exists():
        raise RuntimeError("C546 OOF evaluation must be complete before full-C546 refit")
    out = root / "frozen_models" / MODEL
    require_empty(out)
    arrays = load_npz(root / "prepared_data" / "C546_training_arrays.npz")
    schema = FeatureSchema.from_dict(read_json(root / "prepared_data" / "schema.json"))
    config_path = root / "protocol_lock" / "locked_configs" / "vascmtl.json"
    config = read_json(config_path)
    device = resolve_device(args.device)
    started = time.perf_counter()

    preprocessor = FoldPreprocessor.fit(arrays, patient_id_hash(arrays["patient_ids"]))
    preprocessor.save(out / "preprocessor.json")
    transformed = preprocessor.transform(arrays)
    selector = fit_c_decision(root, arrays, schema, config, device)
    with (out / "decision_layer.pkl").open("wb") as handle:
        pickle.dump(selector, handle)

    summary_path = root / "models" / MODEL / "summary.json"
    summary = read_json(summary_path)
    epoch_codes = [int(record["selected_epochs"]) for record in summary["fold_records"]]
    if len(epoch_codes) != OUTER_FOLDS:
        raise RuntimeError("Expected five selected epoch codes")
    epoch_code = int(np.median(epoch_codes))
    refit_shared._seed_everything_v60(FULL_REFIT_SEED)
    model_config = refit_shared._config_for_model(schema, config, MODEL)
    model = refit_shared.build_torch_model(MODEL, model_config).to(device)
    model, history = refit_shared.fit_v60_multitask_fixed_epochs(
        model, transformed, config, device, epoch_code, FULL_REFIT_SEED
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": MODEL,
            "model_config": model_config.to_dict(),
            "schema": schema.to_dict(),
            "locked_epoch_code": epoch_code,
            "decoded_stage_epochs": decode_stage_epochs(epoch_code),
            "full_refit_seed": FULL_REFIT_SEED,
            "external_validation_locked": True,
        },
        out / "model.pt",
    )
    write_json(out / "refit_history.json", history)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    manifest = {
        "status": "full_C546_final_model_frozen",
        "protocol": "C546_nested_fivefold_OOF_then_full_C546_refit_then_external_A323_B31_prediction",
        "patient_count": N_C,
        "patient_hash": patient_id_hash(arrays["patient_ids"]),
        "input_dimensions": EXPECTED_DIMS,
        "source_summary_sha256": sha256(summary_path),
        "source_config_sha256": sha256(config_path),
        "selected_fold_epoch_codes": epoch_codes,
        "locked_epoch_code": epoch_code,
        "decoded_stage_epochs": decode_stage_epochs(epoch_code),
        "full_refit_seed": FULL_REFIT_SEED,
        "A_or_B_outcomes_used": False,
        "point_prediction_source_for_A_and_B": "single_full_C546_frozen_final_model",
        "five_fold_models_to_be_averaged": False,
        "device": str(device),
        "runtime_versions": _runtime_versions(),
        "fit_seconds": time.perf_counter() - started,
        "artifacts": {path.name: sha256(path) for path in out.iterdir() if path.is_file()},
        "protocol_lock_sha256": args.expected_lock_sha256,
        "C_arrays_sha256": lock["cohorts"]["C_development"]["arrays_sha256"],
    }
    write_json(root / "frozen_models" / "full_refit_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def external_paths(root: Path, centre: str) -> dict[str, Path]:
    prefix = "A323" if centre == "A" else "B31"
    out = root / "external_validation" / centre
    return {
        "out": out,
        "predictors": root / "prepared_data" / f"{prefix}_predictors_without_outcomes.npz",
        "outcomes": root / "prepared_data" / f"{prefix}_sealed_outcomes.npz",
        "predictions": out / f"C546_to_{prefix}_predictions_blinded.csv",
        "lock": out / "prediction_lock_manifest.json",
        "metrics": out / f"C546_to_{prefix}_change_metrics_95ci.csv",
        "patients": out / f"C546_to_{prefix}_VascMTL_patient_predictions.csv",
    }


def predict_external(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    lock = verify_lock(root, args.expected_lock_sha256)
    full_manifest_path = root / "frozen_models" / "full_refit_manifest.json"
    full_manifest = read_json(full_manifest_path)
    if full_manifest["status"] != "full_C546_final_model_frozen":
        raise RuntimeError("Full-C546 frozen model manifest mismatch")
    schema = FeatureSchema.from_dict(read_json(root / "prepared_data" / "schema.json"))
    config = read_json(root / "protocol_lock" / "locked_configs" / "vascmtl.json")
    model_dir = root / "frozen_models" / MODEL
    device = resolve_device(args.device)
    for centre, expected_n in (("A", N_A), ("B", N_B)):
        paths = external_paths(root, centre)
        paths["out"].mkdir(parents=True, exist_ok=True)
        if paths["predictions"].exists() or paths["lock"].exists():
            raise FileExistsError(f"External predictions already locked for Centre {centre}")
        source = load_npz(paths["predictors"])
        if "targets" in source:
            raise RuntimeError(f"Centre {centre} predictor file contains outcomes")
        raw = external_shared.add_dummy_targets(source)
        prediction = external_shared.predict_fold(MODEL, model_dir, raw, schema, config, device)
        if prediction.shape != (expected_n, 2) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid Centre {centre} predictions")
        frame = pd.DataFrame(
            {
                "patient_id": raw["patient_ids"].astype(str),
                "baseline_tbr": raw["baseline"][:, 0].astype(float),
                "baseline_tac": raw["baseline"][:, 1].astype(float),
                "pred_tbr": prediction[:, 0],
                "pred_tac": prediction[:, 1],
            }
        )
        frame.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
        prediction_lock = {
            "status": f"C546_to_{centre}_predictions_locked_before_outcome_evaluation",
            "external_patient_count": expected_n,
            "patient_id_hash": patient_id_hash(frame.patient_id.to_numpy(str)),
            "prediction_file_sha256": sha256(paths["predictions"]),
            "prediction_has_outcomes": False,
            "point_prediction_source": "single_full_C546_frozen_final_model",
            "five_fold_models_averaged": False,
            "full_refit_manifest_sha256": sha256(full_manifest_path),
            "model_artifact_sha256": sha256(model_dir / "model.pt"),
            "preprocessor_sha256": sha256(model_dir / "preprocessor.json"),
            "decision_layer_sha256": sha256(model_dir / "decision_layer.pkl"),
            "external_outcomes_read": False,
            "protocol_lock_sha256": args.expected_lock_sha256,
            "expected_external_outcome_sha256": lock["cohorts"][f"{centre}_external"]["sealed_outcomes_sha256"],
        }
        write_json(paths["lock"], prediction_lock)
        print(json.dumps(prediction_lock, indent=2), flush=True)


def evaluate_one_external(root: Path, centre: str, expected_n: int) -> list[dict[str, Any]]:
    paths = external_paths(root, centre)
    lock = read_json(paths["lock"])
    if sha256(paths["predictions"]) != lock["prediction_file_sha256"]:
        raise RuntimeError(f"Centre {centre} prediction lock mismatch")
    if lock["prediction_has_outcomes"] or lock["external_outcomes_read"]:
        raise RuntimeError(f"Centre {centre} blinded-prediction certification failed")
    predictors = load_npz(paths["predictors"])
    outcomes = load_npz(paths["outcomes"])
    if not np.array_equal(predictors["patient_ids"].astype(str), outcomes["patient_ids"].astype(str)):
        raise RuntimeError(f"Centre {centre} predictor/outcome order mismatch")
    frame = pd.read_csv(paths["predictions"], dtype={"patient_id": str})
    if not np.array_equal(frame.patient_id.to_numpy(str), outcomes["patient_ids"].astype(str)):
        raise RuntimeError(f"Centre {centre} prediction/outcome order mismatch")
    baseline = predictors["baseline"].astype(float)
    target = outcomes["targets"].astype(float)
    prediction = frame[["pred_tbr", "pred_tac"]].to_numpy(float)
    point, ci = external_shared.bootstrap_metrics(baseline, target, prediction)
    setting = "C→A" if centre == "A" else "C→B"
    rows = normalized_ci_rows(setting, expected_n, point, ci)
    pd.DataFrame(rows).to_csv(paths["metrics"], index=False, encoding="utf-8-sig")

    true_delta_tbr = target[:, 0] - baseline[:, 0]
    pred_delta_tbr = prediction[:, 0] - baseline[:, 0]
    true_delta_log_tac = np.log1p(np.maximum(target[:, 1], 0)) - np.log1p(np.maximum(baseline[:, 1], 0))
    pred_delta_log_tac = np.log1p(np.maximum(prediction[:, 1], 0)) - np.log1p(np.maximum(baseline[:, 1], 0))
    patient = pd.DataFrame(
        {
            "patient_id": outcomes["patient_ids"].astype(str),
            "baseline_tbr": baseline[:, 0],
            "baseline_tac": baseline[:, 1],
            "observed_delta_tbr": true_delta_tbr,
            "predicted_delta_tbr": pred_delta_tbr,
            "observed_delta_log_tac": true_delta_log_tac,
            "predicted_delta_log_tac": pred_delta_log_tac,
            "delta_tbr_residual_predicted_minus_observed": pred_delta_tbr - true_delta_tbr,
            "delta_log_tac_residual_predicted_minus_observed": pred_delta_log_tac - true_delta_log_tac,
        }
    )
    patient.to_csv(paths["patients"], index=False, encoding="utf-8-sig")
    evaluation = {
        "status": f"C546_to_{centre}_external_validation_complete",
        "external_patient_count": expected_n,
        "prediction_file_sha256": sha256(paths["predictions"]),
        "outcome_file_sha256": sha256(paths["outcomes"]),
        "metrics_file_sha256": sha256(paths["metrics"]),
        "patient_predictions_sha256": sha256(paths["patients"]),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "external_centre_used_for_fit_preprocessing_early_stopping_or_selection": False,
    }
    write_json(paths["out"] / "external_validation_manifest.json", evaluation)
    return rows


def table_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def evaluate_and_summarize(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    lock = verify_lock(root, args.expected_lock_sha256)
    c_path = root / "results" / "C546_OOF_change_metrics_95ci.csv"
    c_rows = pd.read_csv(c_path).to_dict("records")
    a_rows = evaluate_one_external(root, "A", N_A)
    b_rows = evaluate_one_external(root, "B", N_B)
    long_frame = pd.DataFrame(c_rows + a_rows + b_rows)
    ordered_settings = ["C OOF", "C→A", "C→B"]
    long_frame["validation_setting"] = pd.Categorical(
        long_frame["validation_setting"], ordered_settings, ordered=True
    )
    long_frame["metric_key"] = pd.Categorical(long_frame["metric_key"], METRIC_ORDER, ordered=True)
    long_frame = long_frame.sort_values(["validation_setting", "metric_key"]).reset_index(drop=True)
    long_path = root / "results" / "C_OOF_C_to_A_C_to_B_performance_long.csv"
    long_frame.to_csv(long_path, index=False, encoding="utf-8-sig")

    display_names = {
        "delta_tbr_mae": "ΔTBR MAE",
        "delta_tbr_rmse": "ΔTBR RMSE",
        "delta_tbr_r2": "ΔTBR R²",
        "delta_log_tac_mae": "Δlog-TAC MAE",
        "delta_log_tac_rmse": "Δlog-TAC RMSE",
        "delta_log_tac_r2": "Δlog-TAC R²",
    }
    wide_rows = []
    for setting in ordered_settings:
        subset = long_frame[long_frame.validation_setting == setting].set_index("metric_key")
        row: dict[str, Any] = {
            "Validation": setting,
            "n": int(subset["n"].iloc[0]),
        }
        for key in METRIC_ORDER:
            values = subset.loc[key]
            row[display_names[key]] = f"{values.estimate:.4f} [{values.ci_low:.4f}, {values.ci_high:.4f}]"
        wide_rows.append(row)
    table = pd.DataFrame(wide_rows)
    table_path = root / "results" / "C_OOF_C_to_A_C_to_B_performance_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    report = "\n".join(
        [
            "# C中心开发及A、B中心独立外部验证：六项连续预测性能",
            "",
            table_markdown(table),
            "",
            "## 统计口径",
            "",
            "- C OOF：Centre C 546例患者级嵌套五折完整OOF预测。",
            "- C→A / C→B：在完整C546上全量重训并冻结的单一最终模型，分别一次性预测A323和B31；未对五折模型取均值。",
            "- 所有结局均按变化量计算：ΔTBR=随访TBR−基线TBR；Δlog-TAC=log1p(随访TAC)−log1p(基线TAC)。",
            f"- 95%CI为{N_BOOT}次患者级percentile bootstrap（seed={SEED}）；bootstrap过程中不重新训练模型。",
            "- A、B均未参与C中心的预处理拟合、超参数/训练轮数选择、早停、决策层拟合或full-C546重训。",
            "",
        ]
    )
    report_path = root / "results" / "C_DEVELOPMENT_EXTERNAL_AB_PERFORMANCE_REPORT_CN.md"
    report_path.write_text(report, encoding="utf-8")

    audit = {
        "status": "PASS",
        "cohorts": {"C_development_n": N_C, "A_external_n": N_A, "B_external_n": N_B},
        "C_excluded_patient": lock["cohorts"]["C_development"]["excluded_patient_id"],
        "patient_overlap": lock["cohorts"]["overlaps"],
        "C_OOF_complete": len(c_rows) == 6,
        "full_C546_single_model_frozen": True,
        "A_and_B_predictions_from_full_C546_not_fold_average": True,
        "A_or_B_used_in_C_development": False,
        "A_and_B_outcomes_opened_only_after_prediction_lock": True,
        "input_dimensions": EXPECTED_DIMS,
        "targets_change_scale": True,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "TAC_public_naming_only": all("cac" not in column.lower() for column in table.columns),
        "outputs": {
            "long_table": {"path": str(long_path), "sha256": sha256(long_path)},
            "wide_table": {"path": str(table_path), "sha256": sha256(table_path)},
            "report": {"path": str(report_path), "sha256": sha256(report_path)},
        },
    }
    write_json(root / "results" / "independent_audit.json", audit)
    print(report, flush=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "run", "aggregate", "refit", "predict", "evaluate"),
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument(
        "--source-c547",
        default=str(ROOT / "private_data" / "C547_prepared_arrays.npz"),
    )
    parser.add_argument(
        "--source-schema",
        default=str(ROOT / "configs" / "schema.json"),
    )
    parser.add_argument(
        "--source-a323",
        default=str(ROOT / "runs" / "A323" / "prepared_data" / "A323_training_arrays.npz"),
    )
    parser.add_argument(
        "--source-b31-predictors",
        default=str(ROOT / "runs" / "A323" / "prepared_data" / "B31_predictors_without_outcomes.npz"),
    )
    parser.add_argument(
        "--source-b31-outcomes",
        default=str(ROOT / "runs" / "A323" / "prepared_data" / "B31_sealed_outcomes.npz"),
    )
    parser.add_argument(
        "--source-vascmtl-config",
        default=str(ROOT / "runs" / "A323" / "protocol_lock" / "locked_configs" / "vascmtl.json"),
    )
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exclude-c-id")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        if not args.expected_lock_sha256:
            parser.error(f"{args.stage} requires --expected-lock-sha256")
        if args.stage == "run":
            run_oof(args)
        elif args.stage == "aggregate":
            aggregate_oof(args)
        elif args.stage == "refit":
            refit_full_c(args)
        elif args.stage == "predict":
            predict_external(args)
        else:
            evaluate_and_summarize(args)


if __name__ == "__main__":
    main()
