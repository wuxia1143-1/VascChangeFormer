"""A323 internal nested fivefold comparison after sealing 31 patients as B.

The split is identifier-only.  Centre-B outcomes are saved separately and are
never read by the model-training stage.  The A323 outer folds are the original
locked A354 folds restricted to the remaining patients; every inner fold is
rebuilt inside its corresponding outer-training pool by the audited runners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_INPUT_ROOT = ROOT / "private_data"
SOURCE_SNAPSHOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_SNAPSHOT))

from lac_itransformer.data.schema import FeatureSchema  # noqa: E402
from lac_itransformer.training.folds import (  # noqa: E402
    fold_plan_checksum,
    save_patient_fold_plan,
)
from lac_itransformer.training.nested import (  # noqa: E402
    run_nested_cross_validation,
)
from lac_itransformer.training.trainer import patient_id_hash  # noqa: E402
from lac_itransformer.training.v60_nested import (  # noqa: E402
    run_v60_nested_cross_validation,
)


VERSION = os.environ.get(
    "A323_INTERNAL_VERSION", "A323_B31_SPLIT_STATIC19_LONG16_20260926"
)
MODELS = (
    "vascmtl",
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
)
LABELS = {
    "vascmtl": "VascMTL",
    "persistence": "Persistence",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "apn_dr": "APN-DR",
    "itransformer_mtl": "iTransformer-MTL",
    "first_icu_mtl": "FIRST-ICU-MTL",
    "learning_to_route": "Learning-to-Route",
}
N_A = 323
N_B = 31
OUTER_FOLDS = 5
INNER_FOLDS = 5
SEED = 2026
N_BOOT = 2_000
METRICS = (
    "tbr_mae",
    "tbr_rmse",
    "tbr_r2_endpoint",
    "tbr_r2_change",
    "log_cac_mae",
    "log_cac_rmse",
    "log_cac_r2_endpoint",
    "log_cac_r2_change",
)


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
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        text = text.lstrip("0") or "0"
    return text


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty path: {path}")
    path.mkdir(parents=True, exist_ok=True)


def subset(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in arrays.items()}


def restrict_input_profile(
    arrays: dict[str, np.ndarray], profile: str
) -> dict[str, np.ndarray]:
    """Select a prespecified input profile without changing patients or outcomes."""
    if profile == "full_93_27":
        expected = (93, 27, 2, 5)
        selected = dict(arrays)
    elif profile == "legacy_19_16":
        selected = dict(arrays)
        selected["static"] = arrays["static"][:, :19]
        for key in (
            "values",
            "mask",
            "delta",
            "irregular_values",
            "irregular_mask",
        ):
            selected[key] = arrays[key][..., :16]
        expected = (19, 16, 2, 5)
    else:
        raise ValueError(f"Unknown input profile: {profile}")
    observed = (
        int(selected["static"].shape[1]),
        int(selected["values"].shape[2]),
        int(selected["baseline"].shape[1]),
        int(selected["treatments"].shape[2]),
    )
    if observed != expected:
        raise RuntimeError(f"Input dimensions {observed} do not match {expected}")
    return selected


def load_plan(path: Path, expected_n: int) -> dict[str, int]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if len(frame) != expected_n or frame.patient_id.nunique() != expected_n:
        raise RuntimeError(f"Fold plan is incomplete: {path}")
    return {
        str(row.patient_id): int(row.test_fold)
        for row in frame.itertuples(index=False)
    }


def verify_lock(root: Path, expected_hash: str) -> dict[str, Any]:
    path = root / "protocol_lock" / "protocol_lock.json"
    observed = sha256(path)
    if observed != expected_hash:
        raise RuntimeError(f"Protocol lock changed: {observed} != {expected_hash}")
    lock = read_json(path)
    if lock["version"] != VERSION or tuple(lock["models"]) != MODELS:
        raise RuntimeError("Protocol lock content changed")
    return lock


def prepare(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment_root)
    require_empty(experiment)
    lock_dir = experiment / "protocol_lock"
    data_dir = experiment / "prepared_data"
    config_dir = lock_dir / "locked_configs"
    lock_dir.mkdir()
    data_dir.mkdir()
    config_dir.mkdir()

    external_ids = tuple(line.strip() for line in Path(args.b31_ids_file).read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if len(external_ids) != N_B or len({canonical_id(value) for value in external_ids}) != N_B:
        raise ValueError("The private B31 ID file must contain 31 unique identifiers")
    source_arrays = Path(args.source_arrays)
    source_schema = Path(args.source_schema)
    source_plan = Path(args.source_plan)
    arrays = restrict_input_profile(load_npz(source_arrays), args.input_profile)
    schema_payload = read_json(source_schema)
    expected_schema = (19, 16, 5) if args.input_profile == "legacy_19_16" else (93, 27, 5)
    observed_schema = (
        len(schema_payload["static_features"]),
        len(schema_payload["longitudinal_features"]),
        len(schema_payload["treatment_features"]),
    )
    if observed_schema != expected_schema:
        raise RuntimeError(
            f"Schema dimensions {observed_schema} do not match {expected_schema}"
        )
    patient_ids = arrays["patient_ids"].astype(str)
    if len(patient_ids) != 354 or len(set(patient_ids)) != 354:
        raise RuntimeError("Expected the unique A354 full-input cohort")
    canonical_to_index: dict[str, int] = {}
    for index, patient_id in enumerate(patient_ids):
        canonical = canonical_id(patient_id)
        if canonical in canonical_to_index:
            raise RuntimeError(f"Canonical patient-ID collision: {canonical}")
        canonical_to_index[canonical] = index
    missing = [value for value in external_ids if canonical_id(value) not in canonical_to_index]
    if missing:
        raise RuntimeError(f"Requested B31 identifiers absent from A354: {missing}")
    external_canonical = {canonical_id(value) for value in external_ids}
    b_indices = np.asarray(
        [index for index, value in enumerate(patient_ids) if canonical_id(value) in external_canonical]
    )
    a_indices = np.asarray(
        [index for index, value in enumerate(patient_ids) if canonical_id(value) not in external_canonical]
    )
    if len(a_indices) != N_A or len(b_indices) != N_B:
        raise RuntimeError("A323/B31 split count mismatch")
    a_arrays = subset(arrays, a_indices)
    b_arrays = subset(arrays, b_indices)
    if set(a_arrays["patient_ids"]) & set(b_arrays["patient_ids"]):
        raise RuntimeError("A/B patient overlap")

    a_path = data_dir / "A323_training_arrays.npz"
    b_predictor_path = data_dir / "B31_predictors_without_outcomes.npz"
    b_outcome_path = data_dir / "B31_sealed_outcomes.npz"
    np.savez_compressed(a_path, **a_arrays)
    np.savez_compressed(
        b_predictor_path,
        **{key: value for key, value in b_arrays.items() if key != "targets"},
    )
    np.savez_compressed(
        b_outcome_path,
        patient_ids=b_arrays["patient_ids"],
        targets=b_arrays["targets"],
    )
    shutil.copy2(source_schema, data_dir / "schema.json")

    original_plan = load_plan(source_plan, 354)
    if set(original_plan) != set(patient_ids):
        raise RuntimeError("A354 arrays and frozen fold plan use different patients")
    a_plan = {
        patient_id: original_plan[patient_id]
        for patient_id in patient_ids
        if canonical_id(patient_id) not in external_canonical
    }
    checksum = fold_plan_checksum(a_plan)
    plan_manifest = save_patient_fold_plan(
        a_plan,
        lock_dir / "A323_locked_patient_outer_fold_plan.csv",
        seed=SEED,
        n_folds=OUTER_FOLDS,
    )
    if plan_manifest["checksum_sha256"] != checksum:
        raise RuntimeError("Fold-plan checksum changed during serialization")
    fold_counts = {
        str(fold): int(sum(value == fold for value in a_plan.values()))
        for fold in range(1, OUTER_FOLDS + 1)
    }

    baseline_root = Path(args.baseline_config_root)
    config_records = {}
    for model in MODELS:
        source = (
            Path(args.vascmtl_config)
            if model == "vascmtl"
            else baseline_root / model / "training_config.json"
        )
        config = read_json(source)
        config.update(
            {
                "seed": SEED,
                "device": args.device,
                "num_folds": OUTER_FOLDS,
                "bootstrap_replicates": N_BOOT,
                "result_scope": VERSION,
                "locked_outer_fold_checksum": checksum,
                "model_name": model,
            }
        )
        if model != "vascmtl":
            nested = dict(config.get("nested", {}))
            nested["inner_folds"] = INNER_FOLDS
            config["nested"] = nested
        destination = config_dir / f"{model}.json"
        write_json(destination, config)
        config_records[model] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "effective_sha256": sha256(destination),
        }

    b_stored_ids = [str(value) for value in b_arrays["patient_ids"]]
    lock = {
        "status": "locked_before_A323_model_training",
        "version": VERSION,
        "split_requested_date": "2026-09-26",
        "split_basis": "31 user-specified patient identifiers only; no outcomes used",
        "cohorts": {
            "A_development": {
                "n": N_A,
                "patient_hash": patient_id_hash(a_arrays["patient_ids"]),
                "arrays_sha256": sha256(a_path),
            },
            "B_independent_external": {
                "n": N_B,
                "requested_ids": list(external_ids),
                "stored_ids": b_stored_ids,
                "patient_hash": patient_id_hash(b_arrays["patient_ids"]),
                "predictors_without_outcomes_sha256": sha256(b_predictor_path),
                "sealed_outcomes_sha256": sha256(b_outcome_path),
                "outcomes_used_for_A_model_development": False,
                "external_performance_in_current_request": "not_evaluated",
            },
            "overlap": 0,
        },
        "data": {
            "input_profile": args.input_profile,
            "source_A354_arrays": str(source_arrays),
            "source_A354_arrays_sha256": sha256(source_arrays),
            "schema_sha256": sha256(data_dir / "schema.json"),
            "input_dimensions": {
                "static": int(a_arrays["static"].shape[1]),
                "baseline": int(a_arrays["baseline"].shape[1]),
                "longitudinal": int(a_arrays["values"].shape[2]),
                "treatment": int(a_arrays["treatments"].shape[2]),
            },
        },
        "cross_validation": {
            "unit": "patient",
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "outer_plan": "original frozen A354 assignments restricted to A323",
            "outer_plan_checksum": checksum,
            "outer_fold_counts": fold_counts,
            "same_outer_plan_for_all_models": True,
            "preprocessing_and_selection_fit_inside_training_partitions_only": True,
            "one_complete_OOF_prediction_per_A_patient_required": True,
        },
        "models": list(MODELS),
        "model_labels": LABELS,
        "model_configs": config_records,
        "metrics": list(METRICS),
        "confidence_intervals": {
            "method": "patient-level percentile bootstrap of complete OOF predictions",
            "level": 0.95,
            "replicates": N_BOOT,
            "seed": SEED,
            "same_patient_indices_for_all_models_and_metrics": True,
            "retraining_inside_bootstrap": False,
        },
        "script_sha256": sha256(Path(__file__)),
    }
    lock_path = lock_dir / "protocol_lock.json"
    write_json(lock_path, lock)
    manifest = {
        "status": "A323_B31_split_and_protocol_locked",
        "protocol_lock_sha256": sha256(lock_path),
        "A_n": N_A,
        "B_n": N_B,
        "A_B_overlap": 0,
        "outer_fold_counts": fold_counts,
        "outer_fold_checksum": checksum,
    }
    write_json(data_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def run_model(args: argparse.Namespace) -> None:
    if args.model not in MODELS:
        raise KeyError(args.model)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(args.experiment_root)
    lock = verify_lock(root, args.expected_lock_sha256)
    arrays_path = root / "prepared_data" / "A323_training_arrays.npz"
    if sha256(arrays_path) != lock["cohorts"]["A_development"]["arrays_sha256"]:
        raise RuntimeError("A323 training arrays changed")
    b_outcomes = root / "prepared_data" / "B31_sealed_outcomes.npz"
    if sha256(b_outcomes) != lock["cohorts"]["B_independent_external"]["sealed_outcomes_sha256"]:
        raise RuntimeError("B31 sealed outcomes changed")
    arrays = load_npz(arrays_path)
    if len(arrays["patient_ids"]) != N_A:
        raise RuntimeError("A323 training cohort changed")
    schema = FeatureSchema.from_dict(read_json(root / "prepared_data" / "schema.json"))
    plan = load_plan(root / "protocol_lock" / "A323_locked_patient_outer_fold_plan.csv", N_A)
    if fold_plan_checksum(plan) != lock["cross_validation"]["outer_plan_checksum"]:
        raise RuntimeError("A323 outer-fold plan changed")
    config_path = root / "protocol_lock" / "locked_configs" / f"{args.model}.json"
    if sha256(config_path) != lock["model_configs"][args.model]["effective_sha256"]:
        raise RuntimeError("Locked model configuration changed")
    config = read_json(config_path)
    output = root / "models" / args.model
    if (output / "execution_manifest.json").exists():
        print(f"MODEL_ALREADY_COMPLETE {args.model}", flush=True)
        return
    require_empty(output)
    started = time.perf_counter()
    print(f"MODEL_START {args.model}", flush=True)
    if args.model == "vascmtl":
        summary = run_v60_nested_cross_validation(
            arrays,
            schema,
            config,
            output,
            args.model,
            outer_fold_assignments=plan,
        )
    else:
        summary = run_nested_cross_validation(
            arrays,
            schema,
            config,
            output,
            outer_fold_assignments=plan,
        )
    if summary["patient_count"] != N_A:
        raise RuntimeError(f"Incomplete OOF cohort: {args.model}")
    if summary["cv_protocol"]["outer_fold_plan_checksum"] != lock["cross_validation"]["outer_plan_checksum"]:
        raise RuntimeError(f"Outer-fold checksum mismatch: {args.model}")
    oof_path = output / "out_of_fold_predictions.csv"
    oof = pd.read_csv(oof_path, dtype={"patient_id": str})
    if len(oof) != N_A or oof.patient_id.nunique() != N_A:
        raise RuntimeError(f"Incomplete OOF predictions: {args.model}")
    manifest = {
        "status": "complete",
        "model": args.model,
        "display_name": LABELS[args.model],
        "patient_count": N_A,
        "fit_seconds": time.perf_counter() - started,
        "protocol_lock_sha256": args.expected_lock_sha256,
        "config_sha256": sha256(config_path),
        "summary_sha256": sha256(output / "summary.json"),
        "oof_sha256": sha256(oof_path),
        "B31_outcomes_read_by_training_stage": False,
        "outer_test_patients_used_for_preprocessing_selection_or_training": False,
    }
    write_json(output / "execution_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def safe_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(prediction - target).sum() / denominator)


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    true_tbr = frame.true_tbr.to_numpy(float)
    pred_tbr = frame.pred_tbr.to_numpy(float)
    base_tbr = frame.baseline_tbr.to_numpy(float)
    true_log_cac = np.log1p(np.maximum(frame.true_cac.to_numpy(float), 0.0))
    pred_log_cac = np.log1p(np.maximum(frame.pred_cac.to_numpy(float), 0.0))
    base_log_cac = np.log1p(np.maximum(frame.baseline_cac.to_numpy(float), 0.0))
    tbr_error = pred_tbr - true_tbr
    cac_error = pred_log_cac - true_log_cac
    return {
        "tbr_mae": float(np.abs(tbr_error).mean()),
        "tbr_rmse": float(np.sqrt(np.square(tbr_error).mean())),
        "tbr_r2_endpoint": safe_r2(true_tbr, pred_tbr),
        "tbr_r2_change": safe_r2(true_tbr - base_tbr, pred_tbr - base_tbr),
        "log_cac_mae": float(np.abs(cac_error).mean()),
        "log_cac_rmse": float(np.sqrt(np.square(cac_error).mean())),
        "log_cac_r2_endpoint": safe_r2(true_log_cac, pred_log_cac),
        "log_cac_r2_change": safe_r2(true_log_cac - base_log_cac, pred_log_cac - base_log_cac),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> None:
    model_names = tuple(args.models) if args.models else MODELS
    if not model_names or model_names[0] != "vascmtl" or len(set(model_names)) != len(model_names):
        raise ValueError("Aggregation models must be unique and start with vascmtl")
    root = Path(args.experiment_root)
    lock = verify_lock(root, args.expected_lock_sha256)
    result_dir = root / "results"
    require_empty(result_dir)
    arrays = load_npz(root / "prepared_data" / "A323_training_arrays.npz")
    plan = load_plan(root / "protocol_lock" / "A323_locked_patient_outer_fold_plan.csv", N_A)
    order = list(plan)
    array_index = {str(value): index for index, value in enumerate(arrays["patient_ids"])}
    indices = np.asarray([array_index[value] for value in order])
    truth = pd.DataFrame(
        {
            "patient_id": order,
            "baseline_tbr": arrays["baseline"][indices, 0].astype(float),
            "baseline_cac": arrays["baseline"][indices, 1].astype(float),
            "true_tbr": arrays["targets"][indices, 0].astype(float),
            "true_cac": arrays["targets"][indices, 1].astype(float),
        }
    )
    frames: dict[str, pd.DataFrame] = {}
    oof_hashes = {}
    for model in model_names:
        model_dir = root / "models" / model
        execution = read_json(model_dir / "execution_manifest.json")
        oof_path = model_dir / "out_of_fold_predictions.csv"
        if sha256(oof_path) != execution["oof_sha256"]:
            raise RuntimeError(f"OOF artifact changed: {model}")
        frame = pd.read_csv(oof_path, dtype={"patient_id": str})
        if len(frame) != N_A or set(frame.patient_id) != set(order):
            raise RuntimeError(f"OOF patient mismatch: {model}")
        frame = frame.set_index("patient_id").loc[order].reset_index()
        expected_folds = np.asarray([plan[value] for value in order])
        if not np.array_equal(frame.outer_fold.to_numpy(int), expected_folds):
            raise RuntimeError(f"OOF fold mismatch: {model}")
        for column in ("baseline_tbr", "baseline_cac", "true_tbr", "true_cac"):
            tolerance = 1e-3 if "cac" in column else 2e-7
            difference = np.max(np.abs(frame[column].to_numpy(float) - truth[column].to_numpy(float)))
            if difference > tolerance:
                raise RuntimeError(f"Truth serialization mismatch: {model}/{column}/{difference}")
            frame[column] = truth[column].to_numpy(float)
        frames[model] = frame
        oof_hashes[model] = sha256(oof_path)

    point = {model: calculate_metrics(frame) for model, frame in frames.items()}
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(0, N_A, size=(N_BOOT, N_A))
    samples = {
        model: {metric: np.empty(N_BOOT, float) for metric in METRICS}
        for model in model_names
    }
    for replicate, sample_indices in enumerate(bootstrap_indices):
        for model in model_names:
            values = calculate_metrics(frames[model].iloc[sample_indices])
            for metric in METRICS:
                samples[model][metric][replicate] = values[metric]

    metric_rows = []
    for model in model_names:
        for metric in METRICS:
            low, high = np.nanquantile(samples[model][metric], [0.025, 0.975])
            metric_rows.append(
                {
                    "model": model,
                    "display_name": LABELS[model],
                    "metric": metric,
                    "estimate": point[model][metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_replicates": N_BOOT,
                }
            )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_path = result_dir / "A323_internal_fivefold_metrics_95ci.csv"
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    difference_rows = []
    for baseline in model_names[1:]:
        for metric in METRICS:
            differences = samples["vascmtl"][metric] - samples[baseline][metric]
            low, high = np.nanquantile(differences, [0.025, 0.975])
            difference_rows.append(
                {
                    "comparison": f"VascMTL minus {LABELS[baseline]}",
                    "baseline": baseline,
                    "metric": metric,
                    "difference": point["vascmtl"][metric] - point[baseline][metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "favors_vascmtl_when": "negative" if "r2" not in metric else "positive",
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            )
    differences_frame = pd.DataFrame(difference_rows)
    differences_path = result_dir / "VascMTL_vs_baselines_paired_differences_95ci.csv"
    differences_frame.to_csv(differences_path, index=False, encoding="utf-8-sig")

    np.save(result_dir / "bootstrap_index_matrix.npy", bootstrap_indices, allow_pickle=False)
    fold_rows = []
    for model, frame in frames.items():
        for fold in range(1, OUTER_FOLDS + 1):
            fold_rows.append(
                {
                    "model": model,
                    "display_name": LABELS[model],
                    "outer_fold": fold,
                    "patient_count": int((frame.outer_fold == fold).sum()),
                    **calculate_metrics(frame[frame.outer_fold == fold]),
                }
            )
    pd.DataFrame(fold_rows).to_csv(
        result_dir / "A323_internal_fivefold_fold_metrics.csv", index=False
    )

    main_metrics = (
        "tbr_mae",
        "tbr_rmse",
        "tbr_r2_change",
        "log_cac_mae",
        "log_cac_rmse",
        "log_cac_r2_change",
    )
    table_rows = []
    for model in model_names:
        selected = metrics_frame[metrics_frame.model == model].set_index("metric")
        row = {"模型": LABELS[model]}
        for metric in main_metrics:
            value = selected.loc[metric]
            row[metric] = f"{value.estimate:.4f} [{value.ci_low:.4f}, {value.ci_high:.4f}]"
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table.to_csv(
        result_dir / "A323_internal_fivefold_change_metrics_table_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = "\n".join(
        [
            "# A中心323例内部嵌套五折与基线比较（变化量口径）",
            "",
            "31例用户指定患者已在模型开发前作为独立外部B中心封存；以下仅为A中心323例的完整OOF内部验证结果。",
            "点估计后括号为患者级percentile bootstrap 95%CI。全部按变化量汇报：ΔTBR=随访TBR−基线TBR，Δlog-CAC=log1p(随访TAC)−log1p(基线TAC)；MAE/RMSE越低越好，R²越高越好。",
            "",
            markdown_table(table),
            "",
            "## 方法",
            "",
            f"- 患者级外层{OUTER_FOLDS}折；使用原A354冻结折号删除B31后的A323计划。",
            f"- 每个外层训练池内重新进行患者级内层{INNER_FOLDS}折，用于选参、训练轮数和VascMTL决策层拟合。",
            f"- 95%CI来自完整OOF预测的{N_BOOT}次患者级bootstrap（seed={SEED}）；所有模型和指标共享同一重采样索引。",
            "- Bootstrap不重新训练模型；CI反映固定训练流程下的患者抽样不确定性，不包含训练随机性。",
            "- B31未参与任何预处理、选参、早停、校准或训练；本报告不查看或汇报B31结局。",
            "",
        ]
    )
    report_path = result_dir / "A323_INTERNAL_FIVEFOLD_REPORT_CN.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "status": "A323_internal_nested_fivefold_all_models_complete",
        "A_patient_count": N_A,
        "B_external_patient_count": N_B,
        "A_B_overlap": 0,
        "models": list(model_names),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "outer_fold_checksum": fold_plan_checksum(plan),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "same_bootstrap_indices_all_models_and_metrics": True,
        "B31_outcomes_used": False,
        "protocol_lock_sha256": args.expected_lock_sha256,
        "oof_sha256": oof_hashes,
        "outputs": {
            "metrics": {"path": metrics_path.name, "sha256": sha256(metrics_path)},
            "paired_differences": {"path": differences_path.name, "sha256": sha256(differences_path)},
            "report": {"path": report_path.name, "sha256": sha256(report_path)},
        },
    }
    write_json(result_dir / "result_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument(
        "--source-arrays",
        default=str(FULL_INPUT_ROOT / "prepared_arrays.npz"),
    )
    parser.add_argument(
        "--source-schema",
        default=str(ROOT / "configs" / "schema.json"),
    )
    parser.add_argument(
        "--source-plan",
        default=str(FULL_INPUT_ROOT / "patient_outer_fold_plan.csv"),
    )
    parser.add_argument(
        "--vascmtl-config",
        default=str(ROOT / "configs" / "vascmtl.json"),
    )
    parser.add_argument(
        "--baseline-config-root",
        default=str(ROOT / "configs" / "baselines"),
    )
    parser.add_argument(
        "--input-profile",
        choices=("full_93_27", "legacy_19_16"),
        default="legacy_19_16",
    )
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--b31-ids-file", default=str(ROOT / "private_data" / "B31_ids.txt"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "run":
        if not args.expected_lock_sha256 or not args.model:
            parser.error("run requires --expected-lock-sha256 and --model")
        run_model(args)
    else:
        if not args.expected_lock_sha256:
            parser.error("aggregate requires --expected-lock-sha256")
        aggregate(args)


if __name__ == "__main__":
    main()
