from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.synthetic import SyntheticCohortGenerator, load_npz
from ..training.external_all import (
    evaluate_external_arrays_all_models,
    file_sha256,
    finalize_models_for_external_validation,
)
from .comparison import PRESPECIFIED_MODELS
from .core_ablation import CORE_ABLATION_VARIANTS, run_core_ablation_suite
from .internal_cv import INTERNAL_FIVEFOLD_MODELS, run_internal_fivefold_suite


DEFAULT_DIAGNOSTIC_SEEDS = (2026, 2027, 2028, 2029, 2030)
METRIC_COLUMNS = (
    "tbr_mae",
    "tbr_rmse",
    "tbr_r2",
    "log_cac_mae",
    "log_cac_rmse",
    "log_cac_r2",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_new_or_complete(directory: Path, completion_file: str) -> bool:
    """Return True for a completed phase and reject ambiguous partial output."""
    marker = directory / completion_file
    if marker.is_file():
        return True
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Incomplete output exists at {directory}. Inspect it before restarting; "
            "the diagnostic runner never deletes partial model artifacts automatically."
        )
    return False


def _run_development_seed(job: dict[str, Any]) -> dict[str, Any]:
    """Generate and analyze one development seed in an isolated process."""
    seed = int(job["seed"])
    root = Path(job["root"])
    profile = Path(job["profile"])
    dataset_dir = root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    development_path = dataset_dir / "center_B_development.npz"
    if not development_path.exists():
        SyntheticCohortGenerator.from_json(profile, seed=seed).save(
            development_path, n_patients=int(job["n_development"])
        )
    arrays, schema = load_npz(development_path)
    seed_config = copy.deepcopy(job["training_config"])
    seed_config["seed"] = seed
    seed_config["bootstrap_replicates"] = int(job["bootstrap_replicates"])

    internal_dir = root / "internal"
    if not _require_new_or_complete(
        internal_dir, "internal_fivefold_manifest.json"
    ):
        run_internal_fivefold_suite(
            arrays,
            schema,
            seed_config,
            internal_dir,
            models=INTERNAL_FIVEFOLD_MODELS,
        )

    ablation_dir = root / "ablation"
    if not _require_new_or_complete(
        ablation_dir, "core_ablation_manifest.json"
    ):
        run_core_ablation_suite(
            arrays,
            schema,
            internal_dir,
            ablation_dir,
            variants=CORE_ABLATION_VARIANTS,
            bootstrap_replicates=int(job["bootstrap_replicates"]),
            device_name=str(job["device_name"]),
        )
    return {
        "seed": seed,
        "development_path": str(development_path),
        "internal_complete": True,
        "ablation_complete": True,
    }


def _run_external_seed(job: dict[str, Any]) -> dict[str, Any]:
    """Refit and evaluate one locked seed after the global internal lock."""
    seed = int(job["seed"])
    root = Path(job["root"])
    profile = Path(job["profile"])
    development_path = root / "datasets" / "center_B_development.npz"
    arrays, schema = load_npz(development_path)
    bundle_dir = root / "bundle"
    if not _require_new_or_complete(
        bundle_dir, "locked_bundle_manifest.json"
    ):
        finalize_models_for_external_validation(
            arrays,
            schema,
            root / "internal",
            bundle_dir,
            models=PRESPECIFIED_MODELS,
            device_name=str(job["device_name"]),
        )

    external_path = root / "datasets" / "center_A_external.npz"
    external_seed = seed + 100_000
    if not external_path.exists():
        SyntheticCohortGenerator.from_json(profile, seed=external_seed).save(
            external_path, n_patients=int(job["n_external"])
        )
    external_arrays, external_schema = load_npz(external_path)
    external_dir = root / "external"
    if not _require_new_or_complete(
        external_dir, "external_validation_manifest.json"
    ):
        evaluate_external_arrays_all_models(
            external_arrays,
            external_schema,
            bundle_dir,
            external_dir,
            device_name=str(job["device_name"]),
            n_bootstrap=int(job["bootstrap_replicates"]),
            center_name=f"center_A_synthetic_seed_{external_seed}",
            source_type="standardized_synthetic_npz",
            source_sha256=file_sha256(external_path),
        )
    return {
        "seed": seed,
        "bundle_complete": True,
        "external_complete": True,
    }


def _run_seed_jobs(
    worker: Any, jobs: list[dict[str, Any]], max_workers: int
) -> list[dict[str, Any]]:
    if max_workers == 1:
        return [worker(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=context
    ) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        return [future.result() for future in futures]


def validate_diagnostic_design(
    training_config: dict[str, Any],
    seeds: Iterable[int],
    n_development: int,
    n_external: int,
    bootstrap_replicates: int,
) -> tuple[int, ...]:
    selected_seeds = tuple(int(seed) for seed in seeds)
    if len(selected_seeds) < 5 or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Synthetic diagnostics require at least five unique seeds")
    if not 600 <= int(n_development) <= 700:
        raise ValueError("Center-B synthetic size must be between 600 and 700")
    if int(n_external) < 400:
        raise ValueError("Center-A synthetic size must be at least 400")
    if int(training_config.get("num_folds", 0)) != 5:
        raise ValueError("Synthetic diagnostics require patient-level five-fold CV")
    epochs = int(training_config.get("epochs", 0))
    if not 100 <= epochs <= 200:
        raise ValueError("Deep-model maximum epochs must be between 100 and 200")
    patience = int(training_config.get("patience", 0))
    if patience < 1 or patience >= epochs:
        raise ValueError("Early stopping patience must be positive and below max epochs")
    if int(bootstrap_replicates) < 2000:
        raise ValueError("Formal synthetic diagnostics require at least 2000 bootstraps")
    scope = str(training_config.get("result_scope", ""))
    if not scope.startswith("synthetic_diagnostic"):
        raise ValueError("result_scope must explicitly identify a synthetic diagnostic")
    return selected_seeds


def _aggregate_metric_csv(
    paths_by_seed: dict[int, Path],
    output: Path,
    phase: str,
) -> pd.DataFrame:
    frames = []
    for seed, path in paths_by_seed.items():
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / f"{phase}_metrics_by_seed.csv", index=False)
    keys = ["model"]
    numeric = [
        column
        for column in METRIC_COLUMNS
        if column in combined.columns
    ]
    summary = (
        combined.groupby(keys, sort=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(output / f"{phase}_metrics_mean_std.csv", index=False)
    return summary


def _patient_error_differences(
    full: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, np.ndarray]:
    left = full.sort_values("patient_id").reset_index(drop=True)
    right = candidate.sort_values("patient_id").reset_index(drop=True)
    if not np.array_equal(left["patient_id"], right["patient_id"]):
        raise ValueError("Ablation patient sets differ during cross-seed aggregation")
    if not np.array_equal(left["fold"], right["fold"]):
        raise ValueError("Ablation fold plans differ during cross-seed aggregation")
    full_tbr = np.abs(left["true_tbr"].to_numpy() - left["pred_tbr"].to_numpy())
    candidate_tbr = np.abs(
        right["true_tbr"].to_numpy() - right["pred_tbr"].to_numpy()
    )
    full_cac = np.abs(
        np.log1p(np.maximum(left["true_cac"].to_numpy(), 0))
        - np.log1p(np.maximum(left["pred_cac"].to_numpy(), 0))
    )
    candidate_cac = np.abs(
        np.log1p(np.maximum(right["true_cac"].to_numpy(), 0))
        - np.log1p(np.maximum(right["pred_cac"].to_numpy(), 0))
    )
    return {
        "delta_tbr_mae": candidate_tbr - full_tbr,
        "delta_log_cac_mae": candidate_cac - full_cac,
    }


def _hierarchical_bootstrap(
    differences: list[np.ndarray], n_bootstrap: int, seed: int
) -> dict[str, float]:
    """Resample data-generating seeds, then patients within selected seeds."""
    estimate = float(np.mean([np.mean(values) for values in differences]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        selected_seeds = rng.integers(0, len(differences), len(differences))
        seed_means = []
        for seed_index in selected_seeds:
            values = differences[int(seed_index)]
            selected_patients = rng.integers(0, len(values), len(values))
            seed_means.append(float(np.mean(values[selected_patients])))
        samples[replicate] = float(np.mean(seed_means))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_ablation_worse": float(np.mean(samples > 0)),
    }


def aggregate_core_ablation_across_seeds(
    seed_roots: dict[int, Path],
    output: Path,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    per_variant: dict[str, dict[str, list[np.ndarray]]] = {}
    for seed, root in seed_roots.items():
        frame = pd.read_csv(root / "ablation_metrics.csv")
        frame.insert(0, "seed", seed)
        metric_frames.append(frame)
        full = pd.read_csv(root / "full" / "out_of_fold_predictions.csv")
        for variant in frame["variant"]:
            candidate = pd.read_csv(
                root / str(variant) / "out_of_fold_predictions.csv"
            )
            differences = _patient_error_differences(full, candidate)
            record = per_variant.setdefault(
                str(variant),
                {"delta_tbr_mae": [], "delta_log_cac_mae": []},
            )
            for endpoint, values in differences.items():
                record[endpoint].append(values)
    combined = pd.concat(metric_frames, ignore_index=True)
    combined.to_csv(output / "ablation_metrics_by_seed.csv", index=False)
    delta_columns = (
        "delta_tbr_mae",
        "delta_log_cac_mae",
        "delta_raw_cac_mae",
    )
    summary = (
        combined.groupby("variant", sort=False)[list(delta_columns)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(output / "ablation_delta_mean_std.csv", index=False)

    bootstrap_rows = []
    for variant_index, (variant, endpoints) in enumerate(per_variant.items()):
        for endpoint_index, (endpoint, differences) in enumerate(endpoints.items()):
            result = _hierarchical_bootstrap(
                differences,
                n_bootstrap=bootstrap_replicates,
                seed=91_000 + 100 * variant_index + endpoint_index,
            )
            bootstrap_rows.append(
                {
                    "variant": variant,
                    "endpoint": endpoint,
                    "bootstrap_level": "seed_then_patient",
                    "bootstrap_replicates": bootstrap_replicates,
                    "delta_definition": "ablation_minus_full",
                }
                | result
            )
    hierarchical = pd.DataFrame(bootstrap_rows)
    hierarchical.to_csv(
        output / "ablation_hierarchical_bootstrap.csv", index=False
    )
    return summary, hierarchical


def assess_mechanism_recovery(
    ablation_summary: pd.DataFrame,
    hierarchical: pd.DataFrame,
) -> pd.DataFrame:
    """Check prespecified directions implied by the synthetic generator."""
    checks = (
        (
            "baseline_anchoring_tbr",
            "no_baseline_anchor",
            "delta_tbr_mae",
            "Baseline endpoints directly generate both synthetic endpoints.",
        ),
        (
            "baseline_anchoring_cac",
            "no_baseline_anchor",
            "delta_log_cac_mae",
            "Baseline endpoints directly generate both synthetic endpoints.",
        ),
        (
            "coupling_exists_for_cac",
            "no_coupling",
            "delta_log_cac_mae",
            "Synthetic CAC contains lagged inflammation-to-calcification coupling.",
        ),
        (
            "inflammation_to_calcification",
            "ci_only",
            "delta_log_cac_mae",
            "ci_only removes the generated inflammation-to-calcification path.",
        ),
        (
            "weak_reverse_path_for_tbr",
            "ic_only",
            "delta_tbr_mae",
            "ic_only removes the weaker generated calcification-to-inflammation path.",
        ),
        (
            "asymmetry_over_symmetry_cac",
            "symmetric",
            "delta_log_cac_mae",
            "The generator uses strong forward and weak reverse coupling.",
        ),
        (
            "treatment_conditioning_tbr",
            "no_treatment_conditioning",
            "delta_tbr_mae",
            "Treatment exposures modify the generated inflammation trajectory.",
        ),
        (
            "treatment_conditioning_cac",
            "no_treatment_conditioning",
            "delta_log_cac_mae",
            "Treatment exposures modify the generated calcification trajectory.",
        ),
    )
    rows = []
    for mechanism, variant, endpoint, rationale in checks:
        seed_column = f"{endpoint}_mean"
        seed_row = ablation_summary.loc[ablation_summary["variant"] == variant]
        boot_row = hierarchical.loc[
            (hierarchical["variant"] == variant)
            & (hierarchical["endpoint"] == endpoint)
        ]
        if len(seed_row) != 1 or len(boot_row) != 1:
            raise ValueError(f"Missing ablation result for mechanism check: {mechanism}")
        mean_delta = float(seed_row.iloc[0][seed_column])
        ci_low = float(boot_row.iloc[0]["ci_low"])
        ci_high = float(boot_row.iloc[0]["ci_high"])
        rows.append(
            {
                "mechanism": mechanism,
                "ablated_variant": variant,
                "endpoint": endpoint,
                "expected_direction": "positive_ablation_minus_full",
                "mean_delta_across_seeds": mean_delta,
                "hierarchical_ci_low": ci_low,
                "hierarchical_ci_high": ci_high,
                "direction_recovered": mean_delta > 0,
                "statistically_clear": ci_low > 0,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def aggregate_synthetic_diagnostic(
    output_root: str | Path,
    seeds: Iterable[int],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    output = Path(output_root)
    selected_seeds = tuple(int(seed) for seed in seeds)
    seed_dirs = {seed: output / "runs" / f"seed_{seed}" for seed in selected_seeds}
    summary_dir = output / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    internal = _aggregate_metric_csv(
        {
            seed: root / "internal" / "internal_fivefold_metrics.csv"
            for seed, root in seed_dirs.items()
        },
        summary_dir,
        "internal",
    )
    external = _aggregate_metric_csv(
        {
            seed: root / "external" / "external_metrics.csv"
            for seed, root in seed_dirs.items()
        },
        summary_dir,
        "external",
    )
    ablation, hierarchical = aggregate_core_ablation_across_seeds(
        {seed: root / "ablation" for seed, root in seed_dirs.items()},
        summary_dir,
        bootstrap_replicates,
    )
    recovery = assess_mechanism_recovery(ablation, hierarchical)
    recovery.to_csv(summary_dir / "mechanism_recovery.csv", index=False)
    result = {
        "status": "synthetic_diagnostic_complete_non_clinical",
        "formal_clinical_conclusions_generated": False,
        "seeds": list(selected_seeds),
        "seed_count": len(selected_seeds),
        "bootstrap_replicates_per_seed": bootstrap_replicates,
        "hierarchical_bootstrap_replicates": bootstrap_replicates,
        "reported_across_seed_statistics": ["mean", "sample_standard_deviation"],
        "internal_models": internal["model"].tolist(),
        "external_models": external["model"].tolist(),
        "mechanism_checks": recovery.to_dict(orient="records"),
        "external_data_used_for_hyperparameter_adjustment": False,
    }
    _write_json(summary_dir / "diagnostic_summary.json", result)
    return result


def run_synthetic_diagnostic(
    profile_path: str | Path,
    training_config: dict[str, Any],
    output_root: str | Path,
    seeds: Iterable[int] = DEFAULT_DIAGNOSTIC_SEEDS,
    n_development: int = 650,
    n_external: int = 420,
    bootstrap_replicates: int = 2000,
    device_name: str = "auto",
    parallel_seed_workers: int = 1,
) -> dict[str, Any]:
    """Run the prespecified multi-seed synthetic diagnostic without external tuning."""
    selected_seeds = validate_diagnostic_design(
        training_config,
        seeds,
        n_development,
        n_external,
        bootstrap_replicates,
    )
    parallel_seed_workers = int(parallel_seed_workers)
    if not 1 <= parallel_seed_workers <= len(selected_seeds):
        raise ValueError("parallel_seed_workers must be between 1 and seed count")
    profile = Path(profile_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "locked_before_external_generation",
        "purpose": "non_clinical_synthetic_diagnostic",
        "development_center": "center_B_synthetic",
        "external_center": "center_A_synthetic",
        "development_patient_count": int(n_development),
        "external_patient_count": int(n_external),
        "seeds": list(selected_seeds),
        "external_seed_offset": 100_000,
        "num_folds": 5,
        "maximum_epochs": int(training_config["epochs"]),
        "early_stopping_patience": int(training_config["patience"]),
        "bootstrap_replicates": int(bootstrap_replicates),
        "parallel_seed_workers": parallel_seed_workers,
        "training_config_sha256": _canonical_hash(training_config),
        "aggregate_profile_sha256": file_sha256(profile),
        "hyperparameter_source": "prespecified_and_internal_development_only",
        "external_data_may_not_be_generated_until_internal_lock": True,
        "external_data_used_for_fit_selection_or_epoch_choice": False,
        "models": list(PRESPECIFIED_MODELS),
        "core_ablation_variants": list(CORE_ABLATION_VARIANTS),
    }
    protocol_path = output / "locked_protocol.json"
    if protocol_path.exists():
        if _read_json(protocol_path) != protocol:
            raise RuntimeError("Existing locked synthetic protocol differs from this run")
    else:
        _write_json(protocol_path, protocol)

    seed_dirs = {seed: output / "runs" / f"seed_{seed}" for seed in selected_seeds}
    development_paths = {
        seed: root / "datasets" / "center_B_development.npz"
        for seed, root in seed_dirs.items()
    }

    # Phase 1: every seed finishes internal comparison and ablation. Seeds may
    # run in isolated processes, but no external dataset exists in this phase.
    development_jobs = [
        {
            "seed": seed,
            "root": str(root),
            "profile": str(profile),
            "training_config": training_config,
            "n_development": n_development,
            "bootstrap_replicates": bootstrap_replicates,
            "device_name": device_name,
        }
        for seed, root in seed_dirs.items()
    ]
    _run_seed_jobs(
        _run_development_seed, development_jobs, parallel_seed_workers
    )

    # Phase 2: freeze all internal choices before any center-A dataset is created.
    internal_lock = {
        "status": "all_internal_runs_locked_before_external_generation",
        "seeds": list(selected_seeds),
        "records": {
            str(seed): {
                "development_data_sha256": file_sha256(development_paths[seed]),
                "internal_manifest_sha256": file_sha256(
                    seed_dirs[seed]
                    / "internal"
                    / "internal_fivefold_manifest.json"
                ),
                "ablation_manifest_sha256": file_sha256(
                    seed_dirs[seed]
                    / "ablation"
                    / "core_ablation_manifest.json"
                ),
            }
            for seed in selected_seeds
        },
        "external_data_exists_at_lock_time": False,
        "external_data_used_for_hyperparameter_adjustment": False,
    }
    internal_lock_path = output / "internal_lock_before_external.json"
    if internal_lock_path.exists():
        existing_lock = _read_json(internal_lock_path)
        if existing_lock != internal_lock:
            raise RuntimeError("Internal lock changed after external phase began")
    else:
        external_exists_before_lock = any(
            (seed_dirs[seed] / "datasets" / "center_A_external.npz").exists()
            for seed in selected_seeds
        )
        if external_exists_before_lock:
            raise RuntimeError(
                "External synthetic data already exists before the internal lock"
            )
        _write_json(internal_lock_path, internal_lock)

    # Phase 3: all internal locks exist before any process generates center A.
    external_jobs = [
        {
            "seed": seed,
            "root": str(root),
            "profile": str(profile),
            "n_external": n_external,
            "bootstrap_replicates": bootstrap_replicates,
            "device_name": device_name,
        }
        for seed, root in seed_dirs.items()
    ]
    _run_seed_jobs(_run_external_seed, external_jobs, parallel_seed_workers)

    return aggregate_synthetic_diagnostic(
        output, selected_seeds, bootstrap_replicates
    )
