from __future__ import annotations

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
from ..experiments.prediction import build_torch_model
from .metrics import bootstrap_change_metrics, change_space_metrics, regression_metrics
from .trainer import _git_revision, _runtime_versions, patient_id_hash, resolve_device
from .v27_nested import _predict_base_with_features, _residual_to_endpoint
from .v32_nested import _componentwise_epoch_median
from .v37_external import (
    COMPARISON_MODELS as V37_COMPARISON_MODELS,
    LOCKED_DEVELOPMENT_FOLLOWUP_MEDIAN_MONTHS,
    LOCKED_DEVELOPMENT_PATIENT_HASH,
    PRIMARY_METRICS,
    _canonical_hash,
    _combine_bundles,
    _load_reused_prediction_frame,
    _paired_bootstrap,
    _prediction_frame,
    _prepare_output,
    _read_json,
    _sha256,
    _validate_arrays,
    _fold_epoch_values,
    load_npz,
)
from .v40_final_nested import (
    V40_FINAL_MODELS,
    _config_for_model,
    _decision_modes,
    _seed_everything_v40_final,
)
from .v40_final_selector import V40FinalSelector
from .v41_trainer import fit_v41_fixed_epochs


DERIVED_DECISION_MODELS = {
    "lac_v40_final_no_calibration",
    "lac_v40_final_no_tail",
    "lac_v40_final_central_only",
}
V40_FROZEN_CANDIDATE = "lac_v40_final_no_tail"
V40_COMPARISON_MODELS = tuple(V37_COMPARISON_MODELS) + (V40_FROZEN_CANDIDATE,)


def _source_for_v40(root: Path, model_name: str) -> Path:
    selected_policy = root / "selected_policy" / model_name
    if (selected_policy / "summary.json").is_file():
        return selected_policy
    candidates = (
        root / model_name,
        root / "full" / model_name,
        root / "ablations" / model_name,
    )
    existing = [path for path in candidates if (path / "summary.json").is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected one internal source for {model_name}; found {existing}"
        )
    return existing[0]


def _neural_source_for_v40(root: Path, model_name: str) -> Path:
    if model_name in DERIVED_DECISION_MODELS:
        return _source_for_v40(root, "lac_v40_final_full")
    return _source_for_v40(root, model_name)


def _final_refit_seed(config: dict[str, Any]) -> int:
    return int(config.get("seed", 2026)) + 900_000


def _fit_locked_v40_decision(
    development: dict[str, np.ndarray],
    source: Path,
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[V40FinalSelector, dict[str, Any]]:
    plan_frame = pd.read_csv(source / "patient_outer_fold_plan.csv")
    plan = {
        str(row.patient_id): int(row.test_fold)
        for row in plan_frame.itertuples(index=False)
    }
    raw_index = {
        str(patient_id): index
        for index, patient_id in enumerate(development["patient_ids"])
    }
    bundles: list[dict[str, Any]] = []
    for fold in range(1, 6):
        identifiers = [
            patient_id for patient_id, assigned in plan.items() if assigned == fold
        ]
        indices = np.asarray([raw_index[value] for value in identifiers], dtype=int)
        raw = subset_arrays(development, indices)
        fold_dir = source / f"fold_{fold}"
        checkpoint = torch.load(
            fold_dir / "model.pt", map_location="cpu", weights_only=False
        )
        schema = FeatureSchema.from_dict(checkpoint["schema"])
        model = build_torch_model(
            model_name, _config_for_model(schema, config, model_name)
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
        bundle["meta_fold"] = np.full(len(identifiers), fold, dtype=int)
        bundles.append(bundle)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = _combine_bundles(bundles)
    if len(combined["patient_ids"]) != len(development["patient_ids"]):
        raise RuntimeError("V4.0-final OOF decision reconstruction is incomplete")
    options = dict(config.get("decision", {}))
    selector = V40FinalSelector(
        candidate_weights=tuple(options.get("candidate_weights", (0, .1, .2, .35, .5, .75, 1))),
        ridge_alphas=tuple(options.get("ridge_alphas", (1.0,))),
        degradation_tolerance=float(options.get("degradation_tolerance", .005)),
        required_consistent_folds=int(options.get("required_consistent_folds", 3)),
        seed=int(config.get("seed", 2026)),
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


def finalize_v40_external_bundle(
    development_npz: str | Path,
    schema_path: str | Path,
    v40_root: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Freeze V4.0-final on all 443 patients without reading external data."""
    output = _prepare_output(output_dir)
    schema = FeatureSchema.from_dict(_read_json(schema_path))
    development = load_npz(development_npz)
    _validate_arrays(development, schema, "development_center_443")
    if len(development["patient_ids"]) != 443:
        raise ValueError("V4.0-final refit requires the locked 443-patient cohort")
    if patient_id_hash(development["patient_ids"]) != LOCKED_DEVELOPMENT_PATIENT_HASH:
        raise ValueError("Development patient identity differs from the locked cohort")
    root = Path(v40_root)
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
    full_artifact: Path | None = None
    full_decision: Path | None = None
    for model_name in V40_FINAL_MODELS:
        source = _source_for_v40(root, model_name)
        neural_source = _neural_source_for_v40(root, model_name)
        summary = _read_json(source / "summary.json")
        neural_summary = _read_json(neural_source / "summary.json")
        config = _read_json(neural_source / "training_config.json")
        if int(summary["patient_count"]) != 443:
            raise ValueError(f"{model_name} does not contain 443 internal OOF patients")
        checksum = summary["cv_protocol"]["outer_fold_plan_checksum"]
        if checksum != "7c18f9bd52a884639839a5a2f8e8664f1db8f016bcf46ab632ebf1db9db8c655":
            raise ValueError(f"{model_name} does not use the locked outer split")
        model_dir = output / model_name
        model_dir.mkdir()
        started = time.perf_counter()
        epoch_values = _fold_epoch_values(neural_summary)
        locked_epochs = _componentwise_epoch_median(epoch_values)
        seed = _final_refit_seed(config)

        if model_name in DERIVED_DECISION_MODELS:
            if full_artifact is None or full_decision is None:
                raise RuntimeError("Full V4.0-final must be frozen before derived ablations")
            artifact = model_dir / "model.pt"
            decision_path = model_dir / "decision_layer.pkl"
            shutil.copy2(full_artifact, artifact)
            shutil.copy2(full_decision, decision_path)
            decision_audit = records["lac_v40_final_full"]["decision_audit"]
            history = records["lac_v40_final_full"]["fit_history"]
            exact_shared_state = True
        else:
            _seed_everything_v40_final(seed)
            selector, decision_audit = _fit_locked_v40_decision(
                development, neural_source, model_name, config, device
            )
            decision_path = model_dir / "decision_layer.pkl"
            with decision_path.open("wb") as handle:
                pickle.dump(selector, handle)
            # Reset after read-only OOF reconstruction so every architecture uses
            # the same controlled full-refit initialization seed.
            _seed_everything_v40_final(seed)
            model_config = _config_for_model(schema, config, model_name)
            model = build_torch_model(model_name, model_config).to(device)
            model, history = fit_v41_fixed_epochs(
                model, transformed, config, device, locked_epochs, seed
            )
            artifact = model_dir / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_config": model_config.to_dict(),
                    "schema": schema.to_dict(),
                    "source_internal_git_revision": neural_summary["git_revision"],
                },
                artifact,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            exact_shared_state = False
            if model_name == "lac_v40_final_full":
                full_artifact = artifact
                full_decision = decision_path

        record = {
            "model_name": model_name,
            "source_internal_summary": str(source / "summary.json"),
            "source_internal_summary_sha256": _sha256(source / "summary.json"),
            "source_internal_git_revision": summary["git_revision"],
            "source_outer_fold_checksum": checksum,
            "source_internal_oof_change_metrics": summary["pooled_change_metrics"],
            "locked_stage_epoch_codes": epoch_values,
            "locked_final_stage_epoch_code": locked_epochs,
            "decision_modes": summary["cv_protocol"].get(
                "decision_mode", _decision_modes(model_name)
            ),
            "decision_audit": decision_audit,
            "artifact_file": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "decision_sha256": _sha256(decision_path),
            "seed": seed,
            "fit_seconds": time.perf_counter() - started,
            "fit_history": history,
            "exact_full_neural_state_reused": exact_shared_state,
            "external_data_seen": False,
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records[model_name] = record

    result = {
        "status": "locked_v40_final_full_443_refit_bundle",
        "protocol": "locked_internal_choices_full_443_refit_one_pass_external",
        "development_patient_count": 443,
        "development_patient_hash": patient_id_hash(development["patient_ids"]),
        "v40_models": list(V40_FINAL_MODELS),
        "frozen_candidate": V40_FROZEN_CANDIDATE,
        "derived_decision_models": sorted(DERIVED_DECISION_MODELS),
        "preprocessor_fitted_on_all_development_patients": True,
        "external_data_used_for_fit_selection_or_epoch_choice": False,
        "schema_sha256": _canonical_hash(schema.to_dict()),
        "preprocessor_sha256": _sha256(output / "preprocessor.json"),
        "common_refit_seed": _final_refit_seed(
            _read_json(_source_for_v40(root, "lac_v40_final_full") / "training_config.json")
        ),
        "bundle_git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "device": str(device),
        "model_records": records,
    }
    (output / "locked_bundle_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _load_v40_model(model_dir: Path, device: torch.device):
    checkpoint = torch.load(
        model_dir / "model.pt", map_location="cpu", weights_only=False
    )
    schema = FeatureSchema.from_dict(checkpoint["schema"])
    model_name = checkpoint["model_name"]
    config = _config_for_model(
        schema, {"model_v40_final": checkpoint["model_config"]}, model_name
    )
    model = build_torch_model(model_name, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, schema, model_name


def _gate_records(model_name: str, diagnostics: dict[str, np.ndarray]):
    rows = []
    for gate_name, values in diagnostics.items():
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "model": model_name,
                "quantity": gate_name,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "q1": float(np.quantile(values, .25)),
                "median": float(np.median(values)),
                "q3": float(np.quantile(values, .75)),
                "max": float(np.max(values)),
                "near_zero_fraction": float(np.mean(np.abs(values) < .01)),
                "near_one_fraction": float(np.mean(np.abs(values - 1) < .01)),
            }
        )
    return rows


def evaluate_v40_external_bundle(
    bundle_dir: str | Path,
    external_npz: str | Path,
    external_schema_path: str | Path,
    external_manifest_path: str | Path,
    reused_v37_results: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """One-pass frozen V4.0 external inference; all comparators are reused."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    output = _prepare_output(output_dir)
    bundle = Path(bundle_dir)
    reused = Path(reused_v37_results)
    lock = _read_json(bundle / "locked_bundle_manifest.json")
    if lock["external_data_used_for_fit_selection_or_epoch_choice"]:
        raise RuntimeError("Frozen V4.0 bundle reports external leakage")
    schema = FeatureSchema.from_dict(_read_json(external_schema_path))
    if _canonical_hash(schema.to_dict()) != lock["schema_sha256"]:
        raise ValueError("External schema differs from frozen development schema")
    external = load_npz(external_npz)
    _validate_arrays(external, schema, "external_shandong")
    source_manifest = _read_json(external_manifest_path)
    if len(external["patient_ids"]) != int(source_manifest["external_patient_count"]):
        raise ValueError("External patient count differs from audited manifest")
    missing_followup = ~np.isfinite(external["followup_months"].astype(float))
    if np.any(missing_followup):
        if lock["development_patient_hash"] != LOCKED_DEVELOPMENT_PATIENT_HASH:
            raise RuntimeError("Locked follow-up rule does not apply to this bundle")
        external = dict(external)
        external["followup_months"] = external["followup_months"].astype(float, copy=True)
        external["followup_months"][missing_followup] = LOCKED_DEVELOPMENT_FOLLOWUP_MEDIAN_MONTHS
    followup_imputation = {
        "missing_external_values": int(missing_followup.sum()),
        "fill_value_months": LOCKED_DEVELOPMENT_FOLLOWUP_MEDIAN_MONTHS,
        "source": "median_of_locked_443_development_cohort",
        "external_values_used_to_fit_imputation": False,
    }
    preprocessor_path = bundle / "preprocessor.json"
    if _sha256(preprocessor_path) != lock["preprocessor_sha256"]:
        raise RuntimeError("Frozen V4.0 preprocessor checksum mismatch")
    transformed = FoldPreprocessor.load(preprocessor_path).transform(external)
    device = resolve_device(device_name)
    all_models = tuple(dict.fromkeys(V40_COMPARISON_MODELS + V40_FINAL_MODELS))
    predictions: dict[str, np.ndarray] = {}
    metric_rows = []
    ci_rows = []
    gate_rows = []
    reuse_records = {}
    attempt = {
        "status": "v40_final_external_validation_started",
        "models": list(all_models),
        "bundle_manifest_sha256": _sha256(bundle / "locked_bundle_manifest.json"),
        "external_prepared_arrays_sha256": _sha256(external_npz),
        "external_data_manifest_sha256": _sha256(external_manifest_path),
        "fit_preprocessing_selection_or_epoch_choice_on_external_data": False,
    }
    (output / "evaluation_attempt_manifest.json").write_text(
        json.dumps(attempt, indent=2), encoding="utf-8"
    )

    for model_index, model_name in enumerate(all_models):
        diagnostics = None
        if model_name in V37_COMPARISON_MODELS:
            reused_path = reused / model_name / "external_predictions.csv"
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
        else:
            model_dir = bundle / model_name
            model_lock = _read_json(model_dir / "manifest.json")
            artifact = model_dir / model_lock["artifact_file"]
            if _sha256(artifact) != model_lock["artifact_sha256"]:
                raise RuntimeError(f"{model_name} artifact checksum mismatch")
            if _sha256(model_dir / "decision_layer.pkl") != model_lock["decision_sha256"]:
                raise RuntimeError(f"{model_name} decision checksum mismatch")
            model, locked_schema, checkpoint_name = _load_v40_model(model_dir, device)
            if locked_schema != schema:
                raise RuntimeError(f"{model_name} checkpoint schema mismatch")
            base = _predict_base_with_features(
                model, transformed, external, 64, device, checkpoint_name
            )
            with (model_dir / "decision_layer.pkl").open("rb") as handle:
                decision = pickle.load(handle)
            residual, decision_gates = decision.predict(
                base["heads"], base["features"], model_lock["decision_modes"]
            )
            prediction = _residual_to_endpoint(base["baseline"], residual)
            diagnostics = {
                "decision_gate_tbr": decision_gates["tbr"],
                "decision_gate_cac": decision_gates["cac"],
                "adapter_gate_inflammation": base["neural_diagnostics"]["adapter_i"],
                "adapter_gate_calcification": base["neural_diagnostics"]["adapter_c"],
                "coupling_gate": base["neural_diagnostics"]["coupling_gate"],
                "lag_elapsed_years": base["neural_diagnostics"]["lag_elapsed"],
            }
            gate_rows.extend(_gate_records(model_name, diagnostics))
            del model
            frame = _prediction_frame(
                external["patient_ids"],
                external["baseline"],
                external["targets"],
                prediction,
                diagnostics,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        prediction = np.asarray(prediction, dtype=float)
        predictions[model_name] = prediction
        model_output = output / model_name
        model_output.mkdir()
        if model_name in V37_COMPARISON_MODELS:
            shutil.copy2(reused / model_name / "external_predictions.csv", model_output / "external_predictions.csv")
        else:
            frame.to_csv(model_output / "external_predictions.csv", index=False)
        endpoint = regression_metrics(external["targets"], prediction)
        change = change_space_metrics(external["targets"], prediction, external["baseline"])
        metric_rows.append({"model": model_name} | endpoint | change)
        for metric, record in bootstrap_change_metrics(
            external["targets"],
            prediction,
            external["baseline"],
            n_bootstrap=n_bootstrap,
            seed=2026 + model_index,
        ).items():
            ci_rows.append({"model": model_name, "metric": metric} | record)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "all_external_metrics.csv", index=False)
    metrics.loc[metrics["model"].isin(V40_COMPARISON_MODELS)].to_csv(
        output / "comparison_metrics.csv", index=False
    )
    metrics.loc[metrics["model"].isin(V40_FINAL_MODELS)].to_csv(
        output / "v40_ablation_metrics.csv", index=False
    )
    pd.DataFrame(ci_rows).to_csv(output / "metric_bootstrap_95ci.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output / "gate_diagnostics.csv", index=False)
    paired_rows = []
    for model_index, model_name in enumerate(all_models):
        if model_name == V40_FROZEN_CANDIDATE:
            continue
        for row in _paired_bootstrap(
            external["targets"],
            external["baseline"],
            predictions[V40_FROZEN_CANDIDATE],
            predictions[model_name],
            n_bootstrap,
            42026 + model_index,
        ):
            paired_rows.append({"other_model": model_name} | row)
    pd.DataFrame(paired_rows).to_csv(
        output / "paired_bootstrap_v40_candidate_vs_all.csv", index=False
    )
    comparison = metrics.set_index("model").loc[list(V40_COMPARISON_MODELS)]
    ranks = {
        metric: int(
            comparison[metric]
            .rank(ascending=not metric.endswith("_r2"), method="min")
            .loc[V40_FROZEN_CANDIDATE]
        )
        for metric in PRIMARY_METRICS
    }
    ablations = metrics.set_index("model").loc[list(V40_FINAL_MODELS)]
    full = ablations.loc[V40_FROZEN_CANDIDATE]
    pareto_dominators = []
    for model_name, row in ablations.iterrows():
        if model_name == V40_FROZEN_CANDIDATE:
            continue
        no_worse = all(
            row[key] <= full[key] for key in PRIMARY_METRICS if not key.endswith("_r2")
        ) and all(
            row[key] >= full[key] for key in PRIMARY_METRICS if key.endswith("_r2")
        )
        strict = any(
            row[key] < full[key] for key in PRIMARY_METRICS if not key.endswith("_r2")
        ) or any(
            row[key] > full[key] for key in PRIMARY_METRICS if key.endswith("_r2")
        )
        if no_worse and strict:
            pareto_dominators.append(model_name)
    result = {
        "status": "v40_final_external_validation_complete",
        "external_center": "Shandong_Provincial_Cancer_Hospital",
        "external_patient_count": len(external["patient_ids"]),
        "models": list(all_models),
        "comparison_models": list(V40_COMPARISON_MODELS),
        "v40_models": list(V40_FINAL_MODELS),
        "bootstrap_replicates": n_bootstrap,
        "frozen_candidate": V40_FROZEN_CANDIDATE,
        "v40_candidate_comparison_ranks": ranks,
        "v40_candidate_ablation_pareto_dominators": pareto_dominators,
        "followup_imputation": followup_imputation,
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


def verify_v40_external(bundle_dir: str | Path, result_dir: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    result = Path(result_dir)
    lock = _read_json(bundle / "locked_bundle_manifest.json")
    external = _read_json(result / "external_validation_manifest.json")
    if lock["external_data_used_for_fit_selection_or_epoch_choice"]:
        raise RuntimeError("Bundle reports external leakage")
    if external["fit_preprocessing_selection_or_epoch_choice_on_external_data"]:
        raise RuntimeError("External result reports leakage")
    n = int(external["external_patient_count"])
    for model_name in V40_FINAL_MODELS:
        model_dir = bundle / model_name
        manifest = _read_json(model_dir / "manifest.json")
        artifact = model_dir / manifest["artifact_file"]
        if _sha256(artifact) != manifest["artifact_sha256"]:
            raise RuntimeError(f"{model_name} artifact checksum mismatch")
    for model_name in external["models"]:
        frame = pd.read_csv(result / model_name / "external_predictions.csv")
        if len(frame) != n or frame["patient_id"].nunique() != n:
            raise RuntimeError(f"{model_name} prediction rows are invalid")
    metrics = pd.read_csv(result / "all_external_metrics.csv")
    if set(metrics["model"]) != set(external["models"]):
        raise RuntimeError("External metric model set differs from manifest")
    return {
        "status": "passed",
        "bundle_manifest_sha256": _sha256(bundle / "locked_bundle_manifest.json"),
        "external_manifest_sha256": _sha256(result / "external_validation_manifest.json"),
        "models_verified": len(external["models"]),
        "external_patient_count": n,
    }
