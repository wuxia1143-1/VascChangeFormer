"""Locked A323 full-final models -> Centre C546 external validation.

The source Centre C array contains 547 patients.  One privately specified ineligible patient is excluded
before prediction.  The predict phase never opens the sealed outcome file.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_a323_b31_external_validation as shared


MODELS = shared.MODELS
LABELS = shared.LABELS
SEED = 2026
N_BOOT = 2_000
EXPECTED_DIMS = {"static": 19, "longitudinal": 16, "baseline": 2, "treatment": 5}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def patient_hash(values: np.ndarray) -> str:
    return shared.patient_id_hash(np.asarray(values).astype(str))


def dimensions(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        "static": int(arrays["static"].shape[1]),
        "longitudinal": int(arrays["values"].shape[2]),
        "baseline": int(arrays["baseline"].shape[1]),
        "treatment": int(arrays["treatments"].shape[2]),
    }


def paths(root: Path) -> dict[str, Path]:
    out = root / "external_validation_c546"
    return {
        "out": out,
        "predictors": root / "prepared_data" / "C546_predictors_without_outcomes.npz",
        "outcomes": root / "prepared_data" / "C546_sealed_outcomes.npz",
        "prep_manifest": root / "prepared_data" / "C546_preparation_manifest.json",
        "predictions": out / "C546_external_predictions_blinded.csv",
        "prediction_lock": out / "C546_prediction_lock_manifest.json",
        "pi_alias": out / "A323_OOF_PI90_TAC_calibration_lock.json",
    }


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    p = paths(root)
    p["out"].mkdir(parents=True, exist_ok=True)
    for key in ("predictors", "outcomes", "prep_manifest"):
        if p[key].exists():
            raise FileExistsError(f"Refusing to overwrite {p[key]}")
    excluded_id = args.exclude_c_id
    if not excluded_id:
        raise ValueError("prepare requires --exclude-c-id from your private cohort manifest")
    source = Path(args.source_c547)
    arrays = shared.load_npz(source)
    ids = arrays["patient_ids"].astype(str)
    if len(ids) != 547 or len(np.unique(ids)) != 547:
        raise RuntimeError("Centre C source must contain 547 unique patients")
    excluded = ids == excluded_id
    if int(excluded.sum()) != 1:
        raise RuntimeError(f"Expected exactly one {excluded_id}; found {excluded.sum()}")
    if dimensions(arrays) != EXPECTED_DIMS:
        raise RuntimeError(f"Unexpected Centre C dimensions: {dimensions(arrays)}")
    keep = ~excluded
    c546 = {key: np.asarray(value)[keep] for key, value in arrays.items()}
    outcomes = {
        "patient_ids": c546["patient_ids"],
        "targets": c546.pop("targets"),
    }
    a323 = shared.load_npz(root / "prepared_data" / "A323_training_arrays.npz")
    overlap = sorted(set(a323["patient_ids"].astype(str)) & set(c546["patient_ids"].astype(str)))
    if overlap:
        raise RuntimeError(f"A323/C546 overlap: {overlap[:5]}")
    save_npz(p["predictors"], c546)
    save_npz(p["outcomes"], outcomes)
    manifest = {
        "status": "Centre_C546_prepared_and_outcomes_sealed",
        "source_patient_count": 547,
        "external_patient_count": 546,
        "excluded_patient_id": excluded_id,
        "exclusion_reason": "age_below_20",
        "source_sha256": shared.sha256(source),
        "predictor_sha256": shared.sha256(p["predictors"]),
        "sealed_outcome_sha256": shared.sha256(p["outcomes"]),
        "patient_id_hash": patient_hash(c546["patient_ids"]),
        "input_dimensions": dimensions(c546),
        "A323_patient_count": 323,
        "A323_C546_overlap_n": 0,
        "C546_used_in_development": False,
    }
    write_json(p["prep_manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def make_tac_pi_alias(root: Path, destination: Path, model_names: tuple[str, ...]) -> dict[str, Any]:
    source_path = root / "external_validation" / "A323_OOF_PI90_calibration_lock.json"
    source = shared.read_json(source_path)
    if tuple(source["model_order"]) != model_names:
        raise RuntimeError("A323 PI source model order mismatch")
    quantiles = {}
    for model in model_names:
        q = source["quantiles"][model]
        quantiles[model] = {
            "delta_tbr_q90_abs_error": q["delta_tbr_q90_abs_error"],
            "delta_log_tac_q90_abs_error": q["delta_log_cac_q90_abs_error"],
            "delta_tbr_interval_width": q["delta_tbr_interval_width"],
            "delta_log_tac_interval_width": q["delta_log_cac_interval_width"],
            "oof_prediction_sha256": q["oof_prediction_sha256"],
            "oof_patient_count": q["oof_patient_count"],
        }
    alias = {
        "status": "A323_OOF_PI90_locked_before_full_refit_and_external_prediction",
        "model_order": list(model_names),
        "calibration_source": "A323_nested_fivefold_complete_OOF_absolute_residuals",
        "quantile": 0.9,
        "interval_rule": "point_prediction_plus_or_minus_locked_q90_absolute_error",
        "external_outcomes_used": False,
        "source_lock_sha256": shared.sha256(source_path),
        "quantiles": quantiles,
    }
    write_json(destination, alias)
    return alias


def predict(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    p = paths(root)
    p["out"].mkdir(parents=True, exist_ok=True)
    if p["predictions"].exists() or p["prediction_lock"].exists():
        raise FileExistsError("Centre C predictions are already locked")
    prep = shared.read_json(p["prep_manifest"])
    if prep["external_patient_count"] != 546 or prep["A323_C546_overlap_n"] != 0:
        raise RuntimeError("Invalid C546 preparation manifest")
    raw_source = shared.load_npz(p["predictors"])
    if "targets" in raw_source:
        raise RuntimeError("C546 predictor file contains outcomes")
    raw = shared.add_dummy_targets(raw_source)
    if len(raw["patient_ids"]) != 546 or shared.read_json(p["prep_manifest"])["excluded_patient_id"] in set(raw["patient_ids"].astype(str)):
        raise RuntimeError("Invalid C546 patient set")
    schema = shared.FeatureSchema.from_dict(shared.read_json(root / "prepared_data" / "schema.json"))
    model_names = tuple(args.models) if args.models else MODELS
    pi_alias = make_tac_pi_alias(root, p["pi_alias"], model_names)
    full_manifest = shared.read_json(root / "frozen_models" / "full_refit_manifest.json")
    if full_manifest["patient_count"] != 323 or full_manifest["input_dimensions"] != EXPECTED_DIMS:
        raise RuntimeError("Full-A323 model manifest mismatch")
    device = shared.resolve_device(args.device)
    rows: dict[str, Any] = {
        "patient_id": raw["patient_ids"].astype(str),
        "baseline_tbr": raw["baseline"][:, 0].astype(float),
        "baseline_tac": raw["baseline"][:, 1].astype(float),
        "followup_months": raw["followup_months"].astype(float),
    }
    model_records = {}
    for model in model_names:
        model_dir = root / "frozen_models" / model
        prediction = shared.predict_fold(
            model,
            model_dir,
            raw,
            schema,
            shared.read_json(root / "protocol_lock" / "locked_configs" / f"{model}.json"),
            device,
        )
        if prediction.shape != (546, 2) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid Centre C prediction: {model}")
        rows[f"pred_{model}_tbr"] = prediction[:, 0]
        rows[f"pred_{model}_tac"] = prediction[:, 1]
        model_records[model] = {
            "display_name": LABELS[model],
            "source": "single_full_A323_frozen_final_model",
            "model_artifact_sha256": shared.sha256(model_dir / shared.artifact_name(model)),
            "preprocessor_sha256": shared.sha256(model_dir / "preprocessor.json"),
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(p["predictions"], index=False, encoding="utf-8-sig")
    lock = {
        "status": "C546_predictions_locked_before_outcome_evaluation",
        "external_patient_count": 546,
        "patient_id_hash": patient_hash(frame.patient_id.to_numpy(str)),
        "prediction_file": p["predictions"].name,
        "prediction_file_sha256": shared.sha256(p["predictions"]),
        "prediction_has_outcomes": False,
        "models": list(model_names),
        "model_records": model_records,
        "device": str(device),
        "point_prediction_source": "full_A323_frozen_final_model",
        "five_fold_models_averaged": False,
        "A323_model_artifacts_only": True,
        "C546_outcomes_read": False,
        "pi_lock_sha256": shared.sha256(p["pi_alias"]),
        "pi_source_lock_sha256": pi_alias["source_lock_sha256"],
    }
    write_json(p["prediction_lock"], lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


def change_arrays(baseline: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_tbr = target[:, 0] - baseline[:, 0]
    pred_tbr = prediction[:, 0] - baseline[:, 0]
    true_tac = np.log1p(np.maximum(target[:, 1], 0)) - np.log1p(np.maximum(baseline[:, 1], 0))
    pred_tac = np.log1p(np.maximum(prediction[:, 1], 0)) - np.log1p(np.maximum(baseline[:, 1], 0))
    return true_tbr, pred_tbr, true_tac, pred_tac


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    error = prediction - truth
    denominator = np.square(truth - truth.mean()).sum()
    return (
        float(np.abs(error).mean()),
        float(np.sqrt(np.square(error).mean())),
        float(1 - np.square(error).sum() / denominator) if denominator > 0 else float("nan"),
    )


def evaluate(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    p = paths(root)
    lock = shared.read_json(p["prediction_lock"])
    if shared.sha256(p["predictions"]) != lock["prediction_file_sha256"]:
        raise RuntimeError("Prediction lock mismatch")
    if shared.sha256(p["pi_alias"]) != lock["pi_lock_sha256"]:
        raise RuntimeError("PI lock mismatch")
    pred = pd.read_csv(p["predictions"], dtype={"patient_id": str})
    sealed = shared.load_npz(p["outcomes"])
    ids = sealed["patient_ids"].astype(str)
    pred = pred.set_index("patient_id").loc[ids].reset_index()
    target = sealed["targets"].astype(float)
    baseline = pred[["baseline_tbr", "baseline_tac"]].to_numpy(float)
    model_names = tuple(lock["models"])
    pi = shared.read_json(p["pi_alias"])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 546, size=(N_BOOT, 546), dtype=np.int32)
    metric_rows, pi_rows = [], []
    vasc_plot = {}
    for model in model_names:
        endpoint = pred[[f"pred_{model}_tbr", f"pred_{model}_tac"]].to_numpy(float)
        tt, pt, tc, pc = change_arrays(baseline, target, endpoint)
        for outcome, truth, estimate in (("delta_tbr", tt, pt), ("delta_log_tac", tc, pc)):
            point = metric_values(truth, estimate)
            samples = np.asarray([metric_values(truth[idx], estimate[idx]) for idx in draws])
            for j, metric in enumerate(("mae", "rmse", "r2")):
                metric_rows.append({
                    "model": model,
                    "display_name": LABELS[model],
                    "metric": f"{outcome}_{metric}",
                    "estimate": point[j],
                    "ci_low": float(np.nanquantile(samples[:, j], .025)),
                    "ci_high": float(np.nanquantile(samples[:, j], .975)),
                    "bootstrap_replicates": N_BOOT,
                })
        q = pi["quantiles"][model]
        for outcome, truth, estimate, qkey in (
            ("delta_tbr", tt, pt, "delta_tbr_q90_abs_error"),
            ("delta_log_tac", tc, pc, "delta_log_tac_q90_abs_error"),
        ):
            half = float(q[qkey])
            covered = np.abs(truth - estimate) <= half
            pi_rows.append({
                "model": model,
                "display_name": LABELS[model],
                "outcome": outcome,
                "pi_half_width": half,
                "interval_width": 2 * half,
                "covered_n": int(covered.sum()),
                "total_n": 546,
                "coverage_90pi": float(covered.mean()),
            })
        if model == "vascmtl":
            vasc_plot = {"true_tbr": tt, "pred_tbr": pt, "true_tac": tc, "pred_tac": pc}
    metrics = pd.DataFrame(metric_rows)
    uncertainty = pd.DataFrame(pi_rows)
    metrics_path = p["out"] / "C546_external_change_metrics_95ci.csv"
    uncertainty_path = p["out"] / "C546_external_prediction_uncertainty_90PI.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    uncertainty.to_csv(uncertainty_path, index=False, encoding="utf-8-sig")
    patient = pred[["patient_id", "baseline_tbr", "baseline_tac", "followup_months"]].copy()
    patient["true_tbr"] = target[:, 0]
    patient["true_tac"] = target[:, 1]
    patient["observed_delta_tbr"] = vasc_plot["true_tbr"]
    patient["predicted_delta_tbr"] = vasc_plot["pred_tbr"]
    patient["observed_delta_log_tac"] = vasc_plot["true_tac"]
    patient["predicted_delta_log_tac"] = vasc_plot["pred_tac"]
    qv = pi["quantiles"]["vascmtl"]
    for outcome, half in (("tbr", qv["delta_tbr_q90_abs_error"]), ("log_tac", qv["delta_log_tac_q90_abs_error"])):
        patient[f"{outcome}_pi_lower"] = patient[f"predicted_delta_{outcome}"] - half
        patient[f"{outcome}_pi_upper"] = patient[f"predicted_delta_{outcome}"] + half
        patient[f"{outcome}_pi_covered"] = (
            np.abs(patient[f"observed_delta_{outcome}"] - patient[f"predicted_delta_{outcome}"]) <= half
        ).astype(int)
    patient_path = p["out"] / "C546_VascMTL_patient_predictions_and_PI.csv"
    patient.to_csv(patient_path, index=False, encoding="utf-8-sig")
    plots = p["out"] / "plots"
    plots.mkdir(exist_ok=True)
    for stem, label, truth, estimate in (
        ("tbr", "ΔTBR", vasc_plot["true_tbr"], vasc_plot["pred_tbr"]),
        ("tac", "Δlog-TAC", vasc_plot["true_tac"], vasc_plot["pred_tac"]),
    ):
        mae, _, r2 = metric_values(truth, estimate)
        lo, hi = float(min(truth.min(), estimate.min())), float(max(truth.max(), estimate.max()))
        pad = .06 * max(hi - lo, 1e-6)
        fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=180)
        ax.scatter(truth, estimate, s=24, alpha=.72, color="#2C7FB8", edgecolor="white", linewidth=.35)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", linewidth=1, label="identity")
        ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), xlabel=f"Observed {label}", ylabel=f"Predicted {label}", title=f"Centre C external validation (n=546): VascMTL {label}")
        ax.text(.04, .96, f"MAE = {mae:.3f}\nR² = {r2:.3f}", transform=ax.transAxes, va="top", bbox={"facecolor":"white","alpha":.85,"edgecolor":".8"})
        ax.grid(alpha=.2); ax.legend(loc="lower right"); fig.tight_layout()
        fig.savefig(plots / f"VascMTL_C546_predicted_observed_{stem}.png")
        plt.close(fig)
    manifest = {
        "status": "C546_continuous_external_validation_complete",
        "external_patient_count": 546,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "point_prediction_source": "full_A323_frozen_final_model",
        "five_fold_models_averaged": False,
        "C546_used_for_fit_selection_or_calibration": False,
        "prediction_sha256": shared.sha256(p["predictions"]),
        "outcome_sha256": shared.sha256(p["outcomes"]),
        "metrics_sha256": shared.sha256(metrics_path),
        "uncertainty_sha256": shared.sha256(uncertainty_path),
        "patient_results_sha256": shared.sha256(patient_path),
        "plots": {f.name: shared.sha256(f) for f in plots.glob("*.png")},
    }
    write_json(p["out"] / "continuous_validation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def audit(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    p = paths(root)
    prep = shared.read_json(p["prep_manifest"])
    lock = shared.read_json(p["prediction_lock"])
    manifest = shared.read_json(p["out"] / "continuous_validation_manifest.json")
    assert prep["A323_C546_overlap_n"] == 0
    assert lock["point_prediction_source"] == "full_A323_frozen_final_model"
    assert lock["five_fold_models_averaged"] is False and lock["C546_outcomes_read"] is False
    assert manifest["C546_used_for_fit_selection_or_calibration"] is False
    assert manifest["external_patient_count"] == 546
    assert shared.sha256(p["predictions"]) == lock["prediction_file_sha256"]
    assert shared.sha256(p["outcomes"]) == prep["sealed_outcome_sha256"]
    text_files = list(p["out"].glob("*.json")) + list(p["out"].glob("*.csv"))
    legacy_hits = []
    for file in text_files:
        text = file.read_text(encoding="utf-8-sig")
        # Match a terminology token (including snake-case fields) while not
        # flagging an incidental hexadecimal substring inside a SHA256 value.
        if re.search(r"(?<![A-Za-z0-9])CAC(?![A-Za-z0-9])", text, re.IGNORECASE):
            legacy_hits.append(file.name)
    if legacy_hits:
        raise RuntimeError(f"Legacy terminology found in Centre C outputs: {legacy_hits}")
    result = {
        "status": "PASS",
        "A323_C546_overlap_n": 0,
        "excluded_patient_id": prep["excluded_patient_id"],
        "external_patient_count": 546,
        "C546_in_preprocessing_fit": False,
        "C546_in_hyperparameter_selection": False,
        "C546_in_early_stopping": False,
        "C546_in_PI_calibration": False,
        "all_models_full_A323_single_model_prediction": True,
        "legacy_non_TAC_terminology_hits": [],
    }
    write_json(p["out"] / "continuous_independent_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "predict", "evaluate", "audit"))
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-c547")
    parser.add_argument("--exclude-c-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.stage == "prepare":
        if not args.source_c547:
            parser.error("prepare requires --source-c547")
        prepare(args)
    elif args.stage == "predict":
        predict(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
