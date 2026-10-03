"""Independent audit of the A323-full-refit -> B31 external result."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

CLASSICAL = {"elastic_net", "xgboost"}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def artifact_name(model):
    return "model_meta.json" if model == "persistence" else ("model.pkl" if model in CLASSICAL else "model.pt")

def metrics(baseline, target, prediction):
    true_tbr, pred_tbr = target[:, 0] - baseline[:, 0], prediction[:, 0] - baseline[:, 0]
    true_cac = np.log1p(np.maximum(target[:, 1], 0)) - np.log1p(np.maximum(baseline[:, 1], 0))
    pred_cac = np.log1p(np.maximum(prediction[:, 1], 0)) - np.log1p(np.maximum(baseline[:, 1], 0))
    def r2(y, p):
        den = np.square(y - y.mean()).sum()
        return float("nan") if den <= 0 else float(1 - np.square(p-y).sum()/den)
    return {
        "delta_tbr_mae": float(np.abs(pred_tbr-true_tbr).mean()),
        "delta_tbr_rmse": float(np.sqrt(np.square(pred_tbr-true_tbr).mean())),
        "delta_tbr_r2": r2(true_tbr, pred_tbr),
        "delta_log_cac_mae": float(np.abs(pred_cac-true_cac).mean()),
        "delta_log_cac_rmse": float(np.sqrt(np.square(pred_cac-true_cac).mean())),
        "delta_log_cac_r2": r2(true_cac, pred_cac),
    }

def oof_pi(root, model):
    frame = pd.read_csv(root/"models"/model/"out_of_fold_predictions.csv")
    base = frame[["baseline_tbr", "baseline_cac"]].to_numpy(float)
    target = frame[["true_tbr", "true_cac"]].to_numpy(float)
    pred = frame[["pred_tbr", "pred_cac"]].to_numpy(float)
    true_tbr = target[:, 0] - base[:, 0]
    pred_tbr = pred[:, 0] - base[:, 0]
    true_cac = np.log1p(np.maximum(target[:, 1], 0)) - np.log1p(np.maximum(base[:, 1], 0))
    pred_cac = np.log1p(np.maximum(pred[:, 1], 0)) - np.log1p(np.maximum(base[:, 1], 0))
    return {
        "delta_tbr_q90_abs_error": float(np.quantile(np.abs(pred_tbr-true_tbr), 0.90)),
        "delta_log_cac_q90_abs_error": float(np.quantile(np.abs(pred_cac-true_cac), 0.90)),
    }

def main():
    p = argparse.ArgumentParser(); p.add_argument("--experiment-root", required=True); args=p.parse_args()
    root = Path(args.experiment_root); ext = root / "external_validation"
    lock = json.loads((ext/"prediction_lock_manifest.json").read_text())
    assert lock["point_prediction_source"] == "full_A323_frozen_final_model"
    assert lock["five_fold_models_averaged"] is False
    assert lock["B31_outcomes_read"] is False
    pi_path = ext/"A323_OOF_PI90_calibration_lock.json"
    assert sha(pi_path) == lock["pi_calibration_lock_sha256"]
    assert lock["pi_calibration_locked_before_prediction"] is True
    pi_lock = json.loads(pi_path.read_text())
    assert pi_lock["status"] == "A323_OOF_PI90_calibration_locked_before_full_refit"
    assert pi_lock["external_outcomes_used"] is False
    assert pi_lock["full_A323_refit_manifest_present_at_lock_time"] is False
    frame = pd.read_csv(ext/"external_predictions_blinded.csv", dtype={"patient_id": str})
    assert not any(c.startswith("true_") for c in frame.columns)
    for model in lock["models"]:
        d = root/"frozen_models"/model
        assert sha(d/artifact_name(model)) == lock["model_records"][model]["model_artifact_sha256"]
        assert sha(d/"preprocessor.json") == lock["model_records"][model]["preprocessor_sha256"]
    with np.load(root/"prepared_data/B31_predictors_without_outcomes.npz", allow_pickle=False) as z:
        predictors = {k:z[k] for k in z.files}
    with np.load(root/"prepared_data/B31_sealed_outcomes.npz", allow_pickle=False) as z:
        outcomes = {k:z[k] for k in z.files}
    assert "targets" not in predictors
    assert np.array_equal(frame.patient_id.to_numpy(str), outcomes["patient_ids"].astype(str))
    # Match the evaluator's serialized float32 baseline container exactly.
    baseline = outcomes["targets"].astype(np.float32) * 0.0
    baseline[:, 0] = predictors["baseline"][:, 0]
    baseline[:, 1] = predictors["baseline"][:, 1]
    target = outcomes["targets"].astype(float)
    stored = pd.read_csv(ext/"B31_external_change_metrics_95ci.csv")
    uncertainty = pd.read_csv(ext/"B31_external_prediction_uncertainty_90PI.csv")
    max_diff = 0.0
    max_pi_diff = 0.0
    for model in lock["models"]:
        pred = frame[[f"pred_{model}_tbr", f"pred_{model}_cac"]].to_numpy(float)
        point = metrics(baseline, target, pred)
        rows = stored[stored.model == model]
        for metric, value in point.items():
            observed = float(rows.loc[rows.metric == metric, "estimate"].iloc[0])
            max_diff = max(max_diff, abs(observed-value))
        recomputed_pi = oof_pi(root, model)
        locked_pi = pi_lock["quantiles"][model]
        for outcome, key in (("delta_tbr", "delta_tbr_q90_abs_error"), ("delta_log_cac", "delta_log_cac_q90_abs_error")):
            max_pi_diff = max(max_pi_diff, abs(recomputed_pi[key]-float(locked_pi[key])))
            row = uncertainty[(uncertainty.model == model) & (uncertainty.outcome == outcome)].iloc[0]
            max_pi_diff = max(max_pi_diff, abs(float(row.calibration_q90_abs_error)-float(locked_pi[key])))
            max_pi_diff = max(max_pi_diff, abs(float(row.interval_width)-2.0*float(locked_pi[key])))
    assert max_diff < 1e-12
    assert max_pi_diff < 1e-12
    audit = {"status":"PASS", "models":list(lock["models"]), "external_patient_count":len(frame), "point_source":"full_A323_frozen_final_model", "five_fold_mean_used":False, "pi_source":"A323_nested_OOF_locked_before_full_refit_and_prediction", "pi_calibration_lock_sha256":sha(pi_path), "outcomes_loaded_only_after_prediction_lock":True, "max_metric_reconciliation_abs_diff":max_diff, "max_pi_reconciliation_abs_diff":max_pi_diff}
    (ext/"independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
