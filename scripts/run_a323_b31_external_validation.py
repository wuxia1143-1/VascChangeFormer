"""Frozen A323 full-data model -> blinded B31 external validation.

Stage ``predict`` never reads B31 outcomes.  Stage ``evaluate`` is run only
after the prediction file and its hash have been locked.  All reported metrics
are change-scale metrics: delta TBR and delta log1p(TAC). The legacy internal
field name for TAC is ``cac`` and is retained in machine-readable artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from lac_itransformer.data.preprocessing import FoldPreprocessor  # noqa: E402
from lac_itransformer.data.schema import FeatureSchema  # noqa: E402
from lac_itransformer.experiments.prediction import build_torch_model  # noqa: E402
from lac_itransformer.models.lac import LACConfig  # noqa: E402
from lac_itransformer.models.lac_v60 import LACV60Config  # noqa: E402
from lac_itransformer.training import v27_nested as v27  # noqa: E402
from lac_itransformer.training.dataset import model_inputs  # noqa: E402
from lac_itransformer.training.trainer import (  # noqa: E402
    _loader,
    patient_id_hash,
    predict,
    resolve_device,
)
from lac_itransformer.training.v60_nested import (  # noqa: E402
    _config_for_model,
    _decision_modes,
    _feature_flags,
    _generic_temporal_context,
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
CLASSICAL = {"persistence", "elastic_net", "xgboost"}
N_BOOT = 2_000
SEED = 2026


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_name(model_name: str) -> str:
    if model_name == "persistence":
        return "model_meta.json"
    return "model.pkl" if model_name in CLASSICAL else "model.pt"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def add_dummy_targets(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value).copy() for key, value in arrays.items()}
    result["targets"] = np.zeros((len(result["patient_ids"]), 2), dtype=np.float32)
    return result


def residual_to_endpoint(baseline: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            baseline[:, 0] + residual[:, 0],
            np.maximum(
                0.0,
                np.expm1(
                    np.log1p(np.maximum(baseline[:, 1], 0.0)) + residual[:, 1]
                ),
            ),
        ]
    )


def predict_vascmtl_fold(
    model: torch.nn.Module,
    transformed: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    decision: Any,
) -> np.ndarray:
    original_flags = v27._feature_flags
    try:
        v27._feature_flags = _feature_flags
        bundle = v27._predict_base_with_features(
            model, transformed, raw, batch_size, device, "vascmtl"
        )
    finally:
        v27._feature_flags = original_flags
    context = _generic_temporal_context(transformed)
    ids = [str(value) for value in transformed["patient_ids"]]
    order = {value: index for index, value in enumerate(ids)}
    bundle_ids = [str(value) for value in bundle["patient_ids"]]
    bundle_order = np.asarray([order[value] for value in bundle_ids])
    bundle["features"]["cac"] = np.column_stack(
        [bundle["features"]["cac"], context[bundle_order]]
    )
    residual, _ = decision.predict(
        bundle["heads"], bundle["features"], _decision_modes("vascmtl")
    )
    return residual_to_endpoint(bundle["baseline"], residual)


def predict_fold(
    model_name: str,
    fold_dir: Path,
    raw: dict[str, np.ndarray],
    schema: FeatureSchema,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    preprocessor = FoldPreprocessor.load(fold_dir / "preprocessor.json")
    transformed = preprocessor.transform(raw)
    batch_size = int(config.get("batch_size", 64))
    if model_name == "persistence":
        return raw["baseline"].astype(float).copy()
    if model_name in CLASSICAL:
        with (fold_dir / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)
        return np.asarray(model.predict(transformed), dtype=float)
    checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
    if model_name == "vascmtl":
        model_config = LACV60Config(**checkpoint["model_config"])
        model = build_torch_model("vascmtl", model_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        with (fold_dir / "decision_layer.pkl").open("rb") as handle:
            decision = pickle.load(handle)
        prediction = predict_vascmtl_fold(
            model, transformed, raw, device, batch_size, decision
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return prediction
    model_config = LACConfig(**checkpoint["model_config"])
    model = build_torch_model(model_name, model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    prediction, _, ids = predict(model, _loader(transformed, batch_size, False), device)
    expected = [str(value) for value in raw["patient_ids"]]
    if ids != expected:
        raise RuntimeError(f"Patient order changed during B31 prediction: {model_name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(prediction, dtype=float)


def align_frames(frame: pd.DataFrame, patient_ids: np.ndarray) -> pd.DataFrame:
    expected = [str(value) for value in patient_ids]
    if len(frame) != len(expected) or set(frame.patient_id) != set(expected):
        raise RuntimeError("Patient set mismatch")
    return frame.set_index("patient_id").loc[expected].reset_index()


def change_metrics(
    baseline: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    true_tbr = target[:, 0] - baseline[:, 0]
    pred_tbr = prediction[:, 0] - baseline[:, 0]
    true_cac = np.log1p(np.maximum(target[:, 1], 0.0)) - np.log1p(
        np.maximum(baseline[:, 1], 0.0)
    )
    pred_cac = np.log1p(np.maximum(prediction[:, 1], 0.0)) - np.log1p(
        np.maximum(baseline[:, 1], 0.0)
    )

    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denominator = float(np.square(y_true - y_true.mean()).sum())
        return float("nan") if denominator <= 0 else float(
            1.0 - np.square(y_pred - y_true).sum() / denominator
        )

    return {
        "delta_tbr_mae": float(np.abs(pred_tbr - true_tbr).mean()),
        "delta_tbr_rmse": float(np.sqrt(np.square(pred_tbr - true_tbr).mean())),
        "delta_tbr_r2": r2(true_tbr, pred_tbr),
        "delta_log_cac_mae": float(np.abs(pred_cac - true_cac).mean()),
        "delta_log_cac_rmse": float(np.sqrt(np.square(pred_cac - true_cac).mean())),
        "delta_log_cac_r2": r2(true_cac, pred_cac),
    }


def bootstrap_metrics(
    baseline: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    point = change_metrics(baseline, target, prediction)
    rng = np.random.default_rng(SEED)
    samples = {key: np.empty(N_BOOT, dtype=float) for key in point}
    for index in range(N_BOOT):
        selected = rng.integers(0, len(target), len(target))
        values = change_metrics(
            baseline[selected], target[selected], prediction[selected]
        )
        for key in samples:
            samples[key][index] = values[key]
    ci = {
        key: {
            "estimate": point[key],
            "ci_low": float(np.nanquantile(values, 0.025)),
            "ci_high": float(np.nanquantile(values, 0.975)),
        }
        for key, values in samples.items()
    }
    return point, ci


def freeze_predict(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    output = root / "external_validation"
    output.mkdir(parents=True, exist_ok=True)
    blind_path = output / "external_predictions_blinded.csv"
    if blind_path.exists():
        raise FileExistsError(f"Refusing to overwrite {blind_path}")
    schema = FeatureSchema.from_dict(read_json(root / "prepared_data" / "schema.json"))
    predictor_path = root / "prepared_data" / "B31_predictors_without_outcomes.npz"
    raw = add_dummy_targets(load_npz(predictor_path))
    if "targets" in load_npz(predictor_path):
        raise RuntimeError("B31 predictor file contains outcomes")
    device = resolve_device(args.device)
    rows: dict[str, Any] = {
        "patient_id": [str(value) for value in raw["patient_ids"]],
        "baseline_tbr": raw["baseline"][:, 0].astype(float),
        "baseline_cac": raw["baseline"][:, 1].astype(float),
    }
    model_records: dict[str, Any] = {}
    model_names = tuple(args.models) if args.models else MODELS
    pi_lock_path = output / "A323_OOF_PI90_calibration_lock.json"
    if not pi_lock_path.exists():
        raise FileNotFoundError("A323 OOF PI calibration must be locked before prediction")
    pi_lock = read_json(pi_lock_path)
    if tuple(pi_lock["model_order"]) != model_names:
        raise RuntimeError("PI calibration model order differs from prediction model order")
    for model_name in model_names:
        # External point predictions come from the single final model refit on
        # all 323 A-centre patients.  The five OOF folds are used only for the
        # locked residual/PI calibration and are never averaged at inference.
        model_dir = root / "frozen_models" / model_name
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing full-A323 frozen model: {model_name}")
        prediction = predict_fold(
            model_name,
            model_dir,
            raw,
            schema,
            read_json(root / "protocol_lock" / "locked_configs" / f"{model_name}.json"),
            device,
        )
        if prediction.shape != (len(raw["patient_ids"]), 2) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid B31 predictions: {model_name}")
        rows[f"pred_{model_name}_tbr"] = prediction[:, 0]
        rows[f"pred_{model_name}_cac"] = prediction[:, 1]
        model_records[model_name] = {
            "display_name": LABELS[model_name],
            "source": "single_full_A323_refit",
            "model_artifact_sha256": sha256(
                model_dir / artifact_name(model_name)
            ),
            "preprocessor_sha256": sha256(model_dir / "preprocessor.json"),
            "point_prediction_rule": "one frozen final model fitted on all 323 A patients",
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(blind_path, index=False, encoding="utf-8-sig")
    lock = {
        "status": "B31_predictions_locked_before_outcome_evaluation",
        "external_patient_count": len(frame),
        "patient_id_hash": patient_id_hash(frame.patient_id.to_numpy(str)),
        "prediction_file": blind_path.name,
        "prediction_file_sha256": sha256(blind_path),
        "prediction_has_outcomes": False,
        "models": list(model_names),
        "model_records": model_records,
        "device": str(device),
        "A323_model_artifacts_only": True,
        "point_prediction_source": "full_A323_frozen_final_model",
        "five_fold_models_averaged": False,
        "pi_calibration_lock_sha256": sha256(pi_lock_path),
        "pi_calibration_locked_before_prediction": True,
        "B31_outcomes_read": False,
    }
    write_json(output / "prediction_lock_manifest.json", lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


def fit_pi_quantiles(root: Path, model_name: str) -> dict[str, float]:
    oof = pd.read_csv(
        root / "models" / model_name / "out_of_fold_predictions.csv",
        dtype={"patient_id": str},
    )
    oof_baseline = oof[["baseline_tbr", "baseline_cac"]].to_numpy(float)
    oof_target = oof[["true_tbr", "true_cac"]].to_numpy(float)
    oof_prediction = oof[["pred_tbr", "pred_cac"]].to_numpy(float)
    true_tbr = oof_target[:, 0] - oof_baseline[:, 0]
    pred_tbr = oof_prediction[:, 0] - oof_baseline[:, 0]
    true_cac = np.log1p(np.maximum(oof_target[:, 1], 0)) - np.log1p(
        np.maximum(oof_baseline[:, 1], 0)
    )
    pred_cac = np.log1p(np.maximum(oof_prediction[:, 1], 0)) - np.log1p(
        np.maximum(oof_baseline[:, 1], 0)
    )
    return {
        "delta_tbr_q90_abs_error": float(np.quantile(np.abs(pred_tbr - true_tbr), 0.90)),
        "delta_log_cac_q90_abs_error": float(
            np.quantile(np.abs(pred_cac - true_cac), 0.90)
        ),
    }


def calibrate_pi(args: argparse.Namespace) -> None:
    """Lock A323 OOF-derived PI widths before full refit or external prediction."""
    root = Path(args.experiment_root)
    output = root / "external_validation"
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "A323_OOF_PI90_calibration_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Refusing to overwrite {lock_path}")
    if (root / "frozen_models" / "full_refit_manifest.json").exists():
        raise RuntimeError("PI calibration must be locked before full-A323 refit")
    model_names = tuple(args.models) if args.models else MODELS
    quantiles: dict[str, Any] = {}
    for model_name in model_names:
        oof_path = root / "models" / model_name / "out_of_fold_predictions.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing A323 OOF predictions: {model_name}")
        values = fit_pi_quantiles(root, model_name)
        quantiles[model_name] = {
            **values,
            "delta_tbr_interval_width": 2.0 * values["delta_tbr_q90_abs_error"],
            "delta_log_cac_interval_width": 2.0 * values["delta_log_cac_q90_abs_error"],
            "oof_prediction_sha256": sha256(oof_path),
            "oof_patient_count": int(
                len(pd.read_csv(oof_path, usecols=["patient_id"]))
            ),
        }
    lock = {
        "status": "A323_OOF_PI90_calibration_locked_before_full_refit",
        "model_order": list(model_names),
        "calibration_source": "A323_nested_fivefold_complete_OOF_absolute_residuals",
        "quantile": 0.90,
        "interval_rule": "point_prediction_plus_or_minus_locked_q90_absolute_error",
        "external_outcomes_used": False,
        "full_A323_refit_manifest_present_at_lock_time": False,
        "quantiles": quantiles,
    }
    write_json(lock_path, lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    output = root / "external_validation"
    lock = read_json(output / "prediction_lock_manifest.json")
    model_names = tuple(lock["models"])
    blind_path = output / "external_predictions_blinded.csv"
    if sha256(blind_path) != lock["prediction_file_sha256"]:
        raise RuntimeError("Prediction file hash changed after lock")
    if lock["prediction_has_outcomes"] or lock["B31_outcomes_read"]:
        raise RuntimeError("Prediction lock does not certify blinded inference")
    pi_lock_path = output / "A323_OOF_PI90_calibration_lock.json"
    if sha256(pi_lock_path) != lock["pi_calibration_lock_sha256"]:
        raise RuntimeError("A323 OOF PI calibration lock changed after prediction")
    pi_lock = read_json(pi_lock_path)
    if tuple(pi_lock["model_order"]) != model_names:
        raise RuntimeError("PI calibration model order differs from prediction lock")
    predictors = load_npz(root / "prepared_data" / "B31_predictors_without_outcomes.npz")
    outcomes = load_npz(root / "prepared_data" / "B31_sealed_outcomes.npz")
    if set(predictors) & {"targets"}:
        raise RuntimeError("B31 predictor artifact contains targets")
    if not np.array_equal(predictors["patient_ids"].astype(str), outcomes["patient_ids"].astype(str)):
        raise RuntimeError("B31 predictor/outcome patient order mismatch")
    frame = pd.read_csv(blind_path, dtype={"patient_id": str})
    expected_ids = outcomes["patient_ids"].astype(str)
    if not np.array_equal(frame.patient_id.to_numpy(str), expected_ids):
        raise RuntimeError("Prediction and B31 outcome orders differ")
    baseline = outcomes["targets"] * 0.0
    baseline[:, 0] = predictors["baseline"][:, 0]
    baseline[:, 1] = predictors["baseline"][:, 1]
    target = outcomes["targets"].astype(float)
    metrics_rows = []
    pi_rows = []
    plots_dir = output / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_data = {}
    for model_name in model_names:
        prediction = frame[[f"pred_{model_name}_tbr", f"pred_{model_name}_cac"]].to_numpy(float)
        point, ci = bootstrap_metrics(baseline, target, prediction)
        for metric, values in ci.items():
            metrics_rows.append(
                {
                    "model": model_name,
                    "display_name": LABELS[model_name],
                    "metric": metric,
                    "estimate": values["estimate"],
                    "ci_low": values["ci_low"],
                    "ci_high": values["ci_high"],
                    "bootstrap_replicates": N_BOOT,
                }
            )
        q90 = pi_lock["quantiles"][model_name]
        true_tbr = target[:, 0] - baseline[:, 0]
        pred_tbr = prediction[:, 0] - baseline[:, 0]
        true_cac = np.log1p(np.maximum(target[:, 1], 0)) - np.log1p(
            np.maximum(baseline[:, 1], 0)
        )
        pred_cac = np.log1p(np.maximum(prediction[:, 1], 0)) - np.log1p(
            np.maximum(baseline[:, 1], 0)
        )
        tbr_lower, tbr_upper = pred_tbr - q90["delta_tbr_q90_abs_error"], pred_tbr + q90["delta_tbr_q90_abs_error"]
        cac_lower, cac_upper = pred_cac - q90["delta_log_cac_q90_abs_error"], pred_cac + q90["delta_log_cac_q90_abs_error"]
        pi_rows.extend(
            [
                {
                    "model": model_name,
                    "display_name": LABELS[model_name],
                    "outcome": "delta_tbr",
                    "calibration_q90_abs_error": q90["delta_tbr_q90_abs_error"],
                    "coverage_90pi": float(np.mean((true_tbr >= tbr_lower) & (true_tbr <= tbr_upper))),
                    "interval_width": float(2 * q90["delta_tbr_q90_abs_error"]),
                    "n": len(target),
                },
                {
                    "model": model_name,
                    "display_name": LABELS[model_name],
                    "outcome": "delta_log_cac",
                    "calibration_q90_abs_error": q90["delta_log_cac_q90_abs_error"],
                    "coverage_90pi": float(np.mean((true_cac >= cac_lower) & (true_cac <= cac_upper))),
                    "interval_width": float(2 * q90["delta_log_cac_q90_abs_error"]),
                    "n": len(target),
                },
            ]
        )
        if model_name == "vascmtl":
            plot_data = {
                "true_tbr": true_tbr,
                "pred_tbr": pred_tbr,
                "tbr_q90": q90["delta_tbr_q90_abs_error"],
                "true_cac": true_cac,
                "pred_cac": pred_cac,
                "cac_q90": q90["delta_log_cac_q90_abs_error"],
            }

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output / "B31_external_change_metrics_95ci.csv", index=False, encoding="utf-8-sig")
    pi_frame = pd.DataFrame(pi_rows)
    pi_frame.to_csv(output / "B31_external_prediction_uncertainty_90PI.csv", index=False, encoding="utf-8-sig")

    for outcome, label, key, qkey, file_stem in (
        ("tbr", "ΔTBR", "true_tbr", "tbr_q90", "tbr"),
        ("cac", "Δlog-TAC", "true_cac", "cac_q90", "tac"),
    ):
        truth = plot_data[key]
        pred = plot_data[f"pred_{outcome}"]
        q90 = plot_data[qkey]
        fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=180)
        ax.scatter(truth, pred, s=34, alpha=0.82, edgecolor="white", linewidth=0.4)
        lo = float(min(truth.min(), pred.min()))
        hi = float(max(truth.max(), pred.max()))
        pad = 0.06 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0, label="identity")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Observed {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plots_dir / f"VascMTL_B31_predicted_observed_{file_stem}.png")
        plt.close(fig)

    report_rows = []
    for model_name in model_names:
        subset = metrics_frame[metrics_frame.model == model_name].set_index("metric")
        row = {"模型": LABELS[model_name]}
        for metric in (
            "delta_tbr_mae", "delta_tbr_rmse", "delta_tbr_r2",
            "delta_log_cac_mae", "delta_log_cac_rmse", "delta_log_cac_r2",
        ):
            if metric in subset.index:
                value = subset.loc[metric]
                row[metric] = f"{value.estimate:.4f} [{value.ci_low:.4f}, {value.ci_high:.4f}]"
            else:
                row[metric] = "—"
        report_rows.append(row)
    report_table = pd.DataFrame(report_rows)
    pi_table = pi_frame[["display_name", "outcome", "coverage_90pi", "interval_width", "calibration_q90_abs_error"]].copy()
    pi_table["coverage_90pi"] = pi_table.coverage_90pi.map(lambda value: f"{value:.3f}")
    pi_table["interval_width"] = pi_table.interval_width.map(lambda value: f"{value:.4f}")
    pi_table["calibration_q90_abs_error"] = pi_table.calibration_q90_abs_error.map(lambda value: f"{value:.4f}")
    report = "\n".join(
        [
            "# A323冻结模型→B31独立外部验证",
            "",
            "## 验证口径",
            "",
            "- A中心323例用于模型开发；B中心31例仅作为独立外部验证集。",
            "- 每个模型先按A323内部五折开发结果锁定超参数，再在全部323例A患者上全量重训并冻结；B31点预测来自该单一full-A323最终模型，不对五折模型取均值。",
            "- 所有指标均为变化量：ΔTBR=随访TBR−基线TBR；Δlog-TAC=log1p(随访TAC)−log1p(基线TAC)。机器可读文件为兼容既有管线，仍使用内部字段名`delta_log_cac`。",
            f"- 95%CI为B31患者级Bootstrap percentile区间（{N_BOOT}次，seed={SEED}）。",
            "- 90% PI为A323 OOF绝对误差90%分位数校准的对称区间，区间宽度在B31内固定；未用B31结局校准。",
            "",
            "## 所有模型的连续预测性能",
            "",
            "| 模型 | ΔTBR MAE | ΔTBR RMSE | ΔTBR R² | Δlog-TAC MAE | Δlog-TAC RMSE | Δlog-TAC R² |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        + [
            "| " + " | ".join(str(row[column]) for column in report_table.columns) + " |"
            for _, row in report_table.iterrows()
        ]
        + [
            "",
            "## VascMTL Predicted–Observed图",
            "",
            "图中横轴为B31真实变化量，纵轴为VascMTL预测变化量；对应图文件位于`plots/`目录。",
            "",
            "## VascMTL预测不确定性",
            "",
            "| 结局 | 90% PI coverage | interval width | OOF校准误差90%分位数 |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| {row.display_name} / {'delta_log_tac' if row.outcome == 'delta_log_cac' else row.outcome} | {row.coverage_90pi} | {row.interval_width} | {row.calibration_q90_abs_error} |"
                for row in pi_table.itertuples(index=False)
                if row.display_name == LABELS["vascmtl"]
            ],
            "",
            "## 结果解释",
            "",
            "- 外部验证应主要看模型排序和方向，不将31例上的点估计差异直接解释为显著性证据。",
            "- B31仅31例，95%CI预计较宽；PI coverage是经验覆盖率，不是正式校准检验。",
            "- 如果外部MAE/RMSE排序与A323内部排序一致，可视为方向性复现；若不一致，应优先检查中心分布漂移和区间覆盖，而不是据此重调模型。",
            "",
        ]
    )
    report_path = output / "B31_EXTERNAL_VALIDATION_REPORT_CN.md"
    report_path.write_text(report, encoding="utf-8")
    evaluation = {
        "status": "B31_external_validation_complete",
        "external_patient_count": len(target),
        "models": list(model_names),
        "prediction_file_sha256": sha256(blind_path),
        "outcome_file_sha256": sha256(root / "prepared_data" / "B31_sealed_outcomes.npz"),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "pi_calibration": "A323_OOF_absolute_error_q90_fixed_width",
        "pi_calibration_lock_sha256": sha256(pi_lock_path),
        "pi_locked_before_full_refit_and_external_prediction": True,
        "B31_used_for_fit_or_selection": False,
        "metrics_file_sha256": sha256(output / "B31_external_change_metrics_95ci.csv"),
        "uncertainty_file_sha256": sha256(output / "B31_external_prediction_uncertainty_90PI.csv"),
        "report_sha256": sha256(report_path),
        "plots": {
            name: sha256(plots_dir / name)
            for name in (
                "VascMTL_B31_predicted_observed_tbr.png",
                "VascMTL_B31_predicted_observed_tac.png",
            )
        },
    }
    write_json(output / "external_validation_manifest.json", evaluation)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibrate", "predict", "evaluate"))
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "calibrate":
        calibrate_pi(args)
    elif args.stage == "predict":
        freeze_predict(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
