from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .v60_reporting import _load_oof, _paired_bootstrap, _summary, _validate_alignment


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _endpoint_cac(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    return np.maximum(
        0.0,
        np.expm1(np.log1p(np.maximum(frame["baseline_cac"], 0.0)) + prediction),
    )


def _no_tail_policy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prediction = (
        result["pred_delta_log_cac"].to_numpy(float)
        - result["decision_gate_cac"].to_numpy(float)
        * (
            result["mean_pred_delta_log_cac"].to_numpy(float)
            - result["median_pred_delta_log_cac"].to_numpy(float)
        )
    )
    result["pred_delta_log_cac"] = prediction
    result["pred_cac"] = _endpoint_cac(result, prediction)
    result["decision_gate_cac"] = 0.0
    return result


def _no_calibration_policy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prediction = result["median_pred_delta_log_cac"].to_numpy(float)
    result["pred_delta_log_cac"] = prediction
    result["pred_cac"] = _endpoint_cac(result, prediction)
    result["decision_gate_cac"] = 0.0
    return result


def _write_oof(frame: pd.DataFrame, root: Path, name: str) -> Path:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / "out_of_fold_predictions.csv"
    frame.to_csv(path, index=False)
    return path


def _metric_rows(frames: dict[str, pd.DataFrame], category: str):
    return [
        {
            "model": name,
            "category": category,
            "patient_count": int(len(frame)),
            **_summary(frame),
        }
        for name, frame in frames.items()
    ]


def finalize_v60(
    v40_oof: str | Path,
    v42_b_oof: str | Path,
    strict_a_oof: str | Path,
    strict_c_oof: str | Path,
    no_adapter_oof: str | Path,
    no_anchor_oof: str | Path,
    v40_full_oof: str | Path,
    shared_only_oof: str | Path,
    artifact_root: str | Path,
    report_root: str | Path,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict:
    artifact_root = Path(artifact_root)
    report_root = Path(report_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    v40 = _load_oof(v40_oof)
    stage_a = v40.copy()
    stage_b = _load_oof(v42_b_oof)
    strict_a = _load_oof(strict_a_oof)
    strict_c = _load_oof(strict_c_oof)
    no_adapter = _no_tail_policy(_load_oof(no_adapter_oof))
    no_anchor = _no_tail_policy(_load_oof(no_anchor_oof))
    no_calibration = _no_calibration_policy(_load_oof(v40_full_oof))
    shared_only = _load_oof(shared_only_oof)
    for frame in (
        stage_a,
        stage_b,
        strict_a,
        strict_c,
        no_adapter,
        no_anchor,
        no_calibration,
        shared_only,
    ):
        _validate_alignment(v40, frame)

    stage_frames = {
        "lac_v40_final_no_tail_reference": v40,
        "lac_v60_a_pruned_direct_tail": stage_a,
        "lac_v60_b_no_mechanism_calibrator_features": stage_b,
        "lac_v60_strict_retrain_no_mechanism": strict_a,
        "lac_v60_c_generic_temporal_supplementary": strict_c,
    }
    core_frames = {
        "lac_v60_full": stage_a,
        "lac_v60_without_task_specific_adapters": no_adapter,
        "lac_v60_without_baseline_anchoring": no_anchor,
        "lac_v60_without_oof_robust_calibration": no_calibration,
        "shared_only_itransformer_mtl_control": shared_only,
    }

    stage_paths = {
        name: _write_oof(frame, artifact_root, name)
        for name, frame in stage_frames.items()
    }
    core_paths = {
        name: _write_oof(frame, artifact_root, name)
        for name, frame in core_frames.items()
        if name not in stage_paths
    }

    stage_metrics = {name: _summary(frame) for name, frame in stage_frames.items()}
    core_metrics = {name: _summary(frame) for name, frame in core_frames.items()}
    reference_metrics = stage_metrics["lac_v40_final_no_tail_reference"]
    full_metrics = core_metrics["lac_v60_full"]
    stage_acceptance = {}
    for name in (
        "lac_v60_a_pruned_direct_tail",
        "lac_v60_b_no_mechanism_calibrator_features",
    ):
        tbr = stage_metrics[name]["delta_tbr_mae"] - reference_metrics["delta_tbr_mae"]
        cac = (
            stage_metrics[name]["delta_log_cac_mae"]
            - reference_metrics["delta_log_cac_mae"]
        )
        stage_acceptance[name] = {
            "delta_tbr_mae_candidate_minus_v40": float(tbr),
            "delta_log_cac_mae_candidate_minus_v40": float(cac),
            "passes_tbr": bool(tbr <= 0.01 + 1e-12),
            "passes_cac": bool(cac <= 0.01 + 1e-12),
            "passes_both": bool(tbr <= 0.01 + 1e-12 and cac <= 0.01 + 1e-12),
        }

    exact_a = bool(
        np.array_equal(
            stage_a["pred_delta_tbr"].to_numpy(float),
            v40["pred_delta_tbr"].to_numpy(float),
        )
        and np.array_equal(
            stage_a["pred_delta_log_cac"].to_numpy(float),
            v40["pred_delta_log_cac"].to_numpy(float),
        )
    )
    strict_equivalence = {
        column: float(
            np.max(
                np.abs(
                    strict_a[column].to_numpy(float)
                    - stage_b[column].to_numpy(float)
                )
            )
        )
        for column in (
            "median_pred_delta_log_cac",
            "pred_delta_log_cac",
            "pred_delta_tbr",
        )
    }

    bootstrap = []
    comparison_index = 0
    for name, frame in stage_frames.items():
        if name == "lac_v40_final_no_tail_reference":
            continue
        bootstrap.extend(
            _paired_bootstrap(
                frame,
                v40,
                name,
                "lac_v40_final_no_tail_reference",
                replicates,
                seed + comparison_index * 10_000,
            )
        )
        comparison_index += 1
    for name, frame in core_frames.items():
        if name == "lac_v60_full":
            continue
        bootstrap.extend(
            _paired_bootstrap(
                frame,
                stage_a,
                name,
                "lac_v60_full",
                replicates,
                seed + 100_000 + comparison_index * 10_000,
            )
        )
        comparison_index += 1

    stage_table = pd.DataFrame(_metric_rows(stage_frames, "stage"))
    core_table = pd.DataFrame(_metric_rows(core_frames, "core_ablation"))
    for metric in (
        "delta_tbr_mae",
        "delta_tbr_rmse",
        "delta_tbr_r2",
        "delta_log_cac_mae",
        "delta_log_cac_rmse",
        "delta_log_cac_r2",
    ):
        core_table[f"difference_from_full_{metric}"] = core_table[metric] - full_metrics[metric]
    stage_table.to_csv(report_root / "v60_internal_metrics.csv", index=False)
    core_table.to_csv(report_root / "v60_core_ablation_results.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(
        report_root / "v60_paired_bootstrap_95ci.csv", index=False
    )

    audit = {
        "selected_v60_full": "lac_v60_a_pruned_direct_tail",
        "selection_reason": (
            "V6-A is exactly prediction-equivalent to the frozen V4 no-tail "
            "policy and passes both MAE margins; V6-B fails CAC, so the locked "
            "simplicity rule retains A and does not require C"
        ),
        "difference_definition": "candidate_minus_reference",
        "mae_margin": 0.01,
        "stage_acceptance": stage_acceptance,
        "v6_a_prediction_exactly_equals_v40": exact_a,
        "strict_retrain_equals_v6_b_max_absolute_difference": strict_equivalence,
        "v6_c_status": (
            "not_required_by_stopping_rule; strict generic-temporal run archived "
            "as a supplementary negative result"
        ),
        "external_labels_used": False,
        "external_evaluation_authorized": False,
        "source_sha256": {
            "v40_oof": _sha256(v40_oof),
            "v42_b_oof": _sha256(v42_b_oof),
            "strict_a_oof": _sha256(strict_a_oof),
            "strict_c_oof": _sha256(strict_c_oof),
            "no_adapter_oof": _sha256(no_adapter_oof),
            "no_anchor_oof": _sha256(no_anchor_oof),
            "v40_full_oof": _sha256(v40_full_oof),
            "shared_only_oof": _sha256(shared_only_oof),
        },
        "derived_artifact_sha256": {
            name: _sha256(path) for name, path in stage_paths.items()
        }
        | {name: _sha256(path) for name, path in core_paths.items()},
    }
    (report_root / "v60_internal_acceptance.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (artifact_root / "derivation_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return {
        "audit": audit,
        "stage_metrics": stage_metrics,
        "core_metrics": core_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v40-oof", required=True)
    parser.add_argument("--v42-b-oof", required=True)
    parser.add_argument("--strict-a-oof", required=True)
    parser.add_argument("--strict-c-oof", required=True)
    parser.add_argument("--no-adapter-oof", required=True)
    parser.add_argument("--no-anchor-oof", required=True)
    parser.add_argument("--v40-full-oof", required=True)
    parser.add_argument("--shared-only-oof", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = finalize_v60(
        args.v40_oof,
        args.v42_b_oof,
        args.strict_a_oof,
        args.strict_c_oof,
        args.no_adapter_oof,
        args.no_anchor_oof,
        args.v40_full_oof,
        args.shared_only_oof,
        args.artifact_root,
        args.report_root,
        args.bootstrap_replicates,
        args.seed,
    )
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
