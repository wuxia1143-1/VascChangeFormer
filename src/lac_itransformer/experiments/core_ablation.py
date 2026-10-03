from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from ..data.schema import FeatureSchema
from ..models.lac import LACConfig
from ..training.folds import (
    fold_plan_checksum,
    save_patient_fold_plan,
    validate_patient_fold_plan,
)
from ..training.trainer import run_cross_validation
from .ablation import (
    ABLATION_DESCRIPTIONS,
    ABLATION_OVERRIDES,
    COUPLING_ABLATIONS,
    ROUTING_ABLATIONS,
)


CORE_ABLATION_VARIANTS = tuple(ABLATION_OVERRIDES)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty ablation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_locked_comparison_protocol(
    comparison_dir: str | Path,
    patient_ids: np.ndarray,
    schema: FeatureSchema,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    """Load, verify and reuse the comparison experiment's exact patient folds."""
    root = Path(comparison_dir)
    manifest_path = root / "internal_fivefold_manifest.json"
    config_path = root / "training_config.json"
    plan_path = root / "patient_fold_plan.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    if manifest.get("same_patient_folds_for_all_models") is not True:
        raise ValueError("Comparison suite does not certify one shared patient fold plan")
    if manifest.get("external_center_used") is not False:
        raise ValueError("Comparison suite is not certified as development-center-only")
    if int(base_config.get("num_folds", 0)) != 5:
        raise ValueError("Core ablations require the fixed five-fold comparison protocol")
    if "lac_itransformer" not in set(manifest.get("models", [])):
        raise ValueError("Comparison suite does not contain the full LAC-iTransformer model")
    if manifest.get("training_config_sha256") != _canonical_hash(base_config):
        raise ValueError("Comparison training configuration checksum mismatch")
    frame = pd.read_csv(plan_path, dtype={"patient_id": str})
    if set(frame.columns) != {"patient_id", "test_fold"}:
        raise ValueError("Comparison patient fold table has an invalid schema")
    plan = dict(zip(frame["patient_id"], frame["test_fold"].astype(int)))
    ids = np.asarray([str(value) for value in patient_ids])
    validate_patient_fold_plan(ids, plan, n_folds=5)
    checksum = fold_plan_checksum(plan)
    if checksum != manifest["fold_plan"]["checksum_sha256"]:
        raise ValueError("Comparison patient fold table checksum mismatch")
    checkpoint = torch.load(
        root / "lac_itransformer" / "fold_1" / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["schema"] != schema.to_dict():
        raise ValueError("Development data schema differs from the comparison experiment")
    source = {
        "comparison_manifest_sha256": _file_sha256(manifest_path),
        "comparison_training_config_sha256": _file_sha256(config_path),
        "comparison_fold_plan_sha256": _file_sha256(plan_path),
        "fold_plan_checksum": checksum,
        "comparison_git_revision": checkpoint["git_revision"],
        "n_patients": len(ids),
    }
    return base_config, plan, source


def _paired_delta(
    ablated_error: np.ndarray,
    full_error: np.ndarray,
    n_bootstrap: int,
    seed: int,
    reference: bool = False,
) -> dict[str, float]:
    difference = np.asarray(ablated_error) - np.asarray(full_error)
    estimate = float(np.mean(difference))
    if reference:
        return {
            "estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "probability_ablation_worse": 0.5,
            "bootstrap_two_sided_p": 1.0,
        }
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(difference), len(difference))
        samples[index] = float(np.mean(difference[selected]))
    probability_positive = float(np.mean(samples > 0))
    probability_nonpositive = float(np.mean(samples <= 0))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_ablation_worse": probability_positive,
        "bootstrap_two_sided_p": min(1.0, 2.0 * min(probability_positive, probability_nonpositive)),
    }


def _aligned_oof(reference: pd.DataFrame, candidate: pd.DataFrame, variant: str) -> pd.DataFrame:
    required = {"patient_id", "fold", "true_tbr", "pred_tbr", "true_cac", "pred_cac"}
    if not required.issubset(candidate):
        raise ValueError(f"{variant} OOF prediction contract is incomplete")
    left = reference.sort_values("patient_id").reset_index(drop=True)
    right = candidate.sort_values("patient_id").reset_index(drop=True)
    if not np.array_equal(left["patient_id"], right["patient_id"]):
        raise ValueError(f"{variant} patient set differs from full model")
    if not np.array_equal(left["fold"], right["fold"]):
        raise ValueError(f"{variant} fold assignments differ from full model")
    for endpoint in ("tbr", "cac"):
        if not np.allclose(left[f"true_{endpoint}"], right[f"true_{endpoint}"]):
            raise ValueError(f"{variant} {endpoint} truth differs from full model")
    return right


def run_core_ablation_suite(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    comparison_dir: str | Path,
    output_dir: str | Path,
    variants: Iterable[str] = CORE_ABLATION_VARIANTS,
    bootstrap_replicates: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Retrain every core ablation in the comparison suite's exact five folds."""
    selected = tuple(dict.fromkeys(variants))
    unknown = set(selected) - set(ABLATION_OVERRIDES)
    if unknown:
        raise KeyError(f"Unknown core ablation variants: {sorted(unknown)}")
    if "full" not in selected:
        raise ValueError("The full model must be retrained as the paired ablation reference")
    base_config, fold_plan, comparison_source = load_locked_comparison_protocol(
        comparison_dir, arrays["patient_ids"], schema
    )
    base_lac = LACConfig(
        static_dim=len(schema.static_features),
        num_variables=len(schema.longitudinal_features),
        treatment_dim=len(schema.treatment_features),
        num_patches=schema.time_patches,
        **base_config.get("model", {}),
    )
    if not (
        base_lac.competitive_routing
        and base_lac.background_route
        and base_lac.dynamic_anchors
        and base_lac.baseline_anchoring
        and base_lac.coupling_mode == "asymmetric"
        and base_lac.treatment_conditioning
    ):
        raise ValueError("Comparison LAC configuration is not the prespecified full model")
    n_bootstrap = int(
        bootstrap_replicates
        if bootstrap_replicates is not None
        else base_config.get("bootstrap_replicates", 2000)
    )
    if n_bootstrap < 1:
        raise ValueError("bootstrap_replicates must be positive")
    output = _prepare_output(output_dir)
    fold_manifest = save_patient_fold_plan(
        fold_plan,
        output / "patient_fold_plan.csv",
        seed=int(base_config.get("seed", 2026)),
        n_folds=5,
    )
    summaries: dict[str, Any] = {}
    variant_configs: dict[str, dict[str, Any]] = {}
    for variant in selected:
        config = copy.deepcopy(base_config)
        if device_name is not None:
            config["device"] = device_name
        config["bootstrap_replicates"] = n_bootstrap
        config["model_name"] = "lac_itransformer"
        config["experiment_variant"] = variant
        model_options = copy.deepcopy(config.get("model", {}))
        model_options.update(ABLATION_OVERRIDES[variant])
        config["model"] = model_options
        variant_configs[variant] = config
        summaries[variant] = run_cross_validation(
            arrays,
            schema,
            config,
            output / variant,
            fold_assignments=fold_plan,
        )
        if summaries[variant]["cv_protocol"]["fold_plan_checksum"] != fold_manifest["checksum_sha256"]:
            raise RuntimeError(f"{variant} did not reuse the comparison fold plan")
    full = pd.read_csv(output / "full" / "out_of_fold_predictions.csv", dtype={"patient_id": str})
    full = full.sort_values("patient_id").reset_index(drop=True)
    full_tbr_error = np.abs(full["true_tbr"].to_numpy() - full["pred_tbr"].to_numpy())
    full_cac_error = np.abs(
        np.log1p(np.maximum(full["true_cac"].to_numpy(), 0))
        - np.log1p(np.maximum(full["pred_cac"].to_numpy(), 0))
    )
    full_raw_cac_error = np.abs(
        full["true_cac"].to_numpy() - full["pred_cac"].to_numpy()
    )
    paired_rows = []
    fold_rows = []
    metric_rows = []
    paired_by_variant: dict[str, Any] = {}
    seed = int(base_config.get("seed", 2026))
    for variant_index, variant in enumerate(selected):
        candidate = _aligned_oof(
            full,
            pd.read_csv(
                output / variant / "out_of_fold_predictions.csv",
                dtype={"patient_id": str},
            ),
            variant,
        )
        candidate_tbr_error = np.abs(
            candidate["true_tbr"].to_numpy() - candidate["pred_tbr"].to_numpy()
        )
        candidate_cac_error = np.abs(
            np.log1p(np.maximum(candidate["true_cac"].to_numpy(), 0))
            - np.log1p(np.maximum(candidate["pred_cac"].to_numpy(), 0))
        )
        candidate_raw_cac_error = np.abs(
            candidate["true_cac"].to_numpy() - candidate["pred_cac"].to_numpy()
        )
        tbr_delta = _paired_delta(
            candidate_tbr_error,
            full_tbr_error,
            n_bootstrap,
            seed + 100 * variant_index,
            reference=variant == "full",
        )
        cac_delta = _paired_delta(
            candidate_cac_error,
            full_cac_error,
            n_bootstrap,
            seed + 100 * variant_index + 1,
            reference=variant == "full",
        )
        raw_cac_delta = _paired_delta(
            candidate_raw_cac_error,
            full_raw_cac_error,
            n_bootstrap,
            seed + 100 * variant_index + 2,
            reference=variant == "full",
        )
        paired_by_variant[variant] = {
            "delta_tbr_mae": tbr_delta,
            "delta_log_cac_mae": cac_delta,
            "delta_raw_cac_mae": raw_cac_delta,
        }
        groups = []
        if variant in ROUTING_ABLATIONS:
            groups.append("routing")
        if variant in COUPLING_ABLATIONS:
            groups.append("coupling")
        for endpoint, result in (
            ("TBR_MAE", tbr_delta),
            ("log1p_CAC_MAE", cac_delta),
            ("raw_CAC_MAE", raw_cac_delta),
        ):
            paired_rows.append({
                "variant": variant,
                "groups": "+".join(groups),
                "endpoint": endpoint,
                "delta_definition": "ablation_minus_full",
            } | result)
        for fold in range(1, 6):
            selected_rows = candidate["fold"].to_numpy() == fold
            fold_rows.append({
                "variant": variant,
                "fold": fold,
                "delta_tbr_mae": float(
                    np.mean(candidate_tbr_error[selected_rows] - full_tbr_error[selected_rows])
                ),
                "delta_log_cac_mae": float(
                    np.mean(candidate_cac_error[selected_rows] - full_cac_error[selected_rows])
                ),
                "delta_raw_cac_mae": float(
                    np.mean(
                        candidate_raw_cac_error[selected_rows]
                        - full_raw_cac_error[selected_rows]
                    )
                ),
            })
        summary = summaries[variant]
        fold_records = summary["fold_records"]
        metric_rows.append({
            "variant": variant,
            "description": ABLATION_DESCRIPTIONS[variant],
            "parameter_count": fold_records[0]["parameter_count"],
            "selection_seconds": float(sum(row["selection_seconds"] for row in fold_records)),
            "refit_seconds": float(sum(row["refit_seconds"] for row in fold_records)),
            "inference_seconds": float(sum(row["inference_seconds"] for row in fold_records)),
            **summary["pooled_oof_metrics"],
            "raw_cac_mae": float(np.mean(candidate_raw_cac_error)),
            "delta_tbr_mae": tbr_delta["estimate"],
            "delta_log_cac_mae": cac_delta["estimate"],
            "delta_raw_cac_mae": raw_cac_delta["estimate"],
        })
    pd.DataFrame(metric_rows).to_csv(output / "ablation_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "paired_delta_mae.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_delta_mae.csv", index=False)
    (output / "variant_configs.json").write_text(
        json.dumps(variant_configs, indent=2), encoding="utf-8"
    )
    result_scope = str(base_config.get("result_scope", ""))
    if result_scope.startswith("synthetic_smoke"):
        status = "synthetic_smoke_only"
    elif result_scope.startswith("synthetic"):
        status = "synthetic_diagnostic_only"
    else:
        status = "development_internal_ablation_completed"
    result = {
        "status": status,
        "formal_conclusions_generated": False,
        "protocol": "core_ablation_retrained_in_locked_comparison_patient_fivefold",
        "same_development_patients_as_comparison": True,
        "same_patient_fold_plan_as_comparison": True,
        "each_variant_retrained_in_all_five_folds": True,
        "external_center_used": False,
        "variants": list(selected),
        "variant_count": len(selected),
        "groups": {
            "routing": [name for name in ROUTING_ABLATIONS if name in selected],
            "coupling": [name for name in COUPLING_ABLATIONS if name in selected],
        },
        "descriptions": {name: ABLATION_DESCRIPTIONS[name] for name in selected},
        "comparison_source": comparison_source,
        "fold_plan": fold_manifest,
        "bootstrap_replicates": n_bootstrap,
        "delta_sign_convention": "positive means the ablation has worse MAE than full",
        "paired_delta_mae": paired_by_variant,
        "prespecified_questions": [
            "Is phenotype competition necessary?",
            "Does the background path absorb forced-routing errors?",
            "Are patient/patch-specific anchors necessary?",
            "Does baseline phenotype anchoring help?",
            "Does removing inflammation-to-calcification mainly harm CAC prediction?",
            "Does calcification-to-inflammation add complementary TBR information?",
            "Is symmetric coupling inferior to the medically hypothesized asymmetry?",
            "Does treatment conditioning capture heterogeneous vascular response?",
        ],
    }
    (output / "core_ablation_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
