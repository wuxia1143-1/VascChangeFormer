from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.schema import FeatureSchema
from ..training.folds import build_patient_fold_plan, save_patient_fold_plan
from ..training.trainer import run_cross_validation
from .prediction import model_registry


V2_QUICK_MODELS = (
    "lac_itransformer",
    "lac_v1_no_competitive_routing",
    "lac_v2",
    "lac_v2_no_adapters",
    "lac_v2_no_coupling",
    "lac_v2_forward_only",
    "lac_v2_symmetric",
    "lac_v2_no_treatment",
    "elastic_net",
    "xgboost",
    "itransformer_mtl",
)

DISPLAY_NAMES = {
    "lac_itransformer": "V1 Full",
    "lac_v1_no_competitive_routing": "V1 w/o competitive routing",
    "lac_v2": "V2 Full",
    "lac_v2_no_adapters": "V2 w/o phenotype adapters",
    "lac_v2_no_coupling": "V2 w/o coupling",
    "lac_v2_forward_only": "V2 forward-only (supporting reverse-path probe)",
    "lac_v2_symmetric": "V2 symmetric coupling",
    "lac_v2_no_treatment": "V2 w/o treatment conditioning",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "itransformer_mtl": "iTransformer-MTL",
}

ENDPOINT_METRICS = (
    "tbr_mae",
    "tbr_rmse",
    "tbr_r2",
    "log_cac_mae",
    "log_cac_rmse",
    "log_cac_r2",
)
CHANGE_METRICS = (
    "delta_tbr_mae",
    "delta_tbr_rmse",
    "delta_tbr_r2",
    "delta_log_cac_mae",
    "delta_log_cac_rmse",
    "delta_log_cac_r2",
)


def _hash_config(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_completed_summary(directory: Path) -> dict[str, Any] | None:
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"Partial output exists at {directory}; it is not overwritten automatically"
        )
    return None


def run_v2_quick_comparison_seed(
    arrays: dict[str, np.ndarray],
    schema: FeatureSchema,
    base_config: dict[str, Any],
    output_dir: str | Path,
    models: Iterable[str] = V2_QUICK_MODELS,
) -> dict[str, Any]:
    """Run one development-only seed with a locked patient-level fold plan."""
    if int(base_config.get("num_folds", 5)) != 5:
        raise ValueError("The V2 quick comparison requires exactly five folds")
    scope = str(base_config.get("result_scope", ""))
    if not scope.startswith("synthetic_v2_quick"):
        raise ValueError("The result scope must identify a synthetic V2 quick run")
    selected = tuple(dict.fromkeys(models))
    unknown = set(selected) - set(model_registry())
    if unknown:
        raise KeyError(f"Unknown V2 quick-comparison models: {sorted(unknown)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(base_config.get("seed", 2026))
    patient_ids = np.asarray([str(value) for value in arrays["patient_ids"]])
    fold_plan = build_patient_fold_plan(patient_ids, n_folds=5, seed=seed)
    fold_manifest = save_patient_fold_plan(
        fold_plan, output / "patient_fold_plan.csv", seed=seed, n_folds=5
    )
    (output / "training_config.json").write_text(
        json.dumps(base_config, indent=2), encoding="utf-8"
    )

    summaries: dict[str, Any] = {}
    rows = []
    for model_name in selected:
        model_dir = output / model_name
        summary = _load_completed_summary(model_dir)
        if summary is None:
            config = copy.deepcopy(base_config)
            config["model_name"] = model_name
            summary = run_cross_validation(
                arrays,
                schema,
                config,
                model_dir,
                fold_assignments=fold_plan,
            )
        if (
            summary["cv_protocol"]["fold_plan_checksum"]
            != fold_manifest["checksum_sha256"]
        ):
            raise RuntimeError(f"{model_name} did not use the locked fold plan")
        summaries[model_name] = summary
        parameter_values = [
            record["parameter_count"]
            for record in summary["fold_records"]
            if record["parameter_count"] is not None
        ]
        rows.append(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "parameter_count": (
                    int(np.median(parameter_values))
                    if parameter_values
                    else None
                ),
            }
            | summary["pooled_oof_metrics"]
            | summary["pooled_change_metrics"]
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "v2_quick_metrics.csv", index=False)
    manifest = {
        "status": "synthetic_v2_quick_seed_complete_non_clinical",
        "formal_clinical_conclusions_generated": False,
        "seed": seed,
        "patient_count": len(patient_ids),
        "protocol": "patient-level five-fold; four folds train and one untouched fold tests",
        "same_patient_folds_for_all_models": True,
        "internal_data_only": True,
        "external_center_used": False,
        "external_data_used_for_hyperparameter_selection": False,
        "config_sha256": _hash_config(base_config),
        "fold_plan": fold_manifest,
        "models": list(selected),
        "supporting_probe": {
            "lac_v2_forward_only": (
                "Included in addition to the requested priority table so the "
                "weak C-to-I contribution can be isolated."
            )
        },
        "summaries": summaries,
    }
    (output / "v2_quick_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _comparison(
    by_seed: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    question: str,
) -> dict[str, Any]:
    candidate_rows = by_seed.loc[
        by_seed["model"] == candidate, ["seed", metric]
    ]
    reference_rows = by_seed.loc[
        by_seed["model"] == reference, ["seed", metric]
    ]
    paired = candidate_rows.merge(
        reference_rows, on="seed", suffixes=("_candidate", "_reference")
    )
    delta = (
        paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
    ).to_numpy()
    return {
        "question": question,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "delta_definition": "candidate_minus_reference",
        "mean_delta": float(np.mean(delta)),
        "standard_deviation": (
            float(np.std(delta, ddof=1)) if len(delta) > 1 else float("nan")
        ),
        "seeds_positive": int(np.sum(delta > 0)),
        "seeds_negative": int(np.sum(delta < 0)),
        "lower_is_better": metric.endswith(("mae", "rmse")),
    }


def aggregate_v2_quick_comparison(
    seed_directories: dict[int, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Aggregate exactly the completed development results; no external data."""
    if len(seed_directories) < 3:
        raise ValueError("V2 quick comparison requires at least three seeds")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    manifests = {}
    for seed, directory_value in seed_directories.items():
        directory = Path(directory_value)
        manifest = json.loads(
            (directory / "v2_quick_manifest.json").read_text(encoding="utf-8")
        )
        if int(manifest["seed"]) != int(seed):
            raise ValueError("Seed directory and manifest disagree")
        frame = pd.read_csv(directory / "v2_quick_metrics.csv")
        frame.insert(0, "seed", int(seed))
        frames.append(frame)
        manifests[str(seed)] = str(directory / "v2_quick_manifest.json")
    by_seed = pd.concat(frames, ignore_index=True)
    by_seed.to_csv(output / "metrics_by_seed.csv", index=False)
    numeric = [
        column
        for column in ("parameter_count",) + ENDPOINT_METRICS + CHANGE_METRICS
        if column in by_seed
    ]
    mean_std = (
        by_seed.groupby(["model", "display_name"], sort=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_std.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in mean_std.columns
    ]
    mean_std.to_csv(output / "metrics_mean_std.csv", index=False)

    checks = [
        _comparison(
            by_seed,
            "lac_v1_no_competitive_routing",
            "lac_itransformer",
            endpoint,
            "Does removing V1 competitive routing improve performance?",
        )
        for endpoint in ("tbr_mae", "log_cac_mae")
    ]
    checks += [
        _comparison(
            by_seed,
            "lac_v2",
            "lac_itransformer",
            endpoint,
            "Does the complete lighter V2 improve over V1 Full?",
        )
        for endpoint in ("tbr_mae", "log_cac_mae")
    ]
    checks += [
        _comparison(
            by_seed,
            "lac_v2_no_adapters",
            "lac_v2",
            endpoint,
            "Do non-exclusive phenotype adapters add predictive value?",
        )
        for endpoint in ("tbr_mae", "log_cac_mae")
    ]
    checks.append(
        _comparison(
            by_seed,
            "lac_v2_no_coupling",
            "lac_v2_forward_only",
            "log_cac_mae",
            "Does the isolated I-to-C lag path improve CAC prediction?",
        )
    )
    checks.append(
        _comparison(
            by_seed,
            "lac_v2_forward_only",
            "lac_v2",
            "tbr_mae",
            "Does removing the already weak C-to-I path reduce TBR performance?",
        )
    )
    checks.append(
        _comparison(
            by_seed,
            "lac_v2_no_treatment",
            "lac_v2",
            "log_cac_mae",
            "Does treatment conditioning improve CAC prediction?",
        )
    )
    checks.append(
        _comparison(
            by_seed,
            "lac_v2_symmetric",
            "lac_v2",
            "log_cac_mae",
            "Is asymmetric capacity preferable to symmetric coupling for CAC?",
        )
    )
    mechanism = pd.DataFrame(checks)
    mechanism.to_csv(output / "mechanism_diagnostics.csv", index=False)

    parameter_lookup = (
        by_seed.groupby("model")["parameter_count"].median().to_dict()
    )
    v1_parameters = float(parameter_lookup["lac_itransformer"])
    v2_parameters = float(parameter_lookup["lac_v2"])
    result = {
        "status": "synthetic_v2_quick_comparison_complete_non_clinical",
        "formal_clinical_conclusions_generated": False,
        "seeds": sorted(int(seed) for seed in seed_directories),
        "seed_count": len(seed_directories),
        "patient_level_fivefold": True,
        "development_internal_data_only": True,
        "external_center_used": False,
        "external_data_used_for_model_selection": False,
        "parameter_count": {
            "v1_full": int(v1_parameters),
            "v2_full": int(v2_parameters),
            "v2_minus_v1": int(v2_parameters - v1_parameters),
            "v2_fraction_of_v1": v2_parameters / v1_parameters,
        },
        "model_order": list(V2_QUICK_MODELS),
        "seed_manifests": manifests,
        "interpretation_guardrail": (
            "Rapid synthetic diagnostics assess implementation and mechanism "
            "recovery only; they are not clinical validation."
        ),
    }
    (output / "v2_quick_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
