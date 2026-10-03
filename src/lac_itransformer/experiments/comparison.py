from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..data.schema import FeatureSchema
from ..training.folds import build_patient_fold_plan, save_patient_fold_plan
from ..training.trainer import run_cross_validation
from .prediction import model_registry


PRESPECIFIED_MODELS = (
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
    "lac_itransformer",
)


def run_comparison_suite(
    arrays: dict,
    schema: FeatureSchema,
    base_config: dict[str, Any],
    output_dir: str | Path,
    models: Iterable[str] = PRESPECIFIED_MODELS,
) -> dict[str, Any]:
    """Run all comparators with identical patient folds and preprocessing."""
    selected = tuple(models)
    registry = model_registry()
    unknown = set(selected) - set(registry)
    if unknown:
        raise KeyError(f"Unknown comparison models: {sorted(unknown)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(base_config.get("seed", 2026))
    n_folds = int(base_config.get("num_folds", 5))
    fold_plan = build_patient_fold_plan(arrays["patient_ids"], n_folds, seed)
    fold_manifest = save_patient_fold_plan(
        fold_plan, output / "patient_fold_plan.csv", seed, n_folds
    )
    summaries: dict[str, Any] = {}
    rows = []
    for name in selected:
        config = copy.deepcopy(base_config)
        config["model_name"] = name
        summary = run_cross_validation(
            arrays, schema, config, output / name, fold_assignments=fold_plan
        )
        summaries[name] = summary
        rows.append({"model": name} | summary["mean_metrics"])
    pd.DataFrame(rows).to_csv(output / "comparison_metrics.csv", index=False)
    manifest = {
        "status": "synthetic_smoke_only" if str(base_config.get("result_scope", "")).startswith("synthetic") else "development_only",
        "same_patient_folds": True,
        "fold_plan": fold_manifest,
        "same_fold_fitted_preprocessing": True,
        "external_center_used": False,
        "models": selected,
        "summaries": summaries,
    }
    (output / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
