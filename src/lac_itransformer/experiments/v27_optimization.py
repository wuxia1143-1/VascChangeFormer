from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_yaml
from ..training.folds import (
    fold_plan_checksum,
    validate_patient_fold_plan,
)
from ..training.v27_nested import (
    V27_MODELS,
    run_v27_nested_cross_validation,
)
from .real_internal import load_prepared_real_cohort


def _load_locked_plan(
    path: str | Path,
    patient_ids,
) -> dict[str, int]:
    frame = pd.read_csv(path)
    plan = {
        str(row.patient_id): int(row.test_fold)
        for row in frame.itertuples(index=False)
    }
    validate_patient_fold_plan(patient_ids, plan, 5)
    return plan


def run_v27_model(
    prepared_path: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    model_name: str,
    locked_plan_path: str | Path,
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    if model_name not in V27_MODELS:
        raise KeyError(model_name)
    arrays, schema = load_prepared_real_cohort(prepared_path)
    config = load_yaml(config_path)
    plan = _load_locked_plan(locked_plan_path, arrays["patient_ids"])
    expected_checksum = str(
        config.get(
            "locked_outer_fold_checksum",
            fold_plan_checksum(plan),
        )
    )
    if fold_plan_checksum(plan) != expected_checksum:
        raise RuntimeError("Locked outer patient split checksum changed")
    suffix = (
        model_name
        if tuple(outer_folds) == (1, 2, 3, 4, 5)
        else f"{model_name}_smoke_folds_{'-'.join(map(str, outer_folds))}"
    )
    output = Path(output_root) / suffix
    summary_path = output / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial output will not be overwritten: {output}")
    return run_v27_nested_cross_validation(
        arrays,
        schema,
        config,
        output,
        model_name,
        outer_fold_assignments=plan,
        outer_folds=outer_folds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--locked-plan", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lac_v27_full"],
    )
    parser.add_argument("--outer-folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    arguments = parser.parse_args()
    summaries = {}
    for model_name in arguments.models:
        summaries[model_name] = run_v27_model(
            arguments.data,
            arguments.config,
            arguments.output,
            model_name,
            arguments.locked_plan,
            tuple(arguments.outer_folds),
        )
    print(
        json.dumps(
            {
                model: {
                    "patient_count": summary["patient_count"],
                    "pooled_oof_metrics": summary["pooled_oof_metrics"],
                    "pooled_change_metrics": summary[
                        "pooled_change_metrics"
                    ],
                }
                for model, summary in summaries.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
