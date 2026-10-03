from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import load_yaml
from ..training.folds import fold_plan_checksum, validate_patient_fold_plan
from ..training.v32_nested import V32_MODELS, run_v32_nested_cross_validation
from .real_internal import load_prepared_real_cohort


def _locked_plan(path, patient_ids):
    frame = pd.read_csv(path)
    plan = {
        str(row.patient_id): int(row.test_fold)
        for row in frame.itertuples(index=False)
    }
    validate_patient_fold_plan(patient_ids, plan, 5)
    return plan


def run_v32_model(
    data,
    config_path,
    output_root,
    model_name,
    locked_plan,
    outer_folds=(1, 2, 3, 4, 5),
    training_seed_offset=0,
):
    arrays, schema = load_prepared_real_cohort(data)
    config = load_yaml(config_path)
    config["training_seed_offset"] = int(training_seed_offset)
    plan = _locked_plan(locked_plan, arrays["patient_ids"])
    if fold_plan_checksum(plan) != str(config["locked_outer_fold_checksum"]):
        raise RuntimeError("Locked outer patient split checksum changed")
    complete = tuple(outer_folds) == (1, 2, 3, 4, 5)
    suffix = (
        model_name if complete
        else f"{model_name}_smoke_folds_{'-'.join(map(str, outer_folds))}"
    )
    output = Path(output_root) / suffix
    if (output / "summary.json").is_file():
        return json.loads((output / "summary.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial output will not be overwritten: {output}")
    return run_v32_nested_cross_validation(
        arrays, schema, config, output, model_name,
        outer_fold_assignments=plan, outer_folds=tuple(outer_folds),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--locked-plan", required=True)
    parser.add_argument("--models", nargs="+", default=["lac_v32_full"])
    parser.add_argument("--outer-folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--training-seed-offset", type=int, default=0)
    args = parser.parse_args()
    result = {}
    for name in args.models:
        if name not in V32_MODELS:
            raise KeyError(name)
        result[name] = run_v32_model(
            args.data, args.config, args.output, name, args.locked_plan,
            tuple(args.outer_folds), args.training_seed_offset,
        )
    print(json.dumps({
        name: value["pooled_change_metrics"] for name, value in result.items()
    }, indent=2))


if __name__ == "__main__":
    main()
