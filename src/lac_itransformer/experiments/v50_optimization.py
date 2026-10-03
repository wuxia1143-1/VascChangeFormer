from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import load_yaml
from ..training.folds import fold_plan_checksum, validate_patient_fold_plan
from ..training.v50_nested import V50_MODELS, run_v50_nested_cross_validation
from ..training.v50_uncertainty import (
    fit_detached_conformal_sidecar,
    save_detached_conformal_sidecar,
)
from .real_internal import load_prepared_real_cohort


def _locked_plan(path, patient_ids):
    frame = pd.read_csv(path)
    plan = {
        str(row.patient_id): int(row.test_fold)
        for row in frame.itertuples(index=False)
    }
    validate_patient_fold_plan(patient_ids, plan, 5)
    return plan


def _attach_internal_uncertainty(output, arrays, config, summary):
    output = Path(output)
    oof_path = output / "out_of_fold_predictions.csv"
    uncertainty_path = output / "uncertainty_summary.json"
    interval_path = output / "out_of_fold_prediction_intervals.csv"
    sidecar_path = output / "uncertainty_sidecar.pkl"
    if (
        uncertainty_path.is_file()
        and interval_path.is_file()
        and sidecar_path.is_file()
    ):
        uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    else:
        oof = pd.read_csv(oof_path, dtype={"patient_id": str})
        sidecar, interval_oof, uncertainty = fit_detached_conformal_sidecar(
            oof,
            arrays,
            options=dict(config.get("uncertainty", {})),
            seed=int(config.get("seed", 2026)),
        )
        interval_oof.to_csv(interval_path, index=False)
        uncertainty["artifact_manifest"] = save_detached_conformal_sidecar(
            sidecar,
            sidecar_path,
            source_oof_path=oof_path,
        )
        uncertainty_path.write_text(
            json.dumps(uncertainty, indent=2), encoding="utf-8"
        )
    summary["uncertainty"] = uncertainty
    summary["cv_protocol"]["uncertainty_fit_source"] = (
        "outer-OOF point predictions with fold-cross-fitted scale estimates"
    )
    summary["cv_protocol"]["uncertainty_can_modify_point_prediction"] = False
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_v50_model(
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
        model_name
        if complete
        else f"{model_name}_smoke_folds_{'-'.join(map(str, outer_folds))}"
    )
    output = Path(output_root) / suffix
    if (output / "summary.json").is_file():
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8")
        )
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Partial output will not be overwritten: {output}")
        summary = run_v50_nested_cross_validation(
            arrays,
            schema,
            config,
            output,
            model_name,
            outer_fold_assignments=plan,
            outer_folds=tuple(outer_folds),
        )
    if complete and model_name == "lac_v50_full":
        summary = _attach_internal_uncertainty(output, arrays, config, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--locked-plan", required=True)
    parser.add_argument("--models", nargs="+", default=list(V50_MODELS))
    parser.add_argument(
        "--outer-folds", nargs="+", type=int, default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--training-seed-offset", type=int, default=0)
    args = parser.parse_args()
    result = {}
    for name in args.models:
        if name not in V50_MODELS:
            raise KeyError(name)
        result[name] = run_v50_model(
            args.data,
            args.config,
            args.output,
            name,
            args.locked_plan,
            tuple(args.outer_folds),
            args.training_seed_offset,
        )
    print(
        json.dumps(
            {name: value["pooled_change_metrics"] for name, value in result.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
