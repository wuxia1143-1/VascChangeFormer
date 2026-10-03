from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_yaml
from .data.external_validation import ExternalWorkbookReader
from .data.synthetic import SyntheticCohortGenerator, load_npz
from .experiments.protocol import build_experiment_plan
from .experiments.comparison import PRESPECIFIED_MODELS, run_comparison_suite
from .experiments.internal_cv import INTERNAL_FIVEFOLD_MODELS, run_internal_fivefold_suite
from .experiments.core_ablation import CORE_ABLATION_VARIANTS, run_core_ablation_suite
from .training.external import evaluate_external_workbook
from .training.external_all import (
    evaluate_external_arrays_all_models,
    evaluate_external_workbook_all_models,
    file_sha256,
    finalize_models_for_external_validation,
)
from .training.trainer import run_cross_validation


def _write_json(value: dict, path: str | Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lac-it")
    sub = parser.add_subparsers(dest="command", required=True)

    synthetic = sub.add_parser("synthetic", help="Generate privacy-preserving synthetic unit-test data")
    synthetic.add_argument("--profile", required=True)
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--n-patients", type=int, default=443)
    synthetic.add_argument("--seed", type=int, default=2026)

    train = sub.add_parser("train", help="Run patient-level cross-validation on a development cohort")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)

    compare = sub.add_parser("compare", help="Run the prespecified comparator suite on development data")
    compare.add_argument("--config", required=True)
    compare.add_argument("--data", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--models", nargs="+", default=list(PRESPECIFIED_MODELS))

    internal_cv = sub.add_parser(
        "internal-cv", help="Run locked patient-level 5-fold validation for all models"
    )
    internal_cv.add_argument("--config", required=True)
    internal_cv.add_argument("--data", required=True)
    internal_cv.add_argument("--output", required=True)
    internal_cv.add_argument(
        "--models", nargs="+", default=list(INTERNAL_FIVEFOLD_MODELS)
    )

    ablation = sub.add_parser(
        "core-ablation",
        help="Retrain core LAC ablations in the comparison experiment's locked five folds",
    )
    ablation.add_argument("--data", required=True)
    ablation.add_argument("--comparison-dir", required=True)
    ablation.add_argument("--output", required=True)
    ablation.add_argument(
        "--variants", nargs="+", default=list(CORE_ABLATION_VARIANTS)
    )
    ablation.add_argument("--bootstrap", type=int)
    ablation.add_argument("--device")

    audit = sub.add_parser("audit-external", help="Audit the legacy external workbook without fitting")
    audit.add_argument("--workbook", required=True)
    audit.add_argument("--output")

    external = sub.add_parser("external-evaluate", help="Run locked-fold external validation")
    external.add_argument("--workbook", required=True)
    external.add_argument("--model-dir", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--device", default="auto")
    external.add_argument("--bootstrap", type=int, default=1000)

    finalize_external = sub.add_parser(
        "external-finalize",
        help="Freeze center-B five-fold choices and refit all models on full center B",
    )
    finalize_external.add_argument("--development-data", required=True)
    finalize_external.add_argument("--internal-cv", required=True)
    finalize_external.add_argument("--output", required=True)
    finalize_external.add_argument("--device", default="auto")
    finalize_external.add_argument(
        "--models", nargs="+", default=list(PRESPECIFIED_MODELS)
    )

    validate_all = sub.add_parser(
        "external-validate-all",
        help="Evaluate all locked full-center-B models once on external center A",
    )
    validate_all.add_argument("--bundle", required=True)
    validate_all.add_argument("--output", required=True)
    source = validate_all.add_mutually_exclusive_group(required=True)
    source.add_argument("--workbook")
    source.add_argument("--external-data")
    validate_all.add_argument("--center-name", default="center_A")
    validate_all.add_argument("--device", default="auto")
    validate_all.add_argument("--bootstrap", type=int, default=2000)

    plan = sub.add_parser("experiment-plan", help="Write the prespecified experiment manifest")
    plan.add_argument("--output")

    args = parser.parse_args()
    if args.command == "synthetic":
        path = SyntheticCohortGenerator.from_json(args.profile, seed=args.seed).save(args.output, args.n_patients)
        _write_json({"output": str(path), "n_patients": args.n_patients, "formal_result": False}, None)
    elif args.command == "train":
        arrays, schema = load_npz(args.data)
        _write_json(run_cross_validation(arrays, schema, load_yaml(args.config), args.output), None)
    elif args.command == "compare":
        arrays, schema = load_npz(args.data)
        _write_json(run_comparison_suite(arrays, schema, load_yaml(args.config), args.output, args.models), None)
    elif args.command == "internal-cv":
        arrays, schema = load_npz(args.data)
        _write_json(
            run_internal_fivefold_suite(
                arrays, schema, load_yaml(args.config), args.output, args.models
            ),
            None,
        )
    elif args.command == "core-ablation":
        arrays, schema = load_npz(args.data)
        _write_json(
            run_core_ablation_suite(
                arrays,
                schema,
                args.comparison_dir,
                args.output,
                variants=args.variants,
                bootstrap_replicates=args.bootstrap,
                device_name=args.device,
            ),
            None,
        )
    elif args.command == "audit-external":
        _write_json(ExternalWorkbookReader().audit(args.workbook).to_dict(), args.output)
    elif args.command == "external-evaluate":
        _write_json(evaluate_external_workbook(args.workbook, args.model_dir, args.output, args.device, args.bootstrap), None)
    elif args.command == "external-finalize":
        arrays, schema = load_npz(args.development_data)
        _write_json(
            finalize_models_for_external_validation(
                arrays,
                schema,
                args.internal_cv,
                args.output,
                models=args.models,
                device_name=args.device,
            ),
            None,
        )
    elif args.command == "external-validate-all":
        if args.workbook:
            result = evaluate_external_workbook_all_models(
                args.workbook,
                args.bundle,
                args.output,
                device_name=args.device,
                n_bootstrap=args.bootstrap,
                center_name=args.center_name,
            )
        else:
            arrays, schema = load_npz(args.external_data)
            result = evaluate_external_arrays_all_models(
                arrays,
                schema,
                args.bundle,
                args.output,
                device_name=args.device,
                n_bootstrap=args.bootstrap,
                center_name=args.center_name,
                source_type="standardized_npz",
                source_sha256=file_sha256(args.external_data),
            )
        _write_json(result, None)
    elif args.command == "experiment-plan":
        _write_json(build_experiment_plan(), args.output)


if __name__ == "__main__":
    main()
