"""Extract native outputs from frozen runs and evaluate them without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ["persistence", "elastic_net", "xgboost", "apn_dr", "itransformer_mtl",
             "first_icu_mtl", "learning_to_route"]


def run(name: str, *args: object) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], check=True)


def jobs_for_run(root: Path, development: str, models: list[str]) -> list[dict]:
    jobs = []
    n = 323 if development == "A" else 546
    experiment = "primary" if development == "A" else "reciprocal"
    order = root / "protocol_lock" / "A323_locked_patient_outer_fold_plan.csv" if development == "A" else root / "prepared_data" / "C546_training_arrays.npz"
    for model in models:
        for fold in range(1, 6):
            job = {"key": f"{experiment}_{development}_{model}_{fold}", "experiment": experiment,
                   "development": development, "centre": development, "model": model,
                   "fold": fold, "folder": str(root / "models" / model / f"fold_{fold}"),
                   "arrays": str(root / "prepared_data" / f"{development}{n}_training_arrays.npz"),
                   "reference": str(root / "models" / model / "out_of_fold_predictions.csv"),
                   "order": str(order)}
            if development == "A" and model == "vascmtl":
                job["truth_policy"] = "archived_change_columns"
            jobs.append(job)
        for centre in (["B", "C"] if development == "A" else ["A", "B"]):
            size = {"A": 323, "B": 31, "C": 546}[centre]
            prefix = f"{centre}{size}"
            if development == "C":
                reference = root / "external_validation" / centre / f"C546_to_{prefix}_predictions_blinded.csv"
                columns = ["pred_tbr", "pred_tac"]
            else:
                directory = "external_validation" if centre == "B" else "external_validation_c546"
                filename = "external_predictions_blinded.csv" if centre == "B" else "C546_external_predictions_blinded.csv"
                reference = root / directory / filename
                columns = [f"pred_{model}_tbr", f"pred_{model}_{'cac' if centre == 'B' else 'tac'}"]
            job = {"key": f"{experiment}_{centre}_{model}", "experiment": experiment,
                   "development": development, "centre": centre, "model": model,
                   "folder": str(root / "frozen_models" / model),
                   "arrays": str(root / "prepared_data" / f"{prefix}_predictors_without_outcomes.npz"),
                   "outcomes": str(root / "prepared_data" / f"{prefix}_sealed_outcomes.npz"),
                   "reference": str(reference), "reference_columns": columns,
                   "order": str(root / "prepared_data" / f"{prefix}_sealed_outcomes.npz")}
            if development == "A" and centre == "B" and model == "vascmtl":
                job["truth_policy"] = "b31_legacy_baseline_log_float32"
            jobs.append(job)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-root", type=Path, required=True)
    parser.add_argument("--c-root", type=Path)
    parser.add_argument("--models", nargs="+", default=["vascmtl"], choices=["vascmtl", *BASELINES])
    parser.add_argument("--source", type=Path, default=ROOT / "src")
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--ablation-source", type=Path)
    parser.add_argument("--endpoint-ablation-root", type=Path)
    parser.add_argument("--endpoint-source", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty --output directory")
    if args.endpoint_ablation_root and not args.endpoint_source:
        parser.error("The endpoint ablation requires its original --endpoint-source snapshot")
    a_root = args.a_root.resolve()
    groups = [(args.source, jobs_for_run(a_root, "A", args.models))]
    if args.c_root:
        groups[0][1].extend(jobs_for_run(args.c_root.resolve(), "C", ["vascmtl"]))
    for ablation_root, names, source in (
        (args.ablation_root, ["vascmtl_no_baseline_anchoring", "vascmtl_no_cac_calibration"], args.ablation_source or args.source),
        (args.endpoint_ablation_root, ["vasctimeformer_direct_endpoint_prediction"], args.endpoint_source),
    ):
        if not ablation_root:
            continue
        jobs = []
        for model in names:
            for fold in range(1, 6):
                job = {"key": f"ablation_{model}_{fold}", "experiment": "ablation", "development": "A",
                       "centre": "A", "model": model, "fold": fold,
                       "folder": str(ablation_root.resolve() / "models" / model / f"fold_{fold}"),
                       "arrays": str(a_root / "prepared_data" / "A323_training_arrays.npz"),
                       "reference": str(ablation_root.resolve() / "models" / model / "out_of_fold_predictions.csv"),
                       "order": str(a_root / "protocol_lock" / "A323_locked_patient_outer_fold_plan.csv")}
                if "direct_endpoint" in model:
                    job["public_model"] = "direct_endpoint_prediction"
                jobs.append(job)
        groups.append((source, jobs))
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = args.output / "extracted"
    for number, (source, jobs) in enumerate(groups):
        jobfile = args.output / f"jobs_{number}.json"
        jobfile.write_text(json.dumps(jobs, indent=2) + "\n")
        run("paper_direct_extract.py", "--source", source.resolve(), "--jobs", jobfile,
            "--output", extracted, "--device", args.device)
    run("evaluate_direct_changes.py", "--predictions", *sorted(extracted.glob("*.csv")),
        "--output", args.output / "metrics")


if __name__ == "__main__":
    main()
