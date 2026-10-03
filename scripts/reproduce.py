"""Run the published A323/C546 workflow using private, locally held inputs."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ["persistence", "elastic_net", "xgboost", "apn_dr",
             "itransformer_mtl", "first_icu_mtl", "learning_to_route"]


def run(script: str, *args: object) -> None:
    # Arguments can include a private cohort identifier: do not echo commands.
    print(f"Running {script}", flush=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUTF8="1")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                    *map(str, args)], cwd=ROOT, env=env, check=True)


def lock_hash(root: Path) -> str:
    return hashlib.sha256((root / "protocol_lock" / "protocol_lock.json").read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", choices=["a", "c", "all"], default="all")
    parser.add_argument("--private-data", type=Path, default=ROOT / "private_data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--with-baselines", action="store_true", help="Also run the seven A323 comparators")
    args = parser.parse_args()
    private = args.private_data.resolve()
    output = args.output_root.resolve()
    a_root, c_root = output / "A323", output / "C546"
    required = [private / "C547_prepared_arrays.npz", private / "C_excluded_id.txt"]
    if args.development in ("a", "all"):
        required += [private / "A354_prepared_arrays.npz",
                     private / "A354_patient_outer_fold_plan.csv", private / "B31_ids.txt"]
        if a_root.exists() and any(a_root.iterdir()):
            parser.error("A323 output is not empty. Use a new --output-root or resume with individual stage scripts.")
    if args.development in ("c", "all") and c_root.exists() and any(c_root.iterdir()):
        parser.error("C546 output is not empty. Use a new --output-root or resume with individual stage scripts.")
    if args.development == "c":
        required += [a_root / "prepared_data" / name for name in (
            "A323_training_arrays.npz", "B31_predictors_without_outcomes.npz", "B31_sealed_outcomes.npz")]
        required.append(a_root / "protocol_lock" / "locked_configs" / "vascmtl.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing private inputs (see docs/DATA_CONTRACT.md):\n" + "\n".join(missing))
    excluded = [line.strip() for line in (private / "C_excluded_id.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(excluded) != 1:
        parser.error("C_excluded_id.txt must contain exactly one identifier")
    models = ["vascmtl"] + (BASELINES if args.with_baselines else [])
    output.mkdir(parents=True, exist_ok=True)
    if args.development in ("a", "all"):
        common = ("--experiment-root", a_root)
        run("run_a323_b31_internal_comparison.py", "prepare", *common,
            "--source-arrays", private / "A354_prepared_arrays.npz",
            "--source-plan", private / "A354_patient_outer_fold_plan.csv",
            "--b31-ids-file", private / "B31_ids.txt", "--device", args.device)
        lock = lock_hash(a_root)
        for model in models:
            run("run_a323_b31_internal_comparison.py", "run", *common,
                "--expected-lock-sha256", lock, "--model", model)
        run("run_a323_b31_internal_comparison.py", "aggregate", *common,
            "--expected-lock-sha256", lock, "--models", *models)
        # Calibration must precede full refit and external point prediction.
        run("run_a323_b31_external_validation.py", "calibrate", *common, "--models", *models)
        run("run_a323_full_refit.py", *common, "--device", args.device, "--models", *models)
        for stage in ("predict", "evaluate"):
            run("run_a323_b31_external_validation.py", stage, *common,
                "--device", args.device, "--models", *models)
        run("run_a323_c546_external_validation.py", "prepare", *common,
            "--source-c547", private / "C547_prepared_arrays.npz", "--exclude-c-id", excluded[0])
        for stage in ("predict", "evaluate", "audit"):
            run("run_a323_c546_external_validation.py", stage, *common,
                "--device", args.device, "--models", *models)
    if args.development in ("c", "all"):
        common = ("--experiment-root", c_root)
        run("run_c546_development_external_ab.py", "prepare", *common,
            "--source-c547", private / "C547_prepared_arrays.npz",
            "--source-a323", a_root / "prepared_data" / "A323_training_arrays.npz",
            "--source-b31-predictors", a_root / "prepared_data" / "B31_predictors_without_outcomes.npz",
            "--source-b31-outcomes", a_root / "prepared_data" / "B31_sealed_outcomes.npz",
            "--source-vascmtl-config", a_root / "protocol_lock" / "locked_configs" / "vascmtl.json",
            "--exclude-c-id", excluded[0], "--device", args.device)
        lock = lock_hash(c_root)
        for stage in ("run", "aggregate", "refit", "predict", "evaluate"):
            run("run_c546_development_external_ab.py", stage, *common,
                "--expected-lock-sha256", lock, "--device", args.device)
    print("Workflow complete. Outputs and patient-level artifacts remain in the local output directory.")


if __name__ == "__main__":
    main()
