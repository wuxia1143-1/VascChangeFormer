from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..training.metrics import (
    bootstrap_change_metrics,
    bootstrap_metrics,
    change_space_metrics,
    regression_metrics,
)
from ..training.v27_nested import _v27_diagnostics


TRAINING_SOURCE_PATHS = (
    "configs/real_internal_v40_final_nested_fivefold.yaml",
    "lac_itransformer/experiments/prediction.py",
    "lac_itransformer/experiments/v40_final_optimization.py",
    "lac_itransformer/models/lac_v40_final.py",
    "lac_itransformer/models/lac_v41.py",
    "lac_itransformer/training/v27_nested.py",
    "lac_itransformer/training/v40_final_nested.py",
    "lac_itransformer/training/v40_final_selector.py",
    "lac_itransformer/training/v41_trainer.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source_stability(
    source_revision: str, observed_revisions: set[str]
) -> dict[str, Any]:
    checks = []
    for revision in sorted(observed_revisions):
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                source_revision,
                revision,
                "--",
                *TRAINING_SOURCE_PATHS,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Core training sources changed between declared source "
                f"{source_revision} and observed revision {revision}"
            )
        checks.append(
            {
                "observed_revision": revision,
                "training_sources_identical_to_declared_revision": True,
            }
        )
    return {
        "declared_training_source_revision": source_revision,
        "verified_paths": list(TRAINING_SOURCE_PATHS),
        "revision_checks": checks,
    }


def combine_nested_fold_outputs(
    part_directories: list[str | Path],
    output_directory: str | Path,
    expected_folds: tuple[int, ...] = (1, 2, 3, 4, 5),
    training_source_revision: str | None = None,
) -> dict[str, Any]:
    parts = [Path(path).resolve() for path in part_directories]
    output = Path(output_directory).resolve()
    if len(parts) < 2:
        raise ValueError("At least two partial result directories are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    required = (
        "summary.json",
        "out_of_fold_predictions.csv",
        "outer_fold_metrics.csv",
        "patient_outer_fold_plan.csv",
        "patient_outer_fold_plan.json",
        "training_config.json",
    )
    for part in parts:
        if not part.is_dir():
            raise FileNotFoundError(part)
        for name in required:
            if not (part / name).is_file():
                raise FileNotFoundError(part / name)

    summaries = [_read_json(part / "summary.json") for part in parts]
    model_names = {str(summary["model_name"]) for summary in summaries}
    revisions = {str(summary["git_revision"]) for summary in summaries}
    plan_checksums = {
        str(summary["cv_protocol"]["outer_fold_plan_checksum"])
        for summary in summaries
    }
    if len(model_names) != 1:
        raise RuntimeError(f"Model names differ across parts: {model_names}")
    if len(plan_checksums) != 1:
        raise RuntimeError("Outer-fold plan checksums differ across parts")

    reference_plan = (parts[0] / "patient_outer_fold_plan.csv").read_bytes()
    reference_config = (parts[0] / "training_config.json").read_bytes()
    for part in parts[1:]:
        if (part / "patient_outer_fold_plan.csv").read_bytes() != reference_plan:
            raise RuntimeError("Patient outer-fold plans differ across parts")
        if (part / "training_config.json").read_bytes() != reference_config:
            raise RuntimeError("Training configurations differ across parts")

    predictions = []
    fold_metrics = []
    fold_records: list[dict[str, Any]] = []
    seen_folds: set[int] = set()
    source_files: list[dict[str, Any]] = []
    checkpoint_revisions: set[str] = set()
    for part, summary in zip(parts, summaries):
        part_predictions = pd.read_csv(part / "out_of_fold_predictions.csv")
        part_metrics = pd.read_csv(part / "outer_fold_metrics.csv")
        folds = {int(value) for value in part_metrics["outer_fold"].tolist()}
        overlap = folds & seen_folds
        if overlap:
            raise RuntimeError(f"Duplicate evaluated folds: {sorted(overlap)}")
        seen_folds |= folds
        if set(part_predictions["outer_fold"].astype(int).unique()) != folds:
            raise RuntimeError(f"Prediction/metric fold mismatch in {part}")
        predictions.append(part_predictions)
        fold_metrics.append(part_metrics)
        fold_records.extend(summary["fold_records"])
        source_files.append(
            {
                "directory": str(part),
                "folds": sorted(folds),
                "summary_sha256": _sha256(part / "summary.json"),
                "oof_sha256": _sha256(part / "out_of_fold_predictions.csv"),
            }
        )
        for fold in folds:
            checkpoint = torch.load(
                part / f"fold_{fold}" / "model.pt",
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("model_name") != summary["model_name"]:
                raise RuntimeError(f"Checkpoint model mismatch in {part}/fold_{fold}")
            checkpoint_revisions.add(str(checkpoint.get("git_revision", "unknown")))

    observed_revisions = revisions | checkpoint_revisions
    if len(observed_revisions) == 1:
        source_revision = next(iter(observed_revisions))
        source_stability = {
            "declared_training_source_revision": source_revision,
            "verified_paths": list(TRAINING_SOURCE_PATHS),
            "revision_checks": [],
        }
    else:
        if training_source_revision is None:
            raise RuntimeError(
                "Source revisions differ across partial results; provide the "
                "declared training source revision for an exact git-diff audit: "
                f"{sorted(observed_revisions)}"
            )
        source_revision = str(training_source_revision)
        source_stability = _verify_source_stability(
            source_revision, observed_revisions
        )

    expected = set(expected_folds)
    if seen_folds != expected:
        raise RuntimeError(
            f"Expected folds {sorted(expected)}, received {sorted(seen_folds)}"
        )
    oof = pd.concat(predictions, ignore_index=True)
    if oof["patient_id"].duplicated().any():
        duplicated = oof.loc[oof["patient_id"].duplicated(), "patient_id"].tolist()
        raise RuntimeError(f"Duplicate OOF patient identifiers: {duplicated[:5]}")
    plan = pd.read_csv(parts[0] / "patient_outer_fold_plan.csv")
    if len(oof) != len(plan) or set(oof["patient_id"].astype(str)) != set(
        plan["patient_id"].astype(str)
    ):
        raise RuntimeError("Combined OOF patients do not exactly match locked plan")

    output.mkdir(parents=True)
    try:
        for name in (
            "patient_outer_fold_plan.csv",
            "patient_outer_fold_plan.json",
            "training_config.json",
        ):
            shutil.copy2(parts[0] / name, output / name)
        for part, summary in zip(parts, summaries):
            for fold in summary["cv_protocol"]["outer_folds_evaluated"]:
                shutil.copytree(part / f"fold_{int(fold)}", output / f"fold_{int(fold)}")

        oof = oof.sort_values("patient_id").reset_index(drop=True)
        metrics = (
            pd.concat(fold_metrics, ignore_index=True)
            .sort_values("outer_fold")
            .reset_index(drop=True)
        )
        oof.to_csv(output / "out_of_fold_predictions.csv", index=False)
        metrics.to_csv(output / "outer_fold_metrics.csv", index=False)

        targets = oof[["true_tbr", "true_cac"]].to_numpy()
        prediction = oof[["pred_tbr", "pred_cac"]].to_numpy()
        baseline = oof[["baseline_tbr", "baseline_cac"]].to_numpy()
        config = _read_json(output / "training_config.json")
        replicates = int(config.get("bootstrap_replicates", 2000))
        seed = int(config.get("seed", 2026))
        combined = dict(summaries[0])
        combined.update(
            {
                "git_revision": source_revision,
                "patient_count": len(oof),
                "parameter_count": int(np.median(metrics["parameter_count"])),
                "fold_records": sorted(
                    fold_records, key=lambda record: int(record["outer_fold"])
                ),
                "pooled_oof_metrics": regression_metrics(targets, prediction),
                "pooled_change_metrics": change_space_metrics(
                    targets, prediction, baseline
                ),
                "diagnostics": _v27_diagnostics(oof),
                "bootstrap_95_ci": bootstrap_metrics(
                    targets, prediction, n_bootstrap=replicates, seed=seed
                ),
                "change_bootstrap_95_ci": bootstrap_change_metrics(
                    targets,
                    prediction,
                    baseline,
                    n_bootstrap=replicates,
                    seed=seed,
                ),
                "bootstrap_replicates": replicates,
                "assembly_audit": {
                    "method": "validated non-overlapping partial-fold assembly",
                    "source_git_revision": source_revision,
                    "observed_summary_git_revisions": sorted(revisions),
                    "observed_checkpoint_git_revisions": sorted(
                        checkpoint_revisions
                    ),
                    "training_source_stability": source_stability,
                    "sources": source_files,
                    "combined_oof_sha256": _sha256(
                        output / "out_of_fold_predictions.csv"
                    ),
                },
            }
        )
        combined["cv_protocol"] = dict(combined["cv_protocol"])
        combined["cv_protocol"]["outer_folds_evaluated"] = sorted(seen_folds)
        (output / "summary.json").write_text(
            json.dumps(combined, indent=2), encoding="utf-8"
        )
    except Exception:
        # Leave a partial directory for audit; never silently remove or overwrite it.
        raise
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-source-revision")
    args = parser.parse_args()
    summary = combine_nested_fold_outputs(
        args.parts,
        args.output,
        training_source_revision=args.training_source_revision,
    )
    print(json.dumps(summary["pooled_change_metrics"], indent=2))


if __name__ == "__main__":
    main()
