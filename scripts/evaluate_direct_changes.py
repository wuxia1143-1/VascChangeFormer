"""Evaluate private native-change CSV exports; no model fitting or endpoint conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from direct_change_metrics import (
    METRICS, TARGETS, bootstrap_samples, direct_metrics, patient_draws,
)

KEYS = ["experiment", "development", "centre", "model"]
SIZES = {"A": 323, "B": 31, "C": 546}


def evaluate(frame: pd.DataFrame, replicates: int = 2000, seed: int = 2026
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = KEYS + ["patient_id"] + [f"{prefix}_{t}" for t in TARGETS
                                                     for prefix in ("true", "pred")]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f"Missing native-change columns: {missing}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Missing group, patient, or change values")
    if frame.duplicated(KEYS + ["patient_id"]).any():
        raise ValueError("Each patient must appear once per evaluation/model")
    records, cache = [], {}
    for key, group in frame.groupby(KEYS, sort=False):
        experiment, development, centre, model = key
        if centre not in SIZES or len(group) != SIZES[centre]:
            raise ValueError(f"Incomplete cohort: {key}, n={len(group)}")
        if "patient_order" in group:
            order = group.patient_order.to_numpy(float)
            if (not np.isfinite(order).all() or len(np.unique(order)) != len(order)
                    or not np.equal(order, np.floor(order)).all()):
                raise ValueError(f"Invalid patient_order: {key}")
            group = group.sort_values("patient_order", kind="stable")
        draws = patient_draws(len(group), centre, experiment, replicates, seed)
        for target in TARGETS:
            truth = group[f"true_{target}"].to_numpy(float)
            prediction = group[f"pred_{target}"].to_numpy(float)
            point = direct_metrics(truth, prediction)
            samples = bootstrap_samples(truth, prediction, draws)
            bounds = np.nanquantile(samples, [.025, .975], axis=0)
            cache[(*key, target)] = (group.patient_id.tolist(), truth, prediction)
            for j, metric in enumerate(METRICS):
                records.append(dict(zip(KEYS, key)) | {
                    "n": len(group), "target": target, "metric": metric,
                    "estimate": point[j], "ci_low": bounds[0, j], "ci_high": bounds[1, j],
                    "bootstrap_replicates": replicates, "bootstrap_seed": seed,
                    "valid_bootstrap_replicates": int(np.isfinite(samples[:, j]).sum()),
                    "prediction_scale": "native_change_before_endpoint_reconstruction",
                })
    paired = []
    for key, (ids, truth, prediction) in cache.items():
        experiment, development, centre, model, target = key
        if experiment != "ablation":
            continue
        full_key = ("primary", development, centre, "vascmtl", target)
        if full_key not in cache:
            raise ValueError("Paired ablations require the corresponding full-model OOF export")
        full_ids, full_truth, full_prediction = cache[full_key]
        index = pd.Index(ids).get_indexer(full_ids)
        if len(ids) != len(full_ids) or (index < 0).any():
            raise ValueError("Full and ablation patients differ")
        truth, prediction = truth[index], prediction[index]
        if not np.array_equal(truth, full_truth):
            raise ValueError("Full and ablation observed changes differ")
        draws = patient_draws(len(ids), centre, "primary", replicates, seed)
        point = direct_metrics(truth, prediction) - direct_metrics(truth, full_prediction)
        difference = (bootstrap_samples(truth, prediction, draws)
                      - bootstrap_samples(truth, full_prediction, draws))
        # Positive benefit always favours the full model.
        point[2] *= -1
        difference[:, 2] *= -1
        bounds = np.nanquantile(difference, [.025, .975], axis=0)
        for j, metric in enumerate(METRICS):
            paired.append({"development": development, "centre": centre,
                           "variant": model, "target": target, "metric": metric,
                           "full_model_benefit": point[j], "ci_low": bounds[0, j],
                           "ci_high": bounds[1, j], "bootstrap_replicates": replicates,
                           "bootstrap_seed": seed})
    return pd.DataFrame(records), pd.DataFrame(paired)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    frames, inputs = [], []
    for path in args.predictions:
        inputs.append({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        frames.append(pd.read_csv(path, dtype={"patient_id": str}, float_precision="round_trip"))
    metrics, paired = evaluate(pd.concat(frames, ignore_index=True))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "direct_change_metrics_95ci.csv", index=False, float_format="%.17g")
    if not paired.empty:
        paired.to_csv(args.output / "paired_ablation_bootstrap_95ci.csv", index=False, float_format="%.17g")
    manifest = {"status": "PASS", "evaluation_scale": "direct native changes",
                "endpoint_conversion_or_prediction_clipping": False, "model_fitting": False,
                "bootstrap_replicates": 2000, "bootstrap_seed": 2026,
                "continuous_metrics": len(metrics), "paired_metrics": len(paired),
                "input_sha256": inputs}
    (args.output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
