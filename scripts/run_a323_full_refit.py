"""Full-A323 refit after the locked nested five-fold development run.

The nested OOF artifacts are used only to recover the already-selected epoch /
classical-parameter decisions and to fit the VascMTL decision layer.  This
script never reads the sealed B31 outcomes.  It writes one frozen model per
method under ``frozen_models/``; external inference must use those models,
never an average of the five OOF models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from lac_itransformer.data.preprocessing import FoldPreprocessor, subset_arrays  # noqa: E402
from lac_itransformer.data.schema import FeatureSchema  # noqa: E402
from lac_itransformer.experiments.prediction import (  # noqa: E402
    build_classical_model,
    build_torch_model,
)
from lac_itransformer.models.lac import LACConfig  # noqa: E402
from lac_itransformer.models.lac_v60 import LACV60Config  # noqa: E402
from lac_itransformer.training import v27_nested as v27  # noqa: E402
from lac_itransformer.training.dataset import model_inputs  # noqa: E402
from lac_itransformer.training.trainer import (  # noqa: E402
    _loader,
    _runtime_versions,
    fit_model_fixed_epochs,
    patient_id_hash,
    resolve_device,
    seed_everything,
)
from lac_itransformer.training.v60_multitask_trainer import (  # noqa: E402
    fit_v60_multitask_fixed_epochs,
)
from lac_itransformer.training.v60_nested import (  # noqa: E402
    _config_for_model,
    _decision_modes,
    _feature_flags,
    _generic_temporal_context,
    _seed_everything_v60,
)
from lac_itransformer.training.v60_selector import V60GenericDecisionLayer  # noqa: E402
from lac_itransformer.training.v32_trainer import decode_stage_epochs  # noqa: E402


MODELS = (
    "vascmtl",
    "persistence",
    "elastic_net",
    "xgboost",
    "apn_dr",
    "itransformer_mtl",
    "first_icu_mtl",
    "learning_to_route",
)
CLASSICAL = {"elastic_net", "xgboost"}
DEEP = {"apn_dr", "itransformer_mtl", "first_icu_mtl", "learning_to_route"}
SEED = 3_232_026
EXPECTED_INPUT_DIMENSIONS = {
    "static": 19,
    "longitudinal": 16,
    "baseline": 2,
    "treatment": 5,
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x),
        encoding="utf-8",
    )


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def median_epochs(summary: dict[str, Any]) -> int:
    values = [int(x["selected_epochs"]) for x in summary["fold_records"]]
    if len(values) != 5:
        raise RuntimeError("Expected five selected epoch records")
    return int(np.median(values))


def modal_parameters(summary: dict[str, Any]) -> dict[str, Any]:
    values = []
    for record in summary["fold_records"]:
        value = record.get("selected_classical_parameters")
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise RuntimeError("Missing classical hyperparameter selection")
        values.append(json.dumps(value, sort_keys=True))
    counts = Counter(values)
    best = max(counts.values())
    modes = sorted(k for k, v in counts.items() if v == best)
    if len(modes) != 1:
        raise RuntimeError("Classical selection has no unique mode")
    return json.loads(modes[0])


def append_context(bundle: dict[str, Any], transformed: dict[str, np.ndarray]) -> None:
    context = _generic_temporal_context(transformed)
    ids = [str(x) for x in transformed["patient_ids"]]
    order = {pid: i for i, pid in enumerate(ids)}
    bundle_ids = [str(x) for x in bundle["patient_ids"]]
    indices = np.asarray([order[x] for x in bundle_ids])
    bundle["features"]["cac"] = np.column_stack([bundle["features"]["cac"], context[indices]])


def vasc_base(model, transformed, raw, device, batch_size):
    old = v27._feature_flags
    try:
        v27._feature_flags = _feature_flags
        bundle = v27._predict_base_with_features(model, transformed, raw, batch_size, device, "vascmtl")
    finally:
        v27._feature_flags = old
    append_context(bundle, transformed)
    return bundle


def fit_decision(root: Path, arrays, schema, config, device):
    plan_frame = pd.read_csv(root / "protocol_lock" / "A323_locked_patient_outer_fold_plan.csv", dtype={"patient_id": str})
    plan = {str(r.patient_id): int(r.test_fold) for r in plan_frame.itertuples(index=False)}
    index = {str(x): i for i, x in enumerate(arrays["patient_ids"])}
    bundles = []
    generic_feature_count = int(
        _generic_temporal_context(
            FoldPreprocessor.load(root / "models" / "vascmtl" / "fold_1" / "preprocessor.json").transform(arrays)
        ).shape[1]
    )
    model_config = _config_for_model(schema, config, "vascmtl")
    for fold in range(1, 6):
        ids = [pid for pid, f in plan.items() if f == fold]
        raw = subset_arrays(arrays, np.asarray([index[x] for x in ids], dtype=int))
        fold_dir = root / "models" / "vascmtl" / f"fold_{fold}"
        checkpoint = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=False)
        model = build_torch_model("vascmtl", model_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        transformed = FoldPreprocessor.load(fold_dir / "preprocessor.json").transform(raw)
        bundle = vasc_base(model, transformed, raw, device, int(config.get("batch_size", 64)))
        bundle["meta_fold"] = np.full(len(ids), fold, dtype=int)
        bundles.append(bundle)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    joined = {
        "heads": {k: np.concatenate([b["heads"][k] for b in bundles]) for k in ("tbr", "cac")},
        "features": {k: np.concatenate([b["features"][k] for b in bundles]) for k in ("tbr", "cac")},
        "residual_targets": np.concatenate([b["residual_targets"] for b in bundles]),
        "meta_fold": np.concatenate([b["meta_fold"] for b in bundles]),
    }
    old_count = V60GenericDecisionLayer.configured_generic_feature_count
    old_leaf = V60GenericDecisionLayer.configured_min_samples_leaf
    try:
        V60GenericDecisionLayer.configured_generic_feature_count = generic_feature_count
        V60GenericDecisionLayer.configured_min_samples_leaf = int(config.get("decision", {}).get("generic_min_samples_leaf", 20))
        options = dict(config.get("decision", {}))
        selector = V60GenericDecisionLayer(
            candidate_weights=tuple(options["candidate_weights"]),
            degradation_tolerance=float(options["degradation_tolerance"]),
            required_consistent_folds=int(options["required_consistent_folds"]),
            seed=2026,
        ).fit(
            heads=joined["heads"],
            targets={"tbr": joined["residual_targets"][:, 0], "cac": joined["residual_targets"][:, 1]},
            features=joined["features"],
            meta_fold=joined["meta_fold"],
        )
    finally:
        V60GenericDecisionLayer.configured_generic_feature_count = old_count
        V60GenericDecisionLayer.configured_min_samples_leaf = old_leaf
    return selector


def refit_one(root: Path, model_name: str, arrays, schema, device):
    config_path = root / "protocol_lock" / "locked_configs" / f"{model_name}.json"
    config = read_json(config_path)
    source_dir = root / "models" / model_name
    summary = read_json(source_dir / "summary.json")
    out = root / "frozen_models" / model_name
    out.mkdir(parents=True, exist_ok=True)
    preprocessor = FoldPreprocessor.fit(arrays, patient_id_hash(arrays["patient_ids"]))
    preprocessor.save(out / "preprocessor.json")
    transformed = preprocessor.transform(arrays)
    record: dict[str, Any] = {"model": model_name, "source_summary_sha256": sha256(source_dir / "summary.json"), "source_config_sha256": sha256(config_path), "B31_outcomes_used": False}
    if model_name == "persistence":
        (out / "model_meta.json").write_text("{}", encoding="utf-8")
        record.update({"refit": False, "locked_epochs": None, "locked_parameters": None})
    elif model_name in CLASSICAL:
        params = modal_parameters(summary)
        model = build_classical_model(model_name, seed=SEED, **params).fit(transformed)
        with (out / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        record.update({"refit": True, "locked_epochs": None, "locked_parameters": params})
    elif model_name in DEEP:
        epochs = median_epochs(summary)
        checkpoint = torch.load(source_dir / "fold_1" / "model.pt", map_location="cpu", weights_only=False)
        model_config = LACConfig(**checkpoint["model_config"])
        seed_everything(SEED)
        model = build_torch_model(model_name, model_config).to(device)
        model, history = fit_model_fixed_epochs(model, transformed, config, device, epochs, seed=SEED)
        torch.save({"state_dict": model.state_dict(), "model_name": model_name, "model_config": checkpoint["model_config"], "schema": schema.to_dict(), "locked_epochs": epochs, "full_refit_seed": SEED, "external_validation_locked": True}, out / "model.pt")
        dump_json(out / "refit_history.json", history)
        record.update({"refit": True, "locked_epochs": epochs, "locked_parameters": None})
        del model
    elif model_name == "vascmtl":
        selector = fit_decision(root, arrays, schema, config, device)
        with (out / "decision_layer.pkl").open("wb") as f:
            pickle.dump(selector, f)
        code_values = [int(x["selected_epochs"]) for x in summary["fold_records"]]
        epoch_code = int(np.median(code_values))
        _seed_everything_v60(SEED)
        model_config = _config_for_model(schema, config, "vascmtl")
        model = build_torch_model("vascmtl", model_config).to(device)
        model, history = fit_v60_multitask_fixed_epochs(model, transformed, config, device, epoch_code, SEED)
        torch.save({"state_dict": model.state_dict(), "model_name": "vascmtl", "model_config": model_config.to_dict(), "schema": schema.to_dict(), "locked_epoch_code": epoch_code, "decoded_stage_epochs": decode_stage_epochs(epoch_code), "full_refit_seed": SEED, "external_validation_locked": True}, out / "model.pt")
        dump_json(out / "refit_history.json", history)
        record.update({"refit": True, "locked_epoch_code": epoch_code, "decoded_stage_epochs": decode_stage_epochs(epoch_code), "locked_parameters": None})
        del model
    else:
        raise KeyError(model_name)
    record["artifacts"] = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    dump_json(out / "frozen_model_manifest.json", record)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = p.parse_args()
    root = Path(args.experiment_root)
    arrays = load_npz(root / "prepared_data" / "A323_training_arrays.npz")
    if len(arrays["patient_ids"]) != 323:
        raise RuntimeError("Expected A323 training cohort")
    observed_dimensions = {
        "static": int(arrays["static"].shape[1]),
        "longitudinal": int(arrays["values"].shape[2]),
        "baseline": int(arrays["baseline"].shape[1]),
        "treatment": int(arrays["treatments"].shape[2]),
    }
    if observed_dimensions != EXPECTED_INPUT_DIMENSIONS:
        raise RuntimeError(
            f"Final development input mismatch: {observed_dimensions} != "
            f"{EXPECTED_INPUT_DIMENSIONS}"
        )
    schema = FeatureSchema.from_dict(read_json(root / "prepared_data" / "schema.json"))
    device = resolve_device(args.device)
    started = time.perf_counter()
    records = {}
    for model_name in args.models:
        print(f"FULL_REFIT_START {model_name}", flush=True)
        records[model_name] = refit_one(root, model_name, arrays, schema, device)
        print(f"FULL_REFIT_DONE {model_name}", flush=True)
    prior_manifest_path = root / "frozen_models" / "full_refit_manifest.json"
    prior_models = {}
    if prior_manifest_path.exists():
        prior_models = read_json(prior_manifest_path).get("models", {})
    prior_models.update(records)
    manifest = {"status": "A323_full_data_models_frozen", "protocol": "nested_A323_OOF_then_full_A323_refit_then_external_point_prediction", "patient_count": 323, "patient_hash": patient_id_hash(arrays["patient_ids"]), "input_profile": "static19_longitudinal16_baseline2_treatment5", "input_dimensions": observed_dimensions, "models": prior_models, "pi_source": "A323_nested_OOF_absolute_residual_q90_locked_before_refit", "B31_outcomes_used": False, "device": str(device), "runtime_versions": _runtime_versions(), "fit_seconds": time.perf_counter() - started}
    dump_json(root / "frozen_models" / "full_refit_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x))


if __name__ == "__main__":
    main()
