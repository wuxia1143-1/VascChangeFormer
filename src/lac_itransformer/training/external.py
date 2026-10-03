from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.external_validation import ExternalWorkbookReader
from ..data.preprocessing import FoldPreprocessor
from ..data.schema import FeatureSchema
from ..models.lac import LACConfig, LACiTransformer
from .dataset import ArrayDataset, collate_batch
from .metrics import bootstrap_metrics, regression_metrics
from .trainer import _loader, predict, resolve_device


def evaluate_external_workbook(
    workbook_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    n_bootstrap: int = 1000,
) -> dict:
    model_root, output = Path(model_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_dirs = sorted(path for path in model_root.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No locked fold artifacts under {model_root}")
    device = resolve_device(device_name)
    fold_predictions = []
    arrays = None
    schema = None
    for fold_dir in fold_dirs:
        checkpoint_path, preprocessing_path = fold_dir / "model.pt", fold_dir / "preprocessor.json"
        ExternalWorkbookReader.assert_locked_artifacts(checkpoint_path, preprocessing_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not checkpoint.get("external_validation_locked"):
            raise RuntimeError(f"Checkpoint is not marked external-validation locked: {checkpoint_path}")
        fold_schema = FeatureSchema.from_dict(checkpoint["schema"])
        if schema is None:
            schema = fold_schema
            arrays = ExternalWorkbookReader().prepare_locked_arrays(workbook_path, schema)
        elif fold_schema != schema:
            raise RuntimeError("Feature schema mismatch across folds")
        preprocessor = FoldPreprocessor.load(preprocessing_path)
        transformed = preprocessor.transform(arrays)
        model = LACiTransformer(LACConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        prediction, targets, ids = predict(model, _loader(transformed, 64, False), device)
        fold_predictions.append(prediction)
    prediction = np.mean(np.stack(fold_predictions), axis=0)
    targets = arrays["targets"]
    ids = [str(value) for value in arrays["patient_ids"]]
    metrics = regression_metrics(targets, prediction)
    confidence_intervals = bootstrap_metrics(targets, prediction, n_bootstrap=n_bootstrap)
    pd.DataFrame({
        "patient_id": ids,
        "true_tbr": targets[:, 0],
        "pred_tbr": prediction[:, 0],
        "true_cac": targets[:, 1],
        "pred_cac": prediction[:, 1],
    }).to_csv(output / "external_predictions.csv", index=False)
    result = {
        "status": "external_validation_complete",
        "n_patients": len(ids),
        "n_locked_folds": len(fold_dirs),
        "metrics": metrics,
        "bootstrap_95_ci": confidence_intervals,
        "fit_or_selection_on_external_data": False,
    }
    (output / "external_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
