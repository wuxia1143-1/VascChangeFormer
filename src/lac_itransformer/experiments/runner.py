from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..data.schema import FeatureSchema
from ..training.trainer import run_cross_validation
from .ablation import ABLATION_OVERRIDES


def run_ablation_suite(
    arrays: dict,
    schema: FeatureSchema,
    base_config: dict[str, Any],
    output_dir: str | Path,
    variants: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Execute only on development data; never pass the external cohort."""
    selected = variants or tuple(ABLATION_OVERRIDES)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in selected:
        config = copy.deepcopy(base_config)
        config.setdefault("model", {}).update(ABLATION_OVERRIDES[name])
        results[name] = run_cross_validation(arrays, schema, config, output / name)
    (output / "ablation_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
