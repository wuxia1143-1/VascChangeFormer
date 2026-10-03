from __future__ import annotations

import numpy as np


def comparison_models() -> tuple[str, ...]:
    return (
        "single_tbr", "single_cac", "itransformer_mtl", "first_icu_mtl",
        "learning_to_route", "lac_itransformer",
    )


def error_plane(metrics_by_model: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    """Return the prespecified TBR-vs-CAC error plane for plotting."""
    return [
        {"model": name, "tbr_mae": metrics["tbr_mae"], "log_cac_mae": metrics["log_cac_mae"]}
        for name, metrics in metrics_by_model.items()
    ]


def pareto_nondominated(metrics_by_model: dict[str, dict[str, float]]) -> set[str]:
    names = list(metrics_by_model)
    keep = set(names)
    for name in names:
        own = metrics_by_model[name]
        for rival in names:
            if rival == name:
                continue
            other = metrics_by_model[rival]
            weak = other["tbr_mae"] <= own["tbr_mae"] and other["log_cac_mae"] <= own["log_cac_mae"]
            strict = other["tbr_mae"] < own["tbr_mae"] or other["log_cac_mae"] < own["log_cac_mae"]
            if weak and strict:
                keep.discard(name)
                break
    return keep
