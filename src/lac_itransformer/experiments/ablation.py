from __future__ import annotations

from dataclasses import replace

from ..models.lac import LACConfig


ABLATION_OVERRIDES = {
    "full": {},
    "no_competitive_routing": {"competitive_routing": False},
    "no_background": {"background_route": False},
    "static_anchors": {"dynamic_anchors": False},
    "no_baseline_anchor": {"baseline_anchoring": False},
    "no_coupling": {"coupling_mode": "none"},
    "symmetric": {"coupling_mode": "symmetric"},
    "ic_only": {"coupling_mode": "ic_only"},
    "ci_only": {"coupling_mode": "ci_only"},
    "no_treatment_conditioning": {"treatment_conditioning": False},
}

ROUTING_ABLATIONS = (
    "full",
    "no_competitive_routing",
    "no_background",
    "static_anchors",
    "no_baseline_anchor",
)

COUPLING_ABLATIONS = (
    "full",
    "no_coupling",
    "symmetric",
    "ic_only",
    "ci_only",
    "no_treatment_conditioning",
)

ABLATION_DESCRIPTIONS = {
    "full": "complete competitive-routing and lag-aware asymmetric-coupling model",
    "no_competitive_routing": "shared representation with no phenotype competition",
    "no_background": "remove the background routing path",
    "static_anchors": "replace patient/patch-specific anchors with learned global anchors",
    "no_baseline_anchor": "remove baseline TBR/CAC context and residual endpoint anchoring",
    "no_coupling": "remove cross-phenotype information exchange",
    "symmetric": "use one shared lag-coupling mechanism in both directions",
    "ic_only": "retain inflammation-to-calcification coupling only",
    "ci_only": "retain calcification-to-inflammation coupling only",
    "no_treatment_conditioning": "remove treatment modulation from coupling",
}


def ablation_configs(base: LACConfig) -> dict[str, LACConfig]:
    return {name: replace(base, **overrides) for name, overrides in ABLATION_OVERRIDES.items()}


def sample_efficiency_fractions() -> tuple[float, ...]:
    return (0.2, 0.4, 0.6, 0.8, 1.0)


def missingness_stress_levels() -> tuple[float, ...]:
    return (0.1, 0.2, 0.3)
