from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    question: str
    models: tuple[str, ...]
    primary_outputs: tuple[str, ...]
    status: str = "framework_ready_no_formal_result"


def build_experiment_plan() -> dict:
    specs = [
        ExperimentSpec(
            "prediction_performance",
            "Can the model accurately predict endpoint TBR and CAC?",
            ("persistence", "elastic_net", "xgboost", "apn_dr", "itransformer_mtl", "first_icu_mtl", "learning_to_route", "lac_itransformer"),
            ("TBR MAE/RMSE/R2", "log1p(CAC) MAE/RMSE/R2", "raw CAC median AE", "patient bootstrap 95% CI"),
        ),
        ExperimentSpec(
            "single_vs_multi_task",
            "Does coupled multi-task learning add value over single-task and parameter sharing?",
            ("single_tbr", "single_cac", "itransformer_mtl", "first_icu_mtl", "learning_to_route", "lac_itransformer"),
            ("Table 3", "two-dimensional TBR/CAC error plane"),
        ),
        ExperimentSpec(
            "core_ablation",
            "Are competitive routing and lag-aware asymmetric coupling useful?",
            ("full", "no_competitive_routing", "no_background", "static_anchors", "no_baseline_anchor", "no_coupling", "symmetric", "ic_only", "ci_only", "no_treatment_conditioning"),
            ("delta MAE per endpoint", "parameter count", "runtime"),
        ),
        ExperimentSpec(
            "interpretability",
            "Are route explanations faithful and stable across centers?",
            ("lac_itransformer",),
            ("variable-patch heatmaps", "top/random/bottom deletion", "cross-center rank correlation", "representative patients"),
        ),
        ExperimentSpec(
            "clinical_patterns",
            "Can the model identify prespecified vascular response patterns?",
            ("lac_itransformer",),
            ("stable", "inflammation-dominant", "calcification-dominant", "dual-active"),
        ),
        ExperimentSpec(
            "external_validation",
            "Does a locked development-center model generalize to the external center?",
            (
                "persistence", "elastic_net", "xgboost", "apn_dr",
                "itransformer_mtl", "first_icu_mtl", "learning_to_route",
                "lac_itransformer",
            ),
            (
                "full-center-B refit external metrics",
                "internal OOF to external gap",
                "patient bootstrap 95% CI",
            ),
        ),
    ]
    return {
        "formal_conclusions_generated": False,
        "external_center_role": "locked_external_validation_only",
        "specifications": [asdict(spec) for spec in specs],
    }
