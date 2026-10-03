from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet

from ..models.baselines import PersistenceBaseline, SharedMTLiTransformer, SingleTaskiTransformer
from ..models.lac import LACConfig, LACiTransformer
from ..models.lac_v2 import LACV2Config, LACiTransformerV2
from ..models.lac_v21 import LACV21Config, LACiTransformerV21
from ..models.lac_v22 import LACV22Config, LACiTransformerV22
from ..models.lac_v27 import LACV27Config, LACiTransformerV27
from ..models.lac_v28 import LACV28Config, LACiTransformerV28
from ..models.lac_v30 import LACV30Config, LACiTransformerV30
from ..models.lac_v31 import LACV31Config, LACiTransformerV31
from ..models.lac_v32 import LACV32Config, LACiTransformerV32
from ..models.lac_v33 import LACV33Config, LACiTransformerV33
from ..models.lac_v34 import LACV34Config, LACiTransformerV34
from ..models.lac_v35 import LACV35Config, LACiTransformerV35
from ..models.lac_v36 import LACV36Config, LACiTransformerV36
from ..models.lac_v37 import LACV37Config, LACiTransformerV37
from ..models.lac_v41 import LACV41Config, LACiTransformerV41
from ..models.lac_v40_final import LACV40FinalConfig, LACiTransformerV40Final
from ..models.lac_v42 import LACV42Config, LACiTransformerV42
from ..models.lac_v50 import LACV50Config, LACiTransformerV50
from ..models.lac_v60 import (
    LACV60Config,
    LACV60SelectedConfig,
    LACiTransformerV60,
    LACiTransformerV60Selected,
)
from ..models.official_baselines import (
    APNDRBaseline,
    FIRSTICUMTLBaseline,
    ITransformerMTLBaseline,
    LearningToRouteBaseline,
)


@dataclass(frozen=True)
class ModelEntry:
    key: str
    implementation: str
    availability: str
    purpose: str
    upstream: str | None = None


def model_registry() -> dict[str, ModelEntry]:
    entries = (
        ModelEntry("persistence", "PersistenceBaseline", "ready", "baseline carry-forward"),
        ModelEntry("elastic_net", "ResidualElasticNet", "ready", "sparse linear residual comparator"),
        ModelEntry("xgboost", "ResidualXGBoost", "ready", "nonlinear tree residual comparator", "xgboost/xgboost"),
        ModelEntry("apn_dr", "APNDRBaseline", "ready", "irregular adaptive-patching comparator", "decisionintelligence/APN"),
        ModelEntry("itransformer_mtl", "ITransformerMTLBaseline", "ready", "inverted Transformer MTL comparator", "thuml/iTransformer"),
        ModelEntry("first_icu_mtl", "FIRSTICUMTLBaseline", "ready", "clinical graph-temporal MTL comparator", "zenodo:19609882"),
        ModelEntry("learning_to_route", "LearningToRouteBaseline", "ready", "per-sample modality/task routing comparator", "Grosenick-Lab-Cornell/learning-to-route"),
        ModelEntry("lac_itransformer", "LACiTransformer", "ready", "proposed complete model"),
        ModelEntry("lac_v1_no_competitive_routing", "LACiTransformer", "ready", "V1 without mutually exclusive competitive routing"),
        ModelEntry("lac_v2", "LACiTransformerV2", "ready", "V2 non-exclusive phenotype adapters and asymmetric lag coupling"),
        ModelEntry("lac_v2_no_adapters", "LACiTransformerV2", "ready", "V2 without phenotype adapters"),
        ModelEntry("lac_v2_no_coupling", "LACiTransformerV2", "ready", "V2 without cross-phenotype coupling"),
        ModelEntry("lac_v2_forward_only", "LACiTransformerV2", "ready", "V2 supporting probe without the weak reverse path"),
        ModelEntry("lac_v2_symmetric", "LACiTransformerV2", "ready", "V2 with symmetric bidirectional coupling"),
        ModelEntry("lac_v2_no_treatment", "LACiTransformerV2", "ready", "V2 without treatment conditioning"),
        ModelEntry(
            "lac_v21",
            "LACiTransformerV21",
            "ready",
            "V2.1 task-decoupled readouts and forward-only I-to-C coupling",
        ),
        ModelEntry("lac_v22_full", "LACiTransformerV22", "ready", "V2.2 cross-fitted clinically calibrated full model"),
        ModelEntry("lac_v22_no_adapters", "LACiTransformerV22", "ready", "V2.2 without phenotype adapters"),
        ModelEntry("lac_v22_no_coupling", "LACiTransformerV22", "ready", "V2.2 without lag coupling"),
        ModelEntry("lac_v22_no_treatment", "LACiTransformerV22", "ready", "V2.2 without treatment conditioning"),
        ModelEntry("lac_v27_full", "LACiTransformerV27", "ready", "V2.7 inner-OOF safe statistical decision model"),
        ModelEntry("lac_v28_full", "LACiTransformerV28", "ready", "V2.8 coordinated dual-task lag-residual model"),
        ModelEntry("lac_v30_full", "LACiTransformerV30", "ready", "V3.0 protected TBR-supervised directional dual-task model"),
        ModelEntry("lac_v31_full", "LACiTransformerV31", "ready", "V3.1 mechanism-isolated protected dual-task model"),
        ModelEntry("lac_v32_full", "LACiTransformerV32", "ready", "V3.2 observed-inflammation-history causal lag model"),
        ModelEntry("lac_v33_full", "LACiTransformerV33", "ready", "V3.3 interpretable inflammation-history progression model"),
        ModelEntry("lac_v34_full", "LACiTransformerV34", "ready", "V3.4 protected phenotype-view dual-task model with medical auxiliary learning"),
        ModelEntry("lac_v35_full", "LACiTransformerV35", "ready", "V3.5 Pareto-safe protected shared/private dual-task model"),
        ModelEntry("lac_v36_full", "LACiTransformerV36", "ready", "V3.6 RNG-isolated protected dual-task model with inner-OOF CAC calibration"),
        ModelEntry("lac_v37_full", "LACiTransformerV37", "ready", "V3.7 soft-phenotype protected dual-task model with direction-magnitude CAC decoding"),
        ModelEntry("lac_v41_full", "LACiTransformerV41", "ready", "V4.1 V3.7-preserving CAC progression-hurdle model"),
        ModelEntry("lac_v40_final_full", "LACiTransformerV40Final", "ready", "V4.0-final isolated direct CAC plus protected tail correction"),
        ModelEntry("lac_v42_strict_no_tail", "LACiTransformerV42", "ready", "V4.2 strict tail-feature-free robust CAC calibration"),
        ModelEntry("lac_v42_strict_no_tail_varcal", "LACiTransformerV42", "ready", "V4.2 strict no-tail plus OOF variance calibration"),
        ModelEntry("lac_v42_strict_no_tail_two_stage", "LACiTransformerV42", "ready", "V4.2 strict no-tail plus OOF two-stage CAC mixture"),
        ModelEntry("lac_v50_full", "LACiTransformerV50", "ready", "V5.0 protected dual-task model with mechanism-free reliability gate"),
        ModelEntry("lac_v50_no_task_adapter", "LACiTransformerV50", "ready", "V5.0 without task-specific low-rank adapters"),
        ModelEntry("lac_v50_no_baseline_anchoring", "LACiTransformerV50", "ready", "V5.0 without baseline anchoring"),
        ModelEntry("lac_v50_no_reliability_gate", "LACiTransformerV50", "ready", "V5.0 without the statistical reliability gate"),
        ModelEntry("lac_v50_no_oof_calibration", "LACiTransformerV50", "ready", "V5.0 without inner-OOF calibration"),
        ModelEntry("lac_v60_full", "LACiTransformerV60Selected", "ready", "Selected V6-A with the direct I-to-C/tail residual permanently pruned"),
        ModelEntry("lac_v60_a_no_i2c_tail", "LACiTransformerV60", "ready", "V6-A mechanism-free neural path with the otherwise unchanged robust calibration"),
        ModelEntry("lac_v60_b_no_mechanism_features", "LACiTransformerV60", "ready", "V6-B mechanism-free neural and decision paths"),
        ModelEntry("lac_v60_c_generic_temporal", "LACiTransformerV60", "ready", "V6-C plus task-agnostic generic temporal calibration context"),
        ModelEntry("lac_v60_no_task_adapters", "LACiTransformerV60", "ready", "V6 core ablation without task-specific adapters"),
        ModelEntry("lac_v60_no_baseline_anchoring", "LACiTransformerV60", "ready", "V6 core ablation without baseline anchoring"),
        ModelEntry("lac_v60_no_oof_calibration", "LACiTransformerV60", "ready", "V6 core ablation without OOF robust calibration"),
        ModelEntry("lac_v60_shared_only_mtl", "LACiTransformerV60", "ready", "V6 core shared-only MTL control"),
        ModelEntry("lac_v60_joint_hps_mtl", "LACiTransformerV60", "ready", "V6 simultaneous hard-parameter-sharing MTL control"),
        ModelEntry("lac_v60_joint_shared_adapter_mtl", "LACiTransformerV60", "ready", "V6 simultaneous shared-backbone MTL with task adapters"),
        ModelEntry("lac_v60_single_task_tbr", "LACiTransformerV60", "ready", "V6 TBR-only training control"),
        ModelEntry("lac_v60_single_task_cac", "LACiTransformerV60", "ready", "V6 CAC-only training control"),
        ModelEntry("vascmtl", "LACiTransformerV60", "ready", "Final VascMTL simultaneous hard-parameter-sharing dual-task model"),
        ModelEntry("vascmtl_no_baseline_anchoring", "LACiTransformerV60", "ready", "VascMTL without baseline anchoring"),
        ModelEntry("vascmtl_no_cac_calibration", "LACiTransformerV60", "ready", "VascMTL without CAC inner-OOF calibration"),
        ModelEntry("vascmtl_tbr_single", "LACiTransformerV60", "ready", "Architecture-matched VascMTL TBR-only control"),
        ModelEntry("vascmtl_cac_single", "LACiTransformerV60", "ready", "Architecture-matched VascMTL CAC-only control"),
    )
    return {entry.key: entry for entry in entries}


def build_torch_model(
    name: str,
    config: LACConfig | LACV2Config | LACV21Config | LACV22Config | LACV27Config | LACV28Config | LACV30Config | LACV31Config | LACV32Config | LACV33Config | LACV34Config | LACV35Config | LACV36Config | LACV37Config | LACV41Config | LACV40FinalConfig | LACV42Config | LACV50Config | LACV60Config | LACV60SelectedConfig,
):
    if name == "persistence":
        return PersistenceBaseline()
    if name == "shared_mtl":
        return SharedMTLiTransformer(config)
    if name == "single_tbr":
        return SingleTaskiTransformer(config, "tbr")
    if name == "single_cac":
        return SingleTaskiTransformer(config, "cac")
    if name == "lac_itransformer":
        if not isinstance(config, LACConfig):
            raise TypeError("V1 requires LACConfig")
        return LACiTransformer(config)
    if name == "lac_v1_no_competitive_routing":
        if not isinstance(config, LACConfig):
            raise TypeError("V1 requires LACConfig")
        options = config.to_dict() | {"competitive_routing": False}
        return LACiTransformer(LACConfig(**options))
    v2_overrides = {
        "lac_v2": {},
        "lac_v2_no_adapters": {"phenotype_adapters": False},
        "lac_v2_no_coupling": {"coupling_mode": "none"},
        "lac_v2_forward_only": {"coupling_mode": "ic_only"},
        "lac_v2_symmetric": {"coupling_mode": "symmetric"},
        "lac_v2_no_treatment": {"treatment_conditioning": False},
    }
    if name in v2_overrides:
        if not isinstance(config, LACV2Config):
            raise TypeError("V2 requires LACV2Config")
        options = config.to_dict() | v2_overrides[name]
        return LACiTransformerV2(LACV2Config(**options))
    if name == "lac_v21":
        if not isinstance(config, LACV21Config):
            raise TypeError("V2.1 requires LACV21Config")
        return LACiTransformerV21(config)
    if name.startswith("lac_v22_"):
        if not isinstance(config, LACV22Config):
            raise TypeError("V2.2 requires LACV22Config")
        options = config.to_dict()
        if name == "lac_v22_no_adapters":
            options["phenotype_adapters"] = False
        elif name == "lac_v22_no_coupling":
            options["coupling_enabled"] = False
        elif name == "lac_v22_no_treatment":
            options["treatment_conditioning"] = False
        elif name != "lac_v22_full":
            raise KeyError(name)
        return LACiTransformerV22(LACV22Config(**options))
    if name.startswith("lac_v27_"):
        if not isinstance(config, LACV27Config):
            raise TypeError("V2.7 requires LACV27Config")
        options = config.to_dict()
        overrides = {
            "lac_v27_full": {},
            "lac_v27_median_only": {},
            "lac_v27_mean_only": {},
            "lac_v27_no_adapters": {"phenotype_adapters": False},
            "lac_v27_no_coupling": {"coupling_enabled": False},
            "lac_v27_no_treatment": {"treatment_conditioning": False},
            "lac_v27_no_decision_gate": {},
            "lac_v27_cac_central_only": {},
            "lac_v27_cac_lag_mean_only": {},
            "lac_v27_no_stop_gradient": {
                "stop_gradient_lag_source": False,
                "isolate_cac_shared_gradient": False,
            },
        }
        if name not in overrides:
            raise KeyError(name)
        return LACiTransformerV27(LACV27Config(**(options | overrides[name])))
    if name.startswith("lac_v28_"):
        if not isinstance(config, LACV28Config):
            raise TypeError("V2.8 requires LACV28Config")
        return LACiTransformerV28(config)
    if name.startswith("lac_v30_"):
        if not isinstance(config, LACV30Config):
            raise TypeError("V3.0 requires LACV30Config")
        return LACiTransformerV30(config)
    if name.startswith("lac_v31_"):
        if not isinstance(config, LACV31Config):
            raise TypeError("V3.1 requires LACV31Config")
        return LACiTransformerV31(config)
    if name.startswith("lac_v32_"):
        if not isinstance(config, LACV32Config):
            raise TypeError("V3.2 requires LACV32Config")
        return LACiTransformerV32(config)
    if name.startswith("lac_v33_"):
        if not isinstance(config, LACV33Config):
            raise TypeError("V3.3 requires LACV33Config")
        return LACiTransformerV33(config)
    if name.startswith("lac_v34_"):
        if not isinstance(config, LACV34Config):
            raise TypeError("V3.4 requires LACV34Config")
        return LACiTransformerV34(config)
    if name.startswith("lac_v35_"):
        if not isinstance(config, LACV35Config):
            raise TypeError("V3.5 requires LACV35Config")
        return LACiTransformerV35(config)
    if name.startswith("lac_v36_"):
        if not isinstance(config, LACV36Config):
            raise TypeError("V3.6 requires LACV36Config")
        return LACiTransformerV36(config)
    if name.startswith("lac_v37_"):
        if not isinstance(config, LACV37Config):
            raise TypeError("V3.7 requires LACV37Config")
        return LACiTransformerV37(config)
    if name.startswith("lac_v41_"):
        if not isinstance(config, LACV41Config):
            raise TypeError("V4.1 requires LACV41Config")
        return LACiTransformerV41(config)
    if name.startswith("lac_v40_final_"):
        if not isinstance(config, LACV40FinalConfig):
            raise TypeError("V4.0-final requires LACV40FinalConfig")
        return LACiTransformerV40Final(config)
    if name.startswith("lac_v42_"):
        if not isinstance(config, LACV42Config):
            raise TypeError("V4.2 requires LACV42Config")
        return LACiTransformerV42(config)
    if name.startswith("lac_v50_"):
        if not isinstance(config, LACV50Config):
            raise TypeError("V5.0 requires LACV50Config")
        return LACiTransformerV50(config)
    if name == "lac_v60_full":
        if not isinstance(config, LACV60SelectedConfig):
            raise TypeError("Selected V6.0 requires LACV60SelectedConfig")
        return LACiTransformerV60Selected(config)
    if name.startswith("lac_v60_"):
        if not isinstance(config, LACV60Config):
            raise TypeError("V6.0 requires LACV60Config")
        return LACiTransformerV60(config)
    if name.startswith("vascmtl"):
        if not isinstance(config, LACV60Config):
            raise TypeError("VascMTL requires LACV60Config")
        return LACiTransformerV60(config)
    if name == "apn_dr":
        return APNDRBaseline(config)
    if name == "itransformer_mtl":
        return ITransformerMTLBaseline(config)
    if name == "first_icu_mtl":
        return FIRSTICUMTLBaseline(config)
    if name == "learning_to_route":
        return LearningToRouteBaseline(config)
    raise KeyError(f"Unknown torch model: {name}")


def summarize_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Fixed-length, leakage-free clinical summaries shared by EN/XGBoost."""
    values = arrays["values"].astype(float)
    mask = arrays["mask"].astype(bool)
    count = mask.sum(axis=1)
    mean = np.where(count > 0, (values * mask).sum(axis=1) / np.maximum(count, 1), 0.0)
    minimum = np.where(count > 0, np.where(mask, values, np.inf).min(axis=1), 0.0)
    maximum = np.where(count > 0, np.where(mask, values, -np.inf).max(axis=1), 0.0)
    first = np.take_along_axis(values, mask.argmax(axis=1)[:, None, :], axis=1).squeeze(1)
    reverse_index = mask[:, ::-1].argmax(axis=1)
    last_index = values.shape[1] - 1 - reverse_index
    last = np.take_along_axis(values, last_index[:, None, :], axis=1).squeeze(1)
    slope = np.where(count > 0, last - first, 0.0)
    summaries = [mean, minimum, maximum, slope, mask.mean(axis=1), arrays["delta"].mean(axis=1)]
    treatment = np.concatenate([arrays["treatments"].mean(1), arrays["treatments"].max(1)], axis=1)
    result = np.concatenate([arrays["static"], arrays["baseline"], *summaries, treatment, arrays["times"]], axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=8.0, neginf=-8.0)


class ResidualElasticNet:
    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, seed: int = 2026):
        self.models = [ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=10000) for _ in range(2)]

    def fit(self, arrays: dict[str, np.ndarray]) -> "ResidualElasticNet":
        features = summarize_features(arrays)
        target_tbr = arrays["targets"][:, 0] - arrays["baseline"][:, 0]
        target_cac = np.log1p(arrays["targets"][:, 1]) - np.log1p(arrays["baseline"][:, 1])
        self.models[0].fit(features, target_tbr)
        self.models[1].fit(features, target_cac)
        return self

    def predict(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        features = summarize_features(arrays)
        tbr = arrays["baseline"][:, 0] + self.models[0].predict(features)
        cac = np.maximum(0, np.expm1(np.log1p(arrays["baseline"][:, 1]) + self.models[1].predict(features)))
        return np.column_stack([tbr, cac])


class ResidualXGBoost:
    def __init__(self, seed: int = 2026, **kwargs: Any):
        from xgboost import XGBRegressor

        defaults = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)
        defaults.update(kwargs)
        self.models = [XGBRegressor(random_state=seed + index, **defaults) for index in range(2)]

    def fit(self, arrays: dict[str, np.ndarray]) -> "ResidualXGBoost":
        features = summarize_features(arrays)
        targets = np.column_stack([
            arrays["targets"][:, 0] - arrays["baseline"][:, 0],
            np.log1p(arrays["targets"][:, 1]) - np.log1p(arrays["baseline"][:, 1]),
        ])
        for index, model in enumerate(self.models):
            model.fit(features, targets[:, index])
        return self

    def predict(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        features = summarize_features(arrays)
        delta_tbr, delta_cac = (model.predict(features) for model in self.models)
        return np.column_stack([
            arrays["baseline"][:, 0] + delta_tbr,
            np.maximum(0, np.expm1(np.log1p(arrays["baseline"][:, 1]) + delta_cac)),
        ])


def build_optional_xgboost(seed: int = 2026, **kwargs: Any):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("Install the optional 'xgboost' extra before running this comparator") from exc
    return XGBRegressor(random_state=seed, n_estimators=300, max_depth=3, learning_rate=0.03, **kwargs)


def build_classical_model(name: str, seed: int = 2026, **kwargs: Any):
    if name == "elastic_net":
        return ResidualElasticNet(seed=seed, **kwargs)
    if name == "xgboost":
        return ResidualXGBoost(seed=seed, **kwargs)
    raise KeyError(f"Unknown classical model: {name}")
