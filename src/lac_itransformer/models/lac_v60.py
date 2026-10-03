from __future__ import annotations

from dataclasses import dataclass

import torch

from .lac_v40_final import LACV40FinalConfig, LACiTransformerV40Final
from .lac_v42 import LACV42Config, LACiTransformerV42


@dataclass
class LACV60SelectedConfig(LACV40FinalConfig):
    """Selected V6-A configuration.

    Tail and I-to-C quantities remain diagnostic inputs for the already fitted
    robust calibrator, but the explicit residual addition is permanently
    removed by :class:`V60PrunedTailSelector`.
    """

    direct_tail_prediction_enabled: bool = False


class LACiTransformerV60Selected(LACiTransformerV40Final):
    """V4-compatible diagnostic backbone for the selected pruned V6 policy."""

    architecture_version = "V6.0-selected"

    def __init__(self, config: LACV60SelectedConfig):
        config.direct_tail_prediction_enabled = False
        super().__init__(config)
        self.config = config


@dataclass
class LACV60Config(LACV42Config):
    """Mechanism-free V6 model used by the locked simplification sequence.

    V6 retains the protected TBR path, the private direct CAC path, baseline
    anchoring and task-specific low-rank adapters.  It removes every neural
    inflammation-to-calcification, progression-hurdle and tail-residual route.
    Generic temporal context, when requested by V6-C, is supplied only to the
    outer-training-pool OOF calibrator and is therefore not represented here.
    """

    shared_only_mtl: bool = False
    shared_backbone_mtl: bool = False
    training_objective: str = "staged_dual_task"


class LACiTransformerV60(LACiTransformerV42):
    """Protected shared/private MTL predictor without a medical mechanism path."""

    architecture_version = "V6.0-strict"

    def __init__(self, config: LACV60Config):
        shared_only = bool(config.shared_only_mtl)
        shared_backbone = bool(shared_only or config.shared_backbone_mtl)
        config.progression_hurdle_enabled = False
        config.hurdle_i_to_c_enabled = False
        config.hurdle_risk_gate_enabled = False
        config.hurdle_history_enabled = False
        config.hurdle_treatment_enabled = False
        config.shared_transfer_enabled = False
        config.historical_dose_enabled = False
        config.cac_historical_inflammation_enabled = False
        config.direction_magnitude_enabled = False
        config.mechanism_aux_enabled = False
        config.mechanism_history_enabled = False
        config.coupling_enabled = False
        config.lag_residual_enabled = False
        config.treatment_conditioning = False
        if shared_only:
            config.soft_phenotype_adapters = False
        if shared_backbone:
            config.isolate_cac_shared_gradient = False
        super().__init__(config)
        self.config = config

        # LACiTransformerV40Final deliberately forces the private CAC encoder
        # for its main candidates. Shared-backbone controls are the locked
        # exceptions: both tasks read the same encoder and gradients are shared.
        if shared_backbone:
            config.use_private_cac_encoder = False
            for module in (
                self.cac_token_projection,
                self.cac_variable_embedding,
                self.cac_patch_embedding,
                self.cac_encoder,
                self.cac_encoder_norm,
                self.cac_static_context,
            ):
                self._freeze_module(module)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        output = super().forward(*args, **kwargs)
        # V4.1 represents a disabled hurdle by a constant prior probability.
        # V6 removes the route rather than merely setting its correction to
        # zero, so mechanism-facing compatibility fields must also be zero.
        if "gate_inflammation_to_calcification" in output:
            output["gate_inflammation_to_calcification"] = torch.zeros_like(
                output["gate_inflammation_to_calcification"]
            )
        if "mechanism_probability" in output:
            output["mechanism_probability"] = torch.zeros_like(
                output["mechanism_probability"]
            )
        return output
