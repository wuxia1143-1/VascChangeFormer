from __future__ import annotations

from dataclasses import dataclass

import torch

from .lac_v40_final import LACV40FinalConfig, LACiTransformerV40Final


@dataclass
class LACV50Config(LACV40FinalConfig):
    """Mechanism-free V5.0 point predictor with detached uncertainty sidecar.

    V5.0 keeps the empirically supported baseline anchoring, private CAC path,
    task-specific low-rank adapters and inner-OOF calibration.  It removes the
    explicit inflammation-to-calcification transfer and every dedicated
    inflammation-history/treatment input from the residual gate.  The remaining
    gate is therefore a statistical reliability gate over the private CAC
    representation, not a biological mechanism claim.

    The heteroscedastic Laplace/conformal component is fitted after point-model
    training from honest OOF residuals.  It is intentionally absent from this
    module so its loss cannot alter the encoder or either point prediction.
    """

    reliability_gate_enabled: bool = True
    detached_uncertainty_sidecar: bool = True


class LACiTransformerV50(LACiTransformerV40Final):
    """Protected dual-task point model with no explicit I-to-C pathway."""

    architecture_version = "V5.0"

    def __init__(self, config: LACV50Config):
        config.use_private_cac_encoder = True
        config.shared_transfer_enabled = False
        config.historical_dose_enabled = False
        config.cac_historical_inflammation_enabled = False
        config.direction_magnitude_enabled = False
        config.mechanism_aux_enabled = False
        config.mechanism_history_enabled = False
        config.coupling_enabled = False
        config.lag_residual_enabled = False
        config.hurdle_i_to_c_enabled = False
        config.hurdle_history_enabled = False
        config.hurdle_treatment_enabled = False
        config.treatment_conditioning = False
        config.hurdle_risk_gate_enabled = bool(config.reliability_gate_enabled)
        super().__init__(config)
        self.config = config

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        output = super().forward(*args, **kwargs)
        if "v41_progression_probability" in output:
            output["reliability_logit"] = output["v41_progression_logit"]
            output["reliability_gate"] = output["v41_progression_probability"]
            output["reliability_residual"] = output["v41_hurdle_correction"]
            # V4.1 exposed its detached hurdle through legacy mechanism keys.
            # Keeping a non-zero value there would make V5.0 artifacts look as
            # if an inflammation-to-calcification or lag path still existed.
            # V5-aware prediction code reads ``reliability_gate`` directly.
            output["gate_inflammation_to_calcification"] = torch.zeros_like(
                output["gate_inflammation_to_calcification"]
            )
            output["mechanism_probability"] = torch.zeros_like(
                output["mechanism_probability"]
            )
            output["lag_correction"] = torch.zeros_like(output["lag_correction"])
        return output
