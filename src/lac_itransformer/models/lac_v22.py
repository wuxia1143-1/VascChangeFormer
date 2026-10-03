from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v21 import LACV21Config, LACiTransformerV21


@dataclass
class LACV22Config(LACV21Config):
    """Neural core for the cross-fitted, clinically calibrated V2.2 model."""

    auxiliary_progression: bool = True
    auxiliary_magnitude: bool = True


class LACiTransformerV22(LACiTransformerV21):
    """V2.2 neural core with CAC progression and magnitude probes.

    The final V2.2 estimator adds a calibration expert trained exclusively on
    inner-fold out-of-fold predictions. These auxiliary probes expose
    clinically useful CAC state to that expert without changing the strict
    TBR/CAC information-flow contract inherited from V2.1.
    """

    architecture_version = "V2.2"

    def __init__(self, config: LACV22Config):
        super().__init__(config)
        self.config = config
        hidden = config.hidden_dim
        self.cac_progression_head = nn.Linear(hidden, 1)
        self.cac_magnitude_head = nn.Linear(hidden, 1)
        if not config.auxiliary_progression:
            self._freeze_module(self.cac_progression_head)
        if not config.auxiliary_magnitude:
            self._freeze_module(self.cac_magnitude_head)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        output = super().forward(*args, **kwargs)
        calcification_summary = (
            output["trajectory_calcification"]
            * output["patch_pool_calcification"].unsqueeze(-1)
        ).sum(dim=1)
        output["cac_progression_logit"] = self.cac_progression_head(
            calcification_summary
        ).squeeze(-1)
        output["cac_change_magnitude"] = F.softplus(
            self.cac_magnitude_head(calcification_summary).squeeze(-1)
        )
        return output
