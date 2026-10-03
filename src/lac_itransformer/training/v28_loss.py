from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class V28Loss(nn.Module):
    """Separated task losses used by V2.8 and PCGrad."""

    def __init__(
        self,
        target_standardization: bool = True,
        huber_delta: float = 1.0,
        tbr_mse_weight: float = 0.05,
        cac_mse_weight: float = 0.10,
        lag_aux_weight: float = 0.20,
        lag_aux_mse_weight: float = 0.50,
        progression_weight: float = 0.05,
        magnitude_weight: float = 0.025,
        progression_threshold: float = 0.05,
    ):
        super().__init__()
        self.target_standardization = bool(target_standardization)
        self.huber_delta = float(huber_delta)
        self.tbr_mse_weight = float(tbr_mse_weight)
        self.cac_mse_weight = float(cac_mse_weight)
        self.lag_aux_weight = float(lag_aux_weight)
        self.lag_aux_mse_weight = float(lag_aux_mse_weight)
        self.progression_weight = float(progression_weight)
        self.magnitude_weight = float(magnitude_weight)
        self.progression_threshold = float(progression_threshold)

    def forward(
        self,
        model: nn.Module,
        output: dict[str, torch.Tensor],
        baseline: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_tbr_raw = targets[:, 0] - baseline[:, 0]
        target_cac_raw = (
            torch.log1p(targets[:, 1].clamp_min(0))
            - torch.log1p(baseline[:, 1].clamp_min(0))
        )
        pred_tbr = output["delta_tbr"]
        pred_central = output["delta_log_cac_central"]
        pred_lag = output["delta_log_cac_lag_aux"]
        target_tbr = target_tbr_raw
        target_cac = target_cac_raw
        if self.target_standardization:
            center = model.target_center
            scale = model.target_scale.clamp_min(1e-6)
            target_tbr = (target_tbr - center[0]) / scale[0]
            target_cac = (target_cac - center[1]) / scale[1]
            pred_tbr = (pred_tbr - center[0]) / scale[0]
            pred_central = (pred_central - center[1]) / scale[1]
            pred_lag = (pred_lag - center[1]) / scale[1]

        tbr_l1 = F.l1_loss(pred_tbr, target_tbr)
        tbr_mse = F.mse_loss(pred_tbr, target_tbr)
        tbr = (
            (1.0 - self.tbr_mse_weight) * tbr_l1
            + self.tbr_mse_weight * tbr_mse
        )
        cac_huber = F.huber_loss(
            pred_central,
            target_cac,
            delta=self.huber_delta,
        )
        cac_mse = F.mse_loss(pred_central, target_cac)
        cac_central = (
            (1.0 - self.cac_mse_weight) * cac_huber
            + self.cac_mse_weight * cac_mse
        )
        lag_huber = F.huber_loss(
            pred_lag,
            target_cac,
            delta=self.huber_delta,
        )
        lag_mse = F.mse_loss(pred_lag, target_cac)
        lag_aux = (
            (1.0 - self.lag_aux_mse_weight) * lag_huber
            + self.lag_aux_mse_weight * lag_mse
        )
        progression = F.binary_cross_entropy_with_logits(
            output["cac_progression_logit"],
            (target_cac_raw > self.progression_threshold).to(targets.dtype),
        )
        magnitude = F.smooth_l1_loss(
            output["cac_change_magnitude"],
            target_cac_raw.abs(),
        )
        cac = (
            cac_central
            + self.lag_aux_weight * lag_aux
            + self.progression_weight * progression
            + self.magnitude_weight * magnitude
        )
        return {
            "total": 0.5 * (tbr + cac),
            "tbr": tbr,
            "cac": cac,
            "tbr_l1": tbr_l1,
            "tbr_mse": tbr_mse,
            "cac_central": cac_central,
            "cac_huber": cac_huber,
            "cac_mse": cac_mse,
            "lag_aux": lag_aux,
            "progression": progression,
            "magnitude": magnitude,
        }
