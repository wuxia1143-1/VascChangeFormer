from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LACLoss(nn.Module):
    def __init__(
        self,
        lambda_competition: float = 0.01,
        lambda_sparse: float = 0.01,
        target_standardization: bool = False,
        task_weighting: str = "uncertainty",
        huber_delta: float = 1.0,
        mse_weight_tbr: float = 0.0,
        mse_weight_cac: float = 0.0,
        lambda_cac_progression: float = 0.0,
        lambda_cac_magnitude: float = 0.0,
        cac_progression_threshold: float = 0.05,
        v27_mean_weight_tbr: float = 0.05,
        v27_mean_weight_cac: float = 0.20,
    ):
        super().__init__()
        self.lambda_competition = lambda_competition
        self.lambda_sparse = lambda_sparse
        self.target_standardization = target_standardization
        self.task_weighting = task_weighting
        self.huber_delta = float(huber_delta)
        self.mse_weight_tbr = float(mse_weight_tbr)
        self.mse_weight_cac = float(mse_weight_cac)
        self.lambda_cac_progression = float(lambda_cac_progression)
        self.lambda_cac_magnitude = float(lambda_cac_magnitude)
        self.cac_progression_threshold = float(cac_progression_threshold)
        self.v27_mean_weight_tbr = float(v27_mean_weight_tbr)
        self.v27_mean_weight_cac = float(v27_mean_weight_cac)
        if task_weighting not in {"uncertainty", "fixed_equal"}:
            raise ValueError(f"Unsupported task weighting: {task_weighting}")
        for name, weight in (
            ("mse_weight_tbr", self.mse_weight_tbr),
            ("mse_weight_cac", self.mse_weight_cac),
            ("v27_mean_weight_tbr", self.v27_mean_weight_tbr),
            ("v27_mean_weight_cac", self.v27_mean_weight_cac),
        ):
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def forward(self, model: nn.Module, output: dict[str, torch.Tensor], baseline: torch.Tensor, targets: torch.Tensor):
        anchored = bool(getattr(getattr(model, "config", None), "baseline_anchoring", True))
        if anchored:
            target_tbr = targets[:, 0] - baseline[:, 0]
            target_cac = torch.log1p(targets[:, 1].clamp_min(0)) - torch.log1p(baseline[:, 1].clamp_min(0))
            pred_tbr, pred_cac = output["delta_tbr"], output["delta_log_cac"]
        else:
            target_tbr = targets[:, 0]
            target_cac = torch.log1p(targets[:, 1].clamp_min(0))
            pred_tbr = output["endpoint_tbr"]
            pred_cac = torch.log1p(output["endpoint_cac"].clamp_min(0))
        raw_target_cac = target_cac
        if self.target_standardization:
            center = getattr(model, "target_center", None)
            scale = getattr(model, "target_scale", None)
            if center is None or scale is None:
                raise ValueError(
                    "Target standardization requires model target_center and "
                    "target_scale buffers"
                )
            target_tbr = (target_tbr - center[0]) / scale[0].clamp_min(1e-6)
            target_cac = (target_cac - center[1]) / scale[1].clamp_min(1e-6)
            pred_tbr = (pred_tbr - center[0]) / scale[0].clamp_min(1e-6)
            pred_cac = (pred_cac - center[1]) / scale[1].clamp_min(1e-6)
        if "delta_tbr_median" in output:
            tbr_median = output["delta_tbr_median"]
            tbr_mean = output["delta_tbr_mean"]
            cac_median = output["delta_log_cac_median"]
            cac_mean = output["delta_log_cac_mean"]
            if self.target_standardization:
                tbr_median = (
                    tbr_median - center[0]
                ) / scale[0].clamp_min(1e-6)
                tbr_mean = (
                    tbr_mean - center[0]
                ) / scale[0].clamp_min(1e-6)
                cac_median = (
                    cac_median - center[1]
                ) / scale[1].clamp_min(1e-6)
                cac_mean = (
                    cac_mean - center[1]
                ) / scale[1].clamp_min(1e-6)
            # Median/robust heads protect MAE; mean heads target squared loss.
            robust_tbr = F.l1_loss(tbr_median, target_tbr)
            mean_tbr = F.mse_loss(tbr_mean, target_tbr)
            robust_cac = F.huber_loss(
                cac_median,
                target_cac,
                delta=self.huber_delta,
            )
            mean_cac = F.mse_loss(cac_mean, target_cac)
            loss_tbr = (
                (1.0 - self.v27_mean_weight_tbr) * robust_tbr
                + self.v27_mean_weight_tbr * mean_tbr
            )
            loss_cac = (
                (1.0 - self.v27_mean_weight_cac) * robust_cac
                + self.v27_mean_weight_cac * mean_cac
            )
        else:
            robust_tbr = F.huber_loss(
                pred_tbr,
                target_tbr,
                delta=self.huber_delta,
            )
            robust_cac = F.huber_loss(
                pred_cac,
                target_cac,
                delta=self.huber_delta,
            )
            mean_tbr = F.mse_loss(pred_tbr, target_tbr)
            mean_cac = F.mse_loss(pred_cac, target_cac)
            loss_tbr = (
                (1.0 - self.mse_weight_tbr) * robust_tbr
                + self.mse_weight_tbr * mean_tbr
            )
            loss_cac = (
                (1.0 - self.mse_weight_cac) * robust_cac
                + self.mse_weight_cac * mean_cac
            )
        log_var_tbr = getattr(model, "log_var_tbr", torch.zeros((), device=targets.device))
        log_var_cac = getattr(model, "log_var_cac", torch.zeros((), device=targets.device))
        selected_task = getattr(model, "task", None)
        if selected_task == "tbr":
            task = loss_tbr
        elif selected_task == "cac":
            task = loss_cac
        elif self.task_weighting == "fixed_equal":
            task = 0.5 * (loss_tbr + loss_cac)
        else:
            task = torch.exp(-log_var_tbr) * loss_tbr + log_var_tbr + torch.exp(-log_var_cac) * loss_cac + log_var_cac
        routes = output.get("routes")
        competitive = bool(getattr(getattr(model, "config", None), "competitive_routing", True))
        competition = (
            (routes[..., 0] * routes[..., 1]).mean()
            if routes is not None and competitive
            else torch.zeros((), device=targets.device)
        )
        forward_gate = output.get("gate_inflammation_to_calcification")
        reverse_gate = output.get("gate_calcification_to_inflammation")
        coupling_mode = getattr(getattr(model, "config", None), "coupling_mode", "asymmetric")
        if coupling_mode == "symmetric" and forward_gate is not None and reverse_gate is not None:
            sparse = 0.5 * (forward_gate.abs().mean() + reverse_gate.abs().mean())
        else:
            sparse = reverse_gate.abs().mean() if reverse_gate is not None else torch.zeros((), device=targets.device)
        total = task + self.lambda_competition * competition + self.lambda_sparse * sparse
        progression = torch.zeros((), device=targets.device)
        if (
            self.lambda_cac_progression > 0
            and "cac_progression_logit" in output
        ):
            progression_target = (
                raw_target_cac > self.cac_progression_threshold
            ).to(targets.dtype)
            progression = F.binary_cross_entropy_with_logits(
                output["cac_progression_logit"],
                progression_target,
            )
            total = total + self.lambda_cac_progression * progression
        magnitude = torch.zeros((), device=targets.device)
        if (
            self.lambda_cac_magnitude > 0
            and "cac_change_magnitude" in output
        ):
            magnitude = F.smooth_l1_loss(
                output["cac_change_magnitude"],
                raw_target_cac.abs(),
            )
            total = total + self.lambda_cac_magnitude * magnitude
        return {
            "total": total,
            "tbr": loss_tbr,
            "cac": loss_cac,
            "competition": competition,
            "sparse": sparse,
            "cac_progression": progression,
            "cac_magnitude": magnitude,
            "robust_tbr": robust_tbr,
            "mean_tbr": mean_tbr,
            "robust_cac": robust_cac,
            "mean_cac": mean_cac,
        }


def masked_reconstruction_loss(output: dict[str, torch.Tensor], raw_values: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    selected = selected_mask.bool()
    if not selected.any():
        return output["reconstruction"].sum() * 0.0
    return F.smooth_l1_loss(output["reconstruction"][selected], raw_values[selected])
