from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v33 import LACV33Config, LACiTransformerV33


@dataclass
class LACV34Config(LACV33Config):
    """Protected shared dual-task model with a medical mechanism auxiliary head."""

    history_mode: str = "none"
    hurdle_enabled: bool = False
    coupling_enabled: bool = False
    lag_residual_enabled: bool = False
    treatment_conditioning: bool = False
    use_time_decay_kernel: bool = False
    phenotype_adapters: bool = False
    isolate_cac_shared_gradient: bool = True
    hard_phenotype_views: bool = True
    mechanism_aux_enabled: bool = True
    mechanism_monotonic: bool = True
    mechanism_hidden_dim: int = 8
    mechanism_feature_indices: tuple[int, ...] = ()


class LACiTransformerV34(LACiTransformerV33):
    """Asymmetric shared multi-task predictor with an interpretable auxiliary task.

    The two primary predictions are the validated V3.3 central heads.  A
    strictly historical, sign-constrained inflammation AUC predicts CAC
    progression as an auxiliary task.  It regularizes only the CAC-private
    trajectory/pooling layers and never adds a residual to the CAC prediction.
    """

    architecture_version = "V3.4"

    def __init__(self, config: LACV34Config):
        # V3.4 never uses the V3.3 residual/hurdle route.
        config.history_mode = "none"
        config.hurdle_enabled = False
        config.coupling_enabled = False
        config.lag_residual_enabled = False
        config.use_time_decay_kernel = False
        super().__init__(config)
        self.config = config

        if not config.hard_phenotype_views:
            self.inflammation_feature_mask.fill_(1.0)
            self.calcification_feature_mask.fill_(1.0)

        hidden = int(config.hidden_dim)
        mechanism_hidden = max(4, int(config.mechanism_hidden_dim))
        self.mechanism_susceptibility = nn.Sequential(
            nn.Linear(2 * hidden, mechanism_hidden),
            nn.LayerNorm(mechanism_hidden),
            nn.GELU(),
            nn.Linear(mechanism_hidden, 1),
        )
        # Separate non-negative effects for CAC initiation (baseline=0) and
        # progression (baseline>0).  Both begin weakly positive.
        initial = torch.full((2,), -2.25 if config.mechanism_monotonic else 0.10)
        self.mechanism_burden_coefficient_raw = nn.Parameter(initial)
        nn.init.zeros_(self.mechanism_susceptibility[-1].weight)
        nn.init.zeros_(self.mechanism_susceptibility[-1].bias)

        mechanism_indices = (
            tuple(config.mechanism_feature_indices)
            if config.mechanism_feature_indices
            else tuple(config.inflammation_feature_indices)
        )
        if not mechanism_indices:
            raise ValueError("V3.4 requires at least one mechanism feature")
        self.register_buffer(
            "mechanism_feature_index",
            torch.as_tensor(mechanism_indices, dtype=torch.long),
        )
        signs = torch.ones(len(mechanism_indices))
        if len(mechanism_indices) >= 4:
            signs[3] = -1.0
        self.register_buffer("mechanism_feature_sign", signs)

        if not config.mechanism_aux_enabled:
            self._freeze_module(self.mechanism_susceptibility)
            self.mechanism_burden_coefficient_raw.requires_grad_(False)

    def _cumulative_inflammation_burden(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-continuous time AUC using only strictly earlier measurements."""
        indices = self.mechanism_feature_index.to(values.device)
        history_values = values[:, :-1].index_select(-1, indices)
        history_mask = mask[:, :-1].index_select(-1, indices)
        signed = history_values * self.mechanism_feature_sign[None, None, :]
        observed_count = history_mask.sum(dim=-1).clamp_min(1.0)
        patch_burden = (signed * history_mask).sum(dim=-1) / observed_count
        patch_observed = (history_mask.sum(dim=-1) > 0).to(values.dtype)
        intervals = (times[:, 1:] - times[:, :-1]).clamp_min(0)
        interval_weight = intervals * patch_observed
        burden = (patch_burden * interval_weight).sum(dim=1) / interval_weight.sum(
            dim=1
        ).clamp_min(1e-6)
        burden = burden.clamp(min=-5.0, max=5.0)
        coverage = patch_observed.mean(dim=1)
        return burden, coverage

    def _mechanism_coefficient(self) -> torch.Tensor:
        if self.config.mechanism_monotonic:
            return F.softplus(self.mechanism_burden_coefficient_raw)
        return self.mechanism_burden_coefficient_raw

    def forward(
        self,
        static: torch.Tensor,
        baseline: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(
            static=static,
            baseline=baseline,
            values=values,
            mask=mask,
            delta=delta,
            times=times,
            treatments=treatments,
            **kwargs,
        )
        trajectory_c = output["trajectory_calcification"]
        pool_c = output["patch_pool_calcification"]
        summary_c = (trajectory_c * pool_c.unsqueeze(-1)).sum(dim=1)
        if self.config.baseline_anchoring:
            baseline_c = torch.log1p(baseline[:, 1:2].clamp_min(0))
        else:
            baseline_c = torch.zeros_like(baseline[:, 1:2])
        shared_static = self.static_context(static).detach()
        context_c = shared_static + self.baseline_context_c(baseline_c)
        mechanism_state = torch.cat([summary_c, context_c], dim=-1)
        burden, coverage = self._cumulative_inflammation_burden(values, mask, times)
        coefficient = self._mechanism_coefficient()
        baseline_zero = (baseline[:, 1] <= 0).to(values.dtype)
        patient_coefficient = (
            baseline_zero * coefficient[0]
            + (1.0 - baseline_zero) * coefficient[1]
        )
        if self.config.mechanism_aux_enabled:
            mechanism_logit = (
                self.mechanism_susceptibility(mechanism_state).squeeze(-1)
                + patient_coefficient * burden
            )
        else:
            mechanism_logit = torch.zeros_like(burden)
            patient_coefficient = torch.zeros_like(burden)
        mechanism_probability = torch.sigmoid(mechanism_logit)

        # The audited CV runner persists the mean coupling gate.  Reuse that
        # diagnostic channel for the auxiliary progression probability; it is
        # not used in either primary prediction or decision layer.
        diagnostic_gate = mechanism_probability[:, None].expand(
            -1, values.shape[1]
        )
        if not self.config.mechanism_aux_enabled:
            diagnostic_gate = torch.zeros_like(diagnostic_gate)
        output.update(
            {
                "cac_progression_logit": mechanism_logit,
                "cac_change_magnitude": torch.zeros_like(mechanism_logit),
                "mechanism_probability": mechanism_probability,
                "mechanism_burden": burden,
                "mechanism_coverage": coverage,
                "mechanism_coefficient": coefficient,
                "mechanism_monotonic_violation": F.relu(-coefficient).sum(),
                "gate_inflammation_to_calcification": diagnostic_gate,
                # Explicitly preserve the central prediction as the only CAC
                # regression output.
                "delta_log_cac": output["delta_log_cac_central"],
                "delta_log_cac_mean": output["delta_log_cac_central"],
                "delta_log_cac_raw_final": output["delta_log_cac_central"],
                "lag_correction": torch.zeros_like(output["delta_log_cac_central"]),
            }
        )
        if self.config.baseline_anchoring:
            output["endpoint_cac"] = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0))
                + output["delta_log_cac_central"]
            ).clamp_min(0)
        return output
