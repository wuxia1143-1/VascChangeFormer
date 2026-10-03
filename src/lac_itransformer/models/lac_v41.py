from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v2 import LowRankAdapter
from .lac_v37 import LACV37Config, LACiTransformerV37


@dataclass
class LACV41Config(LACV37Config):
    """Exploratory V3.7-preserving CAC progression-hurdle candidate."""

    progression_hurdle_enabled: bool = True
    hurdle_i_to_c_enabled: bool = True
    hurdle_risk_gate_enabled: bool = True
    hurdle_history_enabled: bool = True
    hurdle_treatment_enabled: bool = True
    hurdle_hidden_dim: int = 12
    hurdle_history_dim: int = 8
    hurdle_i_to_c_rank: int = 2
    hurdle_initial_logit: float = -2.0
    hurdle_residual_initial_bias: float = -2.3


class LACiTransformerV41(LACiTransformerV37):
    """V3.7 central prediction plus a skippable progression correction.

    TBR and the V3.7 CAC central statistic are trained exactly as before.  A
    separately trained hurdle stage estimates whether the central statistic
    underestimates clinically relevant positive CAC progression and, if so,
    applies a non-negative conditional correction.  The correction can read
    only earlier inflammation patches, elapsed time, historical treatment and
    the protected CAC representation.  Its inputs from V3.7 are detached, so
    hurdle loss cannot alter either primary prediction path.
    """

    architecture_version = "V4.1"

    def __init__(self, config: LACV41Config):
        super().__init__(config)
        self.config = config
        hidden = int(config.hidden_dim)
        hurdle_hidden = max(4, int(config.hurdle_hidden_dim))
        history_hidden = max(4, int(config.hurdle_history_dim))
        descriptor_dim = 4 + int(config.treatment_dim)

        # Preserve the exact post-construction RNG state of V3.7.  Otherwise
        # initializing the auxiliary stage would change dropout draws in the
        # unchanged primary paths even though those paths never call V4.1.
        primary_rng_state = torch.random.get_rng_state()

        self.v41_history_projection = nn.Sequential(
            nn.Linear(descriptor_dim, history_hidden),
            nn.LayerNorm(history_hidden),
            nn.GELU(),
        )
        self.v41_i2c_adapter = LowRankAdapter(
            hidden,
            max(1, int(config.hurdle_i_to_c_rank)),
            config.dropout,
        )
        hurdle_input = 3 * hidden + history_hidden
        self.v41_risk_head = nn.Sequential(
            nn.Linear(hurdle_input, hurdle_hidden),
            nn.LayerNorm(hurdle_hidden),
            nn.GELU(),
            nn.Linear(hurdle_hidden, 1),
        )
        self.v41_residual_head = nn.Sequential(
            nn.Linear(hurdle_input, hurdle_hidden),
            nn.LayerNorm(hurdle_hidden),
            nn.GELU(),
            nn.Linear(hurdle_hidden, 1),
        )
        nn.init.zeros_(self.v41_i2c_adapter.up.weight)
        nn.init.zeros_(self.v41_risk_head[-1].weight)
        nn.init.constant_(
            self.v41_risk_head[-1].bias, float(config.hurdle_initial_logit)
        )
        nn.init.normal_(self.v41_residual_head[-1].weight, 0.0, 1e-3)
        nn.init.constant_(
            self.v41_residual_head[-1].bias,
            float(config.hurdle_residual_initial_bias),
        )
        torch.random.set_rng_state(primary_rng_state)

        if not config.progression_hurdle_enabled:
            for module in (
                self.v41_history_projection,
                self.v41_i2c_adapter,
                self.v41_risk_head,
                self.v41_residual_head,
            ):
                self._freeze_module(module)

    def _hurdle_stage_active(self) -> bool:
        return any(
            parameter.requires_grad
            for name, parameter in self.named_parameters()
            if name.startswith("v41_")
        )

    def _strict_history_transfer(
        self,
        inflammation_trajectory: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patches, hidden = inflammation_trajectory.shape
        if not self.config.hurdle_i_to_c_enabled or patches < 2:
            return (
                inflammation_trajectory.new_zeros((batch, hidden)),
                inflammation_trajectory.new_zeros((batch, max(1, patches - 1))),
            )
        elapsed = (times[:, -1:] - times[:, :-1]).clamp_min(0)
        valid = torch.isfinite(elapsed) & (elapsed > 1e-8)
        weights = torch.exp(
            -float(self.config.historical_decay_per_year) * elapsed
        ) * valid.to(inflammation_trajectory.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        historical = inflammation_trajectory[:, :-1].detach()
        summary = (historical * weights.unsqueeze(-1)).sum(dim=1)
        return self.v41_i2c_adapter(summary), weights

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
        if "delta_log_cac_central" not in output:
            return output

        central = output["delta_log_cac_central"]
        batch = central.shape[0]
        hurdle_stage_active = self._hurdle_stage_active()
        if not self.config.progression_hurdle_enabled or not hurdle_stage_active:
            risk_logit = central.new_full(
                (batch,), float(self.config.hurdle_initial_logit)
            )
            risk = torch.sigmoid(risk_logit)
            magnitude = torch.zeros_like(central)
            correction = torch.zeros_like(central)
            # While the unchanged V3.7 CAC stage is active, retain its original
            # direction-magnitude auxiliary output.  Replacing it by `central`
            # would silently change the primary training objective.  A model
            # with the new hurdle explicitly disabled instead exposes central
            # as both decision heads for an exact identity ablation.
            corrected = (
                central
                if not self.config.progression_hurdle_enabled
                else output["delta_log_cac_mean"]
            )
            transfer = central.new_zeros((batch, int(self.config.hidden_dim)))
            history_embedding = central.new_zeros(
                (batch, int(self.config.hurdle_history_dim))
            )
            history_weights = central.new_zeros(
                (batch, max(1, values.shape[1] - 1))
            )
        else:
            shared_static = self.static_context(static).detach()
            baseline_c = (
                torch.log1p(baseline[:, 1:2].clamp_min(0))
                if self.config.baseline_anchoring
                else torch.zeros_like(baseline[:, 1:2])
            )
            context_c = (
                self._cac_static(static, shared_static)
                + self.baseline_context_c(baseline_c)
            ).detach()
            trajectory_c = output["trajectory_calcification"].detach()
            pool_c = output["patch_pool_calcification"].detach()
            summary_c = (trajectory_c * pool_c.unsqueeze(-1)).sum(dim=1)
            central_input = torch.cat([summary_c, context_c], dim=-1)

            descriptor, _, _, _ = self._history_descriptor(
                values, mask, times, treatments
            )
            if not self.config.hurdle_history_enabled:
                descriptor = torch.zeros_like(descriptor)
            elif not self.config.hurdle_treatment_enabled:
                descriptor = descriptor.clone()
                descriptor[:, 4:] = 0.0
            history_embedding = self.v41_history_projection(descriptor)
            transfer, history_weights = self._strict_history_transfer(
                output["trajectory_inflammation"], times
            )
            hurdle_input = torch.cat(
                [central_input, transfer, history_embedding], dim=-1
            )
            risk_logit = self.v41_risk_head(hurdle_input).squeeze(-1)
            risk = torch.sigmoid(risk_logit)
            if not self.config.hurdle_risk_gate_enabled:
                risk = torch.ones_like(risk)
            magnitude = F.softplus(
                self.v41_residual_head(hurdle_input).squeeze(-1)
            )
            correction = risk * magnitude
            corrected = central.detach() + correction

        patches = values.shape[1]
        coupling_gate = risk[:, None].expand(-1, patches)
        output.update(
            {
                "delta_log_cac": central,
                "delta_log_cac_median": central,
                "delta_log_cac_mean": corrected,
                "delta_log_cac_central": central,
                "delta_log_cac_raw_final": central,
                "v41_progression_logit": risk_logit,
                "v41_progression_probability": risk,
                "v41_positive_residual": magnitude,
                "v41_hurdle_correction": correction,
                "v41_history_embedding": history_embedding,
                "v41_i2c_transfer": transfer,
                "mechanism_probability": risk,
                "gate_inflammation_to_calcification": coupling_gate,
                "lag_correction": correction,
            }
        )
        lag_attention = output.get("lag_attention")
        if lag_attention is not None and lag_attention.shape[-1] == history_weights.shape[1]:
            lag_attention = lag_attention.clone()
            lag_attention[:, -1, :] = history_weights
            output["lag_attention"] = lag_attention
        return output
