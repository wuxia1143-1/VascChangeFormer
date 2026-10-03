from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v2 import LowRankAdapter
from .lac_v36 import LACV36Config, LACiTransformerV36, StrictHistoricalLowRankAdapter


@dataclass
class LACV37Config(LACV36Config):
    """Parsimonious soft-phenotype dual-task candidate.

    V3.7 removes fixed hard phenotype masks.  Two identity-initialized soft
    residual adapters create inflammation and calcification views.  TBR owns
    the shared encoder; CAC receives a protected representation, a strictly
    historical I->C transfer, and a direction/magnitude auxiliary decoder.
    """

    hard_phenotype_views: bool = False
    cac_residual_adapter: bool = False
    soft_phenotype_adapters: bool = True
    soft_adapter_rank_i: int = 4
    soft_adapter_rank_c: int = 4
    soft_adapter_initial_bias: float = -2.0
    direction_magnitude_enabled: bool = True
    direction_hidden_dim: int = 8
    historical_dose_enabled: bool = True
    cac_historical_inflammation_enabled: bool = True
    treatment_conditioning: bool = True


class LACiTransformerV37(LACiTransformerV36):
    """Soft-view, task-protected model with an interpretable CAC distribution head.

    The primary CAC central prediction remains the low-MAE V3.6 regression
    head.  A three-state direction head and two conditional magnitude heads
    provide a second, distribution-aware prediction to the inner-OOF-only
    selector.  No outer-fold label is read by this module.
    """

    architecture_version = "V3.7"

    def __init__(self, config: LACV37Config):
        config.hard_phenotype_views = False
        config.cac_residual_adapter = False
        config.mechanism_aux_enabled = False
        super().__init__(config)
        self.config = config
        hidden = int(config.hidden_dim)
        adapter_bias = float(config.soft_adapter_initial_bias)

        self.soft_adapter_i = LowRankAdapter(
            hidden, max(1, int(config.soft_adapter_rank_i)), config.dropout
        )
        self.soft_adapter_c = LowRankAdapter(
            hidden, max(1, int(config.soft_adapter_rank_c)), config.dropout
        )
        self.soft_adapter_gate_i = nn.Linear(2 * hidden, 1)
        self.soft_adapter_gate_c = nn.Linear(2 * hidden, 1)
        for adapter, gate in (
            (self.soft_adapter_i, self.soft_adapter_gate_i),
            (self.soft_adapter_c, self.soft_adapter_gate_c),
        ):
            nn.init.zeros_(adapter.up.weight)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, adapter_bias)

        direction_hidden = max(4, int(config.direction_hidden_dim))
        descriptor_dim = 4 + int(config.treatment_dim)
        self.history_dose_projection = nn.Sequential(
            nn.Linear(descriptor_dim, direction_hidden),
            nn.LayerNorm(direction_hidden),
            nn.GELU(),
        )
        direction_input = 2 * hidden + direction_hidden
        self.cac_direction_head = nn.Sequential(
            nn.Linear(direction_input, direction_hidden),
            nn.GELU(),
            nn.Linear(direction_hidden, 3),
        )
        self.cac_magnitude_head = nn.Sequential(
            nn.Linear(direction_input, direction_hidden),
            nn.GELU(),
            nn.Linear(direction_hidden, 2),
        )
        nn.init.zeros_(self.cac_direction_head[-1].weight)
        nn.init.zeros_(self.cac_direction_head[-1].bias)
        nn.init.normal_(self.cac_magnitude_head[-1].weight, 0.0, 1e-3)
        nn.init.constant_(self.cac_magnitude_head[-1].bias, -2.5)

        if not config.soft_phenotype_adapters:
            self._freeze_module(self.soft_adapter_i)
            self._freeze_module(self.soft_adapter_c)
            self._freeze_module(self.soft_adapter_gate_i)
            self._freeze_module(self.soft_adapter_gate_c)
        if not config.direction_magnitude_enabled:
            self._freeze_module(self.history_dose_projection)
            self._freeze_module(self.cac_direction_head)
            self._freeze_module(self.cac_magnitude_head)

    def _cac_hidden(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Prevent endpoint inflammation leakage without fixed phenotype views.

        Full V3.7 permits inflammation only in strictly historical CAC patches.
        The complete I->C ablation removes that historical access as well.
        """
        safe_values = values.clone()
        safe_mask = mask.clone()
        safe_delta = delta.clone()
        indices = torch.as_tensor(
            self.config.inflammation_feature_indices,
            device=values.device,
            dtype=torch.long,
        )
        safe_values[:, -1, indices] = 0.0
        safe_mask[:, -1, indices] = 0.0
        safe_delta[:, -1, indices] = 0.0
        if not self.config.cac_historical_inflammation_enabled:
            safe_values[:, :, indices] = 0.0
            safe_mask[:, :, indices] = 0.0
            safe_delta[:, :, indices] = 0.0
        return super()._cac_hidden(
            safe_values, safe_mask, safe_delta, feature_mask
        )

    def _soft_adapt(
        self,
        trajectory: torch.Tensor,
        context: torch.Tensor,
        task: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.soft_phenotype_adapters:
            gate = trajectory.new_zeros((trajectory.shape[0], 1))
            return trajectory, gate
        pooled = trajectory.mean(dim=1)
        if task == "i":
            gate = torch.sigmoid(
                self.soft_adapter_gate_i(torch.cat([pooled, context], dim=-1))
            )
            adapted = trajectory + gate[:, None, :] * self.soft_adapter_i(trajectory)
        elif task == "c":
            gate = torch.sigmoid(
                self.soft_adapter_gate_c(torch.cat([pooled, context], dim=-1))
            )
            adapted = trajectory + gate[:, None, :] * self.soft_adapter_c(trajectory)
        else:
            raise KeyError(task)
        return adapted, gate

    def _history_descriptor(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        burden, coverage = self._cumulative_inflammation_burden(values, mask, times)
        if not self.config.historical_dose_enabled:
            burden = torch.zeros_like(burden)
            coverage = torch.zeros_like(coverage)
        followup = (times[:, -1] - times[:, 0]).clamp_min(0)
        if self.config.treatment_conditioning:
            treatment = treatments[:, :-1].amax(dim=1)
        else:
            treatment = treatments.new_zeros(
                (treatments.shape[0], treatments.shape[-1])
            )
        descriptor = torch.cat(
            [
                burden[:, None],
                coverage[:, None],
                torch.log1p(followup)[:, None],
                (followup > 0).to(values.dtype)[:, None],
                treatment,
            ],
            dim=-1,
        )
        return descriptor, burden, coverage, followup

    def _forward_tbr_only(
        self,
        static: torch.Tensor,
        baseline: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_i = self._masked_encode(
            values,
            mask,
            delta,
            self.inflammation_feature_mask,
            cac_encoder=False,
        )
        baseline_i = (
            baseline[:, 0:1]
            if self.config.baseline_anchoring
            else torch.zeros_like(baseline[:, 0:1])
        )
        context_i = self.static_context(static) + self.baseline_context_i(baseline_i)
        trajectory_i = self._trajectory(hidden_i.mean(dim=2), context_i, "i")
        trajectory_i, _ = self._soft_adapt(trajectory_i, context_i, "i")
        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context_i], dim=-1)).squeeze(-1)
        endpoint_tbr = (
            baseline[:, 0] + delta_tbr
            if self.config.baseline_anchoring
            else delta_tbr
        )
        return {"delta_tbr": delta_tbr, "endpoint_tbr": endpoint_tbr}

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
        if self._tbr_stage_active():
            return self._forward_tbr_only(static, baseline, values, mask, delta)
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

        if self.config.baseline_anchoring:
            baseline_i = baseline[:, 0:1]
            baseline_c = torch.log1p(baseline[:, 1:2].clamp_min(0))
        else:
            baseline_i = torch.zeros_like(baseline[:, 0:1])
            baseline_c = torch.zeros_like(baseline[:, 1:2])
        shared_static = self.static_context(static)
        context_i = shared_static + self.baseline_context_i(baseline_i)
        context_c = self._cac_static(static, shared_static) + self.baseline_context_c(
            baseline_c
        )

        trajectory_i, gate_i = self._soft_adapt(
            output["trajectory_inflammation"], context_i, "i"
        )
        trajectory_c, gate_c = self._soft_adapt(
            output["trajectory_calcification"], context_c, "c"
        )
        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        pool_c = torch.softmax(self.pool_c(trajectory_c).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (trajectory_c * pool_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context_i], dim=-1)).squeeze(-1)
        delta_central = self.head_c(torch.cat([summary_c, context_c], dim=-1)).squeeze(-1)

        descriptor, burden, coverage, followup = self._history_descriptor(
            values, mask, times, treatments
        )
        history_embedding = self.history_dose_projection(descriptor)
        distribution_input = torch.cat(
            [summary_c, context_c, history_embedding], dim=-1
        )
        if self.config.direction_magnitude_enabled:
            direction_logits = self.cac_direction_head(distribution_input)
            direction_probability = torch.softmax(direction_logits, dim=-1)
            magnitudes = F.softplus(self.cac_magnitude_head(distribution_input))
            delta_distribution = (
                direction_probability[:, 2] * magnitudes[:, 1]
                - direction_probability[:, 0] * magnitudes[:, 0]
            )
        else:
            direction_logits = distribution_input.new_zeros(
                (distribution_input.shape[0], 3)
            )
            direction_probability = torch.softmax(direction_logits, dim=-1)
            magnitudes = distribution_input.new_zeros(
                (distribution_input.shape[0], 2)
            )
            delta_distribution = delta_central

        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0)) + delta_central
            ).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_central)

        batch, patches = values.shape[:2]
        history_patches = max(1, patches - 1)
        elapsed = (times[:, -1:] - times[:, :-1]).clamp_min(0)
        historical = torch.isfinite(elapsed) & (elapsed > 1e-8)
        weights = torch.exp(-float(self.config.historical_decay_per_year) * elapsed)
        weights = weights * historical.to(values.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        lag_attention = values.new_zeros((batch, patches, history_patches))
        lag_elapsed = values.new_zeros((batch, patches, history_patches))
        lag_attention[:, -1, : weights.shape[1]] = weights
        lag_elapsed[:, -1, : elapsed.shape[1]] = elapsed
        coupling_gate = direction_probability[:, 2, None].expand(-1, patches)
        if not self.config.historical_dose_enabled:
            coupling_gate = torch.zeros_like(coupling_gate)

        output.update(
            {
                "endpoint_tbr": endpoint_tbr,
                "endpoint_cac": endpoint_cac,
                "delta_tbr": delta_tbr,
                "delta_tbr_median": delta_tbr,
                "delta_tbr_mean": delta_tbr,
                "delta_log_cac": delta_central,
                "delta_log_cac_median": delta_central,
                "delta_log_cac_mean": delta_distribution,
                "delta_log_cac_central": delta_central,
                "delta_log_cac_raw_final": delta_central,
                "cac_direction_logits": direction_logits,
                "cac_direction_probability": direction_probability,
                "cac_direction_magnitudes": magnitudes,
                "cac_progression_logit": (
                    direction_logits[:, 2]
                    - torch.logsumexp(direction_logits[:, :2], dim=-1)
                ),
                "cac_change_magnitude": (
                    direction_probability[:, 0] * magnitudes[:, 0]
                    + direction_probability[:, 2] * magnitudes[:, 1]
                ),
                "mechanism_probability": direction_probability[:, 2],
                "mechanism_burden": burden,
                "mechanism_coverage": coverage,
                "history_followup": followup,
                "phenotype_representation_inflammation": trajectory_i[:, :, None, :],
                "phenotype_representation_calcification": trajectory_c[:, :, None, :],
                "phenotype_gate_inflammation": gate_i[:, None, None, :],
                "phenotype_gate_calcification": gate_c[:, None, None, :],
                "trajectory_inflammation": trajectory_i,
                "trajectory_calcification": trajectory_c,
                "patch_pool_inflammation": pool_i,
                "patch_pool_calcification": pool_c,
                "patch_pool_calcification_lag": pool_c,
                "gate_inflammation_to_calcification": coupling_gate,
                "lag_attention": lag_attention,
                "lag_time_deltas": lag_elapsed,
                "lag_correction": delta_distribution - delta_central,
            }
        )
        if isinstance(self.shared_transfer_adapter, StrictHistoricalLowRankAdapter):
            self.shared_transfer_adapter.set_times(None)
        return output
