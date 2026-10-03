from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac import BottleneckMLP
from .lac_v2 import LowRankAdapter
from .lac_v22 import LACV22Config, LACiTransformerV22


@dataclass
class LACV27Config(LACV22Config):
    """Configuration for V2.7's decoupled statistical prediction paths."""

    lag_adapter_rank: int = 4
    stop_gradient_lag_source: bool = True
    isolate_cac_shared_gradient: bool = True
    dual_statistical_heads: bool = True


class LACiTransformerV27(LACiTransformerV22):
    """V2.7: phenotype interaction retained, prediction objectives decoupled.

    The robust/median heads target central errors.  The mean heads target
    squared error, and only the CAC mean head reads the strictly historical
    inflammation-to-calcification update.  The final single prediction is
    produced outside the neural core by an inner-OOF-only safe decision layer.
    """

    architecture_version = "V2.7"

    def __init__(self, config: LACV27Config):
        super().__init__(config)
        self.config = config
        h = config.hidden_dim
        self.head_i_mean = BottleneckMLP(2 * h, 1, config.dropout)
        self.pool_c_mean = nn.Linear(h, 1)
        self.head_c_mean = BottleneckMLP(2 * h, 1, config.dropout)
        self.lag_adapter_i = LowRankAdapter(
            h,
            config.lag_adapter_rank,
            config.dropout,
        )
        if not config.dual_statistical_heads:
            self._freeze_module(self.head_i_mean)
            self._freeze_module(self.pool_c_mean)
            self._freeze_module(self.head_c_mean)
        if not config.coupling_enabled:
            self._freeze_module(self.lag_adapter_i)

    def _single_phenotype_adapter(
        self,
        hidden: torch.Tensor,
        baseline: torch.Tensor,
        phenotype: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.phenotype_adapters:
            return hidden, hidden.new_zeros(hidden.shape)
        if phenotype == "i":
            query = self.q_i
            bias_layer = self.baseline_query_bias_i
            gate_layer = self.adapter_gate_i
            adapter = self.adapter_i
        elif phenotype == "c":
            query = self.q_c
            bias_layer = self.baseline_query_bias_c
            gate_layer = self.adapter_gate_c
            adapter = self.adapter_c
        else:
            raise KeyError(phenotype)
        if self.config.lightweight_baseline_bias:
            bias = bias_layer(baseline)
        else:
            bias = hidden.new_zeros((hidden.shape[0], self.config.hidden_dim))
        query_value = query[None, None, None, :] + bias[:, None, None, :]
        gate = torch.sigmoid(gate_layer(hidden) + query_value)
        return hidden + gate * adapter(hidden), gate

    def forward(
        self,
        static: torch.Tensor,
        baseline: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self._encode_tokens(values, mask, delta)
        if self.config.baseline_anchoring:
            baseline_i = baseline[:, 0:1]
            baseline_c = torch.log1p(baseline[:, 1:2].clamp_min(0))
        else:
            baseline_i = torch.zeros_like(baseline[:, 0:1])
            baseline_c = torch.zeros_like(baseline[:, 1:2])

        shared_static = self.static_context(static)
        context_i = shared_static + self.baseline_context_i(baseline_i)
        phenotype_i, adapter_gate_i = self._single_phenotype_adapter(
            hidden,
            baseline_i,
            "i",
        )
        trajectory_i = self._trajectory(
            phenotype_i.mean(dim=2),
            context_i,
            "i",
        )

        # CAC sees the same numerical shared representation but, by default,
        # its loss cannot rewrite the TBR/shared representation.
        if self.config.isolate_cac_shared_gradient:
            hidden_c = hidden.detach()
            shared_static_c = shared_static.detach()
        else:
            hidden_c = hidden
            shared_static_c = shared_static
        context_c = shared_static_c + self.baseline_context_c(baseline_c)
        phenotype_c, adapter_gate_c = self._single_phenotype_adapter(
            hidden_c,
            baseline_c,
            "c",
        )
        trajectory_c_pre = self._trajectory(
            phenotype_c.mean(dim=2),
            context_c,
            "c",
        )

        lag_source = (
            trajectory_i.detach()
            if self.config.stop_gradient_lag_source
            else trajectory_i
        )
        lag_source = lag_source + self.lag_adapter_i(lag_source)
        (
            trajectory_c,
            gate_ic,
            lag_attention,
            lag_time_deltas,
            lag_interval_treatments,
        ) = self._couple_forward(
            lag_source,
            trajectory_c_pre,
            times,
            treatments,
            context_c,
        )

        pool_weight_i = torch.softmax(
            self.pool_i(trajectory_i).squeeze(-1),
            dim=1,
        )
        pool_weight_c_central = torch.softmax(
            self.pool_c(trajectory_c_pre).squeeze(-1),
            dim=1,
        )
        pool_weight_c_mean = torch.softmax(
            self.pool_c_mean(trajectory_c).squeeze(-1),
            dim=1,
        )
        summary_i = (
            trajectory_i * pool_weight_i.unsqueeze(-1)
        ).sum(dim=1)
        summary_c_central = (
            trajectory_c_pre * pool_weight_c_central.unsqueeze(-1)
        ).sum(dim=1)
        summary_c_mean = (
            trajectory_c * pool_weight_c_mean.unsqueeze(-1)
        ).sum(dim=1)

        tbr_input = torch.cat([summary_i, context_i], dim=-1)
        cac_central_input = torch.cat(
            [summary_c_central, context_c],
            dim=-1,
        )
        cac_mean_input = torch.cat([summary_c_mean, context_c], dim=-1)
        delta_tbr_median = self.head_i(tbr_input).squeeze(-1)
        delta_tbr_mean = self.head_i_mean(tbr_input).squeeze(-1)
        delta_log_cac_median = self.head_c(cac_central_input).squeeze(-1)
        delta_log_cac_mean = self.head_c_mean(cac_mean_input).squeeze(-1)

        # The neural-core default is the robust prediction.  Formal V2.7
        # evaluation replaces these aliases using the fitted safe decision
        # layer and never fits that layer on an outer test fold.
        delta_tbr = delta_tbr_median
        delta_log_cac = delta_log_cac_median
        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0)) + delta_log_cac
            ).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_log_cac)

        return {
            "endpoint_tbr": endpoint_tbr,
            "endpoint_cac": endpoint_cac,
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac,
            "delta_tbr_median": delta_tbr_median,
            "delta_tbr_mean": delta_tbr_mean,
            "delta_log_cac_median": delta_log_cac_median,
            "delta_log_cac_mean": delta_log_cac_mean,
            "shared_representation": hidden,
            "phenotype_representation_inflammation": phenotype_i,
            "phenotype_representation_calcification": phenotype_c,
            "phenotype_gate_inflammation": adapter_gate_i,
            "phenotype_gate_calcification": adapter_gate_c,
            "trajectory_inflammation": trajectory_i,
            "trajectory_inflammation_lag_source": lag_source,
            "trajectory_calcification_pre_coupling": trajectory_c_pre,
            "trajectory_calcification": trajectory_c,
            "gate_inflammation_to_calcification": gate_ic,
            "lag_attention": lag_attention,
            "lag_time_deltas": lag_time_deltas,
            "lag_interval_treatment": lag_interval_treatments,
            "current_treatment_status": treatments,
            "patch_pool_inflammation": pool_weight_i,
            "patch_pool_calcification_central": pool_weight_c_central,
            "patch_pool_calcification": pool_weight_c_mean,
            "cac_progression_logit": self.cac_progression_head(
                summary_c_mean
            ).squeeze(-1),
            "cac_change_magnitude": F.softplus(
                self.cac_magnitude_head(summary_c_mean).squeeze(-1)
            ),
            "reconstruction": self.reconstruction_head(hidden).squeeze(-1),
        }
