from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac import BottleneckMLP
from .lac_v2 import LowRankAdapter
from .lac_v22 import LACV22Config, LACiTransformerV22


@dataclass
class LACV28Config(LACV22Config):
    """Configuration for V2.8's coordinated dual-task residual model."""

    lag_adapter_rank: int = 4
    patient_scalar_adapters: bool = True
    cac_shared_gradient: bool = True
    gradient_coordination: bool = True
    stop_gradient_lag: bool = True
    use_real_time_intervals: bool = True


class LACiTransformerV28(LACiTransformerV22):
    """Single-output dual-task core with an auxiliary historical I-to-C path.

    The neural core emits one TBR prediction and one central CAC prediction.
    A lag auxiliary prediction trains the strictly historical I-to-C module,
    but the final CAC residual correction is fitted only from inner OOF errors
    by the V2.8 cross-fitted residual corrector.
    """

    architecture_version = "V2.8"

    def __init__(self, config: LACV28Config):
        super().__init__(config)
        self.config = config
        h = config.hidden_dim
        self.patient_adapter_gate_i = nn.Linear(2 * h, 1)
        self.patient_adapter_gate_c = nn.Linear(2 * h, 1)
        self.lag_adapter_i = LowRankAdapter(
            h,
            config.lag_adapter_rank,
            config.dropout,
        )
        self.pool_c_lag = nn.Linear(h, 1)
        self.lag_aux_head = BottleneckMLP(2 * h, 1, config.dropout)

        # Identity initialization protects the single-task central predictors.
        nn.init.zeros_(self.adapter_i.up.weight)
        nn.init.zeros_(self.adapter_c.up.weight)
        nn.init.zeros_(self.lag_adapter_i.up.weight)
        nn.init.zeros_(self.patient_adapter_gate_i.weight)
        nn.init.zeros_(self.patient_adapter_gate_c.weight)
        nn.init.constant_(self.patient_adapter_gate_i.bias, -2.0)
        nn.init.constant_(self.patient_adapter_gate_c.bias, -2.0)

        # V2.8 replaces V2.2's feature-wise gates and query biases with two
        # low-capacity patient gates.  Freeze the inherited unused capacity.
        for module in (
            self.adapter_gate_i,
            self.adapter_gate_c,
            self.baseline_query_bias_i,
            self.baseline_query_bias_c,
        ):
            self._freeze_module(module)
        self.q_i.requires_grad_(False)
        self.q_c.requires_grad_(False)
        if not config.phenotype_adapters:
            self._freeze_module(self.patient_adapter_gate_i)
            self._freeze_module(self.patient_adapter_gate_c)
        if not config.coupling_enabled:
            self._freeze_module(self.lag_adapter_i)
            self._freeze_module(self.pool_c_lag)
            self._freeze_module(self.lag_aux_head)

    def _patient_adapt(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        phenotype: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.phenotype_adapters:
            gate = hidden.new_zeros((hidden.shape[0], 1, 1, 1))
            return hidden, gate
        pooled = hidden.mean(dim=(1, 2))
        if phenotype == "i":
            gate_layer = self.patient_adapter_gate_i
            adapter = self.adapter_i
        elif phenotype == "c":
            gate_layer = self.patient_adapter_gate_c
            adapter = self.adapter_c
        else:
            raise KeyError(phenotype)
        gate = torch.sigmoid(
            gate_layer(torch.cat([pooled, context], dim=-1))
        )[:, None, None, :]
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
        phenotype_i, adapter_gate_i = self._patient_adapt(
            hidden,
            context_i,
            "i",
        )
        trajectory_i = self._trajectory(
            phenotype_i.mean(dim=2),
            context_i,
            "i",
        )

        hidden_c = hidden if self.config.cac_shared_gradient else hidden.detach()
        shared_static_c = (
            shared_static
            if self.config.cac_shared_gradient
            else shared_static.detach()
        )
        context_c = shared_static_c + self.baseline_context_c(baseline_c)
        phenotype_c, adapter_gate_c = self._patient_adapt(
            hidden_c,
            context_c,
            "c",
        )
        trajectory_c_pre = self._trajectory(
            phenotype_c.mean(dim=2),
            context_c,
            "c",
        )

        if self.config.stop_gradient_lag:
            lag_source = trajectory_i.detach()
            lag_target = trajectory_c_pre.detach()
            lag_context = context_c.detach()
        else:
            lag_source = trajectory_i
            lag_target = trajectory_c_pre
            lag_context = context_c
        lag_source = lag_source + self.lag_adapter_i(lag_source)
        if self.config.use_real_time_intervals:
            lag_times = times
        else:
            lag_times = torch.arange(
                times.shape[1],
                device=times.device,
                dtype=times.dtype,
            )[None, :].expand_as(times)
        (
            trajectory_c_lag,
            gate_ic,
            lag_attention,
            lag_time_deltas,
            lag_interval_treatments,
        ) = self._couple_forward(
            lag_source,
            lag_target,
            lag_times,
            treatments,
            lag_context,
        )

        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        pool_c = torch.softmax(
            self.pool_c(trajectory_c_pre).squeeze(-1),
            dim=1,
        )
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (trajectory_c_pre * pool_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(
            torch.cat([summary_i, context_i], dim=-1)
        ).squeeze(-1)
        delta_log_cac_central = self.head_c(
            torch.cat([summary_c, context_c], dim=-1)
        ).squeeze(-1)

        if self.config.coupling_enabled:
            pool_c_lag = torch.softmax(
                self.pool_c_lag(trajectory_c_lag).squeeze(-1),
                dim=1,
            )
            summary_c_lag = (
                trajectory_c_lag * pool_c_lag.unsqueeze(-1)
            ).sum(dim=1)
            delta_log_cac_lag_aux = self.lag_aux_head(
                torch.cat([summary_c_lag, lag_context], dim=-1)
            ).squeeze(-1)
        else:
            pool_c_lag = pool_c
            summary_c_lag = summary_c.detach()
            delta_log_cac_lag_aux = delta_log_cac_central.detach()

        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0))
                + delta_log_cac_central
            ).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_log_cac_central)

        return {
            "endpoint_tbr": endpoint_tbr,
            "endpoint_cac": endpoint_cac,
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac_central,
            # Compatibility aliases used by the audited nested-CV feature
            # extraction path.  TBR intentionally has one statistical head;
            # CAC exposes central and lag-auxiliary predictions separately.
            "delta_tbr_median": delta_tbr,
            "delta_tbr_mean": delta_tbr,
            "delta_log_cac_median": delta_log_cac_central,
            "delta_log_cac_mean": delta_log_cac_lag_aux,
            "delta_log_cac_central": delta_log_cac_central,
            "delta_log_cac_lag_aux": delta_log_cac_lag_aux,
            "shared_representation": hidden,
            "phenotype_representation_inflammation": phenotype_i,
            "phenotype_representation_calcification": phenotype_c,
            "phenotype_gate_inflammation": adapter_gate_i,
            "phenotype_gate_calcification": adapter_gate_c,
            "trajectory_inflammation": trajectory_i,
            "trajectory_calcification_pre_coupling": trajectory_c_pre,
            "trajectory_calcification": trajectory_c_lag,
            "gate_inflammation_to_calcification": gate_ic,
            "lag_attention": lag_attention,
            "lag_time_deltas": lag_time_deltas,
            "lag_interval_treatment": lag_interval_treatments,
            "current_treatment_status": treatments,
            "patch_pool_inflammation": pool_i,
            "patch_pool_calcification": pool_c,
            "patch_pool_calcification_lag": pool_c_lag,
            "cac_progression_logit": self.cac_progression_head(
                summary_c
            ).squeeze(-1),
            "cac_change_magnitude": F.softplus(
                self.cac_magnitude_head(summary_c).squeeze(-1)
            ),
            "reconstruction": self.reconstruction_head(hidden).squeeze(-1),
        }
