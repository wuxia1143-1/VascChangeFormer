from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac import BottleneckMLP
from .lac_v28 import LACV28Config, LACiTransformerV28


@dataclass
class LACV30Config(LACV28Config):
    """Directionally protected, TBR-supervised dual-task configuration."""

    separate_cac_encoder: bool = False
    tbr_supervised_lag_teacher: bool = True
    staged_training: bool = True
    lag_residual_enabled: bool = True


class LACiTransformerV30(LACiTransformerV28):
    """V3.0 directional teacher-coupled dual-task estimator.

    TBR supervision shapes the inflammation representation first.  CAC reads
    the frozen numerical representation and a strictly historical copy of the
    inflammation trajectory, but CAC gradients cannot rewrite the TBR path.
    The lag module predicts only a residual around one central CAC estimate.
    """

    architecture_version = "V3.0"

    def __init__(self, config: LACV30Config):
        super().__init__(config)
        self.config = config
        h = config.hidden_dim
        self.lag_residual_head = BottleneckMLP(2 * h, 1, config.dropout)
        self.lag_residual_gate = nn.Linear(3 * h, 1)
        nn.init.zeros_(self.lag_residual_head.net[-1].weight)
        nn.init.zeros_(self.lag_residual_head.net[-1].bias)
        nn.init.zeros_(self.lag_residual_gate.weight)
        nn.init.constant_(self.lag_residual_gate.bias, -2.0)

        # The V2.8 auxiliary complete-prediction head is structurally replaced
        # by the V3.0 residual head.
        self._freeze_module(self.lag_aux_head)

        if config.separate_cac_encoder:
            self.cac_token_projection = copy.deepcopy(self.token_projection)
            self.cac_variable_embedding = copy.deepcopy(self.variable_embedding)
            self.cac_patch_embedding = copy.deepcopy(self.patch_embedding)
            self.cac_encoder = copy.deepcopy(self.encoder)
            self.cac_encoder_norm = copy.deepcopy(self.encoder_norm)
            self.cac_static_context = copy.deepcopy(self.static_context)
        else:
            self.cac_token_projection = None
            self.cac_variable_embedding = None
            self.cac_patch_embedding = None
            self.cac_encoder = None
            self.cac_encoder_norm = None
            self.cac_static_context = None
        if not config.coupling_enabled or not config.lag_residual_enabled:
            self._freeze_module(self.lag_residual_head)
            self._freeze_module(self.lag_residual_gate)

    def _encode_cac_tokens(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.separate_cac_encoder:
            return self._encode_tokens(values, mask, delta).detach()
        _, patches, variables = values.shape
        raw = torch.stack([values, mask, delta], dim=-1)
        variable_ids = torch.arange(variables, device=values.device)
        patch_ids = torch.arange(patches, device=values.device)
        tokens = self.cac_token_projection(raw)
        tokens = tokens + self.cac_variable_embedding(variable_ids)[None, None, :, :]
        tokens = tokens + self.cac_patch_embedding(patch_ids)[None, :, None, :]
        encoded = [self.cac_encoder(tokens[:, step]) for step in range(patches)]
        return self.cac_encoder_norm(torch.stack(encoded, dim=1))

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
        hidden_i = self._encode_tokens(values, mask, delta)
        if self.config.baseline_anchoring:
            baseline_i = baseline[:, 0:1]
            baseline_c = torch.log1p(baseline[:, 1:2].clamp_min(0))
        else:
            baseline_i = torch.zeros_like(baseline[:, 0:1])
            baseline_c = torch.zeros_like(baseline[:, 1:2])

        shared_static_i = self.static_context(static)
        context_i = shared_static_i + self.baseline_context_i(baseline_i)
        phenotype_i, adapter_gate_i = self._patient_adapt(
            hidden_i, context_i, "i"
        )
        trajectory_i = self._trajectory(
            phenotype_i.mean(dim=2), context_i, "i"
        )

        if self.config.separate_cac_encoder:
            hidden_c = self._encode_cac_tokens(values, mask, delta)
            static_c = self.cac_static_context(static)
        else:
            hidden_c = hidden_i.detach()
            static_c = shared_static_i.detach()
        context_c = static_c + self.baseline_context_c(baseline_c)
        phenotype_c, adapter_gate_c = self._patient_adapt(
            hidden_c, context_c, "c"
        )
        trajectory_c_pre = self._trajectory(
            phenotype_c.mean(dim=2), context_c, "c"
        )

        teacher_source = (
            trajectory_i.detach()
            if self.config.tbr_supervised_lag_teacher
            else trajectory_c_pre.detach()
        )
        teacher_source = teacher_source + self.lag_adapter_i(teacher_source)
        lag_target = trajectory_c_pre.detach()
        lag_context = context_c.detach()
        lag_times = times
        if not self.config.use_real_time_intervals:
            lag_times = torch.arange(
                times.shape[1], device=times.device, dtype=times.dtype
            )[None, :].expand_as(times)
        (
            trajectory_c_lag,
            gate_ic,
            lag_attention,
            lag_time_deltas,
            lag_interval_treatments,
        ) = self._couple_forward(
            teacher_source,
            lag_target,
            lag_times,
            treatments,
            lag_context,
        )

        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        pool_c = torch.softmax(self.pool_c(trajectory_c_pre).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (trajectory_c_pre * pool_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(
            torch.cat([summary_i, context_i], dim=-1)
        ).squeeze(-1)
        delta_log_cac_central = self.head_c(
            torch.cat([summary_c, context_c], dim=-1)
        ).squeeze(-1)

        if self.config.coupling_enabled and self.config.lag_residual_enabled:
            pool_c_lag = torch.softmax(
                self.pool_c_lag(trajectory_c_lag).squeeze(-1), dim=1
            )
            summary_c_lag = (
                trajectory_c_lag * pool_c_lag.unsqueeze(-1)
            ).sum(dim=1)
            lag_residual = self.lag_residual_head(
                torch.cat([summary_c_lag - summary_c.detach(), lag_context], dim=-1)
            ).squeeze(-1)
            lag_gate = torch.sigmoid(
                self.lag_residual_gate(
                    torch.cat(
                        [summary_c.detach(), summary_c_lag, lag_context], dim=-1
                    )
                )
            ).squeeze(-1)
        else:
            pool_c_lag = pool_c
            summary_c_lag = summary_c.detach()
            lag_residual = delta_log_cac_central.new_zeros(
                delta_log_cac_central.shape
            )
            lag_gate = lag_residual.clone()
        lag_correction = lag_gate * lag_residual
        delta_log_cac_raw_final = delta_log_cac_central + lag_correction

        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0))
                + delta_log_cac_raw_final
            ).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_log_cac_raw_final)

        return {
            "endpoint_tbr": endpoint_tbr,
            "endpoint_cac": endpoint_cac,
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac_raw_final,
            "delta_tbr_median": delta_tbr,
            "delta_tbr_mean": delta_tbr,
            "delta_log_cac_median": delta_log_cac_central,
            "delta_log_cac_mean": delta_log_cac_raw_final,
            "delta_log_cac_central": delta_log_cac_central,
            "delta_log_cac_raw_final": delta_log_cac_raw_final,
            "lag_residual": lag_residual,
            "lag_residual_gate": lag_gate,
            "lag_correction": lag_correction,
            "shared_representation": hidden_i,
            "phenotype_representation_inflammation": phenotype_i,
            "phenotype_representation_calcification": phenotype_c,
            "phenotype_gate_inflammation": adapter_gate_i,
            "phenotype_gate_calcification": adapter_gate_c,
            "trajectory_inflammation": trajectory_i,
            "trajectory_inflammation_lag_source": teacher_source,
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
            "reconstruction": self.reconstruction_head(hidden_i).squeeze(-1),
        }
