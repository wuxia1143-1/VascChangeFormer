from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v30 import LACV30Config, LACiTransformerV30


@dataclass
class LACV31Config(LACV30Config):
    """Mechanism-isolated protected dual-task configuration."""

    inflammation_feature_indices: tuple[int, ...] = ()
    calcification_feature_indices: tuple[int, ...] = ()
    lag_residual_rank: int = 2
    patient_lag_gate: bool = True
    permute_patient_history: bool = False
    shift_history_to_past: bool = False


class LACiTransformerV31(LACiTransformerV30):
    """V3.1 protected dual-task model with an exclusive historical I-to-C route.

    The central CAC branch cannot observe longitudinal inflammation variables.
    Historical inflammation can reach CAC only through a strictly causal,
    low-rank residual route.  CAC and treatment gradients remain unable to
    rewrite the TBR teacher or TBR-specific parameters.
    """

    architecture_version = "V3.1"

    def __init__(self, config: LACV31Config):
        super().__init__(config)
        self.config = config
        h = config.hidden_dim
        if not config.inflammation_feature_indices:
            raise ValueError("V3.1 requires preregistered inflammation features")
        if not config.calcification_feature_indices:
            raise ValueError("V3.1 requires preregistered calcification features")
        overlap = set(config.inflammation_feature_indices).intersection(
            config.calcification_feature_indices
        )
        if overlap:
            raise ValueError(f"Phenotype feature masks overlap: {sorted(overlap)}")

        inflammation_mask = torch.zeros(config.num_variables)
        calcification_mask = torch.zeros(config.num_variables)
        inflammation_mask[list(config.inflammation_feature_indices)] = 1.0
        calcification_mask[list(config.calcification_feature_indices)] = 1.0
        self.register_buffer("inflammation_feature_mask", inflammation_mask)
        self.register_buffer("calcification_feature_mask", calcification_mask)

        rank = max(1, int(config.lag_residual_rank))
        self.mechanistic_residual_down = nn.Linear(h, rank, bias=False)
        self.mechanistic_residual_up = nn.Linear(rank, 1, bias=False)
        self.patient_lag_gate = nn.Linear(4, 1)
        nn.init.zeros_(self.mechanistic_residual_up.weight)
        nn.init.zeros_(self.patient_lag_gate.weight)
        nn.init.constant_(self.patient_lag_gate.bias, -2.0)

        # V3.1 replaces V3.0's unconstrained MLP residual and high-dimensional
        # residual gate with the low-rank mechanism head and four-input gate.
        self._freeze_module(self.lag_residual_head)
        self._freeze_module(self.lag_residual_gate)
        if not config.coupling_enabled or not config.lag_residual_enabled:
            self._freeze_module(self.mechanistic_residual_down)
            self._freeze_module(self.mechanistic_residual_up)
            self._freeze_module(self.patient_lag_gate)
        elif not config.patient_lag_gate:
            self._freeze_module(self.patient_lag_gate)

    def _masked_inputs(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = feature_mask.to(values.dtype)[None, None, :]
        return values * selected, mask * selected, delta * selected

    def _masked_encode(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        feature_mask: torch.Tensor,
        *,
        cac_encoder: bool,
    ) -> torch.Tensor:
        values, mask, delta = self._masked_inputs(
            values, mask, delta, feature_mask
        )
        if cac_encoder and self.config.separate_cac_encoder:
            _, patches, variables = values.shape
            raw = torch.stack([values, mask, delta], dim=-1)
            variable_ids = torch.arange(variables, device=values.device)
            patch_ids = torch.arange(patches, device=values.device)
            tokens = self.cac_token_projection(raw)
            tokens = tokens + self.cac_variable_embedding(variable_ids)[
                None, None, :, :
            ]
            tokens = tokens + self.cac_patch_embedding(patch_ids)[
                None, :, None, :
            ]
            encoded = [self.cac_encoder(tokens[:, step]) for step in range(patches)]
            hidden = self.cac_encoder_norm(torch.stack(encoded, dim=1))
        else:
            hidden = self._encode_tokens(values, mask, delta)
        selected = feature_mask.to(hidden.dtype)[None, None, :, None]
        scale = float(self.config.num_variables) / float(feature_mask.sum().item())
        return hidden * selected * scale

    def _history_negative_control(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self.config.permute_patient_history and trajectory.shape[0] > 1:
            trajectory = torch.roll(trajectory, shifts=1, dims=0)
        if self.config.shift_history_to_past:
            shifted = torch.zeros_like(trajectory)
            shifted[:, 1:] = trajectory[:, :-1]
            trajectory = shifted
        return trajectory

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
        hidden_i = self._masked_encode(
            values,
            mask,
            delta,
            self.inflammation_feature_mask,
            cac_encoder=False,
        )
        if self.config.baseline_anchoring:
            baseline_i = baseline[:, 0:1]
            baseline_c = torch.log1p(baseline[:, 1:2].clamp_min(0))
        else:
            baseline_i = torch.zeros_like(baseline[:, 0:1])
            baseline_c = torch.zeros_like(baseline[:, 1:2])

        shared_static_i = self.static_context(static)
        context_i = shared_static_i + self.baseline_context_i(baseline_i)
        phenotype_i, adapter_gate_i = self._patient_adapt(hidden_i, context_i, "i")
        trajectory_i = self._trajectory(
            phenotype_i.mean(dim=2), context_i, "i"
        )

        hidden_c = self._masked_encode(
            values,
            mask,
            delta,
            self.calcification_feature_mask,
            cac_encoder=True,
        )
        if self.config.separate_cac_encoder:
            static_c = self.cac_static_context(static)
        else:
            hidden_c = hidden_c.detach()
            static_c = shared_static_i.detach()
        context_c = static_c + self.baseline_context_c(baseline_c)
        phenotype_c, adapter_gate_c = self._patient_adapt(hidden_c, context_c, "c")
        trajectory_c_pre = self._trajectory(
            phenotype_c.mean(dim=2), context_c, "c"
        )

        teacher_source = self._history_negative_control(trajectory_i.detach())
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
            lag_delta = trajectory_c_lag - trajectory_c_pre.detach()
            pooled_lag_delta = (
                lag_delta * pool_c_lag.unsqueeze(-1)
            ).sum(dim=1)
            lag_residual = self.mechanistic_residual_up(
                torch.tanh(self.mechanistic_residual_down(pooled_lag_delta))
            ).squeeze(-1)
            followup = (times[:, -1] - times[:, 0]).clamp_min(0)
            history_strength = torch.sqrt(
                teacher_source.square().mean(dim=(1, 2)).clamp_min(1e-8)
            )
            if self.config.treatment_conditioning:
                treatment_burden = treatments.mean(dim=(1, 2))
            else:
                treatment_burden = torch.zeros_like(followup)
            gate_features = torch.stack(
                [
                    baseline_c.squeeze(-1),
                    torch.log1p(followup),
                    history_strength,
                    treatment_burden,
                ],
                dim=-1,
            )
            if self.config.patient_lag_gate:
                lag_gate = torch.sigmoid(
                    self.patient_lag_gate(gate_features)
                ).squeeze(-1)
            else:
                lag_gate = torch.ones_like(lag_residual)
        else:
            pool_c_lag = pool_c
            lag_residual = delta_log_cac_central.new_zeros(
                delta_log_cac_central.shape
            )
            lag_gate = lag_residual.clone()
            gate_features = delta_log_cac_central.new_zeros(
                (delta_log_cac_central.shape[0], 4)
            )
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
            "lag_gate_features": gate_features,
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
            "cac_progression_logit": self.cac_progression_head(summary_c).squeeze(-1),
            "cac_change_magnitude": F.softplus(
                self.cac_magnitude_head(summary_c).squeeze(-1)
            ),
            "reconstruction": self.reconstruction_head(hidden_i).squeeze(-1),
        }
