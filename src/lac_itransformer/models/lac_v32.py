from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v31 import LACV31Config, LACiTransformerV31


@dataclass
class LACV32Config(LACV31Config):
    """Observed-history augmented directional dual-task configuration."""

    direct_observed_lag: bool = True
    tbr_teacher_in_lag: bool = True
    persistence_gate_features: bool = True
    lag_residual_rank: int = 4


class LACiTransformerV32(LACiTransformerV31):
    """V3.2 with a CAC-specific observed inflammation history encoder.

    TBR remains protected.  CAC receives longitudinal inflammation only through
    a causal lag route that combines a TBR-supervised teacher with a separate
    encoder of the observed biomarker history.  This prevents TBR supervision
    from discarding inflammation information that is useful only for CAC.
    """

    architecture_version = "V3.2"

    def __init__(self, config: LACV32Config):
        super().__init__(config)
        self.config = config
        hidden = config.hidden_dim
        feature_count = len(config.inflammation_feature_indices)
        self.direct_inflammation_encoder = nn.Sequential(
            nn.Linear(3 * feature_count, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.teacher_mix_logit = nn.Parameter(torch.tensor(-0.5))

        rank = max(1, int(config.lag_residual_rank))
        self.mechanistic_residual_down = nn.Linear(hidden, rank, bias=False)
        self.mechanistic_residual_up = nn.Linear(rank, 1, bias=False)
        nn.init.normal_(self.mechanistic_residual_up.weight, mean=0.0, std=1e-3)

        # baseline log-CAC, log follow-up, inflammation AUC, slope, coverage,
        # and treatment burden.  Small nonzero initialization lets gradients
        # reach the gate from the first lag-stage update.
        self.patient_lag_gate = nn.Linear(6, 1)
        nn.init.normal_(self.patient_lag_gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.patient_lag_gate.bias, -1.5)

        if not config.direct_observed_lag:
            self._freeze_module(self.direct_inflammation_encoder)
        if not config.tbr_teacher_in_lag:
            self.teacher_mix_logit.requires_grad_(False)
            self._freeze_module(self.lag_adapter_i)
        if not config.coupling_enabled or not config.lag_residual_enabled:
            self._freeze_module(self.direct_inflammation_encoder)
            self.teacher_mix_logit.requires_grad_(False)
            self._freeze_module(self.mechanistic_residual_down)
            self._freeze_module(self.mechanistic_residual_up)
            self._freeze_module(self.patient_lag_gate)
        elif not config.patient_lag_gate:
            self._freeze_module(self.patient_lag_gate)

        signs = torch.ones(feature_count)
        names = list(config.inflammation_feature_indices)
        # The fourth preregistered variable is lymphocyte in the V3.2 schema;
        # higher lymphocyte values lower the simple persistence burden.
        if len(names) >= 4:
            signs[3] = -1.0
        self.register_buffer("inflammation_burden_sign", signs)

    def _lag_metadata(
        self,
        times: torch.Tensor,
        treatments: torch.Tensor,
        step: int,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate prior interval exposure from current-stage treatment."""
        lag_indices = range(start, step)
        time_delta = torch.stack(
            [(times[:, step] - times[:, lag]).clamp_min(0) for lag in lag_indices],
            dim=1,
        )
        interval_exposure = torch.stack(
            [treatments[:, lag:step].amax(dim=1) for lag in lag_indices],
            dim=1,
        )
        features = torch.cat(
            [
                interval_exposure,
                time_delta.unsqueeze(-1),
                torch.log1p(time_delta).unsqueeze(-1),
            ],
            dim=-1,
        )
        return time_delta, interval_exposure, features

    def _observed_inflammation_history(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.as_tensor(
            self.config.inflammation_feature_indices,
            device=values.device,
            dtype=torch.long,
        )
        observed_values = values.index_select(-1, indices)
        observed_mask = mask.index_select(-1, indices)
        observed_delta = delta.index_select(-1, indices)
        direct_input = torch.cat(
            [
                observed_values * observed_mask,
                observed_mask,
                torch.log1p(observed_delta.clamp_min(0)) * observed_mask,
            ],
            dim=-1,
        )
        direct_source = self.direct_inflammation_encoder(direct_input)

        signed = observed_values * self.inflammation_burden_sign[None, None, :]
        count = observed_mask.sum(dim=-1).clamp_min(1.0)
        patch_burden = (signed * observed_mask).sum(dim=-1) / count
        patch_observed = (observed_mask.sum(dim=-1) > 0).to(values.dtype)
        coverage = patch_observed.mean(dim=1)
        auc = (patch_burden * patch_observed).sum(dim=1) / patch_observed.sum(
            dim=1
        ).clamp_min(1.0)
        elapsed = (times[:, -1] - times[:, 0]).clamp_min(1e-6)
        slope = (patch_burden[:, -1] - patch_burden[:, 0]) / elapsed
        return direct_source, auc, slope, coverage

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
            values, mask, delta, self.inflammation_feature_mask,
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
        trajectory_i = self._trajectory(phenotype_i.mean(dim=2), context_i, "i")

        hidden_c = self._masked_encode(
            values, mask, delta, self.calcification_feature_mask,
            cac_encoder=True,
        )
        if self.config.separate_cac_encoder:
            static_c = self.cac_static_context(static)
        else:
            hidden_c = hidden_c.detach()
            static_c = shared_static_i.detach()
        context_c = static_c + self.baseline_context_c(baseline_c)
        phenotype_c, adapter_gate_c = self._patient_adapt(hidden_c, context_c, "c")
        trajectory_c_pre = self._trajectory(phenotype_c.mean(dim=2), context_c, "c")

        direct_source, inflammation_auc, inflammation_slope, inflammation_coverage = (
            self._observed_inflammation_history(values, mask, delta, times)
        )
        if self.config.direct_observed_lag:
            teacher_source = direct_source
        else:
            teacher_source = torch.zeros_like(direct_source)
        if self.config.tbr_teacher_in_lag:
            tbr_teacher = trajectory_i.detach()
            tbr_teacher = tbr_teacher + self.lag_adapter_i(tbr_teacher)
            teacher_source = teacher_source + torch.sigmoid(
                self.teacher_mix_logit
            ) * tbr_teacher
        teacher_source = self._history_negative_control(teacher_source)

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
            trajectory_c_pre.detach(),
            lag_times,
            treatments,
            context_c.detach(),
        )

        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        pool_c = torch.softmax(self.pool_c(trajectory_c_pre).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (trajectory_c_pre * pool_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context_i], dim=-1)).squeeze(-1)
        delta_log_cac_central = self.head_c(
            torch.cat([summary_c, context_c], dim=-1)
        ).squeeze(-1)

        if self.config.coupling_enabled and self.config.lag_residual_enabled:
            pool_c_lag = torch.softmax(
                self.pool_c_lag(trajectory_c_lag).squeeze(-1), dim=1
            )
            lag_delta = trajectory_c_lag - trajectory_c_pre.detach()
            pooled_lag_delta = (lag_delta * pool_c_lag.unsqueeze(-1)).sum(dim=1)
            lag_residual = self.mechanistic_residual_up(
                torch.tanh(self.mechanistic_residual_down(pooled_lag_delta))
            ).squeeze(-1)
            followup = (times[:, -1] - times[:, 0]).clamp_min(0)
            treatment_burden = (
                treatments.mean(dim=(1, 2))
                if self.config.treatment_conditioning
                else torch.zeros_like(followup)
            )
            if not self.config.persistence_gate_features:
                inflammation_auc = torch.zeros_like(inflammation_auc)
                inflammation_slope = torch.zeros_like(inflammation_slope)
                inflammation_coverage = torch.zeros_like(inflammation_coverage)
            gate_features = torch.stack(
                [
                    baseline_c.squeeze(-1),
                    torch.log1p(followup),
                    inflammation_auc,
                    inflammation_slope,
                    inflammation_coverage,
                    treatment_burden,
                ],
                dim=-1,
            )
            lag_gate = (
                torch.sigmoid(self.patient_lag_gate(gate_features)).squeeze(-1)
                if self.config.patient_lag_gate
                else torch.ones_like(lag_residual)
            )
        else:
            pool_c_lag = pool_c
            lag_residual = delta_log_cac_central.new_zeros(delta_log_cac_central.shape)
            lag_gate = lag_residual.clone()
            gate_features = delta_log_cac_central.new_zeros(
                (delta_log_cac_central.shape[0], 6)
            )
        lag_correction = lag_gate * lag_residual
        delta_log_cac_raw_final = delta_log_cac_central + lag_correction

        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0)) + delta_log_cac_raw_final
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
            "observed_inflammation_auc": inflammation_auc,
            "observed_inflammation_slope": inflammation_slope,
            "observed_inflammation_coverage": inflammation_coverage,
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
