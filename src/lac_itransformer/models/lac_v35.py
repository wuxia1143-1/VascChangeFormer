from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v2 import LowRankAdapter
from .lac_v34 import LACV34Config, LACiTransformerV34


@dataclass
class LACV35Config(LACV34Config):
    """Pareto-safe protected dual-task model configuration."""

    use_private_cac_encoder: bool = True
    shared_transfer_enabled: bool = True
    shared_transfer_rank: int = 4
    shared_transfer_initial_logit: float = -2.0
    cac_residual_adapter: bool = True
    cac_adapter_rank: int = 4
    cac_adapter_initial_bias: float = -2.0
    mechanism_history_enabled: bool = True
    mechanism_gradient_conflict_protection: bool = True
    mechanism_gradient_scale: float = 0.05


class LACiTransformerV35(LACiTransformerV34):
    """Protected shared/private CAC model with a safe mechanism auxiliary task.

    TBR owns the inflammation-trained shared encoder.  CAC combines a private
    calcification encoder with a skip-safe transfer from that protected shared
    trajectory and an identity-initialized CAC-only all-variable adapter.  The
    historical mechanism remains an auxiliary classifier and is never added to
    the continuous CAC prediction.
    """

    architecture_version = "V3.5"

    def __init__(self, config: LACV35Config):
        # Always allocate the private encoder so every ablation is checkpoint
        # compatible.  `use_private_cac_encoder` controls whether it is read.
        config.separate_cac_encoder = True
        config.phenotype_adapters = False
        super().__init__(config)
        self.config = config
        hidden = int(config.hidden_dim)

        self.shared_transfer_adapter = LowRankAdapter(
            hidden, max(1, int(config.shared_transfer_rank)), config.dropout
        )
        self.shared_transfer_logit = nn.Parameter(
            torch.tensor(float(config.shared_transfer_initial_logit))
        )
        nn.init.zeros_(self.shared_transfer_adapter.up.weight)

        self.cac_only_adapter = LowRankAdapter(
            hidden, max(1, int(config.cac_adapter_rank)), config.dropout
        )
        self.cac_only_adapter_gate = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.cac_only_adapter.up.weight)
        nn.init.zeros_(self.cac_only_adapter_gate.weight)
        nn.init.constant_(
            self.cac_only_adapter_gate.bias,
            float(config.cac_adapter_initial_bias),
        )

        if not config.shared_transfer_enabled:
            self._freeze_module(self.shared_transfer_adapter)
            self.shared_transfer_logit.requires_grad_(False)
        if not config.cac_residual_adapter:
            self._freeze_module(self.cac_only_adapter)
            self._freeze_module(self.cac_only_adapter_gate)
        if not config.use_private_cac_encoder:
            for module in (
                self.cac_token_projection,
                self.cac_variable_embedding,
                self.cac_patch_embedding,
                self.cac_encoder,
                self.cac_encoder_norm,
                self.cac_static_context,
            ):
                self._freeze_module(module)
        if not config.mechanism_history_enabled:
            self.mechanism_burden_coefficient_raw.requires_grad_(False)

    def _cac_hidden(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.use_private_cac_encoder:
            return self._masked_encode(
                values, mask, delta, feature_mask, cac_encoder=True
            )
        hidden = self._masked_encode(
            values, mask, delta, feature_mask, cac_encoder=False
        )
        return hidden if not self.config.isolate_cac_shared_gradient else hidden.detach()

    def _cac_static(self, static: torch.Tensor, shared_static: torch.Tensor) -> torch.Tensor:
        if self.config.use_private_cac_encoder:
            return self.cac_static_context(static)
        return (
            shared_static
            if not self.config.isolate_cac_shared_gradient
            else shared_static.detach()
        )

    def _transfer_alpha(self) -> torch.Tensor:
        if not self.config.shared_transfer_enabled:
            return self.shared_transfer_logit.new_zeros(())
        return torch.sigmoid(self.shared_transfer_logit)

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

        shared_static = self.static_context(static)
        context_i = shared_static + self.baseline_context_i(baseline_i)
        trajectory_i = self._trajectory(hidden_i.mean(dim=2), context_i, "i")

        core_mask = self.calcification_feature_mask
        hidden_c = self._cac_hidden(values, mask, delta, core_mask)
        static_c = self._cac_static(static, shared_static)
        context_c = static_c + self.baseline_context_c(baseline_c)
        trajectory_private = self._trajectory(hidden_c.mean(dim=2), context_c, "c")

        transfer_source = (
            trajectory_i
            if not self.config.isolate_cac_shared_gradient
            else trajectory_i.detach()
        )
        transfer_alpha = self._transfer_alpha()
        transfer = transfer_alpha * self.shared_transfer_adapter(transfer_source)
        trajectory_fused = trajectory_private + transfer

        if self.config.cac_residual_adapter:
            all_mask = torch.ones_like(self.calcification_feature_mask)
            hidden_all = self._cac_hidden(values, mask, delta, all_mask)
            trajectory_all = self._trajectory(hidden_all.mean(dim=2), context_c, "c")
            pooled_all = trajectory_all.mean(dim=1)
            adapter_gate = torch.sigmoid(
                self.cac_only_adapter_gate(
                    torch.cat([pooled_all, context_c], dim=-1)
                )
            )
            trajectory_c = trajectory_fused + adapter_gate[:, None, :] * self.cac_only_adapter(
                trajectory_all
            )
            diagnostic_adapter_gate = adapter_gate[:, None, None, :]
        else:
            trajectory_all = trajectory_private
            trajectory_c = trajectory_fused
            diagnostic_adapter_gate = trajectory_c.new_zeros(
                (trajectory_c.shape[0], 1, 1, 1)
            )

        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        pool_c = torch.softmax(self.pool_c(trajectory_c).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (trajectory_c * pool_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context_i], dim=-1)).squeeze(-1)
        delta_log_cac = self.head_c(
            torch.cat([summary_c, context_c], dim=-1)
        ).squeeze(-1)

        burden, coverage = self._cumulative_inflammation_burden(values, mask, times)
        if not self.config.mechanism_history_enabled:
            burden = torch.zeros_like(burden)
        coefficient = self._mechanism_coefficient()
        baseline_zero = (baseline[:, 1] <= 0).to(values.dtype)
        patient_coefficient = (
            baseline_zero * coefficient[0]
            + (1.0 - baseline_zero) * coefficient[1]
        )
        mechanism_state = torch.cat([summary_c, context_c], dim=-1)
        if self.config.mechanism_aux_enabled:
            mechanism_logit = (
                self.mechanism_susceptibility(mechanism_state).squeeze(-1)
                + patient_coefficient * burden
            )
        else:
            mechanism_logit = torch.zeros_like(burden)
            patient_coefficient = torch.zeros_like(burden)
        mechanism_probability = torch.sigmoid(mechanism_logit)

        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0)) + delta_log_cac
            ).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_log_cac)

        batch, patches = values.shape[:2]
        history_patches = max(1, patches - 1)
        lag_attention = values.new_zeros((batch, patches, history_patches))
        lag_time_deltas = values.new_zeros((batch, patches, history_patches))
        diagnostic_gate = mechanism_probability[:, None].expand(-1, patches)
        if not self.config.mechanism_aux_enabled:
            diagnostic_gate = torch.zeros_like(diagnostic_gate)
        zeros_treatment = treatments.new_zeros(
            (batch, patches, history_patches, treatments.shape[-1])
        )

        return {
            "endpoint_tbr": endpoint_tbr,
            "endpoint_cac": endpoint_cac,
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac,
            "delta_tbr_median": delta_tbr,
            "delta_tbr_mean": delta_tbr,
            "delta_log_cac_median": delta_log_cac,
            "delta_log_cac_mean": delta_log_cac,
            "delta_log_cac_central": delta_log_cac,
            "delta_log_cac_raw_final": delta_log_cac,
            "lag_residual": torch.zeros_like(delta_log_cac),
            "lag_residual_gate": torch.zeros_like(delta_log_cac),
            "lag_correction": torch.zeros_like(delta_log_cac),
            "shared_representation": hidden_i,
            "phenotype_representation_inflammation": hidden_i,
            "phenotype_representation_calcification": hidden_c,
            "phenotype_gate_inflammation": hidden_i.new_zeros(
                (batch, 1, 1, 1)
            ),
            "phenotype_gate_calcification": diagnostic_adapter_gate,
            "trajectory_inflammation": trajectory_i,
            "trajectory_inflammation_lag_source": transfer_source,
            "trajectory_calcification_pre_coupling": trajectory_private,
            "trajectory_calcification": trajectory_c,
            "trajectory_calcification_all_view": trajectory_all,
            "gate_inflammation_to_calcification": diagnostic_gate,
            "lag_attention": lag_attention,
            "lag_time_deltas": lag_time_deltas,
            "lag_interval_treatment": zeros_treatment,
            "current_treatment_status": treatments,
            "patch_pool_inflammation": pool_i,
            "patch_pool_calcification": pool_c,
            "patch_pool_calcification_lag": pool_c,
            "cac_progression_logit": mechanism_logit,
            "cac_change_magnitude": torch.zeros_like(mechanism_logit),
            "mechanism_probability": mechanism_probability,
            "mechanism_burden": burden,
            "mechanism_coverage": coverage,
            "mechanism_coefficient": coefficient,
            "mechanism_monotonic_violation": F.relu(-coefficient).sum(),
            "shared_transfer_alpha": transfer_alpha.expand(batch),
            "cac_adapter_gate": diagnostic_adapter_gate.reshape(batch),
            "reconstruction": self.reconstruction_head(hidden_i).squeeze(-1),
        }
