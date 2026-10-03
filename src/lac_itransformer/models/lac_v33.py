from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .lac_v32 import LACV32Config, LACiTransformerV32


@dataclass
class LACV33Config(LACV32Config):
    """Interpretable direct-history progression model configuration."""

    phenotype_adapters: bool = False
    tbr_teacher_in_lag: bool = False
    history_mode: str = "causal"  # causal, current, none, permuted
    hurdle_enabled: bool = True
    use_time_decay_kernel: bool = True
    isolate_cac_shared_gradient: bool = True
    history_hidden_dim: int = 8


class LACiTransformerV33(LACiTransformerV32):
    """Protected dual-task model with an interpretable inflammation kernel.

    The CAC central predictor reads only calcification/metabolic variables.
    A small, strictly historical branch reads observed inflammation directly
    and can add only a non-negative progression correction.  No TBR teacher,
    phenotype adapter, or black-box lag attention is used by the full model.
    """

    architecture_version = "V3.3"

    def __init__(self, config: LACV33Config):
        if config.history_mode not in {"causal", "current", "none", "permuted"}:
            raise ValueError(f"Unknown V3.3 history mode: {config.history_mode}")
        super().__init__(config)
        self.config = config
        hidden = config.hidden_dim
        history_hidden = max(4, int(config.history_hidden_dim))
        feature_count = len(config.inflammation_feature_indices)

        self.history_feature_logits = nn.Parameter(torch.zeros(feature_count))
        self.history_decay_raw = nn.Parameter(torch.tensor(-2.0))
        descriptor_dim = 5 + config.treatment_dim
        self.history_projection = nn.Sequential(
            nn.Linear(descriptor_dim, history_hidden),
            nn.LayerNorm(history_hidden),
            nn.GELU(),
        )
        progression_input = 2 * hidden + history_hidden
        self.progression_hurdle = nn.Sequential(
            nn.Linear(progression_input, history_hidden),
            nn.GELU(),
            nn.Linear(history_hidden, 1),
        )
        self.progression_magnitude = nn.Sequential(
            nn.Linear(progression_input, history_hidden),
            nn.GELU(),
            nn.Linear(history_hidden, 1),
        )
        nn.init.zeros_(self.progression_hurdle[-1].weight)
        nn.init.constant_(self.progression_hurdle[-1].bias, -1.5)
        nn.init.normal_(self.progression_magnitude[-1].weight, 0.0, 1e-3)
        nn.init.constant_(self.progression_magnitude[-1].bias, -4.0)

        # V3.3 replaces all inherited learned lag attention and residual heads.
        for module in (
            self.direct_inflammation_encoder,
            self.lag_adapter_i,
            self.ic_query,
            self.ic_key,
            self.ic_value_down,
            self.ic_value_up,
            self.ic_time_score,
            self.ic_gate,
            self.pool_c_lag,
            self.mechanistic_residual_down,
            self.mechanistic_residual_up,
            self.patient_lag_gate,
        ):
            self._freeze_module(module)
        self.teacher_mix_logit.requires_grad_(False)
        self.ic_log_time_decay.requires_grad_(False)
        if hasattr(self, "interval_treatment_score"):
            self._freeze_module(self.interval_treatment_score)
        if hasattr(self, "interval_treatment_encoder"):
            self._freeze_module(self.interval_treatment_encoder)
        if hasattr(self, "current_treatment_encoder"):
            self._freeze_module(self.current_treatment_encoder)
        # These V2.2 auxiliary heads are replaced by the explicit V3.3
        # progression hurdle/magnitude heads and must not inflate the declared
        # trainable parameter count or enter an optimizer without gradients.
        if hasattr(self, "cac_progression_head"):
            self._freeze_module(self.cac_progression_head)
        if hasattr(self, "cac_magnitude_head"):
            self._freeze_module(self.cac_magnitude_head)

        if config.history_mode == "none":
            self.history_feature_logits.requires_grad_(False)
            self.history_decay_raw.requires_grad_(False)
            self._freeze_module(self.history_projection)
        if not config.use_time_decay_kernel:
            self.history_decay_raw.requires_grad_(False)
        if not config.hurdle_enabled:
            self.history_feature_logits.requires_grad_(False)
            self.history_decay_raw.requires_grad_(False)
            self._freeze_module(self.history_projection)
            self._freeze_module(self.progression_hurdle)
            self._freeze_module(self.progression_magnitude)

    def _history_descriptors(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.as_tensor(
            self.config.inflammation_feature_indices,
            device=values.device,
            dtype=torch.long,
        )
        # The final input patch is excluded: only strictly earlier inflammation
        # can influence the endpoint progression correction.
        observed_values = values[:, :-1].index_select(-1, indices)
        observed_mask = mask[:, :-1].index_select(-1, indices)
        signed = observed_values * self.inflammation_burden_sign[None, None, :]
        feature_weights = torch.softmax(self.history_feature_logits, dim=0)
        observed_weights = observed_mask * feature_weights[None, None, :]
        patch_burden = (signed * observed_weights).sum(dim=-1) / observed_weights.sum(
            dim=-1
        ).clamp_min(1e-6)
        patch_observed = (observed_mask.sum(dim=-1) > 0).to(values.dtype)

        endpoint_time = times[:, -1:]
        history_times = times[:, :-1]
        elapsed = (endpoint_time - history_times).clamp_min(0)
        if self.config.use_time_decay_kernel:
            decay = F.softplus(self.history_decay_raw) + 1e-4
            temporal = torch.exp(-decay * elapsed) * patch_observed
        else:
            temporal = patch_observed
        if self.config.history_mode == "current":
            temporal = torch.zeros_like(temporal)
            temporal[:, -1] = patch_observed[:, -1]
        normalized = temporal / temporal.sum(dim=1, keepdim=True).clamp_min(1e-6)
        history_auc = (normalized * patch_burden).sum(dim=1)

        if patch_burden.shape[1] > 1:
            interval = (history_times[:, -1] - history_times[:, 0]).clamp_min(1e-6)
            slope = (patch_burden[:, -1] - patch_burden[:, 0]) / interval
        else:
            slope = torch.zeros_like(history_auc)
        persistence = (
            torch.sigmoid(2.0 * patch_burden) * patch_observed
        ).sum(dim=1) / patch_observed.sum(dim=1).clamp_min(1.0)
        coverage = patch_observed.mean(dim=1)
        followup = (times[:, -1] - times[:, 0]).clamp_min(0)
        if self.config.treatment_conditioning:
            treatment_history = treatments[:, :-1].amax(dim=1)
        else:
            treatment_history = treatments.new_zeros(
                (treatments.shape[0], treatments.shape[-1])
            )
        descriptors = torch.cat(
            [
                history_auc[:, None],
                slope[:, None],
                persistence[:, None],
                coverage[:, None],
                torch.log1p(followup)[:, None],
                treatment_history,
            ],
            dim=-1,
        )
        if self.config.history_mode == "none":
            descriptors = torch.zeros_like(descriptors)
        elif self.config.history_mode == "permuted" and descriptors.shape[0] > 1:
            descriptors = torch.roll(descriptors, shifts=1, dims=0)
        return descriptors, normalized, elapsed

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
        phenotype_i, adapter_gate_i = self._patient_adapt(hidden_i, context_i, "i")
        trajectory_i = self._trajectory(phenotype_i.mean(dim=2), context_i, "i")

        hidden_c = self._masked_encode(
            values, mask, delta, self.calcification_feature_mask,
            cac_encoder=True,
        )
        if self.config.separate_cac_encoder:
            static_c = self.cac_static_context(static)
        elif self.config.isolate_cac_shared_gradient:
            hidden_c = hidden_c.detach()
            static_c = shared_static.detach()
        else:
            static_c = shared_static
        context_c = static_c + self.baseline_context_c(baseline_c)
        phenotype_c, adapter_gate_c = self._patient_adapt(hidden_c, context_c, "c")
        trajectory_c = self._trajectory(phenotype_c.mean(dim=2), context_c, "c")

        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        pool_c = torch.softmax(self.pool_c(trajectory_c).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (trajectory_c * pool_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context_i], dim=-1)).squeeze(-1)
        delta_log_cac_central = self.head_c(
            torch.cat([summary_c, context_c], dim=-1)
        ).squeeze(-1)

        descriptors, history_weights, history_elapsed = self._history_descriptors(
            values, mask, times, treatments
        )
        history_embedding = self.history_projection(descriptors)
        progression_input = torch.cat(
            [summary_c.detach(), context_c.detach(), history_embedding], dim=-1
        )
        progression_logit = self.progression_hurdle(progression_input).squeeze(-1)
        progression_probability = torch.sigmoid(progression_logit)
        progression_magnitude = F.softplus(
            self.progression_magnitude(progression_input).squeeze(-1)
        )
        if self.config.hurdle_enabled:
            lag_correction = progression_probability * progression_magnitude
        else:
            lag_correction = torch.zeros_like(delta_log_cac_central)
        delta_log_cac_final = delta_log_cac_central + lag_correction

        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0)) + delta_log_cac_final
            ).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_log_cac_final)

        batch, patches = values.shape[:2]
        max_lag = history_weights.shape[1]
        lag_attention = values.new_zeros((batch, patches, max_lag))
        lag_time_deltas = values.new_zeros((batch, patches, max_lag))
        lag_attention[:, -1] = history_weights
        lag_time_deltas[:, -1] = history_elapsed
        coupling_gate = values.new_zeros((batch, patches))
        coupling_gate[:, -1] = progression_probability
        zero_treatment = treatments.new_zeros(
            (batch, patches, max_lag, treatments.shape[-1])
        )

        return {
            "endpoint_tbr": endpoint_tbr,
            "endpoint_cac": endpoint_cac,
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac_final,
            "delta_tbr_median": delta_tbr,
            "delta_tbr_mean": delta_tbr,
            "delta_log_cac_median": delta_log_cac_central,
            "delta_log_cac_mean": delta_log_cac_final,
            "delta_log_cac_central": delta_log_cac_central,
            "delta_log_cac_raw_final": delta_log_cac_final,
            "lag_residual": progression_magnitude,
            "lag_residual_gate": progression_probability,
            "lag_correction": lag_correction,
            "history_descriptors": descriptors,
            "history_feature_weights": torch.softmax(
                self.history_feature_logits, dim=0
            ),
            "history_decay": F.softplus(self.history_decay_raw),
            "shared_representation": hidden_i,
            "phenotype_representation_inflammation": phenotype_i,
            "phenotype_representation_calcification": phenotype_c,
            "phenotype_gate_inflammation": adapter_gate_i,
            "phenotype_gate_calcification": adapter_gate_c,
            "trajectory_inflammation": trajectory_i,
            "trajectory_inflammation_lag_source": history_embedding[:, None, :],
            "trajectory_calcification_pre_coupling": trajectory_c,
            "trajectory_calcification": trajectory_c,
            "gate_inflammation_to_calcification": coupling_gate,
            "lag_attention": lag_attention,
            "lag_time_deltas": lag_time_deltas,
            "lag_interval_treatment": zero_treatment,
            "current_treatment_status": treatments,
            "patch_pool_inflammation": pool_i,
            "patch_pool_calcification": pool_c,
            "patch_pool_calcification_lag": pool_c,
            "cac_progression_logit": progression_logit,
            "cac_change_magnitude": progression_magnitude,
            "reconstruction": self.reconstruction_head(hidden_i).squeeze(-1),
        }
