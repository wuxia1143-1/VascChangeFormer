from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .lac import BottleneckMLP
from .lac_v2 import LowRankAdapter


@dataclass
class LACV21Config:
    """Configuration for the independently callable V2.1 refinement."""

    static_dim: int
    num_variables: int
    treatment_dim: int
    num_patches: int = 3
    hidden_dim: int = 32
    num_heads: int = 2
    num_layers: int = 1
    adapter_rank: int = 8
    coupling_rank_ic: int = 4
    max_lag: int = 2
    dropout: float = 0.10
    phenotype_adapters: bool = True
    lightweight_baseline_bias: bool = True
    baseline_anchoring: bool = True
    coupling_enabled: bool = True
    treatment_conditioning: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LACiTransformerV21(nn.Module):
    """V2.1 with task-decoupled readouts and forward-only I→C coupling.

    The TBR head reads only the pre-coupling inflammation trajectory, static
    covariates and baseline TBR. The CAC head reads the calcification
    trajectory after causal, lagged I→C updating, static covariates and
    baseline CAC. Treatment is used only inside the I→C attention and gate.
    """

    architecture_version = "V2.1"

    def __init__(self, config: LACV21Config):
        super().__init__()
        self.config = config
        h, d, p = config.hidden_dim, config.num_variables, config.num_patches
        if h % config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.max_lag < 1:
            raise ValueError("max_lag must be positive")

        self.token_projection = nn.Linear(3, h)
        self.variable_embedding = nn.Embedding(d, h)
        self.patch_embedding = nn.Embedding(p, h)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.num_heads,
            dim_feedforward=2 * h,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(h)

        # Shared/static information remains shared, while each task sees only
        # its own baseline anchor. Neither context contains treatment.
        self.static_context = BottleneckMLP(config.static_dim, h, config.dropout)
        self.baseline_context_i = nn.Linear(1, h, bias=False)
        self.baseline_context_c = nn.Linear(1, h, bias=False)

        self.q_i = nn.Parameter(torch.randn(h) * 0.02)
        self.q_c = nn.Parameter(torch.randn(h) * 0.02)
        self.baseline_query_bias_i = nn.Linear(1, h, bias=False)
        self.baseline_query_bias_c = nn.Linear(1, h, bias=False)
        self.adapter_i = LowRankAdapter(h, config.adapter_rank, config.dropout)
        self.adapter_c = LowRankAdapter(h, config.adapter_rank, config.dropout)
        self.adapter_gate_i = nn.Linear(h, h)
        self.adapter_gate_c = nn.Linear(h, h)

        self.update_gate_i = nn.Linear(3 * h, h)
        self.update_gate_c = nn.Linear(3 * h, h)
        self.update_candidate_i = nn.Linear(2 * h, h)
        self.update_candidate_c = nn.Linear(2 * h, h)
        self.init_i = nn.Linear(h, h)
        self.init_c = nn.Linear(h, h)

        rank = config.coupling_rank_ic
        self.ic_query = nn.Linear(h, rank, bias=False)
        self.ic_key = nn.Linear(h, rank, bias=False)
        self.ic_value_down = nn.Linear(h, rank, bias=False)
        self.ic_value_up = nn.Linear(rank, h, bias=False)
        self.ic_time_score = nn.Sequential(
            nn.Linear(2, max(2, rank)),
            nn.Tanh(),
            nn.Linear(max(2, rank), 1, bias=False),
        )
        self.ic_log_time_decay = nn.Parameter(torch.tensor(-1.0))
        self.interval_treatment_score = nn.Sequential(
            nn.Linear(config.treatment_dim + 2, max(4, rank)),
            nn.GELU(),
            nn.Linear(max(4, rank), 1, bias=False),
        )
        self.interval_treatment_encoder = nn.Linear(
            config.treatment_dim + 2,
            h,
            bias=False,
        )
        self.current_treatment_encoder = nn.Linear(
            config.treatment_dim,
            h,
            bias=False,
        )
        self.ic_gate = nn.Linear(5 * h, 1)

        self.pool_i = nn.Linear(h, 1)
        self.pool_c = nn.Linear(h, 1)
        self.head_i = BottleneckMLP(2 * h, 1, config.dropout)
        self.head_c = BottleneckMLP(2 * h, 1, config.dropout)
        self.reconstruction_head = nn.Linear(h, 1)

        # These buffers are fitted from the current training partition only.
        # Predictions remain in original residual units.
        self.register_buffer("target_center", torch.zeros(2))
        self.register_buffer("target_scale", torch.ones(2))
        self._freeze_structurally_unused_parameters()

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_structurally_unused_parameters(self) -> None:
        if not self.config.phenotype_adapters:
            for module in (
                self.adapter_i,
                self.adapter_c,
                self.adapter_gate_i,
                self.adapter_gate_c,
                self.baseline_query_bias_i,
                self.baseline_query_bias_c,
            ):
                self._freeze_module(module)
            self.q_i.requires_grad_(False)
            self.q_c.requires_grad_(False)
        elif not self.config.lightweight_baseline_bias:
            self._freeze_module(self.baseline_query_bias_i)
            self._freeze_module(self.baseline_query_bias_c)

        coupling_modules = (
            self.ic_query,
            self.ic_key,
            self.ic_value_down,
            self.ic_value_up,
            self.ic_time_score,
            self.interval_treatment_score,
            self.interval_treatment_encoder,
            self.current_treatment_encoder,
            self.ic_gate,
        )
        if not self.config.coupling_enabled:
            for module in coupling_modules:
                self._freeze_module(module)
            self.ic_log_time_decay.requires_grad_(False)
        elif not self.config.treatment_conditioning:
            for module in (
                self.interval_treatment_score,
                self.interval_treatment_encoder,
                self.current_treatment_encoder,
            ):
                self._freeze_module(module)

    def set_target_scaler(
        self,
        center: Sequence[float],
        scale: Sequence[float],
    ) -> None:
        center_tensor = torch.as_tensor(
            center,
            dtype=self.target_center.dtype,
            device=self.target_center.device,
        )
        scale_tensor = torch.as_tensor(
            scale,
            dtype=self.target_scale.dtype,
            device=self.target_scale.device,
        ).clamp_min(1e-6)
        if center_tensor.shape != (2,) or scale_tensor.shape != (2,):
            raise ValueError("Target center and scale must each contain two values")
        self.target_center.copy_(center_tensor)
        self.target_scale.copy_(scale_tensor)

    def _encode_tokens(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, variables = values.shape
        if (
            patches != self.config.num_patches
            or variables != self.config.num_variables
        ):
            raise ValueError(
                f"Expected [B,{self.config.num_patches},"
                f"{self.config.num_variables}], got {tuple(values.shape)}"
            )
        raw = torch.stack([values, mask, delta], dim=-1)
        variable_ids = torch.arange(variables, device=values.device)
        patch_ids = torch.arange(patches, device=values.device)
        tokens = self.token_projection(raw)
        tokens = tokens + self.variable_embedding(variable_ids)[None, None, :, :]
        tokens = tokens + self.patch_embedding(patch_ids)[None, :, None, :]
        encoded = [
            self.encoder(tokens[:, step])
            for step in range(patches)
        ]
        return self.encoder_norm(torch.stack(encoded, dim=1))

    def _phenotype_adapt(
        self,
        hidden: torch.Tensor,
        baseline_i: torch.Tensor,
        baseline_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.phenotype_adapters:
            zeros = hidden.new_zeros(hidden.shape)
            return hidden, hidden, zeros, zeros
        if self.config.lightweight_baseline_bias:
            bias_i = self.baseline_query_bias_i(baseline_i)
            bias_c = self.baseline_query_bias_c(baseline_c)
        else:
            bias_i = hidden.new_zeros((hidden.shape[0], self.config.hidden_dim))
            bias_c = hidden.new_zeros((hidden.shape[0], self.config.hidden_dim))
        query_i = self.q_i[None, None, None, :] + bias_i[:, None, None, :]
        query_c = self.q_c[None, None, None, :] + bias_c[:, None, None, :]
        gate_i = torch.sigmoid(self.adapter_gate_i(hidden) + query_i)
        gate_c = torch.sigmoid(self.adapter_gate_c(hidden) + query_c)
        phenotype_i = hidden + gate_i * self.adapter_i(hidden)
        phenotype_c = hidden + gate_c * self.adapter_c(hidden)
        return phenotype_i, phenotype_c, gate_i, gate_c

    def _trajectory(
        self,
        patch_repr: torch.Tensor,
        context: torch.Tensor,
        phenotype: str,
    ) -> torch.Tensor:
        gate_layer = self.update_gate_i if phenotype == "i" else self.update_gate_c
        candidate_layer = (
            self.update_candidate_i
            if phenotype == "i"
            else self.update_candidate_c
        )
        init_layer = self.init_i if phenotype == "i" else self.init_c
        previous = init_layer(context)
        states = []
        for step in range(patch_repr.shape[1]):
            current = patch_repr[:, step]
            gate = torch.sigmoid(
                gate_layer(torch.cat([current, previous, context], dim=-1))
            )
            candidate = torch.tanh(
                candidate_layer(torch.cat([current, context], dim=-1))
            )
            previous = gate * candidate + (1.0 - gate) * previous
            states.append(previous)
        return torch.stack(states, dim=1)

    def _lag_metadata(
        self,
        times: torch.Tensor,
        treatments: torch.Tensor,
        step: int,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lag_indices = range(start, step)
        time_delta = torch.stack(
            [
                (times[:, step] - times[:, lag]).clamp_min(0)
                for lag in lag_indices
            ],
            dim=1,
        )
        interval_exposure = torch.stack(
            [
                treatments[:, lag : step + 1].amax(dim=1)
                for lag in lag_indices
            ],
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

    def _forward_message(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        context_c: torch.Tensor,
        current_treatment: torch.Tensor,
        time_delta: torch.Tensor,
        interval_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = (
            self.ic_query(target).unsqueeze(1) * self.ic_key(source)
        ).sum(-1) / math.sqrt(self.config.coupling_rank_ic)
        time_features = torch.stack(
            [time_delta, torch.log1p(time_delta)],
            dim=-1,
        )
        score = score + self.ic_time_score(time_features).squeeze(-1)
        score = score - F.softplus(self.ic_log_time_decay) * time_delta
        if self.config.treatment_conditioning:
            score = score + self.interval_treatment_score(
                interval_features
            ).squeeze(-1)
            interval_encoded = self.interval_treatment_encoder(interval_features)
            current_encoded = self.current_treatment_encoder(current_treatment)
        else:
            interval_encoded = source.new_zeros(source.shape)
            current_encoded = target.new_zeros(target.shape)
        attention = torch.softmax(score, dim=1)
        message = (
            self.ic_value_up(self.ic_value_down(source))
            * attention.unsqueeze(-1)
        ).sum(dim=1)
        interval_summary = (
            interval_encoded * attention.unsqueeze(-1)
        ).sum(dim=1)
        gate = torch.sigmoid(
            self.ic_gate(
                torch.cat(
                    [
                        target,
                        message,
                        context_c,
                        current_encoded,
                        interval_summary,
                    ],
                    dim=-1,
                )
            )
        )
        return message, gate, attention

    def _couple_forward(
        self,
        trajectory_i: torch.Tensor,
        trajectory_c: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
        context_c: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, patches, _ = trajectory_i.shape
        coupled_c = []
        gate_values = []
        lag_weights = []
        lag_time_deltas = []
        lag_interval_treatments = []
        for step in range(patches):
            state_c = trajectory_c[:, step]
            padded_alpha = state_c.new_zeros((batch, self.config.max_lag))
            padded_time = padded_alpha.clone()
            padded_interval = state_c.new_zeros(
                (batch, self.config.max_lag, self.config.treatment_dim)
            )
            gate = state_c.new_zeros((batch, 1))
            if step and self.config.coupling_enabled:
                start = max(0, step - self.config.max_lag)
                time_delta, interval_exposure, interval_features = (
                    self._lag_metadata(times, treatments, step, start)
                )
                # Strictly earlier, pre-coupling inflammation states only.
                lag_i = trajectory_i[:, start:step]
                message, gate, alpha = self._forward_message(
                    lag_i,
                    state_c,
                    context_c,
                    treatments[:, step],
                    time_delta,
                    interval_features,
                )
                state_c = state_c + gate * message
                padded_alpha[:, -alpha.shape[1] :] = alpha
                padded_time[:, -time_delta.shape[1] :] = time_delta
                padded_interval[:, -interval_exposure.shape[1] :] = (
                    interval_exposure
                )
            coupled_c.append(state_c)
            gate_values.append(gate)
            lag_weights.append(padded_alpha)
            lag_time_deltas.append(padded_time)
            lag_interval_treatments.append(padded_interval)
        return (
            torch.stack(coupled_c, dim=1),
            torch.cat(gate_values, dim=1),
            torch.stack(lag_weights, dim=1),
            torch.stack(lag_time_deltas, dim=1),
            torch.stack(lag_interval_treatments, dim=1),
        )

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
        context_c = shared_static + self.baseline_context_c(baseline_c)
        phenotype_i, phenotype_c, adapter_gate_i, adapter_gate_c = (
            self._phenotype_adapt(hidden, baseline_i, baseline_c)
        )
        trajectory_i = self._trajectory(
            phenotype_i.mean(dim=2),
            context_i,
            "i",
        )
        trajectory_c_pre = self._trajectory(
            phenotype_c.mean(dim=2),
            context_c,
            "c",
        )
        (
            trajectory_c,
            gate_ic,
            lag_attention,
            lag_time_deltas,
            lag_interval_treatments,
        ) = self._couple_forward(
            trajectory_i,
            trajectory_c_pre,
            times,
            treatments,
            context_c,
        )

        # TBR is read before all cross-phenotype coupling.
        pool_weight_i = torch.softmax(
            self.pool_i(trajectory_i).squeeze(-1),
            dim=1,
        )
        pool_weight_c = torch.softmax(
            self.pool_c(trajectory_c).squeeze(-1),
            dim=1,
        )
        summary_i = (
            trajectory_i * pool_weight_i.unsqueeze(-1)
        ).sum(dim=1)
        summary_c = (
            trajectory_c * pool_weight_c.unsqueeze(-1)
        ).sum(dim=1)
        delta_tbr = self.head_i(
            torch.cat([summary_i, context_i], dim=-1)
        ).squeeze(-1)
        delta_log_cac = self.head_c(
            torch.cat([summary_c, context_c], dim=-1)
        ).squeeze(-1)
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
            "shared_representation": hidden,
            "phenotype_representation_inflammation": phenotype_i,
            "phenotype_representation_calcification": phenotype_c,
            "phenotype_gate_inflammation": adapter_gate_i,
            "phenotype_gate_calcification": adapter_gate_c,
            "trajectory_inflammation": trajectory_i,
            "trajectory_calcification_pre_coupling": trajectory_c_pre,
            "trajectory_calcification": trajectory_c,
            "gate_inflammation_to_calcification": gate_ic,
            "lag_attention": lag_attention,
            "lag_time_deltas": lag_time_deltas,
            "lag_interval_treatment": lag_interval_treatments,
            "current_treatment_status": treatments,
            "patch_pool_inflammation": pool_weight_i,
            "patch_pool_calcification": pool_weight_c,
            "reconstruction": self.reconstruction_head(hidden).squeeze(-1),
        }
