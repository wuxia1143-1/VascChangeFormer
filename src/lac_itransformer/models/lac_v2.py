from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .lac import BottleneckMLP


@dataclass
class LACV2Config:
    """Configuration for the independent, lightweight V2 architecture."""

    static_dim: int
    num_variables: int
    treatment_dim: int
    num_patches: int = 3
    hidden_dim: int = 32
    num_heads: int = 2
    num_layers: int = 1
    adapter_rank: int = 8
    coupling_rank_ic: int = 4
    coupling_rank_ci: int = 2
    max_lag: int = 2
    dropout: float = 0.10
    phenotype_adapters: bool = True
    lightweight_baseline_bias: bool = True
    baseline_anchoring: bool = True
    coupling_mode: str = "asymmetric"
    treatment_conditioning: bool = True
    reverse_scale: float = 0.10
    reverse_gate_bias: float = 2.0
    competitive_routing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(F.gelu(self.down(value))))


class LACiTransformerV2(nn.Module):
    """Non-exclusive phenotype-adapter LAC-iTransformer V2.

    The returned coupling tensors describe learned predictive associations.
    They are not causal-effect estimates.
    """

    architecture_version = "V2"

    def __init__(self, config: LACV2Config):
        super().__init__()
        self.config = config
        h, d, p = config.hidden_dim, config.num_variables, config.num_patches
        if h % config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.coupling_mode not in {
            "asymmetric",
            "none",
            "symmetric",
            "ic_only",
            "ci_only",
        }:
            raise ValueError(f"Unsupported coupling_mode: {config.coupling_mode}")
        if config.max_lag < 1:
            raise ValueError("max_lag must be positive")
        if config.coupling_rank_ci >= config.coupling_rank_ic:
            raise ValueError(
                "V2 reverse coupling rank must be lower than the forward rank"
            )
        if not 0.0 <= config.reverse_scale < 1.0:
            raise ValueError("V2 reverse_scale must be in [0, 1)")

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

        # Shared H carries common/background information. There is deliberately
        # no background adapter, route, parameter, or auxiliary loss in V2.
        self.context_encoder = BottleneckMLP(config.static_dim + 2, h, config.dropout)
        self.q_i = nn.Parameter(torch.randn(h) * 0.02)
        self.q_c = nn.Parameter(torch.randn(h) * 0.02)
        self.baseline_query_bias = nn.Linear(2, 2 * h, bias=False)
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

        ric, rci = config.coupling_rank_ic, config.coupling_rank_ci
        self.ic_query = nn.Linear(h, ric, bias=False)
        self.ic_key = nn.Linear(h, ric, bias=False)
        self.ic_value_down = nn.Linear(h, ric, bias=False)
        self.ic_value_up = nn.Linear(ric, h, bias=False)
        self.ic_time_score = nn.Sequential(
            nn.Linear(2, max(2, ric)),
            nn.Tanh(),
            nn.Linear(max(2, ric), 1, bias=False),
        )
        self.ic_log_time_decay = nn.Parameter(torch.tensor(-1.0))

        # Treatment appears only in the dominant lag attention and its gate.
        # Interval exposure and current status have separate encoders.
        self.interval_treatment_score = nn.Sequential(
            nn.Linear(config.treatment_dim + 2, max(4, ric)),
            nn.GELU(),
            nn.Linear(max(4, ric), 1, bias=False),
        )
        self.interval_treatment_encoder = nn.Linear(
            config.treatment_dim + 2, h, bias=False
        )
        self.current_treatment_encoder = nn.Linear(
            config.treatment_dim, h, bias=False
        )
        self.ic_gate = nn.Linear(5 * h, 1)

        # The exploratory reverse path is intentionally lower-rank, sparse and
        # scaled. It receives no additional treatment controller.
        self.ci_value_down = nn.Linear(h, rci, bias=False)
        self.ci_value_up = nn.Linear(rci, h, bias=False)
        self.ci_gate = nn.Linear(3 * h, 1)

        self.pool_i = nn.Linear(h, 1)
        self.pool_c = nn.Linear(h, 1)
        self.head_i = BottleneckMLP(2 * h, 1, config.dropout)
        self.head_c = BottleneckMLP(2 * h, 1, config.dropout)
        self.reconstruction_head = nn.Linear(h, 1)
        self.log_var_tbr = nn.Parameter(torch.zeros(()))
        self.log_var_cac = nn.Parameter(torch.zeros(()))
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
                self.baseline_query_bias,
            ):
                self._freeze_module(module)
            self.q_i.requires_grad_(False)
            self.q_c.requires_grad_(False)
        elif not self.config.lightweight_baseline_bias:
            self._freeze_module(self.baseline_query_bias)

        ic_modules = (
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
        ci_modules = (self.ci_value_down, self.ci_value_up, self.ci_gate)
        if self.config.coupling_mode in {"none", "ci_only"}:
            for module in ic_modules:
                self._freeze_module(module)
            self.ic_log_time_decay.requires_grad_(False)
        if self.config.coupling_mode in {"none", "ic_only", "symmetric"}:
            for module in ci_modules:
                self._freeze_module(module)
        if (
            not self.config.treatment_conditioning
            or self.config.coupling_mode in {"none", "ci_only"}
        ):
            for module in (
                self.interval_treatment_score,
                self.interval_treatment_encoder,
                self.current_treatment_encoder,
            ):
                self._freeze_module(module)

    def _encode_tokens(
        self, values: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        b, p, d = values.shape
        if p != self.config.num_patches or d != self.config.num_variables:
            raise ValueError(
                f"Expected [B,{self.config.num_patches},{self.config.num_variables}], "
                f"got {tuple(values.shape)}"
            )
        raw = torch.stack([values, mask, delta], dim=-1)
        var_ids = torch.arange(d, device=values.device)
        patch_ids = torch.arange(p, device=values.device)
        tokens = self.token_projection(raw)
        tokens = tokens + self.variable_embedding(var_ids)[None, None, :, :]
        tokens = tokens + self.patch_embedding(patch_ids)[None, :, None, :]
        encoded = [self.encoder(tokens[:, step]) for step in range(p)]
        return self.encoder_norm(torch.stack(encoded, dim=1))

    def _phenotype_adapt(
        self, hidden: torch.Tensor, baseline_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.phenotype_adapters:
            zero_gate = hidden.new_zeros(hidden.shape)
            return hidden, hidden, zero_gate, zero_gate
        if self.config.lightweight_baseline_bias:
            bias_i, bias_c = self.baseline_query_bias(baseline_context).chunk(2, dim=-1)
        else:
            bias_i = bias_c = torch.zeros_like(baseline_context[:, :1]).expand(
                -1, self.config.hidden_dim
            )
        query_i = self.q_i[None, None, None, :] + bias_i[:, None, None, :]
        query_c = self.q_c[None, None, None, :] + bias_c[:, None, None, :]
        gate_i = torch.sigmoid(self.adapter_gate_i(hidden) + query_i)
        gate_c = torch.sigmoid(self.adapter_gate_c(hidden) + query_c)
        phenotype_i = hidden + gate_i * self.adapter_i(hidden)
        phenotype_c = hidden + gate_c * self.adapter_c(hidden)
        return phenotype_i, phenotype_c, gate_i, gate_c

    def _trajectory(
        self, patch_repr: torch.Tensor, context: torch.Tensor, phenotype: str
    ) -> torch.Tensor:
        gate_layer = self.update_gate_i if phenotype == "i" else self.update_gate_c
        candidate_layer = (
            self.update_candidate_i if phenotype == "i" else self.update_candidate_c
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
            [(times[:, step] - times[:, lag]).clamp_min(0) for lag in lag_indices],
            dim=1,
        )
        interval_exposure = torch.stack(
            [treatments[:, lag : step + 1].amax(dim=1) for lag in lag_indices],
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

    def _dominant_message(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor,
        current_treatment: torch.Tensor,
        time_delta: torch.Tensor,
        interval_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = (
            self.ic_query(target).unsqueeze(1) * self.ic_key(source)
        ).sum(-1) / math.sqrt(self.config.coupling_rank_ic)
        time_features = torch.stack([time_delta, torch.log1p(time_delta)], dim=-1)
        score = score + self.ic_time_score(time_features).squeeze(-1)
        score = score - F.softplus(self.ic_log_time_decay) * time_delta
        if self.config.treatment_conditioning:
            score = score + self.interval_treatment_score(interval_features).squeeze(-1)
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
                        context,
                        current_encoded,
                        interval_summary,
                    ],
                    dim=-1,
                )
            )
        )
        return message, gate, attention

    def _couple(
        self,
        trajectory_i: torch.Tensor,
        trajectory_c: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, patches, _ = trajectory_i.shape
        coupled_i: list[torch.Tensor] = []
        coupled_c: list[torch.Tensor] = []
        gate_ic_values, gate_ci_values = [], []
        lag_weights, reverse_lag_weights = [], []
        lag_time_deltas, lag_interval_treatments = [], []
        for step in range(patches):
            state_i, state_c = trajectory_i[:, step], trajectory_c[:, step]
            padded_alpha = state_i.new_zeros((batch, self.config.max_lag))
            padded_reverse_alpha = padded_alpha.clone()
            padded_time = padded_alpha.clone()
            padded_interval = state_i.new_zeros(
                (batch, self.config.max_lag, self.config.treatment_dim)
            )
            gate_ic = state_i.new_zeros((batch, 1))
            gate_ci = state_i.new_zeros((batch, 1))
            if step:
                start = max(0, step - self.config.max_lag)
                time_delta, interval_exposure, interval_features = self._lag_metadata(
                    times, treatments, step, start
                )
                if self.config.coupling_mode not in {"none", "ci_only"}:
                    lag_i = torch.stack(coupled_i[start:step], dim=1)
                    message_ic, gate_ic, alpha = self._dominant_message(
                        lag_i,
                        state_c,
                        context,
                        treatments[:, step],
                        time_delta,
                        interval_features,
                    )
                    state_c = state_c + gate_ic * message_ic
                    padded_alpha[:, -alpha.shape[1] :] = alpha
                if self.config.coupling_mode == "symmetric":
                    lag_c = torch.stack(coupled_c[start:step], dim=1)
                    message_ci, gate_ci, reverse_alpha = self._dominant_message(
                        lag_c,
                        state_i,
                        context,
                        treatments[:, step],
                        time_delta,
                        interval_features,
                    )
                    state_i = state_i + gate_ci * message_ci
                    padded_reverse_alpha[:, -reverse_alpha.shape[1] :] = reverse_alpha
                elif self.config.coupling_mode not in {"none", "ic_only"}:
                    previous_c = coupled_c[-1]
                    message_ci = self.ci_value_up(self.ci_value_down(previous_c))
                    raw_gate_ci = torch.sigmoid(
                        self.ci_gate(
                            torch.cat([state_i, message_ci, context], dim=-1)
                        )
                        - self.config.reverse_gate_bias
                    )
                    gate_ci = self.config.reverse_scale * raw_gate_ci
                    state_i = state_i + gate_ci * message_ci
                    padded_reverse_alpha[:, -1] = 1.0
                padded_time[:, -time_delta.shape[1] :] = time_delta
                padded_interval[:, -interval_exposure.shape[1] :] = interval_exposure
            coupled_i.append(state_i)
            coupled_c.append(state_c)
            gate_ic_values.append(gate_ic)
            gate_ci_values.append(gate_ci)
            lag_weights.append(padded_alpha)
            reverse_lag_weights.append(padded_reverse_alpha)
            lag_time_deltas.append(padded_time)
            lag_interval_treatments.append(padded_interval)
        return (
            torch.stack(coupled_i, dim=1),
            torch.stack(coupled_c, dim=1),
            torch.cat(gate_ic_values, dim=1),
            torch.cat(gate_ci_values, dim=1),
            torch.stack(lag_weights, dim=1),
            torch.stack(reverse_lag_weights, dim=1),
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
            baseline_context = torch.stack(
                [baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))],
                dim=-1,
            )
        else:
            baseline_context = torch.zeros_like(baseline)
        context = self.context_encoder(
            torch.cat([static, baseline_context], dim=-1)
        )
        phenotype_i, phenotype_c, adapter_gate_i, adapter_gate_c = (
            self._phenotype_adapt(hidden, baseline_context)
        )
        patch_i = phenotype_i.mean(dim=2)
        patch_c = phenotype_c.mean(dim=2)
        trajectory_i = self._trajectory(patch_i, context, "i")
        trajectory_c = self._trajectory(patch_c, context, "c")
        (
            coupled_i,
            coupled_c,
            gate_ic,
            gate_ci,
            lag_attention,
            reverse_lag_attention,
            lag_time_deltas,
            lag_interval_treatments,
        ) = self._couple(
            trajectory_i, trajectory_c, times, treatments, context
        )

        pool_weight_i = torch.softmax(
            self.pool_i(coupled_i).squeeze(-1), dim=1
        )
        pool_weight_c = torch.softmax(
            self.pool_c(coupled_c).squeeze(-1), dim=1
        )
        summary_i = (coupled_i * pool_weight_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (coupled_c * pool_weight_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(
            torch.cat([summary_i, context], dim=-1)
        ).squeeze(-1)
        delta_log_cac = self.head_c(
            torch.cat([summary_c, context], dim=-1)
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
            "trajectory_inflammation": coupled_i,
            "trajectory_calcification": coupled_c,
            "gate_inflammation_to_calcification": gate_ic,
            "gate_calcification_to_inflammation": gate_ci,
            "lag_attention": lag_attention,
            "reverse_lag_attention": reverse_lag_attention,
            "lag_time_deltas": lag_time_deltas,
            "lag_interval_treatment": lag_interval_treatments,
            "current_treatment_status": treatments,
            "patch_pool_inflammation": pool_weight_i,
            "patch_pool_calcification": pool_weight_c,
            "reconstruction": self.reconstruction_head(hidden).squeeze(-1),
        }
