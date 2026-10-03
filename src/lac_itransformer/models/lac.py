from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LACConfig:
    static_dim: int
    num_variables: int
    treatment_dim: int
    num_patches: int = 3
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    route_rank: int = 16
    coupling_rank_ic: int = 8
    coupling_rank_ci: int = 4
    max_lag: int = 2
    dropout: float = 0.15
    competitive_routing: bool = True
    background_route: bool = True
    dynamic_anchors: bool = True
    baseline_anchoring: bool = True
    coupling_mode: str = "asymmetric"
    treatment_conditioning: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BottleneckMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        bottleneck = max(8, min(output_dim, input_dim // 2 if input_dim > 16 else input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LACiTransformer(nn.Module):
    """Lag-aware Asymmetric Coupled iTransformer.

    Route/coupling tensors are returned for prespecified interpretability and
    robustness experiments. They encode learned directional association and
    must not be interpreted as causal effects.
    """

    def __init__(self, config: LACConfig):
        super().__init__()
        self.config = config
        h, d, p = config.hidden_dim, config.num_variables, config.num_patches
        if h % config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.coupling_mode not in {"asymmetric", "none", "symmetric", "ic_only", "ci_only"}:
            raise ValueError(f"Unsupported coupling_mode: {config.coupling_mode}")

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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(h)

        self.context_encoder = BottleneckMLP(config.static_dim + 2, h, config.dropout)
        self.treatment_encoder = BottleneckMLP(config.treatment_dim + 1, h, config.dropout)
        self.anchor_shared = BottleneckMLP(2 * h, h, config.dropout)
        self.inflammation_anchor = nn.Linear(h, h)
        self.calcification_anchor = nn.Linear(h, h)
        self.static_anchor_i = nn.Parameter(torch.randn(h) * 0.02)
        self.static_anchor_c = nn.Parameter(torch.randn(h) * 0.02)

        r = config.route_rank
        self.route_token = nn.Linear(h, r, bias=False)
        self.route_anchor_i = nn.Linear(h, r, bias=False)
        self.route_anchor_c = nn.Linear(h, r, bias=False)
        self.route_background = nn.Linear(h, 1)
        self.route_bias_i = nn.Parameter(torch.zeros(d))
        self.route_bias_c = nn.Parameter(torch.zeros(d))

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
        self.ic_treatment_score = nn.Linear(h, 1, bias=False)
        self.ic_gate = nn.Linear(4 * h, 1)
        self.ci_value_down = nn.Linear(h, rci, bias=False)
        self.ci_value_up = nn.Linear(rci, h, bias=False)
        self.ci_gate = nn.Linear(4 * h, 1)

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
        """Keep ablation parameter counts aligned with the executed graph."""
        if self.config.competitive_routing:
            if self.config.dynamic_anchors:
                self.static_anchor_i.requires_grad_(False)
                self.static_anchor_c.requires_grad_(False)
            else:
                self._freeze_module(self.anchor_shared)
                self._freeze_module(self.inflammation_anchor)
                self._freeze_module(self.calcification_anchor)
            if not self.config.background_route:
                self._freeze_module(self.route_background)
        else:
            for module in (
                self.anchor_shared,
                self.inflammation_anchor,
                self.calcification_anchor,
                self.route_token,
                self.route_anchor_i,
                self.route_anchor_c,
                self.route_background,
            ):
                self._freeze_module(module)
            for parameter in (
                self.static_anchor_i,
                self.static_anchor_c,
                self.route_bias_i,
                self.route_bias_c,
            ):
                parameter.requires_grad_(False)
        ic_modules = (
            self.ic_query,
            self.ic_key,
            self.ic_value_down,
            self.ic_value_up,
            self.ic_treatment_score,
            self.ic_gate,
        )
        ci_modules = (self.ci_value_down, self.ci_value_up, self.ci_gate)
        if self.config.coupling_mode in {"none", "ci_only"}:
            for module in ic_modules:
                self._freeze_module(module)
        if self.config.coupling_mode in {"none", "ic_only", "symmetric"}:
            for module in ci_modules:
                self._freeze_module(module)
        if self.config.coupling_mode == "none" or not self.config.treatment_conditioning:
            self._freeze_module(self.treatment_encoder)
        if not self.config.treatment_conditioning:
            self._freeze_module(self.ic_treatment_score)

    def _encode_tokens(self, values: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        b, p, d = values.shape
        if p != self.config.num_patches or d != self.config.num_variables:
            raise ValueError(f"Expected [B,{self.config.num_patches},{self.config.num_variables}], got {tuple(values.shape)}")
        raw = torch.stack([values, mask, delta], dim=-1)
        var_ids = torch.arange(d, device=values.device)
        patch_ids = torch.arange(p, device=values.device)
        tokens = self.token_projection(raw)
        tokens = tokens + self.variable_embedding(var_ids)[None, None, :, :]
        tokens = tokens + self.patch_embedding(patch_ids)[None, :, None, :]
        encoded = []
        for step in range(p):
            encoded.append(self.encoder(tokens[:, step]))
        return self.encoder_norm(torch.stack(encoded, dim=1))

    @staticmethod
    def _weighted_patch(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        numerator = (hidden * weights.unsqueeze(-1)).sum(dim=2)
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return numerator / denominator

    def _trajectory(self, patch_repr: torch.Tensor, context: torch.Tensor, phenotype: str) -> torch.Tensor:
        gate_layer = self.update_gate_i if phenotype == "i" else self.update_gate_c
        candidate_layer = self.update_candidate_i if phenotype == "i" else self.update_candidate_c
        init_layer = self.init_i if phenotype == "i" else self.init_c
        previous = init_layer(context)
        states = []
        for step in range(patch_repr.shape[1]):
            current = patch_repr[:, step]
            gate = torch.sigmoid(gate_layer(torch.cat([current, previous, context], dim=-1)))
            candidate = torch.tanh(candidate_layer(torch.cat([current, context], dim=-1)))
            previous = gate * candidate + (1.0 - gate) * previous
            states.append(previous)
        return torch.stack(states, dim=1)

    def _couple(
        self,
        trajectory_i: torch.Tensor,
        trajectory_c: torch.Tensor,
        treatment: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = trajectory_i.shape[1]
        coupled_i, coupled_c = [], []
        gate_ic_values, gate_ci_values, lag_weights = [], [], []
        treatment_for_coupling = treatment if self.config.treatment_conditioning else torch.zeros_like(treatment)
        for step in range(p):
            state_i, state_c = trajectory_i[:, step], trajectory_c[:, step]
            if step:
                start = max(0, step - self.config.max_lag)
                lag_i = torch.stack(coupled_i[start:step], dim=1)
                query = self.ic_query(state_c).unsqueeze(1)
                score = (query * self.ic_key(lag_i)).sum(-1) / math.sqrt(self.config.coupling_rank_ic)
                score = score + self.ic_treatment_score(treatment_for_coupling[:, step]).expand_as(score)
                alpha = torch.softmax(score, dim=1)
                u_ic = (self.ic_value_up(self.ic_value_down(lag_i)) * alpha.unsqueeze(-1)).sum(dim=1)
                gate_ic = torch.sigmoid(self.ic_gate(torch.cat([state_c, u_ic, treatment_for_coupling[:, step], context], dim=-1)))

                previous_c = coupled_c[-1]
                u_ci = self.ci_value_up(self.ci_value_down(previous_c))
                gate_ci = torch.sigmoid(self.ci_gate(torch.cat([state_i, u_ci, treatment_for_coupling[:, step], context], dim=-1)))
                if self.config.coupling_mode == "none":
                    gate_ic = torch.zeros_like(gate_ic)
                    gate_ci = torch.zeros_like(gate_ci)
                elif self.config.coupling_mode == "ic_only":
                    gate_ci = torch.zeros_like(gate_ci)
                elif self.config.coupling_mode == "ci_only":
                    gate_ic = torch.zeros_like(gate_ic)
                elif self.config.coupling_mode == "symmetric":
                    # Strictly symmetric comparator: both directions share the
                    # same lag attention, value transform and gate architecture.
                    lag_c = torch.stack(coupled_c[start:step], dim=1)
                    reverse_query = self.ic_query(state_i).unsqueeze(1)
                    reverse_score = (
                        reverse_query * self.ic_key(lag_c)
                    ).sum(-1) / math.sqrt(self.config.coupling_rank_ic)
                    reverse_score = reverse_score + self.ic_treatment_score(
                        treatment_for_coupling[:, step]
                    ).expand_as(reverse_score)
                    reverse_alpha = torch.softmax(reverse_score, dim=1)
                    u_ci = (
                        self.ic_value_up(self.ic_value_down(lag_c))
                        * reverse_alpha.unsqueeze(-1)
                    ).sum(dim=1)
                    gate_ci = torch.sigmoid(
                        self.ic_gate(
                            torch.cat(
                                [state_i, u_ci, treatment_for_coupling[:, step], context],
                                dim=-1,
                            )
                        )
                    )
                state_c = state_c + gate_ic * u_ic
                state_i = state_i + gate_ci * u_ci
                padded_alpha = F.pad(alpha, (self.config.max_lag - alpha.shape[1], 0))
            else:
                gate_ic = torch.zeros((state_i.shape[0], 1), device=state_i.device)
                gate_ci = torch.zeros_like(gate_ic)
                padded_alpha = torch.zeros((state_i.shape[0], self.config.max_lag), device=state_i.device)
            coupled_i.append(state_i)
            coupled_c.append(state_c)
            gate_ic_values.append(gate_ic)
            gate_ci_values.append(gate_ci)
            lag_weights.append(padded_alpha)
        return (
            torch.stack(coupled_i, dim=1),
            torch.stack(coupled_c, dim=1),
            torch.cat(gate_ic_values, dim=1),
            torch.cat(gate_ci_values, dim=1),
            torch.stack(lag_weights, dim=1),
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
                [baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))], dim=-1
            )
        else:
            # The ablation removes both residual anchoring and baseline endpoint
            # access, so no TBR/CAC baseline signal can leak through context.
            baseline_context = torch.zeros_like(baseline)
        context = self.context_encoder(torch.cat([static, baseline_context], dim=-1))
        time_gap = torch.cat([times[:, :1], times[:, 1:] - times[:, :-1]], dim=1).unsqueeze(-1)
        treatment = self.treatment_encoder(torch.cat([treatments, time_gap], dim=-1))

        mean_hidden = hidden.mean(dim=2)
        anchor_seed = self.anchor_shared(torch.cat([mean_hidden, context[:, None, :].expand_as(mean_hidden)], dim=-1))
        if self.config.dynamic_anchors:
            anchor_i = self.inflammation_anchor(anchor_seed)
            anchor_c = self.calcification_anchor(anchor_seed)
        else:
            anchor_i = self.static_anchor_i[None, None, :].expand_as(anchor_seed)
            anchor_c = self.static_anchor_c[None, None, :].expand_as(anchor_seed)
        route_tokens = self.route_token(hidden)
        score_i = (route_tokens * self.route_anchor_i(anchor_i).unsqueeze(2)).sum(-1) / math.sqrt(self.config.route_rank)
        score_c = (route_tokens * self.route_anchor_c(anchor_c).unsqueeze(2)).sum(-1) / math.sqrt(self.config.route_rank)
        score_i = score_i + self.route_bias_i[None, None, :]
        score_c = score_c + self.route_bias_c[None, None, :]
        score_b = self.route_background(hidden).squeeze(-1)
        if not self.config.competitive_routing:
            routes = torch.zeros((*score_i.shape, 3), device=score_i.device, dtype=score_i.dtype)
            routes[..., 0] = 0.5
            routes[..., 1] = 0.5
        elif not self.config.background_route:
            foreground = torch.softmax(torch.stack([score_i, score_c], dim=-1), dim=-1)
            routes = torch.cat([foreground, torch.zeros_like(foreground[..., :1])], dim=-1)
        else:
            routes = torch.softmax(torch.stack([score_i, score_c, score_b], dim=-1), dim=-1)
        patch_i = self._weighted_patch(hidden, routes[..., 0])
        patch_c = self._weighted_patch(hidden, routes[..., 1])
        trajectory_i = self._trajectory(patch_i, context, "i")
        trajectory_c = self._trajectory(patch_c, context, "c")
        coupled_i, coupled_c, gate_ic, gate_ci, lag_attention = self._couple(trajectory_i, trajectory_c, treatment, context)

        pool_weight_i = torch.softmax(self.pool_i(coupled_i).squeeze(-1), dim=1)
        pool_weight_c = torch.softmax(self.pool_c(coupled_c).squeeze(-1), dim=1)
        summary_i = (coupled_i * pool_weight_i.unsqueeze(-1)).sum(dim=1)
        summary_c = (coupled_c * pool_weight_c.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context], dim=-1)).squeeze(-1)
        delta_log_cac = self.head_c(torch.cat([summary_c, context], dim=-1)).squeeze(-1)
        if self.config.baseline_anchoring:
            endpoint_tbr = baseline[:, 0] + delta_tbr
            endpoint_cac = torch.expm1(torch.log1p(baseline[:, 1].clamp_min(0)) + delta_log_cac).clamp_min(0)
        else:
            endpoint_tbr = delta_tbr
            endpoint_cac = F.softplus(delta_log_cac)
        reconstruction = self.reconstruction_head(hidden).squeeze(-1)
        return {
            "endpoint_tbr": endpoint_tbr,
            "endpoint_cac": endpoint_cac,
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac,
            "routes": routes,
            "route_inflammation": routes[..., 0],
            "route_calcification": routes[..., 1],
            "route_background": routes[..., 2],
            "trajectory_inflammation": coupled_i,
            "trajectory_calcification": coupled_c,
            "gate_inflammation_to_calcification": gate_ic,
            "gate_calcification_to_inflammation": gate_ci,
            "lag_attention": lag_attention,
            "patch_pool_inflammation": pool_weight_i,
            "patch_pool_calcification": pool_weight_c,
            "reconstruction": reconstruction,
        }
