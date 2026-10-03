from __future__ import annotations

"""Task-adapted implementations of the prespecified published comparators.

The architectural cores follow the source-locked upstream implementations in
``configs/upstreams.lock.yaml``.  All models expose the same residual TBR/CAC
contract so that data folds, preprocessing, losses, and metrics stay fixed.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .lac import LACConfig


class EndpointAdapter(nn.Module):
    """Common baseline-anchored two-endpoint output contract."""

    def __init__(self, config: LACConfig):
        super().__init__()
        self.config = config
        self.log_var_tbr = nn.Parameter(torch.zeros(()))
        self.log_var_cac = nn.Parameter(torch.zeros(()))

    def endpoint_output(self, baseline: torch.Tensor, residuals: torch.Tensor, **extra):
        delta_tbr, delta_log_cac = residuals[:, 0], residuals[:, 1]
        return {
            "endpoint_tbr": baseline[:, 0] + delta_tbr,
            "endpoint_cac": torch.expm1(
                torch.log1p(baseline[:, 1].clamp_min(0)) + delta_log_cac
            ).clamp_min(0),
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_log_cac,
        } | extra


class _TimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("time encoding dimension must be at least 2")
        self.scale = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, dim - 1)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.scale(time), torch.sin(self.periodic(time))], dim=-1)


class _AdaptivePatchAggregation(nn.Module):
    """APN TAPA module adapted from decisionintelligence/APN."""

    def __init__(self, variables: int, patches: int, time_dim: int, hidden: int, dropout: float):
        super().__init__()
        width = 1.0 / patches
        self.variables, self.patches = variables, patches
        self.left_offsets = nn.Parameter(torch.zeros(variables, patches))
        self.log_widths = nn.Parameter(torch.full((variables, patches), math.log(width)))
        self.temperatures = nn.Parameter(torch.zeros(variables))
        self.projection = nn.Linear(1 + time_dim, hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, times: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # inputs are stacked as [B*N,L,*], preserving observed times and masks
        bn, _, _ = times.shape
        batch = bn // self.variables
        width = 1.0 / self.patches
        centers = torch.linspace(width / 2, 1.0 - width / 2, self.patches, device=times.device)
        left = centers[None] - width / 2 + self.left_offsets
        right = left + torch.exp(self.log_widths).clamp_min(1e-6)
        tau = F.softplus(self.temperatures).clamp_min(1e-6)
        left = left[None].expand(batch, -1, -1).reshape(bn, self.patches, 1)
        right = right[None].expand(batch, -1, -1).reshape(bn, self.patches, 1)
        tau = tau[None, :, None].expand(batch, -1, -1).reshape(bn, 1, 1)
        raw_time = times.transpose(1, 2)
        weights = torch.sigmoid((right - raw_time) / tau) * torch.sigmoid((raw_time - left) / tau)
        weights = weights * mask.transpose(1, 2)
        pooled = torch.bmm(weights, features) / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        projected = self.projection(pooled)
        return self.norm(projected + self.ffn(projected))


class APNDRBaseline(EndpointAdapter):
    """Official APN adaptive patching/query core with dual residual heads."""

    upstream_key = "apn"

    def __init__(self, config: LACConfig, time_dim: int = 8):
        super().__init__(config)
        h, d, p = config.hidden_dim, config.num_variables, config.num_patches
        self.time_encoding = _TimeEncoding(time_dim)
        self.patching = _AdaptivePatchAggregation(d, p, time_dim, h, config.dropout)
        self.position = nn.Parameter(torch.randn(1, 1, p, h) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, d, 1, h) * 0.02)
        self.query_norm = nn.LayerNorm(h)
        context_dim = config.static_dim + 2 + config.treatment_dim
        self.context = nn.Sequential(nn.Linear(context_dim, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, h))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 2))

    def forward(
        self, static, baseline, values, mask, times, treatments,
        irregular_values=None, irregular_mask=None, irregular_times=None, **_,
    ):
        if irregular_values is not None:
            values = irregular_values
            mask = irregular_mask
            times = irregular_times
        batch, length, variables = values.shape
        # The shared real-data contract stores elapsed years so LAC V2 can use
        # true lag intervals. APN's official adaptive patch coordinates are
        # relative [0, 1], so only APN normalizes its private patching copy.
        patch_times = times / times.amax(dim=1, keepdim=True).clamp_min(1e-6)
        stacked_values = values.permute(0, 2, 1).reshape(batch * variables, length, 1)
        stacked_mask = mask.permute(0, 2, 1).reshape(batch * variables, length, 1)
        stacked_times = patch_times[:, None, :].expand(-1, variables, -1).reshape(batch * variables, length, 1)
        encoded_time = self.time_encoding(stacked_times)
        patches = self.patching(stacked_times, torch.cat([stacked_values, encoded_time], -1), stacked_mask)
        patches = patches.reshape(batch, variables, self.config.num_patches, -1) + self.position
        score = torch.matmul(self.queries.expand(batch, -1, -1, -1), patches.transpose(-1, -2))
        score = score / math.sqrt(self.config.hidden_dim)
        query_weights = torch.softmax(score, dim=-1)
        variable_repr = self.query_norm(torch.matmul(query_weights, patches).squeeze(2))
        sequence_repr = variable_repr.mean(dim=1)
        base = torch.stack([baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))], -1)
        context = self.context(torch.cat([static, base, treatments.mean(dim=1)], -1))
        return self.endpoint_output(
            baseline,
            self.head(torch.cat([sequence_repr, context], -1)),
            adaptive_patch_weights=query_weights.squeeze(2),
        )


class ITransformerMTLBaseline(EndpointAdapter):
    """Faithful inverted-variate Transformer with task-adapted MTL heads."""

    upstream_key = "itransformer"

    def __init__(self, config: LACConfig):
        super().__init__(config)
        h, p = config.hidden_dim, config.num_patches
        self.inverted_embedding = nn.Linear(3 * p, h)
        layer = nn.TransformerEncoderLayer(
            h, config.num_heads, 2 * h, config.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(h)
        context_dim = config.static_dim + 2 + config.treatment_dim + p
        self.context = nn.Sequential(nn.Linear(context_dim, h), nn.GELU(), nn.Linear(h, h))
        self.tbr_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 1))
        self.cac_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 1))

    def forward(self, static, baseline, values, mask, delta, times, treatments, **_):
        # Official inversion: each clinical variable is one token whose features span time.
        token_input = torch.cat(
            [values.transpose(1, 2), mask.transpose(1, 2), delta.transpose(1, 2)], dim=-1
        )
        encoded = self.norm(self.encoder(self.inverted_embedding(token_input))).mean(dim=1)
        base = torch.stack([baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))], -1)
        context = self.context(torch.cat([static, base, treatments.mean(1), times], -1))
        shared = torch.cat([encoded, context], -1)
        residuals = torch.cat([self.tbr_head(shared), self.cac_head(shared)], -1)
        return self.endpoint_output(baseline, residuals)


class _GraphMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim), nn.ELU(),
        )

    def forward(self, value):
        return self.net(value)


class _FIRSTGraphEncoder(nn.Module):
    """Node-edge-node message passing retained from the FIRST-ICU encoder."""

    def __init__(self, nodes: int, time_steps: int, feature_dim: int, hidden: int, dropout: float):
        super().__init__()
        edge = torch.ones(nodes, nodes)
        senders, receivers = torch.where(edge > 0)
        self.register_buffer("rel_send", F.one_hot(senders, nodes).float())
        self.register_buffer("rel_rec", F.one_hot(receivers, nodes).float())
        self.node_mlp = _GraphMLP(time_steps * feature_dim, hidden, dropout)
        self.edge_mlp = _GraphMLP(2 * hidden, hidden, dropout)
        self.node_update = _GraphMLP(hidden, hidden, dropout)
        self.edge_update = _GraphMLP(3 * hidden, hidden, 0.0)

    def node_to_edge(self, node):
        return torch.cat([torch.matmul(self.rel_rec, node), torch.matmul(self.rel_send, node)], -1)

    def edge_to_node(self, edge):
        incoming = torch.matmul(self.rel_rec.transpose(0, 1), edge)
        return incoming / max(1, incoming.shape[1])

    def forward(self, data):
        batch, time_steps, nodes, features = data.shape
        node = self.node_mlp(data.permute(0, 2, 1, 3).reshape(batch, nodes, time_steps * features))
        edge = self.edge_mlp(self.node_to_edge(node))
        skip = edge
        node = self.node_update(self.edge_to_node(edge))
        edge = self.edge_update(torch.cat([self.node_to_edge(node), skip], -1))
        return self.edge_to_node(edge)


class FIRSTICUMTLBaseline(EndpointAdapter):
    """FIRST-ICU graph encoder, temporal attention and endpoint interaction."""

    upstream_key = "first_icu"

    def __init__(self, config: LACConfig):
        super().__init__(config)
        h, d, a, p = config.hidden_dim, config.num_variables, config.treatment_dim, config.num_patches
        self.long_projection = nn.Linear(3, h)
        self.treatment_projection = nn.Linear(3, h)
        self.graph = _FIRSTGraphEncoder(d + a, p, 3, h, config.dropout)
        self.temporal = nn.LSTM(h, h, num_layers=2, batch_first=True, dropout=config.dropout)
        self.temporal_score = nn.Sequential(nn.Linear(h, max(4, h // 2)), nn.Tanh(), nn.Linear(max(4, h // 2), 1))
        self.endpoint_tokens = nn.Parameter(torch.randn(1, 2, h) * 0.02)
        self.interaction = nn.MultiheadAttention(h, config.num_heads, dropout=config.dropout, batch_first=True)
        self.context = nn.Sequential(nn.Linear(config.static_dim + 2, h), nn.GELU(), nn.Linear(h, h))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1)) for _ in range(2)])

    def forward(self, static, baseline, values, mask, delta, times, treatments, **_):
        batch, patches, _ = values.shape
        long_features = torch.stack([values, mask, delta], -1)
        gaps = torch.cat([times[:, :1], times[:, 1:] - times[:, :-1]], 1)
        treatment_features = torch.stack(
            [treatments, torch.ones_like(treatments), gaps[:, :, None].expand_as(treatments)], -1
        )
        graph_data = torch.cat([long_features, treatment_features], dim=2)
        graph_nodes = self.graph(graph_data)
        long_step = self.long_projection(long_features).mean(dim=2)
        treatment_step = self.treatment_projection(treatment_features).mean(dim=2)
        temporal, _ = self.temporal(long_step + treatment_step + graph_nodes.mean(1)[:, None, :])
        weights = torch.softmax(self.temporal_score(temporal).squeeze(-1), dim=1)
        sequence = (temporal * weights[:, :, None]).sum(1)
        endpoint_tokens = self.endpoint_tokens.expand(batch, -1, -1) + sequence[:, None, :]
        interacted, interaction_weights = self.interaction(endpoint_tokens, endpoint_tokens, endpoint_tokens)
        base = torch.stack([baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))], -1)
        context = self.context(torch.cat([static, base], -1))
        residuals = torch.cat(
            [head(torch.cat([interacted[:, index], context], -1)) for index, head in enumerate(self.heads)], -1
        )
        return self.endpoint_output(
            baseline, residuals, graph_node_embeddings=graph_nodes,
            endpoint_interaction=interaction_weights, temporal_attention=weights,
        )


class _RouteExpert(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.stl_tbr = nn.Linear(hidden, 1)
        self.stl_cac = nn.Linear(hidden, 1)
        self.mtl = nn.Linear(hidden, 2)

    def forward(self, path):
        encoded = self.encoder(path)
        return torch.cat([self.stl_tbr(encoded), self.stl_cac(encoded)], -1), self.mtl(encoded)


class LearningToRouteBaseline(EndpointAdapter):
    """Three-modality adaptation of the official per-sample two-stage router."""

    upstream_key = "learning_to_route"

    def __init__(self, config: LACConfig):
        super().__init__(config)
        h, p, d, a = config.hidden_dim, config.num_patches, config.num_variables, config.treatment_dim
        self.static_encoder = nn.Sequential(nn.Linear(config.static_dim + 2, h), nn.ReLU(), nn.Linear(h, h))
        self.long_encoder = nn.Sequential(nn.Linear(p * d * 3, h), nn.ReLU(), nn.Linear(h, h))
        self.treatment_encoder = nn.Sequential(nn.Linear(p * (a + 1), h), nn.ReLU(), nn.Linear(h, h))
        self.fusion = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.modality_router = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 4))
        self.task_routers = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2)) for _ in range(4)])
        self.experts = nn.ModuleList([_RouteExpert(h, config.dropout) for _ in range(4)])

    def forward(self, static, baseline, values, mask, delta, times, treatments, **_):
        base = torch.stack([baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))], -1)
        static_path = self.static_encoder(torch.cat([static, base], -1))
        long_path = self.long_encoder(torch.cat([values, mask, delta], -1).flatten(1))
        treatment_path = self.treatment_encoder(torch.cat([treatments, times[:, :, None]], -1).flatten(1))
        fusion_path = self.fusion(torch.cat([static_path, long_path, treatment_path], -1))
        paths = [static_path, long_path, treatment_path, fusion_path]
        modality_probs = torch.softmax(self.modality_router(torch.cat(paths[:3], -1)), -1)
        predictions, task_probs = [], []
        for path, router, expert in zip(paths, self.task_routers, self.experts):
            probability = torch.softmax(router(path), -1)
            stl, mtl = expert(path)
            predictions.append(probability[:, :1] * stl + probability[:, 1:] * mtl)
            task_probs.append(probability)
        residuals = sum(modality_probs[:, index:index + 1] * pred for index, pred in enumerate(predictions))
        return self.endpoint_output(
            baseline, residuals, modality_routes=modality_probs,
            task_routes=torch.stack(task_probs, dim=1),
        )
