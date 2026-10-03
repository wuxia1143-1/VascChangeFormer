from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .lac import LACConfig


class PersistenceBaseline(nn.Module):
    def forward(self, *, baseline: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "endpoint_tbr": baseline[:, 0],
            "endpoint_cac": baseline[:, 1],
            "delta_tbr": torch.zeros_like(baseline[:, 0]),
            "delta_log_cac": torch.zeros_like(baseline[:, 0]),
        }


class SharedMTLiTransformer(nn.Module):
    """Shared encoder with independent residual heads; no phenotype coupling."""

    def __init__(self, config: LACConfig):
        super().__init__()
        h = config.hidden_dim
        self.config = config
        self.project = nn.Linear(3, h)
        self.variable = nn.Embedding(config.num_variables, h)
        layer = nn.TransformerEncoderLayer(h, config.num_heads, 2 * h, config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        self.context = nn.Sequential(nn.Linear(config.static_dim + 2, h), nn.GELU(), nn.Linear(h, h))
        self.tbr_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.cac_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.log_var_tbr = nn.Parameter(torch.zeros(()))
        self.log_var_cac = nn.Parameter(torch.zeros(()))

    def forward(self, static, baseline, values, mask, delta, **_):
        b, p, d = values.shape
        token = self.project(torch.stack([values, mask, delta], dim=-1)) + self.variable(torch.arange(d, device=values.device))[None, None]
        encoded = self.encoder(token.reshape(b * p, d, -1)).mean(dim=1).reshape(b, p, -1).mean(dim=1)
        base_context = torch.stack([baseline[:, 0], torch.log1p(baseline[:, 1].clamp_min(0))], dim=-1)
        context = self.context(torch.cat([static, base_context], dim=-1))
        summary = torch.cat([encoded, context], dim=-1)
        delta_tbr = self.tbr_head(summary).squeeze(-1)
        delta_cac = self.cac_head(summary).squeeze(-1)
        return {
            "endpoint_tbr": baseline[:, 0] + delta_tbr,
            "endpoint_cac": torch.expm1(torch.log1p(baseline[:, 1].clamp_min(0)) + delta_cac).clamp_min(0),
            "delta_tbr": delta_tbr,
            "delta_log_cac": delta_cac,
        }


class SingleTaskiTransformer(SharedMTLiTransformer):
    def __init__(self, config: LACConfig, task: str):
        super().__init__(config)
        if task not in {"tbr", "cac"}:
            raise ValueError("task must be 'tbr' or 'cac'")
        self.task = task
