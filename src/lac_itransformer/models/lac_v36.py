from __future__ import annotations

from dataclasses import dataclass

import torch

from .lac_v2 import LowRankAdapter
from .lac_v35 import LACV35Config, LACiTransformerV35


@dataclass
class LACV36Config(LACV35Config):
    """Protected shared dual-task core with task-isolated training forward."""

    use_private_cac_encoder: bool = False
    strict_historical_transfer: bool = True
    historical_decay_per_year: float = 1.0


class StrictHistoricalLowRankAdapter(LowRankAdapter):
    """Low-rank transfer whose target patch can read strictly earlier patches."""

    def __init__(self, hidden: int, rank: int, dropout: float, decay: float):
        super().__init__(hidden, rank, dropout)
        self.decay = float(decay)
        self._times: torch.Tensor | None = None

    def set_times(self, times: torch.Tensor | None) -> None:
        self._times = times

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self._times is None:
            return super().forward(trajectory)
        times = self._times
        elapsed = times[:, :, None] - times[:, None, :]
        target_index = torch.arange(times.shape[1], device=times.device)[:, None]
        source_index = torch.arange(times.shape[1], device=times.device)[None, :]
        strict = (
            (source_index < target_index)[None, :, :]
            & torch.isfinite(elapsed)
            & (elapsed > 1e-8)
        )
        weight = torch.exp(-self.decay * elapsed.clamp_min(0)) * strict.to(
            trajectory.dtype
        )
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        history = torch.einsum("bts,bsh->bth", weight, trajectory)
        return super().forward(history)


class LACiTransformerV36(LACiTransformerV35):
    """Exploratory shared-phenotype model with exact TBR path isolation.

    During the staged TBR fit the CAC graph is not executed at all.  This
    prevents CAC architecture ablations from changing the random-number stream
    consumed by the TBR head.  Final inference still emits both tasks from one
    model and CAC reads a protected shared representation plus optional I->C
    transfer and CAC-only residual adapter.
    """

    architecture_version = "V3.6"

    def __init__(self, config: LACV36Config):
        super().__init__(config)
        self.config = config
        if config.strict_historical_transfer:
            self.shared_transfer_adapter = StrictHistoricalLowRankAdapter(
                int(config.hidden_dim),
                max(1, int(config.shared_transfer_rank)),
                float(config.dropout),
                float(config.historical_decay_per_year),
            )
            torch.nn.init.zeros_(self.shared_transfer_adapter.up.weight)
            if not config.shared_transfer_enabled:
                self._freeze_module(self.shared_transfer_adapter)

    def _cac_hidden(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        # V3.5's all-variable CAC adapter could bypass the declared I->C path.
        # In V3.6 the residual adapter remains CAC-specific and inflammation can
        # reach CAC only through the audited strict-history transfer.
        if self.config.strict_historical_transfer and bool(
            torch.all(feature_mask == 1).item()
        ):
            feature_mask = self.calcification_feature_mask
        return super()._cac_hidden(values, mask, delta, feature_mask)

    def _tbr_stage_active(self) -> bool:
        tbr_trainable = any(parameter.requires_grad for parameter in self.head_i.parameters())
        cac_trainable = any(parameter.requires_grad for parameter in self.head_c.parameters())
        return bool(tbr_trainable and not cac_trainable)

    def _forward_tbr_only(
        self,
        static: torch.Tensor,
        baseline: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_i = self._masked_encode(
            values,
            mask,
            delta,
            self.inflammation_feature_mask,
            cac_encoder=False,
        )
        baseline_i = (
            baseline[:, 0:1]
            if self.config.baseline_anchoring
            else torch.zeros_like(baseline[:, 0:1])
        )
        context_i = self.static_context(static) + self.baseline_context_i(baseline_i)
        trajectory_i = self._trajectory(hidden_i.mean(dim=2), context_i, "i")
        pool_i = torch.softmax(self.pool_i(trajectory_i).squeeze(-1), dim=1)
        summary_i = (trajectory_i * pool_i.unsqueeze(-1)).sum(dim=1)
        delta_tbr = self.head_i(torch.cat([summary_i, context_i], dim=-1)).squeeze(-1)
        endpoint_tbr = baseline[:, 0] + delta_tbr if self.config.baseline_anchoring else delta_tbr
        return {
            "delta_tbr": delta_tbr,
            "endpoint_tbr": endpoint_tbr,
        }

    def forward(
        self,
        static: torch.Tensor,
        baseline: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        times: torch.Tensor,
        treatments: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self._tbr_stage_active():
            return self._forward_tbr_only(static, baseline, values, mask, delta)
        if isinstance(self.shared_transfer_adapter, StrictHistoricalLowRankAdapter):
            self.shared_transfer_adapter.set_times(times)
        try:
            return super().forward(
                static=static,
                baseline=baseline,
                values=values,
                mask=mask,
                delta=delta,
                times=times,
                treatments=treatments,
                **kwargs,
            )
        finally:
            if isinstance(
                self.shared_transfer_adapter, StrictHistoricalLowRankAdapter
            ):
                self.shared_transfer_adapter.set_times(None)
