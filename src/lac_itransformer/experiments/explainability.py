from __future__ import annotations

import numpy as np
import torch

from ..training.dataset import model_inputs


def route_heatmaps(output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "inflammation": output["route_inflammation"].detach().cpu().numpy(),
        "calcification": output["route_calcification"].detach().cpu().numpy(),
        "background": output["route_background"].detach().cpu().numpy(),
    }


@torch.no_grad()
def deletion_faithfulness(model, batch, task: str = "tbr", fraction: float = 0.1, seed: int = 2026) -> dict[str, float]:
    output = model(**model_inputs(batch))
    route_key = "route_inflammation" if task == "tbr" else "route_calcification"
    prediction_key = "endpoint_tbr" if task == "tbr" else "endpoint_cac"
    scores = output[route_key].mean(dim=1)
    count = max(1, int(scores.shape[-1] * fraction))
    top = scores.topk(count, dim=-1).indices
    bottom = scores.topk(count, dim=-1, largest=False).indices
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    random_rank = torch.rand(scores.shape, generator=generator, device=scores.device).topk(count, dim=-1).indices

    def perturb(index: torch.Tensor) -> float:
        mask = batch.mask.clone()
        values = batch.values.clone()
        expanded = index[:, None, :].expand(-1, values.shape[1], -1)
        values.scatter_(2, expanded, 0.0)
        mask.scatter_(2, expanded, 0.0)
        changed = model(**(model_inputs(batch) | {"values": values, "mask": mask}))[prediction_key]
        return float((changed - output[prediction_key]).abs().mean())

    return {"top": perturb(top), "random": perturb(random_rank), "bottom": perturb(bottom)}
