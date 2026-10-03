from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.schema import LongitudinalBatch


TENSOR_KEYS = ("static", "baseline", "values", "mask", "delta", "times", "treatments", "targets")
IRREGULAR_KEYS = ("irregular_values", "irregular_mask", "irregular_times", "irregular_treatments")


class ArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.patient_ids = [str(x) for x in arrays["patient_ids"]]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        regular = {key: torch.as_tensor(self.arrays[key][index], dtype=torch.float32) for key in TENSOR_KEYS}
        irregular = {
            key: torch.as_tensor(self.arrays[key][index], dtype=torch.float32)
            for key in IRREGULAR_KEYS if key in self.arrays
        }
        return regular | irregular | {
            "patient_id": self.patient_ids[index]
        }


def collate_batch(items: list[dict]) -> LongitudinalBatch:
    return LongitudinalBatch(
        static=torch.stack([item["static"] for item in items]),
        baseline=torch.stack([item["baseline"] for item in items]),
        values=torch.stack([item["values"] for item in items]),
        mask=torch.stack([item["mask"] for item in items]),
        delta=torch.stack([item["delta"] for item in items]),
        times=torch.stack([item["times"] for item in items]),
        treatments=torch.stack([item["treatments"] for item in items]),
        targets=torch.stack([item["targets"] for item in items]),
        irregular_values=torch.stack([item.get("irregular_values", item["values"]) for item in items]),
        irregular_mask=torch.stack([item.get("irregular_mask", item["mask"]) for item in items]),
        irregular_times=torch.stack([item.get("irregular_times", item["times"]) for item in items]),
        irregular_treatments=torch.stack([item.get("irregular_treatments", item["treatments"]) for item in items]),
        patient_ids=[item["patient_id"] for item in items],
    )


def model_inputs(batch: LongitudinalBatch) -> dict[str, torch.Tensor]:
    return {
        "static": batch.static,
        "baseline": batch.baseline,
        "values": batch.values,
        "mask": batch.mask,
        "delta": batch.delta,
        "times": batch.times,
        "treatments": batch.treatments,
        "irregular_values": batch.irregular_values,
        "irregular_mask": batch.irregular_mask,
        "irregular_times": batch.irregular_times,
        "irregular_treatments": batch.irregular_treatments,
    }
