from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class PartitionResult:
    region_ids: np.ndarray
    labels: np.ndarray
    soft_masks: torch.Tensor
    edges: List[Tuple[int, int, float]]
    metadata: Dict
    n_regions: int = field(init=False)

    def __post_init__(self):
        self.n_regions = len(self.region_ids)
        self._validate()

    def _validate(self):
        # --- region_ids: must be [0, 1, ..., N-1] ---
        assert isinstance(self.region_ids, np.ndarray), (
            f"region_ids must be np.ndarray, got {type(self.region_ids)}"
        )
        assert self.region_ids.dtype == np.int32, (
            f"region_ids must be int32, got {self.region_ids.dtype}"
        )
        assert np.array_equal(self.region_ids, np.arange(len(self.region_ids), dtype=np.int32)), (
            "region_ids must be a contiguous range [0, 1, ..., N-1]"
        )

        # --- labels: [H, W] int32, every pixel a valid id ---
        assert isinstance(self.labels, np.ndarray), (
            f"labels must be np.ndarray, got {type(self.labels)}"
        )
        assert self.labels.ndim == 2, f"labels must be [H, W], got ndim={self.labels.ndim}"
        assert self.labels.dtype == np.int32, f"labels must be int32, got {self.labels.dtype}"
        assert self.labels.min() >= 0, "labels contain negative ids"
        assert self.labels.max() < self.n_regions, "labels contain id >= n_regions"
        assert np.all(np.isin(np.unique(self.labels), self.region_ids)), (
            "labels contain ids outside region_ids"
        )

        # --- soft_masks: [N, Hf, Wf] finite, non-negative ---
        assert isinstance(self.soft_masks, torch.Tensor), (
            f"soft_masks must be torch.Tensor, got {type(self.soft_masks)}"
        )
        assert self.soft_masks.ndim == 3, (
            f"soft_masks must be [N, Hf, Wf], got ndim={self.soft_masks.ndim}"
        )
        assert self.soft_masks.shape[0] == self.n_regions, (
            "soft_masks first dim must equal n_regions"
        )
        assert torch.isfinite(self.soft_masks).all(), "soft_masks contain non-finite values"
        assert (self.soft_masks >= 0).all(), "soft_masks contain negative values"

        # --- edges: undirected, i < j, no duplicates, no self loops ---
        seen = set()
        for e in self.edges:
            assert len(e) == 3, f"Each edge must be (i, j, w), got {e}"
            i, j, w = e
            assert isinstance(i, int) and isinstance(j, int), (
                f"Edge endpoints must be int, got ({type(i)}, {type(j)})"
            )
            assert isinstance(w, (int, float, np.floating, np.integer)), (
                f"Edge weight must be numeric, got {type(w)}"
            )
            assert i < j, f"Edge must have i < j, got ({i}, {j})"
            assert 0 <= i < self.n_regions and 0 <= j < self.n_regions, (
                f"Edge endpoint out of range: ({i}, {j})"
            )
            assert not (np.isnan(float(w)) or np.isinf(float(w))), (
                f"Edge weight must be finite, got {w}"
            )
            assert (i, j) not in seen, f"Duplicate edge: ({i}, {j})"
            seen.add((i, j))


# Internal NumPy partition result, distinct from the public src.contracts.RegionPack.
RegionPack = PartitionResult


def compute_config_fingerprint(config: Dict) -> str:
    """Stable sha1 fingerprint of a config dict (sorted keys)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def compute_array_fingerprint(arr: np.ndarray) -> str:
    """Stable fingerprint of an input array (shape + bytes hash)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha1(a.tobytes()).hexdigest()[:12]
    return f"{a.shape}:{a.dtype}:{h}"
