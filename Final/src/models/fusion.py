"""Validated spatial alpha control; no segmentation or search logic."""

import math

import numpy as np
import torch

from src.contracts import PreparedPair, PreparedRegions, RegionPack
from src.project import digest


def validate_alphas(values, count: int, device: torch.device) -> torch.Tensor:
    alphas = torch.as_tensor(values, device=device, dtype=torch.float32)
    if alphas.shape != (count,):
        raise ValueError(f"Expected {count} region alphas, got shape {tuple(alphas.shape)}")
    if not torch.isfinite(alphas).all().item() or ((alphas < 0) | (alphas > 1)).any().item():
        raise ValueError("Region alphas must be finite values in [0,1]")
    return alphas


def prepare_regions(pair: PreparedPair, regions: RegionPack) -> PreparedRegions:
    """Validate once and copy masks to isolate them from M5's later mutations."""
    count = len(regions.region_ids)
    if count < 1 or regions.region_ids != tuple(range(count)):
        raise ValueError("region_ids must be contiguous integers starting at zero")
    labels = regions.labels
    if not isinstance(labels, np.ndarray) or labels.shape != pair.input_hw:
        raise ValueError(f"Labels must have input shape {pair.input_hw}")
    if not np.issubdtype(labels.dtype, np.integer) or set(np.unique(labels)) != set(range(count)):
        raise ValueError("Labels must contain exactly the declared region IDs")
    masks = regions.soft_masks
    if masks.shape != (count, *pair.feature_hw):
        raise ValueError(
            f"Mask shape must be {(count, *pair.feature_hw)}, got {tuple(masks.shape)}"
        )
    if masks.dtype != torch.float32 or masks.device != pair.content_features.device:
        raise ValueError("Masks must be FP32 and on the same device as the pair")
    if not torch.isfinite(masks).all().item() or (masks < 0).any().item():
        raise ValueError("Masks must be finite and nonnegative")
    coverage = masks.sum(0)
    if not torch.allclose(coverage, torch.ones_like(coverage), rtol=0, atol=1e-5):
        raise ValueError("Masks must sum to one at every feature position; no automatic repair")
    if (masks.flatten(1).sum(1) <= 0).any().item():
        raise ValueError("A region disappeared at feature resolution; return it to M5")
    for source, target, weight in regions.edges:
        if source not in regions.region_ids or target not in regions.region_ids or source == target:
            raise ValueError("Invalid graph edge region ID")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Edge weights must be finite and nonnegative")
    snapshot = masks.detach().clone()
    identity = digest(
        {
            "pair": pair.cache_key,
            "ids": regions.region_ids,
            "masks": snapshot.cpu().numpy().tobytes().hex(),
        }
    )
    return PreparedRegions(pair.cache_key, regions.region_ids, snapshot, identity)


def fuse_region_features(content_features, adain_features, soft_masks, region_alphas):
    """Pure fusion of validated tensors; creates new features and [1,1,Hf,Wf] alpha."""
    feature_alpha_map = torch.einsum("n,nhw->hw", region_alphas, soft_masks)[None, None]
    mixed = feature_alpha_map * adain_features + (1.0 - feature_alpha_map) * content_features
    return mixed, feature_alpha_map
