"""Shared M4/M5/M6 data contracts. Frozen fields do not make tensors immutable.

Callers must treat every prepared tensor, PIL image and metadata mapping as read-only.
M4 never mutates cached features; M5/M6 should copy when they need mutable data.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreparedPair:
    pair_id: str
    content_rgb: Image.Image
    style_rgb: Image.Image  # preprocessed reference for M6's independent evaluator
    content_tensor: torch.Tensor  # [1,3,H,W], FP32 [0,1]
    content_features: torch.Tensor  # [1,512,Hf,Wf]
    adain_features: torch.Tensor  # same shape, device and dtype
    feature_hw: tuple[int, int]
    input_hw: tuple[int, int]
    cache_key: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegionPack:
    region_ids: tuple[int, ...]  # exactly (0, ..., N-1)
    labels: np.ndarray  # [H,W], integer labels aligned with content_rgb
    soft_masks: torch.Tensor  # [N,Hf,Wf], FP32, same device as pair
    edges: tuple[tuple[int, int, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRegions:
    """Validated masks copied once; reuse this object in the candidate loop."""

    pair_cache_key: str
    region_ids: tuple[int, ...]
    soft_masks: torch.Tensor
    cache_key: str


@dataclass(frozen=True)
class GenerationResult:
    image_rgb: torch.Tensor  # [1,3,H,W], finite FP32, clamped [0,1]
    alpha_map: torch.Tensor  # [1,1,Hf,Wf]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """M6 returns real measurements; use None for metrics that were not measured.

    status: improved | no_improvement | no_feasible | budget_exhausted | cancelled
    | failed. best_image_rgb and region_alphas must describe the same candidate.
    Default fallbacks must be identified in metadata['used_default'].
    """

    best_image_rgb: torch.Tensor
    region_alphas: tuple[float, ...]
    status: str
    metrics: dict[str, float | None]
    trajectory: tuple[dict[str, Any], ...]
    candidate_count: int
    decoder_calls: int
    state_cache_hits: int
    metadata: dict[str, Any] = field(default_factory=dict)
