"""Baseline region partitions: uniform grid, global single region, scan alphas."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.contracts import RegionPack

from .graph import build_region_graph
from .masks import generate_soft_masks


def _grid_rows_cols(n_target: int, h: int, w: int) -> Tuple[int, int]:
    """Find grid rows/cols that gives region count closest to n_target."""
    if not 1 <= n_target <= h * w:
        raise ValueError("Target region count must fit the image")
    candidates = [(r, min(w, max(1, round(n_target / r)))) for r in range(1, h + 1)]
    return min(
        candidates,
        key=lambda rc: (abs(rc[0] * rc[1] - n_target), abs(np.log((h / rc[0]) / (w / rc[1])))),
    )


def uniform_grid_partition(
    content_rgb: np.ndarray,
    target_regions: int = 12,
    feature_hw: Tuple[int, int] = (16, 16),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    config: Optional[Dict] = None,
) -> RegionPack:
    """Fixed grid partition, outputting same RegionPack contract."""
    if config is None:
        config = {}

    h, w = content_rgb.shape[:2]
    rows, cols = _grid_rows_cols(target_regions, h, w)

    labels = np.zeros((h, w), dtype=np.int32)
    rid = 0
    cell_h = h // rows
    cell_w = w // cols
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < rows - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < cols - 1 else w
            labels[y0:y1, x0:x1] = rid
            rid += 1

    n_regions = rid
    region_ids = np.arange(n_regions, dtype=np.int32)

    edges = build_region_graph(labels, content_rgb, config=config, mode="coordination")
    edges = [(int(i), int(j), float(w)) for (i, j, w) in edges]

    soft_masks = generate_soft_masks(
        labels,
        region_ids,
        feature_hw=feature_hw,
        device=device,
        dtype=dtype,
    )

    metadata = {
        "method": "uniform_grid",
        "grid_rows": rows,
        "grid_cols": cols,
        "n_final_regions": n_regions,
        "target_regions": target_regions,
        "region_count_diff": n_regions - target_regions,
    }

    return RegionPack(
        region_ids=tuple(int(i) for i in region_ids),
        labels=labels,
        soft_masks=soft_masks,
        edges=tuple(edges),
        metadata=metadata,
    )


def global_single_region(
    content_rgb: np.ndarray,
    feature_hw: Tuple[int, int] = (16, 16),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    config: Optional[Dict] = None,
) -> RegionPack:
    """Entire image as one region."""
    h, w = content_rgb.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    region_ids = np.array([0], dtype=np.int32)

    edges: List[Tuple[int, int, float]] = []

    soft_masks = generate_soft_masks(
        labels,
        region_ids,
        feature_hw=feature_hw,
        device=device,
        dtype=dtype,
    )

    metadata = {
        "method": "global_single_region",
        "n_final_regions": 1,
    }

    return RegionPack(
        region_ids=tuple(int(i) for i in region_ids),
        labels=labels,
        soft_masks=soft_masks,
        edges=tuple(edges),
        metadata=metadata,
    )


def scan_alphas_uniform(n_regions: int, alphas: List[float]) -> List[Dict[int, float]]:
    """Return list of {region_id: alpha} dicts for ablation scan."""
    if len(alphas) != n_regions:
        raise ValueError(f"Length of alphas ({len(alphas)}) must equal n_regions ({n_regions})")
    return [{rid: alphas[rid] for rid in range(n_regions)}]
