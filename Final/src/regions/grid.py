"""Explicit fixed-grid baseline with equal-weight four-neighbour adjacency."""

from __future__ import annotations

import numpy as np
import torch

from src.contracts import RegionPack  # noqa: E402


def build_regions(content_rgb, *, feature_hw, device, grid=(4, 4)):
    """Tile the (already M4-preprocessed) content into a ``gh x gw`` grid.

    ``content_rgb`` is the prepared PIL RGB image. Labels live
    on the content grid; soft masks are one-hot labels resampled to
    ``feature_hw`` and renormalised so they are finite, nonnegative and sum
    to 1 per position (tolerance 1e-5). Edges are uniform 4-neighbourhood
    links (weight 1.0), which is the content-agnostic relationship baseline.
    """
    gh, gw = int(grid[0]), int(grid[1])
    width, height = content_rgb.size
    if not 1 <= gh <= height or not 1 <= gw <= width:
        raise ValueError("Grid dimensions must fit within the content image")
    labels = np.zeros((height, width), dtype=np.int64)
    cell_h = height // gh
    cell_w = width // gw
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * cell_h, height if gy == gh - 1 else (gy + 1) * cell_h
            x0, x1 = gx * cell_w, width if gx == gw - 1 else (gx + 1) * cell_w
            labels[y0:y1, x0:x1] = gy * gw + gx
    n = gh * gw
    region_ids = tuple(range(n))

    device = torch.device(device)
    labels_t = torch.from_numpy(labels).to(device).unsqueeze(0).unsqueeze(0).float()
    masks = torch.zeros((n, 1, height, width), device=device)
    for r in range(n):
        masks[r, 0] = labels_t[0, 0] == float(r)
    hf, wf = int(feature_hw[0]), int(feature_hw[1])
    masks = torch.nn.functional.interpolate(masks, size=(hf, wf), mode="area").clamp_min(0)
    total = masks.sum(dim=0, keepdim=True).clamp_min(1e-12)
    masks = (masks / total).squeeze(1).to(torch.float32)  # [N,Hf,Wf]

    edges = []
    for gy in range(gh):
        for gx in range(gw):
            r = gy * gw + gx
            if gx + 1 < gw:
                edges.append((r, r + 1, 1.0))
            if gy + 1 < gh:
                edges.append((r, r + gw, 1.0))

    return RegionPack(
        region_ids=region_ids,
        labels=labels,
        soft_masks=masks,
        edges=tuple(edges),
        metadata={
            "stub": False,
            "method": "fixed_grid",
            "owner": "M6",
            "kind": "fixed_grid",
            "grid": [gh, gw],
            "feature_hw": [hf, wf],
            "note": "Explicit content-agnostic grid baseline with equal-weight adjacency",
        },
    )
