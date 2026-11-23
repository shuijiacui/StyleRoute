"""Supplementary metric: contour agreement between generated and content images
(README section 6, metrics axis).

Fixed pipeline, no learned components:

* grayscale conversion (mean over RGB);
* Sobel gradient magnitude (3x3, fixed kernels);
* threshold at a fixed upper quantile of the magnitude map -> boolean edge map;
* match edges within a fixed pixel tolerance (chamfer via pairwise distances),
  precision = |A matched| / |A|, recall = |B matched| / |B|, F1 harmonic mean.

Degenerate cases (frozen rules, documented): both edge maps empty -> P=R=F1=1.0
(agreement on "no edges"); exactly one empty -> P=R=F1=0.0. Images that fail to
yield any edges on real content are themselves a reported diagnostic, never
silently re-scored with a different extractor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:  # keep import light for pure-python environments
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


@dataclass(frozen=True)
class MetricsConfig:
    """Frozen contour-extraction settings (recorded in run metadata)."""

    tolerance_px: float = 2.0     # matching radius for chamfer agreement
    edge_quantile: float = 0.95   # fixed upper quantile for the gradient threshold
    min_edge_pixels: int = 8      # fewer than this counts as "no edges" (diagnostic)


_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


def _sobel(gray: "torch.Tensor") -> "torch.Tensor":
    """gray: [1,1,H,W] -> gradient magnitude [1,1,H,W] (fixed 3x3 Sobel)."""
    kx = torch.tensor(_SOBEL_X, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor(_SOBEL_Y, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def edge_map(image: "torch.Tensor", config: MetricsConfig | None = None) -> "torch.Tensor":
    """Fixed edge extraction. image: [1,3,H,W] FP32 in [0,1] -> bool [H,W]."""
    if torch is None:
        raise RuntimeError("torch is required for contour metrics")
    config = config or MetricsConfig()
    gray = image.to(torch.float32).mean(dim=1, keepdim=True)
    mag = _sobel(gray)[0, 0]
    threshold = torch.quantile(mag.flatten(), config.edge_quantile)
    return mag > threshold


def _coords(mask: "torch.Tensor") -> "torch.Tensor":
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return torch.stack([ys.to(torch.float32), xs.to(torch.float32)], dim=1)


def contour_prf(
    pred_edges: "torch.Tensor",
    target_edges: "torch.Tensor",
    config: MetricsConfig | None = None,
) -> dict[str, float | int | bool]:
    """Precision / recall / F1 with fixed pixel tolerance (chamfer matching).

    pred/target: bool [H,W]. Frozen degenerate rules: both empty -> all 1.0;
    one empty -> all 0.0.
    """
    if torch is None:
        raise RuntimeError("torch is required for contour metrics")
    config = config or MetricsConfig()
    pred_edges = pred_edges.bool()
    target_edges = target_edges.bool()
    n_pred = int(pred_edges.sum())
    n_target = int(target_edges.sum())
    pred_empty = n_pred < config.min_edge_pixels
    target_empty = n_target < config.min_edge_pixels

    if pred_empty and target_empty:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "pred_px": n_pred, "target_px": n_target, "both_no_edge": True}
    if pred_empty or target_empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "pred_px": n_pred, "target_px": n_target, "both_no_edge": False}

    pred_c = _coords(pred_edges)
    target_c = _coords(target_edges)
    dist = torch.cdist(pred_c, target_c)          # [P, T] pixel distances
    pred_match = (dist.min(dim=1).values <= config.tolerance_px).float().mean()
    target_match = (dist.min(dim=0).values <= config.tolerance_px).float().mean()
    precision = float(pred_match)
    recall = float(target_match)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "pred_px": n_pred, "target_px": n_target, "both_no_edge": False}


def contour_f1(
    generated: "torch.Tensor",
    content: "torch.Tensor",
    config: MetricsConfig | None = None,
) -> dict[str, float | int | bool]:
    """One-call metric: extract edges with the fixed pipeline, then PRF."""
    config = config or MetricsConfig()
    return contour_prf(edge_map(generated, config), edge_map(content, config), config)


def provenance(config: MetricsConfig | None = None) -> dict[str, Any]:
    config = config or MetricsConfig()
    return {
        "metric": "contour_prf",
        "extractor": "sobel3x3 + fixed upper quantile",
        "tolerance_px": config.tolerance_px,
        "edge_quantile": config.edge_quantile,
        "min_edge_pixels": config.min_edge_pixels,
        "degenerate_rule": "both-empty -> 1.0; one-empty -> 0.0",
    }
