from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def generate_soft_masks(
    region_labels: np.ndarray,
    region_ids: np.ndarray,
    feature_hw: Tuple[int, int],
    sigma: float = 2.0,
    eps: float = 1e-6,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build soft masks [N, Hf, Wf] from hard region labels.

    Pipeline (order is fixed and recorded):
      1. Per-region one-hot coverage on the original content grid.
      2. Area resize to feature resolution, retaining tiny-region coverage.
      3. Gaussian smoothing of each one-hot map (sigma in feature pixels)
         using a separable convolution -> smooth boundary overlap.
      4. Clamp to non-negative.
      5. Normalise across the region dimension so weights sum to 1.

    Args:
        region_labels: [H, W] int array, ids in [0, N-1].
        region_ids: 1D array [0, ..., N-1].
        feature_hw: (Hf, Wf) target feature spatial size.
        sigma: Gaussian sigma in feature pixels. If <= 0 the smoothing
            step is skipped and pure one-hot masks are returned.
        eps: floor for the normalisation denominator (zero-scale guard).
        device: target torch device.
        dtype: target torch dtype.

    Returns:
        soft_masks: [N, Hf, Wf] non-negative, sums to ~1 over dim 0.
    """
    Hf, Wf = feature_hw
    N = len(region_ids)

    if N < 1 or not np.array_equal(region_ids, np.arange(N)):
        raise ValueError("region_ids must be consecutive and nonempty")
    if set(np.unique(region_labels)) != set(range(N)):
        raise ValueError("Every region must have hard-label coverage")
    # Resize one-hot AREA, not label IDs. Tiny regions retain fractional coverage.
    one_hot = torch.from_numpy(np.stack([region_labels == i for i in range(N)]).astype(np.float32))
    masks = (
        torch.nn.functional.interpolate(one_hot.unsqueeze(0), size=(Hf, Wf), mode="area")
        .squeeze(0)
        .numpy()
    )

    # --- Step 3: Gaussian smoothing (separable) ---
    if sigma is not None and sigma > 0.0:
        masks = _gaussian_smooth(masks, sigma)

    # --- Step 4: non-negative clamp ---
    masks = np.maximum(masks, 0.0)

    # --- Step 5: normalise across region dimension ---
    mask_sum = masks.sum(axis=0, keepdims=True)  # [1, Hf, Wf]
    # zero-scale guard: locations with no coverage fall back to uniform
    # weights instead of producing NaN/inf
    if (mask_sum <= 0).any():
        raise ValueError("Soft masks have uncovered locations")
    masks = masks / mask_sum

    return torch.from_numpy(masks).to(device=device, dtype=dtype)


def generate_hard_masks(
    region_labels: np.ndarray,
    region_ids: np.ndarray,
    feature_hw: Tuple[int, int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One-hot hard masks [N, Hf, Wf] from region labels."""
    Hf, Wf = feature_hw
    N = len(region_ids)

    labels_t = torch.from_numpy(region_labels.astype(np.float32))
    labels_resized = (
        torch.nn.functional.interpolate(
            labels_t.unsqueeze(0).unsqueeze(0),
            size=(Hf, Wf),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )

    hard_masks = torch.zeros(N, Hf, Wf, device=device, dtype=dtype)
    for i in range(N):
        hard_masks[i] = (labels_resized == float(i)).to(device=device, dtype=dtype)

    return hard_masks


def compute_effective_areas(
    soft_masks: torch.Tensor,
    threshold: float = 0.01,
) -> Dict[int, float]:
    """
    Fraction of spatial locations where each region has weight > threshold.

    Returns:
        dict region_id -> effective area ratio in (0, 1].
    """
    m = soft_masks.detach().cpu()
    N, Hf, Wf = m.shape
    total = float(Hf * Wf)
    areas: Dict[int, float] = {}
    for i in range(N):
        ratio = float((m[i] > threshold).sum().item()) / total if total > 0 else 0.0
        areas[i] = ratio
    return areas


def validate_soft_masks(
    soft_masks: torch.Tensor,
    atol: float = 1e-3,
) -> bool:
    """
    Validate the soft-mask contract:
      - shape [N, Hf, Wf], N > 0
      - finite, non-negative
      - per-location sum over regions ~ 1
    """
    assert soft_masks.ndim == 3, f"Expected 3D [N,Hf,Wf], got {soft_masks.ndim}D"
    N, Hf, Wf = soft_masks.shape
    assert N > 0, "Number of regions must be positive"
    assert Hf > 0 and Wf > 0, "Feature spatial size must be positive"
    assert torch.isfinite(soft_masks).all(), "Soft masks contain non-finite values"
    assert (soft_masks >= 0).all(), "Soft masks contain negative values"
    assert (soft_masks.sum(dim=(1, 2)) > 0).all(), "A region vanished in feature space"

    mask_sum = soft_masks.sum(dim=0)
    ones = torch.ones_like(mask_sum)
    assert torch.allclose(mask_sum, ones, atol=atol), (
        f"Soft mask sum != 1: max diff {(mask_sum - ones).abs().max().item()}"
    )
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Truncated 1D Gaussian kernel, normalised to sum 1."""
    radius = int(np.ceil(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs**2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def _gaussian_smooth(masks: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separable Gaussian smoothing over the two spatial dims.
    Each spatial axis is filtered independently with 'same'-length convolution
    on edge-padded input, so the output keeps the input shape.
    """
    k = _gaussian_kernel_1d(sigma)
    r = k.size // 2
    out = masks.copy()
    # smooth along axis -2 (rows), vectorised over columns
    padded = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="edge")
    tmp = np.zeros_like(out)
    for idx in range(out.shape[0]):
        for c in range(out.shape[2]):
            tmp[idx, :, c] = np.convolve(padded[idx, :, c], k, mode="valid")
    # smooth along axis -1 (cols), vectorised over rows
    padded = np.pad(tmp, ((0, 0), (0, 0), (r, r)), mode="edge")
    for idx in range(out.shape[0]):
        for rr in range(out.shape[1]):
            out[idx, rr, :] = np.convolve(padded[idx, rr, :], k, mode="valid")
    return out
