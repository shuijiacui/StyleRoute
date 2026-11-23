"""Region adjacency graph construction and edge weight computation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.color import rgb2lab

from .region_types import RegionPack


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {rgb.shape}")
    rgb_uint8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    lab = rgb2lab(rgb_uint8)
    return lab


def _normalised_color_diff(
    lab: np.ndarray, labels: np.ndarray,
    i: int, j: int, max_color_diff: float,
) -> float:
    mask_i = labels == i
    mask_j = labels == j
    if not np.any(mask_i) or not np.any(mask_j):
        return 0.0
    mean_i = lab[mask_i].mean(axis=0)
    mean_j = lab[mask_j].mean(axis=0)
    diff = np.linalg.norm(mean_i - mean_j)
    if max_color_diff <= 0:
        return 0.0
    return float(np.clip(diff / max_color_diff, 0.0, 1.0))


def compute_boundary_gradient(
    img: np.ndarray, labels: np.ndarray,
    i: int, j: int, max_grad: float = 1.0,
    grad_mag: Optional[np.ndarray] = None,
) -> float:
    """Vectorized boundary gradient computation."""
    mask_i = labels == i
    mask_j = labels == j
    if not np.any(mask_i) or not np.any(mask_j):
        return 0.0

    if grad_mag is None:
        gray = img.astype(np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Vectorized boundary detection via dilation
    from scipy import ndimage
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    dilated_j = ndimage.binary_dilation(mask_j, structure=struct)
    boundary = mask_i & dilated_j

    if not np.any(boundary):
        return 0.0

    mean_grad = float(np.mean(grad_mag[boundary]))
    if max_grad <= 0:
        return 0.0
    return float(np.clip(mean_grad / max_grad, 0.0, 1.0))


def build_region_graph(
    labels: np.ndarray,
    img: np.ndarray,
    config: Optional[Dict] = None,
    mode: str = "coordination",
) -> List[Tuple[int, int, float]]:
    """Build undirected region adjacency graph (vectorized + precomputed)."""
    if mode not in ("merge_cost", "coordination"):
        raise ValueError(
            f"Unknown mode '{mode}'. Expected 'merge_cost' or 'coordination'."
        )

    if config is None:
        config = {}

    merge_cfg = config.get("merge", {})
    ew_cfg = config.get("edge_weight", {})
    norm_cfg = config.get("normalization", {})

    max_color_diff = float(norm_cfg.get("max_color_diff", 100.0))
    max_contact = float(norm_cfg.get("max_contact", 1.0))
    if max_contact <= 0:
        max_contact = 1.0

    lab = _rgb_to_lab(img)
    h, w = labels.shape
    n_regions = int(labels.max()) + 1

    # ---- Precompute region Lab means (ONCE) ----
    region_lab_means = {}
    for rid in range(n_regions):
        mask = labels == rid
        if np.any(mask):
            region_lab_means[rid] = lab[mask].mean(axis=0)
        else:
            region_lab_means[rid] = np.zeros(3)

    # ---- Vectorized contact computation ----
    right_diff = labels[:, 1:] != labels[:, :-1]
    bottom_diff = labels[1:, :] != labels[:-1, :]

    contact: Dict[Tuple[int, int], int] = {}

    if np.any(right_diff):
        y_idx, x_idx = np.where(right_diff)
        left_labels = labels[y_idx, x_idx]
        right_labels = labels[y_idx, x_idx + 1]
        for cur, nb in zip(left_labels, right_labels):
            cur_i, nb_i = int(cur), int(nb)
            if cur_i != nb_i:
                key = (min(cur_i, nb_i), max(cur_i, nb_i))
                contact[key] = contact.get(key, 0) + 1

    if np.any(bottom_diff):
        y_idx, x_idx = np.where(bottom_diff)
        top_labels = labels[y_idx, x_idx]
        bottom_labels = labels[y_idx + 1, x_idx]
        for cur, nb in zip(top_labels, bottom_labels):
            cur_i, nb_i = int(cur), int(nb)
            if cur_i != nb_i:
                key = (min(cur_i, nb_i), max(cur_i, nb_i))
                contact[key] = contact.get(key, 0) + 1

    # ---- Precompute gradient magnitude (once) ----
    gray = img.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    edges: List[Tuple[int, int, float]] = []
    skip_bg_in_merge = bool(merge_cfg.get("skip_boundary_gradient", True))

    for (i, j), shared_len in contact.items():
        if i < 0 or j >= n_regions:
            continue

        # ---- Fast color diff (lookup) ----
        mean_i = region_lab_means[i]
        mean_j = region_lab_means[j]
        diff = np.linalg.norm(mean_i - mean_j)
        if max_color_diff <= 0:
            cd = 0.0
        else:
            cd = float(np.clip(diff / max_color_diff, 0.0, 1.0))

        if mode == "merge_cost":
            u = float(merge_cfg.get("u", 1.0))
            v = float(merge_cfg.get("v", 0.0))
            if v > 0 and not skip_bg_in_merge:
                bg = compute_boundary_gradient(
                    img, labels, i, j, max_grad=1.0, grad_mag=grad_mag
                )
            else:
                bg = 0.0
            cost = u * cd + v * bg
            edges.append((i, j, float(np.clip(cost, 0.0, 1.0))))
        else:
            u_w = float(ew_cfg.get("u_w", 0.5))
            v_w = float(ew_cfg.get("v_w", 0.5))
            decay = float(ew_cfg.get("decay", 1.0))
            bg = compute_boundary_gradient(
                img, labels, i, j, max_grad=1.0, grad_mag=grad_mag
            )
            raw = u_w * cd + v_w * bg
            w_coord = math.exp(-decay * float(np.clip(raw, 0.0, 1.0)))
            contact_norm = shared_len / max_contact
            weight = w_coord * contact_norm
            edges.append((i, j, float(np.clip(weight, 0.0, 1.0))))

    edges = [(int(i), int(j), float(w)) for (i, j, w) in edges]
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges


def equal_weight_graph(labels: np.ndarray) -> List[Tuple[int, int, float]]:
    """Return graph with same topology but all edge weights = 1.0."""
    h, w = labels.shape

    right_diff = labels[:, 1:] != labels[:, :-1]
    bottom_diff = labels[1:, :] != labels[:-1, :]

    contact: Dict[Tuple[int, int], int] = {}

    if np.any(right_diff):
        y_idx, x_idx = np.where(right_diff)
        left_labels = labels[y_idx, x_idx]
        right_labels = labels[y_idx, x_idx + 1]
        for cur, nb in zip(left_labels, right_labels):
            cur_i, nb_i = int(cur), int(nb)
            if cur_i != nb_i:
                key = (min(cur_i, nb_i), max(cur_i, nb_i))
                contact[key] = contact.get(key, 0) + 1

    if np.any(bottom_diff):
        y_idx, x_idx = np.where(bottom_diff)
        top_labels = labels[y_idx, x_idx]
        bottom_labels = labels[y_idx + 1, x_idx]
        for cur, nb in zip(top_labels, bottom_labels):
            cur_i, nb_i = int(cur), int(nb)
            if cur_i != nb_i:
                key = (min(cur_i, nb_i), max(cur_i, nb_i))
                contact[key] = contact.get(key, 0) + 1

    n_regions = int(labels.max()) + 1
    edges: List[Tuple[int, int, float]] = []
    for (i, j) in contact.keys():
        if i < 0 or j >= n_regions:
            continue
        edges.append((i, j, 1.0))

    edges = [(int(i), int(j), float(w)) for (i, j, w) in edges]
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges


def coordination_stats(edges: List[Tuple[int, int, float]]) -> Dict:
    """Compute statistics of coordination edge weights."""
    if not edges:
        return {
            "n_edges": 0,
            "weight_sum": 0.0,
            "w_min": 0.0,
            "w_max": 0.0,
            "w_mean": 0.0,
        }
    ws = [e[2] for e in edges]
    return {
        "n_edges": len(ws),
        "weight_sum": float(sum(ws)),
        "w_min": float(min(ws)),
        "w_max": float(max(ws)),
        "w_mean": float(np.mean(ws)),
    }