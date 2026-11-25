"""Four-connected adjacency and appearance weights, with explicit RGB ranges."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab


def normalized_rgb(rgb):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    else:
        raise ValueError("RGB must be uint8 [0,255] or floating point [0,1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Floating RGB must be finite and in [0,1]")
    return image


def _rgb_to_lab(rgb):
    return rgb2lab(normalized_rgb(rgb))


def _contacts(labels):
    pairs = np.concatenate(
        [
            np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], axis=1),
            np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()], axis=1),
        ]
    )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(np.sort(pairs, axis=1), axis=0, return_counts=True)


def compute_boundary_gradient(img, labels, i, j, max_grad=1.0, grad_mag=None):
    from scipy.ndimage import binary_dilation

    if max_grad <= 0:
        return 0.0
    if grad_mag is None:
        gy, gx = np.gradient(normalized_rgb(img).mean(axis=2))
        grad_mag = np.hypot(gx, gy)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    first, second = labels == i, labels == j
    boundary = (first & binary_dilation(second, structure=structure)) | (
        second & binary_dilation(first, structure=structure)
    )
    return float(np.clip(grad_mag[boundary].mean() / max_grad, 0, 1)) if boundary.any() else 0.0


def _normalised_color_diff(lab, labels, i, j, max_color_diff):
    if max_color_diff <= 0 or not (labels == i).any() or not (labels == j).any():
        return 0.0
    return float(
        np.clip(
            np.linalg.norm(lab[labels == i].mean(0) - lab[labels == j].mean(0)) / max_color_diff,
            0,
            1,
        )
    )


def build_region_graph(labels, img, config=None, mode="coordination"):
    if mode not in {"coordination", "merge_cost"}:
        raise ValueError(f"Unknown graph mode: {mode}")
    config = config or {}
    merge, weight, norm = (config.get(k, {}) for k in ("merge", "edge_weight", "normalization"))
    rgb = normalized_rgb(img)
    lab = rgb2lab(rgb)
    flat = labels.ravel()
    count = np.bincount(flat)
    means = np.stack(
        [np.bincount(flat, weights=lab[:, :, c].ravel(), minlength=len(count)) for c in range(3)],
        axis=1,
    ) / np.maximum(count[:, None], 1)
    gy, gx = np.gradient(rgb.mean(2))
    magnitude = np.hypot(gx, gy)
    raw_pairs = np.concatenate(
        [
            np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], axis=1),
            np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()], axis=1),
        ]
    )
    gradients = np.concatenate(
        [
            ((magnitude[:, :-1] + magnitude[:, 1:]) * 0.5).ravel(),
            ((magnitude[:-1, :] + magnitude[1:, :]) * 0.5).ravel(),
        ]
    )
    crossing = raw_pairs[:, 0] != raw_pairs[:, 1]
    pairs, inverse, lengths = np.unique(
        np.sort(raw_pairs[crossing], axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if not len(pairs):
        return []
    boundary_means = np.bincount(inverse, weights=gradients[crossing]) / lengths
    color_scale = float(norm.get("max_color_diff", 100.0))
    gradient_scale = float(norm.get("max_gradient", 1.0))
    edges = []
    for (i, j), length, mean_gradient in zip(pairs, lengths, boundary_means):
        i, j = int(i), int(j)
        cd = (
            float(np.clip(np.linalg.norm(means[i] - means[j]) / color_scale, 0, 1))
            if color_scale > 0
            else 0.0
        )
        bg = float(np.clip(mean_gradient / gradient_scale, 0, 1)) if gradient_scale > 0 else 0.0
        if mode == "merge_cost":
            if merge.get("skip_boundary_gradient", False):
                bg = 0.0
            value = float(merge.get("u", 0.6)) * cd + float(merge.get("v", 0.4)) * bg
        else:
            raw = float(weight.get("u_w", 0.5)) * cd + float(weight.get("v_w", 0.5)) * bg
            value = np.exp(-float(weight.get("decay", 1.0)) * raw)
            # Relative contact factor is bounded; it cannot saturate appearance weights.
            power = float(weight.get("contact_power", 0.0))
            if power < 0:
                raise ValueError("contact_power must be nonnegative")
            value *= (float(length) / float(lengths.max())) ** power
        edges.append((i, j, float(np.clip(value, 0, 1))))
    return edges


def equal_weight_graph(labels):
    pairs, _ = _contacts(labels)
    return [(int(i), int(j), 1.0) for i, j in pairs]


def coordination_stats(edges):
    weights = [e[2] for e in edges]
    return {
        "n_edges": len(edges),
        "weight_sum": float(sum(weights)),
        "w_min": float(min(weights, default=0)),
        "w_max": float(max(weights, default=0)),
        "w_mean": float(np.mean(weights)) if weights else 0.0,
    }
