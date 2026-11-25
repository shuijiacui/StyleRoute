from __future__ import annotations

import os
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .region_types import RegionPack


def _to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x


def _find_boundaries(labels: np.ndarray) -> np.ndarray:
    """Binary boundary map from a label array (4-connected)."""
    H, W = labels.shape
    b = np.zeros((H, W), dtype=np.float32)
    b[:, : W - 1] = (labels[:, : W - 1] != labels[:, 1:]).astype(np.float32)
    b[: H - 1, :] = (labels[: H - 1, :] != labels[1:, :]).astype(np.float32)
    return b


def save_partition_preview(
    regions: RegionPack,
    content_rgb: Optional[np.ndarray] = None,
    out_path: str = "outputs/m5_partition.png",
    max_regions: int = 64,
) -> str:
    """Hard-boundary overlay with region IDs drawn at each region centroid."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    labels = _to_numpy(regions.labels)
    H, W = labels.shape
    N = min(regions.n_regions, max_regions)

    base = (
        plt.colormaps["tab20"].resampled(max(20, N))
        if hasattr(plt, "colormaps")
        else plt.cm.get_cmap("tab20").resampled(max(20, N))
    )  # type: ignore[attr-defined]
    colors = base(np.arange(N))
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=(W / 50.0, H / 50.0), dpi=150)
    if content_rgb is not None:
        ax.imshow(content_rgb, alpha=0.4)
    ax.imshow(labels, cmap=cmap, alpha=0.7, vmin=0, vmax=max(1, N - 1))
    ax.imshow(_find_boundaries(labels), cmap="gray", alpha=0.9)

    for rid in range(N):
        ys, xs = np.where(labels == rid)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        ax.text(
            cx,
            cy,
            str(rid),
            color="white",
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"),
        )

    ax.axis("off")
    ax.set_title(f"Partition: {regions.n_regions} regions", fontsize=8)
    patches = [mpatches.Patch(color=colors[i], label=f"R{i}") for i in range(min(N, 12))]
    if N > 12:
        patches.append(mpatches.Patch(color="gray", label=f"... +{N - 12}"))
    ax.legend(handles=patches, loc="lower right", fontsize=5, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_graph_preview(
    regions: RegionPack,
    out_path: str = "outputs/m5_graph.png",
    max_nodes: int = 48,
) -> str:
    """
    Region adjacency graph. Line thickness encodes edge weight; the caption
    states the weight semantics (coordination weight: high = keep consistent).
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    labels = _to_numpy(regions.labels)
    Hf, Wf = labels.shape
    N = min(regions.n_regions, max_nodes)

    import networkx as nx

    G = nx.Graph()
    for i in range(N):
        ys, xs = np.where(labels == i)
        if len(ys) > 0:
            G.add_node(i, pos=(float(xs.mean()) / Wf, 1.0 - float(ys.mean()) / Hf))
        else:
            G.add_node(i, pos=(0.5, 0.5))

    for edge in regions.edges:
        if len(edge) == 3:
            i, j, w = edge
        elif len(edge) == 2:
            i, j = edge
            w = 1.0
        else:
            continue
        if i < N and j < N:
            G.add_edge(i, j, weight=float(w))

    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

    ws = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    max_w = max(ws) if ws else 1.0
    if max_w <= 0:
        max_w = 1.0
    widths = [1.0 + 3.0 * (w / max_w) for w in ws]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150, node_color="steelblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="orange", alpha=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white")
    ax.set_title(
        f"Region graph: {N} nodes, {len(G.edges())} edges (thickness = coordination weight)",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_mask_overlay(
    regions: RegionPack,
    region_id: int,
    content_rgb: Optional[np.ndarray] = None,
    out_path: str = "outputs/m5_mask_overlay.png",
    alpha: float = 0.55,
) -> str:
    """Soft-mask coverage overlay for a single region (feature resolution)."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if region_id >= regions.n_regions:
        raise ValueError(f"region_id {region_id} >= n_regions {regions.n_regions}")

    mask = _to_numpy(regions.soft_masks[region_id])
    Hf, Wf = mask.shape

    ncols = 2 if content_rgb is not None else 1
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6 * ncols, 5) if content_rgb is not None else (6, 5),
        dpi=150,
    )
    if ncols == 1:
        axes = [axes]

    if content_rgb is not None:
        from PIL import Image as PILImage

        resized = np.asarray(
            PILImage.fromarray(np.asarray(content_rgb).astype(np.uint8)).resize(
                (Wf, Hf), PILImage.Resampling.LANCZOS
            )
        )
        axes[0].imshow(resized)
        axes[0].set_title("Content (feature size)", fontsize=8)
        axes[0].axis("off")
        axes[1].imshow(resized)
        axes[1].imshow(mask, cmap="hot", alpha=alpha, vmin=0, vmax=1)
        axes[1].set_title(f"Region {region_id} soft-mask overlay", fontsize=8)
        axes[1].axis("off")
    else:
        axes[0].imshow(mask, cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"Region {region_id} soft mask", fontsize=8)
        axes[0].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_mask_sum_check(
    regions: RegionPack,
    out_path: str = "outputs/m5_mask_sum.png",
) -> str:
    """Visualise the per-location mask sum; should be flat 1.0 everywhere."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    s = _to_numpy(regions.soft_masks).sum(axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    im = ax.imshow(s, cmap="viridis", vmin=1.0 - 1e-3, vmax=1.0 + 1e-3)
    ax.set_title("Soft-mask sum over regions (target 1.0)", fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_edge_weight_report(
    regions: RegionPack,
    out_path: str = "outputs/m5_edge_weights.png",
) -> str:
    """Histogram of coordination edge weights (direction sanity check)."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ws = np.array([e[2] for e in regions.edges], dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    if ws.size:
        ax.hist(ws, bins=20, color="steelblue", edgecolor="black")
        ax.axvline(float(ws.mean()), color="red", linestyle="--", label="mean")
        ax.legend(fontsize=7)
    ax.set_title("Coordination edge weights (high = keep consistent)", fontsize=8)
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
