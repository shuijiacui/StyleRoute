"""Content-adaptive region partition via SLIC + adjacency-based merging."""

from __future__ import annotations

import json
import hashlib
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .region_types import RegionPack
from .graph import build_region_graph
from .masks import generate_soft_masks


def _run_slic(
    img: np.ndarray,
    n_segments: int = 64,
    compactness: float = 10.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    try:
        from skimage import segmentation
        labels = segmentation.slic(
            img,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
            channel_axis=2,
        )
        return labels
    except ImportError:
        h, w = img.shape[:2]
        grid_h = int(np.sqrt(n_segments * h / w))
        grid_w = int(n_segments / grid_h)
        grid_h = max(1, min(grid_h, h))
        grid_w = max(1, min(grid_w, w))
        labels = np.zeros((h, w), dtype=np.int32)
        region_id = 0
        cell_h = h // grid_h
        cell_w = w // grid_w
        for i in range(grid_h):
            for j in range(grid_w):
                y0, y1 = i * cell_h, (i + 1) * cell_h if i < grid_h - 1 else h
                x0, x1 = j * cell_w, (j + 1) * cell_w if j < grid_w - 1 else w
                labels[y0:y1, x0:x1] = region_id
                region_id += 1
        return labels


def _renormalise_ids(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.vectorize(mapping.get)(labels).astype(np.int32)


def _merge_regions(
    labels: np.ndarray,
    img: np.ndarray,
    merge_cfg: Dict,
    norm_cfg: Dict,
) -> Tuple[np.ndarray, List[Dict]]:
    cur = labels.copy()
    merge_record = []

    max_regions = int(merge_cfg.get("max_regions", 16))
    min_regions = int(merge_cfg.get("min_regions", 8))
    threshold = float(merge_cfg.get("threshold", 0.35))
    min_area = int(merge_cfg.get("min_area", 16))

    while True:
        n_regions = int(cur.max()) + 1
        if n_regions <= max_regions:
            break

        _merge_edges = build_region_graph(
            cur, img,
            config={"merge": merge_cfg, "normalization": norm_cfg},
            mode="merge_cost",
        )

        if not _merge_edges:
            break

        _merge_edges.sort(key=lambda x: x[2])

        for i, j, cost in _merge_edges:
            n_regions = int(cur.max()) + 1
            if n_regions <= max_regions:
                break
            if cost > threshold:
                break

            mask_j = cur == j
            cur[mask_j] = i
            merge_record.append({
                "keep": int(i),
                "drop": int(j),
                "cost": float(cost),
                "merged_area": int(np.sum(mask_j)),
            })

        cur = _renormalise_ids(cur)

    h, w = cur.shape
    while True:
        areas = np.bincount(cur.flatten())
        tiny = [rid for rid, area in enumerate(areas) if 0 < area < min_area]
        if not tiny:
            break
        changed = False
        for t in tiny:
            if t >= (int(cur.max()) + 1):
                continue
            y_idx, x_idx = np.where(cur == t)
            if len(y_idx) == 0:
                continue
            best_nb = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y_idx[0] + dy, x_idx[0] + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nb = cur[ny, nx]
                        if nb != t:
                            best_nb = int(nb)
                            break
                if best_nb is not None:
                    break
            if best_nb is not None:
                cur[cur == t] = best_nb
                changed = True
                merge_record.append({
                    "keep": best_nb,
                    "drop": int(t),
                    "cost": 0.0,
                    "merged_area": int(np.sum(cur == best_nb)),
                    "reason": "tiny_region",
                })
        if not changed:
            break
        cur = _renormalise_ids(cur)

    return cur, merge_record


def build_regions(
    content_rgb: np.ndarray,
    feature_hw: Tuple[int, int] = (16, 16),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    region_config: Optional[Dict] = None,
    return_diagnostics: bool = False,
) -> RegionPack:
    t_start = time.time()

    if region_config is None:
        region_config = {}

    seed = int(region_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    slic_cfg = region_config.get("slic", {})
    merge_cfg = region_config.get("merge", {})
    norm_cfg = region_config.get("normalization", {})

    n_segments = int(slic_cfg.get("n_segments", 64))
    compactness = float(slic_cfg.get("compactness", 10.0))
    sigma = float(slic_cfg.get("sigma", 1.0))

    # Step 1: SLIC
    labels = _run_slic(
        content_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        seed=seed,
    )
    initial_regions = int(labels.max()) + 1

    # Step 2: Merge
    labels, merge_record = _merge_regions(
        labels, content_rgb,
        merge_cfg=merge_cfg,
        norm_cfg=norm_cfg,
    )

    # ---- uniform image → force single region ----
    if np.allclose(content_rgb, content_rgb[0, 0, :], atol=1e-3):
        labels = np.zeros_like(labels, dtype=np.int32)

    # Step 3: Renormalise (returns int32)
    labels = _renormalise_ids(labels)

    # Step 4: Coordination graph
    edges = build_region_graph(
        labels, content_rgb,
        config=region_config,
        mode="coordination",
    )

    # Step 5: Soft masks
    n_regions = int(labels.max()) + 1
    region_ids = np.arange(n_regions, dtype=np.int32)
    soft_masks = generate_soft_masks(
        labels,
        region_ids,
        feature_hw=feature_hw,
        device=device,
        dtype=dtype,
    )

    # Step 6: Metadata
    areas = np.bincount(labels.flatten())
    coverage_ok = int(np.sum(areas)) == labels.size

    from scipy import ndimage
    _, n_conn = ndimage.label(
        labels, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    connectivity = {
        "n_connected_components": int(n_conn),
        "per_region_components": {},
        "all_connected": int(n_conn) == 1,
    }
    vanished_regions = []
    for rid in range(n_regions):
        _, n_comps = ndimage.label(
            labels == rid,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        )
        connectivity["per_region_components"][int(rid)] = int(n_comps)
        if int(n_comps) == 0:
            vanished_regions.append(int(rid))

    try:
        config_blob = json.dumps(region_config, sort_keys=True, default=str).encode("utf-8")
        config_fingerprint = hashlib.sha1(config_blob).hexdigest()[:12]
    except Exception:
        config_fingerprint = "unknown"

    metadata = {
        "method": "content_adaptive",
        "num_regions": n_regions,
        "input_shape": list(content_rgb.shape),
        "feature_hw": list(feature_hw),
        "n_initial_superpixels": int(n_segments),
        "initial_regions": initial_regions,
        "n_final_regions": n_regions,
        "n_merge_ops": len(merge_record),
        "coverage_ok": bool(coverage_ok),
        "merge_steps": len(merge_record),
        "merge_record": merge_record,
        "slic_params": {
            "n_segments": n_segments,
            "compactness": compactness,
            "sigma": sigma,
            "color_space": slic_cfg.get("color_space", "Lab"),
        },
        "merge_params": dict(merge_cfg),
        "connectivity": connectivity,
        "vanished_regions": vanished_regions,
        "config_fingerprint": config_fingerprint,
        "seed": seed,
        "elapsed_sec": float(time.time() - t_start),
    }

    pack = RegionPack(
        region_ids=region_ids,
        labels=labels,
        soft_masks=soft_masks,
        edges=edges,
        metadata=metadata,
    )

    if return_diagnostics:
        return pack, {
            "labels": labels,
            "edges_raw": edges,
            "merge_record": merge_record,
            "metadata": metadata,
        }

    return pack