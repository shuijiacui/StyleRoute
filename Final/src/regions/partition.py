"""Deterministic SLIC; merge one current four-connected neighbour at a time."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import slic

from .graph import build_region_graph, normalized_rgb
from .masks import generate_soft_masks, validate_soft_masks
from .region_types import RegionPack

STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _run_slic(img, n_segments=64, compactness=10.0, sigma=1.0, seed=42):
    # SLIC is deterministic: retain seed in provenance without changing global RNGs.
    return slic(
        normalized_rgb(img),
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=2,
        convert2lab=True,
        enforce_connectivity=True,
    ).astype(np.int32)


def _renormalise_ids(labels):
    return np.unique(labels, return_inverse=True)[1].reshape(labels.shape).astype(np.int32)


def _connected_ids(labels):
    result = np.zeros_like(labels, dtype=np.int32)
    offset = 0
    for rid in np.unique(labels):
        components, count = ndimage.label(labels == rid, structure=STRUCTURE)
        covered = components > 0
        result[covered] = components[covered] - 1 + offset
        offset += count
    return result


def _merge_regions(labels, img, merge_cfg, norm_cfg):
    cur = _connected_ids(labels)
    records = []
    maximum = int(merge_cfg.get("max_regions", 16))
    minimum = int(merge_cfg.get("min_regions", min(8, maximum)))
    threshold = float(merge_cfg.get("threshold", 0.35))
    min_area = int(merge_cfg.get("min_area", 16))
    if not 1 <= minimum <= maximum or min_area < 0 or threshold < 0:
        raise ValueError("Invalid region merge limits")
    while True:
        count = int(cur.max()) + 1
        if count <= minimum:
            break
        areas = np.bincount(cur.ravel())
        tiny = set(np.flatnonzero(areas < min_area).tolist())
        if count <= maximum and not tiny:
            break
        edges = build_region_graph(
            cur, img, {"merge": merge_cfg, "normalization": norm_cfg}, "merge_cost"
        )
        candidates = [
            (i, j, cost)
            for i, j, cost in edges
            if cost <= threshold and (count > maximum or i in tiny or j in tiny)
        ]
        if not candidates:
            break  # Keep actual N when all boundaries are unsafe; never spin or force.
        i, j, cost = min(
            candidates, key=lambda e: (not (e[0] in tiny or e[1] in tiny), e[2], e[0], e[1])
        )
        dropped_area = int(areas[j])
        cur[cur == j] = i
        records.append(
            {
                "keep": i,
                "drop": j,
                "cost": cost,
                "merged_area": dropped_area,
                "reason": "tiny_region" if i in tiny or j in tiny else "region_limit",
                "ids_before_reindex": True,
            }
        )
        cur = _renormalise_ids(cur)
        # Exactly one region disappears; rebuild all adjacency/statistics next iteration.
    return cur, records


def build_regions(
    content_rgb,
    feature_hw=(16, 16),
    device=None,
    dtype=torch.float32,
    region_config=None,
    return_diagnostics=False,
):
    start = time.perf_counter()
    rgb = normalized_rgb(content_rgb)
    cfg = dict(region_config or {})
    slic_cfg, merge_cfg = cfg.get("slic", {}), cfg.get("merge", {})
    if np.allclose(rgb, rgb[0, 0], atol=1e-6):
        labels = np.zeros(rgb.shape[:2], np.int32)
    else:
        labels = _connected_ids(
            _run_slic(
                rgb,
                n_segments=int(slic_cfg.get("n_segments", 64)),
                compactness=float(slic_cfg.get("compactness", 10)),
                sigma=float(slic_cfg.get("sigma", 1)),
                seed=cfg.get("seed", 42),
            )
        )
    initial = int(labels.max()) + 1
    labels, records = _merge_regions(labels, rgb, merge_cfg, cfg.get("normalization", {}))
    count = int(labels.max()) + 1
    ids = np.arange(count, dtype=np.int32)
    mask_cfg = cfg.get("masks", {})
    soft = generate_soft_masks(
        labels,
        ids,
        feature_hw,
        sigma=float(mask_cfg.get("sigma", 2)),
        eps=float(mask_cfg.get("eps", 1e-8)),
        device=device,
        dtype=dtype,
    )
    validate_soft_masks(soft, atol=1e-5)
    components = {int(i): int(ndimage.label(labels == i, structure=STRUCTURE)[1]) for i in ids}
    areas = np.bincount(labels.ravel())
    metadata = {
        "method": "content_adaptive",
        "version": "integrated-v2",
        "num_regions": count,
        "input_shape": list(rgb.shape),
        "input_range": "float32 [0,1]",
        "feature_hw": list(feature_hw),
        "initial_regions": initial,
        "n_initial_superpixels": int(slic_cfg.get("n_segments", 64)),
        "n_final_regions": count,
        "n_merge_ops": len(records),
        "merge_steps": len(records),
        "merge_record": records,
        "coverage_ok": int(areas.sum()) == labels.size,
        "areas": areas.tolist(),
        "mask_effective_areas": soft.sum((1, 2)).cpu().tolist(),
        "connectivity": {
            "n_connected_components": sum(components.values()),
            "per_region_components": components,
            "all_connected": all(n == 1 for n in components.values()),
        },
        "vanished_regions": [],
        "slic_params": slic_cfg,
        "merge_params": merge_cfg,
        "masks": {"resize": "one_hot_area", "sigma": float(mask_cfg.get("sigma", 2))},
        "region_limit_reached": count <= int(merge_cfg.get("max_regions", 16)),
        "small_regions_remaining": np.flatnonzero(
            areas < int(merge_cfg.get("min_area", 16))
        ).tolist(),
        "config_fingerprint": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "input_fingerprint": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "seed": cfg.get("seed", 42),
        "elapsed_sec": time.perf_counter() - start,
    }
    edges = build_region_graph(labels, rgb, cfg)
    result = RegionPack(ids, labels, soft, edges, metadata)
    if return_diagnostics:
        return result, {
            "labels": labels,
            "edges_raw": edges,
            "merge_record": records,
            "metadata": metadata,
        }
    return result
