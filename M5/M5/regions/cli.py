"""Command-line interface for M5 region partition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from .partition import build_regions
from .baselines import uniform_grid_partition, global_single_region
from .region_types import RegionPack
from .graph import equal_weight_graph


def _load_image_rgb(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _save_region_pack(path: str, pack: RegionPack) -> None:
    np.savez_compressed(
        path,
        region_ids=pack.region_ids,
        labels=pack.labels,
        soft_masks=pack.soft_masks,
        edges=np.array(pack.edges, dtype=object),
        metadata=np.array(pack.metadata, dtype=object),
    )


def _save_metadata_json(path: str, pack: RegionPack) -> None:
    meta = pack.metadata.copy()
    meta["edges"] = [(int(i), int(j), float(w)) for (i, j, w) in pack.edges]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _save_edges_json(path: str, edges) -> None:
    data = [{"i": int(i), "j": int(j), "weight": float(w)} for (i, j, w) in edges]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_equal_weight_edges(path: str, labels: np.ndarray) -> None:
    """Save equal-weight graph edges computed from labels."""
    eq = equal_weight_graph(labels)
    data = [{"i": int(i), "j": int(j), "weight": float(w)} for (i, j, w) in eq]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_partition_preview(path: str, labels: np.ndarray) -> None:
    try:
        from .visualize import draw_partition
        draw_partition(labels, path)
    except Exception as e:
        import warnings
        warnings.warn(f"Could not save partition preview: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M5 Region Partition CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--content", required=True,
        help="Path to the content image (RGB PNG/JPG).",
    )
    parser.add_argument(
        "--feature-height", type=int, required=True,
        help="Feature map height Hf.",
    )
    parser.add_argument(
        "--feature-width", type=int, required=True,
        help="Feature map width Wf.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to regions.json config file.",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/regions_out",
        help="Output directory.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device.",
    )
    parser.add_argument(
        "--grid-baseline", action="store_true",
        help="Also generate grid baseline partition.",
    )
    parser.add_argument(
        "--single-baseline", action="store_true",
        help="Also generate single-region baseline.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed in config.",
    )

    args = parser.parse_args()

    # ---- Load config ----
    region_config: Dict = {}
    if args.config is not None:
        if not os.path.isfile(args.config):
            print(f"ERROR: config not found: {args.config}")
            return 1
        with open(args.config, "r") as f:
            region_config = json.load(f)
        print(f"Config      : {args.config}")

    if args.seed is not None:
        region_config["seed"] = args.seed

    # ---- Load image ----
    if not os.path.isfile(args.content):
        print(f"ERROR: image not found: {args.content}")
        return 1

    content_rgb = _load_image_rgb(args.content)
    print(f"Input       : {args.content}")
    print(f"Input shape : {content_rgb.shape} (uint8, no resize/crop)")
    print(f"Input hash  : {content_rgb.shape}:uint8:{hash(content_rgb.tobytes()) & 0xffffffff:x}")
    print(f"Feature hw  : ({args.feature_height}, {args.feature_width})")

    # ---- Output dir ----
    os.makedirs(args.output, exist_ok=True)

    # ---- Device ----
    device = None
    if args.device == "cuda":
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            device = None

    # ---- Build regions ----
    print("\n--- Building regions ---")
    pack = build_regions(
        content_rgb=content_rgb,
        feature_hw=(args.feature_height, args.feature_width),
        device=device,
        region_config=region_config,
    )

    n_regions = int(pack.labels.max()) + 1
    n_edges = len(pack.edges)
    print(f"Regions     : {n_regions} (initial {pack.metadata.get('initial_regions', '?')}, "
          f"merges {pack.metadata.get('n_merge_ops', '?')})")
    print(f"Edges       : {n_edges}")
    print(f"Soft masks  : {pack.soft_masks.shape}")
    print(f"Coverage ok : {pack.metadata.get('coverage_ok', '?')}")
    print(f"Connected   : {pack.metadata.get('connectivity', {}).get('all_connected', '?')}")

    # ---- Save outputs ----
    npz_path = os.path.join(args.output, "region_pack.npz")
    _save_region_pack(npz_path, pack)
    print(f"\nSaved NPZ   : {npz_path}")

    meta_path = os.path.join(args.output, "metadata.json")
    _save_metadata_json(meta_path, pack)
    print(f"Saved meta  : {meta_path}")

    edges_path = os.path.join(args.output, "edges.json")
    _save_edges_json(edges_path, pack.edges)
    print(f"Saved edges : {edges_path}")

    # ---- Equal-weight edges (from labels, not edges) ----
    eq_path = os.path.join(args.output, "edges_equal_weight.json")
    _save_equal_weight_edges(eq_path, pack.labels)
    print(f"Saved eq-edg: {eq_path}")

    # ---- Preview ----
    preview_path = os.path.join(args.output, "partition_preview.png")
    _save_partition_preview(preview_path, pack.labels)
    print(f"Saved preview: {preview_path}")

    # ---- Baselines ----
    if args.grid_baseline:
        print("\n--- Grid baseline ---")
        grid_pack = uniform_grid_partition(
            content_rgb=content_rgb,
            target_regions=12,
            feature_hw=(args.feature_height, args.feature_width),
            device=device,
        )
        grid_npz = os.path.join(args.output, "region_pack_grid.npz")
        _save_region_pack(grid_npz, grid_pack)
        print(f"Saved grid  : {grid_npz}")

    if args.single_baseline:
        print("\n--- Single-region baseline ---")
        single_pack = global_single_region(
            content_rgb=content_rgb,
            feature_hw=(args.feature_height, args.feature_width),
            device=device,
        )
        single_npz = os.path.join(args.output, "region_pack_single.npz")
        _save_region_pack(single_npz, single_pack)
        print(f"Saved single: {single_npz}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())