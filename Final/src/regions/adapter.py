"""M5 -> M4 共同合同适配器。

default.json 的 region_adapter 为 "src.regions.adapter:build_regions"。
本模块把 M5 自有的 RegionPack（region_types.py）转换成 M4 共同合同
src.contracts.RegionPack（region_ids 为 tuple、edges 为 tuple），并处理：
- content_rgb 入参类型（M4 传 PIL Image，M5 partition 收 np.ndarray）；
- device 字符串 -> torch.device；
- 默认 region_config 从 resources/configs/regions.json 读取。

转换不改写 M5 的分区/掩码/图结果，只包装。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.contracts import RegionPack  # noqa: E402

from .partition import build_regions as _m5_build_regions  # noqa: E402

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "configs" / "regions.json"
)
_config_cache: dict | None = None


def _load_default_config() -> dict:
    global _config_cache
    if _config_cache is None:
        _config_cache = json.loads(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    return _config_cache


def build_regions(
    content_rgb, *, feature_hw, device, region_config: dict | None = None
) -> RegionPack:
    """合同入口：与 M4 交接文档的签名一致。"""
    if hasattr(content_rgb, "convert"):  # PIL Image -> np.ndarray (RGB, HxWx3)
        array = np.asarray(content_rgb.convert("RGB"))
    else:
        array = np.asarray(content_rgb)
    cfg = region_config if region_config is not None else _load_default_config()
    pack = _m5_build_regions(
        array,
        feature_hw=tuple(feature_hw),
        device=torch.device(device) if device is not None else None,
        region_config=cfg,
    )
    return RegionPack(
        region_ids=tuple(int(i) for i in pack.region_ids),
        labels=pack.labels,
        soft_masks=pack.soft_masks,
        edges=tuple((int(i), int(j), float(w)) for i, j, w in pack.edges),
        metadata={**pack.metadata, "adapter": "src.regions.adapter"},
    )


def build_equal_weight_regions(content_rgb, *, feature_hw, device):
    """Same adaptive partition and masks, with only coordination weights ablated."""
    from dataclasses import replace

    regions = build_regions(content_rgb, feature_hw=feature_hw, device=device)
    return replace(
        regions,
        edges=tuple((i, j, 1.0) for i, j, _ in regions.edges),
        metadata={**regions.metadata, "edge_weight_ablation": "equal"},
    )
