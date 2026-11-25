"""Project paths, configuration and serializable provenance utilities."""

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Historical run snapshots are evidence and are not rewritten during reorganization.
# Resolve their former resource paths without recreating old top-level directories.
LEGACY_PATHS = {
    Path("vendor/adain/upstream"): Path("resources/upstream/adain"),
    Path("assets/fonts"): Path("resources/fonts"),
    Path("weights"): Path("resources/weights"),
    Path("configs"): Path("resources/configs"),
    Path("data"): Path("resources/images"),
    Path("runs"): Path("outputs/runs"),
    Path("dist"): Path("outputs/packages"),
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for previous, current in LEGACY_PATHS.items():
        if path.is_relative_to(previous):
            return ROOT / current / path.relative_to(previous)
    return ROOT / path


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path = "resources/configs/default.json") -> dict[str, Any]:
    config = json.loads(project_path(path).read_text(encoding="utf-8-sig"))
    height, width = config["content_hw"]
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v < 32 or v % 8 for v in (height, width)
    ):
        raise ValueError("content_hw must contain multiples of 8, each at least 32")
    if config["content_resize_mode"] != "fit_center_crop" or config["dtype"] != "float32":
        raise ValueError("Only fit_center_crop and float32 are supported")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if config["style_long_side"] < config["minimum_style_short_side"] or config["adain_eps"] <= 0:
        raise ValueError("Invalid style sizes or AdaIN epsilon")
    return config


def runtime_info(device=None) -> dict[str, Any]:
    import torch

    versions = {}
    for package in (
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "streamlit",
        "psutil",
        "cma",
        "timm",
        "pyiqa",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    actual = torch.device(device or "cpu")
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": versions,
        "actual_device": str(actual),
        "device_name": torch.cuda.get_device_name(actual)
        if actual.type == "cuda"
        else platform.processor() or "CPU",
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "dtype": "float32",
        "cpu_threads": torch.get_num_threads(),
    }
