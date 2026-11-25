"""Read saved canonical tensors and provenance; never guess a partition or final state."""

import json

import numpy as np
import torch

from src.models.preprocessing import prepare_content
from src.project import file_sha256, project_path


def read_result(row):
    metadata = json.loads(project_path(row["result_metadata"]).read_text(encoding="utf-8"))
    if metadata.get("format_version") != "integrated-v2":
        raise ValueError("Historical results require their original code/configuration")
    path = project_path(row["selected_tensor"])
    if file_sha256(path) != metadata["selected_sha256"]:
        raise ValueError("Selected image checksum mismatch")
    regions = project_path(row["region_artifact"])
    if file_sha256(regions) != metadata["regions_sha256"]:
        raise ValueError("Region artifact checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        image = torch.from_numpy(archive["image_rgb"].copy())
    artifact_dir = project_path(row["artifact_dir"])
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "input_manifest.json").read_text(encoding="utf-8"))
    content_path = artifact_dir / "inputs/content.png"
    if file_sha256(content_path) != manifest["inputs"]["content"]["sha256"]:
        raise ValueError("Saved content checksum mismatch")
    content = prepare_content(content_path, tuple(config["content_hw"])).tensor
    return image, content, metadata


def final_alphas(row):
    metadata = json.loads(project_path(row["result_metadata"]).read_text(encoding="utf-8"))
    return tuple(metadata["region_alphas"])
