"""Save a concrete local run with inputs and provenance; never invent search evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.preprocessing import load_rgb, png_bytes
from src.project import file_sha256, project_path, utc_now, write_json


def alpha_image(alpha_map: torch.Tensor) -> Image.Image:
    pixels = alpha_map.detach()[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(pixels)


def region_overlay(content_rgb, labels):
    colors = np.stack(
        ((labels * 67 + 40) % 255, (labels * 113 + 80) % 255, (labels * 173 + 120) % 255), axis=-1
    ).astype("uint8")
    return Image.blend(content_rgb, Image.fromarray(colors), 0.35)


def input_source_record(source):
    """Link known input bytes to the fixed dataset; never guess an upload's rights."""
    if isinstance(source, bytes):
        fingerprint = hashlib.sha256(source).hexdigest()
    elif isinstance(source, (str, Path)):
        fingerprint = file_sha256(source)
    else:
        return {"source": "in-memory image", "dataset_id": None}
    manifest_path = project_path("resources/images/manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["content_images"] + manifest["style_images"]:
            if item["sha256"] == fingerprint:
                return {
                    key: item.get(key)
                    for key in (
                        "id",
                        "source",
                        "sha256",
                        "object_url",
                        "artist",
                        "license",
                        "license_url",
                        "usage",
                    )
                }
    return {
        "source": "user input",
        "sha256": fingerprint,
        "dataset_id": None,
        "rights_note": "Not inferred from the filename or image contents",
    }


def dataset_pair_id(content_source, style_source):
    content, style = input_source_record(content_source), input_source_record(style_source)
    return f"{content['id']}__{style['id']}" if content.get("id") and style.get("id") else None


def save_run(
    directory,
    engine,
    pair,
    default,
    content_source,
    style_source,
    *,
    best=None,
    regions=None,
    search_result=None,
    postcard=None,
    postcard_metadata=None,
):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "runtime.json").exists():
        raise FileExistsError(f"Run already exists; choose a new output directory: {target}")
    (target / "inputs").mkdir(exist_ok=True)
    (target / "inputs/content.png").write_bytes(png_bytes(load_rgb(content_source)))
    (target / "inputs/style.png").write_bytes(png_bytes(load_rgb(style_source)))
    (target / "content_processed.png").write_bytes(png_bytes(pair.content_rgb))
    (target / "default.png").write_bytes(png_bytes(default.image_rgb))
    selected = best if best is not None else default
    if best is not None:
        (target / "best.png").write_bytes(png_bytes(best.image_rgb))
    alpha_image(selected.alpha_map).save(target / "alpha_map.png")
    if regions is not None:
        labels = regions.labels
        # A deterministic diagnostic overlay, not an alternative segmentation algorithm.
        region_overlay(pair.content_rgb, labels).save(target / "region_overlay.png")
        np.savez_compressed(
            target / "regions.npz",
            labels=labels,
            soft_masks=regions.soft_masks.detach().cpu().numpy(),
            region_ids=np.array(regions.region_ids),
            edges=np.array(regions.edges, dtype=np.float64).reshape(-1, 3),
        )
    if postcard is not None:
        (target / "postcard.png").write_bytes(postcard)
        write_json(target / "postcard.json", postcard_metadata)
    write_json(target / "config.json", engine.config)
    write_json(
        target / "runtime.json",
        {
            **engine.runtime,
            "counters": dict(engine.counters),
            "counters_scope": "cumulative engine lifetime",
        },
    )
    write_json(
        target / "input_manifest.json",
        {
            "pair_id": pair.pair_id,
            "cache_key": pair.cache_key,
            "preparation": pair.metadata,
            "original_sources": {
                "content": input_source_record(content_source),
                "style": input_source_record(style_source),
            },
            "inputs": {
                name: {
                    "path": f"inputs/{name}.png",
                    "sha256": file_sha256(target / f"inputs/{name}.png"),
                }
                for name in ("content", "style")
            },
            "dataset_manifest": "resources/images/manifest.json; original licenses still apply",
        },
    )
    events = [{"at": utc_now(), **default.metadata}]
    if best is not None:
        events.append({"at": utc_now(), **best.metadata})
    (target / "generation.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n" for event in events),
        encoding="utf-8",
    )
    if search_result is not None:
        (target / "search.jsonl").write_text(
            "".join(
                json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                for event in search_result.trajectory
            ),
            encoding="utf-8",
        )
        write_json(
            target / "search_summary.json",
            {
                key: getattr(search_result, key)
                for key in (
                    "status",
                    "region_alphas",
                    "metrics",
                    "candidate_count",
                    "decoder_calls",
                    "state_cache_hits",
                    "metadata",
                )
            },
        )
    return target
