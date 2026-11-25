"""Validate real adaptive regions on every content image, plus M4 fusion and cache."""

import argparse
import json

import numpy as np
import torch

from src.models.adain import AdaINEngine
from src.models.preprocessing import prepare_content
from src.project import load_config, project_path, write_json
from src.regions.adapter import build_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/validation/regions.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(
        project_path("resources/images/manifest.json").read_text(encoding="utf-8")
    )
    rows = []
    engine = AdaINEngine(load_config())
    pair = engine.prepare(
        project_path(manifest["content_images"][0]["local_path"]),
        project_path(manifest["style_images"][0]["local_path"]),
    )
    for entry in manifest["content_images"]:
        image = prepare_content(project_path(entry["local_path"]), pair.input_hw).rgb
        regions = build_regions(image, feature_hw=pair.feature_hw, device="cpu")
        repeated = build_regions(image, feature_hw=pair.feature_hw, device="cpu")
        assert np.array_equal(regions.labels, repeated.labels)
        assert torch.equal(regions.soft_masks, repeated.soft_masks)
        assert regions.edges == repeated.edges
        components = regions.metadata["connectivity"]["per_region_components"]
        assert all(value == 1 for value in components.values())
        assert regions.metadata["connectivity"]["all_connected"]
        assert (regions.soft_masks.sum((1, 2)) > 0).all()
        torch.testing.assert_close(
            regions.soft_masks.sum(0), torch.ones(pair.feature_hw), atol=1e-5, rtol=0
        )
        rows.append(
            {
                "id": entry["id"],
                "region_count": len(regions.region_ids),
                "all_connected": True,
                "all_equal_weights": len(set(w for _, _, w in regions.edges)) <= 1,
                "weight_min": min((w for _, _, w in regions.edges), default=None),
                "weight_max": max((w for _, _, w in regions.edges), default=None),
                "soft_coverage": True,
                "deterministic": True,
            }
        )
        print(f"[PASS] {entry['id']}: {len(regions.region_ids)} regions", flush=True)
    regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=engine.device)
    prepared = engine.prepare_regions(pair, regions)
    global_image = engine.render_global(pair, 0.6)
    regional = engine.render_regions(pair, prepared, (0.6,) * len(regions.region_ids))
    torch.testing.assert_close(global_image.image_rgb, regional.image_rgb, rtol=1e-5, atol=2e-6)
    write_json(
        project_path(args.output),
        {
            "version": "integrated-v2",
            "status": "passed",
            "output_hw": pair.input_hw,
            "feature_hw": pair.feature_hw,
            "content_images": rows,
            "fusion_max_error": float((global_image.image_rgb - regional.image_rgb).abs().max()),
            "scope": "Real SLIC/regions and pretrained AdaIN generation, not search quality",
        },
    )


if __name__ == "__main__":
    main()
