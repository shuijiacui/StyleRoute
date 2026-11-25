"""Freeze objective scales using only the 12 manifest pilot baselines, never evaluation."""

import argparse
import json

import numpy as np

from src.integration import load_adapter
from src.models.adain import AdaINEngine
from src.models.compensation import ContentGeometry
from src.project import digest, file_sha256, load_config, project_path, write_json
from src.search.aesthetics import AestheticMetric
from src.search.landscape import UniformityMetric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Update default and ablation configs")
    parser.add_argument("--out", default="outputs/validation/landscape/calibration.json")
    args = parser.parse_args()
    manifest_path = project_path("resources/images/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = load_config()
    engine = AdaINEngine(config)
    build_regions = load_adapter(config["region_adapter"])
    metric = AestheticMetric(engine.device)
    paths = {
        e["id"]: project_path(e["local_path"])
        for group in ("content_images", "style_images")
        for e in manifest[group]
    }
    rows = []
    for info in manifest["pilot_pairs"]:
        pair = engine.prepare(
            paths[info["content_id"]], paths[info["style_id"]], pair_id=info["pair_id"]
        )
        regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=engine.device)
        baseline = engine.render_global(pair, 0.6).image_rgb
        u = UniformityMetric(ContentGeometry(pair.content_tensor, regions.labels))(baseline)
        rows.append(
            {"pair_id": pair.pair_id, "A": metric(baseline), **u, "pair_cache_key": pair.cache_key}
        )
        print(f"[{len(rows)}/12] {pair.pair_id}: A={rows[-1]['A']:.4f} U={u['U']:.4f}", flush=True)
    scales = {
        "aesthetic_scale": max(
            1.0,
            float(
                np.quantile([r["A"] for r in rows], 0.85)
                - np.quantile([r["A"] for r in rows], 0.15)
            ),
        ),
        "uniformity_scale": max(0.25, float(np.median([r["U"] for r in rows]))),
        "structure_scale": 1.0,
    }
    report = {
        "split": "pilot",
        "count": len(rows),
        "rows": rows,
        "scales": scales,
        "manifest_sha256": file_sha256(manifest_path),
        "aesthetic": metric.provenance,
        "rule": "A=max(1,p85-p15); U=max(.25,median baseline U); Q=baseline-relative",
        "weights": {"A": 0.4, "U": 0.4, "Q": 0.2},
        "style_slack": 0.05,
        "weights_status": "prespecified, not human-fitted",
    }
    report["calibration_id"] = digest(report)
    write_json(project_path(args.out), report)
    if args.apply:
        for name in ("default.json", "landscape_experiment.json"):
            path = project_path("resources/configs/" + name)
            settings = json.loads(path.read_text(encoding="utf-8"))
            search = settings["search_config"] if name == "default.json" else settings
            search["landscape"].update(scales)
            search["calibration"] = report["calibration_id"]
            write_json(path, settings)
        print("Frozen pilot scales in default and landscape experiment configurations.")


if __name__ == "__main__":
    main()
