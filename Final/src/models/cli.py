"""python -m src.models.cli --content ... --style ... --output outputs/runs/demo/default.png"""

import argparse

import numpy as np
import torch

from src.contracts import RegionPack
from src.models.adain import AdaINEngine
from src.outputs import dataset_pair_id, save_run
from src.postcard.render import CANVAS_WH, DESIGN_VERSION, TEMPLATES, render_postcard
from src.project import file_sha256, load_config, project_path


def main():
    parser = argparse.ArgumentParser(description="M4 real AdaIN image generation")
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument(
        "--alpha", type=float, help="Global strength; defaults to config default_alpha"
    )
    parser.add_argument("--config", default="resources/configs/default.json")
    parser.add_argument("--output", required=True, help="PNG filename in a new run directory")
    parser.add_argument("--regions", help="M5 NPZ: labels, soft_masks, region_ids, optional edges")
    parser.add_argument("--alphas", help="Comma-separated region strengths, in region ID order")
    parser.add_argument("--postcard", action="store_true")
    parser.add_argument("--title", default="山海来信")
    parser.add_argument("--message", default="把眼前的风景，寄给远方的你。")
    parser.add_argument("--template", default="minimal", choices=TEMPLATES)
    args = parser.parse_args()
    output = project_path(args.output)
    if output.suffix.lower() != ".png":
        parser.error("--output must be a PNG filename")
    if output.name in {
        "best.png",
        "postcard.png",
        "content_processed.png",
        "alpha_map.png",
        "region_overlay.png",
    }:
        parser.error("--output name conflicts with a reserved run artifact; use default.png")
    if bool(args.regions) != bool(args.alphas):
        parser.error("--regions and --alphas must be supplied together")
    engine = AdaINEngine(load_config(args.config))
    content, style = project_path(args.content), project_path(args.style)
    pair = engine.prepare(content, style, pair_id=dataset_pair_id(content, style))
    default = engine.render_global(
        pair, engine.config["default_alpha"] if args.alpha is None else args.alpha
    )
    regions, best = None, None
    if args.regions:
        with np.load(project_path(args.regions), allow_pickle=False) as pack:
            if not np.issubdtype(pack["region_ids"].dtype, np.integer):
                raise ValueError("NPZ region_ids must have an integer dtype")
            regions = RegionPack(
                tuple(int(v) for v in pack["region_ids"]),
                pack["labels"],
                torch.as_tensor(pack["soft_masks"], device=engine.device),
                tuple((int(i), int(j), float(w)) for i, j, w in pack["edges"])
                if "edges" in pack else (),
            )
        best = engine.render_regions(pair, regions, tuple(float(v) for v in args.alphas.split(",")))
    postcard = None
    font = project_path(engine.config["font_path"])
    heading = project_path(
        engine.config.get("postcard_heading_font_path", engine.config["font_path"])
    )
    if args.postcard:
        postcard = render_postcard(
            (best or default).image_rgb,
            args.title,
            args.message,
            args.template,
            font,
            title_font_path=heading,
        )
    directory = save_run(
        output.parent,
        engine,
        pair,
        default,
        content,
        style,
        best=best,
        regions=regions,
        postcard=postcard,
        postcard_metadata={
            "title": args.title,
            "message": args.message,
            "template": args.template,
            "font_sha256": file_sha256(font) if postcard else None,
            "generation_hw": pair.input_hw,
            "canvas_wh": list(CANVAS_WH),
            "heading_font_sha256": file_sha256(heading) if postcard else None,
            "design_version": DESIGN_VERSION,
        },
    )
    if output.name != "default.png":
        (directory / "default.png").replace(output)
    print(f"Saved real result to {output}; device={engine.device}; feature_hw={pair.feature_hw}")


if __name__ == "__main__":
    main()
