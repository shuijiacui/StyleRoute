"""Check a real M5 adapter, optionally M6; never substitute synthetic production results."""

import argparse
from datetime import datetime

import torch

from src.contracts import RegionPack
from src.integration import load_adapter, run_search
from src.models.adain import AdaINEngine
from src.outputs import dataset_pair_id, save_run
from src.project import load_config, project_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="resources/configs/default.json")
    parser.add_argument(
        "--content", default="resources/images/content/pilot/mountain/mountain_01.jpg"
    )
    parser.add_argument("--style", default="resources/images/styles/met_436535.jpg")
    parser.add_argument("--m5-only", action="store_true", help="Only check real regions and fusion")
    parser.add_argument("--output", help="New run directory, relative to the project root")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["default_alpha"] != 0.6:
        parser.error("Use the formal 0.6 baseline configuration for handoff checks")
    build_regions = load_adapter(config["region_adapter"])
    if not args.m5_only:
        load_adapter(config["search_adapter"])
    directory = project_path(
        args.output or ("outputs/runs/handoff_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if directory.exists():
        raise FileExistsError(f"Choose a new output directory: {directory}")
    engine = AdaINEngine(config)
    content, style = project_path(args.content), project_path(args.style)
    pair = engine.prepare(content, style, pair_id=dataset_pair_id(content, style))
    regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=engine.device)
    if not isinstance(regions, RegionPack):
        raise ValueError("M5 must import and return src.contracts.RegionPack")
    prepared = engine.prepare_regions(pair, regions)
    initial = engine.render_global(pair, 0.6)
    uniform = engine.render_regions(pair, prepared, (0.6,) * len(regions.region_ids))
    tolerance = {"rtol": 1e-5, "atol": 2e-6}
    torch.testing.assert_close(initial.image_rgb, uniform.image_rgb, **tolerance)
    before = dict(engine.counters)
    content_features = pair.content_features.clone()
    adain_features = pair.adain_features.clone()
    candidate_state = (0.3,) + (0.6,) * (len(regions.region_ids) - 1)
    candidate = engine.render_regions(pair, prepared, candidate_state)
    repeated = engine.render_regions(pair, prepared, candidate_state)
    torch.testing.assert_close(candidate.image_rgb, repeated.image_rgb, rtol=0, atol=0)
    torch.testing.assert_close(pair.content_features, content_features, rtol=0, atol=0)
    torch.testing.assert_close(pair.adain_features, adain_features, rtol=0, atol=0)
    assert engine.counters["encoder_calls"] == before["encoder_calls"]
    report = {
        "status": "passed",
        "scope": "real M5 contract and M4 generation; not segmentation quality evaluation",
        "region_count": len(regions.region_ids),
        "feature_hw": pair.feature_hw,
        "output_hw": pair.input_hw,
        "style_hw": pair.metadata["style_hw"],
        "default_alpha": config["default_alpha"],
        "uniform_max_abs_error": float((initial.image_rgb - uniform.image_rgb).abs().max()),
        "repeat_max_abs_error": float((candidate.image_rgb - repeated.image_rgb).abs().max()),
        **tolerance,
        "search_executed": not args.m5_only,
    }
    best, search_result = None, None
    if not args.m5_only:
        before_search = dict(engine.counters)
        regions, best, search_result = run_search(engine, pair, initial)
        report.update(
            scope="real M5/M6 integration contract; result status is reported independently",
            search_status=search_result.status,
            search_decoder_calls_observed=(
                engine.counters["decoder_calls"] - before_search["decoder_calls"]
            ),
        )
    save_run(
        directory,
        engine,
        pair,
        initial,
        content,
        style,
        best=best,
        regions=regions,
        search_result=search_result,
    )
    write_json(directory / "handoff_checks.json", report)
    print(f"Handoff checks saved: {directory}")


if __name__ == "__main__":
    main()
