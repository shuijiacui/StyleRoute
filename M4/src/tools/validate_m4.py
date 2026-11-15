"""Generate real pilot samples and numerical evidence. No M5/M6 results are simulated."""

import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.contracts import RegionPack
from src.models.adain import AdaINEngine
from src.models.preprocessing import load_rgb, png_bytes, tensor_to_pil
from src.outputs import alpha_image
from src.postcard.render import render_postcard
from src.project import load_config, project_path, utc_now, write_json


def main():
    directory = project_path("outputs/runs/validation")
    directory.mkdir(parents=True, exist_ok=True)
    config = load_config()
    engine = AdaINEngine(config)
    manifest = json.loads(
        project_path("resources/images/manifest.json").read_text(encoding="utf-8")
    )
    pilots = sorted(
        (item for item in manifest["content_images"] if item["usage"] == "pilot"),
        key=lambda item: item["class_name"],
    )
    styles = manifest["style_images"]
    font_path = project_path(config["font_path"])
    font = ImageFont.truetype(str(font_path), 18)
    grid = Image.new("RGB", (1320, 1000), "#fffdf7")
    draw = ImageDraw.Draw(grid)
    draw.text(
        (15, 10),
        "真实 AdaIN 试运行样例 · 每行一张内容图 · 不代表区域搜索结果",
        font=font,
        fill="#294034",
    )
    for index, label in enumerate(["处理后内容图"] + [item["label_zh"] for item in styles]):
        draw.text((12 + index * 263, 48), label, font=font, fill="#294034")
    records = []
    for row, content in enumerate(pilots):
        for column, style in enumerate(styles):
            pair = engine.prepare(
                project_path(content["local_path"]), project_path(style["local_path"])
            )
            result = engine.render_global(pair, 0.6)
            filename = f"{content['id']}__{style['id']}.png"
            (directory / filename).write_bytes(png_bytes(result.image_rgb))
            y = 86 + row * 300
            if column == 0:
                grid.paste(pair.content_rgb, (8, y))
                draw.text((12, y + 262), content["label_zh"], font=font, fill="#294034")
            grid.paste(tensor_to_pil(result.image_rgb), (8 + (column + 1) * 263, y))
            records.append(
                {
                    "content_id": content["id"],
                    "style_id": style["id"],
                    "pair_id": pair.pair_id,
                    "output": filename,
                    "kind": "real_global_adain_pilot",
                    "alpha": 0.6,
                }
            )
    grid.save(directory / "pilot_grid.jpg")

    pair = engine.prepare(
        project_path("resources/images/content/pilot/mountain/mountain_01.jpg"),
        project_path("resources/images/styles/met_436535.jpg"),
    )
    labels = np.zeros(pair.input_hw, dtype=np.int64)
    labels[:, pair.input_hw[1] // 2 :] = 1
    ramp = torch.linspace(1, 0, pair.feature_hw[1], device=engine.device)
    left = ramp.repeat(pair.feature_hw[0], 1)
    masks = torch.stack([left, 1 - left])
    raw = RegionPack(
        (0, 1),
        labels,
        masks,
        ((0, 1, 1.0),),
        {"test_only": True, "note": "Controlled left/right soft masks, not M5 segmentation"},
    )
    prepared = engine.prepare_regions(pair, raw)
    before = (pair.content_features.clone(), pair.adain_features.clone())
    default = engine.render_global(pair, 0.6)
    constant = engine.render_regions(pair, prepared, (0.6, 0.6))
    torch.testing.assert_close(default.image_rgb, constant.image_rgb, rtol=1e-5, atol=1e-6)
    first = engine.render_regions(pair, prepared, (0.3, 0.9))
    engine.render_regions(pair, prepared, (0.9, 0.3))
    repeated = engine.render_regions(pair, prepared, (0.3, 0.9))
    permuted = engine.prepare_regions(pair, RegionPack((0, 1), 1 - labels, masks.flip(0)))
    permuted_result = engine.render_regions(pair, permuted, (0.9, 0.3))
    torch.testing.assert_close(first.image_rgb, repeated.image_rgb, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first.image_rgb, permuted_result.image_rgb, rtol=1e-5, atol=1e-6)
    assert torch.equal(before[0], pair.content_features) and torch.equal(
        before[1], pair.adain_features
    )
    (directory / "controlled_regions_test_only.png").write_bytes(png_bytes(first.image_rgb))
    alpha_image(first.alpha_map).save(directory / "controlled_alpha_test_only.png")
    for template in ("minimal", "border"):
        postcard = render_postcard(
            default.image_rgb,
            "山海来信",
            "把群山的轮廓，留在这张明信片里。\n愿远方的你，也能看见此刻的风景。",
            template,
            font_path,
        )
        (directory / f"postcard_{template}.png").write_bytes(postcard)
    strong = engine.render_global(pair, 1.0)
    (directory / "strong_style_limit_case.png").write_bytes(png_bytes(strong.image_rgb))
    evidence = {
        "at": utc_now(),
        "runtime": engine.runtime,
        "pilot_outputs": records,
        "test_mask_is_synthetic": True,
        "real_m5_m6_integration": False,
        "constant_alpha_output_max_abs_error": float(
            (default.image_rgb - constant.image_rgb).abs().max()
        ),
        "constant_alpha_map_max_abs_error": float((constant.alpha_map - 0.6).abs().max()),
        "permutation_max_abs_error": float(
            (first.image_rgb - permuted_result.image_rgb).abs().max()
        ),
        "repeat_max_abs_error": float((first.image_rgb - repeated.image_rgb).abs().max()),
        "feature_cache_unchanged": True,
        "rtol": 1e-5,
        "atol": 1e-6,
        "notes": [
            "Only pilot images used. No formal 72-pair evaluation was run.",
            "Strong stylization and upscaling can distort details; no improvement claim.",
        ],
    }
    write_json(directory / "numerical_checks.json", evidence)
    failure_records = []

    def expect_failure(name, action):
        try:
            action()
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            failure_records.append(
                {
                    "case": name,
                    "status": "rejected_as_expected",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        else:
            raise AssertionError(f"Invalid input was accepted: {name}")

    expect_failure("not_an_image", lambda: load_rgb(b"this is not a PNG"))
    expect_failure("wrong_alpha_length", lambda: engine.render_regions(pair, prepared, (0.6,)))
    expect_failure(
        "negative_mask", lambda: engine.prepare_regions(pair, RegionPack((0, 1), labels, -masks))
    )
    expect_failure(
        "missing_font",
        lambda: render_postcard(
            default.image_rgb, "中文", "", "minimal", directory / "font_does_not_exist.ttf"
        ),
    )
    expect_failure(
        "body_overflow",
        lambda: render_postcard(default.image_rgb, "标题", "长正文" * 150, "minimal", font_path),
    )
    missing_config = dict(config, weight_manifest="resources/weights/missing_manifest.json")
    expect_failure("missing_weight_manifest", lambda: AdaINEngine(missing_config))
    if not torch.cuda.is_available():
        expect_failure(
            "explicit_cuda_unavailable", lambda: AdaINEngine(dict(config, device="cuda"))
        )
    write_json(directory / "failure_cases.json", {"at": utc_now(), "cases": failure_records})
    print(f"Pilot samples and actual numerical evidence saved to {directory}")


if __name__ == "__main__":
    main()
