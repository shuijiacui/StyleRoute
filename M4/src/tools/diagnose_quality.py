"""Compare exact upstream inference and actual resolutions without altering the baseline."""

import importlib.util
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.adain import AdaINEngine
from src.models.preprocessing import prepare_style, tensor_to_pil
from src.project import load_config, project_path, utc_now, write_json


def main():
    output = project_path("outputs/runs/quality_diagnosis")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config()
    engine = AdaINEngine(base)
    content = project_path("resources/images/content/pilot/mountain/mountain_01.jpg")
    style = project_path("resources/images/styles/met_436535.jpg")
    pair = engine.prepare(content, style)
    # Run the retained, unmodified MIT-licensed inference definitions, on exactly
    # the same preprocessed tensors. Training code is not invoked.
    upstream_directory = project_path("resources/upstream/adain")
    sys.path.insert(0, str(upstream_directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "m4_upstream_reference", upstream_directory / "net.py"
        )
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
    finally:
        sys.path.remove(str(upstream_directory))
    upstream.vgg.load_state_dict(
        torch.load(project_path(base["encoder_checkpoint"]), map_location="cpu", weights_only=True),
        strict=True,
    )
    upstream.decoder.load_state_dict(
        torch.load(project_path(base["decoder_checkpoint"]), map_location="cpu", weights_only=True),
        strict=True,
    )
    encoder = torch.nn.Sequential(*list(upstream.vgg.children())[:31]).eval()
    decoder = upstream.decoder.eval()
    style_tensor = prepare_style(style).tensor
    with torch.inference_mode():
        reference_content = encoder(pair.content_tensor)
        reference_style = encoder(style_tensor)
        reference_adain = upstream.adain(reference_content, reference_style)
        reference = decoder(0.6 * reference_adain + 0.4 * reference_content).clamp(0, 1)
    actual = engine.render_global(pair, 0.6).image_rgb
    torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-6)
    report = {
        "created_at": utc_now(),
        "reference_scope": "Original upstream topology and AdaIN; identical preprocessed inputs",
        "upstream_max_abs_error": float((reference - actual).abs().max()),
        "baseline_config_unchanged": True,
        "cases": [],
    }

    variants = [(256, 256, 0.6), (512, 256, 0.6), (512, 512, 0.6), (512, 512, 0.8), (512, 512, 1.0)]
    font = ImageFont.truetype(str(project_path(base["font_path"])), 23)
    canvas = Image.new("RGB", (6 * 400, 2 * 470 + 75), "#fffdf7")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        "真实分辨率与强度对照（全部画面按相同显示尺寸呈现，无锐化）",
        font=font,
        fill="#283d33",
    )
    headings = ["原始内容构图"] + [f"C{c}/S{s} · alpha {a}" for c, s, a in variants]
    for column, heading in enumerate(headings):
        draw.text((10 + column * 400, 43), heading, font=font, fill="#283d33")
    engines = {(256, 256): engine}
    for row, content_id in enumerate(("mountain_01", "downtown_01")):
        category = "mountain" if row == 0 else "downtown"
        source = project_path(f"resources/images/content/pilot/{category}/{content_id}.jpg")
        for column, (content_size, style_size, alpha) in enumerate(variants, 1):
            key = content_size, style_size
            if key not in engines:
                config = dict(
                    base, content_hw=[content_size, content_size], style_long_side=style_size
                )
                engines[key] = AdaINEngine(config)
            active = engines[key]
            prepared = active.prepare(source, style)
            image = tensor_to_pil(active.render_global(prepared, alpha).image_rgb)
            filename = f"{content_id}_c{content_size}_s{style_size}_a{alpha}.png"
            image.save(output / filename)
            canvas.paste(
                image.resize((392, 392), Image.Resampling.LANCZOS),
                (4 + column * 400, 82 + row * 470),
            )
            if column == 2:
                original = prepared.content_rgb
                canvas.paste(
                    original.resize((392, 392), Image.Resampling.LANCZOS), (4, 82 + row * 470)
                )
                original.save(output / f"{content_id}_content_512.png")
            draw.text(
                (8 + column * 400, 480 + row * 470),
                f"{content_size}×{content_size} 实际生成",
                font=font,
                fill="#283d33",
            )
            report["cases"].append(
                {
                    "content_id": content_id,
                    "style_id": "met_436535",
                    "content_size": content_size,
                    "style_long_side": style_size,
                    "alpha": alpha,
                    "output": filename,
                    "config_hash": active.config_hash,
                }
            )
    canvas.save(output / "resolution_strength_comparison.jpg", quality=94)
    # Alpha=0 still travels through the encoder/decoder; isolate reconstruction loss.
    reconstruction = tensor_to_pil(engine.render_global(pair, 0.0).image_rgb)
    reconstruction.save(output / "mountain_01_reconstruction_256.png")
    write_json(output / "diagnosis.json", report)
    print(f"Upstream parity max error: {report['upstream_max_abs_error']}; saved {output}")


if __name__ == "__main__":
    main()
