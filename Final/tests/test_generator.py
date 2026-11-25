from dataclasses import replace

import numpy as np
import pytest
import torch

from src.contracts import RegionPack
from src.models.adain import AdaINEngine
from src.models.preprocessing import prepare_style
from src.project import load_config

pytestmark = pytest.mark.weights


def test_prepared_pair_preserves_evaluation_style(real_engine, real_pair):
    expected = prepare_style(
        "resources/images/styles/met_436535.jpg",
        real_engine.config["style_long_side"],
        real_engine.config["minimum_style_short_side"],
    )
    assert real_pair.style_rgb.mode == "RGB"
    assert real_pair.style_rgb.size == expected.rgb.size
    assert real_pair.style_rgb.tobytes() == expected.rgb.tobytes()
    assert real_pair.metadata["style_hw"] == [expected.rgb.height, expected.rgb.width]


def test_real_endpoint_outputs_and_zero_is_reconstruction(real_engine, real_pair):
    for alpha in (0, 0.6, 1):
        image = real_engine.render_global(real_pair, alpha).image_rgb
        assert image.shape == (1, 3, *real_pair.input_hw)
        assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        assert not image.requires_grad
    with torch.inference_mode():
        reconstruction = real_engine.decoder(real_pair.content_features).clamp(0, 1)
    torch.testing.assert_close(real_engine.render_global(real_pair, 0).image_rgb, reconstruction)


def test_real_regional_equivalence_cache_and_candidate_isolation(real_engine, real_pair):
    height, width = real_pair.input_hw
    fh, fw = real_pair.feature_hw
    labels = np.zeros((height, width), dtype="int64")
    labels[:, width // 2 :] = 1
    masks = torch.zeros((2, fh, fw), device=real_engine.device)
    masks[0, :, : fw // 2], masks[1, :, fw // 2 :] = 1, 1
    raw = RegionPack((0, 1), labels, masks, metadata={"test_only": True})
    prepared = real_engine.prepare_regions(real_pair, raw)
    content_before, adain_before = (
        real_pair.content_features.clone(),
        real_pair.adain_features.clone(),
    )
    encoding_calls = real_engine.counters["encoder_calls"]
    global_image = real_engine.render_global(real_pair, 0.6).image_rgb
    regions_image = real_engine.render_regions(real_pair, prepared, (0.6, 0.6)).image_rgb
    torch.testing.assert_close(global_image, regions_image, rtol=1e-5, atol=2e-6)
    first = real_engine.render_regions(real_pair, prepared, (0.3, 0.9)).image_rgb
    real_engine.render_regions(real_pair, prepared, (0.9, 0.3))
    again = real_engine.render_regions(real_pair, prepared, (0.3, 0.9)).image_rgb
    torch.testing.assert_close(first, again)
    torch.testing.assert_close(real_pair.content_features, content_before)
    torch.testing.assert_close(real_pair.adain_features, adain_before)
    assert real_engine.counters["encoder_calls"] == encoding_calls


def test_fresh_pair_equals_cached_path(real_engine, real_pair):
    fresh = real_engine.prepare(real_pair.content_rgb, "resources/images/styles/met_436535.jpg")
    torch.testing.assert_close(
        real_engine.render_global(real_pair, 0.6).image_rgb,
        real_engine.render_global(fresh, 0.6).image_rgb,
        rtol=1e-5,
        atol=1e-6,
    )


def test_foreign_pair_rejected(real_engine, real_pair):
    with pytest.raises(ValueError, match="another model"):
        real_engine.render_global(
            replace(real_pair, metadata={**real_pair.metadata, "model_hash": "wrong"})
        )


def test_explicit_cuda_never_silently_falls_back():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, so the unavailable-device case does not apply")
    config = load_config()
    config["device"] = "cuda"
    with pytest.raises(RuntimeError, match="CUDA"):
        AdaINEngine(config)
