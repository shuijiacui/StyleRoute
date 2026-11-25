"""Actual ImageNet evaluation is separate from random-weight wiring tests."""

import json

import pytest
import torch

from src.integration import run_search
from src.models.preprocessing import image_tensor
from src.project import project_path
from src.regions.adapter import build_regions
from src.search.objective import ScorerConfig, VGG19Evaluator
from src.tools.prepare_evaluator import EXPECTED_SHA256

pytestmark = [
    pytest.mark.evaluator,
    pytest.mark.skipif(
        not project_path("resources/weights/vgg19_features.pth").is_file(),
        reason="Run python -m src.tools.prepare_evaluator; not a passed quality check",
    ),
]


def test_verified_pretrained_evaluator_calibrates_real_default(real_engine, real_pair):
    manifest = json.loads(
        project_path("resources/weights/vgg19_features.json").read_text(encoding="utf-8")
    )
    assert manifest["source_sha256"] == EXPECTED_SHA256
    regions = build_regions(
        real_pair.content_rgb, feature_hw=real_pair.feature_hw, device=real_engine.device
    )
    style = image_tensor(real_pair.style_rgb)
    evaluator = VGG19Evaluator(
        content_tensor=real_pair.content_tensor,
        style_tensor=style,
        soft_masks=regions.soft_masks,
        edges=regions.edges,
        config=ScorerConfig(),
    )
    assert evaluator.style_loss(style) < 1e-10
    default = real_engine.render_global(real_pair, 0.6)
    evaluator.calibrate(default.image_rgb)
    score = evaluator.compute(default.image_rgb, (0.6,) * len(regions.region_ids))
    assert score.feasible
    assert score.Q >= 0 and score.T >= 0


def test_shared_vgg_forward_matches_separate_losses(monkeypatch, real_engine, real_pair):
    regions = build_regions(
        real_pair.content_rgb, feature_hw=real_pair.feature_hw, device=real_engine.device
    )
    evaluator = VGG19Evaluator(
        content_tensor=real_pair.content_tensor,
        style_tensor=image_tensor(real_pair.style_rgb),
        soft_masks=regions.soft_masks,
        edges=regions.edges,
        config=ScorerConfig(),
    )
    image = real_engine.render_global(real_pair, 0.6).image_rgb
    q, _ = evaluator.structural_loss(image)
    t = evaluator.style_loss(image)
    evaluator.calibrate(image)
    original = evaluator._forward
    calls = []

    def forward(image):
        calls.append(1)
        return original(image)

    monkeypatch.setattr(evaluator, "_forward", forward)
    score = evaluator.compute(image, (0.6,) * len(regions.region_ids))
    assert len(calls) == 1
    assert score.Q == q and score.T == t
    from torchvision.transforms.functional import gaussian_blur

    clean_q, _ = evaluator.structural_loss(real_pair.content_tensor)
    blurred_q, _ = evaluator.structural_loss(gaussian_blur(real_pair.content_tensor, 15, 3))
    assert clean_q < 1e-10 and blurred_q > clean_q + 0.01


def test_continuous_compensation_replay_and_accounting(real_engine, real_pair):
    from src.models.compensation import ContentGeometry, LocalCompensator

    if not project_path("resources/weights/NIMA_VGG16_ava-dc4e8265.pth").exists():
        pytest.skip("NIMA AVA weights required for continuous search integration")
    previous = real_engine.config
    real_engine.config = {
        **previous,
        "search_config": {**previous["search_config"], "method": "cma_compensated", "budget": 8},
    }
    try:
        initial = real_engine.render_global(real_pair, 0.6)
        before = dict(real_engine.counters)
        regions, best, result = run_search(real_engine, real_pair, initial)
        assert result.candidate_count == 8
        assert (
            result.decoder_calls
            == 1 + real_engine.counters["decoder_calls"] - before["decoder_calls"]
        )
        assert real_engine.counters["encoder_calls"] == before["encoder_calls"]
        assert all(0 <= a <= 1 for a in result.region_alphas)
        assert result.metrics["A"] is not None and result.metrics["U"] >= 0
        assert result.metrics["T"] <= 1.05 * result.metadata["t0"] + 1e-9
        if result.metadata["used_default"]:
            assert best.image_rgb is initial.image_rgb
        else:
            raw = (
                initial.image_rgb
                if result.region_alphas == (0.6,) * len(regions.region_ids)
                else (
                    real_engine.render_regions(real_pair, regions, result.region_alphas).image_rgb
                )
            )
            correction = LocalCompensator(
                ContentGeometry(real_pair.content_tensor, regions.labels),
                real_engine.render_global(real_pair, 0).image_rgb,
            )
            replay = correction(raw, result.metadata["beta"])
            torch.testing.assert_close(best.image_rgb, replay, rtol=0, atol=0)
    finally:
        real_engine.config = previous


@pytest.mark.parametrize("method", ["risk", "plain", "random_states", "global_scan"])
def test_pretrained_complete_search_methods(real_engine, real_pair, method):
    previous = real_engine.config
    real_engine.config = {
        **previous,
        "search_config": {
            "method": method,
            "budget": 4,
            "max_restarts": 0,
            "scoring": {"weights": "IMAGENET1K_V1"},
        },
    }
    try:
        initial = real_engine.render_global(real_pair, 0.6)
        regions, best, result = run_search(real_engine, real_pair, initial)
        assert result.metadata["scorer"]["weights"] == "IMAGENET1K_V1"
        assert result.candidate_count <= 4
        if result.metadata["used_default"]:
            assert torch.equal(best.image_rgb, initial.image_rgb)
        else:
            rerendered = real_engine.render_regions(real_pair, regions, result.region_alphas)
            torch.testing.assert_close(best.image_rgb, rerendered.image_rgb)
    finally:
        real_engine.config = previous
