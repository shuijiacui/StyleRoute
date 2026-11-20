"""Real-integration tests (torch required): grid RegionPack stub, evaluator
math invariants, and the adapter -> contracts.SearchResult path validated by
M4's own ``validate_search_result``.

Weights are RANDOM (not ImageNet) on purpose: these tests prove wiring and
math, not generation quality. ImageNet-weight checks belong to Phase B/E and
need the recorded weight download.

Run from the repo root:  python -m pytest M6/tests/test_integration.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "M4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image", reason="PIL required")

from PIL import Image  # noqa: E402

from src.contracts import GenerationResult, PreparedPair, PreparedRegions  # noqa: E402
from src.integration import validate_search_result  # noqa: E402
from src.models.preprocessing import image_tensor  # noqa: E402

from M6.regions_grid import build_regions  # noqa: E402
from M6.search import adapter, objective  # noqa: E402


def _pair(content_seed=0, hw=64, style_hw=(80, 56)):
    """A minimal PreparedPair-like object with real tensor types."""
    g = torch.Generator().manual_seed(content_seed)
    content_t = torch.rand(1, 3, hw, hw, generator=g)
    content_rgb = Image.fromarray(
        (content_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    style_t = torch.rand(1, 3, style_hw[0], style_hw[1], generator=g)
    style_rgb = Image.fromarray(
        (style_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    return PreparedPair(
        pair_id="test_pair",
        content_rgb=content_rgb,
        style_rgb=style_rgb,
        content_tensor=content_t,
        content_features=torch.zeros(1, 512, hw // 8, hw // 8),
        adain_features=torch.zeros(1, 512, hw // 8, hw // 8),
        feature_hw=(hw // 8, hw // 8),
        input_hw=(hw, hw),
        cache_key="test_pair_key",
        metadata={},
    )


def _regions_for(pair, grid=(2, 2)):
    return build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device="cpu", grid=grid)


# ---------------------------------------------------------------------------
# Grid stub contract (M4/M5 interface doc)
# ---------------------------------------------------------------------------

def test_grid_stub_satisfies_region_pack_contract():
    pair = _pair()
    regions = _regions_for(pair, grid=(4, 4))
    assert regions.region_ids == tuple(range(16))
    assert regions.labels.shape == (64, 64)
    assert set(np.unique(regions.labels)) == set(range(16))
    masks = regions.soft_masks
    assert masks.shape == (16, *pair.feature_hw)
    assert masks.dtype == torch.float32
    assert (masks >= 0).all()
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert len(regions.edges) == 24  # 4-neighbourhood of a 4x4 grid
    for i, j, w in regions.edges:
        assert 0 <= i < 16 and 0 <= j < 16 and i != j and w > 0
    assert regions.metadata["stub"] is True


# ---------------------------------------------------------------------------
# Evaluator math invariants (README 3/4)
# ---------------------------------------------------------------------------

def _evaluator(pair, regions, cfg=None):
    return objective.VGG19Evaluator(
        content_tensor=pair.content_tensor,
        style_tensor=image_tensor(pair.style_rgb),
        soft_masks=regions.soft_masks,
        edges=regions.edges,
        config=cfg,
        device="cpu",
    )


def test_q_weights_bounded_and_default_is_feasible():
    pair = _pair()
    regions = _regions_for(pair)
    ev = _evaluator(pair, regions)
    assert float(ev._b.min()) >= ev.config.b_min - 1e-6
    assert float(ev._b.max()) <= ev.config.b_max + 1e-6
    default_image = torch.rand_like(pair.content_tensor)
    Q, q_map = ev.structural_loss(default_image)
    assert math.isfinite(Q) and Q >= 0
    T = ev.style_loss(default_image)
    assert math.isfinite(T) and T >= 0


def test_style_loss_of_reference_is_zero():
    # Gram MSE of the style image against itself must vanish (frozen
    # normalisation consistency check).
    pair = _pair()
    regions = _regions_for(pair)
    ev = _evaluator(pair, regions)
    from src.models.preprocessing import image_tensor

    style = image_tensor(pair.style_rgb)
    assert ev.style_loss(style) < 1e-10


def test_risk_apportionment_sums_match_q_and_s():
    pair = _pair()
    regions = _regions_for(pair)
    cfg = objective.ScorerConfig(weights="RANDOM", lam=2.0, epsilon=0.05)
    ev = _evaluator(pair, regions, cfg)
    default_image = torch.rand_like(pair.content_tensor)
    q0, t0 = ev.calibrate(default_image)
    alphas = (0.3, 0.6, 0.9, 0.3)  # 2x2 grid
    image = torch.rand_like(pair.content_tensor)
    score = ev.compute(image, alphas)
    # recompute the pieces to verify the sums
    Q, q_map = ev.structural_loss(image)
    q_i = objective.apportion(q_map, ev._masks)
    S = objective.coordination(alphas, regions.edges)
    s_i = objective.coordination_contrib(alphas, regions.edges)
    assert math.isclose(sum(q_i), Q, rel_tol=1e-5)
    assert math.isclose(sum(s_i), S, rel_tol=1e-12)
    assert len(score.risks) == 4
    expected = [q / (q0 + cfg.delta) + cfg.lam * si for q, si in zip(q_i, s_i)]
    for got, want in zip(score.risks, expected):
        assert math.isclose(got, want, rel_tol=1e-9)


def test_score_rejects_wrong_alpha_count():
    pair = _pair()
    regions = _regions_for(pair)
    ev = _evaluator(pair, regions)
    with pytest.raises(Exception):
        ev.compute(torch.rand_like(pair.content_tensor), (0.6, 0.6))


# ---------------------------------------------------------------------------
# Adapter end-to-end through M4's validator
# ---------------------------------------------------------------------------

def _run_adapter(method, budget=8, seed=0):
    pair = _pair()
    regions = _regions_for(pair)
    prepared = PreparedRegions(
        pair_cache_key=pair.cache_key,
        region_ids=regions.region_ids,
        soft_masks=regions.soft_masks,
        cache_key="regions_key",
    )
    default_image = torch.rand_like(pair.content_tensor)
    initial = GenerationResult(
        image_rgb=default_image,
        alpha_map=torch.full((1, 1, *pair.feature_hw), 0.6),
        metadata={"pair_id": pair.pair_id, "kind": "default"},
    )
    decode_calls = {"n": 0}

    def generate(alphas):
        decode_calls["n"] += 1
        perturb = 0.02 * (torch.rand_like(default_image) - 0.5)
        scale = 1.0 + 0.1 * sum(abs(a - 0.6) for a in alphas)
        return GenerationResult(
            image_rgb=(default_image + scale * perturb).clamp(0, 1),
            alpha_map=torch.full((1, 1, *pair.feature_hw), 0.6),
            metadata={"alphas": list(alphas)},
        )

    result = adapter.solve(
        generate=generate,
        pair=pair,
        regions=regions,
        initial_result=initial,
        search_config={
            "method": method,
            "budget": budget,
            "seed": seed,
            "weights": "RANDOM",  # logic test only, NOT a quality result
        },
    )
    validate_search_result(result, pair, prepared, allowed=(0.3, 0.6, 0.9))
    return result, decode_calls["n"]


def test_adapter_risk_method_passes_m4_validation():
    result, _ = _run_adapter("risk")
    assert result.status in {"improved", "no_improvement"}
    assert result.candidate_count <= 8
    assert set(result.metrics) >= {"Q", "T", "S", "J"}
    assert result.metadata["scorer"]["weights"] == "RANDOM"
    assert result.metadata["decoder_count_includes_initial"] is True


def test_adapter_all_methods_pass_m4_validation():
    for method in ("risk", "plain", "random_states", "global_scan"):
        result, decode_calls = _run_adapter(method, budget=10, seed=1)
        assert result.candidate_count >= 1
        if result.metadata["used_default"]:
            assert result.status == "no_improvement"
            assert all(abs(a - 0.6) < 1e-6 for a in result.region_alphas)


def test_default_result_is_not_regenerated():
    pair = _pair()
    regions = _regions_for(pair)
    default_image = torch.rand_like(pair.content_tensor)
    initial = GenerationResult(
        image_rgb=default_image,
        alpha_map=torch.full((1, 1, *pair.feature_hw), 0.6),
        metadata={},
    )
    calls = {"n": 0}

    def generate(alphas):
        calls["n"] += 1
        return GenerationResult(
            image_rgb=default_image.clone(),
            alpha_map=torch.full((1, 1, *pair.feature_hw), 0.6),
            metadata={},
        )

    result = adapter.solve(
        generate=generate,
        pair=pair,
        regions=regions,
        initial_result=initial,
        search_config={"method": "risk", "budget": 5, "seed": 0, "weights": "RANDOM"},
    )
    # the default state (evaluation #1) must reuse the provided image
    assert calls["n"] == result.decoder_calls - 1
