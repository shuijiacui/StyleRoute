"""Regressions for the actual interface/algorithm failures found during team review."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from src.integration import run_search
from src.regions.graph import _rgb_to_lab, build_region_graph
from src.regions.masks import generate_soft_masks, validate_soft_masks
from src.regions.partition import _merge_regions
from src.search.baselines import run_global_scan, run_random_states, run_risk_guided
from src.search.objective import VGG19_RELU_LAYERS, ScorerConfig, VGG19Evaluator
from src.search.solver import Score, SearchConfig


def landscape(alphas):
    j = 0.1 + sum(abs(a - 0.3) for a in alphas)
    return Score(j, 1, 0, j, 0, True, (1.0,) * len(alphas)), tuple(alphas)


def test_rgb_ranges_are_equivalent_and_long_boundaries_do_not_saturate():
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:, :16] = [20, 80, 140]
    rgb[:, 16:] = [40, 120, 180]
    labels = np.zeros((32, 32), np.int32)
    labels[:, 16:] = 1
    np.testing.assert_allclose(_rgb_to_lab(rgb), _rgb_to_lab(rgb.astype(np.float32) / 255))
    weights = build_region_graph(labels, rgb)
    assert 0 < weights[0][2] < 1


def test_unsafe_boundary_stops_without_forcing_a_merge():
    labels = np.zeros((16, 16), np.int32)
    labels[:, 8:] = 1
    image = np.zeros((16, 16, 3), np.float32)
    image[:, :8] = [1, 0, 0]
    image[:, 8:] = [0, 0, 1]
    result, records = _merge_regions(labels, image, {"max_regions": 1, "threshold": 0.35}, {})
    np.testing.assert_array_equal(result, labels)
    assert not records


def test_every_merge_reduces_count_without_reviving_ids_or_disconnecting_regions():
    labels = np.tile(np.arange(20, dtype=np.int32), (20, 1))
    image = np.full((20, 20, 3), 0.5, np.float32)
    result, records = _merge_regions(
        labels, image, {"max_regions": 8, "min_regions": 8, "min_area": 0}, {}
    )
    assert len(np.unique(result)) == 8
    assert len(records) == 12
    assert all(record["merged_area"] > 0 for record in records)
    assert all(ndimage.label(result == rid)[1] == 1 for rid in np.unique(result))


def test_tiny_regions_keep_fractional_coverage_and_zero_masks_are_rejected():
    labels = np.zeros((64, 64), np.int32)
    labels[1, 1] = 1
    masks = generate_soft_masks(labels, np.arange(2), (8, 8))
    assert masks[1].sum() > 0
    validate_soft_masks(masks)
    broken = torch.zeros_like(masks)
    broken[0] = 1
    with pytest.raises(AssertionError, match="vanished"):
        validate_soft_masks(broken)


def test_global_result_image_alphas_and_metrics_describe_one_candidate():
    result = run_global_scan(evaluate=landscape, n_regions=2, config=SearchConfig(budget=3))
    assert result.status == "improved"
    assert result.best_image == result.best_alphas == (0.3, 0.3)
    assert result.best_score == landscape(result.best_alphas)[0]
    assert result.candidate_count == 3
    limited = run_global_scan(evaluate=landscape, n_regions=2, config=SearchConfig(budget=1))
    assert limited.used_default and limited.candidate_count == 1
    assert limited.best_score == landscape(limited.best_alphas)[0]


def test_random_archive_and_unique_evaluation_indices_are_truthful():
    result = run_random_states(evaluate=landscape, n_regions=2, config=SearchConfig(budget=9))
    best_records = [r for r in result.trajectory if r.is_new_best]
    assert best_records[-1].alphas == result.best_alphas
    local = run_risk_guided(evaluate=landscape, n_regions=2, config=SearchConfig(budget=9))
    fresh = [r.evaluation_index for r in local.trajectory if not r.cache_hit]
    assert fresh == list(range(1, local.candidate_count + 1))
    assert local.state_cache_hits > 0


def test_evaluator_uses_recorded_relu_layers_and_preserves_global_rng():
    state = torch.random.get_rng_state()
    image = torch.zeros(1, 3, 32, 32)
    evaluator = VGG19Evaluator(
        content_tensor=image,
        style_tensor=image,
        soft_masks=torch.ones(1, 4, 4),
        edges=(),
        config=ScorerConfig(weights="RANDOM"),
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert VGG19_RELU_LAYERS == (1, 6, 11, 20)
    assert all(isinstance(evaluator._features[i], torch.nn.ReLU) for i in VGG19_RELU_LAYERS)
    assert evaluator._style_grams[20].shape == (1, 512, 512)


@pytest.mark.weights
@pytest.mark.parametrize("method", ["risk", "plain", "random_states", "global_scan"])
def test_real_generator_regions_and_each_search_method_share_the_contract(
    real_engine, real_pair, method
):
    initial = real_engine.render_global(real_pair, 0.6)
    previous = real_engine.config
    real_engine.config = {
        **previous,
        "region_adapter": "src.regions.adapter:build_regions",
        "search_adapter": "src.search.adapter:solve",
        "search_config": {
            "method": method,
            "budget": 3,
            "max_restarts": 0,
            "scoring": {"weights": "RANDOM", "b_min": 0.2, "grad_smooth_kernel": 5},
        },
    }
    before = dict(real_engine.counters)
    features = real_pair.content_features.clone()
    try:
        regions, best, result = run_search(real_engine, real_pair, initial)
        assert result.candidate_count <= 3
        assert (
            result.decoder_calls
            == 1 + real_engine.counters["decoder_calls"] - before["decoder_calls"]
        )
        assert result.metadata["scorer"]["b_bounds"][0] == 0.2
        assert result.metadata["scorer"]["grad_smooth_kernel"] == 5
        regenerated = real_engine.render_regions(real_pair, regions, result.region_alphas)
        torch.testing.assert_close(best.image_rgb, regenerated.image_rgb, atol=2e-6, rtol=1e-5)
        assert torch.equal(features, real_pair.content_features)
        assert real_engine.counters["encoder_calls"] == before["encoder_calls"]
    finally:
        real_engine.config = previous
