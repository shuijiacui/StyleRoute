"""Contract/orchestration tests use explicit doubles, never production search evidence."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src import integration
from src.contracts import RegionPack, SearchResult
from src.models.fusion import prepare_regions


def contract_result(pair):
    return SearchResult(
        pair.content_tensor,
        (0.6, 0.6),
        "no_improvement",
        {"Q": 1.0, "T": 1.0, "S": 1.0, "J": 1.0},
        (),
        1,
        1,
        0,
        {"used_default": True, "test_only": True},
    )


@pytest.mark.parametrize("change", ["metric", "domain", "status", "count", "image", "fallback"])
def test_bad_m6_results_are_rejected(fake_pair, two_regions, change):
    prepared = prepare_regions(fake_pair, two_regions)
    result = contract_result(fake_pair)
    if change == "metric":
        result = replace(result, metrics={"Q": float("nan"), "T": 1, "S": 1, "J": 1})
    elif change == "domain":
        result = replace(result, region_alphas=(0.4, 0.6))
    elif change == "status":
        result = replace(result, status="amazing_success")
    elif change == "count":
        result = replace(result, candidate_count=0)
    elif change == "image":
        result = replace(result, best_image_rgb=torch.full_like(fake_pair.content_tensor, 1.1))
    else:
        result = replace(result, status="improved")
    with pytest.raises(ValueError):
        integration.validate_search_result(result, fake_pair, prepared, [0.3, 0.6, 0.9])


@pytest.mark.weights
def test_default_is_passed_to_solver_without_regeneration(monkeypatch, real_engine, real_pair):
    default = real_engine.render_global(real_pair, 0.6)
    before = dict(real_engine.counters)

    def fixture_regions(image, *, feature_hw, device):
        return RegionPack(
            (0,),
            np.zeros(real_pair.input_hw, dtype="int64"),
            torch.ones((1, *feature_hw), device=device),
            metadata={"test_only": True},
        )

    def fixture_solver(*, generate, pair, regions, initial_result, search_config):
        assert initial_result is default
        return SearchResult(
            default.image_rgb,
            (0.6,),
            "no_improvement",
            {"Q": 1, "T": 1, "S": 1, "J": 1},
            (),
            1,
            1,
            0,
            {"used_default": True, "test_only": True},
        )

    monkeypatch.setattr(
        integration,
        "load_adapter",
        lambda spec: fixture_regions if spec.startswith("src.regions.") else fixture_solver,
    )
    _, best, result = integration.run_search(real_engine, real_pair, default)
    assert result.status == "no_improvement" and best.image_rgb is default.image_rgb
    assert real_engine.counters == before


def test_missing_adapters_reported_without_fake_result():
    ready, reasons = integration.integration_status(
        {"region_adapter": "_missing_m5:build", "search_adapter": "_missing_m6:solve"}
    )
    assert not ready and len(reasons) == 2
