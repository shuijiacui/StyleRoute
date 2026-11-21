"""Tests for the contour PRF metric (fixed extractor + tolerance matching)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from M6.search.metrics import (  # noqa: E402
    MetricsConfig,
    contour_f1,
    contour_prf,
    edge_map,
)


def _blank(hw=64):
    return torch.zeros(hw, hw, dtype=torch.bool)


def test_identical_edge_maps_agree_perfectly():
    pred = _blank()
    pred[10:20, 30] = True
    result = contour_prf(pred, pred.clone())
    assert result["precision"] == 1.0 and result["recall"] == 1.0 and result["f1"] == 1.0


def test_far_apart_edges_score_zero():
    pred, target = _blank(), _blank()
    pred[5, 0:40] = True          # >= min_edge_pixels so the degenerate
    target[60, 0:40] = True       # "no edges" rule does not apply
    result = contour_prf(pred, target, MetricsConfig(tolerance_px=2.0))
    assert result["f1"] == 0.0


def test_tolerance_matches_slightly_shifted_edges():
    pred, target = _blank(), _blank()
    for y in range(10, 50):
        pred[y, 20] = True
        target[y, 22] = True  # 2px shift, within tolerance
    strict = contour_prf(pred, target, MetricsConfig(tolerance_px=1.0))
    loose = contour_prf(pred, target, MetricsConfig(tolerance_px=2.0))
    assert loose["f1"] == 1.0
    assert strict["f1"] < loose["f1"]


def test_no_edge_degenerate_rules():
    blank = _blank()
    lined = _blank()
    lined[10, 10:40] = True
    both = contour_prf(blank, _blank())
    assert both["both_no_edge"] and both["f1"] == 1.0
    one = contour_prf(blank, lined)
    assert not one["both_no_edge"] and one["f1"] == 0.0


def test_edge_map_is_boolean_and_nonempty_on_realistic_image():
    g = torch.Generator().manual_seed(0)
    image = torch.rand(1, 3, 64, 64, generator=g)
    edges = edge_map(image, MetricsConfig(edge_quantile=0.90))
    assert edges.dtype == torch.bool
    assert int(edges.sum()) >= 8


def test_contour_f1_identical_images():
    g = torch.Generator().manual_seed(1)
    image = torch.rand(1, 3, 64, 64, generator=g)
    result = contour_f1(image, image.clone())
    assert result["f1"] == 1.0
