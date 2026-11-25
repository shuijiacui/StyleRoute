import numpy as np
import pytest
import torch
from skimage.color import lab2rgb

from src.models.compensation import ContentGeometry, LocalCompensator, image_lab
from src.search.cma_search import optimize
from src.search.landscape import UniformityMetric


def tensor(lab):
    return torch.from_numpy(lab2rgb(lab).astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)


def smooth_scene():
    lab = np.zeros((96, 96, 3), np.float32)
    lab[..., 0] = 60 + np.linspace(0, 4, 96)[None, :]
    lab[..., 1:] = (3, -8)
    image = tensor(lab)
    return lab, image, ContentGeometry(image, np.zeros((96, 96), np.int32))


def test_uniformity_detects_lightness_and_chroma_blocks_without_penalising_offset():
    lab, original, geometry = smooth_scene()
    metric = UniformityMetric(geometry)
    baseline = metric(original)
    offset = lab.copy()
    offset[..., 0] += 5
    assert metric(tensor(offset))["U"] < 0.001
    for channel, key in ((0, "U_L"), (1, "U_ab")):
        damaged = lab.copy()
        damaged[24:72, 24:72, channel] += 10
        assert metric(tensor(damaged))[key] > baseline[key] + 0.2
    # A candidate never changes the cached measurement support.
    np.testing.assert_array_equal(geometry.lab, image_lab(original))


def test_compensation_identity_boundaries_direction_and_intensity():
    lab, original, geometry = smooth_scene()
    reconstructed = lab.copy()
    reconstructed[20:76, 20:48, 0] -= 8
    reconstructed[20:76, 48:76, 0] += 8
    candidate = tensor(reconstructed)
    compensate = LocalCompensator(geometry, candidate)
    assert compensate(candidate, 0) is candidate
    half, full = compensate(candidate, 0.5), compensate(candidate, 1)
    metric = UniformityMetric(geometry)
    assert metric(full)["U"] < metric(half)["U"] < metric(candidate)["U"]
    assert torch.equal(full[:, :, :4], candidate[:, :, :4])
    assert (image_lab(full)[..., 0] - image_lab(candidate)[..., 0]).max() <= 12.001
    # Opposite decoder residual must not be reinforced.
    opposite = lab.copy()
    opposite[..., 0] = 2 * lab[..., 0] - reconstructed[..., 0]
    opposite = tensor(opposite)
    assert torch.max(torch.abs(compensate(opposite, 1) - opposite)) < 1e-4
    assert torch.isfinite(full).all() and full.min() >= 0 and full.max() <= 1
    with pytest.raises(ValueError):
        compensate(original, float("nan"))


def evaluator(theta):
    target = np.array([0.3, 0.8, 0.65])
    error = np.asarray(theta) - target
    # Coupled quadratic rewards simultaneous changes in the full parameter vector.
    objective = float(error @ np.array([[2, 0.4, 0.2], [0.4, 1, 0.1], [0.2, 0.1, 1]]) @ error)
    violation = max(0.0, 0.2 - min(theta[:2]))
    return {"J": objective, "feasible": violation == 0, "violation": violation}, theta


def test_cma_joint_search_reproducible_and_budget_counts_initial_and_warmup():
    first = optimize(evaluator, 2, budget=80, seed=0)
    second = optimize(evaluator, 2, budget=160, seed=0)
    assert first["trajectory"] == second["trajectory"][:80]
    assert len(first["trajectory"]) == 80
    assert [r["selection_reason"] for r in first["trajectory"][:5]] == ["initial"] + ["warmup"] * 4
    assert first["best"][1]["J"] < first["trajectory"][0]["J"] * 0.2
    assert first["best"][1]["feasible"]
    for row in first["trajectory"][5:]:
        assert all(0 <= x <= 1 for x in row["alphas"] + [row["beta"]])
        assert all(abs(x - 0.6) > 1e-9 for x in row["alphas"])


@pytest.mark.parametrize("budget", [1, 2, 4, 6, 17])
def test_cma_small_budgets_and_exact_fallback(budget):
    baseline = object()

    def score(theta):
        return {"J": 1.0, "feasible": True, "violation": 0.0}, baseline

    result = optimize(score, 2, budget=budget)
    assert len(result["trajectory"]) == budget
    assert result["best"][2] is baseline and result["used_default"]
    assert result["best"][0] == (0.6, 0.6, 0)


def test_cma_rejects_infeasible_and_numerically_tiny_gains():
    def score(theta):
        baseline = theta == (0.6, 0.6, 0)
        feasible = baseline or theta[-1] == 0.5
        return {
            "J": 1 if baseline else 0.9999 if feasible else -100,
            "feasible": feasible,
            "violation": 0 if feasible else 0.001,
        }, theta

    result = optimize(score, 2, budget=40)
    assert result["used_default"] and result["best"][0] == (0.6, 0.6, 0)


def test_cma_nonfinite_scores_fail_loudly():
    with pytest.raises(ValueError, match="Nonfinite"):
        optimize(lambda _: ({"J": float("nan")}, None), 2)
