from dataclasses import replace

import pytest
import torch

from src.models.fusion import fuse_region_features, prepare_regions, validate_alphas


def test_constant_strength_matches_global_features(fake_pair, two_regions):
    prepared = prepare_regions(fake_pair, two_regions)
    mixed, alpha = fuse_region_features(
        fake_pair.content_features,
        fake_pair.adain_features,
        prepared.soft_masks,
        torch.tensor([0.6, 0.6]),
    )
    torch.testing.assert_close(alpha, torch.full_like(alpha, 0.6))
    torch.testing.assert_close(
        mixed, 0.6 * fake_pair.adain_features + 0.4 * fake_pair.content_features
    )


def test_synchronized_permutation_preserves_result(fake_pair, two_regions):
    features = (fake_pair.content_features, fake_pair.adain_features)
    first, _ = fuse_region_features(*features, two_regions.soft_masks, torch.tensor([0.3, 0.9]))
    second, _ = fuse_region_features(
        *features, two_regions.soft_masks.flip(0), torch.tensor([0.9, 0.3])
    )
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "kind", ["shape", "nan", "negative", "uncovered", "sum", "empty_region", "dtype"]
)
def test_invalid_masks_are_rejected(fake_pair, two_regions, kind):
    masks = two_regions.soft_masks.clone()
    if kind == "shape":
        masks = masks[:, :3]
    elif kind == "nan":
        masks[0, 0, 0] = float("nan")
    elif kind == "negative":
        masks[0, 0, 0] = -1
    elif kind == "uncovered":
        masks[:, 0, 0] = 0
    elif kind == "sum":
        masks *= 0.5
    elif kind == "empty_region":
        masks[0], masks[1] = 1, 0
    else:
        masks = masks.double()
    with pytest.raises(ValueError):
        prepare_regions(fake_pair, replace(two_regions, soft_masks=masks))


@pytest.mark.parametrize("values", [(0.3,), (0.3, float("inf")), (-0.1, 0.6), (0.6, 1.1)])
def test_invalid_alphas_are_rejected(values):
    with pytest.raises(ValueError):
        validate_alphas(values, 2, torch.device("cpu"))


def test_validated_mask_snapshot_is_isolated(fake_pair, two_regions):
    prepared = prepare_regions(fake_pair, two_regions)
    before = prepared.soft_masks.clone()
    two_regions.soft_masks.zero_()
    torch.testing.assert_close(prepared.soft_masks, before)


def test_bad_region_ids_and_labels(fake_pair, two_regions):
    with pytest.raises(ValueError):
        prepare_regions(fake_pair, replace(two_regions, region_ids=(1, 2)))
    with pytest.raises(ValueError):
        prepare_regions(fake_pair, replace(two_regions, labels=two_regions.labels.astype(float)))
