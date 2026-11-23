import numpy as np
import torch
import pytest

from M5.regions.masks import (
    generate_soft_masks,
    generate_hard_masks,
    validate_soft_masks,
    compute_effective_areas,
)


class TestGenerateSoftMasks:
    def _make_labels(self, H=32, W=32, N=4):
        labels = np.zeros((H, W), dtype=np.int32)
        labels[:H//2, :W//2] = 0
        labels[:H//2, W//2:] = 1
        labels[H//2:, :W//2] = 2
        labels[H//2:, W//2:] = 3
        region_ids = np.arange(N, dtype=np.int32)
        return labels, region_ids

    def test_output_shape(self):
        labels, region_ids = self._make_labels()
        masks = generate_soft_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            sigma=4.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert masks.shape == (4, 16, 16)

    def test_non_negative(self):
        labels, region_ids = self._make_labels()
        masks = generate_soft_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            sigma=4.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert (masks >= 0).all()

    def test_sum_approximately_one(self):
        labels, region_ids = self._make_labels()
        masks = generate_soft_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            sigma=4.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mask_sum = masks.sum(dim=0)
        ones = torch.ones_like(mask_sum)
        assert torch.allclose(mask_sum, ones, atol=1e-3)

    def test_validate_soft_masks_passes(self):
        labels, region_ids = self._make_labels()
        masks = generate_soft_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            sigma=4.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        result = validate_soft_masks(masks, atol=1e-3)
        assert result is True

    def test_single_region(self):
        H, W = 32, 32
        labels = np.zeros((H, W), dtype=np.int32)
        region_ids = np.array([0], dtype=np.int32)
        masks = generate_soft_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            sigma=4.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert masks.shape == (1, 16, 16)
        assert torch.allclose(masks.sum(dim=0), torch.ones(16, 16), atol=1e-3)

    def test_different_feature_sizes(self):
        labels, region_ids = self._make_labels()
        for hw in [(8, 8), (32, 32), (7, 11)]:
            masks = generate_soft_masks(
                region_labels=labels,
                region_ids=region_ids,
                feature_hw=hw,
                sigma=4.0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            assert masks.shape == (4, hw[0], hw[1])

    def test_compute_effective_areas(self):
        labels, region_ids = self._make_labels()
        masks = generate_soft_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            sigma=4.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        areas = compute_effective_areas(masks, threshold=0.01)
        assert len(areas) == 4
        for rid, area in areas.items():
            assert area > 0


class TestGenerateHardMasks:
    def test_output_shape(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[:16, :16] = 0
        labels[:16, 16:] = 1
        labels[16:, :16] = 2
        labels[16:, 16:] = 3
        region_ids = np.arange(4, dtype=np.int32)
        masks = generate_hard_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert masks.shape == (4, 16, 16)

    def test_one_hot_property(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[:16, :16] = 0
        labels[:16, 16:] = 1
        labels[16:, :16] = 2
        labels[16:, 16:] = 3
        region_ids = np.arange(4, dtype=np.int32)
        masks = generate_hard_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mask_sum = masks.sum(dim=0)
        assert torch.allclose(mask_sum, torch.ones(16, 16))

    def test_binary_values(self):
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[:16, :16] = 0
        labels[:16, 16:] = 1
        labels[16:, :16] = 2
        labels[16:, 16:] = 3
        region_ids = np.arange(4, dtype=np.int32)
        masks = generate_hard_masks(
            region_labels=labels,
            region_ids=region_ids,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        unique_vals = torch.unique(masks)
        assert len(unique_vals) == 2
        assert 0.0 in unique_vals
        assert 1.0 in unique_vals


class TestValidateSoftMasks:
    def test_valid_masks(self):
        masks = torch.rand(4, 16, 16)
        masks = masks / masks.sum(dim=0, keepdim=True)
        assert validate_soft_masks(masks) is True

    def test_negative_raises(self):
        masks = torch.rand(4, 16, 16)
        masks[0, 0, 0] = -0.1
        with pytest.raises(AssertionError):
            validate_soft_masks(masks)

    def test_nan_raises(self):
        masks = torch.rand(4, 16, 16)
        masks[0, 0, 0] = float("nan")
        with pytest.raises(AssertionError):
            validate_soft_masks(masks)

    def test_inf_raises(self):
        masks = torch.rand(4, 16, 16)
        masks[0, 0, 0] = float("inf")
        with pytest.raises(AssertionError):
            validate_soft_masks(masks)

    def test_wrong_ndim_raises(self):
        masks = torch.rand(16, 16)
        with pytest.raises(AssertionError):
            validate_soft_masks(masks)

    def test_zero_regions_raises(self):
        masks = torch.rand(0, 16, 16)
        with pytest.raises(AssertionError):
            validate_soft_masks(masks)