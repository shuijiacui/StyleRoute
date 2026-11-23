import numpy as np
import torch
import pytest

from M5.regions.partition import build_regions
from M5.regions.region_types import RegionPack


class TestBuildRegions:
    def _make_dummy_image(self, H=64, W=64):
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[:H//2, :W//2] = [0.8, 0.2, 0.2]
        img[:H//2, W//2:] = [0.2, 0.8, 0.2]
        img[H//2:, :W//2] = [0.2, 0.2, 0.8]
        img[H//2:, W//2:] = [0.9, 0.9, 0.1]
        return img

    def test_returns_region_pack(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert isinstance(regions, RegionPack)

    def test_region_ids_continuous_zero_based(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected = np.arange(regions.n_regions, dtype=np.int32)
        assert np.array_equal(regions.region_ids, expected)

    def test_labels_shape(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert regions.labels.ndim == 2
        assert regions.labels.shape[0] == 64
        assert regions.labels.shape[1] == 64

    def test_soft_masks_shape(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert regions.soft_masks.ndim == 3
        assert regions.soft_masks.shape[0] == regions.n_regions
        assert regions.soft_masks.shape[1] == 16
        assert regions.soft_masks.shape[2] == 16

    def test_soft_masks_non_negative(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert (regions.soft_masks >= 0).all()

    def test_soft_masks_sum_to_one(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mask_sum = regions.soft_masks.sum(dim=0)
        ones = torch.ones_like(mask_sum)
        assert torch.allclose(mask_sum, ones, atol=1e-3)

    def test_edges_format(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for edge in regions.edges:
            assert len(edge) == 3
            i, j, w = edge
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert i < j
            assert isinstance(w, (int, float, np.floating, np.integer))
            assert not (np.isnan(float(w)) or np.isinf(float(w)))
            assert 0.0 <= float(w) <= 1.0

    def test_no_duplicate_edges(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        edge_set = set()
        for i, j, w in regions.edges:
            edge_set.add((i, j))
        assert len(edge_set) == len(regions.edges)

    def test_metadata_contains_required_keys(self):
        img = self._make_dummy_image()
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for key in ["method", "num_regions", "feature_hw"]:
            assert key in regions.metadata

    def test_uniform_image_single_region(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert regions.n_regions == 1
        assert regions.soft_masks.shape[0] == 1

    def test_deterministic_with_seed(self):
        img = self._make_dummy_image()
        r1 = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        r2 = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert r1.n_regions == r2.n_regions
        assert np.array_equal(r1.labels, r2.labels)
        assert torch.allclose(r1.soft_masks, r2.soft_masks)


class TestRegionPackValidation:
    def test_valid_pack_passes(self):
        region_ids = np.arange(4, dtype=np.int32)
        labels = np.zeros((16, 16), dtype=np.int32)
        soft_masks = torch.ones(4, 16, 16) / 4.0
        edges = [(0, 1, 0.5), (1, 2, 0.3), (2, 3, 0.8)]
        metadata = {"method": "test"}
        pack = RegionPack(
            region_ids=region_ids,
            labels=labels,
            soft_masks=soft_masks,
            edges=edges,
            metadata=metadata,
        )
        assert pack.n_regions == 4

    def test_non_continuous_ids_raises(self):
        region_ids = np.array([0, 2, 3, 4], dtype=np.int32)
        labels = np.zeros((16, 16), dtype=np.int32)
        soft_masks = torch.ones(4, 16, 16) / 4.0
        edges = []
        metadata = {"method": "test"}
        with pytest.raises(AssertionError):
            RegionPack(
                region_ids=region_ids,
                labels=labels,
                soft_masks=soft_masks,
                edges=edges,
                metadata=metadata,
            )

    def test_edge_i_not_less_than_j_raises(self):
        region_ids = np.arange(2, dtype=np.int32)
        labels = np.zeros((16, 16), dtype=np.int32)
        soft_masks = torch.ones(2, 16, 16) / 2.0
        edges = [(1, 0, 0.5)]
        metadata = {"method": "test"}
        with pytest.raises(AssertionError):
            RegionPack(
                region_ids=region_ids,
                labels=labels,
                soft_masks=soft_masks,
                edges=edges,
                metadata=metadata,
            )