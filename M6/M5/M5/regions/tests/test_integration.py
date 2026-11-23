import numpy as np
import torch
import pytest

from M5.regions.partition import build_regions
from M5.regions.baselines import uniform_grid_partition, global_single_region
from M5.regions.region_types import RegionPack


class TestIntegration:
    def test_partition_returns_valid_regionpack(self):
        """build_regions should return a valid RegionPack."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert isinstance(regions, RegionPack)
        assert regions.n_regions >= 1

    def test_baseline_grid_returns_valid_regionpack(self):
        """uniform_grid_partition should return a valid RegionPack."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        regions = uniform_grid_partition(
            img,
            target_regions=12,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
        )
        assert isinstance(regions, RegionPack)
        assert regions.n_regions >= 1

    def test_baseline_global_returns_valid_regionpack(self):
        """global_single_region should return a valid RegionPack."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        regions = global_single_region(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert isinstance(regions, RegionPack)
        assert regions.n_regions == 1

    def test_region_ids_continuous(self):
        """region_ids must be [0, 1, ..., N-1]."""
        img = np.random.rand(128, 128, 3).astype(np.float32)
        regions = build_regions(
            img,
            feature_hw=(16, 16),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected = np.arange(regions.n_regions, dtype=np.int32)
        assert np.array_equal(regions.region_ids, expected)

    def test_edges_have_weights(self):
        """Edges must include weights (not just i, j)."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        regions = build_regions(
            img,
            feature_hw=(8, 8),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert len(regions.edges) >= 0
        for edge in regions.edges:
            assert len(edge) == 3
            i, j, w = edge
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert isinstance(w, (int, float))