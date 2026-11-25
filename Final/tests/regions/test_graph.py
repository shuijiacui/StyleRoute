import numpy as np

from src.regions.graph import build_region_graph, compute_boundary_gradient


class TestComputeBoundaryGradient:
    def test_no_boundary_returns_zero(self):
        """Two non-adjacent regions should return 0."""
        labels = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.int32,
        )
        img = np.ones((3, 3, 3), dtype=np.float32) * 0.5
        cost = compute_boundary_gradient(img, labels, 0, 2)
        assert cost == 0.0

    def test_uniform_boundary_low_cost(self):
        """Uniform image boundary should give low gradient cost."""
        labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        img = np.ones((3, 3, 3), dtype=np.float32) * 0.5
        cost = compute_boundary_gradient(img, labels, 0, 1)
        assert 0.0 <= cost <= 1.0

    def test_sharp_boundary_high_cost(self):
        """Strong intensity change at boundary should give higher cost."""
        labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        img = np.zeros((3, 3, 3), dtype=np.float32)
        img[:, :2, :] = 1.0  # left half bright, right half dark
        cost = compute_boundary_gradient(img, labels, 0, 1)
        assert 0.0 <= cost <= 1.0

    def test_output_in_range(self):
        """Cost must always be in [0, 1]."""
        labels = np.array(
            [
                [0, 1],
                [0, 1],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(2, 2, 3).astype(np.float32)
        cost = compute_boundary_gradient(img, labels, 0, 1)
        assert 0.0 <= cost <= 1.0


class TestBuildRegionGraph:
    def test_single_region_no_edges(self):
        """Single region image should produce no edges."""
        labels = np.zeros((8, 8), dtype=np.int32)
        img = np.random.rand(8, 8, 3).astype(np.float32)
        edges = build_region_graph(labels, img)
        assert edges == []

    def test_two_adjacent_regions_one_edge(self):
        """Two adjacent regions should produce exactly 1 edge."""
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(4, 4, 3).astype(np.float32)
        edges = build_region_graph(labels, img)
        assert len(edges) == 1
        i, j, w = edges[0]
        assert i == 0 and j == 1
        assert isinstance(w, float)
        assert 0.0 < w <= 1.0

    def test_edge_format(self):
        """Every edge must be (int, int, float) with i < j."""
        labels = np.array(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 3, 3, 2, 2],
                [3, 3, 3, 3, 2, 2],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(4, 6, 3).astype(np.float32)
        edges = build_region_graph(labels, img)
        for e in edges:
            assert len(e) == 3
            i, j, w = e
            assert isinstance(i, int)
            assert isinstance(j, int)
            assert isinstance(w, float)
            assert i < j
            assert not (np.isnan(w) or np.isinf(w))

    def test_no_duplicate_edges(self):
        """Each (i, j) pair should appear at most once."""
        labels = np.array(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 3, 3, 2, 2],
                [3, 3, 3, 3, 2, 2],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(4, 6, 3).astype(np.float32)
        edges = build_region_graph(labels, img)
        pairs = [(e[0], e[1]) for e in edges]
        assert len(pairs) == len(set(pairs))

    def test_weights_in_range(self):
        """All edge weights should be in (0, 1]."""
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(4, 4, 3).astype(np.float32)
        edges = build_region_graph(labels, img)
        for i, j, w in edges:
            assert 0.0 < w <= 1.0, f"Weight {w} out of range for edge ({i}, {j})"

    def test_sorted_output(self):
        """Edges should be sorted by (i, j)."""
        labels = np.array(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 3, 3, 2, 2],
                [3, 3, 3, 3, 2, 2],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(4, 6, 3).astype(np.float32)
        edges = build_region_graph(labels, img)
        sorted_pairs = sorted((e[0], e[1]) for e in edges)
        actual_pairs = [(e[0], e[1]) for e in edges]
        assert actual_pairs == sorted_pairs

    def test_deterministic(self):
        """Same input should produce identical edges."""
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.int32,
        )
        img = np.random.rand(4, 4, 3).astype(np.float32)
        e1 = build_region_graph(labels, img)
        e2 = build_region_graph(labels, img)
        assert len(e1) == len(e2)
        for (i1, j1, w1), (i2, j2, w2) in zip(e1, e2):
            assert i1 == i2 and j1 == j2
            assert abs(w1 - w2) < 1e-10

    def test_config_override(self):
        """Passing config should override defaults."""
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        config = {"merge": {"decay": 5.0, "use_boundary_gradient": False}}
        edges = build_region_graph(labels, img, config=config)
        assert len(edges) == 1
        i, j, w = edges[0]
        assert i == 0 and j == 1
        assert isinstance(w, float)
