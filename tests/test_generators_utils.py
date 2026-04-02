"""Tests for flyvis_gnn.generators.utils — sequence and graph utilities."""
import numpy as np
import pytest
import torch

from flyvis_gnn.generators.utils import (
    apply_pairwise_knobs_torch,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)

pytestmark = pytest.mark.tier2


class TestMseqBits:
    def test_default_length(self):
        seq = mseq_bits(p=8)
        assert len(seq) == 2 ** 8 - 1

    def test_values_pm1(self):
        seq = mseq_bits(p=6)
        assert set(seq).issubset({-1, 1})

    def test_deterministic(self):
        s1 = mseq_bits(p=5, seed=7)
        s2 = mseq_bits(p=5, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_custom_length(self):
        seq = mseq_bits(p=8, length=100)
        assert len(seq) == 100

    def test_different_seeds_differ(self):
        s1 = mseq_bits(p=6, seed=1)
        s2 = mseq_bits(p=6, seed=2)
        assert not np.array_equal(s1, s2)


class TestBuildNeighborGraph:
    def test_adjacency_structure(self):
        centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        adj = build_neighbor_graph(centers, k=2)
        assert len(adj) == 4
        for neighbors in adj:
            assert len(neighbors) >= 2

    def test_symmetry(self):
        centers = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        adj = build_neighbor_graph(centers, k=2)
        for i in range(len(adj)):
            for j in adj[i]:
                assert i in adj[j]  # symmetric adjacency


class TestGreedyBlueMask:
    def test_output_shape(self):
        adj = [{1}, {0, 2}, {1, 3}, {2, 4}, {3}]
        mask = greedy_blue_mask(adj, n_cols=5, target_density=0.5)
        assert mask.shape == (5,)
        assert mask.dtype == bool

    def test_at_least_one_selected(self):
        adj = [{1}, {0, 2}, {1, 3}, {2, 4}, {3}]
        mask = greedy_blue_mask(adj, n_cols=5, target_density=0.5)
        assert mask.sum() >= 1

    def test_deterministic_with_seed(self):
        adj = [{1}, {0, 2}, {1, 3}, {2, 4}, {3}]
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        m1 = greedy_blue_mask(adj, n_cols=5, target_density=0.5, rng=rng1)
        m2 = greedy_blue_mask(adj, n_cols=5, target_density=0.5, rng=rng2)
        np.testing.assert_array_equal(m1, m2)


class TestApplyPairwiseKnobsTorch:
    def test_no_modification_when_knobs_off(self):
        code = torch.tensor([1.0, -1.0, 1.0, -1.0])
        result = apply_pairwise_knobs_torch(code, corr_strength=0.0, flip_prob=0.0, seed=42)
        torch.testing.assert_close(result, code)

    def test_output_shape_preserved(self):
        code = torch.randn(20)
        result = apply_pairwise_knobs_torch(code, corr_strength=0.5, flip_prob=0.1, seed=42)
        assert result.shape == code.shape

    def test_deterministic(self):
        code = torch.randn(50)
        r1 = apply_pairwise_knobs_torch(code, corr_strength=0.3, flip_prob=0.2, seed=42)
        r2 = apply_pairwise_knobs_torch(code, corr_strength=0.3, flip_prob=0.2, seed=42)
        torch.testing.assert_close(r1, r2)


class TestComputeColumnLabels:
    def test_output_shape(self):
        rng = np.random.RandomState(0)
        n = 50
        u = rng.randn(n).astype(np.float32)
        v = rng.randn(n).astype(np.float32)
        labels, centers = compute_column_labels(u, v, n_columns=5, seed=0)
        assert labels.shape == (n,)
        assert centers.shape == (5, 2)

    def test_all_clusters_assigned(self):
        rng = np.random.RandomState(0)
        n = 100
        u = rng.randn(n).astype(np.float32)
        v = rng.randn(n).astype(np.float32)
        labels, _ = compute_column_labels(u, v, n_columns=5, seed=0)
        assert set(labels) == {0, 1, 2, 3, 4}
