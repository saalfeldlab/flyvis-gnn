"""Tests for flyvis_gnn.models.utils — batch_frames and compute_normalization_value."""
import pytest
import torch

from flyvis_gnn.models.utils import _batch_frames, compute_normalization_value
from flyvis_gnn.neuron_state import NeuronState

pytestmark = pytest.mark.tier2


class TestBatchFrames:
    def test_concatenation(self):
        s1 = NeuronState.zeros(5)
        s2 = NeuronState.zeros(5)
        s2.voltage = torch.ones(5)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        batched, batched_edges = _batch_frames([s1, s2], edge_index)
        assert batched.n_neurons == 10
        assert batched_edges.shape[1] == 4  # 2 edges * 2 frames

    def test_edge_offset(self):
        s1 = NeuronState.zeros(3)
        s2 = NeuronState.zeros(3)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        _, batched_edges = _batch_frames([s1, s2], edge_index)
        # Second frame edges should be offset by 3
        assert batched_edges[0, 2].item() == 3
        assert batched_edges[1, 3].item() == 5

    def test_voltage_concatenated(self):
        s1 = NeuronState.zeros(3)
        s2 = NeuronState.zeros(3)
        s1.voltage = torch.tensor([1.0, 2.0, 3.0])
        s2.voltage = torch.tensor([4.0, 5.0, 6.0])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        batched, _ = _batch_frames([s1, s2], edge_index)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        torch.testing.assert_close(batched.voltage, expected)

    def test_single_frame(self):
        s = NeuronState.zeros(4)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        batched, batched_edges = _batch_frames([s], edge_index)
        assert batched.n_neurons == 4
        torch.testing.assert_close(batched_edges, edge_index)


class TestComputeNormalizationValue:
    def test_max_method(self):
        func_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        x_values = torch.linspace(0, 1, 5)
        val = compute_normalization_value(func_values, x_values, method="max")
        assert val == pytest.approx(5.0)

    def test_mean_method(self):
        func_values = torch.tensor([[2.0, 4.0, 6.0]])
        x_values = torch.linspace(0, 1, 3)
        val = compute_normalization_value(func_values, x_values, method="mean")
        assert val == pytest.approx(4.0)

    def test_per_neuron_max(self):
        func_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_values = torch.linspace(0, 1, 3)
        vals = compute_normalization_value(func_values, x_values, method="max", per_neuron=True)
        assert vals.shape == (2,)
        assert vals[0].item() == pytest.approx(3.0)
        assert vals[1].item() == pytest.approx(6.0)

    def test_median_method(self):
        func_values = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        x_values = torch.linspace(0, 1, 5)
        val = compute_normalization_value(func_values, x_values, method="median")
        assert val == pytest.approx(5.0)
