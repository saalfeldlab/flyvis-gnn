"""Tests for flyvis_gnn.models.flyvis_dataset — Dataset and Sampler interfaces."""
import numpy as np
import pytest
import torch

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.flyvis_dataset import FlyVisDataset, FlyVisFrameSampler
from flyvis_gnn.neuron_state import NeuronState, NeuronTimeSeries

pytestmark = pytest.mark.tier3


@pytest.fixture
def synthetic_dataset():
    """Build a FlyVisDataset from synthetic data with known dimensions."""
    T, N = 100, 10
    cfg_dict = {
        "dataset": "test",
        "simulation": {
            "params": [[1.0]],
            "n_neurons": N,
            "n_frames": T,
            "n_edges": 10,
            "n_neuron_types": 2,
            "seed": 42,
        },
        "graph_model": {
            "signal_model_name": "flyvis_A",
            "aggr_type": "add",
            "embedding_dim": 2,
            "input_size": 3,
            "output_size": 1,
            "hidden_dim": 8,
            "n_layers": 2,
            "input_size_update": 5,
            "n_layers_update": 2,
            "hidden_dim_update": 8,
        },
        "plotting": {"colormap": "tab10"},
        "training": {
            "device": "cpu",
            "n_runs": 1,
            "seed": 42,
            "time_step": 1,
            "time_window": 0,
        },
    }
    config = NeuralGraphConfig(**cfg_dict)

    torch.manual_seed(0)
    x_ts = NeuronTimeSeries(
        index=torch.arange(N),
        pos=torch.randn(N, 2),
        voltage=torch.randn(T, N),
        stimulus=torch.randn(T, N),
    )
    y_ts = np.random.RandomState(0).randn(T, N, 1).astype(np.float32)
    return FlyVisDataset(x_ts, y_ts, config)


class TestFlyVisDataset:
    def test_len_positive(self, synthetic_dataset):
        assert len(synthetic_dataset) > 0

    def test_len_correct(self, synthetic_dataset):
        # _max_k = n_frames - 4 - time_step = 100 - 4 - 1 = 95
        # _min_k = time_window = 0
        assert len(synthetic_dataset) == 95

    def test_getitem_returns_tuple(self, synthetic_dataset):
        x, y, k = synthetic_dataset[0]
        assert isinstance(x, NeuronState)
        assert isinstance(y, torch.Tensor)
        assert isinstance(k, int)

    def test_getitem_shapes(self, synthetic_dataset):
        x, y, k = synthetic_dataset[0]
        assert x.n_neurons == 10
        assert y.shape == (10, 1)

    def test_set_epoch(self, synthetic_dataset):
        synthetic_dataset.set_epoch(5)
        assert synthetic_dataset.loss_noise_level == pytest.approx(0.0)  # config noise = 0


class TestFlyVisFrameSampler:
    def test_len(self, synthetic_dataset):
        sampler = FlyVisFrameSampler(synthetic_dataset, num_samples=50, seed=42)
        assert len(sampler) == 50

    def test_deterministic(self, synthetic_dataset):
        s1 = FlyVisFrameSampler(synthetic_dataset, num_samples=20, seed=42)
        s2 = FlyVisFrameSampler(synthetic_dataset, num_samples=20, seed=42)
        assert list(s1) == list(s2)

    def test_epoch_changes_order(self, synthetic_dataset):
        s = FlyVisFrameSampler(synthetic_dataset, num_samples=20, seed=42)
        s.set_epoch(0)
        order0 = list(s)
        s.set_epoch(1)
        order1 = list(s)
        assert order0 != order1

    def test_indices_in_range(self, synthetic_dataset):
        sampler = FlyVisFrameSampler(synthetic_dataset, num_samples=100, seed=42)
        indices = list(sampler)
        assert all(0 <= i < len(synthetic_dataset) for i in indices)
