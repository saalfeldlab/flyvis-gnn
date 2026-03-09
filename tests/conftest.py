"""Shared fixtures for flyvis-gnn unit tests.

All fixtures produce CPU-only, deterministically-seeded data.
No disk I/O, no GPU, no external data files.
"""
import os

# Prevent matplotlib from trying to open display windows
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch


@pytest.fixture
def rng():
    """Deterministic numpy RNG."""
    return np.random.RandomState(42)


@pytest.fixture
def torch_generator():
    """Deterministic torch Generator (CPU)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    return g


# --------------- Tier 1 fixtures ---------------

@pytest.fixture
def sample_1d_arrays(rng):
    """Matched pair of 1D float64 arrays, length 50, for R^2 / fitting tests."""
    true = rng.randn(50).astype(np.float64)
    noise = rng.randn(50).astype(np.float64) * 0.1
    learned = 2.0 * true + 0.5 + noise  # slope~2, offset~0.5, high R^2
    return true, learned


@pytest.fixture
def sample_2d_tensors():
    """(N, n_pts) torch tensors for vectorized linear fit tests."""
    torch.manual_seed(0)
    N, n_pts = 8, 100
    x = torch.linspace(-1, 1, n_pts).unsqueeze(0).expand(N, -1)
    slopes = torch.arange(1, N + 1, dtype=torch.float32)
    offsets = torch.zeros(N)
    y = slopes[:, None] * x + offsets[:, None] + torch.randn(N, n_pts) * 0.01
    return x, y, slopes, offsets


@pytest.fixture
def neuron_state_packed():
    """Legacy (N, 9) numpy array for NeuronState.from_numpy tests.

    Columns: [index, xpos, ypos, voltage, stimulus,
              group_type, neuron_type, calcium, fluorescence]
    """
    rng = np.random.RandomState(123)
    N = 10
    arr = np.zeros((N, 9), dtype=np.float32)
    arr[:, 0] = np.arange(N)           # index
    arr[:, 1] = np.linspace(0, 1, N)   # xpos
    arr[:, 2] = np.linspace(0, 1, N)   # ypos
    arr[:, 3] = rng.randn(N)           # voltage
    arr[:, 4] = rng.rand(N)            # stimulus
    arr[:, 5] = np.arange(N) % 3       # group_type
    arr[:, 6] = np.arange(N) % 5       # neuron_type
    arr[:, 7] = rng.rand(N)            # calcium
    arr[:, 8] = rng.rand(N)            # fluorescence
    return arr


@pytest.fixture
def neuron_timeseries_packed():
    """Legacy (T, N, 9) numpy array for NeuronTimeSeries.from_numpy tests."""
    rng = np.random.RandomState(456)
    T, N = 20, 10
    arr = np.zeros((T, N, 9), dtype=np.float32)
    arr[:, :, 0] = np.arange(N)[None, :]
    arr[:, :, 1] = np.linspace(0, 1, N)[None, :]
    arr[:, :, 2] = np.linspace(0, 1, N)[None, :]
    arr[:, :, 3] = rng.randn(T, N)     # voltage
    arr[:, :, 4] = rng.rand(T, N)      # stimulus
    arr[:, :, 5] = (np.arange(N) % 3)[None, :]
    arr[:, :, 6] = (np.arange(N) % 5)[None, :]
    arr[:, :, 7] = rng.rand(T, N)      # calcium
    arr[:, :, 8] = rng.rand(T, N)      # fluorescence
    return arr


# --------------- Tier 2 fixtures ---------------

@pytest.fixture
def simple_edge_index():
    """Small (2, E) edge index for a 5-node fully-connected graph (no self-loops)."""
    src, dst = [], []
    for i in range(5):
        for j in range(5):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


@pytest.fixture
def minimal_config_dict():
    """Minimal valid config dict that can construct NeuralGraphConfig."""
    return {
        "dataset": "test_dataset",
        "simulation": {
            "params": [[1.0, 1.0]],
            "n_neurons": 10,
            "n_input_neurons": 2,
            "n_neuron_types": 2,
            "n_edges": 20,
            "n_frames": 100,
            "delta_t": 0.02,
            "seed": 42,
        },
        "graph_model": {
            "signal_model_name": "flyvis_A",
            "aggr_type": "add",
            "embedding_dim": 2,
            "input_size": 3,
            "output_size": 1,
            "hidden_dim": 16,
            "n_layers": 2,
            "input_size_update": 5,
            "n_layers_update": 2,
            "hidden_dim_update": 16,
            "g_phi_positive": True,
            "update_type": "generic",
        },
        "plotting": {
            "colormap": "tab10",
        },
        "training": {
            "device": "cpu",
            "n_epochs": 1,
            "n_runs": 1,
            "batch_size": 1,
            "seed": 42,
        },
    }


@pytest.fixture
def minimal_config(minimal_config_dict):
    """A NeuralGraphConfig object with small sizes for CPU testing."""
    from flyvis_gnn.config import NeuralGraphConfig
    return NeuralGraphConfig(**minimal_config_dict)
