"""Tests for flyvis_gnn.models.flyvis_gnn — FlyVisGNN forward pass shapes."""
import pytest
import torch

from flyvis_gnn.models.flyvis_gnn import FlyVisGNN
from flyvis_gnn.neuron_state import NeuronState

pytestmark = pytest.mark.tier3


@pytest.fixture
def model_and_inputs(minimal_config):
    """Create a tiny FlyVisGNN and matching synthetic inputs."""
    model = FlyVisGNN(aggr_type="add", config=minimal_config, device="cpu")
    model.eval()

    N = minimal_config.simulation.n_neurons
    state = NeuronState.zeros(N)
    state.voltage = torch.randn(N)
    state.stimulus = torch.randn(N)
    state.index = torch.arange(N)

    n_edges = minimal_config.simulation.n_edges
    src = torch.randint(0, N, (n_edges,))
    dst = torch.randint(0, N, (n_edges,))
    edge_index = torch.stack([src, dst])

    data_id = torch.zeros(N, 1, dtype=torch.int)
    return model, state, edge_index, data_id


class TestFlyVisGNNForward:
    def test_output_shape(self, model_and_inputs):
        model, state, edge_index, data_id = model_and_inputs
        pred = model(state, edge_index, data_id=data_id)
        assert pred.shape == (state.n_neurons, 1)

    def test_return_all(self, model_and_inputs):
        model, state, edge_index, data_id = model_and_inputs
        pred, in_features, msg = model(state, edge_index,
                                        data_id=data_id, return_all=True)
        assert pred.shape == (state.n_neurons, 1)
        assert msg.shape == (state.n_neurons, 1)

    def test_no_nans(self, model_and_inputs):
        model, state, edge_index, data_id = model_and_inputs
        pred = model(state, edge_index, data_id=data_id)
        assert torch.isfinite(pred).all()


class TestFlyVisGNNAttributes:
    def test_parameter_counts(self, model_and_inputs):
        model = model_and_inputs[0]
        assert hasattr(model, "g_phi")
        assert hasattr(model, "f_theta")
        assert hasattr(model, "a")
        assert hasattr(model, "W")

    def test_embedding_shape(self, model_and_inputs, minimal_config):
        model = model_and_inputs[0]
        N = minimal_config.simulation.n_neurons
        emb_dim = minimal_config.graph_model.embedding_dim
        assert model.a.shape == (N, emb_dim)

    def test_w_shape(self, model_and_inputs, minimal_config):
        model = model_and_inputs[0]
        n_w = minimal_config.simulation.n_edges + minimal_config.simulation.n_extra_null_edges
        assert model.W.shape == (n_w, 1)
