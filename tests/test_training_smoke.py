"""Smoke tests for training components — LossRegularizer lifecycle and gradient flow.

These tests exercise the training infrastructure without requiring real data or GPU.
"""
import pytest
import torch

from flyvis_gnn.models.flyvis_gnn import FlyVisGNN
from flyvis_gnn.models.regularizer import LossRegularizer
from flyvis_gnn.neuron_state import NeuronState

pytestmark = pytest.mark.tier3


@pytest.fixture
def training_setup(minimal_config):
    """Create a tiny model, state, edges, and regularizer for smoke testing."""
    config = minimal_config
    N = config.simulation.n_neurons
    n_edges = config.simulation.n_edges

    model = FlyVisGNN(aggr_type="add", config=config, device="cpu")
    model.train()

    state = NeuronState.zeros(N)
    state.voltage = torch.randn(N)
    state.stimulus = torch.randn(N)
    state.index = torch.arange(N, dtype=torch.float32)

    src = torch.randint(0, N, (n_edges,))
    dst = torch.randint(0, N, (n_edges,))
    edge_index = torch.stack([src, dst])

    regularizer = LossRegularizer(
        train_config=config.training,
        model_config=config.graph_model,
        activity_column=6,
        plot_frequency=10,
        n_neurons=N,
        trainer_type='flyvis',
    )

    return model, state, edge_index, regularizer, config


class TestLossRegularizerSmoke:
    """Smoke tests for LossRegularizer lifecycle."""

    def test_instantiation(self, training_setup):
        _, _, _, regularizer, _ = training_setup
        assert regularizer.epoch == 0
        assert regularizer.iter_count == 0
        assert len(regularizer.COMPONENTS) > 0

    def test_set_epoch(self, training_setup):
        _, _, _, regularizer, _ = training_setup
        regularizer.set_epoch(1, Niter=100)
        assert regularizer.epoch == 1
        assert regularizer.Niter == 100

    def test_reset_and_finalize(self, training_setup):
        _, _, _, regularizer, _ = training_setup
        regularizer.set_epoch(0, Niter=10)
        regularizer.reset_iteration()
        regularizer.finalize_iteration()
        assert regularizer.iter_count == 1

    def test_history_recording(self, training_setup):
        _, _, _, regularizer, _ = training_setup
        regularizer.set_epoch(0, Niter=10)
        # iter_count=1 triggers recording (should_record returns True)
        regularizer.reset_iteration()
        regularizer.finalize_iteration()
        history = regularizer.get_history()
        assert len(history['iteration']) == 1
        assert len(history['regul_total']) == 1

    def test_backward_compat_import(self):
        """Verify LossRegularizer can still be imported from models.utils."""
        from flyvis_gnn.models.utils import LossRegularizer as LR
        assert LR is LossRegularizer


class TestGradientFlow:
    """Verify gradients flow through the model forward pass."""

    def test_forward_backward(self, training_setup):
        model, state, edge_index, _, _ = training_setup
        data_id = torch.zeros(state.n_neurons, 1, dtype=torch.int)

        pred = model(state, edge_index, data_id=data_id)
        loss = pred.sum()
        loss.backward()

        # At least some parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients flowed through the model"

    def test_loss_is_finite(self, training_setup):
        model, state, edge_index, _, _ = training_setup
        data_id = torch.zeros(state.n_neurons, 1, dtype=torch.int)

        pred = model(state, edge_index, data_id=data_id)
        loss = pred.sum()
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_optimizer_step(self, training_setup):
        """Verify a full optimizer step doesn't produce NaNs."""
        model, state, edge_index, _, _ = training_setup
        data_id = torch.zeros(state.n_neurons, 1, dtype=torch.int)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Forward
        pred = model(state, edge_index, data_id=data_id)
        loss = pred.sum()

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify no NaN parameters after step
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"NaN in parameter {name} after optimizer step"

    def test_two_step_loss_changes(self, training_setup):
        """Verify loss changes after an optimizer step (model is learning)."""
        model, state, edge_index, _, _ = training_setup
        data_id = torch.zeros(state.n_neurons, 1, dtype=torch.int)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Step 1
        pred1 = model(state, edge_index, data_id=data_id)
        loss1 = pred1.pow(2).sum()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # Step 2
        pred2 = model(state, edge_index, data_id=data_id)
        loss2 = pred2.pow(2).sum()

        # Loss should change (not necessarily decrease with 1 step, but should differ)
        assert loss1.item() != loss2.item(), "Loss did not change after optimizer step"
