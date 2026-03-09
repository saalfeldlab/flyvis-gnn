"""Tests for flyvis_gnn.models.MLP — shape validation."""
import pytest
import torch

from flyvis_gnn.models.MLP import MLP

pytestmark = pytest.mark.tier3


class TestMLPShapes:
    @pytest.mark.parametrize("nlayers", [2, 3, 5])
    def test_output_shape(self, nlayers):
        model = MLP(input_size=4, output_size=1, nlayers=nlayers,
                     hidden_size=16, device="cpu")
        x = torch.randn(10, 4)
        out = model(x)
        assert out.shape == (10, 1)

    @pytest.mark.parametrize("activation", [
        "relu", "tanh", "sigmoid", "leaky_relu", "soft_relu", "none",
    ])
    def test_activation_variants(self, activation):
        model = MLP(input_size=3, output_size=2, nlayers=3,
                     hidden_size=8, device="cpu", activation=activation)
        x = torch.randn(5, 3)
        out = model(x)
        assert out.shape == (5, 2)
        assert torch.isfinite(out).all()

    def test_batch_dimensions(self):
        model = MLP(input_size=2, output_size=1, nlayers=2,
                     hidden_size=8, device="cpu")
        for batch in [1, 32, 128]:
            out = model(torch.randn(batch, 2))
            assert out.shape == (batch, 1)

    @pytest.mark.parametrize("init", ["zeros", "ones", None])
    def test_initialisation_modes(self, init):
        model = MLP(input_size=3, output_size=1, nlayers=2,
                     hidden_size=8, device="cpu", initialisation=init)
        x = torch.randn(5, 3)
        out = model(x)
        assert out.shape == (5, 1)
        assert torch.isfinite(out).all()
