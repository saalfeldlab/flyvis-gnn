"""Tests for flyvis_gnn.models.Siren_Network — SIREN shape validation."""
import pytest
import torch

from flyvis_gnn.models.Siren_Network import SineLayer, Siren

pytestmark = pytest.mark.tier3


class TestSineLayer:
    def test_output_shape(self):
        layer = SineLayer(in_features=3, out_features=16, is_first=True, omega_0=30)
        x = torch.randn(10, 3)
        out = layer(x)
        assert out.shape == (10, 16)

    def test_output_bounded(self):
        layer = SineLayer(in_features=2, out_features=8, is_first=True, omega_0=1)
        x = torch.randn(100, 2) * 0.01
        out = layer(x)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_hidden_layer(self):
        layer = SineLayer(in_features=8, out_features=8, is_first=False, omega_0=30)
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 8)


class TestSiren:
    def test_output_shape_linear_out(self):
        model = Siren(in_features=2, out_features=1, hidden_features=32,
                      hidden_layers=2, outermost_linear=True)
        x = torch.randn(50, 2)
        out = model(x)
        assert out.shape == (50, 1)

    def test_output_shape_sine_out(self):
        model = Siren(in_features=3, out_features=1, hidden_features=16,
                      hidden_layers=1, outermost_linear=False)
        x = torch.randn(20, 3)
        out = model(x)
        assert out.shape == (20, 1)

    def test_get_omegas(self):
        model = Siren(in_features=2, out_features=1, hidden_features=16,
                      hidden_layers=2, first_omega_0=30, hidden_omega_0=30)
        omegas = model.get_omegas()
        assert len(omegas) >= 2
        assert all(isinstance(o, (int, float)) for o in omegas)

    def test_no_nans(self):
        model = Siren(in_features=2, out_features=1, hidden_features=16,
                      hidden_layers=2, outermost_linear=True)
        x = torch.randn(30, 2)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_learnable_omega(self):
        model = Siren(in_features=2, out_features=1, hidden_features=16,
                      hidden_layers=1, learnable_omega=True)
        x = torch.randn(10, 2)
        out = model(x)
        assert out.shape == (10, 1)
        # Omega should be a Parameter
        loss = model.get_omega_L2_loss()
        assert loss > 0
