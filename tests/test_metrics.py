"""Tests for flyvis_gnn.metrics — pure math, no I/O."""
import numpy as np
import pytest
import torch

from flyvis_gnn.metrics import (
    ANATOMICAL_ORDER,
    INDEX_TO_NAME,
    _torch_linear_fit,
    _vectorized_linear_fit,
    _vectorized_linspace,
    compute_activity_stats,
    compute_r_squared,
    compute_r_squared_filtered,
    derive_tau,
    derive_vrest,
)

pytestmark = pytest.mark.tier1


# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

class TestConstants:
    def test_index_to_name_has_65_entries(self):
        assert len(INDEX_TO_NAME) == 65

    def test_anatomical_order_length(self):
        # Should cover all 65 neuron indices plus one None sentinel
        assert len(ANATOMICAL_ORDER) == 66


# ------------------------------------------------------------------ #
#  R² computation
# ------------------------------------------------------------------ #

class TestComputeRSquared:
    def test_perfect_linear_fit(self):
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        learned = 2.0 * true + 1.0
        r2, slope = compute_r_squared(true, learned)
        assert r2 == pytest.approx(1.0, abs=1e-6)
        assert slope == pytest.approx(2.0, abs=1e-6)

    def test_noisy_fit_high_r2(self, sample_1d_arrays):
        true, learned = sample_1d_arrays
        r2, slope = compute_r_squared(true, learned)
        assert 0.9 < r2 <= 1.0
        assert slope == pytest.approx(2.0, abs=0.5)

    def test_identity_fit(self):
        true = np.linspace(0, 10, 100)
        r2, slope = compute_r_squared(true, true)
        assert r2 == pytest.approx(1.0, abs=1e-6)
        assert slope == pytest.approx(1.0, abs=1e-6)

    def test_negative_slope(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        learned = -3.0 * true + 10.0
        r2, slope = compute_r_squared(true, learned)
        assert r2 == pytest.approx(1.0, abs=1e-6)
        assert slope == pytest.approx(-3.0, abs=1e-6)


class TestComputeRSquaredFiltered:
    def test_outlier_removal(self):
        true = np.linspace(0, 1, 50)
        learned = true.copy()
        learned[0] = 100.0  # outlier
        r2, slope, mask = compute_r_squared_filtered(true, learned, outlier_threshold=5.0)
        assert mask[0] == False  # outlier removed
        assert mask[1:].all()
        assert r2 == pytest.approx(1.0, abs=1e-4)

    def test_no_outliers(self):
        true = np.linspace(0, 1, 30)
        learned = true * 2.0
        r2, slope, mask = compute_r_squared_filtered(true, learned, outlier_threshold=100.0)
        assert mask.all()
        assert r2 == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ #
#  Vectorized helpers
# ------------------------------------------------------------------ #

class TestVectorizedLinspace:
    def test_shape(self):
        starts = np.array([0.0, 10.0])
        ends = np.array([1.0, 20.0])
        result = _vectorized_linspace(starts, ends, n_pts=5, device="cpu")
        assert result.shape == (2, 5)

    def test_endpoints(self):
        starts = np.array([0.0, 10.0])
        ends = np.array([1.0, 20.0])
        result = _vectorized_linspace(starts, ends, n_pts=5, device="cpu")
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, -1].item() == pytest.approx(1.0)
        assert result[1, 0].item() == pytest.approx(10.0)
        assert result[1, -1].item() == pytest.approx(20.0)

    def test_single_row(self):
        result = _vectorized_linspace(np.array([0.0]), np.array([1.0]), n_pts=11, device="cpu")
        assert result.shape == (1, 11)
        assert result[0, 5].item() == pytest.approx(0.5)


class TestVectorizedLinearFit:
    def test_exact_slopes(self, sample_2d_tensors):
        x, y, expected_slopes, expected_offsets = sample_2d_tensors
        slopes, offsets = _vectorized_linear_fit(x, y)
        np.testing.assert_allclose(slopes, expected_slopes.numpy(), atol=0.15)

    def test_constant_x_returns_zero_slope(self):
        x = np.ones((3, 10))
        y = np.random.randn(3, 10)
        slopes, offsets = _vectorized_linear_fit(x, y)
        np.testing.assert_array_equal(slopes, 0.0)

    def test_accepts_tensors(self):
        x = torch.linspace(-1, 1, 50).unsqueeze(0)
        y = 3.0 * x + 1.0
        slopes, offsets = _vectorized_linear_fit(x, y)
        assert slopes[0] == pytest.approx(3.0, abs=1e-4)
        assert offsets[0] == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------ #
#  Derived quantities
# ------------------------------------------------------------------ #

class TestDeriveTau:
    def test_negative_slope_gives_positive_tau(self):
        slopes = np.array([-2.0, -1.0, -0.5])
        tau = derive_tau(slopes, 3)
        np.testing.assert_allclose(tau, [0.5, 1.0, 1.0])  # clipped to [0, 1]

    def test_zero_slope_gives_one(self):
        slopes = np.array([0.0])
        tau = derive_tau(slopes, 1)
        assert tau[0] == 1.0

    def test_n_neurons_truncation(self):
        slopes = np.array([-1.0, -2.0, -4.0, -5.0])
        tau = derive_tau(slopes, 2)
        assert len(tau) == 2


class TestDeriveVrest:
    def test_basic(self):
        slopes = np.array([-2.0, -1.0])
        offsets = np.array([4.0, 3.0])
        vrest = derive_vrest(slopes, offsets, 2)
        np.testing.assert_allclose(vrest, [2.0, 3.0])

    def test_zero_slope(self):
        slopes = np.array([0.0])
        offsets = np.array([1.0])
        vrest = derive_vrest(slopes, offsets, 1)
        assert vrest[0] == 1.0

    def test_n_neurons_truncation(self):
        slopes = np.array([-1.0, -2.0, -3.0])
        offsets = np.array([2.0, 4.0, 6.0])
        vrest = derive_vrest(slopes, offsets, 2)
        assert len(vrest) == 2


# ------------------------------------------------------------------ #
#  Torch linear fit
# ------------------------------------------------------------------ #

class TestTorchLinearFit:
    def test_matches_numpy_version(self, sample_2d_tensors):
        x, y, _, _ = sample_2d_tensors
        slopes_np, offsets_np = _vectorized_linear_fit(x, y)
        slopes_t, offsets_t = _torch_linear_fit(x, y)
        np.testing.assert_allclose(slopes_t.detach().numpy(), slopes_np, atol=1e-4)
        np.testing.assert_allclose(offsets_t.detach().numpy(), offsets_np, atol=1e-4)

    def test_gradient_flows_through_y(self):
        x = torch.linspace(-1, 1, 50).unsqueeze(0)
        y = (2.0 * x + 1.0).clone().requires_grad_(True)
        slopes, offsets = _torch_linear_fit(x, y)
        loss = slopes.sum()
        loss.backward()
        assert y.grad is not None

    def test_perfect_line(self):
        x = torch.linspace(0, 1, 100).unsqueeze(0)
        y = 5.0 * x - 3.0
        slopes, offsets = _torch_linear_fit(x, y)
        assert slopes[0].item() == pytest.approx(5.0, abs=1e-4)
        assert offsets[0].item() == pytest.approx(-3.0, abs=1e-4)


# ------------------------------------------------------------------ #
#  Activity statistics
# ------------------------------------------------------------------ #

class TestComputeActivityStats:
    def test_shape(self):
        from flyvis_gnn.neuron_state import NeuronTimeSeries
        ts = NeuronTimeSeries(voltage=torch.randn(20, 10))
        mu, sigma = compute_activity_stats(ts)
        assert mu.shape == (10,)
        assert sigma.shape == (10,)

    def test_known_values(self):
        from flyvis_gnn.neuron_state import NeuronTimeSeries
        # constant voltage => std = 0
        v = torch.ones(50, 5) * 3.0
        ts = NeuronTimeSeries(voltage=v)
        mu, sigma = compute_activity_stats(ts)
        torch.testing.assert_close(mu, torch.full((5,), 3.0))
        torch.testing.assert_close(sigma, torch.zeros(5))
