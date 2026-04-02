"""Tests for flyvis_gnn.fitting_models — pure math functions."""
import numpy as np
import pytest

from flyvis_gnn.fitting_models import linear_fit, linear_model, power_model

pytestmark = pytest.mark.tier1


class TestLinearModel:
    def test_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(linear_model(x, 1.0, 0.0), x)

    def test_slope_and_offset(self):
        x = np.array([0.0, 1.0])
        np.testing.assert_allclose(linear_model(x, 2.0, 3.0), [3.0, 5.0])

    def test_zero_slope(self):
        x = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(linear_model(x, 0.0, 5.0), [5.0, 5.0, 5.0])


class TestPowerModel:
    def test_basic(self):
        x = np.array([1.0, 2.0, 4.0])
        result = power_model(x, a=8.0, b=1.0)
        np.testing.assert_allclose(result, [8.0, 4.0, 2.0])

    def test_exponent_two(self):
        x = np.array([1.0, 2.0, 3.0])
        result = power_model(x, a=1.0, b=2.0)
        np.testing.assert_allclose(result, [1.0, 0.25, 1.0 / 9.0])

    def test_scalar(self):
        assert power_model(2.0, a=4.0, b=1.0) == pytest.approx(2.0)


class TestLinearFit:
    def test_perfect_linear_data(self):
        x_data = np.linspace(-1, 1, 100)
        y_data = 3.0 * x_data + 2.0
        lin_fit, r2, _, _, _, _ = linear_fit(x_data, y_data)
        assert r2 == pytest.approx(1.0, abs=1e-6)
        assert lin_fit[0] == pytest.approx(3.0, abs=1e-4)
        assert lin_fit[1] == pytest.approx(2.0, abs=1e-4)

    def test_noisy_data(self):
        rng = np.random.RandomState(99)
        x_data = np.linspace(0, 10, 200)
        y_data = 1.5 * x_data + 0.5 + rng.randn(200) * 0.1
        lin_fit, r2, _, _, _, _ = linear_fit(x_data, y_data)
        assert r2 > 0.99
        assert lin_fit[0] == pytest.approx(1.5, abs=0.1)
