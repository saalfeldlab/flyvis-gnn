"""Tests for flyvis_gnn.utils — utility functions."""
import numpy as np
import pytest
import torch

from flyvis_gnn.utils import (
    choose_boundary_values,
    compute_feve,
    get_equidistant_points,
    migrate_state_dict,
    sort_key,
    to_numpy,
)

pytestmark = pytest.mark.tier1


class TestToNumpy:
    def test_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        assert result is arr  # same object

    def test_grad_tensor(self):
        t = torch.tensor([1.0], requires_grad=True)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)


class TestSortKey:
    def test_graphs_suffix(self):
        assert sort_key("model_graphs_0.pt") == 0

    def test_numeric_suffix(self):
        key = sort_key("model_3_100.pt")
        assert key == 3e7 + 100


class TestChooseBoundaryValues:
    def test_no_boundary_identity(self):
        bc, sbc = choose_boundary_values("no")
        x = torch.tensor([[0.5, 0.5]])
        torch.testing.assert_close(bc(x), x)

    def test_periodic_wraps(self):
        bc, _ = choose_boundary_values("periodic")
        x = torch.tensor([1.5, 2.3])
        result = bc(x)
        assert torch.all(result >= 0) and torch.all(result < 1)

    def test_periodic_preserves_unit_interval(self):
        bc, _ = choose_boundary_values("periodic")
        x = torch.tensor([0.3, 0.7])
        result = bc(x)
        torch.testing.assert_close(result, x)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            choose_boundary_values("unknown_bc")


class TestComputeFeve:
    def test_perfect_prediction(self):
        rng = np.random.RandomState(0)
        true = rng.randn(5, 100)
        feve = compute_feve(true, true)
        np.testing.assert_allclose(feve, 1.0, atol=1e-6)

    def test_random_prediction_low_feve(self):
        rng = np.random.RandomState(0)
        true = rng.randn(5, 100)
        pred = rng.randn(5, 100)
        feve = compute_feve(true, pred)
        assert np.all(feve < 0.5)

    def test_output_shape(self):
        rng = np.random.RandomState(0)
        true = rng.randn(8, 50)
        pred = rng.randn(8, 50)
        feve = compute_feve(true, pred)
        assert feve.shape == (8,)


class TestGetEquidistantPoints:
    def test_returns_correct_count(self):
        x, y = get_equidistant_points(n_points=256)
        assert len(x) == 256
        assert len(y) == 256

    def test_points_inside_unit_disk(self):
        x, y = get_equidistant_points(n_points=1024)
        r = np.sqrt(x ** 2 + y ** 2)
        assert np.all(r <= 1.0 + 1e-6)


class TestMigrateStateDict:
    def test_renames_keys(self):
        sd = {
            "model_state_dict": {
                "lin_edge.weight": torch.tensor([1.0]),
                "lin_phi.bias": torch.tensor([2.0]),
                "other.weight": torch.tensor([3.0]),
            }
        }
        result = migrate_state_dict(sd)
        keys = set(result["model_state_dict"].keys())
        assert "g_phi.weight" in keys
        assert "f_theta.bias" in keys
        assert "other.weight" in keys
        assert "lin_edge.weight" not in keys
        assert "lin_phi.bias" not in keys
