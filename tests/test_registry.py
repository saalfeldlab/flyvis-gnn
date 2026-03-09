"""Tests for flyvis_gnn.models.registry — model registration and lookup."""
import pytest

from flyvis_gnn.models.registry import create_model, list_models

pytestmark = pytest.mark.tier2


class TestListModels:
    def test_returns_list(self):
        models = list_models()
        assert isinstance(models, list)

    def test_flyvis_a_registered(self):
        models = list_models()
        assert "flyvis_A" in models

    def test_sorted(self):
        models = list_models()
        assert models == sorted(models)


class TestCreateModel:
    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            create_model("nonexistent_model_xyz")
