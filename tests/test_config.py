"""Tests for flyvis_gnn.config — Enum values and NeuralGraphConfig construction."""
import pytest

from flyvis_gnn.config import (
    Boundary,
    CalciumType,
    Integration,
    MLPActivation,
    NeuralGraphConfig,
    Prediction,
    UpdateType,
)

pytestmark = pytest.mark.tier2


class TestEnums:
    def test_boundary_values(self):
        assert Boundary.PERIODIC == "periodic"
        assert Boundary.NO == "no"
        assert Boundary.WALL == "wall"

    def test_prediction_values(self):
        assert Prediction.FIRST_DERIVATIVE == "first_derivative"
        assert Prediction.SECOND_DERIVATIVE == "2nd_derivative"

    def test_integration_values(self):
        assert Integration.EULER == "Euler"
        assert Integration.RUNGE_KUTTA == "Runge-Kutta"

    def test_calcium_type_none(self):
        assert CalciumType.NONE == "none"
        assert CalciumType.LEAKY == "leaky"

    def test_update_type_generic(self):
        assert UpdateType.GENERIC == "generic"
        assert UpdateType.MLP == "mlp"

    def test_mlp_activation(self):
        assert MLPActivation.RELU == "relu"
        assert MLPActivation.TANH == "tanh"


class TestNeuralGraphConfig:
    def test_construction_from_dict(self, minimal_config_dict):
        config = NeuralGraphConfig(**minimal_config_dict)
        assert config.dataset == "test_dataset"
        assert config.simulation.n_neurons == 10
        assert config.graph_model.signal_model_name == "flyvis_A"
        assert config.training.device == "cpu"

    def test_defaults_applied(self, minimal_config_dict):
        config = NeuralGraphConfig(**minimal_config_dict)
        assert config.simulation.boundary == Boundary.PERIODIC
        assert config.training.learning_rate_start == 0.001
        assert config.graph_model.prediction == Prediction.SECOND_DERIVATIVE

    def test_simulation_defaults(self, minimal_config):
        assert minimal_config.simulation.dimension == 2
        assert minimal_config.simulation.seed == 42
        assert minimal_config.simulation.delta_t == 0.02

    def test_description_default(self, minimal_config):
        assert minimal_config.description == "flyvis_gnn"
