"""Tests for flyvis_gnn.models.training_utils — config-driven logic."""
import pytest

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.training_utils import determine_load_fields

pytestmark = pytest.mark.tier2


class TestDetermineLoadFields:
    def test_base_fields(self, minimal_config):
        fields = determine_load_fields(minimal_config)
        assert "voltage" in fields
        assert "stimulus" in fields
        assert "neuron_type" in fields

    def test_visual_field_adds_pos(self, minimal_config_dict):
        minimal_config_dict["graph_model"]["field_type"] = "visual_NNR"
        config = NeuralGraphConfig(**minimal_config_dict)
        fields = determine_load_fields(config)
        assert "pos" in fields

    def test_calcium_adds_calcium(self, minimal_config_dict):
        d = minimal_config_dict.copy()
        d["simulation"] = {**d["simulation"], "calcium_type": "leaky"}
        config = NeuralGraphConfig(**d)
        fields = determine_load_fields(config)
        assert "calcium" in fields

    def test_noise_adds_noise(self, minimal_config_dict):
        d = minimal_config_dict.copy()
        d["simulation"] = {**d["simulation"], "measurement_noise_level": 0.05}
        config = NeuralGraphConfig(**d)
        fields = determine_load_fields(config)
        assert "noise" in fields

    def test_no_extra_fields_by_default(self, minimal_config):
        fields = determine_load_fields(minimal_config)
        assert "calcium" not in fields
        assert "noise" not in fields
