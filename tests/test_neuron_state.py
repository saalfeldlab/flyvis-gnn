"""Tests for flyvis_gnn.neuron_state — NeuronState and NeuronTimeSeries dataclasses."""
import numpy as np
import pytest
import torch

from flyvis_gnn.neuron_state import NeuronState, NeuronTimeSeries

pytestmark = pytest.mark.tier1


# ------------------------------------------------------------------ #
#  NeuronState
# ------------------------------------------------------------------ #

class TestNeuronStateFromNumpy:
    def test_field_shapes(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        N = neuron_state_packed.shape[0]
        assert state.n_neurons == N
        assert state.index.shape == (N,)
        assert state.pos.shape == (N, 2)
        assert state.voltage.shape == (N,)
        assert state.stimulus.shape == (N,)
        assert state.calcium.shape == (N,)
        assert state.fluorescence.shape == (N,)

    def test_field_dtypes(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        assert state.index.dtype == torch.long
        assert state.voltage.dtype == torch.float32
        assert state.group_type.dtype == torch.long
        assert state.neuron_type.dtype == torch.long

    def test_round_trip(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        packed_back = state.to_packed()
        np.testing.assert_allclose(
            packed_back.numpy(), neuron_state_packed, atol=1e-5
        )


class TestNeuronStateZeros:
    def test_shape_and_values(self):
        state = NeuronState.zeros(5)
        assert state.n_neurons == 5
        assert torch.all(state.voltage == 0)
        assert state.index.tolist() == [0, 1, 2, 3, 4]

    def test_all_fields_populated(self):
        state = NeuronState.zeros(3)
        assert state.index is not None
        assert state.pos is not None
        assert state.voltage is not None
        assert state.stimulus is not None
        assert state.calcium is not None
        assert state.fluorescence is not None


class TestNeuronStateObservable:
    def test_voltage_mode(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        obs = state.observable(calcium_type="none")
        assert obs.shape == (state.n_neurons, 1)
        torch.testing.assert_close(obs.squeeze(), state.voltage)

    def test_calcium_mode(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        obs = state.observable(calcium_type="leaky")
        assert obs.shape == (state.n_neurons, 1)
        torch.testing.assert_close(obs.squeeze(), state.calcium)


class TestNeuronStateSubset:
    def test_subset_length(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        sub = state.subset([0, 2, 4])
        assert sub.n_neurons == 3

    def test_subset_values(self, neuron_state_packed):
        state = NeuronState.from_numpy(neuron_state_packed)
        sub = state.subset([0, 2, 4])
        assert sub.index.tolist() == [0, 2, 4]
        torch.testing.assert_close(sub.voltage, state.voltage[[0, 2, 4]])


class TestNeuronStateDevice:
    def test_device_property(self):
        state = NeuronState.zeros(5)
        assert state.device == torch.device("cpu")


class TestNeuronStateClone:
    def test_clone_is_independent(self):
        state = NeuronState.zeros(5)
        cloned = state.clone()
        cloned.voltage[0] = 999.0
        assert state.voltage[0] == 0.0


# ------------------------------------------------------------------ #
#  NeuronTimeSeries
# ------------------------------------------------------------------ #

class TestNeuronTimeSeriesFromNumpy:
    def test_shapes(self, neuron_timeseries_packed):
        ts = NeuronTimeSeries.from_numpy(neuron_timeseries_packed)
        T, N = neuron_timeseries_packed.shape[:2]
        assert ts.n_frames == T
        assert ts.n_neurons == N
        assert ts.voltage.shape == (T, N)

    def test_static_fields_not_repeated(self, neuron_timeseries_packed):
        ts = NeuronTimeSeries.from_numpy(neuron_timeseries_packed)
        N = neuron_timeseries_packed.shape[1]
        # Static fields should have shape (N,) or (N, 2), not (T, N)
        assert ts.index.shape == (N,)
        assert ts.pos.shape == (N, 2)


class TestNeuronTimeSeriesFrame:
    def test_frame_extraction(self, neuron_timeseries_packed):
        ts = NeuronTimeSeries.from_numpy(neuron_timeseries_packed)
        frame = ts.frame(5)
        assert isinstance(frame, NeuronState)
        assert frame.n_neurons == ts.n_neurons

    def test_frame_voltage_matches(self, neuron_timeseries_packed):
        ts = NeuronTimeSeries.from_numpy(neuron_timeseries_packed)
        frame = ts.frame(3)
        torch.testing.assert_close(frame.voltage, ts.voltage[3])


class TestNeuronTimeSeriesSubset:
    def test_subset_neurons(self, neuron_timeseries_packed):
        ts = NeuronTimeSeries.from_numpy(neuron_timeseries_packed)
        sub = ts.subset_neurons(np.array([0, 3, 7]))
        assert sub.n_neurons == 3
        assert sub.voltage.shape == (ts.n_frames, 3)
        assert sub.index.shape == (3,)


class TestNeuronTimeSeriesXnorm:
    def test_xnorm_positive(self, neuron_timeseries_packed):
        ts = NeuronTimeSeries.from_numpy(neuron_timeseries_packed)
        xnorm = ts.xnorm
        assert xnorm.item() > 0
