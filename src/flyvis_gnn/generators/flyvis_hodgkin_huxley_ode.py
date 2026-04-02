"""Hodgkin-Huxley ODE for flyvis continuous spiking model.

Implements classic HH dynamics with continuous synaptic coupling:
    C * dv/dt = -g_L*(v - E_L) - g_Na*m^3*h*(v - E_Na) - g_K*n^4*(v - E_K) + I_syn + I_ext

Gate dynamics (alpha/beta rate functions from Hodgkin & Huxley 1952):
    dm/dt = alpha_m(v)*(1-m) - beta_m(v)*m
    dh/dt = alpha_h(v)*(1-h) - beta_h(v)*h
    dn/dt = alpha_n(v)*(1-n) - beta_n(v)*n

Synaptic coupling is continuous and voltage-dependent (no event-triggered spikes):
    I_syn = sum_j W_j * sigma(v_j)  where sigma is a sigmoid activation
    This is structurally similar to the graded model's W * g_phi(v).

Architecture follows flyvis_voltage (continuous Euler integration, no events),
NOT flyvis_adex (event-triggered). The forward() method returns dv/dt,
and the generator loop does standard Euler: v += dt * dv/dt.
"""

import torch
import torch.nn as nn

from flyvis_gnn.generators.ode_params import FlyVisHodgkinHuxleyODEParams
from flyvis_gnn.neuron_state import NeuronState


def _alpha_m(v):
    """Na activation rate. v in mV."""
    dv = v + 40.0
    # Numerically stable form: avoid 0/0 when dv ~ 0
    return torch.where(
        dv.abs() < 1e-6,
        torch.ones_like(v),
        0.1 * dv / (1.0 - torch.exp(-dv / 10.0)),
    )


def _beta_m(v):
    return 4.0 * torch.exp(-(v + 65.0) / 18.0)


def _alpha_h(v):
    return 0.07 * torch.exp(-(v + 65.0) / 20.0)


def _beta_h(v):
    return 1.0 / (1.0 + torch.exp(-(v + 35.0) / 10.0))


def _alpha_n(v):
    dv = v + 55.0
    return torch.where(
        dv.abs() < 1e-6,
        0.1 * torch.ones_like(v),
        0.01 * dv / (1.0 - torch.exp(-dv / 10.0)),
    )


def _beta_n(v):
    return 0.125 * torch.exp(-(v + 65.0) / 80.0)


def _m_inf(v):
    am = _alpha_m(v)
    return am / (am + _beta_m(v))


def _h_inf(v):
    ah = _alpha_h(v)
    return ah / (ah + _beta_h(v))


def _n_inf(v):
    an = _alpha_n(v)
    return an / (an + _beta_n(v))


class FlyVisHodgkinHuxleyODE(nn.Module):
    """Ground-truth Hodgkin-Huxley ODE for continuous spiking dynamics.

    Like FlyVisODE (graded voltage model), this returns dv/dt from forward().
    The generator loop handles Euler integration externally.

    Synaptic coupling uses continuous voltage-dependent activation:
        I_syn_i = sum_{j->i} W_j * sigmoid((v_j - v_half) / slope)
    This is analogous to the graded model's W * relu(v) but with a sigmoid
    activation that represents voltage-dependent transmitter release.
    """

    def __init__(self, ode_params: FlyVisHodgkinHuxleyODEParams, device=None):
        super().__init__()
        self.ode_params = ode_params
        self.device = device
        if self.ode_params is not None:
            self.ode_params.to(device)

    def _compute_synaptic_current(self, state: NeuronState):
        """Continuous voltage-dependent synaptic current via scatter_add.

        Presynaptic activation: sigmoid((v_pre - v_half) / slope)
        Weighted by connectome weights W, aggregated to postsynaptic targets.

        Returns:
            I_syn: (N,) synaptic current per neuron
        """
        p = self.ode_params
        src, dst = p.edge_index
        v_pre = state.voltage[src]

        # Continuous presynaptic activation (sigmoid of voltage)
        activation = torch.sigmoid((v_pre - p.syn_v_half[src]) / p.syn_slope[src])

        # Weighted messages
        edge_msg = p.W * activation

        # Aggregate to postsynaptic neurons
        I_syn = torch.zeros(state.voltage.shape[0], device=self.device, dtype=state.voltage.dtype)
        I_syn.scatter_add_(0, dst, edge_msg)

        return I_syn

    def forward(self, state: NeuronState, edge_index: torch.Tensor, has_field=False, data_id=[]):
        """Compute dv/dt from HH dynamics + continuous synaptic input.

        Also updates gate variables (m, h, n) in-place on the state.

        Args:
            state: NeuronState with voltage, stimulus, hh_m, hh_h, hh_n fields
            edge_index: (2, E) — not used (edge_index is in ode_params), kept for API compat

        Returns:
            dv: (N, 1) voltage derivative (same shape as FlyVisODE output)
        """
        p = self.ode_params
        v = state.voltage
        m = state.hh_m
        h = state.hh_h
        n = state.hh_n

        # Ion channel currents
        I_Na = p.g_Na * (m ** 3) * h * (v - p.E_Na)
        I_K = p.g_K * (n ** 4) * (v - p.E_K)
        I_L = p.g_L * (v - p.E_L)

        # Synaptic current (continuous, voltage-dependent)
        I_syn = self._compute_synaptic_current(state)

        # External input
        I_ext = p.I_bias + p.stim_scale * state.stimulus

        # dv/dt = (-I_Na - I_K - I_L + I_syn + I_ext) / C
        dv = (-I_Na - I_K - I_L + I_syn + I_ext) / p.C

        return dv.unsqueeze(-1)

    def step_gates(self, state: NeuronState, dt: float):
        """Update gate variables m, h, n using Euler integration.

        Called separately from forward() so the generator can control
        the integration order (gates updated after voltage step).
        """
        v = state.voltage
        m, h, n = state.hh_m, state.hh_h, state.hh_n

        # Gate derivatives
        dm = _alpha_m(v) * (1.0 - m) - _beta_m(v) * m
        dh = _alpha_h(v) * (1.0 - h) - _beta_h(v) * h
        dn = _alpha_n(v) * (1.0 - n) - _beta_n(v) * n

        # Euler step
        state.hh_m = torch.clamp(m + dt * dm, 0.0, 1.0)
        state.hh_h = torch.clamp(h + dt * dh, 0.0, 1.0)
        state.hh_n = torch.clamp(n + dt * dn, 0.0, 1.0)

    def init_state(self, n_neurons: int, v_init: float = -65.0) -> NeuronState:
        """Create initial NeuronState for HH simulation.

        Voltage starts at v_init (resting).
        Gates initialized at steady-state values for v_init.
        """
        device = self.device

        v = torch.full((n_neurons,), v_init, dtype=torch.float32, device=device)

        return NeuronState(
            voltage=v,
            stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            hh_m=_m_inf(v),
            hh_h=_h_inf(v),
            hh_n=_n_inf(v),
        )
