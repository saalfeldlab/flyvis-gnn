"""AdEx (Adaptive Exponential Integrate-and-Fire) ODE for flyvis spiking model.

Implements the full AdEx dynamics with event-triggered synaptic transmission:
    dv/dt = (-g_L*(v - v_rest) + g_L*delta_T*exp((v - v_thresh)/delta_T) - w + I) / C
    dw/dt = (-w + a*(v - v_rest)) / tau_w

COBA synapses:
    dge/dt = -ge / tau_ge
    dgi/dt = -gi / tau_gi
    I = I_bias + I_stim + ge*(E_ge - v) + gi*(E_gi - v)

CUBA synapses:
    I = I_bias + I_stim  (spike kicks applied directly to v)

Spike detection: v > v_cut  ->  v = v_reset, w += b
Refractory period: v frozen for t_refrac after spike.
"""

import torch
import torch.nn as nn

from flyvis_gnn.generators.ode_params import FlyVisAdExODEParams
from flyvis_gnn.neuron_state import NeuronState


class FlyVisAdExODE(nn.Module):
    """Ground-truth AdEx ODE for spiking neural dynamics.

    Supports both COBA and CUBA synapse models via ode_params.synapse_model.
    Uses scatter_add for event-triggered spike propagation.
    """

    def __init__(self, ode_params: FlyVisAdExODEParams, device=None):
        super().__init__()
        self.ode_params = ode_params
        self.device = device
        if self.ode_params is not None:
            self.ode_params.to(device)

    def _propagate_spikes(self, state: NeuronState):
        """Event-triggered synaptic transmission: scatter Q to postsynaptic targets.

        Only active for edges whose presynaptic neuron spiked on the previous step.
        Modifies state.ge/gi (COBA) or state.voltage (CUBA) in-place.
        """
        if state.spiked is None or not state.spiked.any():
            return

        p = self.ode_params
        src, dst = p.edge_index

        # Mask: which edges have a spiking presynaptic neuron?
        spiked_src = state.spiked[src]
        if not spiked_src.any():
            return

        active_dst = dst[spiked_src]
        active_src = src[spiked_src]
        is_exc = p.is_excitatory[active_src]

        if p.synapse_model == "COBA":
            # Excitatory: ge_post += Q_ge
            exc_dst = active_dst[is_exc]
            if exc_dst.numel() > 0:
                state.ge.scatter_add_(0, exc_dst, p.Q_ge[exc_dst])
            # Inhibitory: gi_post += Q_gi
            inh_dst = active_dst[~is_exc]
            if inh_dst.numel() > 0:
                state.gi.scatter_add_(0, inh_dst, p.Q_gi[inh_dst])
        else:
            # CUBA: v_post += J_exc or J_inh
            exc_dst = active_dst[is_exc]
            if exc_dst.numel() > 0:
                state.voltage.scatter_add_(0, exc_dst, p.J_exc[exc_dst])
            inh_dst = active_dst[~is_exc]
            if inh_dst.numel() > 0:
                state.voltage.scatter_add_(0, inh_dst, p.J_inh[inh_dst])

    def _compute_derivatives(self, state: NeuronState):
        """Compute dv/dt, dw/dt, dge/dt, dgi/dt from current state.

        Returns:
            dv, dw, dge, dgi — all (N,) tensors. dge/dgi are None for CUBA.
        """
        p = self.ode_params
        v = state.voltage
        w = state.adapt_current

        # Current
        I = p.I_bias + p.stim_scale * state.stimulus

        if p.synapse_model == "COBA":
            I = I + state.ge * (p.E_ge - v) + state.gi * (p.E_gi - v)

        # Voltage: AdEx equation
        exp_term = p.g_L * p.delta_T * torch.exp(
            torch.clamp((v - p.v_thresh) / p.delta_T, max=20.0)
        )
        dv = (-p.g_L * (v - p.v_rest) + exp_term - w + I) / p.C

        # Adaptation
        dw = (-w + p.a * (v - p.v_rest)) / p.tau_w

        # Conductance decay (COBA only)
        dge = dgi = None
        if p.synapse_model == "COBA":
            dge = -state.ge / p.tau_ge
            dgi = -state.gi / p.tau_gi

        return dv, dw, dge, dgi

    def _detect_and_reset_spikes(self, state: NeuronState):
        """Detect spikes (v > v_cut), reset v and increment w. Updates state in-place."""
        p = self.ode_params
        spiked = state.voltage > p.v_cut

        if spiked.any():
            state.voltage[spiked] = p.v_reset[spiked]
            state.adapt_current[spiked] = state.adapt_current[spiked] + p.b[spiked]
            state.refractory_counter[spiked] = p.t_refrac[spiked]

        state.spiked = spiked

    def step(self, state: NeuronState, dt: float) -> NeuronState:
        """Perform one full Euler integration step.

        Algorithm:
            1. Propagate previous step's spikes (event-triggered scatter)
            2. Compute derivatives
            3. Euler integration (skip refractory neurons for v)
            4. Spike detection and reset
            5. Decrement refractory counter

        Args:
            state: current NeuronState (modified in-place and returned)
            dt: integration timestep in ms

        Returns:
            state: updated NeuronState (same object, modified in-place)
        """
        p = self.ode_params

        # 1. Spike propagation from previous step
        self._propagate_spikes(state)

        # 2. Compute derivatives
        dv, dw, dge, dgi = self._compute_derivatives(state)

        # 3. Euler integration
        non_refractory = state.refractory_counter <= 0
        state.voltage[non_refractory] = state.voltage[non_refractory] + dv[non_refractory] * dt
        state.adapt_current = state.adapt_current + dw * dt
        if p.synapse_model == "COBA":
            state.ge = state.ge + dge * dt
            state.gi = state.gi + dgi * dt

        # 4. Spike detection and reset
        self._detect_and_reset_spikes(state)

        # 5. Decrement refractory counter
        state.refractory_counter = state.refractory_counter - dt

        return state

    def init_state(self, n_neurons: int) -> NeuronState:
        """Create initial NeuronState for AdEx simulation.

        Voltage is initialized at v_rest + random perturbation.
        All other dynamic fields start at zero.
        """
        p = self.ode_params
        device = self.device

        v_init = p.v_rest + p.v_0_mean + p.v_0_std * torch.randn(n_neurons, device=device)

        state = NeuronState(
            voltage=v_init,
            stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            adapt_current=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            spiked=torch.zeros(n_neurons, dtype=torch.bool, device=device),
            refractory_counter=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        )

        if p.synapse_model == "COBA":
            state.ge = torch.zeros(n_neurons, dtype=torch.float32, device=device)
            state.gi = torch.zeros(n_neurons, dtype=torch.float32, device=device)

        return state
