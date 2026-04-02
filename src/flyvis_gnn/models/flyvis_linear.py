"""Linear ODE model for FlyVis — simple baseline with fixed ReLU activation.

Matches the ground-truth ODE structure:
    msg_j  = W_j * relu(v_j)
    dv/dt  = (-v + msg + excitation + V_rest) / tau

All three parameter sets (tau, V_rest, W) are directly learned.
No MLP, no embeddings — the activation function g_phi is just ReLU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flyvis_gnn.models.registry import register_model
from flyvis_gnn.neuron_state import NeuronState


@register_model("flyvis_linear", "flyvis_linear_tanh")
class FlyVisLinear(nn.Module):
    """Linear ODE baseline for FlyVis neural signal dynamics.

    Equations:
        msg   = sum_j  W_j * relu(v_j)           (scatter_add over incoming edges)
        dv/dt = (-v + msg + excitation + V_rest) / tau
        (flyvis_linear_tanh adds  + s * tanh(v)  inside the numerator)

    Learnable parameters:
        raw_tau : (n_neurons,)  —  tau = softplus(raw_tau) > 0
        V_rest  : (n_neurons,)  —  per-neuron resting potential
        W       : (n_edges + n_extra_null_edges, 1) — per-edge synaptic weight
    """

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.signal_model_name
        self.calcium_type = simulation_config.calcium_type
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.batch_size = train_config.batch_size
        self.update_type = model_config.update_type  # kept for regularizer compat

        # --- tau (time constant, must be > 0) ---
        # Parameterised as softplus(raw_tau).  Initialised so tau ≈ 1.
        self.raw_tau = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))

        # --- V_rest (resting potential) ---
        self.V_rest = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))

        # --- per-edge weights W ---
        n_w = self.n_edges + self.n_extra_null_edges
        w_init_mode = getattr(train_config, 'w_init_mode', 'zeros')
        if w_init_mode == 'zeros':
            W_init = torch.zeros(n_w, device=device, dtype=torch.float32)
        elif w_init_mode == 'randn_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            W_init = torch.randn(n_w, device=device, dtype=torch.float32) * (w_init_scale / math.sqrt(n_w))
        else:  # 'randn'
            W_init = torch.randn(n_w, device=device, dtype=torch.float32)
        self.W = nn.Parameter(W_init[:, None], requires_grad=True)

        # --- tanh self-coupling (flyvis_linear_tanh only) ---
        if 'tanh' in self.model:
            self.s = nn.Parameter(
                torch.ones(1, device=device, dtype=torch.float32))

    # ------------------------------------------------------------------ #

    def _compute_messages(self, v, edge_index):
        """msg_j = W_j * relu(v_j), aggregated via scatter_add."""
        src, dst = edge_index
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        edge_msg = self.W[edge_W_idx] * F.relu(v[src])  # (E, 1)

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    # ------------------------------------------------------------------ #

    def forward(self, state: NeuronState, edge_index: torch.Tensor,
                data_id=[], k=[], return_all=False, **kwargs):
        """Compute dv/dt from neuron state and connectivity.

        Returns:
            pred          when return_all=False
            (pred, None, msg)  when return_all=True
              — in_features is None (no MLP features to expose)
        """
        self.data_id = data_id.squeeze().long().clone().detach() if hasattr(data_id, 'squeeze') else data_id

        v = state.observable(self.calcium_type)         # (N, 1)
        excitation = state.stimulus.unsqueeze(-1)       # (N, 1)

        msg = self._compute_messages(v, edge_index)     # (N, 1)

        # Map raw_tau → positive tau via softplus, handling batched state
        particle_id = state.index.long()
        tau = F.softplus(self.raw_tau[particle_id]).unsqueeze(-1)      # (N, 1)
        v_rest = self.V_rest[particle_id].unsqueeze(-1)                # (N, 1)

        if 'tanh' in self.model:
            pred = (-v + msg + excitation + v_rest + self.s * torch.tanh(v)) / tau
        else:
            pred = (-v + msg + excitation + v_rest) / tau

        if return_all:
            return pred, None, msg
        return pred
