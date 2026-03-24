import numpy as np
import torch
import torch.nn as nn

from flyvis_gnn.generators.ode_params import FlyVisODEParams
from flyvis_gnn.neuron_state import NeuronState


def group_by_direction_and_function(neuron_type):
    if neuron_type in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
        return 0  # Outer photoreceptors
    elif neuron_type in ['R7', 'R8']:
        return 1  # Inner photoreceptors
    elif neuron_type in ['L1', 'L2', 'L3', 'L4', 'L5']:
        return 2  # Lamina monopolar
    elif neuron_type in ['Am', 'C2', 'C3']:
        return 3  # Lamina interneurons
    elif neuron_type in ['Mi1', 'Mi2', 'Mi3', 'Mi4']:
        return 4  # Early Mi neurons
    elif neuron_type in ['Mi9', 'Mi10', 'Mi11', 'Mi12']:
        return 5  # Mid Mi neurons
    elif neuron_type in ['Mi13', 'Mi14', 'Mi15']:
        return 6  # Late Mi neurons
    elif neuron_type in ['Tm1', 'Tm2', 'Tm3', 'Tm4']:
        return 7  # Early Tm neurons
    elif neuron_type in ['Tm5a', 'Tm5b', 'Tm5c', 'Tm5Y']:
        return 8  # Tm5 family
    elif neuron_type in ['Tm9', 'Tm16', 'Tm20']:
        return 9  # Mid Tm neurons
    elif neuron_type in ['Tm28', 'Tm30']:
        return 10  # Late Tm neurons
    elif neuron_type.startswith('TmY'):
        return 11  # TmY neurons
    elif neuron_type == 'T4a':
        return 12  # T4a (upward motion)
    elif neuron_type == 'T4b':
        return 13  # T4b (rightward motion)
    elif neuron_type == 'T4c':
        return 14  # T4c (downward motion)
    elif neuron_type == 'T4d':
        return 15  # T4d (leftward motion)
    elif neuron_type in ['T5a', 'T5b', 'T5c', 'T5d']:
        return 16  # T5 OFF motion detectors
    elif neuron_type in ['T1', 'T2', 'T2a', 'T3']:
        return 17  # Tangential neurons
    elif neuron_type.startswith('Lawf'):
        return 18  # Wide-field neurons
    else:
        return 19  # Other/CT1


def get_photoreceptor_positions_from_net(net):
    """Extract photoreceptor positions from flyvis network.

    Returns x, y coordinates for all input neurons (R1-R8).
    """
    nodes = net.connectome.nodes

    print(f"total nodes: {len(nodes['u'])}")

    u_coords = np.array(nodes['u'])
    v_coords = np.array(nodes['v'])
    node_types = np.array(nodes['type'])
    node_roles = np.array(nodes['role'])

    node_types_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in node_types]
    node_roles_str = [r.decode('utf-8') if isinstance(r, bytes) else str(r) for r in node_roles]

    print(f"available node types: {set(node_types_str)}")
    print(f"available node roles: {set(node_roles_str)}")

    photoreceptor_types = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    photoreceptor_mask = np.array([t in photoreceptor_types for t in node_types_str])
    input_mask = np.array([r == 'input' for r in node_roles_str])

    print(f"photoreceptor type mask (R1-R8): {np.sum(photoreceptor_mask)} neurons")
    print(f"input role mask: {np.sum(input_mask)} neurons")

    mask = photoreceptor_mask
    print("using photoreceptor type mask (R1-R8)")

    u_photo = u_coords[mask]
    v_photo = v_coords[mask]

    x_coords = u_photo + 0.5 * v_photo
    y_coords = v_photo * np.sqrt(3) / 2

    return x_coords, y_coords, u_photo, v_photo


class FlyVisODE(nn.Module):
    """Ground-truth ODE for flyvis neural signal dynamics.

    Computes dv/dt = (-v + msg + e + v_rest [+ s*tanh(v)]) / tau
    where msg = sum_j w_j * f(v_j) over incoming edges.

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    def __init__(self, aggr_type="add", ode_params=None, params=[], g_phi=torch.nn.functional.relu, model_type=None, n_neuron_types=None, device=None):
        super().__init__()

        # Accept dict (legacy) or ODE_params_class instance
        if isinstance(ode_params, dict):
            ode_params = FlyVisODEParams(**ode_params)
        self.ode_params = ode_params
        self.g_phi = g_phi
        self.model_type = model_type
        self.device = device

        if self.ode_params is not None:
            self.ode_params.to(device)

        if 'multiple_ReLU' in model_type:
            if n_neuron_types is None:
                raise ValueError("n_neuron_types must be provided for multiple_ReLU model type")
            if params[0][0]>0:
                self.params = torch.tensor(params[0], dtype=torch.float32, device=device).expand((n_neuron_types, 1))
            else:
                self.params = torch.abs(1 + 0.5 * torch.randn((n_neuron_types, 1), dtype=torch.float32, device=device))
        else:
            self.params = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    def _compute_messages(self, v, particle_type, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        args:
            v: (N, 1) voltage
            particle_type: (N, 1) long — neuron type indices
            edge_index: (2, E) source/destination indices

        returns:
            msg: (N, 1) aggregated messages per node
        """
        src, dst = edge_index

        v_src = v[src]
        particle_type_src = particle_type[src]

        if 'multiple_ReLU' in self.model_type:
            edge_msg = self.ode_params.W[:, None] * self.g_phi(v_src) * self.params[particle_type_src.squeeze()]
        elif 'NULL' in self.model_type:
            edge_msg = 0 * self.g_phi(v_src)
        else:
            edge_msg = self.ode_params.W[:, None] * self.g_phi(v_src)

        msg = torch.zeros(v.shape[0], edge_msg.shape[1], device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, has_field=False, data_id=[]):
        """Compute dv/dt from neuron state and connectivity.

        args:
            state: NeuronState with voltage, stimulus, neuron_type fields
            edge_index: (2, E) tensor of (src, dst) edge indices

        returns:
            dv: (N, 1) voltage derivative
        """
        v = state.voltage.unsqueeze(-1)
        v_rest = self.ode_params.V_i_rest[:, None]
        e = state.stimulus.unsqueeze(-1)
        particle_type = state.neuron_type.unsqueeze(-1).long()
        msg = self._compute_messages(v, particle_type, edge_index)
        tau = self.ode_params.tau_i[:, None]

        if 'tanh' in self.model_type:
            s = self.params
            dv = (-v + msg + e + v_rest + s * torch.tanh(v)) / tau
        else:
            dv = (-v + msg + e + v_rest) / tau

        return dv

    def func(self, u, type, function):
        if function == 'phi':
            if 'multiple_ReLU' in self.model_type:
                return self.g_phi(u) * self.params[type]
            else:
                return self.g_phi(u)
        elif function == 'update':
            v_rest = self.ode_params.V_i_rest[type]
            tau = self.ode_params.tau_i[type]
            if 'tanh' in self.model_type:
                s = self.params
                return (-u + v_rest + s * torch.tanh(u)) / tau
            else:
                return (-u + v_rest) / tau
