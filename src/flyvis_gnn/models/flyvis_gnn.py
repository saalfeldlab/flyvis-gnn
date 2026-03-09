import math

import numpy as np
import torch
import torch.nn as nn

from flyvis_gnn.models.MLP import MLP
from flyvis_gnn.models.registry import register_model
from flyvis_gnn.models.Siren_Network import Siren
from flyvis_gnn.neuron_state import NeuronState


@register_model(
    "flyvis_A", "flyvis_A_tanh", "flyvis_A_multiple_ReLU", "flyvis_A_NULL",
    "flyvis_B", "flyvis_C", "flyvis_D",
)
class FlyVisGNN(nn.Module):
    """GNN for FlyVis neural signal dynamics with per-edge W.

    Equations:
        msg_j = W[edge] * g_phi(v_j, a_j)^2   (flyvis_A, g_phi_positive=True)
        msg_j = W[edge] * g_phi(v_i, v_j, a_i, a_j)^2   (flyvis_B)
        du/dt = f_theta(v, a, sum(msg), excitation)

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    PARAMS_DOC = {
        "model_name": "FlyVisGNN",
        "description": "GNN for FlyVis neural signal dynamics with per-edge W: "
                       "du/dt = f_theta(v, a, sum(msg), excitation), msg_j = W[edge] * g_phi(v_j, a_j)",
        "key_differences_from_SignalPropagation": {
            "W_shape": "1D per-edge vector W[n_edges + n_extra_null_edges, 1] instead of dense N×N matrix",
            "visual_input": "Supports visual field input (DAVIS/calcium) via excitation channel",
            "calcium": "Can use calcium concentration instead of voltage as observable",
            "g_phi_positive": "When True, g_phi output is squared to enforce positive edge messages",
        },
        "equations": {
            "message_flyvis_A": "msg_j = W[edge_idx] * g_phi(v_j, a_j)^2   (g_phi_positive=True)",
            "message_flyvis_B": "msg_j = W[edge_idx] * g_phi(v_i, v_j, a_i, a_j)^2",
            "update": "du/dt = f_theta(v, a, sum(msg), excitation)",
        },
        "graph_model_config": {
            "description": "Parameters in the graph_model: section of the YAML config.",
            "g_phi (MLP0)": {
                "description": "Edge message function — computes per-edge features, multiplied by W[edge]",
                "input_size": {
                    "flyvis_A": "input_size = 1 + embedding_dim  (v_j, a_j)",
                    "flyvis_B": "input_size = 2 + 2*embedding_dim  (v_i, v_j, a_i, a_j)",
                },
                "output_size": "1 (scalar edge message)",
                "hidden_dim": {"description": "Hidden layer width", "typical_range": [32, 128], "default": 64},
                "n_layers": {"description": "Number of MLP layers", "typical_range": [2, 5], "default": 3},
            },
            "f_theta (MLP1)": {
                "description": "Node update function — computes du/dt from voltage + embedding + messages + excitation",
                "input_size_update": "1 + embedding_dim + output_size + 1  (v, a, msg, excitation)",
                "output_size": "1 (du/dt scalar)",
                "hidden_dim_update": {"description": "Hidden layer width", "typical_range": [32, 128], "default": 64},
                "n_layers_update": {"description": "Number of MLP layers", "typical_range": [2, 5], "default": 3},
            },
            "embedding": {
                "embedding_dim": {"description": "Dimension of learnable node embedding a_i", "typical_range": [1, 8], "default": 2},
            },
            "g_phi_positive": {"description": "If True, square g_phi output to enforce positive messages", "default": True},
            "field_type": {"description": "Visual field type — determines visual input reconstruction model (e.g. 'visual_NNR')"},
            "MLP_activation": {"description": "Activation function for MLPs", "default": "tanh"},
        },
        "simulation_params": {
            "description": "Parameters in the simulation: section of the YAML config",
            "n_neurons": {"description": "Total number of neurons in the connectome"},
            "n_input_neurons": {"description": "Number of input (photoreceptor) neurons"},
            "n_neuron_types": {"description": "Number of neuron cell types"},
            "n_edges": {"description": "Number of synaptic connections in the connectome"},
            "n_extra_null_edges": {"description": "Additional null edges for capacity (default 0)"},
            "n_frames": {"description": "Number of simulation time frames"},
            "visual_input_type": {"description": "Type of visual stimulus (e.g. 'DAVIS', 'youtube-vos')"},
            "noise_model_level": {"description": "Noise level added to observations", "typical_range": [0.0, 0.1]},
            "calcium_type": {"description": "If not 'none', use calcium concentration instead of voltage as observable"},
        },
        "training_params": {
            "description": "Parameters in the training: section that affect model architecture or loss",
            "tunable": [
                {"name": "learning_rate_W_start", "description": "Learning rate for per-edge connectivity W", "typical_range": [1e-4, 5e-2]},
                {"name": "learning_rate_start", "description": "Learning rate for MLPs", "typical_range": [1e-4, 5e-3]},
                {"name": "learning_rate_embedding_start", "description": "Learning rate for embeddings a", "typical_range": [1e-4, 5e-3]},
                {"name": "coeff_W_L1", "description": "L1 sparsity penalty on W", "typical_range": [1e-6, 1e-3]},
                {"name": "coeff_g_phi_diff", "description": "Regularizer: g_phi output variance penalty", "typical_range": [0, 500]},
                {"name": "coeff_g_phi_norm", "description": "Regularizer: edge weight norm penalty", "typical_range": [0, 10]},
                {"name": "coeff_g_phi_weight_L1", "description": "L1 penalty on g_phi weights", "typical_range": [0, 10]},
                {"name": "coeff_f_theta_weight_L1", "description": "L1 penalty on f_theta weights", "typical_range": [0, 10]},
                {"name": "coeff_f_theta_weight_L2", "description": "L2 penalty on f_theta weights", "typical_range": [0, 0.01]},
                {"name": "batch_size", "description": "Number of time frames per batch", "typical_range": [1, 4]},
                {"name": "data_augmentation_loop", "description": "Number of augmentation iterations per epoch", "typical_range": [10, 50]},
                {"name": "w_init_mode", "description": "W initialization: 'zeros' (default), 'randn', or 'randn_scaled'"},
                {"name": "w_init_scale", "description": "Scale factor for randn_scaled init (W * scale/sqrt(n_edges))", "default": 1.0},
            ],
        },
    }

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.aggr_type = aggr_type
        self.model = model_config.signal_model_name
        self.dimension = simulation_config.dimension
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.field_type = model_config.field_type
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity
        self.calcium_type = simulation_config.calcium_type
        self.MLP_activation = config.graph_model.MLP_activation

        self.training_time_window = config.training.time_window

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.g_phi_positive = model_config.g_phi_positive

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.g_phi = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            activation=self.MLP_activation,
            device=self.device,
        )

        self.f_theta = MLP(
            input_size=self.input_size_update,
            output_size=self.output_size,
            nlayers=self.n_layers_update,
            hidden_size=self.hidden_dim_update,
            activation=self.MLP_activation,
            device=self.device,
        )

        self.a = nn.Parameter(
            torch.tensor(
                np.ones((int(self.n_neurons), self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

        train_config = config.training
        n_w = self.n_edges + self.n_extra_null_edges
        w_init_mode = getattr(train_config, 'w_init_mode', 'zeros')
        if w_init_mode == 'zeros':
            W_init = torch.zeros(n_w, device=self.device, dtype=torch.float32)
        elif w_init_mode == 'randn_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            W_init = torch.randn(n_w, device=self.device, dtype=torch.float32) * (w_init_scale / math.sqrt(n_w))
        else:  # 'randn'
            W_init = torch.randn(n_w, device=self.device, dtype=torch.float32)
        self.W = nn.Parameter(W_init[:, None], requires_grad=True)

        if 'visual' in model_config.field_type:

            if 'instantNGP' in model_config.field_type:
                # to be implemented
                pass
            else:
                print('use NNR for visual field reconstruction')
                self.NNR_f = Siren(in_features=model_config.input_size_nnr_f, out_features=model_config.output_size_nnr_f,
                            hidden_features=model_config.hidden_dim_nnr_f,
                            hidden_layers=model_config.n_layers_nnr_f, first_omega_0=model_config.omega_f,
                            hidden_omega_0=model_config.omega_f,
                            outermost_linear=model_config.outermost_linear_nnr_f)
                self.NNR_f.to(self.device)

                # Match training normalization (graph_trainer.py divides by raw period).
                # Previous code divided by 2*pi here — revert if needed:
                self.NNR_f_xy_period = model_config.nnr_f_xy_period / (2*np.pi)
                self.NNR_f_T_period = model_config.nnr_f_T_period / (2*np.pi)
                # self.NNR_f_xy_period = model_config.nnr_f_xy_period
                # self.NNR_f_T_period = model_config.nnr_f_T_period

    def forward_visual(self, state: NeuronState, k):
        """Reconstruct visual field from neuron positions and time step k."""
        if 'instantNGP' in self.field_type:
            # to be implemented
            pass
        else:
            kk = torch.full((state.n_neurons, 1), float(k), device=self.device, dtype=torch.float32)
            in_features = torch.cat((state.pos[:, :self.dimension] / self.NNR_f_xy_period, kk / self.NNR_f_T_period), dim=1)
            reconstructed_field = self.NNR_f(in_features[:self.n_input_neurons]) ** 2

        return reconstructed_field

    def _compute_messages(self, v, embedding, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        args:
            v: (N, 1) observable (voltage or calcium)
            embedding: (N, embedding_dim) node embeddings
            edge_index: (2, E) source/destination indices

        returns:
            msg: (N, 1) aggregated messages per node
        """
        src, dst = edge_index

        # compute edge-to-W indices (supports batched edge_index)
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        # build per-edge features
        if self.model == 'flyvis_B':
            in_features = torch.cat([v[dst], v[src], embedding[dst], embedding[src]], dim=1)
        else:
            in_features = torch.cat([v[src], embedding[src]], dim=1)

        # edge function
        g_phi_out = self.g_phi(in_features)
        if self.g_phi_positive:
            g_phi_out = g_phi_out ** 2

        # weight by per-edge W
        edge_msg = self.W[edge_W_idx] * g_phi_out  # (E, 1)

        # aggregate: scatter_add messages to destination nodes
        msg = torch.zeros(v.shape[0], edge_msg.shape[1], device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, data_id=[], k=[], return_all=False, **kwargs):
        """Forward pass: compute du/dt from neuron state and connectivity.

        args:
            state: NeuronState with voltage, stimulus, index fields
            edge_index: (2, E) tensor of (src, dst) edge indices
            data_id: dataset ID tensor
            k: time step (for visual field reconstruction)
            return_all: if True, return (pred, in_features, msg)

        returns:
            pred: (N, 1) predicted du/dt
        """
        self.data_id = data_id.squeeze().long().clone().detach()

        v = state.observable(self.calcium_type)
        excitation = state.stimulus.unsqueeze(-1)
        particle_id = state.index.long()
        embedding = self.a[particle_id].squeeze()

        msg = self._compute_messages(v, embedding, edge_index)

        in_features = torch.cat([v, embedding, msg, excitation], dim=1)
        pred = self.f_theta(in_features)

        if return_all:
            return pred, in_features, msg
        else:
            return pred
