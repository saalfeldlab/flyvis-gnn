"""
Neural ODE wrapper for FlyVisGNN.

Uses torchdiffeq's adjoint method for memory-efficient training:
- Memory O(1) in rollout steps L (vs O(L) for BPTT)
- Backward pass uses adjoint ODE solve
"""

import logging

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from flyvis_gnn.neuron_state import NeuronState

logger = logging.getLogger(__name__)


class GNNODEFunc_FlyVis(nn.Module):
    """
    Wraps GNN model as ODE vector field: dv/dt = f(t, v).

    Uses NeuronState as template; voltage is updated by the ODE solver each step.
    """

    def __init__(self, model, x_template, edge_index, data_id, neurons_per_sample, batch_size,
                 has_visual_field=False, x_ts=None,
                 device=None, k_batch=None, state_clamp=10.0, stab_lambda=0.0):
        super().__init__()
        self.model = model
        self.x_template = x_template      # NeuronState (batched, B*N neurons)
        self.edge_index = edge_index       # (2, B*E) batched edge index
        self.data_id = data_id
        self.neurons_per_sample = neurons_per_sample
        self.batch_size = batch_size
        self.has_visual_field = has_visual_field
        self.x_ts = x_ts
        self.device = device or torch.device('cpu')
        self.k_batch = k_batch  # per-sample k values, shape (batch_size,)
        self.delta_t = 1.0
        self.state_clamp = state_clamp  # clamp state to [-state_clamp, state_clamp]
        self.stab_lambda = stab_lambda  # damping: dv = GNN(v) - lambda*v

    def set_time_params(self, delta_t):
        self.delta_t = delta_t

    def forward(self, t, v):
        """Compute dv/dt = GNN(v). Called by ODE solver at each integration step."""
        state = self.x_template.clone()
        state.voltage = v.view(-1)

        # k_offset: discrete time step offset from continuous time t
        k_offset = int((t / self.delta_t).item()) if t.numel() == 1 else 0

        if self.has_visual_field and hasattr(self.model, 'forward_visual'):
            # For visual field, process each batch sample separately
            for b in range(self.batch_size):
                start_idx = b * self.neurons_per_sample
                end_idx = (b + 1) * self.neurons_per_sample
                k_current = int(self.k_batch[b].item()) + k_offset

                state_b = state.subset(range(start_idx, end_idx))
                visual_input = self.model.forward_visual(state_b, k_current)
                n_input = getattr(self.model, 'n_input_neurons', self.neurons_per_sample)
                state.stimulus[start_idx:start_idx + n_input] = visual_input.squeeze(-1)
                state.stimulus[start_idx + n_input:end_idx] = 0

        elif self.x_ts is not None:
            # Update visual input for each batch sample from x_ts
            for b in range(self.batch_size):
                start_idx = b * self.neurons_per_sample
                end_idx = (b + 1) * self.neurons_per_sample
                k_current = int(self.k_batch[b].item()) + k_offset

                if k_current < self.x_ts.n_frames:
                    state.stimulus[start_idx:end_idx] = self.x_ts.stimulus[k_current]

        pred = self.model(
            state,
            self.edge_index,
            data_id=self.data_id,
            return_all=False
        )

        dv = pred.view(-1)

        return dv


def integrate_neural_ode_FlyVis(model, v0, x_template, edge_index, data_id, time_steps, delta_t,
                         neurons_per_sample, batch_size, has_visual_field=False,
                         x_ts=None, device=None, k_batch=None,
                         ode_method='dopri5', rtol=1e-4, atol=1e-5,
                         adjoint=True, noise_level=0.0, state_clamp=10.0, stab_lambda=0.0):
    """
    Integrate GNN dynamics using Neural ODE.

    args:
        model: FlyVisGNN model
        v0: initial voltage state (B*N,)
        x_template: NeuronState used as template for ODE steps (batched, B*N neurons)
        edge_index: (2, B*E) batched edge index
        data_id: dataset ID tensor
        time_steps: number of integration steps
        delta_t: time step size
        neurons_per_sample: number of neurons per sample
        batch_size: number of samples in batch
        ode_method: solver ('dopri5', 'rk4', 'euler', etc.)
        rtol, atol: tolerances for adaptive solvers
        adjoint: use adjoint method for O(1) memory
        state_clamp: clamp state to [-state_clamp, state_clamp]
        stab_lambda: damping coefficient for stability

    returns:
        v_final: final state (B*N,)
        v_trajectory: states at all time points (time_steps+1, B*N)
    """

    # adjoint: O(1) memory, standard: faster but O(L) memory
    solver = odeint_adjoint if adjoint else odeint

    ode_func = GNNODEFunc_FlyVis(
        model=model,
        x_template=x_template,
        edge_index=edge_index,
        data_id=data_id,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        has_visual_field=has_visual_field,
        x_ts=x_ts,
        device=device,
        k_batch=k_batch,
        state_clamp=state_clamp,
        stab_lambda=stab_lambda
    )
    ode_func.set_time_params(delta_t)

    # t_span: evaluation points [0, dt, 2*dt, ..., time_steps*dt]
    t_span = torch.linspace(
        0, time_steps * delta_t, time_steps + 1,
        device=device, dtype=v0.dtype
    )

    # v_trajectory shape: (time_steps+1, B*N)
    v_trajectory = solver(
        ode_func,
        v0.flatten(),
        t_span,
        method=ode_method,
        rtol=rtol,
        atol=atol
    )

    if noise_level > 0 and model.training:
        v_trajectory = v_trajectory + noise_level * torch.randn_like(v_trajectory)

    v_final = v_trajectory[-1]

    return v_final, v_trajectory


def neural_ode_loss_FlyVis(model, dataset_batch, edge_index, x_ts, k_batch,
                           time_step, batch_size, n_neurons, ids_batch,
                           delta_t, device,
                           data_id=None, has_visual_field=False,
                           y_batch=None, noise_level=0.0, ode_method='dopri5',
                           rtol=1e-4, atol=1e-5, adjoint=True,
                           iteration=0, state_clamp=10.0, stab_lambda=0.0):
    """
    Compute loss using Neural ODE integration.

    args:
        dataset_batch: list of NeuronState (one per frame in batch)
        edge_index: (2, E) edge index for a single frame (will be batched internally)
    """

    # Batch frames: concatenate NeuronState fields and offset edge indices
    n_per_frame = dataset_batch[0].n_neurons
    x_template = NeuronState(
        index=torch.cat([s.index for s in dataset_batch]),
        pos=torch.cat([s.pos for s in dataset_batch]),
        group_type=torch.cat([s.group_type for s in dataset_batch]),
        neuron_type=torch.cat([s.neuron_type for s in dataset_batch]),
        voltage=torch.cat([s.voltage for s in dataset_batch]),
        stimulus=torch.cat([s.stimulus for s in dataset_batch]),
        calcium=torch.cat([s.calcium for s in dataset_batch]),
        fluorescence=torch.cat([s.fluorescence for s in dataset_batch]),
    )
    batched_edges = torch.cat(
        [edge_index + i * n_per_frame for i in range(len(dataset_batch))], dim=1
    )

    v0 = x_template.voltage.flatten()
    neurons_per_sample = n_per_frame

    # Extract per-sample k values (one per batch sample)
    k_per_sample = torch.tensor([
        k_batch[b * neurons_per_sample, 0].item()
        for b in range(batch_size)
    ], device=device)

    if iteration % 500 == 0:
        logger.debug(f"Neural ODE iter {iteration}: time_step={time_step}, delta_t={delta_t}, "
                      f"batch_size={batch_size}, neurons_per_sample={neurons_per_sample}")
        logger.debug(f"  v0 shape={v0.shape}, mean={v0.mean().item():.4f}, std={v0.std().item():.4f}")
        logger.debug(f"  k_per_sample={k_per_sample.tolist()}, ode_method={ode_method}, adjoint={adjoint}")

    v_final, v_trajectory = integrate_neural_ode_FlyVis(
        model=model,
        v0=v0,
        x_template=x_template,
        edge_index=batched_edges,
        data_id=data_id,
        time_steps=time_step,
        delta_t=delta_t,
        neurons_per_sample=neurons_per_sample,
        batch_size=batch_size,
        has_visual_field=has_visual_field,
        x_ts=x_ts,
        device=device,
        k_batch=k_per_sample,
        ode_method=ode_method,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        noise_level=noise_level,
        state_clamp=state_clamp,
        stab_lambda=stab_lambda
    )

    pred_x = v_final.view(-1, 1)
    loss = ((pred_x[ids_batch] - y_batch[ids_batch]) / (delta_t * time_step)).norm(2)

    if iteration % 500 == 0:
        logger.debug(f"  v_final mean={v_final.mean().item():.4f}, std={v_final.std().item():.4f}")
        logger.debug(f"  pred_x mean={pred_x[ids_batch].mean().item():.4f}, loss={loss.item():.4f}")

    return loss, pred_x


def debug_check_gradients(model, loss, iteration):
    """Log gradient flow diagnostics at DEBUG level after loss.backward()."""
    logger.debug(f"Gradient check (iter {iteration}), loss={loss.item():.6f}")

    if hasattr(model, 'W'):
        if model.W.grad is not None:
            w_grad = model.W.grad
            logger.debug(f"  W.grad: mean={w_grad.mean().item():.8f}, "
                          f"std={w_grad.std().item():.8f}, max={w_grad.abs().max().item():.8f}, "
                          f"nonzero={(w_grad.abs() > 1e-10).sum().item()}/{w_grad.numel()}")
        else:
            logger.debug("  W.grad is None — no gradient flowing to W")

    if hasattr(model, 'phi_edge'):
        phi_grads = []
        for name, param in model.phi_edge.named_parameters():
            if param.grad is not None:
                phi_grads.append(param.grad.abs().mean().item())
        if phi_grads:
            logger.debug(f"  phi_edge grads: mean={sum(phi_grads)/len(phi_grads):.8f}")

    if hasattr(model, 'embedding'):
        if model.embedding.weight.grad is not None:
            emb_grad = model.embedding.weight.grad
            logger.debug(f"  embedding.grad: mean={emb_grad.mean().item():.8f}, "
                          f"std={emb_grad.std().item():.8f}")
