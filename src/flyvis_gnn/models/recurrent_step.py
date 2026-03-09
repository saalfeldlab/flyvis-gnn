"""Recurrent multi-step training loss for FlyVis GNN.

Overview of recurrent / noise-aware training strategies
-------------------------------------------------------
All strategies load a pretrained one-step model and fine-tune it.
The goal is to improve robustness to observation noise (process +
measurement) without sacrificing connectivity R².

1. **Standard recurrent** (``recurrent_training=True``):
   Pick one random frame k, unroll time_step forward using the model's
   own predictions, compare predicted voltage at k+time_step to the
   observed (noisy) target. Forces the model to be self-consistent
   over multiple steps, but gradient flows through a long noisy chain.
   Config: ``recurrent_training: true, time_step: N``

2. **Multi-start recurrent** (``multi_start_recurrent=True``):
   For a target frame T, launch time_step parallel rollouts from
   T-time_step, T-time_step+1, ..., T-1 (lengths time_step down to 1).
   All predictions target the same observed v(T). Each start has
   independent noise on its initial voltage, so gradient noise from
   different starts partially cancels. Short paths (1-step) anchor the
   gradient while long paths enforce trajectory consistency.
   Config: ``recurrent_training: true, multi_start_recurrent: true, time_step: N``

3. **Consecutive batch** (``consecutive_batch=True``):
   Instead of sampling batch_size random frames, pick one random start k
   and use frames k, k+1, ..., k+batch_size-1. Each frame gets a
   standard one-step prediction (no unrolling). Consecutive frames share
   the same local dynamics but have independent noise realisations, so
   the gradient over the batch naturally averages out noise. Simplest
   approach: no extra memory, no multi-step backprop, just a sampling
   change.
   Config: ``consecutive_batch: true, batch_size: N``
   (no recurrent_training needed)

Modes 1 and 2 are implemented in this module. Mode 3 is a sampling
change in graph_trainer.py (no dedicated function needed).
"""

import numpy as np
import torch

from flyvis_gnn.models.utils import _batch_frames


def recurrent_loss(
    model,
    x_ts,
    y_ts,
    edges,
    ids,
    frame_indices,
    iter_idx,
    config,
    device,
    xnorm,
    ynorm,
    regularizer,
    has_visual_field=False,
):
    """Compute one training iteration of recurrent (possibly multi-start) loss.

    Returns:
        loss: scalar tensor (already includes regularisation)
        regul_value: float, regularisation component for logging
    """
    sim = config.simulation
    tc = config.training
    time_step = tc.time_step
    n_neurons = sim.n_neurons
    multi_start = tc.multi_start_recurrent

    if multi_start:
        return _multi_start_loss(
            model, x_ts, edges, ids, frame_indices, iter_idx,
            time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
        )
    else:
        return _standard_recurrent_loss(
            model, x_ts, edges, ids, frame_indices, iter_idx,
            time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
        )


# ------------------------------------------------------------------ #
#  Standard recurrent: single start, unroll time_step forward         #
# ------------------------------------------------------------------ #

def _standard_recurrent_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
):
    batch_size = tc.batch_size
    n_neurons = sim.n_neurons
    data_id = torch.zeros((n_neurons * batch_size, 1), dtype=torch.int, device=device)

    state_batch = []
    y_list = []
    ids_list = []
    ids_index = 0

    for b in range(batch_size):
        k = int(frame_indices[iter_idx * batch_size + b])
        k = k - k % time_step  # align to time_step boundary

        x = x_ts.frame(k)
        if x.noise is not None and sim.measurement_noise_level > 0:
            x.voltage = x.voltage + x.noise

        if has_visual_field:
            visual_input = model.forward_visual(x, k)
            x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
            x.stimulus[model.n_input_neurons:] = 0

        if torch.isnan(x.voltage).any():
            continue

        y = x_ts.voltage[k + time_step].unsqueeze(-1)
        if torch.isnan(y).any():
            continue

        state_batch.append(x)
        y_list.append(y)
        ids_list.append(ids + ids_index)
        ids_index += x.n_neurons

    if not state_batch:
        return torch.zeros(1, device=device, requires_grad=True), 0.0

    y_batch = torch.cat(y_list, dim=0)
    ids_batch = np.concatenate(ids_list, axis=0)
    data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)

    # Regularisation (computed once on initial state)
    regularizer.reset_iteration()
    regul_loss = regularizer.compute(
        model=model, x=state_batch[0], in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
    )
    loss = regul_loss.clone()
    regul_value = regul_loss.item()

    # Forward pass + unroll
    batched_state, batched_edges = _batch_frames(state_batch, edges)
    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
    loss = loss + update_regul

    pred_x = batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

    for step in range(time_step - 1):
        neurons_per_sample = state_batch[0].n_neurons
        for b_idx in range(len(state_batch)):
            s, e = b_idx * neurons_per_sample, (b_idx + 1) * neurons_per_sample
            state_batch[b_idx].voltage = pred_x[s:e].squeeze()
            k_current = int(frame_indices[iter_idx * batch_size + b_idx])
            k_current = k_current - k_current % time_step + step + 1
            if has_visual_field:
                vi = model.forward_visual(state_batch[b_idx], k_current)
                state_batch[b_idx].stimulus[:model.n_input_neurons] = vi.squeeze(-1)
                state_batch[b_idx].stimulus[model.n_input_neurons:] = 0
            else:
                state_batch[b_idx].stimulus = x_ts.frame(k_current).stimulus

        batched_state, batched_edges = _batch_frames(state_batch, edges)
        pred, _, _ = model(batched_state, batched_edges, data_id=data_id, return_all=True)
        pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * time_step)).norm(2)
    return loss, regul_value


# ------------------------------------------------------------------ #
#  Multi-start recurrent: time_step starts all targeting frame T      #
# ------------------------------------------------------------------ #

def _multi_start_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
):
    """Launch time_step rollouts of decreasing length, all targeting frame T.

    Start frames: T - time_step, T - time_step + 1, ..., T - 1
    Rollout lengths: time_step, time_step - 1, ..., 1
    Target: observed v(T) for all.
    """
    n_neurons = sim.n_neurons

    # Pick target frame T (one per iteration, use first frame index)
    k_raw = int(frame_indices[iter_idx * time_step])  # batch_size == time_step
    T = max(time_step, k_raw)  # ensure we have enough history
    T = min(T, x_ts.n_frames - 1)  # stay in bounds

    # Target voltage at T (same for all starts)
    y_target = x_ts.voltage[T].unsqueeze(-1)
    if torch.isnan(y_target).any():
        return torch.zeros(1, device=device, requires_grad=True), 0.0

    # Regularisation (compute once)
    x0 = x_ts.frame(T - time_step)
    if x0.noise is not None and sim.measurement_noise_level > 0:
        x0.voltage = x0.voltage + x0.noise
    regularizer.reset_iteration()
    regul_loss = regularizer.compute(
        model=model, x=x0, in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
    )
    regul_value = regul_loss.item()
    loss = regul_loss.clone()

    # Launch each start independently
    for s in range(time_step):
        start_k = T - time_step + s  # start frame
        n_steps = time_step - s       # rollout length

        x = x_ts.frame(start_k)
        if x.noise is not None and sim.measurement_noise_level > 0:
            x.voltage = x.voltage + x.noise

        if torch.isnan(x.voltage).any():
            continue

        if has_visual_field:
            vi = model.forward_visual(x, start_k)
            x.stimulus[:model.n_input_neurons] = vi.squeeze(-1)
            x.stimulus[model.n_input_neurons:] = 0

        data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

        # Unroll n_steps forward
        for step in range(n_steps):
            batched_state, batched_edges = _batch_frames([x], edges)
            pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

            if s == 0 and step == 0:
                update_regul = regularizer.compute_update_regul(model, in_features, ids, device)
                loss = loss + update_regul

            x.voltage = (x.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)).squeeze(-1)

            # Update stimulus for next step
            k_next = start_k + step + 1
            if k_next < x_ts.n_frames:
                if has_visual_field:
                    vi = model.forward_visual(x, k_next)
                    x.stimulus[:model.n_input_neurons] = vi.squeeze(-1)
                    x.stimulus[model.n_input_neurons:] = 0
                else:
                    x.stimulus = x_ts.frame(k_next).stimulus

        # Loss: predicted voltage vs target at T
        pred_v = x.voltage.unsqueeze(-1)
        loss = loss + ((pred_v[ids] - y_target[ids]) / (sim.delta_t * time_step)).norm(2)

    # Average over the time_step starts
    loss = loss / time_step
    return loss, regul_value
