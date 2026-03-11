"""Pure metrics computation for FlyVis — no matplotlib dependency.

Contains the connectivity R² pipeline (slope correction, grad_msg,
corrected weights) and derived quantities (tau, V_rest).

Used by:
    - plot.py (re-exports for backward compatibility)
    - GNN_PlotFigure.py (post-training analysis)
    - graph_trainer.py (training-time monitoring)
    - sparsify.py (pruning)
"""
import os
from typing import Optional

import numpy as np
import torch
from scipy.optimize import curve_fit

from flyvis_gnn.fitting_models import linear_model
from flyvis_gnn.utils import graphs_data_path, to_numpy

# ------------------------------------------------------------------ #
#  Neuron type constants
# ------------------------------------------------------------------ #

INDEX_TO_NAME: dict[int, str] = {
    0: 'am', 1: 'c2', 2: 'c3', 3: 'ct1(lo1)', 4: 'ct1(m10)',
    5: 'l1', 6: 'l2', 7: 'l3', 8: 'l4', 9: 'l5',
    10: 'lawf1', 11: 'lawf2', 12: 'mi1', 13: 'mi10', 14: 'mi11',
    15: 'mi12', 16: 'mi13', 17: 'mi14', 18: 'mi15', 19: 'mi2',
    20: 'mi3', 21: 'mi4', 22: 'mi9', 23: 'r1', 24: 'r2',
    25: 'r3', 26: 'r4', 27: 'r5', 28: 'r6', 29: 'r7', 30: 'r8',
    31: 't1', 32: 't2', 33: 't2a', 34: 't3', 35: 't4a',
    36: 't4b', 37: 't4c', 38: 't4d', 39: 't5a', 40: 't5b',
    41: 't5c', 42: 't5d', 43: 'tm1', 44: 'tm16', 45: 'tm2',
    46: 'tm20', 47: 'tm28', 48: 'tm3', 49: 'tm30', 50: 'tm4',
    51: 'tm5y', 52: 'tm5a', 53: 'tm5b', 54: 'tm5c', 55: 'tm9',
    56: 'tmy10', 57: 'tmy13', 58: 'tmy14', 59: 'tmy15',
    60: 'tmy18', 61: 'tmy3', 62: 'tmy4', 63: 'tmy5a', 64: 'tmy9',
}

ANATOMICAL_ORDER: list[Optional[int]] = [
    None, 23, 24, 25, 26, 27, 28, 29, 30,
    5, 6, 7, 8, 9, 10, 11, 12,
    19, 20, 21, 22,
    13, 14, 15, 16, 17, 18,
    43, 45, 48, 50, 44, 46, 47, 49, 51, 52, 53, 54, 55,
    61, 62, 63, 56, 57, 58, 59, 60, 64,
    1, 2, 4, 3,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    0,
]


# ------------------------------------------------------------------ #
#  Weight extraction
# ------------------------------------------------------------------ #

def get_model_W(model) -> torch.Tensor:
    """Get the weight matrix from a model, handling low-rank factorization."""
    if hasattr(model, 'W'):
        return model.W
    elif hasattr(model, 'WL') and hasattr(model, 'WR'):
        return model.WL @ model.WR
    else:
        raise AttributeError("Model has neither 'W' nor 'WL'/'WR' attributes")


# ------------------------------------------------------------------ #
#  R² computation
# ------------------------------------------------------------------ #

def compute_r_squared(true: np.ndarray, learned: np.ndarray) -> tuple[float, float]:
    """Compute R² and linear fit slope between true and learned arrays."""
    lin_fit, _ = curve_fit(linear_model, true, learned)
    residuals = learned - linear_model(true, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((learned - np.mean(learned)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r_squared, lin_fit[0]


def compute_r_squared_filtered(true: np.ndarray, learned: np.ndarray, outlier_threshold: float = 5.0) -> tuple[float, float, np.ndarray]:
    """Compute R² with outlier removal.

    Removes points where |learned - true| > outlier_threshold before
    computing the linear fit and R².

    Returns:
        r_squared: float.
        slope: float.
        inlier_mask: (N,) bool array — True for inliers.
    """
    residuals = learned - true
    mask = np.abs(residuals) <= outlier_threshold
    true_in = true[mask]
    learned_in = learned[mask]
    r_squared, slope = compute_r_squared(true_in, learned_in)
    return r_squared, slope, mask


# ------------------------------------------------------------------ #
#  Vectorized helpers
# ------------------------------------------------------------------ #

def _vectorized_linspace(starts: np.ndarray, ends: np.ndarray, n_pts: int, device: torch.device) -> torch.Tensor:
    """Create (N, n_pts) tensor where row n spans [starts[n], ends[n]].

    Instead of calling torch.linspace N times, we parameterize with
    t in [0, 1] and broadcast:  rr[n, i] = start[n] + t[i] * (end[n] - start[n])
    """
    t = torch.linspace(0, 1, n_pts, device=device)                   # (n_pts,)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)  # (N,)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)      # (N,)
    return starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]    # (N, n_pts)


def _batched_mlp_eval(mlp, model_a, rr, build_features_fn,
                      device, chunk_size=2000, post_fn=None):
    """Evaluate an MLP for all neurons at once, in chunks.

    Instead of N individual forward passes on (1000, D) inputs, we
    stack all neurons into (N*1000, D) and run one pass per chunk.

    Args:
        mlp: nn.Module — the MLP to evaluate (model.g_phi or model.f_theta).
        model_a: (N, emb_dim) embedding tensor.
        rr: (N, n_pts) tensor of input values per neuron.
        build_features_fn: callable(rr_flat, emb_flat) -> (chunk*n_pts, D)
            Builds the MLP input features from flattened rr and embeddings.
        device: torch device.
        chunk_size: number of neurons per chunk (limits GPU memory).
        post_fn: optional callable applied to MLP output (e.g. lambda x: x**2).

    Returns:
        (N, n_pts) tensor of MLP outputs.
    """
    N, n_pts = rr.shape
    emb_dim = model_a.shape[1]
    results = []

    for i in range(0, N, chunk_size):
        chunk_rr = rr[i:i + chunk_size]                        # (C, n_pts)
        chunk_a = model_a[i:i + chunk_size]                     # (C, emb_dim)
        C = chunk_rr.shape[0]

        # Flatten: repeat each neuron's values n_pts times
        rr_flat = chunk_rr.reshape(-1, 1)                       # (C*n_pts, 1)
        emb_flat = chunk_a[:, None, :].expand(-1, n_pts, -1)    # (C, n_pts, emb_dim)
        emb_flat = emb_flat.reshape(-1, emb_dim)                 # (C*n_pts, emb_dim)

        in_features = build_features_fn(rr_flat, emb_flat)       # (C*n_pts, D)

        with torch.no_grad():
            out = mlp(in_features.float())                       # (C*n_pts, 1)
            if post_fn is not None:
                out = post_fn(out)

        results.append(out.squeeze(-1).reshape(C, n_pts))        # (C, n_pts)

    return torch.cat(results, dim=0)                              # (N, n_pts)


def _vectorized_linear_fit(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized least-squares linear regression across rows.

    Fits y[n] = slope[n] * x[n] + offset[n] for all N rows in parallel,
    replacing N individual scipy.curve_fit calls.

    Uses the closed-form solution:
        slope  = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
        offset = (Σy − slope·Σx) / n

    Args:
        x: (N, n_pts) numpy array or tensor.
        y: (N, n_pts) numpy array or tensor.

    Returns:
        slopes: (N,) numpy array.
        offsets: (N,) numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    if isinstance(y, torch.Tensor):
        y = to_numpy(y)

    n_pts = x.shape[1]
    sx = x.sum(axis=1)
    sy = y.sum(axis=1)
    sxy = (x * y).sum(axis=1)
    sxx = (x * x).sum(axis=1)

    denom = n_pts * sxx - sx * sx
    # Guard against degenerate cases (constant x)
    safe = np.abs(denom) > 1e-12
    slopes = np.where(safe, (n_pts * sxy - sx * sy) / np.where(safe, denom, 1.0), 0.0)
    offsets = np.where(safe, (sy - slopes * sx) / n_pts, 0.0)

    return slopes, offsets


# ------------------------------------------------------------------ #
#  Feature-building helpers for the two MLPs
# ------------------------------------------------------------------ #

def _build_g_phi_features(rr_flat, emb_flat, signal_model_name):
    """Build input features for g_phi MLP."""
    if 'flyvis_B' in signal_model_name:
        return torch.cat([rr_flat * 0, rr_flat, emb_flat, emb_flat], dim=1)
    else:
        return torch.cat([rr_flat, emb_flat], dim=1)


def _build_f_theta_features(rr_flat, emb_flat):
    """Build input features for f_theta MLP: (v, embedding, msg=0, exc=0)."""
    zeros = torch.zeros_like(rr_flat)
    return torch.cat([rr_flat, emb_flat, zeros, zeros], dim=1)


# ------------------------------------------------------------------ #
#  Activity statistics
# ------------------------------------------------------------------ #

def compute_activity_stats(x_ts, device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-neuron mean and std of voltage activity.

    Args:
        x_ts: NeuronTimeSeries (voltage field is (T, N) tensor).
        device: optional device override.

    Returns:
        mu_activity: (N,) tensor of per-neuron mean voltage.
        sigma_activity: (N,) tensor of per-neuron std voltage.
    """
    voltage = x_ts.voltage  # (T, N), already on device if x_ts was moved
    if device is not None:
        voltage = voltage.to(device)
    mu = voltage.mean(dim=0)
    sigma = voltage.std(dim=0)
    return mu, sigma


# ------------------------------------------------------------------ #
#  Slope extraction
# ------------------------------------------------------------------ #

def extract_g_phi_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope of g_phi for each neuron j (vectorized).

    Evaluates g_phi(a_j, v) over each neuron's activity range [mu-2σ, mu+2σ]
    in one batched forward pass, then fits all slopes with vectorized regression.

    Returns:
        slopes: (n_neurons,) numpy array of g_phi slopes.
    """
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    n_pts = 1000

    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

    # Neurons where activity range includes positive values
    valid = (mu + sigma) > 0
    starts = np.maximum(mu - 2 * sigma, 0.0)
    ends = mu + 2 * sigma

    # For invalid neurons, set dummy range (won't be used)
    starts[~valid] = 0.0
    ends[~valid] = 1.0

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    post_fn = (lambda x: x ** 2) if g_phi_positive else None
    build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, signal_model_name)

    func = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr,
                             build_fn, device, post_fn=post_fn)  # (N, n_pts)

    slopes, _ = _vectorized_linear_fit(rr, func)

    # Invalid neurons get slope = 1.0
    slopes[~valid] = 1.0

    return slopes


def extract_f_theta_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope and offset of f_theta for each neuron i (vectorized).

    Evaluates f_theta(a_i, v_i, msg=0, exc=0) over each neuron's activity range
    in one batched forward pass, then fits all slopes/offsets with vectorized regression.

    Returns:
        slopes: (n_neurons,) numpy array — slope relates to 1/tau.
        offsets: (n_neurons,) numpy array — offset relates to V_rest.
    """
    n_pts = 1000
    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

    starts = mu - 2 * sigma
    ends = mu + 2 * sigma

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    func = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr,
                             lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f),
                             device)  # (N, n_pts)

    slopes, offsets = _vectorized_linear_fit(rr, func)

    return slopes, offsets


# ------------------------------------------------------------------ #
#  Derived quantities from f_theta slopes
# ------------------------------------------------------------------ #

def derive_tau(slopes_f_theta: np.ndarray, n_neurons: int) -> np.ndarray:
    """Convert f_theta slopes to learned tau: tau = 1/(-slope), clipped to [0,1].

    Args:
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        n_neurons: number of neurons to use.

    Returns:
        learned_tau: (n_neurons,) numpy array.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        learned_tau = np.where(slopes_f_theta != 0, 1.0 / -slopes_f_theta, 1.0)[:n_neurons]
    return np.clip(learned_tau, 0, 1)


def derive_vrest(slopes_f_theta: np.ndarray, offsets_f_theta: np.ndarray, n_neurons: int) -> np.ndarray:
    """Convert f_theta slopes/offsets to learned V_rest: V_rest = -offset/slope.

    Args:
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        offsets_f_theta: (N,) numpy array of f_theta offsets.
        n_neurons: number of neurons to use.

    Returns:
        learned_V_rest: (n_neurons,) numpy array.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(slopes_f_theta != 0, -offsets_f_theta / slopes_f_theta, 1.0)[:n_neurons]


def _torch_linear_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable least-squares linear regression in pure torch.

    Same closed-form OLS as _vectorized_linear_fit, but operates on
    torch tensors with gradient tracking preserved through y.

    Args:
        x: (N, n_pts) tensor (no grad needed — voltage grid points).
        y: (N, n_pts) tensor (grad flows through here from f_theta).

    Returns:
        slopes: (N,) tensor.
        offsets: (N,) tensor.
    """
    n_pts = x.shape[1]
    sx = x.sum(dim=1)
    sy = y.sum(dim=1)
    sxy = (x * y).sum(dim=1)
    sxx = (x * x).sum(dim=1)

    denom = n_pts * sxx - sx * sx
    slopes = (n_pts * sxy - sx * sy) / (denom + 1e-12)
    offsets = (sy - slopes * sx) / n_pts

    return slopes, offsets


def compute_f_theta_linearity_loss(model, n_neurons: int, mu: np.ndarray, sigma: np.ndarray, device: torch.device, n_pts: int = 200) -> torch.Tensor:
    """Unsupervised f_theta linearity loss.

    Evaluates f_theta WITH gradient tracking, fits a differentiable OLS
    line through the outputs, and penalizes the residual (non-linear
    component). No ground-truth V_rest is needed.

    Physical motivation: the true neuron dynamics are leaky integrators
    (dv/dt = -(v - V_rest)/tau), which is linear in v. Penalizing
    f_theta's deviation from linearity is an inductive bias toward the
    correct physics, constraining the space of solutions so that
    V_rest = -offset/slope is more uniquely determined.

    Gradients flow through f_theta parameters only:
    - model.a (embeddings) is detached
    - rr (voltage grid) is constructed from cached data stats (no grad)

    Args:
        model: FlyVisGNN model with f_theta and a attributes.
        n_neurons: Number of neurons.
        mu: (N,) numpy array — per-neuron mean voltage.
        sigma: (N,) numpy array — per-neuron std voltage.
        device: Torch device.
        n_pts: Number of voltage grid points (default 200).

    Returns:
        Scalar mean-squared residual loss with gradient through f_theta.
    """
    starts = mu - 2 * sigma
    ends = mu + 2 * sigma

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts), no grad

    # Evaluate f_theta WITHOUT no_grad — gradient flows through f_theta weights
    emb_dim = model.a.shape[1]
    rr_flat = rr.reshape(-1, 1)                                          # (N*n_pts, 1)
    a_detached = model.a[:n_neurons].detach()                             # block grad to embeddings
    emb_flat = a_detached[:, None, :].expand(-1, n_pts, -1).reshape(-1, emb_dim)  # (N*n_pts, emb_dim)

    in_features = _build_f_theta_features(rr_flat, emb_flat)             # (N*n_pts, D)
    out = model.f_theta(in_features.float())                             # (N*n_pts, 1)
    func = out.squeeze(-1).reshape(n_neurons, n_pts)                     # (N, n_pts)

    # Differentiable OLS: fit a line through f_theta outputs
    slopes, offsets = _torch_linear_fit(rr, func)

    # Linear prediction: what f_theta WOULD output if it were perfectly linear
    linear_pred = slopes[:, None] * rr + offsets[:, None]                # (N, n_pts)

    # Residual: the non-linear component of f_theta
    residual = func - linear_pred                                        # (N, n_pts)

    # Mean squared residual across all neurons and points
    loss = (residual ** 2).mean()

    return loss


def compute_f_theta_centering_loss(
    model,
    n_neurons: int,
    mu: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Unsupervised f_theta centering loss — anchors V_rest toward mean voltage.

    Evaluates f_theta at (v=μ_i, a_i, msg=0, exc=0) for each neuron and
    penalizes the output magnitude. If f_theta is approximately linear
    (dv/dt ≈ -(v - V_rest)/tau), then f_theta(μ) = (V_rest - μ)/tau.
    Penalizing this pulls V_rest toward μ (the observed mean voltage),
    providing an unsupervised anchor for the zero-crossing location.

    Unlike the linearity loss (which constrains f_theta's *shape*),
    this constrains f_theta's *location* — where the zero-crossing falls.

    Cost: N f_theta evaluations (trivial — no voltage grid needed).

    Args:
        model: FlyVisGNN model with f_theta and a attributes.
        n_neurons: Number of neurons.
        mu: (N,) numpy array — per-neuron mean voltage.
        device: Torch device.

    Returns:
        Scalar MSE loss with gradient through f_theta.
    """
    mu_t = torch.tensor(mu[:n_neurons], dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)

    emb = model.a[:n_neurons].detach()                # (N, emb_dim) — block grad to embeddings
    zeros = torch.zeros(n_neurons, 1, device=device)  # msg=0, exc=0

    in_features = torch.cat([mu_t, emb, zeros, zeros], dim=1)  # (N, 1+emb_dim+1+1)
    out = model.f_theta(in_features.float())                    # (N, 1)

    # MSE: penalize f_theta output at mean voltage
    loss = (out ** 2).mean()

    return loss


# ------------------------------------------------------------------ #
#  Dynamics R² (V_rest and tau)
# ------------------------------------------------------------------ #

def compute_dynamics_r2(model, x_ts, config, device, n_neurons):
    """Compute V_rest R² and tau R² during training (lightweight, no plots).

    Extracts learned V_rest and tau from f_theta slopes/offsets and compares
    against ground truth V_i_rest.pt and taus.pt.

    Returns:
        (vrest_r2, tau_r2): tuple of float R² values.
    """
    from flyvis_gnn.generators.ode_params import FlyVisODEParams
    ode_params = FlyVisODEParams.load(graphs_data_path(config.dataset), device=device)
    gt_V_rest = to_numpy(ode_params.V_i_rest[:n_neurons])
    gt_tau = to_numpy(ode_params.tau_i[:n_neurons])

    mu, sigma = compute_activity_stats(x_ts, device)
    slopes, offsets = extract_f_theta_slopes(model, config, n_neurons, mu, sigma, device)

    learned_V_rest = derive_vrest(slopes, offsets, n_neurons)
    learned_tau = derive_tau(slopes, n_neurons)

    try:
        vrest_r2, _ = compute_r_squared(gt_V_rest, learned_V_rest)
    except Exception:
        vrest_r2 = 0.0
    try:
        tau_r2, _ = compute_r_squared(gt_tau, learned_tau)
    except Exception:
        tau_r2 = 0.0

    return vrest_r2, tau_r2


def compute_dynamics_r2_linear(model, config, device, n_neurons):
    """Compute V_rest R² and tau R² for FlyVisLinear (direct parameter comparison).

    Unlike GNN models where tau and V_rest must be extracted from f_theta
    slopes, the linear model exposes them as direct learnable parameters.

    Returns:
        (vrest_r2, tau_r2, connectivity_r2): tuple of float R² values.
    """
    import torch.nn.functional as F

    from flyvis_gnn.generators.ode_params import FlyVisODEParams
    ode_params = FlyVisODEParams.load(graphs_data_path(config.dataset), device=device)
    gt_V_rest = to_numpy(ode_params.V_i_rest[:n_neurons])
    gt_tau = to_numpy(ode_params.tau_i[:n_neurons])
    gt_weights = to_numpy(ode_params.W)

    learned_tau = to_numpy(F.softplus(model.raw_tau[:n_neurons]).detach())
    learned_vrest = to_numpy(model.V_rest[:n_neurons].detach())
    learned_W = to_numpy(get_model_W(model).squeeze())

    try:
        vrest_r2, _ = compute_r_squared(gt_V_rest, learned_vrest)
    except Exception:
        vrest_r2 = 0.0
    try:
        tau_r2, _ = compute_r_squared(gt_tau, learned_tau)
    except Exception:
        tau_r2 = 0.0
    try:
        conn_r2, _ = compute_r_squared(gt_weights, learned_W)
    except Exception:
        conn_r2 = 0.0

    return vrest_r2, tau_r2, conn_r2


# ------------------------------------------------------------------ #
#  Gradient of f_theta w.r.t. msg
# ------------------------------------------------------------------ #

def compute_grad_msg(model, in_features, config):
    """Compute d(f_theta)/d(msg) for each neuron from a forward-pass in_features.

    Args:
        model: FlyVisGNN model.
        in_features: (N, D) tensor from model(..., return_all=True).
            Layout: [v(1), embedding(E), msg(1), excitation(1)].
        config: config object with graph_model.embedding_dim.

    Returns:
        grad_msg: (N,) tensor of gradients.
    """
    emb_dim = config.graph_model.embedding_dim
    v = in_features[:, 0:1].clone().detach()
    embedding = in_features[:, 1:1 + emb_dim].clone().detach()
    msg = in_features[:, 1 + emb_dim:2 + emb_dim].clone().detach()
    excitation = in_features[:, 2 + emb_dim:3 + emb_dim].clone().detach()

    msg.requires_grad_(True)
    in_features_grad = torch.cat([v, embedding, msg, excitation], dim=1)
    out = model.f_theta(in_features_grad)

    grad = torch.autograd.grad(
        outputs=out,
        inputs=msg,
        grad_outputs=torch.ones_like(out),
        retain_graph=False,
        create_graph=False,
    )[0]

    return grad.squeeze().detach()


# ------------------------------------------------------------------ #
#  Corrected weights
# ------------------------------------------------------------------ #

def compute_corrected_weights(model, edges, slopes_f_theta, slopes_g_phi, grad_msg):
    """Compute corrected W_ij from raw W, slopes, and grad_msg.

    Formula:
        corrected_W_ij = -W_ij / slope_phi[i] * grad_msg[i] * slope_edge[j]

    Args:
        model: model with .W, .n_edges, .n_extra_null_edges attributes.
        edges: (2, E) edge index tensor.
        slopes_f_theta: (N,) array/tensor of f_theta slopes per neuron.
        slopes_g_phi: (N,) array/tensor of g_phi slopes per neuron.
        grad_msg: (N,) tensor of d(f_theta)/d(msg) per neuron.

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
    """
    device = get_model_W(model).device

    # Convert to tensors if needed
    if not isinstance(slopes_f_theta, torch.Tensor):
        slopes_f_theta = torch.tensor(slopes_f_theta, dtype=torch.float32, device=device)
    if not isinstance(slopes_g_phi, torch.Tensor):
        slopes_g_phi = torch.tensor(slopes_g_phi, dtype=torch.float32, device=device)

    n_w = model.n_edges + model.n_extra_null_edges

    # Map edges to neuron indices (handles batched edges via modulo)
    target_neuron_ids = edges[1, :] % n_w   # i — post-synaptic
    prior_neuron_ids = edges[0, :] % n_w    # j — pre-synaptic

    slopes_phi_per_edge = slopes_f_theta[target_neuron_ids]     # (E,)
    slopes_edge_per_edge = slopes_g_phi[prior_neuron_ids]    # (E,)
    grad_msg_per_edge = grad_msg[target_neuron_ids]             # (E,)

    W = get_model_W(model)  # (E, 1)

    corrected_W = (-W
                   / slopes_phi_per_edge[:, None]
                   * grad_msg_per_edge.unsqueeze(1)
                   * slopes_edge_per_edge.unsqueeze(1))

    # Sanitize: division by near-zero slopes can produce inf/nan
    corrected_W = torch.nan_to_num(corrected_W, nan=0.0, posinf=0.0, neginf=0.0)

    return corrected_W


def compute_all_corrected_weights(model, config, edges, x_ts, device, n_grad_frames=8):
    """High-level: compute corrected W from model state and training data.

    Extracts slopes from g_phi and f_theta, computes grad_msg averaged
    over multiple frames, and applies the correction formula.

    Args:
        model: FlyVisGNN model.
        config: full config object.
        edges: (2, E) edge index tensor.
        x_ts: NeuronTimeSeries (training data).
        device: torch device.
        n_grad_frames: number of frames to sample for grad_msg (default 8).

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
        slopes_f_theta: (N,) numpy array.
        slopes_g_phi: (N,) numpy array.
        offsets_f_theta: (N,) numpy array.
    """
    n_neurons = model.a.shape[0]

    # 1. Activity statistics
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)

    # 2. Slope extraction
    slopes_g_phi = extract_g_phi_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)
    slopes_f_theta, offsets_f_theta = extract_f_theta_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    # 3. Compute grad_msg over multiple frames and take median
    n_frames = x_ts.voltage.shape[0]
    frame_indices = np.linspace(n_frames // 10, n_frames - 100, n_grad_frames, dtype=int)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()

    # Ensure edges are on the correct device
    edges = edges.to(device)

    grad_list = []
    for k in frame_indices:
        state = x_ts.frame(int(k)).to(device)
        with torch.no_grad():
            _, in_features, _ = model(state, edges, data_id=data_id, return_all=True)
        grad_k = compute_grad_msg(model, in_features, config)
        grad_list.append(grad_k)

    if was_training:
        model.train()

    grad_msg = torch.stack(grad_list).median(dim=0).values  # (N,)

    # 4. Corrected weights
    corrected_W = compute_corrected_weights(
        model, edges, slopes_f_theta, slopes_g_phi, grad_msg)

    return corrected_W, slopes_f_theta, slopes_g_phi, offsets_f_theta
