"""Plotting functions for FlyVis.

Used by the training loop (graph_trainer.py), data generation
(graph_data_generator.py), testing (graph_tester.py), and
post-training analysis (GNN_PlotFigure.py).

Metric computation lives in flyvis_gnn.metrics — re-exported here
for backward compatibility.
"""
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

from flyvis_gnn.fitting_models import linear_model

# Re-export all metrics functions for backward compatibility.
# Callers can import from either flyvis_gnn.metrics or flyvis_gnn.plot.
from flyvis_gnn.metrics import (  # noqa: F401
    ANATOMICAL_ORDER,
    INDEX_TO_NAME,
    _batched_mlp_eval,
    _build_f_theta_features,
    _build_g_phi_features,
    _vectorized_linear_fit,
    _vectorized_linspace,
    compute_activity_stats,
    compute_all_corrected_weights,
    compute_corrected_weights,
    compute_dynamics_r2,
    compute_grad_msg,
    compute_r_squared,
    compute_r_squared_filtered,
    derive_tau,
    derive_vrest,
    extract_f_theta_slopes,
    extract_g_phi_slopes,
    get_model_W,
)
from flyvis_gnn.utils import to_numpy

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def plot_training_summary_panels(fig, log_dir, Niter=None):
    """Add embedding, weight comparison, g_phi, and f_theta function panels to a summary figure.

    Finds the last saved training snapshot and loads the PNG images into subplots 2-5
    of a 2x3 grid figure.

    Args:
        fig: matplotlib Figure (expected 2x3 subplot layout, panel 1 already used for loss)
        log_dir: path to the training log directory
        Niter: iterations per epoch (for global iteration x-axis in R² panel)
    """
    import glob
    import os

    import imageio

    from flyvis_gnn.figure_style import default_style
    style = default_style

    embedding_files = glob.glob(f"{log_dir}/tmp_training/embedding/*.png")
    if not embedding_files:
        return

    last_file = max(embedding_files, key=os.path.getctime)
    filename = os.path.basename(last_file)
    last_epoch, last_N = filename.replace('.png', '').split('_')

    panels = [
        (2, f"{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.png", 'learned embedding'),
        (3, f"{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.png", 'weight comparison'),
        (4, f"{log_dir}/tmp_training/function/MLP1/func_{last_epoch}_{last_N}.png", r'$g_\phi$ (MLP1)'),
        (5, f"{log_dir}/tmp_training/function/MLP0/func_{last_epoch}_{last_N}.png", r'$f_\theta$ (MLP0)'),
    ]
    for pos, path, title in panels:
        fig.add_subplot(2, 3, pos)
        img = imageio.imread(path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=style.label_font_size)

    # Panel 6: R² metrics trajectory
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(metrics_log_path):
        r2_iters, conn_vals, vrest_vals, tau_vals = [], [], [], []
        try:
            with open(metrics_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('epoch', 'iteration')):
                        continue
                    parts = line.split(',')
                    r2_iters.append(int(parts[0]))
                    conn_vals.append(float(parts[1]))
                    vrest_vals.append(float(parts[2]) if len(parts) > 2 else 0.0)
                    tau_vals.append(float(parts[3]) if len(parts) > 3 else 0.0)
        except Exception:
            pass
        if conn_vals:
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.plot(r2_iters, conn_vals, color='#d62728', linewidth=style.line_width, label='conn')
            ax6.plot(r2_iters, vrest_vals, color='#1f77b4', linewidth=style.line_width, label=r'$V_{rest}$')
            ax6.plot(r2_iters, tau_vals, color='#2ca02c', linewidth=style.line_width, label=r'$\tau$')
            ax6.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax6.set_ylim(-0.05, 1.05)
            style.xlabel(ax6, 'iteration')
            style.ylabel(ax6, r'$R^2$')
            ax6.set_title(r'$R^2$ metrics', fontsize=style.label_font_size)
            ax6.legend(fontsize=style.annotation_font_size, loc='lower right')
            ax6.grid(True, alpha=0.3)



def _plot_curves_fast(ax, rr, func, type_list, cmap, linewidth=1, alpha=0.1):
    """Plot per-neuron curves using LineCollection (single draw call).

    Instead of N individual ax.plot() calls (high matplotlib overhead),
    build an (N, n_pts, 2) segments array and add one LineCollection.

    Args:
        ax: matplotlib Axes.
        rr: (N, n_pts) or (n_pts,) numpy array of x-values.
        func: (N, n_pts) numpy array of y-values.
        type_list: (N,) int array of neuron type indices.
        cmap: CustomColorMap with .color(int) method.
        linewidth: line width.
        alpha: transparency.
    """
    N, n_pts = func.shape

    # If rr is 1D (shared range), broadcast to (N, n_pts)
    if rr.ndim == 1:
        rr = np.broadcast_to(rr[None, :], (N, n_pts))

    # Build (N, n_pts, 2) segments array: each row is [(x0,y0), (x1,y1), ...]
    segments = np.stack([rr, func], axis=-1)                  # (N, n_pts, 2)

    # Build per-neuron RGBA color array
    type_np = np.asarray(type_list).astype(int).ravel()
    colors = [(*cmap.color(type_np[n])[:3], alpha) for n in range(N)]

    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)
    ax.autoscale_view()






# ------------------------------------------------------------------ #
#  Subplot functions — shared between training and GNN_PlotFigure
# ------------------------------------------------------------------ #

def plot_embedding(ax, model, type_list, n_types, cmap):
    """Plot embedding scatter colored by neuron type.

    Args:
        ax: matplotlib Axes.
        model: model with .a embedding tensor (N, emb_dim).
        type_list: (N,) tensor/array of integer type indices.
        n_types: number of neuron types.
        cmap: CustomColorMap with .color(int) method.
    """
    embedding = to_numpy(model.a)
    type_np = to_numpy(type_list).squeeze()

    for n in range(n_types):
        mask = (type_np == n)
        if np.any(mask):
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=cmap.color(n), s=6, alpha=0.25, edgecolors='none')

    ax.set_xlabel('$a_0$', fontsize=32)
    ax.set_ylabel('$a_1$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def plot_f_theta(ax, model, config, n_neurons, type_list, cmap, device, step=20):
    """Plot f_theta function curves colored by neuron type (vectorized).

    Evaluates all selected neurons in one batched MLP pass and draws
    all curves with a single LineCollection.
    """
    n_pts = 1000
    xlim = config.plotting.xlim

    # Select every step-th neuron
    neuron_ids = np.arange(0, n_neurons, step)
    n_sel = len(neuron_ids)

    # Shared x-range, expanded to (n_sel, n_pts)
    rr_1d = torch.linspace(xlim[0], xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)  # (n_sel, n_pts)

    # Batched MLP evaluation
    func = _batched_mlp_eval(
        model.f_theta, model.a[neuron_ids], rr,
        lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f),
        device)

    # Fast plot with LineCollection
    type_np = to_numpy(type_list).astype(int)
    _plot_curves_fast(ax, to_numpy(rr_1d), to_numpy(func),
                      type_np[neuron_ids], cmap, linewidth=1, alpha=0.2)

    ax.set_xlim(xlim)
    ax.set_ylim(config.plotting.ylim)
    ax.set_xlabel('$v_i$', fontsize=32)
    ax.set_ylabel(r'learned $f_\theta(\mathbf{a}_i, v_i)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_g_phi(ax, model, config, n_neurons, type_list, cmap, device, step=20):
    """Plot g_phi function curves colored by neuron type (vectorized).

    Evaluates all selected neurons in one batched MLP pass and draws
    all curves with a single LineCollection.
    """
    model_config = config.graph_model
    n_pts = 1000

    neuron_ids = np.arange(0, n_neurons, step)
    n_sel = len(neuron_ids)

    rr_1d = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)

    post_fn = (lambda x: x ** 2) if model_config.g_phi_positive else None
    build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, model_config.signal_model_name)

    func = _batched_mlp_eval(
        model.g_phi, model.a[neuron_ids], rr,
        build_fn, device, post_fn=post_fn)

    type_np = to_numpy(type_list).astype(int)
    _plot_curves_fast(ax, to_numpy(rr_1d), to_numpy(func),
                      type_np[neuron_ids], cmap, linewidth=1, alpha=0.2)

    ax.set_xlim(config.plotting.xlim)
    ax.set_ylim([-config.plotting.xlim[1] / 10, config.plotting.xlim[1] * 1.2])
    ax.set_xlabel('$v_j$', fontsize=32)
    ax.set_ylabel(r'learned $g_\phi(\mathbf{a}_j, v_j)$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_weight_scatter(ax, gt_weights, learned_weights, corrected=False,
                        xlim=None, ylim=None, mc=None, scatter_size=0.5,
                        outlier_threshold=None):
    """Plot true vs learned weight scatter with R² and slope.

    Args:
        ax: matplotlib Axes.
        gt_weights: (E,) numpy array of ground truth weights.
        learned_weights: (E,) numpy array of learned (or corrected) weights.
        corrected: if True, use W* label; if False, use W label.
        xlim: optional (lo, hi) for x-axis.
        ylim: optional (lo, hi) for y-axis.
        mc: per-edge color array; if None, uses black.
        scatter_size: scatter point size (default 0.5).
        outlier_threshold: if set, remove points with |residual| > threshold.
    """
    if outlier_threshold is not None:
        residuals = learned_weights - gt_weights
        mask = np.abs(residuals) <= outlier_threshold
        true_in = gt_weights[mask]
        learned_in = learned_weights[mask]
        mc_in = mc[mask] if mc is not None else None
    else:
        true_in = gt_weights
        learned_in = learned_weights
        mc_in = mc

    r_squared, slope = compute_r_squared(true_in, learned_in)

    scatter_color = mc_in if mc_in is not None else 'k'
    ax.scatter(true_in, learned_in, s=scatter_size, c=scatter_color, alpha=0.04)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(true_in)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)

    ylabel = r'learned $W_{ij}^*$' if corrected else r'learned $W_{ij}$'
    ax.set_xlabel(r'true $W_{ij}$', fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared, slope


def plot_tau(ax, slopes_f_theta, gt_taus, n_neurons, mc=None):
    """Plot learned tau vs ground truth tau.

    Args:
        ax: matplotlib Axes.
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        gt_taus: (N,) tensor/array of ground truth taus.
        n_neurons: number of neurons.
        mc: color for scatter points.
    """
    learned_tau = np.where(slopes_f_theta != 0, 1.0 / -slopes_f_theta, 1.0)
    learned_tau = learned_tau[:n_neurons]
    learned_tau = np.clip(learned_tau, 0, 1)
    gt_taus_np = to_numpy(gt_taus[:n_neurons]) if torch.is_tensor(gt_taus) else np.asarray(gt_taus[:n_neurons])

    r_squared, slope = compute_r_squared(gt_taus_np, learned_tau)

    ax.scatter(gt_taus_np, learned_tau, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(gt_taus_np)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel(r'true $\tau$', fontsize=32)
    ax.set_ylabel(r'learned $\tau$', fontsize=32)
    ax.set_xlim([0, 0.35])
    ax.set_ylim([0, 0.35])
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared


def plot_vrest(ax, slopes_f_theta, offsets_f_theta, gt_V_rest, n_neurons, mc=None):
    """Plot learned V_rest vs ground truth V_rest.

    Args:
        ax: matplotlib Axes.
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        offsets_f_theta: (N,) numpy array of f_theta offsets.
        gt_V_rest: (N,) tensor/array of ground truth V_rest.
        n_neurons: number of neurons.
        mc: color for scatter points.
    """
    learned_V_rest = np.where(slopes_f_theta != 0, -offsets_f_theta / slopes_f_theta, 1.0)
    gt_vr_np = to_numpy(gt_V_rest[:n_neurons]) if torch.is_tensor(gt_V_rest) else np.asarray(gt_V_rest[:n_neurons])

    r_squared, slope = compute_r_squared(gt_vr_np, learned_V_rest)

    ax.scatter(gt_vr_np, learned_V_rest, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(gt_vr_np)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel(r'true $V_{rest}$', fontsize=32)
    ax.set_ylabel(r'learned $V_{rest}$', fontsize=32)
    ax.set_xlim([-0.05, 0.9])
    ax.set_ylim([-0.05, 0.9])
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared


# ================================================================== #
#  CONSOLIDATED FROM generators/plots.py
# ================================================================== #

from typing import Optional

from flyvis_gnn.figure_style import FigureStyle, default_style


def plot_spatial_activity_grid(
    positions: np.ndarray,
    voltages: np.ndarray,
    stimulus: np.ndarray,
    neuron_types: np.ndarray,
    output_path: str,
    calcium: Optional[np.ndarray] = None,
    n_input_neurons: Optional[int] = None,
    index_to_name: Optional[dict] = None,
    anatomical_order: Optional[list] = None,
    style: FigureStyle = default_style,
) -> None:
    """8x9 or 16x9 hex scatter grid of per-neuron-type spatial activity.

    Args:
        positions: (N, 2) spatial positions for hex scatter.
        voltages: (N,) voltage per neuron.
        stimulus: (n_input,) stimulus values for input neurons.
        neuron_types: (N,) integer neuron type per neuron.
        output_path: where to save the figure.
        calcium: (N,) calcium values (if not None, adds bottom 8 rows).
        n_input_neurons: number of input neurons (defaults to len(stimulus)).
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        anatomical_order: panel ordering. Defaults to ANATOMICAL_ORDER.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    order = anatomical_order or ANATOMICAL_ORDER
    n_inp = n_input_neurons or len(stimulus)
    include_calcium = calcium is not None

    n_cols = 9
    n_rows = 16 if include_calcium else 8
    panel_w, panel_h = 2.0, 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        facecolor=style.background,
    )
    plt.subplots_adjust(hspace=1.2)
    axes_flat = axes.flatten()

    # hide trailing panels in voltage section
    n_panels = len(order)
    for i in range(n_panels, n_cols * 8):
        if i < len(axes_flat):
            axes_flat[i].set_visible(False)
    if include_calcium:
        for i in range(n_panels + n_cols * 8, len(axes_flat)):
            axes_flat[i].set_visible(False)

    vmin_v, vmax_v = style.hex_voltage_range
    vmin_s, vmax_s = style.hex_stimulus_range
    vmin_ca, vmax_ca = style.hex_calcium_range

    for panel_idx, type_idx in enumerate(order):
        # --- voltage panel ---
        ax_v = axes_flat[panel_idx]
        _draw_hex_panel(
            ax_v, type_idx, positions, voltages, stimulus,
            neuron_types, n_inp, names,
            cmap=style.cmap, vmin=vmin_v, vmax=vmax_v,
            stim_cmap=style.cmap, stim_vmin=vmin_s, stim_vmax=vmax_s,
            style=style,
        )

        # --- calcium panel (if present) ---
        if include_calcium:
            ax_ca = axes_flat[panel_idx + n_cols * 8]
            if type_idx is None:
                # stimulus panel (same as voltage section)
                ax_ca.scatter(
                    positions[:n_inp, 0], positions[:n_inp, 1],
                    s=style.hex_stimulus_marker_size, c=stimulus,
                    cmap=style.cmap, vmin=vmin_s, vmax=vmax_s,
                    marker=style.hex_marker, alpha=1.0, linewidths=0,
                )
                ax_ca.set_title(style._label('stimuli'), fontsize=style.font_size)
            else:
                mask = neuron_types == type_idx
                count = int(np.sum(mask))
                name = names.get(type_idx, f'type_{type_idx}')
                if count > 0:
                    ax_ca.scatter(
                        positions[:count, 0], positions[:count, 1],
                        s=style.hex_marker_size, c=calcium[mask],
                        cmap=style.cmap_calcium, vmin=vmin_ca, vmax=vmax_ca,
                        marker=style.hex_marker, alpha=1, linewidths=0,
                    )
                ax_ca.set_title(style._label(name), fontsize=style.font_size)
            ax_ca.set_facecolor(style.background)
            ax_ca.set_xticks([])
            ax_ca.set_yticks([])
            ax_ca.set_aspect('equal')
            for spine in ax_ca.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if not include_calcium else 0.92, bottom=0.05)
    style.savefig(fig, output_path)


def plot_kinograph(
    activity: np.ndarray,
    stimulus: np.ndarray,
    output_path: str,
    rank_90_act: int = 0,
    rank_99_act: int = 0,
    rank_90_inp: int = 0,
    rank_99_inp: int = 0,
    zoom_size: int = 200,
    zoom_neuron_start: int = 4900,
    style: FigureStyle = default_style,
) -> None:
    """2x2 kinograph: full activity + zoom, full stimulus + zoom.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        stimulus: (n_input_neurons, n_frames) transposed stimulus array.
        output_path: where to save the figure.
        rank_90_act: effective rank at 90% variance (activity).
        rank_99_act: effective rank at 99% variance (activity).
        rank_90_inp: effective rank at 90% variance (input).
        rank_99_inp: effective rank at 99% variance (input).
        zoom_size: size of zoom window in neurons and frames.
        zoom_neuron_start: first neuron index for the activity zoom panel.
        style: FigureStyle instance.
    """
    n_neurons, n_frames = activity.shape
    n_input, _ = stimulus.shape
    vmax_act = np.abs(activity).max()
    vmax_inp = np.abs(stimulus).max() * 1.2
    zoom_f = min(zoom_size, n_frames)
    zoom_n_act = min(zoom_size, n_neurons - zoom_neuron_start)
    zoom_n_inp = min(zoom_size, n_input)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(style.figure_height * 3.5, style.figure_height * 2.5),
        facecolor=style.background,
        gridspec_kw={'width_ratios': [2, 1]},
    )

    imshow_kw = dict(aspect='auto', cmap=style.cmap, origin='lower', interpolation='nearest')

    # top-left: full activity
    ax = axes[0, 0]
    im = ax.imshow(activity, vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, n_frames - 1])
    ax.set_xticklabels([0, n_frames], fontsize=style.tick_font_size)
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels([1, n_neurons], fontsize=style.tick_font_size)
    style.annotate(ax, f'rank(90%)={rank_90_act}  rank(99%)={rank_99_act}', (0.02, 0.97), va='top', ha='left')

    # top-right: zoom activity
    ax = axes[0, 1]
    zoom_neuron_end = zoom_neuron_start + zoom_n_act
    im = ax.imshow(activity[zoom_neuron_start:zoom_neuron_end, :zoom_f], vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_act - 1])
    ax.set_yticklabels([zoom_neuron_start, zoom_neuron_end], fontsize=style.tick_font_size)

    # bottom-left: full stimulus
    ax = axes[1, 0]
    im = ax.imshow(stimulus, vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'input neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, stimulus.shape[1] - 1])
    ax.set_xticklabels([0, stimulus.shape[1]], fontsize=style.tick_font_size)
    ax.set_yticks([0, n_input - 1])
    ax.set_yticklabels([1, n_input], fontsize=style.tick_font_size)
    style.annotate(ax, f'rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}', (0.02, 0.97), va='top', ha='left')

    # bottom-right: zoom stimulus
    ax = axes[1, 1]
    im = ax.imshow(stimulus[:zoom_n_inp, :zoom_f], vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'input neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_inp - 1])
    ax.set_yticklabels([1, zoom_n_inp], fontsize=style.tick_font_size)

    plt.tight_layout()
    style.savefig(fig, output_path)


def plot_activity_traces(
    activity: np.ndarray,
    output_path: str,
    n_traces: int = 100,
    max_frames: int = 10000,
    n_input_neurons: int = 0,
    style: FigureStyle = default_style,
    neuron_indices: np.ndarray | None = None,
    dpi: int | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Sampled neuron voltage traces stacked vertically.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        output_path: where to save the figure.
        n_traces: number of neurons to sample.
        max_frames: truncate x-axis at this frame count.
        n_input_neurons: shown as annotation.
        style: FigureStyle instance.
        neuron_indices: pre-selected neuron indices; if None, random sample.
        dpi: override DPI for this figure; if None, use style default.
        title: optional title for the figure.

    Returns:
        neuron_indices used (for reuse in paired plots).
    """
    n_neurons, n_frames = activity.shape
    n_traces = min(n_traces, n_neurons)
    if neuron_indices is None:
        neuron_indices = np.sort(np.random.choice(n_neurons, n_traces, replace=False))
    sampled = activity[neuron_indices]
    offset = sampled + 2 * np.arange(len(neuron_indices))[:, None]

    fig, ax = style.figure(aspect=1.5)
    ax.plot(offset.T, linewidth=0.5, alpha=0.7, color=style.foreground)
    style.xlabel(ax, 'time (frames)', fontsize=16)
    style.ylabel(ax, f'{len(neuron_indices)} / {n_neurons} neurons')
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlim([0, min(n_frames, max_frames)])
    ax.set_ylim([offset[0].min() - 2, offset[-1].max() + 2])
    if title:
        ax.set_title(title, fontsize=style.font_size)

    plt.tight_layout()
    save_kwargs = {}
    if dpi is not None:
        save_kwargs['dpi'] = dpi
    style.savefig(fig, output_path, **save_kwargs)
    return neuron_indices


def plot_selected_neuron_traces(
    activity: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    selected_types: Optional[list[int]] = None,
    start_frame: int = 63000,
    end_frame: int = 63500,
    index_to_name: Optional[dict] = None,
    step_v: float = 1.5,
    style: FigureStyle = default_style,
) -> None:
    """Traces for specific neuron types over a time window.

    Args:
        activity: (n_neurons, n_frames) full activity array.
        type_list: (n_neurons,) integer neuron type per neuron.
        output_path: where to save the figure.
        selected_types: list of type indices to plot. Defaults to
            [l1, mi1, mi2, r1, t1, t4a, t5a, tm1, tm4, tm9].
        start_frame: start of time window.
        end_frame: end of time window.
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        step_v: vertical offset between traces.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    if selected_types is None:
        selected_types = [5, 12, 19, 23, 31, 35, 39, 43, 50, 55]

    # find one neuron per selected type
    neuron_indices = []
    for stype in selected_types:
        indices = np.where(type_list == stype)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])

    n_sel = len(neuron_indices)
    if n_sel == 0:
        return

    true_slice = activity[neuron_indices, start_frame:end_frame]

    fig, ax = style.figure(aspect=1.5)
    for i in range(n_sel):
        baseline = np.mean(true_slice[i])
        ax.plot(true_slice[i] - baseline + i * step_v,
                linewidth=style.line_width, c='green', alpha=0.75)

    # neuron ids as y-tick labels
    ytick_positions = [i * step_v for i in range(n_sel)]
    ytick_labels = [names.get(selected_types[i], f'type_{selected_types[i]}') for i in range(n_sel)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=style.tick_font_size)
    ax.set_ylim([-step_v, n_sel * step_v])
    style.ylabel(ax, 'neuron')

    ax.set_xticks([0, end_frame - start_frame])
    ax.set_xticklabels([start_frame, end_frame], fontsize=14)
    style.xlabel(ax, 'time (frames)', fontsize=16)

    plt.tight_layout()
    style.savefig(fig, output_path)


# --------------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------------- #

def _draw_hex_panel(
    ax, type_idx, positions, voltages, stimulus, neuron_types,
    n_input_neurons, names, cmap, vmin, vmax,
    stim_cmap, stim_vmin, stim_vmax, style,
):
    """Draw a single hex scatter panel (voltage or stimulus)."""
    if type_idx is None:
        ax.scatter(
            positions[:n_input_neurons, 0], positions[:n_input_neurons, 1],
            s=style.hex_stimulus_marker_size, c=stimulus,
            cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax,
            marker=style.hex_marker, alpha=1.0, linewidths=0,
        )
        ax.set_title(style._label('stimuli'), fontsize=style.font_size)
    else:
        mask = neuron_types == type_idx
        count = int(np.sum(mask))
        name = names.get(type_idx, f'type_{type_idx}')
        if count > 0:
            ax.scatter(
                positions[:count, 0], positions[:count, 1],
                s=style.hex_marker_size, c=voltages[mask],
                cmap=cmap, vmin=vmin, vmax=vmax,
                marker=style.hex_marker, alpha=1, linewidths=0,
            )
        ax.set_title(style._label(name), fontsize=style.font_size)

    ax.set_facecolor(style.background)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)


# ================================================================== #
#  CONSOLIDATED FROM generators/utils.py
# ================================================================== #



def plot_signal_loss(loss_dict, log_dir, epoch=None, Niter=None, epoch_boundaries=None,
                     debug=False, current_loss=None, current_regul=None, total_loss=None,
                     total_loss_regul=None):
    """
    Plot stratified loss components over training iterations.

    Creates a three-panel figure showing loss and regularization terms in both
    linear and log scale, plus connectivity R2 trajectory. Saves to {log_dir}/tmp_training/loss.png.

    Parameters:
    -----------
    loss_dict : dict
        Dictionary containing loss component lists with keys:
        - 'loss': Loss without regularization
        - 'regul_total': Total regularization loss
        - 'iteration': Global iteration numbers (for x-axis)
        - 'W_L1': W L1 sparsity penalty
        - 'W_L2': W L2 regularization penalty
        - 'g_phi_diff': g_phi monotonicity penalty
        - 'g_phi_norm': g_phi normalization
        - 'g_phi_weight': g_phi MLP weight regularization
        - 'f_theta_weight': f_theta MLP weight regularization
        - 'W_sign': W sign consistency penalty
    log_dir : str
        Directory to save the figure
    epoch : int, optional
        Current epoch number
    Niter : int, optional
        Number of iterations per epoch
    debug : bool, optional
        If True, print debug information about loss components
    current_loss : float, optional
        Current iteration total loss (for debug)
    current_regul : float, optional
        Current iteration regularization (for debug)
    total_loss : float, optional
        Accumulated total loss (for debug)
    total_loss_regul : float, optional
        Accumulated regularization loss (for debug)
    """
    if len(loss_dict['loss']) == 0:
        return

    # Debug output if requested
    if debug and current_loss is not None and current_regul is not None:
        current_pred_loss = current_loss - current_regul

        # Get current iteration component values (last element in each list)
        comp_sum = (loss_dict['W_L1'][-1] + loss_dict['W_L2'][-1] +
                   loss_dict['g_phi_diff'][-1] + loss_dict['g_phi_norm'][-1] +
                   loss_dict['g_phi_weight'][-1] + loss_dict['f_theta_weight'][-1] +
                   loss_dict['W_sign'][-1])

        print(f"\n=== DEBUG Loss Components (Epoch {epoch}, Iter {Niter}) ===")
        print("Current iteration:")
        print(f"  loss.item() (total): {current_loss:.6f}")
        print(f"  regul_this_iter: {current_regul:.6f}")
        print(f"  prediction_loss (loss - regul): {current_pred_loss:.6f}")
        print("\nRegularization breakdown:")
        print(f"  W_L1: {loss_dict['W_L1'][-1]:.6f}")
        print(f"  W_L2: {loss_dict['W_L2'][-1]:.6f}")
        print(f"  W_sign: {loss_dict['W_sign'][-1]:.6f}")
        print(f"  g_phi_diff: {loss_dict['g_phi_diff'][-1]:.6f}")
        print(f"  g_phi_norm: {loss_dict['g_phi_norm'][-1]:.6f}")
        print(f"  g_phi_weight: {loss_dict['g_phi_weight'][-1]:.6f}")
        print(f"  f_theta_weight: {loss_dict['f_theta_weight'][-1]:.6f}")
        print(f"  Sum of components: {comp_sum:.6f}")
        if total_loss is not None and total_loss_regul is not None:
            print("\nAccumulated (for reference):")
            print(f"  total_loss (accumulated): {total_loss:.6f}")
            print(f"  total_loss_regul (accumulated): {total_loss_regul:.6f}")
        if current_loss > 0:
            print(f"\nRatio: regul / loss (current iter) = {current_regul / current_loss:.4f}")
        if current_pred_loss < 0:
            print("\n⚠️  WARNING: Negative prediction loss! regul > total loss")
        print("="*60)

    style = default_style
    lw = style.line_width
    fig_loss, (ax1, ax2, ax3) = style.figure(ncols=3, width=3 * style.figure_height * style.default_aspect)

    # x-axis: use global iteration if available, otherwise list index
    x_iter = loss_dict.get('iteration') or list(range(len(loss_dict['loss'])))

    # Linear scale
    legend_fs = 9
    for a in (ax1, ax2, ax3):
        a.tick_params(axis='x', labelsize=9)
    ax1.plot(x_iter, loss_dict['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax1.plot(x_iter, loss_dict['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax1.plot(x_iter, loss_dict['W_L1'], color='r', linewidth=1, label='W l1 sparsity', alpha=0.7)
    ax1.plot(x_iter, loss_dict['W_L2'], color='darkred', linewidth=1, label='W l2 regul', alpha=0.7)
    ax1.plot(x_iter, loss_dict['W_sign'], color='navy', linewidth=1, label='W sign (dale)', alpha=0.7)
    ax1.plot(x_iter, loss_dict['f_theta_weight'], color='lime', linewidth=1, label=r'$f_\theta$ weight regul', alpha=0.7)
    ax1.plot(x_iter, loss_dict['g_phi_diff'], color='orange', linewidth=1, label=r'$g_\phi$ monotonicity', alpha=0.7)
    ax1.plot(x_iter, loss_dict['g_phi_norm'], color='brown', linewidth=1, label=r'$g_\phi$ norm', alpha=0.7)
    ax1.plot(x_iter, loss_dict['g_phi_weight'], color='pink', linewidth=1, label=r'$g_\phi$ weight regul', alpha=0.7)
    style.xlabel(ax1, 'iteration')
    style.ylabel(ax1, 'loss')
    ax1.legend(fontsize=legend_fs, loc='best', ncol=2)

    # Log scale
    ax2.plot(x_iter, loss_dict['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax2.plot(x_iter, loss_dict['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax2.plot(x_iter, loss_dict['W_L1'], color='r', linewidth=1, label='W l1 sparsity', alpha=0.7)
    ax2.plot(x_iter, loss_dict['W_L2'], color='darkred', linewidth=1, label='W l2 regul', alpha=0.7)
    ax2.plot(x_iter, loss_dict['W_sign'], color='navy', linewidth=1, label='W sign (dale)', alpha=0.7)
    ax2.plot(x_iter, loss_dict['f_theta_weight'], color='lime', linewidth=1, label=r'$f_\theta$ weight regul', alpha=0.7)
    ax2.plot(x_iter, loss_dict['g_phi_diff'], color='orange', linewidth=1, label=r'$g_\phi$ monotonicity', alpha=0.7)
    ax2.plot(x_iter, loss_dict['g_phi_norm'], color='brown', linewidth=1, label=r'$g_\phi$ norm', alpha=0.7)
    ax2.plot(x_iter, loss_dict['g_phi_weight'], color='pink', linewidth=1, label=r'$g_\phi$ weight regul', alpha=0.7)
    style.xlabel(ax2, 'iteration')
    style.ylabel(ax2, 'loss')
    ax2.set_yscale('log')
    ax2.legend(fontsize=legend_fs, loc='best', ncol=2)

    # Epoch boundary lines on all three panels
    if epoch_boundaries:
        for xb in epoch_boundaries:
            for ax in (ax1, ax2, ax3):
                ax.axvline(x=xb, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # R2 metrics panel (conn, V_rest, tau)
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(metrics_log_path):
        r2_iters, conn_vals, vrest_vals, tau_vals = [], [], [], []
        try:
            with open(metrics_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('epoch', 'iteration')):
                        continue
                    parts = line.split(',')
                    r2_iters.append(int(parts[0]))
                    conn_vals.append(float(parts[1]))
                    vrest_vals.append(float(parts[2]) if len(parts) > 2 else 0.0)
                    tau_vals.append(float(parts[3]) if len(parts) > 3 else 0.0)
        except Exception:
            pass
        if conn_vals:
            ax3.plot(r2_iters, conn_vals, color='#d62728', linewidth=1,
                     label=r'connectivity $R^2$')
            ax3.plot(r2_iters, vrest_vals, color='#1f77b4', linewidth=1,
                     label=r'$V_{rest}$ $R^2$')
            ax3.plot(r2_iters, tau_vals, color='#2ca02c', linewidth=1,
                     label=r'$\tau$ $R^2$')
            ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax3.set_ylim(-0.05, 1.05)
            style.xlabel(ax3, 'iteration')
            style.ylabel(ax3, r'$R^2$')
            ax3.legend(fontsize=legend_fs, loc='lower right')
            # most recent R2 values
            latest_text = (f"conn={conn_vals[-1]:.3f}\n"
                           f"vrest={vrest_vals[-1]:.3f}\n"
                           f"tau={tau_vals[-1]:.3f}")
            ax3.text(0.02, 0.97, latest_text, transform=ax3.transAxes,
                     fontsize=style.annotation_font_size, verticalalignment='top')
        else:
            ax3.text(0.5, 0.5, 'no r\u00b2 data yet', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=style.label_font_size, color='gray')
    else:
        ax3.text(0.5, 0.5, 'no r\u00b2 data yet', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=style.label_font_size, color='gray')

    style.savefig(fig_loss, f'{log_dir}/tmp_training/loss.png')
    plt.close()


def plot_loss_from_file(log_dir):
    """Load loss_components.pt and plot loss decomposition (log scale).

    Parameters
    ----------
    log_dir : str
        Log directory containing ``loss_components.pt``.

    Returns
    -------
    str
        Path to the saved ``loss.png``, or *None* if the file was not found.
    """
    import torch
    pt_path = os.path.join(log_dir, 'loss_components.pt')
    if not os.path.isfile(pt_path):
        return None
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    epoch_boundaries = data.pop('epoch_boundaries', None)

    style = default_style
    fig, ax = style.figure(ncols=1)
    x_iter = data.get('iteration') or list(range(len(data['loss'])))
    legend_fs = 7

    ax.plot(x_iter, data['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax.plot(x_iter, data['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax.plot(x_iter, data['W_L1'], color='r', linewidth=1, label='W l1 sparsity', alpha=0.7)
    ax.plot(x_iter, data['W_L2'], color='darkred', linewidth=1, label='W l2 regul', alpha=0.7)
    ax.plot(x_iter, data['W_sign'], color='navy', linewidth=1, label='W sign (dale)', alpha=0.7)
    ax.plot(x_iter, data['f_theta_weight'], color='lime', linewidth=1, label=r'$f_\theta$ weight regul', alpha=0.7)
    ax.plot(x_iter, data['g_phi_diff'], color='orange', linewidth=1, label=r'$g_\phi$ monotonicity', alpha=0.7)
    ax.plot(x_iter, data['g_phi_norm'], color='brown', linewidth=1, label=r'$g_\phi$ norm', alpha=0.7)
    ax.plot(x_iter, data['g_phi_weight'], color='pink', linewidth=1, label=r'$g_\phi$ weight regul', alpha=0.7)
    ax.set_yscale('log')
    style.xlabel(ax, 'iteration')
    style.ylabel(ax, 'loss')
    ax.legend(fontsize=legend_fs, loc='best', ncol=2)

    if epoch_boundaries:
        for xb in epoch_boundaries:
            ax.axvline(x=xb, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    out_path = os.path.join(log_dir, 'tmp_training', 'loss_log.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    style.savefig(fig, out_path)
    plt.close()
    return out_path


# ================================================================== #
#  CONSOLIDATED FROM models/utils.py
# ================================================================== #

def plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list,
                         gt_weights, edges, n_neurons=None, n_neuron_types=None):
    from flyvis_gnn.plot import (
        plot_embedding,
        plot_f_theta,
        plot_g_phi,
        plot_weight_scatter,
    )
    from flyvis_gnn.utils import CustomColorMap

    if n_neurons is None:
        n_neurons = len(type_list)

    cmap = CustomColorMap(config=config)

    # Plot 1: Embedding scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_embedding(ax, model, type_list, n_neuron_types, cmap)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/embedding/{epoch}_{N}.png", dpi=87)
    plt.close()

    # Plot 2: Raw W scatter (no correction)
    fig, ax = plt.subplots(figsize=(8, 8))
    raw_W = to_numpy(get_model_W(model).squeeze())
    r_squared_raw, _ = plot_weight_scatter(
        ax,
        gt_weights=to_numpy(gt_weights),
        learned_weights=raw_W,
        corrected=False,
        outlier_threshold=5,
    )
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Compute corrected weights
    corrected_W, _, _, _ = compute_all_corrected_weights(
        model, config, edges, x_ts, device)

    # Plot 3: Corrected weight comparison scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    r_squared, _ = plot_weight_scatter(
        ax,
        gt_weights=to_numpy(gt_weights),
        learned_weights=to_numpy(corrected_W.squeeze()),
        corrected=True,
        xlim=[-1, 2],
        ylim=[-1, 2],
        outlier_threshold=5,
    )
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 4: Edge function visualization (g_phi / MLP1)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_g_phi(ax, model, config, n_neurons, type_list, cmap, device)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/function/MLP1/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    # Plot 5: Phi function visualization (f_theta / MLP0)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_f_theta(ax, model, config, n_neurons, type_list, cmap, device)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/function/MLP0/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    return r_squared


def plot_training_linear(model, config, epoch, N, log_dir, device,
                         gt_weights, n_neurons=None):
    """Training diagnostics for FlyVisLinear — raw W scatter + tau/Vrest vs GT.

    Uses compute_dynamics_r2_linear from metrics for R² computation,
    and generates scatter plots for W, tau, V_rest.

    Returns:
        (connectivity_r2, tau_r2, vrest_r2)
    """
    import torch.nn.functional as F

    from flyvis_gnn.metrics import compute_dynamics_r2_linear
    from flyvis_gnn.plot import plot_weight_scatter

    if n_neurons is None:
        n_neurons = model.n_neurons

    # Compute all R² values via shared metrics function
    vrest_r2, tau_r2, conn_r2 = compute_dynamics_r2_linear(model, config, device, n_neurons)

    # Extract learned parameters for plotting
    learned_tau = to_numpy(F.softplus(model.raw_tau[:n_neurons]).detach())
    learned_vrest = to_numpy(model.V_rest[:n_neurons].detach())

    # Load ground-truth tau and V_rest for scatter plots
    from flyvis_gnn.utils import graphs_data_path
    tau_path = graphs_data_path(config.dataset, 'taus.pt')
    if not os.path.exists(tau_path):
        tau_path = graphs_data_path(config.dataset, 'tau_i.pt')
    gt_tau_np = to_numpy(torch.load(tau_path, map_location=device, weights_only=True)[:n_neurons])
    gt_vrest_np = to_numpy(torch.load(
        graphs_data_path(config.dataset, 'V_i_rest.pt'),
        map_location=device, weights_only=True)[:n_neurons])

    # Plot 1: Raw W scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_weight_scatter(
        ax,
        gt_weights=to_numpy(gt_weights),
        learned_weights=to_numpy(get_model_W(model).squeeze()),
        corrected=False,
        outlier_threshold=5,
    )
    plt.tight_layout()
    os.makedirs(f"{log_dir}/tmp_training/matrix", exist_ok=True)
    plt.savefig(f"{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 2: tau scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_weight_scatter(ax, gt_weights=gt_tau_np, learned_weights=learned_tau, corrected=False)
    ax.set_xlabel(r'true $\tau$', fontsize=24)
    ax.set_ylabel(r'learned $\tau$', fontsize=24)
    plt.tight_layout()
    os.makedirs(f"{log_dir}/tmp_training/dynamics", exist_ok=True)
    plt.savefig(f"{log_dir}/tmp_training/dynamics/tau_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 3: V_rest scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_weight_scatter(ax, gt_weights=gt_vrest_np, learned_weights=learned_vrest, corrected=False)
    ax.set_xlabel(r'true $V_{rest}$', fontsize=24)
    ax.set_ylabel(r'learned $V_{rest}$', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/dynamics/vrest_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    return conn_r2, tau_r2, vrest_r2


def plot_weight_comparison(w_true, w_modified, output_path, xlabel='true $W$', ylabel='modified $W$', color='white'):
    w_true_np = w_true.detach().cpu().numpy().flatten()
    w_modified_np = w_modified.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(w_true_np, w_modified_np, s=8, alpha=0.5, color=color, edgecolors='none')
    # Fit linear model
    lin_fit, _ = curve_fit(linear_model, w_true_np, w_modified_np)
    slope = lin_fit[0]
    lin_fit[1]
    # R2 calculation
    residuals = w_modified_np - linear_model(w_true_np, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((w_modified_np - np.mean(w_modified_np)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # Plot identity line
    plt.plot([w_true_np.min(), w_true_np.max()], [w_true_np.min(), w_true_np.max()], 'r--', linewidth=2, label='identity')
    # Add text
    plt.text(w_true_np.min(), w_true_np.max(), f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}', fontsize=18, va='top', ha='left')
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return slope, r_squared


# ================================================================== #
#  CONSOLIDATED FROM models/plot_utils.py
# ================================================================== #

import warnings

from tqdm import trange

warnings.filterwarnings('ignore')

def render_visual_field_video(model, x_ts, sim, log_dir, epoch, N, logger):
    """Render a 3-panel visual field video (GT hex, predicted hex, rolling traces).

    Computes a linear correction gt = a*pred + b over frames 0..800, then
    renders an MP4 with ground-truth vs corrected-prediction hex scatter
    plots and rolling traces for 10 representative neurons.

    Args:
        model: FlyVisGNN model with forward_visual method
        x_ts: NeuronTimeSeries on GPU
        sim: SimulationConfig
        log_dir: output directory path
        epoch: current epoch number
        N: current iteration number
        logger: logging.Logger instance

    Returns:
        field_R2: R² of corrected predictions vs ground truth
        field_slope: slope coefficient 'a' of the linear fit
    """
    with torch.no_grad():

        # Static XY locations
        X1 = to_numpy(x_ts.pos[:sim.n_input_neurons])

        # group-based selection of 10 traces
        groups = 217
        group_size = sim.n_input_neurons // groups  # expect 8
        assert groups * group_size == sim.n_input_neurons, "Unexpected packing of input neurons"
        picked_groups = np.linspace(0, groups - 1, 10, dtype=int)
        member_in_group = group_size // 2
        trace_ids = (picked_groups * group_size + member_in_group).astype(int)

        # MP4 writer setup
        fps = 10
        metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(12, 4))

        out_dir = f"{log_dir}/tmp_training/external_input"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/field_movie_{epoch}_{N}.mp4"
        if os.path.exists(out_path):
            os.remove(out_path)

        # rolling buffers
        win = 200
        offset = 1.25
        hist_t = deque(maxlen=win)
        hist_gt = {i: deque(maxlen=win) for i in trace_ids}
        hist_pred = {i: deque(maxlen=win) for i in trace_ids}

        step_video = 2

        # First pass: collect all gt and pred, fit linear transform gt = a*pred + b
        all_gt = []
        all_pred = []
        for k_fit in range(0, 800, step_video):
            x_fit = x_ts.frame(k_fit)
            pred_fit = to_numpy(model.forward_visual(x_fit, k_fit)).squeeze()
            gt_fit = to_numpy(x_ts.stimulus[k_fit, :sim.n_input_neurons]).squeeze()
            all_gt.append(gt_fit)
            all_pred.append(pred_fit)
        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        # Least-squares fit: gt = a * pred + b
        A_fit = np.vstack([all_pred, np.ones(len(all_pred))]).T
        a_coeff, b_coeff = np.linalg.lstsq(A_fit, all_gt, rcond=None)[0]
        logger.info(f"field linear fit: gt = {a_coeff:.4f} * pred + {b_coeff:.4f}")

        # Compute field_R2 on corrected predictions
        pred_corrected_all = a_coeff * all_pred + b_coeff
        ss_res = np.sum((all_gt - pred_corrected_all) ** 2)
        ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
        field_R2 = 1 - ss_res / (ss_tot + 1e-16)
        field_slope = a_coeff
        logger.info(f"external input R² (corrected): {field_R2:.4f}")

        # GT value range for consistent color scaling
        gt_vmin = float(all_gt.min())
        gt_vmax = float(all_gt.max())

        with writer.saving(fig, out_path, dpi=200):
            error_list = []

            for k in trange(0, 800, step_video, ncols=100):
                # inputs and predictions
                x = x_ts.frame(k)
                pred = to_numpy(model.forward_visual(x, k))
                pred_vec = np.asarray(pred).squeeze()  # (sim.n_input_neurons,)
                pred_corrected = a_coeff * pred_vec + b_coeff  # corrected to GT scale

                gt_vec = to_numpy(x_ts.stimulus[k, :sim.n_input_neurons]).squeeze()

                # update rolling traces (store corrected predictions)
                hist_t.append(k)
                for i in trace_ids:
                    hist_gt[i].append(gt_vec[i])
                    hist_pred[i].append(pred_corrected[i])

                # draw three panels
                fig.clf()

                # RMSE on corrected predictions
                rmse_frame = float(np.sqrt(((pred_corrected - gt_vec) ** 2).mean()))
                running_rmse = float(np.mean(error_list + [rmse_frame])) if len(error_list) else rmse_frame

                # Traces (both on GT scale)
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_axis_off()
                ax3.set_facecolor("black")

                t = np.arange(len(hist_t))
                for j, i in enumerate(trace_ids):
                    y0 = j * offset
                    ax3.plot(t, np.array(hist_gt[i])   + y0, color='lime',  lw=1.6, alpha=0.95)
                    ax3.plot(t, np.array(hist_pred[i]) + y0, color='k', lw=1.2, alpha=0.95)

                ax3.set_xlim(max(0, len(t) - win), len(t))
                ax3.set_ylim(-offset * 0.5, offset * (len(trace_ids) + 0.5))
                ax3.text(
                    0.02, 0.98,
                    f"frame: {k}   RMSE: {rmse_frame:.3f}   avg RMSE: {running_rmse:.3f}   a={a_coeff:.3f} b={b_coeff:.3f}",
                    transform=ax3.transAxes,
                    va='top', ha='left',
                    fontsize=6, color='k')

                # GT field
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_vec, cmap=default_style.cmap, marker='h', vmin=gt_vmin, vmax=gt_vmax)
                ax1.set_axis_off()
                ax1.set_title('ground truth', fontsize=12)

                # Predicted field (corrected, same scale as GT)
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=pred_corrected, cmap=default_style.cmap, marker='h')
                ax2.set_axis_off()
                ax2.set_title('prediction (corrected)', fontsize=12)

                plt.tight_layout()
                writer.grab_frame()

                error_list.append(rmse_frame)

    return field_R2, field_slope

