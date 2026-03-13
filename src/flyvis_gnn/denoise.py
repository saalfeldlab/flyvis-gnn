"""Wiener / wavelet / spectral-subtraction denoising of noisy derivative targets.

Usage (standalone):
    python -m flyvis_gnn.denoise --config <yaml> --checkpoint <model.pt> [--device cuda]

Usage (from code):
    from flyvis_gnn.denoise import wiener_filter_derivatives
    results = wiener_filter_derivatives(config, model_checkpoint, device='cuda')

The function loads a trained model, generates predictions on the training set,
estimates the signal power spectrum, and applies a configurable filter to the
noisy derivative targets.  The filtered targets are saved as
wiener_y_list_{split}.zarr alongside the existing noisy_y_list.

Config fields (in simulation: section):
    derivative_target:          'clean' | 'noisy' | 'wiener'
    filter_algorithm:           'wiener' | 'wavelet' | 'spectral_subtraction'
    filter_noise_fraction:      0.0 .. 1.0  (aggressiveness knob)
    filter_noise_spectrum:      'analytical' | 'empirical'
    filter_spectrum_smoothing:  int  (frequency-bin smoothing of periodogram)
    filter_h_floor:             float  (minimum filter gain)
    filter_per_neuron_type:     bool
    filter_save_plots:          bool
    filter_wavelet_name:        str  ('db4', 'sym6', ...)
    filter_wavelet_level:       int  (0 = auto)
    filter_wavelet_threshold:   'soft' | 'hard'
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyvis_gnn.log import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def wiener_filter_derivatives(config, model_checkpoint, device='cuda', split='train'):
    """Apply frequency-domain denoising to noisy derivative targets.

    Args:
        config: NeuralGraphConfig with filter fields in simulation section.
        model_checkpoint: path to a trained model .pt file.
        device: torch device string.
        split: 'train' or 'test'.

    Returns:
        dict with keys: mse_noisy, mse_filtered, mse_pred, r2_noisy,
        r2_filtered, out_path.
    """
    from flyvis_gnn.utils import graphs_data_path
    from flyvis_gnn.zarr_io import ZarrArrayWriter, load_raw_array, load_simulation_data
    from flyvis_gnn.models.training_utils import (
        build_model, determine_load_fields, load_flyvis_data,
    )

    sim = config.simulation
    dt = sim.delta_t
    sigma_meas = sim.measurement_noise_level
    fraction = sim.filter_noise_fraction
    algorithm = sim.filter_algorithm
    log_dir = os.path.dirname(os.path.dirname(model_checkpoint))  # models/ -> log_dir

    logger.info(f"wiener_filter_derivatives: algorithm={algorithm}, "
                f"noise_fraction={fraction}, checkpoint={model_checkpoint}")

    # --- Load data ---
    y_noisy = load_raw_array(
        graphs_data_path(config.dataset, f"noisy_y_list_{split}"))  # (T, N, 1)
    y_clean = load_raw_array(
        graphs_data_path(config.dataset, f"y_list_{split}"))         # (T, N, 1)
    T, N, _ = y_noisy.shape

    # Neuron types for per-type spectrum averaging
    x_ts_fields = load_simulation_data(
        graphs_data_path(config.dataset, f"x_list_{split}"),
        fields=['neuron_type'],
    )
    neuron_types = x_ts_fields.neuron_type.numpy()   # (N,)
    unique_types = np.unique(neuron_types)

    # --- Generate model predictions ---
    logger.info("generating model predictions for signal spectrum estimation...")
    y_pred = _generate_predictions(config, model_checkpoint, device, split, T, N)

    # --- Compute spectra ---
    freqs = np.fft.rfftfreq(T, d=dt)  # (T//2+1,)

    S_noise_analytical = _analytical_noise_spectrum(freqs, sigma_meas, dt)

    S_signal_per_neuron, S_noisy_per_neuron = _estimate_signal_spectrum(
        y_pred, y_noisy, neuron_types, unique_types, freqs, T, N,
        per_type=sim.filter_per_neuron_type,
        smooth_bins=sim.filter_spectrum_smoothing,
    )

    # Empirical noise spectrum (for validation / optional use)
    residuals = y_noisy[:, :, 0] - y_pred[:, :, 0]
    S_residual_mean = np.mean(
        np.abs(np.fft.rfft(residuals, axis=0)) ** 2, axis=1) / T

    # Effective noise spectrum for the filter
    if sim.filter_noise_spectrum == 'empirical':
        S_noise_eff = np.broadcast_to(
            S_residual_mean[np.newaxis, :], (N, len(freqs))).copy()
    else:
        S_noise_eff = np.broadcast_to(
            S_noise_analytical[np.newaxis, :], (N, len(freqs))).copy()

    # --- Apply filter ---
    h_floor = sim.filter_h_floor
    y_filtered = _apply_filter(
        algorithm, y_noisy, S_signal_per_neuron, S_noise_eff,
        fraction, h_floor, T, N, sim,
    )
    logger.info(f"filtering complete: algorithm={algorithm}, "
                f"noise_fraction={fraction}")

    # --- Metrics ---
    metrics = _compute_metrics(y_noisy, y_filtered, y_pred, y_clean)
    logger.info(
        f"MSE noisy->clean: {metrics['mse_noisy']:.6f}, "
        f"filtered->clean: {metrics['mse_filtered']:.6f}, "
        f"pred->clean: {metrics['mse_pred']:.6f}")
    logger.info(
        f"R2 noisy->clean: {metrics['r2_noisy']:.4f}, "
        f"filtered->clean: {metrics['r2_filtered']:.4f}")
    logger.info(
        f"MSE reduction: "
        f"{(1 - metrics['mse_filtered'] / metrics['mse_noisy']) * 100:.1f}%")

    # --- Save ---
    out_path = graphs_data_path(config.dataset, f"wiener_y_list_{split}")
    writer = ZarrArrayWriter(
        path=out_path, n_neurons=N, n_features=1, time_chunks=2000)
    for t in range(T):
        writer.append(y_filtered[t])
    writer.finalize()
    logger.info(f"saved filtered targets to {out_path}")

    # --- Diagnostic plots ---
    if sim.filter_save_plots:
        plot_dir = os.path.join(log_dir, 'wiener_diagnostics')
        os.makedirs(plot_dir, exist_ok=True)
        _plot_diagnostics(
            freqs=freqs,
            S_signal_per_neuron=S_signal_per_neuron,
            S_noise_analytical=S_noise_analytical,
            S_residual_mean=S_residual_mean,
            S_noisy_per_neuron=S_noisy_per_neuron,
            y_noisy=y_noisy, y_filtered=y_filtered,
            y_clean=y_clean, y_pred=y_pred,
            neuron_types=neuron_types, unique_types=unique_types,
            fraction=fraction, algorithm=algorithm, h_floor=h_floor,
            metrics=metrics, plot_dir=plot_dir, dt=dt,
        )

    metrics['out_path'] = str(out_path)
    return metrics


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _generate_predictions(config, model_checkpoint, device, split, T, N):
    """Run the trained model on all frames and return predictions."""
    from flyvis_gnn.utils import graphs_data_path
    from flyvis_gnn.zarr_io import load_simulation_data
    from flyvis_gnn.models.training_utils import (
        build_model, determine_load_fields, load_flyvis_data,
    )

    sim = config.simulation
    sigma_meas = sim.measurement_noise_level

    model, _ = build_model(config, device, checkpoint_path=model_checkpoint)
    model.eval()

    load_fields = determine_load_fields(config)
    x_ts, _, _ = load_flyvis_data(
        config.dataset, split=split, fields=load_fields, device=device,
        measurement_noise_level=sigma_meas,
    )

    edges = torch.load(
        graphs_data_path(config.dataset, 'edge_index.pt'),
        map_location=device, weights_only=False)
    data_id = torch.zeros((N, 1), dtype=torch.int, device=device)

    y_pred = np.zeros((T, N, 1), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            state = x_ts.frame(t).to(device)
            if state.noise is not None and sigma_meas > 0:
                state.voltage = state.voltage + state.noise
            pred = model(state, edges, data_id=data_id)
            y_pred[t] = pred.cpu().numpy()

    return y_pred


def _analytical_noise_spectrum(freqs, sigma_meas, dt):
    """Analytical PSD of measurement-noise derivative.

    noise_deriv[t] = (noise[t+1] - noise[t]) / dt,  noise ~ N(0, sigma_meas)
    S(f) = 4 sigma^2 sin^2(pi f dt) / dt^2   (continuous PSD)
    Scaled by dt to match numpy periodogram convention.
    """
    S = 4 * sigma_meas ** 2 * np.sin(np.pi * freqs * dt) ** 2 / dt ** 2
    return S * dt


def _smooth_spectrum(S, n_bins):
    if n_bins <= 1:
        return S
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(S, size=n_bins, mode='nearest')


def _estimate_signal_spectrum(
    y_pred, y_noisy, neuron_types, unique_types, freqs, T, N,
    per_type, smooth_bins,
):
    """Estimate signal and noisy power spectra, optionally averaged by type."""
    n_freq = len(freqs)
    S_signal = np.zeros((N, n_freq), dtype=np.float64)
    S_noisy = np.zeros((N, n_freq), dtype=np.float64)

    if per_type:
        for typ in unique_types:
            mask = neuron_types == typ
            pred_type = y_pred[:, mask, 0]
            noisy_type = y_noisy[:, mask, 0]
            S_p = np.mean(np.abs(np.fft.rfft(pred_type, axis=0)) ** 2, axis=1) / T
            S_n = np.mean(np.abs(np.fft.rfft(noisy_type, axis=0)) ** 2, axis=1) / T
            S_signal[mask] = _smooth_spectrum(S_p, smooth_bins)[np.newaxis, :]
            S_noisy[mask] = _smooth_spectrum(S_n, smooth_bins)[np.newaxis, :]
    else:
        for n in range(N):
            S_signal[n] = _smooth_spectrum(
                np.abs(np.fft.rfft(y_pred[:, n, 0])) ** 2 / T, smooth_bins)
            S_noisy[n] = _smooth_spectrum(
                np.abs(np.fft.rfft(y_noisy[:, n, 0])) ** 2 / T, smooth_bins)

    return S_signal, S_noisy


def _apply_filter(algorithm, y_noisy, S_signal, S_noise_eff,
                  fraction, h_floor, T, N, sim):
    """Dispatch to the chosen filter algorithm."""
    y_filtered = np.zeros_like(y_noisy)

    if algorithm == 'wiener':
        for n in range(N):
            Y = np.fft.rfft(y_noisy[:, n, 0])
            S_noi = fraction * S_noise_eff[n]
            H = S_signal[n] / (S_signal[n] + S_noi + 1e-30)
            H = np.maximum(H, h_floor)
            y_filtered[:, n, 0] = np.fft.irfft(H * Y, n=T)

    elif algorithm == 'wavelet':
        import pywt
        wname = sim.filter_wavelet_name
        level = sim.filter_wavelet_level if sim.filter_wavelet_level > 0 else None
        mode = sim.filter_wavelet_threshold
        for n in range(N):
            coeffs = pywt.wavedec(y_noisy[:, n, 0], wname, level=level)
            sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = fraction * sigma_est * np.sqrt(2 * np.log(T))
            coeffs_t = [coeffs[0]]
            for c in coeffs[1:]:
                coeffs_t.append(pywt.threshold(c, value=thresh, mode=mode))
            y_filtered[:, n, 0] = pywt.waverec(coeffs_t, wname)[:T]

    elif algorithm == 'spectral_subtraction':
        for n in range(N):
            Y = np.fft.rfft(y_noisy[:, n, 0])
            power = np.abs(Y) ** 2 / T
            S_noi = fraction * S_noise_eff[n]
            power_clean = np.maximum(power - S_noi, h_floor * power)
            gain = np.sqrt(power_clean / (power + 1e-30))
            y_filtered[:, n, 0] = np.fft.irfft(gain * Y, n=T)

    else:
        raise ValueError(f"unknown filter_algorithm: {algorithm}")

    return y_filtered


def _compute_metrics(y_noisy, y_filtered, y_pred, y_clean):
    mse_noisy = float(np.mean((y_noisy - y_clean) ** 2))
    mse_filtered = float(np.mean((y_filtered - y_clean) ** 2))
    mse_pred = float(np.mean((y_pred - y_clean) ** 2))

    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    r2_noisy = 1 - float(np.sum((y_noisy - y_clean) ** 2)) / (ss_tot + 1e-30)
    r2_filtered = 1 - float(np.sum((y_filtered - y_clean) ** 2)) / (ss_tot + 1e-30)

    return {
        'mse_noisy': mse_noisy,
        'mse_filtered': mse_filtered,
        'mse_pred': mse_pred,
        'r2_noisy': r2_noisy,
        'r2_filtered': r2_filtered,
    }


# ------------------------------------------------------------------ #
#  Diagnostic plots
# ------------------------------------------------------------------ #

def _plot_diagnostics(
    freqs, S_signal_per_neuron, S_noise_analytical, S_residual_mean,
    S_noisy_per_neuron, y_noisy, y_filtered, y_clean, y_pred,
    neuron_types, unique_types, fraction, algorithm, h_floor,
    metrics, plot_dir, dt,
):
    """Six-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    S_signal_mean = np.mean(S_signal_per_neuron, axis=0)
    S_noisy_mean = np.mean(S_noisy_per_neuron, axis=0)

    # Panel 1 — Power spectra
    ax = axes[0, 0]
    ax.semilogy(freqs[1:], S_noisy_mean[1:], alpha=0.5,
                label='S_noisy (data)', color='gray')
    ax.semilogy(freqs[1:], S_signal_mean[1:],
                label='S_signal (model pred)', color='blue')
    ax.semilogy(freqs[1:], S_noise_analytical[1:],
                label='S_noise (analytical)', color='red', ls='--')
    ax.semilogy(freqs[1:], S_residual_mean[1:],
                label='S_residual (empirical)', color='orange', ls=':')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Power spectra (mean over neurons)')
    ax.legend(fontsize=7)

    # Panel 2 — Filter H(f)
    ax = axes[0, 1]
    S_noi_plot = fraction * S_noise_analytical
    H_mean = S_signal_mean / (S_signal_mean + S_noi_plot + 1e-30)
    H_mean = np.maximum(H_mean, h_floor)
    ax.plot(freqs[1:], H_mean[1:], color='blue')
    ax.axhline(y=h_floor, color='red', ls='--', alpha=0.5,
               label=f'floor={h_floor}')
    cross_idx = np.argmin(np.abs(S_signal_mean[1:] - S_noi_plot[1:]))
    if cross_idx > 0:
        ax.axvline(x=freqs[cross_idx + 1], color='green', ls=':', alpha=0.5,
                   label=f'crossover={freqs[cross_idx+1]:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Filter gain H(f)')
    ax.set_title(f'{algorithm} (fraction={fraction})')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7)

    # Panel 3 — Time-domain
    ax = axes[0, 2]
    T = y_noisy.shape[0]
    t_show = min(2000, T)
    n_ex = 0  # example neuron
    t_ax = np.arange(t_show) * dt
    ax.plot(t_ax, y_noisy[:t_show, n_ex, 0], alpha=0.3, color='gray',
            lw=0.5, label='noisy')
    ax.plot(t_ax, y_filtered[:t_show, n_ex, 0], color='blue',
            lw=0.8, label='filtered')
    ax.plot(t_ax, y_clean[:t_show, n_ex, 0], color='green',
            lw=0.8, label='clean (GT)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('dV/dt')
    ax.set_title(f'Time-domain (neuron {n_ex})')
    ax.legend(fontsize=7)

    # Panel 4 — Noise model validation
    ax = axes[1, 0]
    ax.semilogy(freqs[1:], S_residual_mean[1:],
                label='|FFT(residual)|^2', color='orange')
    ax.semilogy(freqs[1:], S_noise_analytical[1:],
                label='S_noise (analytical)', color='red', ls='--')
    gap = S_residual_mean[1:] - S_noise_analytical[1:]
    ax.semilogy(freqs[1:], np.maximum(gap, 1e-20),
                label='gap (process + model err)', color='purple', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Noise model validation')
    ax.legend(fontsize=7)

    # Panel 5 — Per-type R²
    ax = axes[1, 1]
    r2_typ_n, r2_typ_f = [], []
    for typ in unique_types:
        m = neuron_types == typ
        ss_n = np.sum((y_noisy[:, m, :] - y_clean[:, m, :]) ** 2)
        ss_f = np.sum((y_filtered[:, m, :] - y_clean[:, m, :]) ** 2)
        ss_t = np.sum((y_clean[:, m, :] - np.mean(y_clean[:, m, :])) ** 2)
        r2_typ_n.append(1 - ss_n / (ss_t + 1e-30))
        r2_typ_f.append(1 - ss_f / (ss_t + 1e-30))
    x_pos = np.arange(len(unique_types))
    ax.bar(x_pos - 0.15, r2_typ_n, 0.3, alpha=0.6,
           label='noisy', color='gray')
    ax.bar(x_pos + 0.15, r2_typ_f, 0.3, alpha=0.8,
           label='filtered', color='blue')
    ax.set_xlabel('Neuron type')
    ax.set_ylabel('R^2 vs GT')
    ax.set_title(f'Quality by type (N={len(unique_types)})')
    if len(unique_types) <= 20:
        ax.set_xticks(x_pos)
    ax.legend(fontsize=7)

    # Panel 6 — Summary text
    ax = axes[1, 2]
    ax.axis('off')
    txt = (
        f"Algorithm: {algorithm}\n"
        f"Noise fraction: {fraction}\n"
        f"H floor: {h_floor}\n\n"
        f"MSE noisy->clean:    {metrics['mse_noisy']:.6f}\n"
        f"MSE filtered->clean: {metrics['mse_filtered']:.6f}\n"
        f"MSE reduction:       "
        f"{(1 - metrics['mse_filtered']/metrics['mse_noisy'])*100:.1f}%\n\n"
        f"R2 noisy->clean:    {metrics['r2_noisy']:.4f}\n"
        f"R2 filtered->clean: {metrics['r2_filtered']:.4f}\n"
        f"R2 improvement:     "
        f"{metrics['r2_filtered'] - metrics['r2_noisy']:.4f}"
    )
    ax.text(0.1, 0.9, txt, transform=ax.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    ax.set_title('Summary')

    plt.tight_layout()
    fname = os.path.join(
        plot_dir, f'wiener_diagnostics_{algorithm}_f{fraction}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"saved diagnostic plot: {fname}")

    # Save spectra for later analysis
    np.savez(
        os.path.join(plot_dir, f'spectra_{algorithm}_f{fraction}.npz'),
        freqs=freqs,
        S_signal_mean=S_signal_mean,
        S_noise_analytical=S_noise_analytical,
        S_residual_mean=S_residual_mean,
        S_noisy_mean=S_noisy_mean,
        H_mean=H_mean,
        r2_per_type_noisy=np.array(r2_typ_n),
        r2_per_type_filtered=np.array(r2_typ_f),
    )


# ------------------------------------------------------------------ #
#  CLI entry point
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Denoise derivative targets using a trained GNN model')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model .pt checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', default='train')
    # Allow CLI overrides of key filter params
    parser.add_argument('--filter_noise_fraction', type=float, default=None)
    parser.add_argument('--filter_algorithm', type=str, default=None)
    args = parser.parse_args()

    from flyvis_gnn.config import NeuralGraphConfig
    config = NeuralGraphConfig.from_yaml(args.config)

    if args.filter_noise_fraction is not None:
        config.simulation.filter_noise_fraction = args.filter_noise_fraction
    if args.filter_algorithm is not None:
        config.simulation.filter_algorithm = args.filter_algorithm

    results = wiener_filter_derivatives(
        config, args.checkpoint, device=args.device, split=args.split)

    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
