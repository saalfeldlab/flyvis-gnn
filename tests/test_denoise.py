"""Tests for flyvis_gnn.denoise — filter internals with synthetic data.

All tests are CPU-only, no disk I/O, no model checkpoints.
They exercise the core filter math to verify correctness.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from flyvis_gnn.denoise import (
    _analytical_noise_spectrum,
    _apply_filter,
    _compute_metrics,
    _estimate_signal_spectrum,
    _smooth_spectrum,
)


pytestmark = pytest.mark.tier1


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_signals(rng):
    """Create a synthetic signal + noise scenario.

    Returns y_clean, y_noisy, y_pred (all shape (T, N, 1)),
    plus dt, sigma_meas, neuron_types.
    """
    T, N = 256, 10
    dt = 0.02
    sigma_meas = 0.1
    t = np.arange(T) * dt

    # Clean signal: sum of low-frequency sinusoids (different per neuron)
    y_clean = np.zeros((T, N, 1), dtype=np.float64)
    for n in range(N):
        freq = 1.0 + 0.5 * n
        y_clean[:, n, 0] = np.sin(2 * np.pi * freq * t) + 0.5 * np.cos(2 * np.pi * 0.3 * t)

    # Measurement noise derivative: (noise[t+1] - noise[t]) / dt
    noise_ts = rng.randn(T + 1, N) * sigma_meas
    noise_deriv = np.diff(noise_ts, axis=0) / dt  # (T, N)
    y_noisy = y_clean.copy()
    y_noisy[:, :, 0] += noise_deriv

    # Model predictions: clean + small residual (simulates a good model)
    y_pred = y_clean + rng.randn(T, N, 1) * 0.1

    # Neuron types: 2 types, alternating
    neuron_types = np.array([i % 2 for i in range(N)])

    return {
        'y_clean': y_clean,
        'y_noisy': y_noisy,
        'y_pred': y_pred,
        'T': T,
        'N': N,
        'dt': dt,
        'sigma_meas': sigma_meas,
        'neuron_types': neuron_types,
    }


# ------------------------------------------------------------------ #
#  Tests: _analytical_noise_spectrum
# ------------------------------------------------------------------ #

class TestAnalyticalNoiseSpectrum:
    def test_shape(self):
        freqs = np.fft.rfftfreq(256, d=0.02)
        S = _analytical_noise_spectrum(freqs, sigma_meas=0.1, dt=0.02)
        assert S.shape == freqs.shape

    def test_dc_is_zero(self):
        """DC component (f=0) should be zero for derivative noise."""
        freqs = np.fft.rfftfreq(256, d=0.02)
        S = _analytical_noise_spectrum(freqs, sigma_meas=0.1, dt=0.02)
        assert S[0] == pytest.approx(0.0, abs=1e-15)

    def test_increases_with_frequency(self):
        """Blue noise: PSD should generally increase with frequency."""
        freqs = np.fft.rfftfreq(256, d=0.02)
        S = _analytical_noise_spectrum(freqs, sigma_meas=0.1, dt=0.02)
        # Compare low-freq vs high-freq averages
        n = len(freqs)
        low = np.mean(S[1:n // 4])
        high = np.mean(S[n // 2:])
        assert high > low

    def test_scales_with_sigma_squared(self):
        freqs = np.fft.rfftfreq(256, d=0.02)
        S1 = _analytical_noise_spectrum(freqs, sigma_meas=0.1, dt=0.02)
        S2 = _analytical_noise_spectrum(freqs, sigma_meas=0.2, dt=0.02)
        # S should scale as sigma^2
        ratio = S2[10] / S1[10]
        assert ratio == pytest.approx(4.0, rel=1e-10)


# ------------------------------------------------------------------ #
#  Tests: _smooth_spectrum
# ------------------------------------------------------------------ #

class TestSmoothSpectrum:
    def test_no_smoothing(self):
        S = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _smooth_spectrum(S, n_bins=1)
        np.testing.assert_array_equal(result, S)

    def test_smoothing_preserves_length(self):
        S = np.random.randn(100)
        result = _smooth_spectrum(S, n_bins=5)
        assert result.shape == S.shape

    def test_smoothing_reduces_variance(self):
        rng = np.random.RandomState(0)
        S = rng.randn(100)
        result = _smooth_spectrum(S, n_bins=10)
        assert np.var(result) < np.var(S)


# ------------------------------------------------------------------ #
#  Tests: _estimate_signal_spectrum
# ------------------------------------------------------------------ #

class TestEstimateSignalSpectrum:
    def test_shapes(self, synthetic_signals):
        s = synthetic_signals
        freqs = np.fft.rfftfreq(s['T'], d=s['dt'])
        unique_types = np.unique(s['neuron_types'])

        S_signal, S_noisy = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, s['T'], s['N'], per_type=True, smooth_bins=5,
        )
        assert S_signal.shape == (s['N'], len(freqs))
        assert S_noisy.shape == (s['N'], len(freqs))

    def test_per_neuron_mode(self, synthetic_signals):
        s = synthetic_signals
        freqs = np.fft.rfftfreq(s['T'], d=s['dt'])
        unique_types = np.unique(s['neuron_types'])

        S_signal, S_noisy = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, s['T'], s['N'], per_type=False, smooth_bins=1,
        )
        # Per-neuron mode: different neurons should have different spectra
        assert not np.allclose(S_signal[0], S_signal[1])

    def test_positive_spectra(self, synthetic_signals):
        s = synthetic_signals
        freqs = np.fft.rfftfreq(s['T'], d=s['dt'])
        unique_types = np.unique(s['neuron_types'])

        S_signal, S_noisy = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, s['T'], s['N'], per_type=True, smooth_bins=3,
        )
        assert np.all(S_signal >= 0)
        assert np.all(S_noisy >= 0)


# ------------------------------------------------------------------ #
#  Tests: _apply_filter
# ------------------------------------------------------------------ #

class TestApplyFilter:
    def _make_sim(self, **kwargs):
        defaults = dict(
            filter_wavelet_name='db4',
            filter_wavelet_level=0,
            filter_wavelet_threshold='soft',
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_wiener_reduces_noise(self, synthetic_signals):
        """Wiener filter should reduce MSE vs clean."""
        s = synthetic_signals
        T, N = s['T'], s['N']
        freqs = np.fft.rfftfreq(T, d=s['dt'])

        S_noise = _analytical_noise_spectrum(freqs, s['sigma_meas'], s['dt'])
        S_noise_eff = np.broadcast_to(S_noise[np.newaxis, :], (N, len(freqs))).copy()

        unique_types = np.unique(s['neuron_types'])
        S_signal, _ = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, T, N, per_type=True, smooth_bins=5,
        )

        sim = self._make_sim()
        y_filtered = _apply_filter(
            'wiener', s['y_noisy'], S_signal, S_noise_eff,
            fraction=1.0, h_floor=0.01, T=T, N=N, sim=sim,
        )

        mse_noisy = np.mean((s['y_noisy'] - s['y_clean']) ** 2)
        mse_filtered = np.mean((y_filtered - s['y_clean']) ** 2)
        assert mse_filtered < mse_noisy, (
            f"Wiener filter should reduce MSE: {mse_filtered:.4f} >= {mse_noisy:.4f}")

    def test_wiener_fraction_zero_is_identity(self, synthetic_signals):
        """With fraction=0, filter gain H(f)=1 everywhere => no change."""
        s = synthetic_signals
        T, N = s['T'], s['N']
        freqs = np.fft.rfftfreq(T, d=s['dt'])

        S_noise = _analytical_noise_spectrum(freqs, s['sigma_meas'], s['dt'])
        S_noise_eff = np.broadcast_to(S_noise[np.newaxis, :], (N, len(freqs))).copy()

        unique_types = np.unique(s['neuron_types'])
        S_signal, _ = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, T, N, per_type=True, smooth_bins=5,
        )

        sim = self._make_sim()
        y_filtered = _apply_filter(
            'wiener', s['y_noisy'], S_signal, S_noise_eff,
            fraction=0.0, h_floor=0.0, T=T, N=N, sim=sim,
        )

        np.testing.assert_allclose(y_filtered, s['y_noisy'], atol=1e-10,
                                   err_msg="fraction=0 should be identity")

    def test_spectral_subtraction_reduces_noise(self, synthetic_signals):
        s = synthetic_signals
        T, N = s['T'], s['N']
        freqs = np.fft.rfftfreq(T, d=s['dt'])

        S_noise = _analytical_noise_spectrum(freqs, s['sigma_meas'], s['dt'])
        S_noise_eff = np.broadcast_to(S_noise[np.newaxis, :], (N, len(freqs))).copy()

        unique_types = np.unique(s['neuron_types'])
        S_signal, _ = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, T, N, per_type=True, smooth_bins=5,
        )

        sim = self._make_sim()
        y_filtered = _apply_filter(
            'spectral_subtraction', s['y_noisy'], S_signal, S_noise_eff,
            fraction=1.0, h_floor=0.01, T=T, N=N, sim=sim,
        )

        mse_noisy = np.mean((s['y_noisy'] - s['y_clean']) ** 2)
        mse_filtered = np.mean((y_filtered - s['y_clean']) ** 2)
        assert mse_filtered < mse_noisy

    def test_unknown_algorithm_raises(self, synthetic_signals):
        s = synthetic_signals
        sim = self._make_sim()
        with pytest.raises(ValueError, match="unknown filter_algorithm"):
            _apply_filter(
                'bogus', s['y_noisy'], np.ones((s['N'], 1)), np.ones((s['N'], 1)),
                fraction=1.0, h_floor=0.0, T=s['T'], N=s['N'], sim=sim,
            )


# ------------------------------------------------------------------ #
#  Tests: _compute_metrics
# ------------------------------------------------------------------ #

class TestComputeMetrics:
    def test_perfect_filter(self, rng):
        """If filtered == clean, MSE should be 0 and R2 should be 1."""
        T, N = 100, 5
        y_clean = rng.randn(T, N, 1)
        y_noisy = y_clean + rng.randn(T, N, 1) * 0.5
        y_pred = y_clean + rng.randn(T, N, 1) * 0.1

        metrics = _compute_metrics(y_noisy, y_clean, y_pred, y_clean)
        assert metrics['mse_filtered'] == pytest.approx(0.0, abs=1e-15)
        assert metrics['r2_filtered'] == pytest.approx(1.0, abs=1e-10)

    def test_noisy_worse_than_filtered(self, synthetic_signals):
        """Basic sanity: noisy MSE should be >= 0."""
        s = synthetic_signals
        metrics = _compute_metrics(s['y_noisy'], s['y_noisy'], s['y_pred'], s['y_clean'])
        assert metrics['mse_noisy'] > 0
        # When filtered == noisy, they should match
        assert metrics['mse_filtered'] == pytest.approx(metrics['mse_noisy'])

    def test_returns_all_keys(self, rng):
        T, N = 50, 3
        y = rng.randn(T, N, 1)
        metrics = _compute_metrics(y, y, y, y)
        for key in ['mse_noisy', 'mse_filtered', 'mse_pred', 'r2_noisy', 'r2_filtered']:
            assert key in metrics


# ------------------------------------------------------------------ #
#  Tests: wavelet filter (optional dependency)
# ------------------------------------------------------------------ #

class TestWaveletFilter:
    def test_wavelet_reduces_noise(self, synthetic_signals):
        pywt = pytest.importorskip("pywt")
        s = synthetic_signals
        T, N = s['T'], s['N']

        sim = SimpleNamespace(
            filter_wavelet_name='db4',
            filter_wavelet_level=0,
            filter_wavelet_threshold='soft',
        )

        # Dummy spectra (not used by wavelet path)
        S_signal = np.ones((N, T // 2 + 1))
        S_noise_eff = np.ones((N, T // 2 + 1))

        y_filtered = _apply_filter(
            'wavelet', s['y_noisy'], S_signal, S_noise_eff,
            fraction=1.0, h_floor=0.01, T=T, N=N, sim=sim,
        )

        mse_noisy = np.mean((s['y_noisy'] - s['y_clean']) ** 2)
        mse_filtered = np.mean((y_filtered - s['y_clean']) ** 2)
        assert mse_filtered < mse_noisy


# ------------------------------------------------------------------ #
#  Integration: end-to-end filter pipeline (no model)
# ------------------------------------------------------------------ #

class TestEndToEndFilterPipeline:
    """Test the full filter pipeline math without model or disk I/O."""

    def test_wiener_pipeline(self, synthetic_signals):
        """Simulate the full wiener_filter_derivatives flow with synthetic data."""
        s = synthetic_signals
        T, N, dt = s['T'], s['N'], s['dt']
        sigma_meas = s['sigma_meas']

        freqs = np.fft.rfftfreq(T, d=dt)
        unique_types = np.unique(s['neuron_types'])

        # Step 1: analytical noise spectrum
        S_noise = _analytical_noise_spectrum(freqs, sigma_meas, dt)
        assert S_noise[0] == 0.0

        # Step 2: estimate signal spectrum from predictions
        S_signal, S_noisy = _estimate_signal_spectrum(
            s['y_pred'], s['y_noisy'], s['neuron_types'], unique_types,
            freqs, T, N, per_type=True, smooth_bins=5,
        )

        # Step 3: apply Wiener filter with conservative fraction
        S_noise_eff = np.broadcast_to(S_noise[np.newaxis, :], (N, len(freqs))).copy()
        sim = SimpleNamespace(
            filter_wavelet_name='db4',
            filter_wavelet_level=0,
            filter_wavelet_threshold='soft',
        )

        y_filtered = _apply_filter(
            'wiener', s['y_noisy'], S_signal, S_noise_eff,
            fraction=0.5, h_floor=0.05, T=T, N=N, sim=sim,
        )

        # Step 4: metrics
        metrics = _compute_metrics(s['y_noisy'], y_filtered, s['y_pred'], s['y_clean'])

        # Wiener filter should improve MSE and R2
        assert metrics['mse_filtered'] < metrics['mse_noisy']
        assert metrics['r2_filtered'] > metrics['r2_noisy']

        # Filtered R2 should be meaningfully better
        r2_improvement = metrics['r2_filtered'] - metrics['r2_noisy']
        assert r2_improvement > 0.01, f"Expected meaningful R2 improvement, got {r2_improvement:.4f}"
