"""PyTorch Dataset and Sampler for FlyVis GNN training.

Wraps the in-memory NeuronTimeSeries with a proper Dataset interface
and provides a reproducible frame sampler to replace bare np.random.randint.

Design choice: This Dataset is NOT used with a DataLoader + collate_fn.
The training loop still manually collects frames and calls _batch_frames(),
because batch assembly requires the shared edge_index tensor and the
regularizer needs per-frame model access. The Dataset replaces only the
frame extraction + target construction logic.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class FlyVisDataset(Dataset):
    """Dataset wrapping an in-memory NeuronTimeSeries.

    All data lives on GPU. __getitem__ returns frame indices into the
    timeseries; no disk I/O occurs.

    Args:
        x_ts: NeuronTimeSeries on GPU — (T, N) voltage/stimulus/etc.
        y_ts: numpy array — derivative targets, shape (T, N, 1)
        config: NeuralGraphConfig
        model: FlyVisGNN (needed for forward_visual if has_visual_field)
    """

    def __init__(self, x_ts, y_ts, config, model=None):
        self.x_ts = x_ts
        self.y_ts = y_ts
        self.config = config
        self.model = model

        tc = config.training
        sim = config.simulation
        self.time_step = tc.time_step
        self.time_window = tc.time_window
        self.n_frames = sim.n_frames
        self.recurrent = tc.recurrent_training
        self.neural_ode = tc.neural_ODE_training
        self.has_visual_field = 'visual' in config.graph_model.field_type
        self.test_neural_field = 'test' in config.graph_model.field_type
        self.n_input_neurons = getattr(sim, 'n_input_neurons', 0)

        # Valid frame range: leave room for time_step + 4 safety margin
        # _min_k = time_window so we can look back; _max_k so k + time_step + 4 < n_frames
        self._min_k = self.time_window
        self._max_k = self.n_frames - 4 - self.time_step

        # Epoch-dependent state
        self.loss_noise_level = 0.0

    def __len__(self):
        """Number of valid frame indices."""
        return max(self._max_k - self._min_k, 0)

    def set_epoch(self, epoch):
        """Update epoch-dependent state (noise decay)."""
        self.loss_noise_level = self.config.training.loss_noise_level * (0.95 ** epoch)

    def get_frame(self, k):
        """Extract a single training sample at frame index k.

        Args:
            k: absolute frame index in [_min_k, _max_k)

        Returns:
            x: NeuronState at frame k (with visual field applied if enabled)
            y: target tensor, shape (N, 1)
            k: the frame index (passed through for downstream use)
        """
        if self.recurrent or self.neural_ode:
            k = k - k % self.time_step

        x = self.x_ts.frame(k)

        # Visual field injection
        if self.has_visual_field and self.model is not None:
            visual_input = self.model.forward_visual(x, k)
            x.stimulus[:self.model.n_input_neurons] = visual_input.squeeze(-1)
            x.stimulus[self.model.n_input_neurons:] = 0

        # Target construction
        if self.recurrent or self.neural_ode:
            y = self.x_ts.voltage[k + self.time_step].unsqueeze(-1)
        elif self.test_neural_field:
            y = self.x_ts.stimulus[k, :self.n_input_neurons].unsqueeze(-1)
        else:
            y = torch.tensor(self.y_ts[k], device=x.device)

        # Optional noise injection
        if self.loss_noise_level > 0:
            y = y + torch.randn_like(y) * self.loss_noise_level

        return x, y, k

    def __getitem__(self, idx):
        """Return (NeuronState, target_y, frame_k) for a valid frame index.

        idx is an offset in [0, len(self)), mapped to absolute frame k.
        """
        k = idx + self._min_k
        return self.get_frame(k)


class FlyVisFrameSampler(Sampler):
    """Reproducible random frame sampler with per-epoch seeding.

    Replaces bare np.random.randint calls with a seeded RNG that produces
    reproducible frame index sequences. The seed changes per epoch to ensure
    different sampling order across epochs while remaining deterministic.

    Args:
        dataset: FlyVisDataset (used for valid range)
        num_samples: total frames to sample per epoch (= Niter * batch_size)
        seed: base random seed
    """

    def __init__(self, dataset, num_samples, seed=42):
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Update epoch for seed computation."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        n = len(self.dataset)
        if n <= 0:
            return iter([])
        indices = rng.randint(0, n, size=self.num_samples)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
