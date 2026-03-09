"""Neuron state dataclasses for flyvis simulation.

Replaces the packed (N, 9) tensor with named fields.
Follows the zapbench pattern: data loads directly into dataclass fields,
classmethods handle I/O, no raw tensor layout leaks outside.

All fields default to None so callers can load only what they need
(e.g. fields=['index', 'voltage', 'stimulus'] for training).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# Field classification — used by from_zarr_v3 to pick dtype/shape.
STATIC_FIELDS = {'index', 'pos', 'group_type', 'neuron_type'}
DYNAMIC_FIELDS = {'voltage', 'stimulus', 'calcium', 'fluorescence', 'noise'}
ALL_FIELDS = STATIC_FIELDS | DYNAMIC_FIELDS


def _apply(tensor, fn):
    """Apply fn to tensor if not None, else return None."""
    return fn(tensor) if tensor is not None else None


@dataclass
class NeuronState:
    """Single-frame neuron state.

    Static fields (set once, never change per frame):
        index, pos, group_type, neuron_type

    Dynamic fields (updated every simulation frame):
        voltage, stimulus, calcium, fluorescence

    All fields default to None — only populated fields are used.
    """

    # static
    index: torch.Tensor | None = None        # (N,) long — neuron IDs 0..N-1
    pos: torch.Tensor | None = None          # (N, 2) float32 — spatial (x, y)
    group_type: torch.Tensor | None = None   # (N,) long — grouped neuron type
    neuron_type: torch.Tensor | None = None  # (N,) long — integer neuron type

    # dynamic
    voltage: torch.Tensor | None = None      # (N,) float32 — membrane voltage u
    stimulus: torch.Tensor | None = None     # (N,) float32 — visual input / excitation
    calcium: torch.Tensor | None = None      # (N,) float32 — calcium concentration
    fluorescence: torch.Tensor | None = None # (N,) float32 — fluorescence readout
    noise: torch.Tensor | None = None       # (N,) float32 — measurement noise

    @property
    def n_neurons(self) -> int:
        """Infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("NeuronState has no populated fields")

    @property
    def device(self) -> torch.device:
        """Infer device from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("NeuronState has no populated fields")

    def observable(self, calcium_type: str = "none") -> torch.Tensor:
        """Return the observable signal: voltage or calcium, as (N, 1)."""
        if calcium_type != "none":
            return self.calcium.unsqueeze(-1)
        return self.voltage.unsqueeze(-1)

    @classmethod
    def from_numpy(cls, x: np.ndarray) -> NeuronState:
        """Create from legacy (N, 9) numpy array.

        Column layout: [index, xpos, ypos, voltage, stimulus,
                        group_type, neuron_type, calcium, fluorescence]
        """
        t = torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
        return cls(
            index=t[:, 0].long(),
            pos=t[:, 1:3].float(),
            group_type=t[:, 5].long(),
            neuron_type=t[:, 6].long(),
            voltage=t[:, 3].float(),
            stimulus=t[:, 4].float(),
            calcium=t[:, 7].float(),
            fluorescence=t[:, 8].float(),
        )

    def to_packed(self) -> torch.Tensor:
        """Pack back into (N, 9) tensor for legacy compatibility."""
        x = torch.zeros(self.n_neurons, 9, dtype=torch.float32, device=self.device)
        if self.index is not None:
            x[:, 0] = self.index.float()
        if self.pos is not None:
            x[:, 1:3] = self.pos
        if self.voltage is not None:
            x[:, 3] = self.voltage
        if self.stimulus is not None:
            x[:, 4] = self.stimulus
        if self.group_type is not None:
            x[:, 5] = self.group_type.float()
        if self.neuron_type is not None:
            x[:, 6] = self.neuron_type.float()
        if self.calcium is not None:
            x[:, 7] = self.calcium
        if self.fluorescence is not None:
            x[:, 8] = self.fluorescence
        return x

    def to(self, device: torch.device) -> NeuronState:
        """Move all non-None tensors to device."""
        return NeuronState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def clone(self) -> NeuronState:
        """Deep clone all non-None tensors."""
        return NeuronState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.clone())
            for f in dc_fields(self)
        })

    def detach(self) -> NeuronState:
        """Detach all non-None tensors from computation graph."""
        return NeuronState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.detach())
            for f in dc_fields(self)
        })

    def subset(self, ids) -> NeuronState:
        """Select a subset of neurons by index."""
        return NeuronState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t[ids])
            for f in dc_fields(self)
        })

    @classmethod
    def zeros(cls, n_neurons: int, device: torch.device = None) -> NeuronState:
        """Create zero-initialized NeuronState (all fields populated)."""
        return cls(
            index=torch.arange(n_neurons, dtype=torch.long, device=device),
            pos=torch.zeros(n_neurons, 2, dtype=torch.float32, device=device),
            group_type=torch.zeros(n_neurons, dtype=torch.long, device=device),
            neuron_type=torch.zeros(n_neurons, dtype=torch.long, device=device),
            voltage=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            calcium=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        )


@dataclass
class NeuronTimeSeries:
    """Full simulation timeseries — static metadata + dynamic per-frame data.

    Static fields are stored once (same for all frames).
    Dynamic fields have a leading time dimension (T, N).

    All fields default to None — only populated fields are used.
    """

    # static (stored once)
    index: torch.Tensor | None = None        # (N,)
    pos: torch.Tensor | None = None          # (N, 2)
    group_type: torch.Tensor | None = None   # (N,)
    neuron_type: torch.Tensor | None = None  # (N,)

    # dynamic (stored per frame)
    voltage: torch.Tensor | None = None      # (T, N)
    stimulus: torch.Tensor | None = None     # (T, N)
    calcium: torch.Tensor | None = None      # (T, N)
    fluorescence: torch.Tensor | None = None # (T, N)
    noise: torch.Tensor | None = None        # (T, N) — measurement noise

    @property
    def n_frames(self) -> int:
        """Infer T from the first non-None dynamic field."""
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        raise ValueError("NeuronTimeSeries has no dynamic fields")

    @property
    def n_neurons(self) -> int:
        """Infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[-1] if f.name in DYNAMIC_FIELDS else val.shape[0]
        raise ValueError("NeuronTimeSeries has no populated fields")

    @property
    def xnorm(self) -> torch.Tensor:
        """Voltage normalization: 1.5 * std of all valid voltage values."""
        v = self.voltage
        valid = v[~torch.isnan(v)]
        if len(valid) > 0:
            return 1.5 * valid.std()
        return torch.tensor(1.0, device=v.device if v is not None else 'cpu')

    def frame(self, t: int) -> NeuronState:
        """Extract single-frame NeuronState at time t.

        Static fields are shared (not cloned).
        Dynamic fields are cloned so the caller can modify them
        without corrupting the timeseries data.
        """
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in DYNAMIC_FIELDS:
                kwargs[f.name] = val[t].clone()
            else:
                kwargs[f.name] = val
        return NeuronState(**kwargs)

    def to(self, device: torch.device) -> NeuronTimeSeries:
        """Move all non-None tensors to device."""
        return NeuronTimeSeries(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def subset_neurons(self, ids: np.ndarray | torch.Tensor) -> NeuronTimeSeries:
        """Select a subset of neurons by index."""
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in DYNAMIC_FIELDS:
                kwargs[f.name] = val[:, ids]
            else:
                kwargs[f.name] = val[ids]
        return NeuronTimeSeries(**kwargs)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> NeuronTimeSeries:
        """Create from legacy (T, N, 9) numpy array.

        Column layout: [index, xpos, ypos, voltage, stimulus,
                        group_type, neuron_type, calcium, fluorescence]
        """
        t = torch.from_numpy(arr) if not isinstance(arr, torch.Tensor) else arr
        return cls(
            # static — take from first frame
            index=t[0, :, 0].long(),
            pos=t[0, :, 1:3].float(),
            group_type=t[0, :, 5].long(),
            neuron_type=t[0, :, 6].long(),
            # dynamic — all frames
            voltage=t[:, :, 3].float(),
            stimulus=t[:, :, 4].float(),
            calcium=t[:, :, 7].float(),
            fluorescence=t[:, :, 8].float(),
        )

    @classmethod
    def from_zarr_v3(cls, path: str | Path, fields: Sequence[str] | None = None) -> NeuronTimeSeries:
        """Load from V3 per-field zarr format.

        Args:
            path: directory containing per-field .zarr arrays
            fields: list of field names to load (e.g. ['voltage', 'stimulus']).
                    None means load all available fields.

        Expects per-field zarr arrays under path/:
            pos.zarr          — (N, 2) float32    (static)
            group_type.zarr   — (N,) int32        (static)
            neuron_type.zarr  — (N,) int32        (static)
            voltage.zarr      — (T, N) float32    (dynamic)
            stimulus.zarr     — (T, N) float32    (dynamic)
            calcium.zarr      — (T, N) float32    (dynamic)
            fluorescence.zarr — (T, N) float32    (dynamic)

        Note: index is not saved on disk — it is constructed as arange(n_neurons).
              Legacy data with index.zarr is still supported.
        """
        import tensorstore as ts

        path = Path(path)
        if fields is None:
            fields = list(ALL_FIELDS)

        def _read(name):
            zarr_path = path / f'{name}.zarr'
            if not zarr_path.exists():
                return None
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
            }
            return ts.open(spec).result().read().result()

        kwargs = {}
        for name in ALL_FIELDS:
            if name in fields:
                raw = _read(name)
                if raw is not None:
                    if name in ('index', 'group_type', 'neuron_type'):
                        kwargs[name] = torch.from_numpy(raw.copy()).long()
                    else:
                        kwargs[name] = torch.from_numpy(raw.copy()).float()
                else:
                    kwargs[name] = None
            else:
                kwargs[name] = None

        ts_obj = cls(**kwargs)

        # construct index as arange if not loaded from disk
        if ts_obj.index is None and 'index' in fields:
            ts_obj.index = torch.arange(ts_obj.n_neurons, dtype=torch.long)

        return ts_obj

    @classmethod
    def load(cls, path: str | Path, fields: Sequence[str] | None = None) -> NeuronTimeSeries:
        """Load simulation data. V3 zarr is the primary format.

        Falls back to .npy for legacy data.

        Args:
            path: base path (directory for V3, or path without extension for npy)
            fields: field names to load (V3 only). None = all.
        """
        path = Path(path)
        base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

        # V3 zarr (per-field arrays)
        if base_path.exists() and base_path.is_dir():
            if (base_path / 'voltage.zarr').exists():
                return cls.from_zarr_v3(base_path, fields=fields)

        # npy fallback (loads all fields, ignores fields param)
        npy_path = Path(str(base_path) + '.npy')
        if npy_path.exists():
            return cls.from_numpy(np.load(npy_path))

        raise FileNotFoundError(f"no V3 zarr or .npy found at {base_path}")
