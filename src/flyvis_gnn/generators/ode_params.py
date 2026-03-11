"""ODE parameter classes and registry for flyvis-gnn.

Maps config signal_model_name strings to ODE parameter dataclasses.
Each ODE_params_class knows how to construct itself from a source,
save/load to disk, and expose its fields by name.

Usage:
    @register_ode_params("flyvis_A", "flyvis_B")
    class FlyVisODEParams(ODEParamsBase):
        ...

    ODE_params_class = get_ode_params_class("flyvis_A")
    p = ODE_params_class.from_flyvis_network(net, device=device)
    p.save(folder)
    p = ODE_params_class.load(folder)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any

import torch

from flyvis_gnn.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ODE_PARAMS_REGISTRY: dict[str, type] = {}


def register_ode_params(*names: str):
    """Class decorator that registers an ODE params class under config names."""
    def decorator(cls):
        for name in names:
            if name in _ODE_PARAMS_REGISTRY:
                raise ValueError(
                    f"ODE params name '{name}' already registered to "
                    f"{_ODE_PARAMS_REGISTRY[name].__name__}"
                )
            _ODE_PARAMS_REGISTRY[name] = cls
        return cls
    return decorator


def get_ode_params_class(name: str) -> type:
    """Look up ODE params class by config signal_model_name."""
    if name not in _ODE_PARAMS_REGISTRY:
        available = sorted(_ODE_PARAMS_REGISTRY.keys())
        raise KeyError(
            f"Unknown ODE params '{name}'. Available: {available}"
        )
    return _ODE_PARAMS_REGISTRY[name]


def list_ode_params() -> list[str]:
    """Return sorted list of all registered ODE params names."""
    return sorted(_ODE_PARAMS_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class ODEParamsBase:
    """Base class for ODE parameter dataclasses.

    Provides to(), clone(), save(), load(), and dict-style access
    for backward compatibility (p["tau_i"] still works).
    """

    def __getitem__(self, key: str) -> Any:
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """Dict-style assignment for backward compatibility."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Support `"key" in params`."""
        return hasattr(self, key) and getattr(self, key) is not None

    def __iter__(self):
        """Iterate over field names (for `for key in params`)."""
        return iter(f.name for f in dc_fields(self))

    def to(self, device: torch.device) -> ODEParamsBase:
        """Move all tensor fields to device."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(device))
        return self

    def clone(self) -> ODEParamsBase:
        """Deep clone all tensor fields."""
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            kwargs[f.name] = val.clone() if isinstance(val, torch.Tensor) else val
        return self.__class__(**kwargs)

    def save(self, folder: str):
        """Save all fields as a single ode_params.pt dict."""
        os.makedirs(folder, exist_ok=True)
        state = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                state[f.name] = val.cpu()
            else:
                state[f.name] = val
        torch.save(state, os.path.join(folder, "ode_params.pt"))

    @classmethod
    def load(cls, folder: str, device: torch.device | str = "cpu"):
        """Load from ode_params.pt, or fall back to legacy individual .pt files."""
        unified_path = os.path.join(folder, "ode_params.pt")
        if os.path.exists(unified_path):
            state = torch.load(unified_path, map_location=device, weights_only=True)
            return cls(**state)
        return cls._load_legacy(folder, device)

    @classmethod
    def _load_legacy(cls, folder: str, device: torch.device | str = "cpu"):
        """Override in subclass to support legacy per-file loading."""
        raise FileNotFoundError(
            f"No ode_params.pt found at {folder} and no legacy loader defined "
            f"for {cls.__name__}"
        )


# ---------------------------------------------------------------------------
# FlyVis graded-voltage model params
# ---------------------------------------------------------------------------

@register_ode_params(
    "flyvis_A", "flyvis_B", "flyvis_C", "flyvis_D",
    "flyvis_A_multiple_ReLU", "flyvis_B_multiple_ReLU", "flyvis_C_multiple_ReLU",
    "flyvis_A_tanh", "flyvis_B_tanh", "flyvis_C_tanh",
    "flyvis_A_NULL", "flyvis_B_NULL", "flyvis_C_NULL",
    "flyvis_linear", "flyvis_linear_tanh",
)
@dataclass
class FlyVisODEParams(ODEParamsBase):
    """Parameters for the graded-voltage FlyVis ODE.

    Node params (indexed by neuron — one value per node in the graph):
        tau_i:     (N,) time constants
        V_i_rest:  (N,) resting potentials

    Edge params:
        edge_index: (2, E) source/destination indices
        w:          (E,) effective synaptic weights
    """
    tau_i: torch.Tensor = None       # (N,)
    V_i_rest: torch.Tensor = None    # (N,)
    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None           # (E,) effective synaptic weights

    @classmethod
    def from_flyvis_network(cls, net, device: torch.device | str = "cpu"):
        """Construct from a flyvis Network object."""
        params = net._param_api()
        tau_i = params.nodes.time_const
        V_i_rest = params.nodes.bias
        W = params.edges.syn_strength * params.edges.syn_count * params.edges.sign
        edge_index = torch.stack([
            torch.tensor(net.connectome.edges.source_index[:]),
            torch.tensor(net.connectome.edges.target_index[:]),
        ], dim=0)
        return cls(
            tau_i=tau_i.to(device),
            V_i_rest=V_i_rest.to(device),
            edge_index=edge_index.to(device),
            W=W.to(device),
        )

    @classmethod
    def _load_legacy(cls, folder: str, device: torch.device | str = "cpu"):
        """Load from legacy individual .pt files (taus.pt, V_i_rest.pt, etc.)."""
        def _load(name):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return torch.load(path, map_location=device, weights_only=True)
            return None

        tau_i = _load("taus.pt")
        V_i_rest = _load("V_i_rest.pt")
        W = _load("weights.pt")
        edge_index = _load("edge_index.pt")

        if tau_i is None and V_i_rest is None and W is None and edge_index is None:
            raise FileNotFoundError(
                f"No ode_params.pt or legacy .pt files found at {folder}"
            )

        logger.info(f"loaded legacy ODE params from {folder}")
        return cls(tau_i=tau_i, V_i_rest=V_i_rest, edge_index=edge_index, W=W)
