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


# ---------------------------------------------------------------------------
# FlyVis AdEx spiking model params
# ---------------------------------------------------------------------------

# Default values from Zerlaut et al. 2018 (AutoMind ADEX_NEURON_DEFAULTS_ZERLAUT).
# Units: mV, pF, nS, pA, ms, Hz.  Stored as dimensionless floats in those units.
ADEX_DEFAULTS = dict(
    # Membrane
    C=200.0,             # pF  — membrane capacitance
    g_L=10.0,            # nS  — leak conductance
    v_rest=-65.0,        # mV  — resting (leak reversal) potential
    v_thresh=-50.0,      # mV  — spike initiation threshold (exp onset)
    delta_T=2.0,         # mV  — exponential nonlinearity sharpness
    v_cut=0.0,           # mV  — hard spike cutoff for detection
    v_reset=-65.0,       # mV  — post-spike reset voltage
    t_refrac=5.0,        # ms  — absolute refractory period
    # Adaptation
    a=4.0,               # nS  — subthreshold adaptation coupling
    b=20.0,              # pA  — spike-triggered adaptation increment
    tau_w=500.0,         # ms  — adaptation time constant
    # Synaptic (COBA)
    E_ge=0.0,            # mV  — excitatory reversal potential
    E_gi=-80.0,          # mV  — inhibitory reversal potential
    Q_ge=1.0,            # nS  — excitatory quantal conductance
    Q_gi=5.0,            # nS  — inhibitory quantal conductance
    tau_ge=5.0,          # ms  — excitatory conductance decay
    tau_gi=5.0,          # ms  — inhibitory conductance decay
    # Synaptic (CUBA) — no defaults from Zerlaut, set to 0 as placeholder
    J_exc=0.0,           # mV  — excitatory spike kick
    J_inh=0.0,           # mV  — inhibitory spike kick
    # External input
    I_bias=0.0,          # pA  — constant bias current
    stim_scale=1.0,      # pA per unit stimulus — converts visual input to current
    # Initial conditions
    v_0_mean=0.0,        # mV  — mean offset from v_rest for initial v
    v_0_std=4.0,         # mV  — std of initial v perturbation
)


@register_ode_params("flyvis_adex_coba", "flyvis_adex_cuba")
@dataclass
class FlyVisSpikingODEParams(ODEParamsBase):
    """Parameters for the AdEx spiking FlyVis ODE.

    Per-neuron static params (indexed by neuron, one value per node):
        Membrane: C, g_L, v_rest, v_thresh, delta_T, v_cut, v_reset, t_refrac
        Adaptation: a, b, tau_w

    Per-neuron synaptic params:
        COBA: E_ge, E_gi, Q_ge, Q_gi, tau_ge, tau_gi
        CUBA: J_exc, J_inh

    Per-neuron external input:
        I_bias, stim_scale

    Network topology:
        edge_index: (2, E) source/destination indices
        is_excitatory: (N,) bool — True for excitatory neurons

    Synapse model selector:
        synapse_model: "COBA" or "CUBA"
    """
    # Membrane — (N,) per neuron
    C: torch.Tensor = None
    g_L: torch.Tensor = None
    v_rest: torch.Tensor = None
    v_thresh: torch.Tensor = None
    delta_T: torch.Tensor = None
    v_cut: torch.Tensor = None
    v_reset: torch.Tensor = None
    t_refrac: torch.Tensor = None

    # Adaptation — (N,)
    a: torch.Tensor = None
    b: torch.Tensor = None
    tau_w: torch.Tensor = None

    # Synaptic COBA — (N,)
    E_ge: torch.Tensor = None
    E_gi: torch.Tensor = None
    Q_ge: torch.Tensor = None
    Q_gi: torch.Tensor = None
    tau_ge: torch.Tensor = None
    tau_gi: torch.Tensor = None

    # Synaptic CUBA — (N,)
    J_exc: torch.Tensor = None
    J_inh: torch.Tensor = None

    # External input — (N,)
    I_bias: torch.Tensor = None
    stim_scale: torch.Tensor = None

    # Initial conditions (scalars, not per-neuron)
    v_0_mean: float = 0.0
    v_0_std: float = 4.0

    # Topology
    edge_index: torch.Tensor = None       # (2, E)
    is_excitatory: torch.Tensor = None    # (N,) bool

    # Synapse model selector
    synapse_model: str = "COBA"

    @classmethod
    def from_defaults(cls, n_neurons: int, is_excitatory: torch.Tensor,
                      edge_index: torch.Tensor, synapse_model: str = "COBA",
                      device: torch.device | str = "cpu",
                      overrides: dict | None = None) -> FlyVisSpikingODEParams:
        """Construct from Zerlaut defaults with per-neuron expansion.

        Args:
            n_neurons: total number of neurons
            is_excitatory: (N,) bool tensor — True for excitatory neurons
            edge_index: (2, E) connectivity
            synapse_model: "COBA" or "CUBA"
            device: target device
            overrides: dict of param_name -> value to override defaults
        """
        d = {**ADEX_DEFAULTS}
        if overrides:
            d.update(overrides)

        def _expand(val):
            return torch.full((n_neurons,), val, dtype=torch.float32, device=device)

        return cls(
            C=_expand(d["C"]),
            g_L=_expand(d["g_L"]),
            v_rest=_expand(d["v_rest"]),
            v_thresh=_expand(d["v_thresh"]),
            delta_T=_expand(d["delta_T"]),
            v_cut=_expand(d["v_cut"]),
            v_reset=_expand(d["v_reset"]),
            t_refrac=_expand(d["t_refrac"]),
            a=_expand(d["a"]),
            b=_expand(d["b"]),
            tau_w=_expand(d["tau_w"]),
            E_ge=_expand(d["E_ge"]),
            E_gi=_expand(d["E_gi"]),
            Q_ge=_expand(d["Q_ge"]),
            Q_gi=_expand(d["Q_gi"]),
            tau_ge=_expand(d["tau_ge"]),
            tau_gi=_expand(d["tau_gi"]),
            J_exc=_expand(d["J_exc"]),
            J_inh=_expand(d["J_inh"]),
            I_bias=_expand(d["I_bias"]),
            stim_scale=_expand(d["stim_scale"]),
            v_0_mean=d["v_0_mean"],
            v_0_std=d["v_0_std"],
            edge_index=edge_index.to(device),
            is_excitatory=is_excitatory.to(device),
            synapse_model=synapse_model,
        )

    @classmethod
    def from_flyvis_network(cls, net, synapse_model: str = "COBA",
                            device: torch.device | str = "cpu",
                            overrides: dict | None = None) -> FlyVisSpikingODEParams:
        """Construct from a flyvis Network, using Zerlaut defaults for AdEx params.

        E/I identity is inferred from the sign of synaptic weights:
        neurons with net positive outgoing weight are excitatory.
        """
        params = net._param_api()
        W = (params.edges.syn_strength * params.edges.syn_count * params.edges.sign)
        W = W.detach().cpu().float()
        src_idx = net.connectome.edges.source_index[:]
        dst_idx = net.connectome.edges.target_index[:]
        # Ensure numpy conversion for torch.tensor to always produce CPU tensors
        if hasattr(src_idx, 'cpu'):
            src_idx = src_idx.detach().cpu().numpy()
        if hasattr(dst_idx, 'cpu'):
            dst_idx = dst_idx.detach().cpu().numpy()
        edge_index = torch.stack([
            torch.tensor(src_idx, dtype=torch.long),
            torch.tensor(dst_idx, dtype=torch.long),
        ], dim=0)

        n_neurons = len(params.nodes.time_const)
        src = edge_index[0]

        # Infer E/I from net outgoing weight sign per neuron
        sum_w = torch.zeros(n_neurons)
        sum_w.scatter_add_(0, src, W)
        is_excitatory = (sum_w >= 0)

        return cls.from_defaults(
            n_neurons=n_neurons,
            is_excitatory=is_excitatory,
            edge_index=edge_index,
            synapse_model=synapse_model,
            device=device,
            overrides=overrides,
        )
