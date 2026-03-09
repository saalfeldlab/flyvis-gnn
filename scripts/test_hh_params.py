#!/usr/bin/env python3
"""Standalone HH parameter sweep: compare old vs new defaults.

Uses realistic connectome weight budget AND convergence.
Shows the effect of each parameter fix incrementally.

Usage:
    python scripts/test_hh_params.py
"""

import sys
sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flyvis_gnn.generators.flyvis_hodgkin_huxley_ode import FlyVisHodgkinHuxleyODE
from flyvis_gnn.generators.ode_params import FlyVisHodgkinHuxleyODEParams


def build_network():
    """Build a network matching real flyvis connectivity statistics.

    Real connectome: total exc W≈1.5/neuron, in-degree≈19-32.
    Layout (80 neurons):
        L0 (photoreceptors): 0-29, L1 (lamina): 30-59, L2 (medulla): 60-79
    """
    n_neurons = 80
    edges, weights = [], []
    rng = np.random.RandomState(42)

    for dst in range(30, 60):
        n_in = rng.randint(20, 30)
        for src in rng.choice(30, size=n_in, replace=False):
            w = rng.exponential(1.5 / (n_in * 0.7)) if rng.rand() < 0.7 else -rng.exponential(0.8 / (n_in * 0.3))
            edges.append([src, dst]); weights.append(w)

    for dst in range(60, 80):
        n_in = rng.randint(20, 30)
        for src in rng.choice(range(30, 60), size=n_in, replace=False):
            w = rng.exponential(1.5 / (n_in * 0.7)) if rng.rand() < 0.7 else -rng.exponential(0.8 / (n_in * 0.3))
            edges.append([src, dst]); weights.append(w)

    for src in range(30, 60):
        for dst in rng.choice([n for n in range(30, 60) if n != src], size=rng.randint(5, 10), replace=False):
            edges.append([src, dst]); weights.append(rng.exponential(0.04) * (1 if rng.rand() < 0.5 else -1))

    ei = torch.tensor(edges, dtype=torch.long).t()
    W = torch.tensor(weights, dtype=torch.float32)

    w_np = np.array(weights)
    dst_arr = np.array([e[1] for e in edges])
    print(f"Network: {n_neurons} neurons, {len(weights)} edges, |W| mean={np.abs(w_np).mean():.3f}")
    for name, lo, hi in [("L1", 30, 60), ("L2", 60, 80)]:
        exc = [w_np[(dst_arr == n) & (w_np > 0)].sum() for n in range(lo, hi)]
        print(f"  {name} exc W/neuron: mean={np.mean(exc):.2f}")
    return n_neurons, ei, W


def run_simulation(n_neurons, edge_index, W, overrides,
                   stim_neurons=range(0, 30), stim_value=1.0,
                   dt=0.01, t_total=100.0, warmup=30.0, w_scale=1.0):
    """Run HH with warmup then stimulus. w_scale applied to W before simulation."""
    W_scaled = W * w_scale
    params = FlyVisHodgkinHuxleyODEParams.from_defaults(
        n_neurons, edge_index, W_scaled, overrides=overrides)
    ode = FlyVisHodgkinHuxleyODE(params, device="cpu")
    state = ode.init_state(n_neurons)
    state.stimulus = torch.zeros(n_neurons)

    for _ in range(int(warmup / dt)):
        dv = ode(state, edge_index)
        state.voltage = state.voltage + dt * dv.squeeze()
        ode.step_gates(state, dt)

    for sn in stim_neurons:
        state.stimulus[sn] = stim_value

    n_steps = int(t_total / dt)
    voltages = np.zeros((n_neurons, n_steps))
    for step in range(n_steps):
        dv = ode(state, edge_index)
        state.voltage = state.voltage + dt * dv.squeeze()
        ode.step_gates(state, dt)
        voltages[:, step] = state.voltage.detach().numpy()

    return np.arange(n_steps) * dt, voltages


CONFIGS = [
    ("OLD: syn_v_half=−20, I_bias=0, stim=10, w_scale=1",
     dict(syn_v_half=-20.0, I_bias=0.0, stim_scale=10.0), 1.0),
    ("+ syn_v_half=−45",
     dict(syn_v_half=-45.0, I_bias=0.0, stim_scale=10.0), 1.0),
    ("+ I_bias=3",
     dict(syn_v_half=-45.0, I_bias=3.0, stim_scale=10.0), 1.0),
    ("+ stim_scale=50",
     dict(syn_v_half=-45.0, I_bias=3.0, stim_scale=50.0), 1.0),
    ("+ w_scale=2  (NEW defaults)",
     dict(syn_v_half=-45.0, I_bias=3.0, stim_scale=50.0), 2.0),
]

LAYER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
LAYER_NAMES = ["L0 photo", "L1", "L2"]
LAYER_RANGES = [range(0, 30), range(30, 60), range(60, 80)]


def main():
    n_neurons, edge_index, W = build_network()

    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(16, 3.5 * len(CONFIGS)),
                             sharex=True, sharey=True)

    for ax, (label, overrides, w_scale) in zip(axes, CONFIGS):
        print(f"Running: {label} ...")
        times, voltages = run_simulation(
            n_neurons, edge_index, W, overrides, w_scale=w_scale)

        spike_parts, range_parts = [], []
        for lrange, color, lname in zip(LAYER_RANGES, LAYER_COLORS, LAYER_NAMES):
            lv = voltages[list(lrange)]
            for row in lv:
                ax.plot(times, row, color=color, linewidth=0.15, alpha=0.12)
            ax.plot(times, lv.mean(axis=0), color=color, linewidth=1.8,
                    label=lname, alpha=0.95)

            total_spikes = sum(np.sum((r[1:] > 0) & (r[:-1] <= 0)) for r in lv)
            spike_parts.append(f"{lname}:{total_spikes}")
            range_parts.append(f"{lname}:[{lv.min():.0f},{lv.max():.0f}]")

        ax.set_title(f"{label}    spikes: {'  '.join(spike_parts)}    "
                     f"V: {'  '.join(range_parts)}",
                     fontsize=9, loc="left", fontfamily="monospace")
        ax.set_ylabel("V (mV)")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.7)

    axes[-1].set_xlabel("time (ms)")
    axes[0].set_ylim(-90, 60)

    fig.suptitle("HH parameter sweep — incremental fixes, realistic connectome weights",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = "scripts/hh_param_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
