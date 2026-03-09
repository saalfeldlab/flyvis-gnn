"""Generate the 3-panel R² bar plot comparing default vs agentic-optimized configs.

Standalone script — run once to produce assets/Fig_r2_barplot.png.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────

EXPLORATIONS = {
    "noise_free": {
        "label": "Noise-free ($\\sigma = 0$)",
        "log_dir": "LLM_flyvis_noise_free",
        "log_prefix": "flyvis_noise_free_Claude",
        "default_metrics": "flyvis_noise_free",
    },
    "noise_005": {
        "label": "Low noise ($\\sigma = 0.05$)",
        "log_dir": "LLM_flyvis_noise_005",
        "log_prefix": "flyvis_noise_005_Claude",
        "default_metrics": "flyvis_noise_005",
    },
    "noise_05": {
        "label": "High noise ($\\sigma = 0.5$)",
        "log_dir": "LLM_flyvis_noise_05",
        "log_prefix": "flyvis_noise_05_Claude",
        "default_metrics": "flyvis_noise_05",
    },
}

BASE = "/workspace/flyvis-gnn"
EXPLORATION_ROOT = os.path.join(BASE, "log", "Claude_exploration")
FLY_ROOT = os.path.join(BASE, "log", "fly")

METRICS = ["conn $R^2$", r"$\tau$ $R^2$", "$V_{rest}$ $R^2$"]
METRIC_KEYS = ["connectivity_R2", "tau_R2", "V_rest_R2"]


def parse_analysis_log(path):
    """Extract connectivity_R2, tau_R2, V_rest_R2 from an analysis log."""
    vals = {}
    with open(path) as f:
        for line in f:
            for key in METRIC_KEYS:
                if line.startswith(f"{key}:"):
                    vals[key] = float(line.split(":")[1].strip())
    return vals


def get_default_r2(dataset_name):
    """Get final R2 from default config metrics.log (single seed)."""
    metrics_path = os.path.join(FLY_ROOT, dataset_name, "tmp_training", "metrics.log")
    last_line = ""
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("iteration"):
                last_line = line
    # format: iteration,connectivity_r2,vrest_r2,tau_r2
    parts = last_line.split(",")
    return {
        "connectivity_R2": float(parts[1]),
        "V_rest_R2": float(parts[2]),
        "tau_R2": float(parts[3]),
    }


# Collect data
data = {}
for key, info in EXPLORATIONS.items():
    # Default (single seed)
    default = get_default_r2(info["default_metrics"])

    # Agentic (4 seeds)
    agentic_seeds = []
    for seed in range(4):
        log_path = os.path.join(
            EXPLORATION_ROOT, info["log_dir"],
            f"{info['log_prefix']}_{seed:02d}_analysis.log",
        )
        if os.path.exists(log_path):
            agentic_seeds.append(parse_analysis_log(log_path))

    data[key] = {"default": default, "agentic": agentic_seeds, "info": info}

# ── Plot ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

c_default = "#AAAAAA"
c_agentic = "#2ca02c"
bar_width = 0.32

for panel_idx, (key, d) in enumerate(data.items()):
    ax = axes[panel_idx]
    info = d["info"]
    default = d["default"]
    seeds = d["agentic"]

    n_metrics = len(METRICS)
    x = np.arange(n_metrics)

    # Default values (single seed — no error bar)
    default_vals = [default[k] for k in METRIC_KEYS]

    # Agentic values (4 seeds — mean ± std)
    agentic_matrix = np.array([[s[k] for k in METRIC_KEYS] for s in seeds])
    agentic_mean = agentic_matrix.mean(axis=0)
    agentic_std = agentic_matrix.std(axis=0)

    bars_def = ax.bar(
        x - bar_width / 2, default_vals, bar_width,
        color=c_default, edgecolor="white", linewidth=1.2,
        label="Default config" if panel_idx == 0 else None,
    )
    bars_agt = ax.bar(
        x + bar_width / 2, agentic_mean, bar_width,
        yerr=agentic_std, capsize=4, color=c_agentic,
        edgecolor="white", linewidth=1.2,
        label="Agentic-optimized" if panel_idx == 0 else None,
        error_kw=dict(lw=1.5, capthick=1.5),
    )

    # Value annotations
    for bar in bars_def:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#555")
    for i, bar in enumerate(bars_agt):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + agentic_std[i] + 0.02,
                f"{agentic_mean[i]:.3f}", ha="center", va="bottom", fontsize=9,
                color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("$R^2$", fontsize=14)
    ax.set_title(info["label"], fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)

    # Iteration count
    n_iters = len([
        f for f in os.listdir(os.path.join(EXPLORATION_ROOT, info["log_dir"], "r2_trajectory"))
        if f.startswith("iter_") and f.endswith(".log")
    ])
    ax.text(0.02, 0.97, f"{n_iters} iterations", transform=ax.transAxes,
            fontsize=10, va="top", color="#888")

# Legend only in left panel
axes[0].legend(fontsize=11, loc="upper left", frameon=False,
               bbox_to_anchor=(0.0, 0.92))

plt.tight_layout()
out_path = os.path.join(BASE, "assets", "Fig_r2_barplot.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
