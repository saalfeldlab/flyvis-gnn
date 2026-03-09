# %% [raw]
# ---
# title: "Robustness to Measurement Noise"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Measurement Noise
# execute:
#   echo: false
# image: "graphs_data/fly/flyvis_noise_005_010/activity_traces_noisy.png"
# description: "Train the GNN on voltage observations corrupted by additive Gaussian measurement noise at five levels. Evaluate whether the model can still recover synaptic weights, biophysical parameters, and neuron-type identity from noisy observations."
# ---

# %% [markdown]
# ## Robustness to Measurement Noise
#
# In all previous experiments, the GNN received clean voltage traces
# from the simulator (up to intrinsic dynamics noise).  In a real
# experimental setting, however, voltage recordings are corrupted by
# **measurement noise** --- instrument noise, shot noise, or
# fluorescence noise in calcium imaging.  This notebook investigates
# how measurement noise degrades the GNN's ability to recover the
# circuit.
#
# ### Distinction Between Intrinsic and Measurement Noise
#
# Recall the simulated dynamics
# ([Notebook 00](Notebook_00.html)):
#
# $$\tau_i\frac{dv_i(t)}{dt} = -v_i(t) + V_i^{\text{rest}}
# + \sum_{j\in\mathcal{N}_i} \mathbf{W}_{ij}\,
# \text{ReLU}\!\big(v_j(t)\big) + I_i(t)
# + \sigma_{\text{dyn}}\,\xi_i(t),$$
#
# where $\sigma_{\text{dyn}}\,\xi_i(t)$ with
# $\xi_i(t) \sim \mathcal{N}(0,1)$ is **intrinsic dynamics noise**
# that drives stochastic fluctuations *within* the ODE.  This noise
# is part of the true dynamics and produces voltage trajectories
# that genuinely differ from the deterministic solution.
#
# **Measurement noise** is fundamentally different: it corrupts the
# *observations* of the voltage, not the dynamics themselves.  The
# GNN receives
#
# $$\tilde{v}_i(t) = v_i(t) + \sigma_{\text{meas}}\,\eta_i(t),
# \qquad \eta_i(t) \sim \mathcal{N}(0,1),$$
#
# and the derivative targets become
#
# $$\frac{d\widetilde{v}}{dt} \approx
# \frac{\tilde{v}_i(t+\Delta t) - \tilde{v}_i(t)}{\Delta t},$$
#
# which amplifies the measurement noise by a factor
# $\sim 1/\Delta t$.  Both the input voltage and the derivative
# targets are noisy, making this a harder inverse problem than
# intrinsic noise alone.
#
# ### Experimental Setup
#
# We fix the intrinsic dynamics noise at
# $\sigma_{\text{dyn}} = 0.05$ (the same as the baseline
# [Notebook 04](Notebook_04.html) low-noise condition) and vary the
# measurement noise level across five conditions:
#
# | Config | $\sigma_{\text{meas}}$ | Voltage SNR | Derivative SNR |
# |:--|:--|:--|:--|
# | `flyvis_noise_005_002` | 0.02 | high | moderate |
# | `flyvis_noise_005_004` | 0.04 | moderate | low |
# | `flyvis_noise_005_006` | 0.06 | moderate | low |
# | `flyvis_noise_005_008` | 0.08 | low | very low |
# | `flyvis_noise_005_010` | 0.10 | low | very low |
#
# To change the intrinsic noise level, edit `noise_model_level` in
# the respective config files under `config/fly/`.

# %%
#| output: false
import glob
import os
import warnings

from IPython.display import Image, Markdown, display

sys_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
import sys
sys.path.insert(0, sys_path)

from GNN_PlotFigure import data_plot
from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_test
from flyvis_gnn.plot import plot_loss_from_file
from flyvis_gnn.utils import set_device, add_pre_folder, graphs_data_path, log_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    if os.path.isfile(path):
        display(Image(filename=path, width=width))
    else:
        display(Markdown(f"*Image not found: `{os.path.basename(path)}`*"))


# %% [markdown]
# ## Results
#
# Each config shares the same intrinsic dynamics noise
# ($\sigma_{\text{dyn}} = 0.05$) and GNN architecture.  Only the
# measurement noise level $\sigma_{\text{meas}}$ varies.
#
# ### Optimized Training Hyperparameters
#
# A systematic LLM-driven exploration on the
# $\sigma_{\text{meas}} = 0.04$ condition (36 iterations across
# 5 intervention categories) identified noise-robust
# hyperparameters that significantly improve connectivity
# recovery under measurement noise:
#
# | Parameter | Default | Optimized | Rationale |
# |:--|:--|:--|:--|
# | `batch_size` | 4 | **6** | Larger batches average out noisy gradients |
# | `data_augmentation_loop` | 35 | **30** | Balances noise averaging with convergence |
# | `coeff_g_phi_diff` | 750 | **1200** | Stronger monotonicity constraint stabilizes $g_\phi$ |
#
# These parameters are applied uniformly across all five
# measurement noise conditions.  The exploration also established
# that **noise averaging (batch size $\times$ augmentation loop)
# is the dominant lever** --- all other interventions tested
# (LR scheduling, recurrent training, derivative smoothing,
# stronger L1/L2 regularization, $f_\theta$ message monotonicity)
# either degraded or catastrophically broke training.
#
# On the $\sigma_{\text{meas}} = 0.04$ condition, the optimized
# config achieves connectivity $R^2 = 0.925 \pm 0.003$
# (CV = 0.3\% across 4 seeds), compared to $R^2 \approx 0.82$
# with default parameters.
#
# Synaptic weight recovery decreases from
# $R^2 = 0.96$ at $\sigma_{\text{meas}} = 0.02$ to $0.76$ at
# $\sigma_{\text{meas}} = 0.10$, while time constants remain
# robust ($R^2 > 0.85$ across all conditions).
# Resting potentials are the most sensitive to measurement noise,
# dropping below $R^2 = 0.05$ for
# $\sigma_{\text{meas}} \geq 0.06$, consistent with the
# $\sim 1/\Delta t$ amplification of noise in the derivative
# targets that $V^{\text{rest}}$ depends on.

# %%
#| output: false
datasets = [
    ('flyvis_noise_005_002', '0.02', r'$\sigma_{\text{meas}} = 0.02$'),
    ('flyvis_noise_005_004', '0.04', r'$\sigma_{\text{meas}} = 0.04$'),
    ('flyvis_noise_005_006', '0.06', r'$\sigma_{\text{meas}} = 0.06$'),
    ('flyvis_noise_005_008', '0.08', r'$\sigma_{\text{meas}} = 0.08$'),
    ('flyvis_noise_005_010', '0.10', r'$\sigma_{\text{meas}} = 0.10$'),
]

config_root = "./config"
configs = {}
graphs_dirs = {}

for config_name, table_label, label in datasets:
    config_file, pre_folder = add_pre_folder(config_name)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name
    configs[config_name] = config
    graphs_dirs[config_name] = graphs_data_path(pre_folder + config_name)

device = set_device(configs[datasets[0][0]].training.device)

# Check that data exists
missing_data = []
for config_name, table_label, label in datasets:
    gdir = graphs_dirs[config_name]
    has_data = (os.path.isfile(os.path.join(gdir, "x_list_train.pt"))
                or os.path.isfile(os.path.join(gdir, "x_list_train.npy"))
                or os.path.isdir(os.path.join(gdir, "x_list_train")))
    if not has_data:
        missing_data.append(f"{table_label} ({config_name})")

if missing_data:
    msg = ", ".join(missing_data)
    raise RuntimeError(
        f"Training data not found for: {msg}. "
        f"Please run data generation first."
    )

# Check that trained models exist
missing_models = []
for config_name, table_label, label in datasets:
    log_dir = log_path(configs[config_name].config_file)
    model_files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
    if not model_files:
        missing_models.append(f"{table_label} ({config_name})")

if missing_models:
    msg = ", ".join(missing_models)
    raise RuntimeError(
        f"No trained models found for: {msg}. "
        f"Please train the GNN models first."
    )

# %%
def parse_plot_results(log_dir):
    """Extract key metrics from data_plot metrics.txt."""
    metrics = {}
    path = os.path.join(log_dir, "results", "metrics.txt")
    if not os.path.isfile(path):
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                metrics[key.strip()] = val.strip()
    return metrics

header = "| Metric | " + " | ".join(tl for _, tl, _ in datasets) + " |"
sep = "|:--|" + "|".join(":--:" for _ in datasets) + "|"
rows = [header, sep]

metric_keys = [
    ('W_corrected_R2', '$W$ corrected $R^2$'),
    ('W_corrected_slope', '$W$ corrected slope'),
    ('tau_R2', r'$\tau$ $R^2$'),
    ('V_rest_R2', r'$V^{\text{rest}}$ $R^2$'),
    ('clustering_accuracy', 'Clustering accuracy'),
]

for key, display_name in metric_keys:
    cells = []
    for config_name, _, _ in datasets:
        log_dir = log_path(configs[config_name].config_file)
        m = parse_plot_results(log_dir)
        cells.append(m.get(key, '\u2014'))
    rows.append(f"| {display_name} | " + " | ".join(cells) + " |")

display(Markdown("\n".join(rows)))

# %% [markdown]
# ## Activity Traces
#
# Each figure shows 20 sampled neuron traces over a 500-frame
# window.  The **left panel** displays the clean voltage from the
# ODE simulation (intrinsic dynamics noise
# $\sigma_{\text{dyn}} = 0.05$ only).
#
# The **right panel** shows the noisy observations
# $\tilde{v}_i(t) = v_i(t) + \sigma_{\text{meas}}\,\eta_i(t)$
# that the GNN actually receives during training.  As
# $\sigma_{\text{meas}}$ increases, the high-frequency measurement
# noise becomes clearly visible.

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"### {label}"))
    gdir = graphs_dirs[config_name]
    display_image(os.path.join(gdir, "activity_traces_noisy.png"), width=850)

# %% [markdown]
# ## Generate Analysis Plots
#
# For each measurement noise condition, `data_plot` loads the best
# model checkpoint and generates the full suite of results
# visualizations.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("ANALYSIS - Generating results plots for all measurement noise conditions")
print("=" * 80)

for config_name, table_label, label in datasets:
    config = configs[config_name]
    print(f"\n--- {label} ---")
    data_plot(
        config=config,
        config_file=config.config_file,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
    )

# %% [markdown]
# ## Connectivity Recovery
#
# The scatter plots below compare the learned (corrected) synaptic
# weights $W_{ij}^{\text{corr}}$ against the ground-truth connectome
# weights for all 434,112 edges.  As measurement noise increases,
# the derivative targets become noisier and the weight recovery
# degrades.

# %%
#| output: false
def show_result(filename, config_name, width=600):
    log_dir = log_path(configs[config_name].config_file)
    config_indices = config_name.replace('flyvis_', '')
    path = os.path.join(log_dir, "results", filename.format(idx=config_indices))
    if os.path.isfile(path):
        display_image(path, width=width)

def show_mlp(mlp_name, config_name, suffix=""):
    log_dir = log_path(configs[config_name].config_file)
    config_indices = config_name.replace('flyvis_', '')
    path = os.path.join(log_dir, "results", f"{mlp_name}_{config_indices}{suffix}.png")
    if os.path.isfile(path):
        display_image(path, width=700)

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"### {label}"))
    show_result("weights_comparison_corrected.png", config_name)

# %% [markdown]
# ## Neural Embeddings
#
# Each neuron is assigned a learned embedding $\mathbf{a}_i$ that
# captures its functional identity.  Tight clustering by cell type
# indicates that the GNN discovers neuron-type identity despite
# measurement noise.

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"### {label}"))
    show_result("embedding_{idx}.png", config_name)

# %% [markdown]
# ## UMAP Projections

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"### {label}"))
    show_result("embedding_augmented_{idx}.png", config_name)

# %% [markdown]
# ## Learned Functions
#
# The domain-restricted plots show the true ground-truth function
# (left panel) alongside the learned function (right panel) for
# each measurement noise condition.  As noise increases, the
# learned functions deviate more from the ground truth, but the
# overall shape is often preserved.

# %% [markdown]
# ### $f_\theta$ (MLP$_0$): Neuron Update Function

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_mlp("MLP0", config_name, "_domain")

# %% [markdown]
# ### Time Constants ($\tau$)

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_result("tau_comparison_{idx}.png", config_name, width=500)

# %% [markdown]
# ### Resting Potentials ($V^{\text{rest}}$)

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_result("V_rest_comparison_{idx}.png", config_name, width=500)

# %% [markdown]
# ### $g_\phi$ (MLP$_1$): Edge Message Function

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_mlp("MLP1", config_name, "_domain")

# %% [markdown]
# ### Neural Embeddings

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_result("embedding_{idx}.png", config_name, width=500)

# %% [markdown]
# ## Spectral Analysis

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"### {label}"))
    show_result("eigen_comparison.png", config_name, width=900)

# %% [markdown]
# ## References
#
# [1] J. K. Lappalainen et al., "Connectome-constrained networks predict
# neural activity across the fly visual system," *Nature*, 2024.
# [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)
#
# [2] C. Allier, L. Heinrich, M. Schneider, S. Saalfeld, "Graph
# neural networks uncover structure and functions underlying the
# activity of simulated neural assemblies," *arXiv:2602.13325*,
# 2026.
# [doi:10.48550/arXiv.2602.13325](https://doi.org/10.48550/arXiv.2602.13325)

# %%
