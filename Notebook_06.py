# %% [raw]
# ---
# title: "Robustness to Extra Null Edges"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Null Edges
# execute:
#   echo: false
# image: "log/fly/flyvis_noise_005_null_edges_pc_200/results/weights_comparison_corrected.png"
# description: "Train the GNN with 100%, 200%, and 400% extra random null edges appended to the true connectome. Evaluate whether the model can still recover synaptic weights, biophysical parameters, and neuron-type identity despite the corrupted adjacency matrix."
# ---

# %% [markdown]
# ## Robustness to Extra Null Edges
#
# In a real experimental setting, the connectome may contain false
# positives: spurious synaptic connections that do not carry
# functional weight.  To test robustness to such noise in the
# adjacency matrix, we augmented the true connectome (434,112 edges)
# with random **null edges** — connections between randomly chosen
# neuron pairs that carry zero true weight.  The GNN must learn to
# assign near-zero weights to these null edges while still recovering
# the true synaptic structure.
#
# We tested three levels of null-edge contamination:
#
# | Config | Extra null edges | Total edges | Ratio |
# |:--|:--|:--|:--|
# | `flyvis_noise_005_null_edges_pc_100` | 434,112 | 868,224 | 1:1 (100%) |
# | `flyvis_noise_005_null_edges_pc_200` | 868,224 | 1,302,336 | 2:1 (200%) |
# | `flyvis_noise_005_null_edges_pc_400` | 1,736,448 | 2,170,560 | 4:1 (400%) |

# %% [markdown]
# ## Noise Level
#
# Recall that the simulated dynamics include an intrinsic noise term
# $\sigma\,\xi_i(t)$ where $\xi_i(t) \sim \mathcal{N}(0,1)$
# ([Notebook 00](Notebook_00.html)).  All null-edge experiments
# presented here use a fixed noise level of $\sigma = 0.05$
# (low noise).  To change the noise level, edit the
# `noise_model_level` field in the respective config files:
#
# - `config/fly/flyvis_noise_005_null_edges_pc_100.yaml`
# - `config/fly/flyvis_noise_005_null_edges_pc_200.yaml`
# - `config/fly/flyvis_noise_005_null_edges_pc_400.yaml`

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
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.models.graph_trainer import data_test, data_train
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
# Each config extends the base `flyvis_noise_005` setup with a
# different number of extra null edges injected into the adjacency
# matrix.  The null edges are sampled uniformly at random among
# neuron pairs not already connected.  The GNN architecture and
# training hyperparameters are otherwise identical across conditions.
#
# The GNN is robust to null-edge contamination: even
# with 4x as many spurious edges as real ones, it recovers
# synaptic weights, biophysical parameters, and neuron-type
# identity with only modest degradation.  The model effectively
# learns to assign near-zero weights to the null edges while
# preserving the true synaptic structure.

# %%
#| output: false
datasets = [
    ('flyvis_noise_005_null_edges_pc_100', '100%', '100% extra null edges (434,112 null / 868,224 total)'),
    ('flyvis_noise_005_null_edges_pc_200', '200%', '200% extra null edges (868,224 null / 1,302,336 total)'),
    ('flyvis_noise_005_null_edges_pc_400', '400%', '400% extra null edges (1,736,448 null / 2,170,560 total)'),
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
    graphs_dirs[config_name] = graphs_data_path(config.dataset)

device = set_device(configs[datasets[0][0]].training.device)

# %%
#| output: false
print()
print("=" * 80)
print("GENERATE - Simulating fly visual system (null-edge variants)")
print("=" * 80)

for config_name, table_label, label in datasets:
    config = configs[config_name]
    graphs_dir = graphs_dirs[config_name]
    print()
    print(f"--- {label} ---")
    data_exists = os.path.isdir(os.path.join(graphs_dir, 'x_list_train'))
    if data_exists:
        print(f"  data already exists at {graphs_dir}/")
        print("  skipping simulation...")
    else:
        print(f"  generating data at {graphs_dir}/")
        data_generate(
            config,
            device=device,
            visualize=False,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=False,
            save=True,
            step=100,
        )

print()
print("=" * 80)
print("TRAIN - GNN on fly visual system (null-edge variants)")
print("=" * 80)

for config_name, table_label, label in datasets:
    config = configs[config_name]
    log_dir = log_path(config.config_file)
    model_dir = os.path.join(log_dir, "models")
    model_exists = os.path.isdir(model_dir) and any(
        f.startswith("best_model") for f in os.listdir(model_dir)
    ) if os.path.isdir(model_dir) else False
    print()
    print(f"--- {label} ---")
    if model_exists:
        print(f"  trained model already present in {model_dir}/")
        print("  skipping training. To retrain, delete the log folder:")
        print(f"    rm -rf {log_dir}")
    else:
        print(f"  training on {config.simulation.n_frames} frames")
        print(f"  {config.training.n_epochs} epochs, batch_size={config.training.batch_size}")
        print()
        data_train(config, device=device)

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


header = "| Metric | " + " | ".join(table_label for _, table_label, _ in datasets) + " |"
sep = "|:--|" + "|".join(":--:" for _ in datasets) + "|"
rows = [header, sep]

metric_keys = [
    ('W_corrected_R2', '$W$ corrected $R^2$'),
    ('W_corrected_slope', '$W$ corrected slope'),
    ('tau_R2', '$\\tau$ $R^2$'),
    ('V_rest_R2', '$V^{\\text{rest}}$ $R^2$'),
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
# ## Loss Curves

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    gnn_log_dir = log_path(configs[config_name].config_file)
    display(Markdown(f"### {label}"))
    loss_png = plot_loss_from_file(gnn_log_dir)
    if loss_png:
        display_image(loss_png, width=900)
    else:
        print(f"[{label}] loss_components.pt not found.")

# %% [markdown]
# ## Testing
#
# We evaluate each trained model on held-out stimuli and compute
# rollout predictions.

# %%
#| echo: true
#| output: false
for config_name, table_label, label in datasets:
    config = configs[config_name]
    gnn_log_dir = log_path(config.config_file)
    print(f"\n--- Testing {label} ---")
    data_test(
        config=config,
        visualize=True,
        style="color name continuous_slice",
        verbose=False,
        best_model='best',
        run=0,
        step=10,
        n_rollout_frames=250,
        device=device,
    )

# %% [markdown]
# ## Rollout Traces

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    gnn_log_dir = log_path(configs[config_name].config_file)
    results_dir = os.path.join(gnn_log_dir, "results")
    if not os.path.isdir(results_dir):
        continue
    display(Markdown(f"### {label}"))
    rollout_all = sorted([f for f in os.listdir(results_dir)
                          if f.startswith("rollout_") and "_all" in f
                          and "on_" not in f and f.endswith(".png")])
    rollout_sel = sorted([f for f in os.listdir(results_dir)
                          if f.startswith("rollout_") and "selected" in f
                          and "on_" not in f and f.endswith(".png")])
    if rollout_all:
        display_image(os.path.join(results_dir, rollout_all[0]), width=900)
    if rollout_sel:
        display_image(os.path.join(results_dir, rollout_sel[0]), width=900)

# %% [markdown]
# ## GNN Analysis: Learned Representations
#
# We run the same analysis as [Notebook 04](Notebook_04.html) on
# each null-edge model to assess whether circuit recovery is
# preserved despite the corrupted adjacency matrix.

# %%
#| echo: true
#| output: false
for config_name, table_label, label in datasets:
    config = configs[config_name]
    print(f"\n--- Generating analysis plots for {label} ---")
    data_plot(
        config=config,
        config_file=config.config_file,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
    )

# %%
#| output: false
def get_config_indices(config_name):
    config = configs[config_name]
    return config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else config_name.replace('flyvis_', '')


def show_result(filename, config_name, width=600):
    gnn_log_dir = log_path(configs[config_name].config_file)
    idx = get_config_indices(config_name)
    path = os.path.join(gnn_log_dir, "results", filename.format(idx=idx))
    display_image(path, width=width)


def show_mlp(mlp_name, config_name, suffix=""):
    gnn_log_dir = log_path(configs[config_name].config_file)
    idx = get_config_indices(config_name)
    path = os.path.join(gnn_log_dir, "results", f"{mlp_name}_{idx}{suffix}.png")
    display_image(path, width=700)


# %% [markdown]
# ### Corrected Weights ($W$)

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_result("weights_comparison_corrected.png", config_name)

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
    show_result("embedding_{idx}.png", config_name)

# %% [markdown]
# ### UMAP Projections

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_result("embedding_augmented_{idx}.png", config_name)

# %% [markdown]
# ### Spectral Analysis

# %%
#| lightbox: true
for config_name, table_label, label in datasets:
    display(Markdown(f"#### {label}"))
    show_result("eigen_comparison.png", config_name, width=900)

# %% [markdown]
# ## Rollout Metrics

# %%
def parse_results_log(path):
    """Parse a results log file into a dict of metric_name -> value string."""
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    with open(path) as f:
        for line in f:
            for key in ['RMSE', 'Pearson r']:
                if line.startswith(f'{key}:'):
                    metrics[key] = line.split(':', 1)[1].strip()
    return metrics


header = "| Metric | " + " | ".join(table_label for _, table_label, _ in datasets) + " |"
sep = "|:--|" + "|".join(":--:" for _ in datasets) + "|"
rows = [header, sep]

for key in ['RMSE', 'Pearson r']:
    cells = []
    for config_name, _, _ in datasets:
        log_dir = log_path(configs[config_name].config_file)
        m = parse_results_log(os.path.join(log_dir, "results_rollout.log"))
        cells.append(m.get(key, '\u2014'))
    rows.append(f"| {key} | " + " | ".join(cells) + " |")

display(Markdown("\n".join(rows)))

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
