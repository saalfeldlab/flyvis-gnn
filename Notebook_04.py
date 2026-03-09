# %% [raw]
# ---
# title: "GNN Results: Connectivity Recovery and Learned Representations"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Analysis
# execute:
#   echo: false
# image: "log/fly/flyvis_noise_05/results/MLP0_noise_05_domain.png"
# description: "Extract and compare the learned synaptic weights, time constants, resting potentials, neuron-type embeddings, and MLP functions against the ground-truth simulator parameters across noise conditions."
# ---

# %% [markdown]
# ## Analysis of Learned Representations
#

#
# After training ([Notebook 01](Notebook_01.html)), we analyzed what the GNN had learned
# about the circuit.  For each noise condition we extracted the learned synaptic
# weights $\widehat{W}_{ij}$, neural embeddings $\mathbf{a}_i$, and
# the two MLP functions ($f_\theta$ and $g_\phi$), and compared them
# to the ground-truth parameters of the simulator.
#
# The analysis addressed several questions: (1) how accurately did
# the GNN recover the 434,112 synaptic weights from voltage data
# alone? (2) did the learned embeddings capture cell-type identity?
# (3) were the learned functions biologically interpretable?  We
# generated all results plots via `data_plot` and then display the
# key results side by side across noise conditions for comparison.
# Recall that the simulated dynamics include an intrinsic noise term
# $\sigma\,\xi_i(t)$ where $\xi_i(t) \sim \mathcal{N}(0,1)$ ([Notebook 00](Notebook_00.html)).
# We compared results at three noise levels: $\sigma = 0$
# (noise-free), $\sigma = 0.05$ (low noise), and $\sigma = 0.5$
# (high noise).

# %%
#| output: false
import glob
import os
import sys
import warnings

from IPython.display import Image, Markdown, display

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.utils import set_device, add_pre_folder, graphs_data_path, log_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
from GNN_PlotFigure import data_plot

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    display(Image(filename=path, width=width))

# %% [markdown]
# ## Configuration

# %%
#| output: false
datasets = [
    ('flyvis_noise_free', 'Noise-free'),
    ('flyvis_noise_005', 'Noise 0.05'),
    ('flyvis_noise_05', 'Noise 0.5'),
]

config_root = "./config"
configs = {}
graphs_dirs = {}

for config_name, label in datasets:
    config_file, pre_folder = add_pre_folder(config_name)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name
    configs[config_name] = config
    graphs_dirs[config_name] = graphs_data_path(pre_folder + config_name)

device = set_device(configs[datasets[0][0]].training.device)

# Check that data exists
missing_data = []
for config_name, label in datasets:
    gdir = graphs_dirs[config_name]
    has_data = (os.path.isfile(os.path.join(gdir, "x_list_train.pt"))
                or os.path.isfile(os.path.join(gdir, "x_list_train.npy"))
                or os.path.isdir(os.path.join(gdir, "x_list_train")))
    if not has_data:
        missing_data.append(label)

if missing_data:
    msg = ", ".join(missing_data)
    raise RuntimeError(
        f"Training data not found for: {msg}. "
        f"Please run Notebook_00 first to generate the data."
    )

# Check that trained models exist
missing_models = []
for config_name, label in datasets:
    log_dir = log_path(configs[config_name].config_file)
    model_files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
    if not model_files:
        missing_models.append(label)

if missing_models:
    msg = ", ".join(missing_models)
    raise RuntimeError(
        f"No trained models found for: {msg}. "
        f"Please run Notebook_01 first to train the GNN models."
    )

# %% [markdown]
# ## Generate Analysis Plots
#
# For each noise condition, `data_plot` loads the best model
# checkpoint and generates the full suite of results
# visualizations: weight scatter plots (raw and corrected), neural
# embeddings, MLP function curves, spectral analysis, and UMAP
# projections.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("ANALYSIS - Generating results plots for all noise conditions")
print("=" * 80)

for config_name, label in datasets:
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
# The scatter plots below compare the learned synaptic weights
# $\widehat{W}_{ij}$ against the ground-truth connectome weights
# $\mathbf{W}_{ij}$ for all 434,112 edges.  Because the GNN can
# absorb arbitrary gain factors into $f_\theta$ and $g_\phi$, the
# raw model parameter $\widehat{\mathbf{W}}$ differs from the true
# weights by a per-neuron scaling.  The plots show **corrected**
# weights that factor out these gains to reveal the true synaptic
# structure (see below).

# %%
#| output: false
def show_result(filename, config_name, width=600):
    log_dir = log_path(configs[config_name].config_file)
    config_indices = config_name.replace('flyvis_', '')
    path = os.path.join(log_dir, "results", filename.format(idx=config_indices))
    if os.path.isfile(path):
        display_image(path, width=width)

# %% [markdown]
# **Weight–Gain Entanglement.**
# In the GNN forward pass, the message arriving at neuron $i$ is
# $\text{msg}_i = \sum_{j} \widehat{W}_{ij}\,g_\phi(v_j,\mathbf{a}_j)^2$.
# The model therefore learns the *product*
# $\widehat{W}_{ij} \cdot g_\phi^2$, not $\widehat{W}_{ij}$ alone:
# an arbitrary gain absorbed into $g_\phi$ can be compensated by
# rescaling $\widehat{W}_{ij}$, and likewise for the postsynaptic
# gain of $f_\theta$.  To disentangle the true synaptic weight from
# these gain factors, we fit a linear model to each function in its
# natural activity range (mean $\pm$ 2 std), extracting per-neuron
# slopes $s_g(j)$ from $g_\phi$ and $s_f(i)$ from $f_\theta$,
# together with $\partial f_\theta / \partial\text{msg}$ evaluated at
# typical operating points.  The corrected weight is then
#
# $$W_{ij}^{\text{corr}} = -\,\frac{\widehat{W}_{ij}}{s_f(i)}
# \;\frac{\partial f_\theta}{\partial\text{msg}}\bigg|_i
# \; s_g(j)$$
#
# which factors out the gain ambiguity and recovers the true
# synaptic structure, as shown below.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_free")

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_005")

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_result("weights_comparison_corrected.png", "flyvis_noise_05")

# %% [markdown]
# ## Neural Embeddings
#
# Each neuron $i$ is assigned a learned embedding vector
# $\mathbf{a}_i \in \mathbb{R}^{d_\text{emb}}$ that captures its
# functional identity.  The 2D scatter below shows these embeddings
# colored by ground-truth cell type.  Tight clustering by type
# indicates that the GNN has discovered cell-type identity from
# voltage dynamics alone, without any explicit labels.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_result("embedding_{idx}.png", "flyvis_noise_free")

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_result("embedding_{idx}.png", "flyvis_noise_005")

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_result("embedding_{idx}.png", "flyvis_noise_05")

# %% [markdown]
# ## UMAP Projections
#
# To further assess how well the learned representations capture
# cell-type structure, we apply UMAP [3] to an augmented feature vector
# that combines the learned embedding $\mathbf{a}_i$ with the
# extracted biophysical parameters ($\tau_i$, $V_i^{\text{rest}}$)
# and connectivity statistics (mean and standard deviation of
# incoming and outgoing weights).  Points are colored by Gaussian
# mixture model (GMM) cluster labels ($n_{\text{components}} = 100$),
# and the clustering accuracy relative to the ground-truth cell
# types is reported.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_result("embedding_augmented_{idx}.png", "flyvis_noise_free")

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_result("embedding_augmented_{idx}.png", "flyvis_noise_005")

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_result("embedding_augmented_{idx}.png", "flyvis_noise_05")

# %% [markdown]
# ## Learned Functions
#
# The GNN uses two MLP functions.  The edge message function
# $g_\phi$ (MLP$_1$) maps presynaptic voltage and embedding to a
# nonnegative message (via squaring).  The monotonicity regularizer
# ($\mu_0$) enforces that $g_\phi$ increases with voltage, ensuring
# that stronger presynaptic activity produces larger messages.  The
# neuron update function $f_\theta$ (MLP$_0$) combines postsynaptic
# voltage, aggregated input, and external stimulus to predict
# $\widehat{dv}/dt$.
#
# Each curve corresponds to one of the 65 cell types, colored
# consistently across plots.  The voltage axis is restricted to
# each neuron type's natural activity range (mean $\pm$ 2 std),
# where the functions are actually evaluated during inference.

# %%
#| output: false
def show_mlp(mlp_name, config_name, suffix=""):
    log_dir = log_path(configs[config_name].config_file)
    config_indices = config_name.replace('flyvis_', '')
    path = os.path.join(log_dir, "results", f"{mlp_name}_{config_indices}{suffix}.png")
    if os.path.isfile(path):
        display_image(path, width=700)

# %% [markdown]
# ## $f_\theta$ (MLP$_0$): Neuron Update Function
#
# Each curve shows $f_\theta$ restricted to the neuron type's natural
# activity range (mean $\pm$ 2 std).  A linear fit in this domain
# yields the effective time constant $\tau_i$ (slope) and resting
# potential $V_i^{\text{rest}}$ (zero-crossing), compared to the
# ground-truth simulator parameters below.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_mlp("MLP0", "flyvis_noise_free", "_domain")

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_mlp("MLP0", "flyvis_noise_005", "_domain")

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_mlp("MLP0", "flyvis_noise_05", "_domain")

# %% [markdown]
# ## Biophysical Parameters from $f_\theta$
#
# The linear fit to $f_\theta$ in the natural activity domain directly
# yields two biophysical parameters for each neuron: the time constant
# $\tau_i$ (from the slope) and the resting potential
# $V_i^{\text{rest}}$ (from the zero-crossing).  The scatter plots
# below compare these extracted values to the ground-truth simulator
# parameters.

# %% [markdown]
# ### Time Constants ($\tau$)

# %% [markdown]
# #### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_result("tau_comparison_{idx}.png", "flyvis_noise_free", width=500)

# %% [markdown]
# #### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_result("tau_comparison_{idx}.png", "flyvis_noise_005", width=500)

# %% [markdown]
# #### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_result("tau_comparison_{idx}.png", "flyvis_noise_05", width=500)

# %% [markdown]
# ### Resting Potentials ($V^{\text{rest}}$)

# %% [markdown]
# #### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_result("V_rest_comparison_{idx}.png", "flyvis_noise_free", width=500)

# %% [markdown]
# #### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_result("V_rest_comparison_{idx}.png", "flyvis_noise_005", width=500)

# %% [markdown]
# #### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_result("V_rest_comparison_{idx}.png", "flyvis_noise_05", width=500)

# %% [markdown]
# ## $g_\phi$ (MLP$_1$): Edge Message Function
#
# Each curve shows $g_\phi$ restricted to the neuron type's natural
# activity range (mean $\pm$ 2 std).  The slopes extracted from the
# linear fit in this domain are used for the weight correction
# described above.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_mlp("MLP1", "flyvis_noise_free", "_domain")

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_mlp("MLP1", "flyvis_noise_005", "_domain")

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_mlp("MLP1", "flyvis_noise_05", "_domain")

# %% [markdown]
# ## Spectral Analysis
#
# Beyond comparing individual synaptic weights, we ask whether the
# GNN has recovered the global dynamical structure of the circuit.
# The weight matrix $\mathbf{W} \in \mathbb{R}^{N \times N}$ (with
# $N{=}13{,}741$ neurons) governs the linear stability and
# oscillatory modes of the network.  Its eigenvalues
# $\lambda_k = \text{Re}(\lambda_k) + i\,\text{Im}(\lambda_k)$
# determine the time scales and frequencies of intrinsic network
# modes, while its singular values $\sigma_k$ capture the gain
# along principal directions of signal flow.
#
# The $2 \times 3$ figure below compares the spectral properties of
# the ground-truth and learned (corrected) weight matrices:
#
# **Top row.** *Eigenvalues and singular values.*
# (Left) The 200 largest-magnitude eigenvalues plotted in the
# complex plane; overlap between green (true) and black (learned)
# clouds indicates that the GNN preserves the circuit's oscillatory
# and decay modes.
# (Center) A scatter of matched singular values; black points near
# the diagonal mean the learned matrix preserves the gain spectrum.
# (Right) The singular value spectrum on a log scale; parallel
# decay curves confirm that the rank structure and effective
# dimensionality of the connectivity are faithfully reproduced.
#
# **Bottom row.** *Singular vector alignment.*
# (Left and center) Alignment matrices between the top 100 right
# and left singular vectors of the true and learned matrices.
# A strong diagonal indicates one-to-one correspondence between
# the principal connectivity modes.  Off-diagonal mass would
# signal that the learned matrix mixes true modes.
# (Right) The best alignment score per singular vector.  Values
# near 1.0 mean the corresponding mode is recovered almost
# exactly; the gray dashed line marks the expected alignment for
# random vectors.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_result("eigen_comparison.png", "flyvis_noise_free", width=900)

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_result("eigen_comparison.png", "flyvis_noise_005", width=900)

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_result("eigen_comparison.png", "flyvis_noise_05", width=900)

# %% [markdown]
# ## Summary
#
# The table below summarizes the key quantitative metrics across
# the three noise conditions.  Weight correlation ($R^2$) is
# reported for the corrected weights.  These metrics are extracted
# from the `results.log` files generated by `data_plot`.

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

header = "| Metric | " + " | ".join(label for _, label in datasets) + " |"
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
    for config_name, _ in datasets:
        log_dir = log_path(configs[config_name].config_file)
        m = parse_plot_results(log_dir)
        cells.append(m.get(key, '\u2014'))
    rows.append(f"| {display_name} | " + " | ".join(cells) + " |")

display(Markdown("\n".join(rows)))

# %% [markdown]
# ## References
#
# [1] J. K. Lappalainen et al., "Connectome-constrained networks predict
# neural activity across the fly visual system," *Nature*, 2024.
# [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)
#
# [2] J. Gilmer et al., "Neural Message Passing for Quantum Chemistry,"
# 2017.
# [doi:10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)
#
# [3] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold
# Approximation and Projection for Dimension Reduction," 2018.
# [doi:10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426)
