# %% [raw]
# ---
# title: "GNN + INR: Joint Stimulus and Dynamics Recovery"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - INR
#   - SIREN
# execute:
#   echo: false
# image: "log/fly/flyvis_noise_005_INR/results/weights_comparison_corrected.png"
# description: "Train a SIREN implicit neural representation (INR) jointly with the GNN to recover the visual stimulus field from neural activity alone. Discuss the inherent scale/offset degeneracy and the corrected R²."
# ---

# %% [markdown]
# ## Joint Stimulus and Dynamics Recovery with GNN + INR
#
# In the previous notebooks the visual stimulus $I_i(t)$ was provided
# as a known input to the GNN.  Here we ask: can the stimulus itself
# be recovered from neural activity alone?
#
# We replaced the ground-truth stimulus with a learnable **implicit
# neural representation** (INR), specifically a
# [SIREN](https://arxiv.org/abs/2006.09661) network, that maps
# continuous coordinates $(t, x, y)$ to the stimulus value at each
# neuron position and time step.  The SIREN was trained jointly with
# the GNN.  This amounted to solving a harder inverse problem: recovering not
# only the circuit parameters ($W$, $\tau$, $V^{\text{rest}}$,
# $f_\theta$, $g_\phi$) but also the stimulus field from voltage
# data alone.

# %% [markdown]
# ## SIREN Architecture
#
# The SIREN (Sinusoidal Representation Network) uses periodic
# activation functions $\phi(x) = \sin(\omega_0 \cdot x)$ instead
# of ReLU, enabling it to represent fine spatial and temporal
# detail in the stimulus field.
#
# The key hyperparameters explored by the agentic hyper-parameter optimization
# ([Notebook 09](Notebook_09.html)) are:
#
# - **$\omega_0$** (frequency scaling): controls the spectral
#   bandwidth of the representation.  Higher $\omega_0$ allows the
#   network to capture faster temporal fluctuations and sharper
#   spatial edges.
# - **hidden_dim**: network width (number of hidden units per
#   layer).
# - **n_layers**: network depth.
# - **learning rate**: must scale inversely with $\omega_0$ for
#   stable training.
#
# The input is a 3D coordinate $(t, x, y)$ normalized to the
# training domain, and the output is a scalar stimulus value for
# each neuron at each time step.
#
# ### Scale/Offset Degeneracy and Corrected $R^2$
#
# The SIREN output enters the GNN through $f_\theta$, which receives
# the concatenated input $[v_i,\, \mathbf{a}_i,\, \text{msg}_i,\, I_i(t)]$.
# This creates an inherent **scale/offset degeneracy**: $f_\theta$'s
# biases absorb any constant offset, and its weights on the excitation
# dimension compensate any scale factor (including sign inversion).
# The SIREN and $f_\theta$ jointly optimize along a degenerate manifold
# where the stimulus *pattern* is learned correctly but the linear
# mapping between SIREN output and true stimulus is unidentifiable.
# We therefore apply a **global linear fit**
# $I^{\text{true}} = a \cdot I^{\text{pred}} + b$ and report the
# corrected $R^2$.

# %%
#| output: false
import glob
import os
import warnings

from IPython.display import Image, Markdown, Video, display

sys_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
import sys
sys.path.insert(0, sys_path)

from GNN_PlotFigure import data_plot
from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.models.graph_trainer import data_train, data_test
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
# ## Noise Level
#
# Recall that the simulated dynamics include an intrinsic noise term
# $\sigma\,\xi_i(t)$ where $\xi_i(t) \sim \mathcal{N}(0,1)$
# ([Notebook 00](Notebook_00.html)).  The joint GNN+INR experiment
# presented here uses $\sigma = 0.05$ (low noise).  To change the
# noise level, edit the `noise_model_level` field in the config file
# `config/fly/flyvis_noise_005_INR.yaml`.

# %% [markdown]
# ## Results
#
# The joint GNN+SIREN model uses the `flyvis_noise_005_INR` config,
# which extends the noise 0.05 setup with SIREN parameters
# (hidden_dim=2048, 4 layers, $\omega_0$=4096).  Training alternates
# between full GNN learning rates in epoch 0 and reduced rates
# (x0.05) in subsequent epochs, while the SIREN learning rate
# remains constant.
#
# Despite the added complexity of jointly learning the stimulus
# field, the GNN still recovers synaptic weights, time constants,
# resting potentials, and neuron-type embeddings with quality
# comparable to the known-stimulus baseline.  This confirms that
# the inverse problem remains well-posed even when the input
# drive is unknown.

# %%
#| output: false
config_name = "flyvis_noise_005_INR"
config_file, pre_folder = add_pre_folder(config_name)
config = NeuralGraphConfig.from_yaml(f"./config/{config_file}.yaml")
config.dataset = pre_folder + config.dataset
config.config_file = pre_folder + config_name
gnn_log_dir = log_path(config.config_file)
device = set_device(config.training.device)

graphs_dir = graphs_data_path(config.dataset)

# %% [markdown]
# ## Data Generation
#
# The INR config uses the same `flyvis_noise_005` simulation data.
# If it has not been generated yet (via [Notebook 00](Notebook_00.html)),
# we generate it here.

# %%
#| echo: true
#| output: false
data_exists = os.path.isdir(os.path.join(graphs_dir, 'x_list_train')) or \
              os.path.isdir(os.path.join(graphs_dir, 'x_list_0'))

if data_exists:
    print(f"Data already exists at {graphs_dir}/")
    print("Skipping simulation.")
else:
    print(f"Generating simulation data for {config_name}...")
    data_generate(
        config,
        device=device,
        visualize=False,
        run_vizualized=0,
        style="color",
        alpha=1,
        erase=True,
        save=True,
        step=100,
    )

# %% [markdown]
# ## Training
#
# The joint GNN+SIREN model is trained end-to-end.  The GNN learns
# synaptic weights, embeddings, and MLPs while the SIREN learns to
# reconstruct the stimulus field from $(t,x,y)$ coordinates.
# Training uses 3 epochs with alternate training: full GNN learning
# rates in epoch 0, then 0.05x in epochs 1+.

# %%
#| echo: true
#| output: false
model_dir = os.path.join(gnn_log_dir, "models")
model_exists = os.path.isdir(model_dir) and any(
    f.startswith("best_model") for f in os.listdir(model_dir)
) if os.path.isdir(model_dir) else False

if model_exists:
    print(f"Trained model already present in {model_dir}/")
    print("Skipping training. To retrain, delete the log folder:")
    print(f"  rm -rf {gnn_log_dir}")
else:
    print(f"Training joint GNN+SIREN model ({config_name})...")
    data_train(config=config, erase=True, device=device)

# %% [markdown]
# ## Loss Decomposition

# %%
#| lightbox: true
loss_png = plot_loss_from_file(gnn_log_dir)
if loss_png:
    display_image(loss_png, width=900)
else:
    print("loss_components.pt not found. Run training above first.")

# %% [markdown]
# ## Testing: Rollout and Stimulus Recovery
#
# We run `data_test` on the joint model.  For INR models, the rollout
# uses training data (since the SIREN was fit to it) and additionally
# computes the corrected stimulus $R^2$ and generates a GT vs Pred
# video showing the recovered stimulus on the hexagonal photoreceptor
# array.

# %%
#| echo: true
#| output: false
print("\n--- Testing joint GNN+SIREN model ---")
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
# ## Stimulus Recovery Video
#
# The video below shows the SIREN result with three panels:
#
# - **Left**: ground-truth stimulus on the hexagonal photoreceptor
#   array.
# - **Center**: SIREN prediction after global linear correction
#   ($I^{\text{true}} = a \cdot I^{\text{pred}} + b$).
# - **Right**: rolling voltage traces for selected neurons
#   (ground truth in green, prediction in black).

# %%
results_dir = os.path.join(gnn_log_dir, "results")
_video_files = sorted(glob.glob(os.path.join(results_dir, "*stimulus*gt_vs_pred*.mp4")))
if _video_files:
    display(Video(_video_files[-1], embed=True, width=800))
else:
    display(Markdown("*No stimulus video found. Check that data_test completed successfully.*"))

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
            for key in ['RMSE', 'Pearson r', 'stimuli_R2']:
                if line.startswith(f'{key}:'):
                    metrics[key] = line.split(':', 1)[1].strip()
    return metrics


rollout_log = os.path.join(gnn_log_dir, "results_rollout.log")
m = parse_results_log(rollout_log)
if m:
    rows = ["| Metric | Value |", "|:--|:--|"]
    for key in ['RMSE', 'Pearson r', 'stimuli_R2']:
        if key in m:
            label = 'Stimulus $R^2$ (corrected)' if key == 'stimuli_R2' else key
            rows.append(f"| {label} | {m[key]} |")
    display(Markdown("\n".join(rows)))
else:
    print(f"Rollout log not found at {rollout_log}")

# %% [markdown]
# ## Rollout Traces

# %%
#| lightbox: true
if os.path.isdir(results_dir):
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
# Beyond stimulus recovery, the joint GNN+SIREN model also learns
# synaptic weights, neural embeddings, and MLP functions.  Below we
# run the same analysis as [Notebook 04](Notebook_04.html) on the
# joint model to verify that circuit recovery is preserved.

# %%
#| echo: true
#| output: false
print("\n--- Generating GNN analysis plots for noise_005_INR ---")
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
config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else config_name.replace('flyvis_', '')


def show_result(filename, width=600):
    path = os.path.join(gnn_log_dir, "results", filename.format(idx=config_indices))
    display_image(path, width=width)


def show_mlp(mlp_name, suffix=""):
    path = os.path.join(gnn_log_dir, "results", f"{mlp_name}_{config_indices}{suffix}.png")
    display_image(path, width=700)


# %% [markdown]
# ### Corrected Weights ($W$)

# %%
#| lightbox: true
show_result("weights_comparison_corrected.png")

# %% [markdown]
# ### $f_\theta$ (MLP$_0$): Neuron Update Function

# %%
#| lightbox: true
show_mlp("MLP0", "_domain")

# %% [markdown]
# ### Time Constants ($\tau$)

# %%
#| lightbox: true
show_result("tau_comparison_{idx}.png", width=500)

# %% [markdown]
# ### Resting Potentials ($V^{\text{rest}}$)

# %%
#| lightbox: true
show_result("V_rest_comparison_{idx}.png", width=500)

# %% [markdown]
# ### $g_\phi$ (MLP$_1$): Edge Message Function

# %%
#| lightbox: true
show_mlp("MLP1", "_domain")

# %% [markdown]
# ### Neural Embeddings

# %%
#| lightbox: true
show_result("embedding_{idx}.png")

# %% [markdown]
# ### UMAP Projections

# %%
#| lightbox: true
show_result("embedding_augmented_{idx}.png")

# %% [markdown]
# ### Spectral Analysis

# %%
#| lightbox: true
show_result("eigen_comparison.png", width=900)

# %% [markdown]
# ## References
#
# [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell,
# and G. Wetzstein, "Implicit Neural Representations with Periodic
# Activation Functions," *NeurIPS*, 2020.
# [doi:10.48550/arXiv.2006.09661](https://doi.org/10.48550/arXiv.2006.09661)
#
# [2] C. Allier, L. Heinrich, M. Schneider, S. Saalfeld, "Graph
# neural networks uncover structure and functions underlying the
# activity of simulated neural assemblies," *arXiv:2602.13325*,
# 2026.
# [doi:10.48550/arXiv.2602.13325](https://doi.org/10.48550/arXiv.2602.13325)
