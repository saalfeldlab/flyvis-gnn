# %% [raw]
# ---
# title: "GNN Testing: Evaluating Learned Dynamics"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Testing
# execute:
#   echo: false
# image: "log/fly/flyvis_noise_free/results/rollout_noise_free_DAVIS_selected.png"
# description: "Evaluate the trained GNN on held-out test stimuli never seen during training, comparing one-step predictions and multi-step rollouts against the ground-truth simulator trajectories."
# ---

# %% [markdown]
# ## Test
#

#
# ### Test Data Generation
#
# The test set is generated from a separate pool of DAVIS video
# stimuli.  During data generation ([Notebook 00](Notebook_00.html)), the 71 DAVIS video
# subdirectories are split 80/20: 56 videos are used
# for training and 15 for testing.  All augmentations (flips, rotations)
# of a given video stay in the same split, so the test visual stimuli
# are entirely unseen during training.  The simulator then generates
# new voltage traces from these test stimuli using the same connectivity
# and dynamics parameters (see the train/test first-frame previews in
# [Notebook 00](Notebook_00.html)).
# Recall that the simulated dynamics include an intrinsic noise term
# $\sigma\,\xi_i(t)$ where $\xi_i(t) \sim \mathcal{N}(0,1)$.  
# We evaluated the GNN at three noise levels: $\sigma = 0$
# (noise-free), $\sigma = 0.05$ (low noise), and $\sigma = 0.5$
# (high noise).

# %%
#| output: false
import glob
import os
import warnings

from IPython.display import Image, Markdown, display

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.models.graph_trainer import data_test
from flyvis_gnn.utils import set_device, add_pre_folder, graphs_data_path, log_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    display(Image(filename=path, width=width))


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


def metrics_table(datasets, configs, log_suffix):
    """Build a markdown table string from results log files."""
    header = "| Metric | " + " | ".join(label for _, label in datasets) + " |"
    sep = "|:--|" + "|".join(":--:" for _ in datasets) + "|"
    rows = [header, sep]
    for key in ['RMSE', 'Pearson r']:
        cells = []
        for config_name, _ in datasets:
            log_dir = log_path(configs[config_name].config_file)
            m = parse_results_log(os.path.join(log_dir, log_suffix))
            cells.append(m.get(key, '\u2014'))
        rows.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(rows)

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

# Check that test data exists for all configs
missing_data = []
for config_name, label in datasets:
    gdir = graphs_dirs[config_name]
    has_test = (os.path.isfile(os.path.join(gdir, "x_list_test.pt"))
                or os.path.isfile(os.path.join(gdir, "x_list_test.npy"))
                or os.path.isdir(os.path.join(gdir, "x_list_test")))
    if not has_test:
        missing_data.append(label)

if missing_data:
    msg = ", ".join(missing_data)
    raise RuntimeError(
        f"Test data not found for: {msg}. "
        f"Please run Notebook_00 first to generate the data."
    )

# Check that trained models exist for all configs
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
        f"Run Notebook_01 first to train the GNN models."
    )

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("TEST - Evaluating on new test data (unseen stimuli)")
print("=" * 80)

for config_name, label in datasets:
    config = configs[config_name]
    print()
    print(f"--- {label} ---")
    data_test(config, best_model='best', device=device)

# %% [markdown]
# ## Rollout Results
#
# Starting from the initial voltages at $t{=}0$, the model receives
# only the external stimulus from the test set (unseen video
# sequences) and autoregressively integrates its own predicted
# derivatives to produce the full voltage trajectory.  In the plots
# below, ground-truth traces appear in green and GNN predictions in
# black.  The red trace corresponds to one of the R1–R6 outer
# photoreceptors, which receive the visual stimulus directly from
# the compound eye while also integrating excitatory feedback from
# lamina interneurons (L2, L4, and amacrine cells).The **all-types** plot displays one representative neuron per cell
# type (65 traces in total), giving a broad overview of how well the
# GNN captures the diversity of circuit dynamics across all cell
# classes.  The **selected** plot zooms into a smaller subset of
# neurons chosen to highlight fine temporal structure and allow a more
# detailed comparison between prediction and ground truth.

# %%
#| lightbox: true
def show_rollout(config_name, suffix_filter="on_"):
    log_dir = log_path(configs[config_name].config_file)
    results_dir = os.path.join(log_dir, "results")
    if not os.path.isdir(results_dir):
        return
    rollout_all = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "_all" in f
                   and suffix_filter not in f and f.endswith(".png")]
    rollout_sel = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "selected" in f
                   and suffix_filter not in f and f.endswith(".png")]
    if rollout_all:
        display_image(os.path.join(results_dir, sorted(rollout_all)[0]), width=900)
    if rollout_sel:
        display_image(os.path.join(results_dir, sorted(rollout_sel)[0]), width=900)

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_rollout('flyvis_noise_free')

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_rollout('flyvis_noise_005')

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_rollout('flyvis_noise_05')

# %% [markdown]
# ## Test Metrics
#
# The model was evaluated in two modes: one-step prediction (ground-truth voltages at each frame, predicting $\widehat{dv}/dt$) and autoregressive rollout (integrating its own predictions from the first frame, receiving only the external stimulus).
# The tables below summarize the quantitative evaluation for each
# noise condition.  RMSE measures the average magnitude of prediction
# errors across all neurons, while Pearson $r$ captures how well the
# predicted and ground-truth temporal profiles are correlated on a
# per-neuron basis (reported as mean $\pm$ standard deviation over
# neurons).
#
# ### One-Step Prediction (test set)

# %%
display(Markdown(metrics_table(datasets, configs, "results_test.log")))

# %% [markdown]
# ### Autoregressive Rollout (test set)

# %%
display(Markdown(metrics_table(datasets, configs, "results_rollout.log")))

# %% [markdown]
# ## Noise-Free Evaluation
#
# A key question is whether models trained on noisy data have learned
# the underlying deterministic dynamics. To test this, the noisy models ($\sigma{=}0.05$ and
# $\sigma{=}0.5$) are evaluated on the **noise-free** test data.  If
# the GNN has correctly identified the noiseless update rule, its
# rollout on clean data should track the deterministic ground truth
# closely.

# %%
#| echo: true
#| output: false
noise_free_config = configs['flyvis_noise_free']
noisy_datasets = [ds for ds in datasets if ds[0] != 'flyvis_noise_free']

for config_name, label in noisy_datasets:
    config = configs[config_name]
    print()
    print(f"--- {label} model on noise-free test data ---")
    data_test(config, best_model='best', device=device, test_config=noise_free_config)

# %% [markdown]
# ### Rollout: Noisy Models on Noise-Free Data

# %%
#| lightbox: true
def show_nf_rollout(config_name):
    log_dir = log_path(configs[config_name].config_file)
    results_dir = os.path.join(log_dir, "results")
    if not os.path.isdir(results_dir):
        return
    rollout_all = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "_all" in f
                   and "on_noise_free" in f and f.endswith(".png")]
    rollout_sel = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "selected" in f
                   and "on_noise_free" in f and f.endswith(".png")]
    if rollout_all:
        display_image(os.path.join(results_dir, sorted(rollout_all)[0]), width=900)
    if rollout_sel:
        display_image(os.path.join(results_dir, sorted(rollout_sel)[0]), width=900)

# %% [markdown]
# ### Low noise ($\sigma = 0.05$) model on noise-free data

# %%
#| lightbox: true
show_nf_rollout('flyvis_noise_005')

# %% [markdown]
# ### High noise ($\sigma = 0.5$) model on noise-free data

# %%
#| lightbox: true
show_nf_rollout('flyvis_noise_05')

# %% [markdown]
# ### Noise-Free Rollout Metrics

# %%
nf_header = "| Metric | " + " | ".join(label for _, label in noisy_datasets) + " |"
nf_sep = "|:--|" + "|".join(":--:" for _ in noisy_datasets) + "|"
nf_rows = [nf_header, nf_sep]
for key in ['RMSE', 'Pearson r']:
    cells = []
    for config_name, _ in noisy_datasets:
        log_dir = log_path(configs[config_name].config_file)
        m = parse_results_log(os.path.join(log_dir, "results_rollout_on_noise_free.log"))
        cells.append(m.get(key, '\u2014'))
    nf_rows.append(f"| {key} | " + " | ".join(cells) + " |")
display(Markdown("\n".join(nf_rows)))

# %% [markdown]
# ### Noise and Denoising
#
# When the training data contains process noise ($\sigma = 0.05$ or
# $\sigma = 0.5$), the stochastic component of the derivatives
# $\sigma\,\xi_i(t)$ is, by definition, unpredictable from the
# current state.  A model that minimizes mean-squared error will
# therefore learn to predict only the deterministic part of
# $dv_i/dt$, effectively ignoring the noise.  The GNN has in fact
# learned a *noise-free* dynamical model: it recovers the
# deterministic update rule underlying the noisy observations.
#
# The rollout traces and metrics above confirm this interpretation.
# Models trained on noisy data, when evaluated on noise-free test
# stimuli, track the clean ground truth with high fidelity.  This
# demonstrates that the GNN implicitly denoises the dynamics.  It
# extracts the systematic circuit computation from stochastic
# observations without any explicit noise model or denoising
# objective.

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
