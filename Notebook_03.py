# %% [raw]
# ---
# title: "GNN Testing: Robustness to Edge Removal"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Ablation
# execute:
#   echo: false
# image: "log/fly/flyvis_noise_free/results/rollout_noise_free_DAVIS_selected_on_noise_free_mask_50.png"
# description: "Test whether the GNN has learned the circuit's computational rules by randomly ablating 50% of synaptic connections and comparing the model's predictions to the simulator under the same reduced connectivity."
# ---

# %% [markdown]
# ## Ablation
#

#
# A well-trained dynamical model should capture the circuit's
# computational rules, not merely memorize its specific activities.To test this, we randomly ablated 50% of the synaptic connections and
# regenerated the ground truth under the ablated connectivity. The same edge mask is then applied to the GNN's learned weights
# $\widehat{W}_{ij}$, so both the simulator and the model operate on identical reduced circuits.  If the GNN has learned the correct
# message-passing functions $f_\theta$ and $g_\phi$, it should generalize to the reduced connectivity without retraining.
# We tested robustness at three noise levels: $\sigma = 0$ noise-free), $\sigma = 0.05$ (low noise), and $\sigma = 0.5$
# (high noise). The test is performed using the best models in ([Notebook 01](Notebook_01.html)).

# %%
#| output: false
import glob
import os
import warnings

from IPython.display import Image, Markdown, display

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
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


# %%
#| output: false
# Base configs (trained models)
base_datasets = [
    ('flyvis_noise_free', 'Noise-free'),
    ('flyvis_noise_005', 'Noise 0.05'),
    ('flyvis_noise_05', 'Noise 0.5'),
]

# Ablated configs (50% edge mask)
mask_datasets = [
    ('flyvis_noise_free_mask_50', 'Noise-free (50% ablation)'),
    ('flyvis_noise_005_mask_50', 'Noise 0.05 (50% ablation)'),
    ('flyvis_noise_05_mask_50', 'Noise 0.5 (50% ablation)'),
]

config_root = "./config"
base_configs = {}
mask_configs = {}

for config_name, label in base_datasets:
    config_file, pre_folder = add_pre_folder(config_name)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name
    base_configs[config_name] = config

for config_name, label in mask_datasets:
    config_file, pre_folder = add_pre_folder(config_name)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name
    mask_configs[config_name] = config

device = set_device(base_configs[base_datasets[0][0]].training.device)

# %% [markdown]
# ## Generate Ablated Data
#
# For each noise condition, the simulator regenerates voltage
# traces with 50% of edges randomly zeroed.
# The resulting ablation mask (a boolean tensor
# indicating which edges survive) is saved alongside the data so
# that the exact same mask can be applied to the GNN's learned
# weights at test time.  This ensures a fair comparison: both the
# ground-truth simulator and the learned model operate on precisely
# the same reduced circuit.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("GENERATE - data with 50% edge ablation")
print("=" * 80)

for config_name, label in mask_datasets:
    config = mask_configs[config_name]
    graphs_dir = graphs_data_path(config.dataset)
    mask_path = os.path.join(graphs_dir, "ablation_mask.pt")
    has_test_data = (os.path.isfile(os.path.join(graphs_dir, "x_list_test.pt"))
                     or os.path.isfile(os.path.join(graphs_dir, "x_list_test.npy"))
                     or os.path.isdir(os.path.join(graphs_dir, "x_list_test")))

    if os.path.exists(mask_path) and has_test_data:
        print(f"\n--- {label} ---")
        print(f"  ablated data already exists at {graphs_dir}/")
        print("  skipping generation...")
    else:
        print(f"\n--- {label} ---")
        print(f"  generating with ablation_ratio={config.simulation.ablation_ratio}")
        data_generate(config, device=device, visualize=False, style='color')

# %% [markdown]
# ## Test: GNN on Ablated Data
#
# Each model, trained on the full non-ablated connectivity, is now
# evaluated on the ablated test data.  Before evaluation, the saved
# ablation mask is loaded and applied to the model's learned weight
# vector $\widehat{\mathbf{W}}$, zeroing out the same edges that were
# removed in the simulator.  No retraining or fine-tuning is
# performed; the model must rely on the message-passing functions it
# learned from the original circuit.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("TEST - GNN models on ablated data")
print("=" * 80)

# Check that trained models exist for all base configs
pairs = list(zip(base_datasets, mask_datasets))
missing_models = []
for base_name, base_label in base_datasets:
    log_dir = log_path(base_configs[base_name].config_file)
    model_files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
    if not model_files:
        missing_models.append(base_label)

if missing_models:
    msg = ", ".join(missing_models)
    raise RuntimeError(
        f"No trained models found for: {msg}. "
        f"Please run Notebook_01 first to train the GNN models."
    )

for (base_name, base_label), (mask_name, mask_label) in pairs:
    print(f"\n--- {base_label} model on ablated data ---")
    data_test(
        base_configs[base_name],
        best_model='best',
        device=device,
        test_config=mask_configs[mask_name],
    )

# %% [markdown]
# ## Ablation Rollout Traces
#
# The rollout plots below compare ground-truth voltages (green) and
# GNN predictions (black) under 50% edge ablation.  The red trace
# corresponds to one of the R1–R6 outer photoreceptors, which receive
# the visual stimulus directly from the compound eye and also
# integrate excitatory feedback from lamina interneurons (L2, L4, and
# amacrine cells). The **all-types** plot shows one representative neuron per cell type
# (65 traces), providing a global view of how the ablated GNN
# captures the circuit dynamics.  The **selected** plot zooms into a
# subset for more detailed inspection.

# %%
#| lightbox: true
def show_ablation_rollout(base_name, mask_name):
    log_dir = log_path(base_configs[base_name].config_file)
    results_dir = os.path.join(log_dir, "results")
    if not os.path.isdir(results_dir):
        return
    mask_suffix = mask_name.replace('flyvis_', '')
    rollout_all = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "_all" in f
                   and mask_suffix in f and f.endswith(".png")]
    rollout_sel = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "selected" in f
                   and mask_suffix in f and f.endswith(".png")]
    if rollout_all:
        display_image(os.path.join(results_dir, sorted(rollout_all)[0]), width=900)
    if rollout_sel:
        display_image(os.path.join(results_dir, sorted(rollout_sel)[0]), width=900)

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
show_ablation_rollout('flyvis_noise_free', 'flyvis_noise_free_mask_50')

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
show_ablation_rollout('flyvis_noise_005', 'flyvis_noise_005_mask_50')

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
show_ablation_rollout('flyvis_noise_05', 'flyvis_noise_05_mask_50')

# %% [markdown]
# ## Ablation Metrics
#
# The tables below report RMSE and Pearson $r$ (mean $\pm$ std over
# neurons) for the ablated evaluation, for both one-step prediction
# and autoregressive rollout.

# %%
header = "| Metric | " + " | ".join(bl for _, bl in base_datasets) + " |"
sep = "|:--|" + "|".join(":--:" for _ in base_datasets) + "|"

# One-step prediction
rows_1s = [header, sep]
for key in ['RMSE', 'Pearson r']:
    cells = []
    for (base_name, _), (mask_name, _) in pairs:
        log_dir = log_path(base_configs[base_name].config_file)
        mask_suffix = mask_name.replace('flyvis_', '')
        m = parse_results_log(os.path.join(log_dir, f"results_test_on_{mask_suffix}.log"))
        cells.append(m.get(key, '\u2014'))
    rows_1s.append(f"| {key} | " + " | ".join(cells) + " |")

display(Markdown("### One-Step Prediction (ablated data)\n\n" + "\n".join(rows_1s)))

# Rollout
rows_ro = [header, sep]
for key in ['RMSE', 'Pearson r']:
    cells = []
    for (base_name, _), (mask_name, _) in pairs:
        log_dir = log_path(base_configs[base_name].config_file)
        mask_suffix = mask_name.replace('flyvis_', '')
        m = parse_results_log(os.path.join(log_dir, f"results_rollout_on_{mask_suffix}.log"))
        cells.append(m.get(key, '\u2014'))
    rows_ro.append(f"| {key} | " + " | ".join(cells) + " |")

display(Markdown("### Autoregressive Rollout (ablated data)\n\n" + "\n".join(rows_ro)))

# %% [markdown]
# ## Noise-Free Ablation Evaluation
#
# As in the non-ablated case (Notebook 02), we cross-test the noisy
# models on clean data to verify that the denoising property is
# preserved under ablation.  The models trained on noisy data
# ($\sigma{=}0.05$ and $\sigma{=}0.5$) are evaluated on the
# **noise-free ablated** test data.  If the GNN has learned the
# deterministic dynamics, it should
# still track the clean ground truth even after losing half its
# synaptic connections.

# %%
#| echo: true
#| output: false
noise_free_mask_config = mask_configs['flyvis_noise_free_mask_50']
noisy_base = [ds for ds in base_datasets if ds[0] != 'flyvis_noise_free']

for base_name, base_label in noisy_base:
    print()
    print(f"--- {base_label} model on noise-free ablated data ---")
    data_test(
        base_configs[base_name],
        best_model='best',
        device=device,
        test_config=noise_free_mask_config,
    )

# %% [markdown]
# ### Rollout: Noisy Models on Noise-Free Ablated Data

# %%
#| lightbox: true
def show_nf_ablation_rollout(base_name):
    log_dir = log_path(base_configs[base_name].config_file)
    results_dir = os.path.join(log_dir, "results")
    if not os.path.isdir(results_dir):
        return
    rollout_all = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "_all" in f
                   and "on_noise_free_mask_50" in f and f.endswith(".png")]
    rollout_sel = [f for f in os.listdir(results_dir)
                   if f.startswith("rollout_") and "selected" in f
                   and "on_noise_free_mask_50" in f and f.endswith(".png")]
    if rollout_all:
        display_image(os.path.join(results_dir, sorted(rollout_all)[0]), width=900)
    if rollout_sel:
        display_image(os.path.join(results_dir, sorted(rollout_sel)[0]), width=900)

# %% [markdown]
# ### Low noise ($\sigma = 0.05$) on noise-free ablated data

# %%
#| lightbox: true
show_nf_ablation_rollout('flyvis_noise_005')

# %% [markdown]
# ### High noise ($\sigma = 0.5$) on noise-free ablated data

# %%
#| lightbox: true
show_nf_ablation_rollout('flyvis_noise_05')

# %% [markdown]
# ### Noise-Free Ablation Metrics

# %%
nf_header = "| Metric | " + " | ".join(bl for _, bl in noisy_base) + " |"
nf_sep = "|:--|" + "|".join(":--:" for _ in noisy_base) + "|"
nf_rows = [nf_header, nf_sep]
for key in ['RMSE', 'Pearson r']:
    cells = []
    for base_name, _ in noisy_base:
        log_dir = log_path(base_configs[base_name].config_file)
        m = parse_results_log(os.path.join(log_dir, "results_rollout_on_noise_free_mask_50.log"))
        cells.append(m.get(key, '\u2014'))
    nf_rows.append(f"| {key} | " + " | ".join(cells) + " |")
display(Markdown("\n".join(nf_rows)))

# %% [markdown]
# ### Denoising Under Ablation
#
# The results confirm that the implicit denoising property observed
# in Notebook 02 is preserved under ablation.  Models trained on noisy
# data still recover the deterministic dynamics when evaluated on
# noise-free ablated data, tracking the clean ground truth closely.
# This is a strong indication that the GNN has learned the true
# message-passing computation.  The functions $f_\theta$ and $g_\phi$
# generalize across both noise conditions and connectivity
# perturbations, rather than being overfitted to the specific training
# dataset.
