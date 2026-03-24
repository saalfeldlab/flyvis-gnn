# %% [raw]
# ---
# title: "Data Generation: Drosophila Visual System"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - Simulation
#   - Data Generation
# execute:
#   echo: false
# image: "graphs_data/fly/flyvis_noise_free/activity_traces.png"
# description: "Simulate the Drosophila visual system with 13,741 neurons across 65 cell types at three intrinsic noise levels ($\\sigma = 0$, $0.05$, $0.5$) and generate voltage traces for GNN training and testing."
# ---

# %% [markdown]
# ## Simulation
#
# We simulated neural activity in the *Drosophila* visual system using
# [*flyvis*](https://github.com/TuragaLab/flyvis)' pretrained models [1].  The recurrent neural network contains
# 13,741 neurons from 65 cell types and 434,122 synaptic connections,
# corresponding to real neurons and their synapses.  We restricted the original
# 721 retinotopic columns to the central subset of 217. Each neuron is modeled
# as a non-spiking compartment governed by
#
# $$\tau_i\frac{dv_i(t)}{dt} = -v_i(t) + V_i^{\text{rest}} + \sum_{j\in\mathcal{N}_i} \mathbf{W}_{ij}\,\text{ReLU}\!\big(v_j(t)\big) + I_i(t) + \sigma\,\xi_i(t),$$
#
# where $\tau_i$ and $V_i^{\text{rest}}$ are cell-type parameters,
# $\mathbf{W}_{ij}$ is the connectome-constrained synaptic weight,
# $I_i(t)$ the visual input, and
# $\xi_i(t) \sim \mathcal{N}(0,1)$ is independent Gaussian noise scaled
# by $\sigma$. The noise term $\sigma\,\xi_i(t)$ models intrinsic stochasticity in the
# membrane dynamics (e.g. channel noise, synaptic variability).
#

# Unlike measurement noise added post hoc, this intrinsic
# noise alters the dynamical trajectory of $v_i(t)$ and couples through the
# connectivity matrix $\mathbf{W}$.  As a dynamical perturbation, it widens
# the distribution of visited voltages and propagates through synaptic
# connections, enriching the training signal.
#
# We generated data at three noise levels $\sigma$:
#
# | Dataset | $\sigma$ | Description |
# |---------|----------|-------------|
# | `flyvis_noise_free` | 0.0 | Deterministic (no intrinsic noise) |
# | `flyvis_noise_005` | 0.05 | Low intrinsic noise |
# | `flyvis_noise_05` | 0.5 | High intrinsic noise |

# %%
#| output: false
import os
import warnings

from IPython.display import Image, display

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.utils import set_device, add_pre_folder, load_and_display, graphs_data_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)


def display_image(path, width=700):
    """Display a full-resolution image; width controls inline size (px)."""
    if not os.path.isfile(path):
        print(f"  image not found: {path} (run data generation first)")
        return
    display(Image(filename=path, width=width))

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

for config_name, label in datasets:
    print(f"{label}: {graphs_dirs[config_name]}")

# %% [markdown]
# ## Visual Stimulus
#
# The visual input $I_i(t)$ is derived from the DAVIS dataset [2, 3], a
# collection of natural video sequences at 480p resolution originally
# developed for optical flow and video segmentation benchmarks.  Each RGB
# frame is center-cropped to 60% of its spatial extent, converted to
# grayscale luminance, and resampled onto the hexagonal photoreceptor
# lattice of 217 columns via a Gaussian box-eye filter (extent 8,
# kernel size 13).  Each column feeds 8 photoreceptor types (R1–R8),
# giving 1,736 input neurons.  Videos longer than 80 frames are split
# into 50-frame chunks; temporal interpolation resamples each chunk to
# match the simulation time step $\Delta t = 0.02$.
#
# The full set of sequences is augmented with horizontal and vertical
# flips and four rotations (0°, 90°, 180°, 270°) of the hexagonal
# array.  All augmentations of the same base video are kept together and
# the dataset is split 80/20 at the *base-video level* to prevent data
# leakage between training and testing.  The sequences within each split
# are shuffled (seed 42) and concatenated into a continuous stimulus
# stream.

# %%
#| echo: true
#| output: false
print()
print("=" * 80)
print("GENERATE - Simulating fly visual system at three noise levels")
print("=" * 80)

for config_name, label in datasets:
    config = configs[config_name]
    graphs_dir = graphs_dirs[config_name]
    print()
    print(f"--- {label} (noise_model_level={config.simulation.noise_model_level}) ---")

    data_exists = os.path.isdir(f'{graphs_dir}/x_list_train') or os.path.isdir(f'{graphs_dir}/x_list_0')
    if data_exists:
        print(f"  data already exists at {graphs_dir}/")
        print("  skipping simulation...")
    else:
        print(f"  simulating {config.simulation.n_neurons} neurons, {config.simulation.n_neuron_types} types")
        print(f"  generating {config.simulation.n_frames} time frames")
        print(f"  visual input: {config.simulation.visual_input_type}")
        print(f"  output: {graphs_dir}/")
        print()
        data_generate(
            config,
            device=device,
            visualize=True,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=False,
            save=True,
            step=100,
        )

# %% [markdown]
# ### Training sequences
# First frames of the shuffled DAVIS sequences assigned to training
# (shown for the noise-free dataset).

# %%
#| lightbox: true
graphs_dir_0 = graphs_dirs[datasets[0][0]]
display_image(f"{graphs_dir_0}/shuffle_first_frames_train.png", width=900)

# %% [markdown]
# ### Test sequences
# First frames of the shuffled DAVIS sequences assigned to testing
# (shown for the noise-free dataset).

# %%
#| lightbox: true
display_image(f"{graphs_dir_0}/shuffle_first_frames_test.png", width=900)

# %% [markdown]
# ### Visual Stimulus Movie
#
# The animation below shows the visual input $I_i(t)$ as seen by the
# 217 hexagonal columns of the compound eye.  Each hexagon represents
# one retinotopic column (8 photoreceptors, R1–R8, share the same
# input), and the color encodes the grayscale luminance at each time
# step.  The stimulus is derived from the DAVIS natural video dataset
# and resampled onto the hexagonal lattice.

# %%
import zarr
import numpy as np
from IPython.display import Video
from matplotlib.animation import FFMpegWriter

stimulus_video_path = os.path.join(graphs_dir_0, "stimulus_hexagonal.mp4")

if not os.path.isfile(stimulus_video_path):
    import matplotlib.pyplot as plt
    # Load stimulus and positions from zarr
    stim = zarr.open(store=os.path.join(graphs_dir_0, "x_list_train", "stimulus.zarr"), mode='r')
    pos = np.array(zarr.open(store=os.path.join(graphs_dir_0, "x_list_train", "pos.zarr"), mode='r'))

    n_input = configs[datasets[0][0]].simulation.n_input_neurons  # 1736
    n_columns = 217
    group_size = n_input // n_columns  # 8 photoreceptors per column

    # Use one photoreceptor per column (first in each group) for display
    col_indices = np.arange(0, n_input, group_size)
    X = pos[col_indices]  # (217, 2)

    # Render video: subsample frames (every 2nd frame, first 2000 frames)
    frame_step = 2
    n_frames_video = min(2000, stim.shape[0])
    frame_indices = range(0, n_frames_video, frame_step)

    # Precompute color range from a sample
    sample_frames = np.array(stim[:n_frames_video:frame_step*10])
    vmin = float(sample_frames[:, col_indices].min())
    vmax = float(sample_frames[:, col_indices].max())
    del sample_frames

    from matplotlib.patches import RegularPolygon
    from matplotlib.collections import PatchCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Hex radius: half the minimum distance between column centres, scaled up
    # to close gaps between adjacent hexagons.
    from scipy.spatial import KDTree
    dists, _ = KDTree(X).query(X, k=2)
    hex_radius = float(dists[:, 1].min()) / 2 * 1.15

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(5, 5))
    fps = 10
    writer = FFMpegWriter(fps=fps, metadata={'title': 'Visual Stimulus'})

    with writer.saving(fig, stimulus_video_path, dpi=150):
        for k in frame_indices:
            ax.clear()
            ax.set_axis_off()
            frame = np.array(stim[k])
            vals = frame[col_indices]
            patches = [RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                      orientation=0) for x, y in X]
            pc = PatchCollection(patches, edgecolors='none')
            pc.set_facecolor(cmap(norm(vals)))
            ax.add_collection(pc)
            ax.set_xlim(X[:, 0].min() - hex_radius, X[:, 0].max() + hex_radius)
            ax.set_ylim(X[:, 1].min() - hex_radius, X[:, 1].max() + hex_radius)
            ax.set_aspect('equal')
            ax.set_title(f'frame {k}', fontsize=10)
            writer.grab_frame()
    plt.close(fig)

# %%
if os.path.isfile(stimulus_video_path):
    display(Video(stimulus_video_path, embed=True, width=800))

# %% [markdown]
# ## Activity Traces
#
# Each plot below shows 100 randomly selected voltage traces $v_i(t)$
# over the first 10,000 time steps (out of 64,000 total). The three plots corresponds to the three level of intrinsinc noise used in simulations.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)

# %%
#| lightbox: true
display_image(f"{graphs_dirs['flyvis_noise_free']}/activity.png", width=850)

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)

# %%
#| lightbox: true
display_image(f"{graphs_dirs['flyvis_noise_005']}/activity.png", width=850)

# %% [markdown]
# ### High noise ($\sigma = 0.5$)

# %%
#| lightbox: true
display_image(f"{graphs_dirs['flyvis_noise_05']}/activity.png", width=850)

# %% [markdown]
# ## References
#
# [1] J. K. Lappalainen et al., "Connectome-constrained networks predict
# neural activity across the fly visual system," *Nature*, 2024.
# [doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)
#
# [2] D. J. Butler et al., "A Naturalistic Open Source Movie for Optical
# Flow Evaluation," *ECCV*, 2012.
# [doi:10.1007/978-3-642-33783-3_44](https://doi.org/10.1007/978-3-642-33783-3_44)
#
# [3] F. Perazzi et al., "A Benchmark Dataset and Evaluation Methodology
# for Video Object Segmentation," *CVPR*, 2016.
# [doi:10.1109/CVPR.2016.85](https://doi.org/10.1109/CVPR.2016.85)
