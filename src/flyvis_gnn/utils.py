import gc
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


import tensorstore as ts

# ---------------------------------------------------------------------------
# Configurable data root (graphs_data/ and log/ location)
# ---------------------------------------------------------------------------

_DATA_ROOT_CACHE = None


def get_data_root() -> str:
    """Return the data root directory from data_paths.json, or '.' if not found.

    Searches for data_paths.json in two locations (first match wins):
      1. Current working directory
      2. Two levels up from this file (repo root when running from source)
    """
    global _DATA_ROOT_CACHE
    if _DATA_ROOT_CACHE is not None:
        return _DATA_ROOT_CACHE
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), 'data_paths.json'),
        os.path.normpath(os.path.join(_this_dir, '..', 'data_paths.json')),       # src/data_paths.json
        os.path.normpath(os.path.join(_this_dir, '..', '..', 'data_paths.json')), # repo root
    ]
    for json_path in candidates:
        print(f"[get_data_root] checking {json_path}  exists={os.path.isfile(json_path)}")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                _DATA_ROOT_CACHE = json.load(f)['data_root']
            print(f"[get_data_root] loaded data_root = {_DATA_ROOT_CACHE!r}")
            return _DATA_ROOT_CACHE
    _DATA_ROOT_CACHE = '.'
    print("[get_data_root] data_paths.json not found, using '.'")
    return _DATA_ROOT_CACHE


def graphs_data_path(*parts: str) -> str:
    """Build path under graphs_data/: graphs_data_path('fly', 'x.npy') -> '{data_root}/graphs_data/fly/x.npy'"""
    return os.path.join(get_data_root(), 'graphs_data', *parts)


def log_path(*parts: str) -> str:
    """Build path under log/: log_path('fly', 'models') -> '{data_root}/log/fly/models'"""
    return os.path.join(get_data_root(), 'log', *parts)


def config_path(*parts):
    """Build path under config/: config_path('fly', 'x.yaml') -> '{data_root}/config/fly/x.yaml'"""
    return os.path.join(get_data_root(), 'config', *parts)


def setup_flyvis_model_path():
    """Install bundled model 000 into the flyvis package if not already present.

    The repo ships a copy of model 000 in assets/flyvis_model/.  If flyvis
    doesn't already have the pretrained results at its results_dir, we copy
    the bundled files there so that ``NetworkView('flow/0000/000')`` works.
    """
    import flyvis
    target_dir = os.path.join(str(flyvis.results_dir), "flow", "0000", "000")
    if os.path.exists(os.path.join(target_dir, "best_chkpt")):
        return  # flyvis already has the model
    # Look for bundled model relative to this file (src/flyvis_gnn/utils.py -> repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bundled = os.path.join(repo_root, "assets", "flyvis_model", "flow", "0000", "000")
    if not os.path.isdir(bundled):
        return
    import shutil as _shutil
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    _shutil.copytree(bundled, target_dir)
    logging.getLogger(__name__).info(
        f"installed bundled flyvis model 000 to {target_dir}"
    )


def open_gcs_zarr(url: str):
    # Strip accidental prefixes like:  'str = "gs://.../aligned"'
    url = url.strip()
    if url.startswith("str"):
        # remove leading 'str ='
        url = re.sub(r'^str\s*=\s*', '', url).strip()
        # strip surrounding quotes
        url = url.strip('\'"')

    if not url.startswith("gs://"):
        raise ValueError(f"Expected gs:// URL, got: {url}")

    # First try zarr3 with a plain kvstore string
    try:
        return ts.open({'driver': 'zarr3', 'kvstore': url, 'open': True}).result()
    except Exception:
        pass

    # Fall back to explicit GCS kvstore spec (works across TS versions), zarr3 → zarr2
    bucket_path = url[len("gs://"):]
    bucket, path = bucket_path.split('/', 1)
    for drv in ('zarr3', 'zarr'):
        try:
            return ts.open({
                'driver': drv,
                'kvstore': {'driver': 'gcs', 'bucket': bucket, 'path': path},
                'open': True
            }).result()
        except Exception:
            continue
    raise RuntimeError(f"Could not open Zarr at {url} with zarr3 or zarr2")


def migrate_state_dict(state_dict: dict) -> dict:
    """Migrate legacy checkpoint keys (lin_edge->g_phi, lin_phi->f_theta)."""
    state_dict['model_state_dict'] = {
        k.replace('lin_edge.', 'g_phi.').replace('lin_phi.', 'f_theta.'): v
        for k, v in state_dict['model_state_dict'].items()
    }
    return state_dict


def sort_key(filename: str) -> int:
            # Extract the numeric parts using regular expressions
            if filename.split('_')[-2] == 'graphs':
                return 0
            else:
                return 1E7 * int(filename.split('_')[-2]) + int(filename.split('_')[-1][:-3])


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def set_device(device: str = 'auto') -> str:
    """
    Set the device to use for computations. If 'auto' is specified, the device is chosen automatically:
     * if GPUs are available, the GPU with the most free memory is chosen
     * if MPS is available, MPS is used
     * otherwise, the CPU is used
    :param device: The device to use for computations. Automatically chosen if 'auto' is specified (default).
    :return: The torch.device object that is used for computations.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Unset CUDA_VISIBLE_DEVICES

    if device == 'auto':
        if torch.cuda.is_available():
            try:
                # Query free memory via PyTorch to avoid index mismatch with nvidia-smi
                num_gpus = torch.cuda.device_count()
                max_free_memory = -1
                best_device_id = -1
                for i in range(num_gpus):
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    if free_mem > max_free_memory:
                        max_free_memory = free_mem
                        best_device_id = i
                if best_device_id == -1:
                    raise ValueError("Could not determine the GPU with the most free memory.")

                device = f'cuda:{best_device_id}'
                torch.cuda.set_device(best_device_id)
                total_memory_gb = torch.cuda.get_device_properties(best_device_id).total_memory / 1024 ** 3
                free_memory_gb = max_free_memory / 1024 ** 3
                print(
                    f"using device: {device}, name: {torch.cuda.get_device_name(best_device_id)}, "
                    f"total memory: {total_memory_gb:.2f} GB, free memory: {free_memory_gb:.2f} GB")
            except Exception as e:
                print(f"Failed to get GPU information: {e}")
                device = 'cpu'
                print(f"using device: {device}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"using device: {device}")
        else:
            device = 'cpu'
            print(f"using device: {device}")
    return device


def choose_boundary_values(bc_name):
    def identity(x):
        return x

    def periodic(x):
        return torch.remainder(x, 1.0)

    def periodic_wall(x):
        y = torch.remainder(x[:,0:1], 1.0)
        return torch.cat((y,x[:,1:2]), 1)

    def shifted_periodic(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5

    def shifted_periodic_wall(x):
        y = torch.remainder(x[:,0:1] - 0.5, 1.0) - 0.5
        return torch.cat((y,x[:,1:2]), 1)


    match bc_name:
        case 'no':
            return identity, identity
        case 'periodic':
            return periodic, shifted_periodic
        case 'wall':
            return periodic_wall, shifted_periodic_wall
        case _:
            raise ValueError(f'unknown boundary condition {bc_name}')


class CustomColorMap:
    def __init__(self, config):
        self.cmap_name = config.plotting.colormap
        self.model_name = config.graph_model.particle_model_name

        if self.cmap_name == 'tab10':
            self.nmap = 8
        else:
            self.nmap = config.simulation.n_neurons

        self.has_mesh = 'Mesh' in self.model_name

    def color(self, index):

        if ('PDE_F' in self.model_name) | ('PDE_MLPs' in self.model_name):
            match index:
                case 0:
                    color = (0.75, 0.75, 0.75)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.5, 0.75, 0)
                case 4:
                    color = (0, 0.75, 0)
                case 5:
                    color = (0.5, 0, 0.25)
                case _:
                    color = (1, 1, 1)
        elif self.model_name == 'PDE_E':
            match index:
                case 0:
                    color = (1, 1, 1)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.75, 0, 0)
                case _:
                    color = (0.5, 0.5, 0.5)
        elif self.has_mesh:
            if index == 0:
                color = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.cmap_name)
                color = color_map(index / self.nmap)
        else:
            color_map = plt.colormaps.get_cmap(self.cmap_name)
            if self.cmap_name == 'tab20':
                color = color_map(index % 20)
            else:
                color = color_map(index)

        return color


def add_pre_folder(config_file_):

    if 'arbitrary' in config_file_:
        config_file = os.path.join('arbitrary', config_file_)
        pre_folder = 'arbitrary/'
    elif 'boids' in config_file_:
        config_file = os.path.join('boids', config_file_)
        pre_folder = 'boids/'
    elif 'Coulomb' in config_file_:
        config_file = os.path.join('Coulomb', config_file_)
        pre_folder = 'Coulomb/'
    elif 'fluids' in config_file_:
        config_file = os.path.join('fluids', config_file_)
        pre_folder = 'fluids/'
    elif 'gravity' in config_file_:
        config_file = os.path.join('gravity', config_file_)
        pre_folder = 'gravity/'
    elif 'springs' in config_file_:
        config_file = os.path.join('springs', config_file_)
        pre_folder = 'springs/'
    elif 'CElegans' in config_file_:
        config_file = os.path.join('CElegans', config_file_)
        pre_folder = 'CElegans/'
    elif 'fly' in config_file_:
        config_file = os.path.join('fly', config_file_)
        pre_folder = 'fly/'
    elif 'zebra' in config_file_:
        config_file = os.path.join('zebrafish', config_file_)
        pre_folder = 'zebrafish/'
    elif 'signal' in config_file_:
        config_file = os.path.join('signal', config_file_)
        pre_folder = 'signal/'
    elif 'falling_water_ramp' in config_file_:
        config_file = os.path.join('falling_water_ramp', config_file_)
        pre_folder = 'falling_water_ramp/'
    elif 'multimaterial' in config_file_:
        config_file = os.path.join('multimaterial', config_file_)
        pre_folder = 'multimaterial/'
    elif 'RD_RPS' in config_file_:
        config_file = os.path.join('reaction_diffusion', config_file_)
        pre_folder = 'reaction_diffusion/'
    elif 'wave' in config_file_:
        config_file = os.path.join('wave', config_file_)
        pre_folder = 'wave/'
    elif ('cell' in config_file_) | ('cardio' in config_file_) | ('U2OS' in config_file_):
        config_file = os.path.join('cell', config_file_)
        pre_folder = 'cell/'
    elif 'mouse' in config_file_:
        config_file = os.path.join('mouse_city', config_file_)
        pre_folder = 'mouse_city/'


    return config_file, pre_folder


def create_log_dir(config=[], erase=True):

    log_dir = log_path(config.config_file)
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/external_input'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/MLP0'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/MLP1'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
    if config.training.n_ghosts > 0:
        os.makedirs(os.path.join(log_dir, 'tmp_training/ghost'), exist_ok=True)

    if erase:
        # erase old models to avoid mismatch with new ground truth
        files = glob.glob(f"{log_dir}/models/*")
        for f in files:
            os.remove(f)
        # erase entire results folder and recreate
        results_dir = os.path.join(log_dir, 'results')
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        # erase entire tmp_training folder and recreate
        tmp_training_dir = os.path.join(log_dir, 'tmp_training')
        if os.path.exists(tmp_training_dir):
            shutil.rmtree(tmp_training_dir)
        os.makedirs(os.path.join(log_dir, 'tmp_training/external_input'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/function/MLP0'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/function/MLP1'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
        if config.training.n_ghosts > 0:
            os.makedirs(os.path.join(log_dir, 'tmp_training/ghost'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)


    return log_dir, logger


def fig_init(fontsize=48, formatx='%.2f', formaty='%.2f'):
    # from matplotlib import rc, font_manager
    # from numpy import arange, cos, pi
    # from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title, \
    #     grid, savefig, show
    # sizeOfFont = 12
    # fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
    #                   'weight': 'normal', 'size': sizeOfFont}
    # ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    #                                          size=sizeOfFont, weight='normal', stretch='normal')
    # rc('text', usetex=True)
    # rc('font', **fontProperties)
    # figure(1, figsize=(6, 4))
    # ax = axes([0.1, 0.1, 0.8, 0.7])
    # t = arange(0.0, 1.0 + 0.01, 0.01)
    # s = cos(2 * 2 * pi * t) + 2
    # plot(t, s)
    # for label in ax.get_xticklabels():
    #     label.set_fontproperties(ticks_font)
    # for label in ax.get_yticklabels():
    #     label.set_fontproperties(ticks_font)
    # xlabel(r'\textbf{time (s)}')
    # ylabel(r'\textit{voltage (mV)}', fontsize=16, family='Helvetica')
    # title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
    #       fontsize=16, color='r')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks([])
    plt.yticks([])
    # ax.xaxis.get_major_formatter()._usetex = False
    # ax.yaxis.get_major_formatter()._usetex = False
    ax.tick_params(axis='both', which='major', pad=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
    ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Set axis line alpha to 0.75
    for spine in ax.spines.values():
        spine.set_alpha(0.75)

    return fig, ax


def check_and_clear_memory(
        device: str = None,
        iteration_number: int = None,
        every_n_iterations: int = 100,
        memory_percentage_threshold: float = 0.6
):
    """
    Check the memory usage of a GPU and clear the cache every n iterations or if it exceeds a certain threshold.
    :param device: The device to check the memory usage for.
    :param iteration_number: The current iteration number.
    :param every_n_iterations: Clear the cache every n iterations.
    :param memory_percentage_threshold: Percentage of memory usage that triggers a clearing.
    """

    if device and 'cuda' in device:
        logging.getLogger(__name__)

        if (iteration_number % every_n_iterations == 0):

            # logger.info(f"Recurrent cuda cleanining")
            # logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            torch.cuda.memory_allocated(device)
            gc.collect()
            torch.cuda.empty_cache()

            # if (iteration_number==0):
            #     logger.info(f"total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            #     logger.info(f"total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")


        if torch.cuda.memory_allocated(device) > memory_percentage_threshold * torch.cuda.get_device_properties(device).total_memory:
            print ("memory usage is high. Calling garbage collector and clearing cache.")
            # logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
            gc.collect()
            torch.cuda.empty_cache()

def large_tensor_nonzero(tensor, chunk_size=2**30):
    indices = []
    num_chunks = (tensor.numel() + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        chunk = tensor.flatten()[i * chunk_size:(i + 1) * chunk_size]
        chunk_indices = chunk.nonzero(as_tuple=True)[0] + i * chunk_size
        indices.append(chunk_indices)
    indices = torch.cat(indices)
    row_indices = indices // tensor.size(1)
    col_indices = indices % tensor.size(1)
    return torch.stack([row_indices, col_indices])


def get_equidistant_points(n_points=1024):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    r = np.sqrt(indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)

    return x, y


def compute_feve(true, pred, n_repeats=None):
    """
    Compute FEVE metric (Stringer et al.)
    true/pred: (n_neurons, n_timepoints)
    n_repeats: number of trial repeats (if data has repeated stimuli)
    """
    if n_repeats is not None:
        # Reshape to (n_neurons, n_trials, n_frames_per_trial)
        n_frames_per_trial = true.shape[1] // n_repeats
        true_trials = true.reshape(-1, n_repeats, n_frames_per_trial)

        # Trial-averaged response (stimulus-locked component)
        mean_response = np.mean(true_trials, axis=1, keepdims=True)

        # Explainable variance (trial-to-trial variability)
        var_explainable = np.var(true_trials - mean_response, axis=(1,2))
    else:
        # Without trial structure, use total variance
        var_explainable = np.var(true, axis=1)

    # Prediction error variance
    var_error = np.var(true - pred, axis=1)

    # FEVE per neuron
    feve = 1 - var_error / (var_explainable + 1e-8)
    feve = np.clip(feve, -np.inf, 1.0)  # Can be negative if pred worse than mean

    return feve


def compute_trace_metrics(true, pred, label=""):
    """compute RMSE, Pearson correlation metrics, FEVE, and R²."""
    n_samples = true.shape[0]
    rmse_list, pearson_list, r2_list = [], [], []

    for i in range(n_samples):
        valid = ~(np.isnan(true[i]) | np.isnan(pred[i]))
        if valid.sum() > 0:
            rmse_list.append(np.sqrt(np.mean((true[i,valid] - pred[i,valid])**2)))
            if valid.sum() > 1 and np.std(true[i,valid]) > 1e-8 and np.std(pred[i,valid]) > 1e-8:
                pearson_list.append(pearsonr(true[i,valid], pred[i,valid])[0])
            else:
                pearson_list.append(np.nan)

            # Compute R²
            ss_res = np.sum((true[i,valid] - pred[i,valid]) ** 2)
            ss_tot = np.sum((true[i,valid] - np.mean(true[i,valid])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else np.nan
            r2_list.append(r2)

    rmse = np.array(rmse_list) if rmse_list else np.array([np.nan])
    pearson = np.array(pearson_list) if pearson_list else np.array([np.nan])
    r2 = np.array(r2_list) if r2_list else np.array([np.nan])

    if len(rmse_list) == 0:
        print("\033[91mERROR: all neurons contain NaN — model diverged\033[0m")
    else:
        print(f"Pearson r: \033[92m{np.nanmean(pearson):.3f}\033[0m ± {np.nanstd(pearson):.3f} [{np.nanmin(pearson):.3f}, {np.nanmax(pearson):.3f}]")
        print(f"RMSE: \033[92m{np.nanmean(rmse):.4f}\033[0m ± {np.nanstd(rmse):.4f} [{np.nanmin(rmse):.4f}, {np.nanmax(rmse):.4f}]")
        # print(f"R²: \033[92m{np.nanmean(r2):.3f}\033[0m ± {np.nanstd(r2):.3f} [{np.nanmin(r2):.3f}, {np.nanmax(r2):.3f}]")

    feve = compute_feve(true, pred, None)
    if np.all(np.isnan(feve)):
        print("\033[91mFEVE: NaN (model diverged)\033[0m")
    else:
        # print(f"FEVE: \033[92m{np.nanmean(feve):.3f}\033[0m ± {np.nanstd(feve):.3f} [{np.nanmin(feve):.3f}, {np.nanmax(feve):.3f}]")
        pass

    return rmse, pearson, feve, r2


def get_datavis_root_dir() -> str:
    """Location of downloaded DAVIS data.

    Set the DATAVIS_ROOT environment variable to the directory containing
    ``JPEGImages/480p/``.
    """

    datavis_root = os.environ.get("DATAVIS_ROOT", "")
    if not datavis_root or not os.path.exists(datavis_root):
        raise RuntimeError(
            "DAVIS data not found. Set the DATAVIS_ROOT environment variable "
            "to the directory containing JPEGImages/480p/. "
            "Example: export DATAVIS_ROOT=/path/to/DAVIS"
        )
    return datavis_root


def load_and_display(file_name, *, title=None, **kwargs):
    """Load and show an image file in the current matplotlib figure."""
    img = imageio.imread(file_name)
    plt.imshow(img, **kwargs)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
