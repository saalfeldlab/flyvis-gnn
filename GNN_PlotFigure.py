import os
import glob
import time
import logging
import warnings

import umap
# Fix umap / scikit-learn >=1.6 incompatibility (force_all_finite renamed to ensure_all_finite)
try:
    import sklearn.utils.validation as _skval
    _orig_check_array = _skval.check_array
    def _check_array_compat(*args, **kwargs):
        kwargs.pop('force_all_finite', None)
        return _orig_check_array(*args, **kwargs)
    _skval.check_array = _check_array_compat
    # Also patch the reference cached in umap's module namespace
    import umap.umap_ as _umap_mod
    if hasattr(_umap_mod, 'check_array'):
        _umap_mod.check_array = _check_array_compat
except Exception:
    pass
import torch
import numpy as np
import seaborn as sns
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD

from flyvis_gnn.figure_style import default_style as fig_style
from flyvis_gnn.zarr_io import load_simulation_data, load_raw_array
from flyvis_gnn.sparsify import clustering_gmm
from flyvis_gnn.models.flyvis_gnn import FlyVisGNN  # noqa: F401 — kept for backwards compat
from flyvis_gnn.models.registry import create_model
from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.metrics import (
    get_model_W,
    compute_r_squared,
    compute_r_squared_filtered,
    compute_all_corrected_weights,
    compute_activity_stats,
    extract_g_phi_slopes,
    extract_f_theta_slopes,
    derive_tau,
    derive_vrest,
    INDEX_TO_NAME,
    _vectorized_linspace,
    _batched_mlp_eval,
    _vectorized_linear_fit,
    _build_g_phi_features,
    _build_f_theta_features,
)
from flyvis_gnn.plot import _plot_curves_fast
from flyvis_gnn.utils import (
    to_numpy,
    CustomColorMap,
    sort_key,
    create_log_dir,
    add_pre_folder,
    graphs_data_path,
    log_path,
    config_path,
    migrate_state_dict,
)

# Optional imports
try:
    from flyvis_gnn.models.Ising_analysis import analyze_ising_model
except ImportError:
    analyze_ising_model = None

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

# Configure matplotlib for Helvetica-style fonts (no LaTeX)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',  # sans-serif math text
})


def get_training_files(log_dir, n_runs):
    files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_*.pt")
    if len(files) == 0:
        return [], np.array([])
    files.sort(key=sort_key)

    # Find the first file with positive sort_key
    file_id = 0
    while file_id < len(files) and sort_key(files[file_id]) <= 0:
        file_id += 1

    # If all files have non-positive sort_key, use all files
    if file_id >= len(files):
        file_id = 0

    files = files[file_id:]

    # Filter out files without the expected X_Y.pt suffix (e.g., "graphs_0.pt" has no Y)
    files = [f for f in files if f.split('_')[-2].isdigit()]

    if len(files) == 0:
        return [], np.array([])

    # Filter based on the Y value (number after "graphs")
    files_with_0 = [file for file in files if int(file.split('_')[-2]) == 0]
    files_without_0 = [file for file in files if int(file.split('_')[-2]) != 0]

    indices_with_0 = np.arange(0, len(files_with_0) - 1, dtype=int)
    indices_without_0 = np.linspace(0, len(files_without_0) - 1, 50, dtype=int)

    # Select the files using the generated indices
    selected_files_with_0 = [files_with_0[i] for i in indices_with_0]
    if len(files_without_0) > 0:
        selected_files_without_0 = [files_without_0[i] for i in indices_without_0]
        selected_files = selected_files_with_0 + selected_files_without_0
    else:
        selected_files = selected_files_with_0

    return selected_files, np.arange(0, len(selected_files), 1)


def _plot_synaptic_linear(model, config, config_indices, log_dir, logger, mc,
                          edges, gt_weights, gt_taus, gt_V_Rest,
                          type_list, n_types, n_neurons, cmap, device,
                          extended, log_file, mu_activity, sigma_activity):
    """Analysis plots for FlyVisLinear: W, tau, V_rest R² + clustering."""
    import torch.nn.functional as F
    sim = config.simulation

    # --- Parameter table ---
    w_params = get_model_W(model).numel()
    tau_params = model.raw_tau.numel()
    vrest_params = model.V_rest.numel()
    total_params = w_params + tau_params + vrest_params
    if hasattr(model, 's'):
        total_params += model.s.numel()
    print('learnable parameters:')
    print(f'  W (connectivity): {w_params:,}')
    print(f'  tau (time constant): {tau_params:,}')
    print(f'  V_rest (resting potential): {vrest_params:,}')
    print(f'  total: {total_params:,}')

    gt_taus_np = to_numpy(gt_taus[:n_neurons])
    gt_V_rest_np = to_numpy(gt_V_Rest[:n_neurons])
    gt_w_np = to_numpy(gt_weights)

    learned_tau = to_numpy(F.softplus(model.raw_tau[:n_neurons]).detach())
    learned_V_rest = to_numpy(model.V_rest[:n_neurons].detach())
    learned_weights = to_numpy(get_model_W(model).squeeze())

    # --- Plot 1: Loss curve ---
    if os.path.exists(os.path.join(log_dir, 'loss.pt')):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_alpha(0.75)
        list_loss = torch.load(os.path.join(log_dir, 'loss.pt'), weights_only=False)
        plt.plot(list_loss, color=mc, linewidth=2)
        plt.xlim([0, len(list_loss)])
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Training Loss')
        plt.tight_layout()
        plt.savefig(f'{log_dir}/results/loss.png', dpi=300)
        plt.close()

    # --- Plot 2: Raw W comparison ---
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(gt_w_np, learned_weights, c=mc, s=0.1, alpha=0.1)
    r_squared_W, slope_W = compute_r_squared(gt_w_np, learned_weights)
    plt.text(0.05, 0.95, f'R²: {r_squared_W:.3f}\nslope: {slope_W:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $W_{ij}$', fontsize=48)
    plt.ylabel(r'learned $W_{ij}$', fontsize=48)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/weights_comparison_raw.png', dpi=300)
    plt.close()
    print(f"weights R²: \033[92m{r_squared_W:.4f}\033[0m  slope: {slope_W:.4f}")
    logger.info(f"weights R²: {r_squared_W:.4f}  slope: {slope_W:.4f}")

    # --- Plot 3: tau comparison ---
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(gt_taus_np, learned_tau, c=mc, s=1, alpha=0.3)
    r_squared_tau, slope_tau = compute_r_squared(gt_taus_np, learned_tau)
    plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {slope_tau:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $\tau$', fontsize=48)
    plt.ylabel(r'learned $\tau$', fontsize=48)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
    plt.close()
    print(f"tau R²: \033[92m{r_squared_tau:.3f}\033[0m  slope: {slope_tau:.2f}")
    logger.info(f"tau R²: {r_squared_tau:.3f}  slope: {slope_tau:.2f}")

    # --- Plot 4: V_rest comparison ---
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(gt_V_rest_np, learned_V_rest, c=mc, s=1, alpha=0.3)
    r_squared_V_rest, slope_V_rest = compute_r_squared(gt_V_rest_np, learned_V_rest)
    plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {slope_V_rest:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $V_{rest}$', fontsize=48)
    plt.ylabel(r'learned $V_{rest}$', fontsize=48)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
    plt.close()
    print(f"V_rest R²: \033[92m{r_squared_V_rest:.3f}\033[0m  slope: {slope_V_rest:.2f}")
    logger.info(f"V_rest R²: {r_squared_V_rest:.3f}  slope: {slope_V_rest:.2f}")

    # --- Plot 5: tau and V_rest per neuron ---
    fig = plt.figure(figsize=(10, 9))
    ax = plt.subplot(2, 1, 1)
    plt.scatter(np.arange(n_neurons), learned_tau,
                c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
    plt.ylabel(r'$\tau_i$', fontsize=48)
    plt.xticks([])
    plt.yticks(fontsize=24)
    ax = plt.subplot(2, 1, 2)
    plt.scatter(np.arange(n_neurons), learned_V_rest,
                c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
    plt.xlabel('neuron index', fontsize=48)
    plt.ylabel(r'$V^{\mathrm{rest}}_i$', fontsize=48)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/results/dynamics_params_{config_indices}.png", dpi=300)
    plt.close()

    # --- Write R² to log file ---
    if log_file:
        log_file.write(f"connectivity_R2: {r_squared_W:.4f}\n")
        log_file.write(f"tau_R2: {r_squared_tau:.4f}\n")
        log_file.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")

    # --- Eigenvalue / SVD analysis ---
    print('plot eigenvalue spectrum and eigenvector comparison ...')
    edges_np = to_numpy(edges)
    true_sparse = scipy.sparse.csr_matrix(
        (gt_w_np.flatten(), (edges_np[1], edges_np[0])),
        shape=(n_neurons, n_neurons))
    learned_sparse = scipy.sparse.csr_matrix(
        (learned_weights.flatten(), (edges_np[1], edges_np[0])),
        shape=(n_neurons, n_neurons))

    n_components = min(100, n_neurons - 1)
    svd_true = TruncatedSVD(n_components=n_components, random_state=42)
    svd_learned = TruncatedSVD(n_components=n_components, random_state=42)
    svd_true.fit(true_sparse)
    svd_learned.fit(learned_sparse)
    sv_true = svd_true.singular_values_
    sv_learned = svd_learned.singular_values_

    n_eigs = min(200, n_neurons - 2)
    try:
        eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
        eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
    except Exception:
        n_eigs = min(50, n_neurons - 2)
        eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
        eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')

    V_true = svd_true.components_
    V_learned = svd_learned.components_
    alignment = np.abs(V_true @ V_learned.T)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    axes[0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
    axes[0].scatter(eig_learned.real, eig_learned.imag, s=100, c='black', alpha=0.7, label='learned')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('real', fontsize=32)
    axes[0].set_ylabel('imag', fontsize=32)
    axes[0].legend(fontsize=20)
    axes[0].tick_params(labelsize=20)
    axes[0].set_title('eigenvalues in complex plane', fontsize=28)

    n_compare = min(len(sv_true), len(sv_learned))
    axes[1].scatter(sv_true[:n_compare], sv_learned[:n_compare], s=100, c='black', edgecolors='black', alpha=0.7)
    max_val = max(sv_true.max(), sv_learned.max())
    axes[1].plot([0, max_val], [0, max_val], 'g--', linewidth=2)
    axes[1].set_xlabel('true singular value', fontsize=32)
    axes[1].set_ylabel('learned singular value', fontsize=32)
    axes[1].tick_params(labelsize=20)
    axes[1].set_title('singular value comparison', fontsize=28)

    axes[2].plot(sv_true, color='green', linewidth=2, label='true')
    axes[2].plot(sv_learned, color='black', linewidth=2, label='learned')
    axes[2].set_xlabel('index', fontsize=32)
    axes[2].set_ylabel('singular value', fontsize=32)
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=20)
    axes[2].tick_params(labelsize=20)
    axes[2].set_title('singular value spectrum (log scale)', fontsize=28)

    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/eigen_comparison.png', dpi=87)
    plt.close()

    true_spectral_radius = np.max(np.abs(eig_true))
    learned_spectral_radius = np.max(np.abs(eig_learned))
    print(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
    logger.info(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')

    # --- Clustering (no embeddings — use tau, V_rest, W stats) ---
    print('clustering learned features...')
    src, dst = edges_np[0], edges_np[1]

    def _connectivity_stats(w, src, dst, n):
        in_count = np.bincount(dst, minlength=n).astype(np.float64)
        out_count = np.bincount(src, minlength=n).astype(np.float64)
        in_sum = np.bincount(dst, weights=w, minlength=n)
        out_sum = np.bincount(src, weights=w, minlength=n)
        in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
        out_sq = np.bincount(src, weights=w ** 2, minlength=n)
        safe_in = np.where(in_count > 0, in_count, 1)
        safe_out = np.where(out_count > 0, out_count, 1)
        in_mean = in_sum / safe_in
        out_mean = out_sum / safe_out
        in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
        out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
        in_mean[in_count == 0] = 0
        out_mean[out_count == 0] = 0
        in_std[in_count == 0] = 0
        out_std[out_count == 0] = 0
        return in_mean, in_std, out_mean, out_std

    w_in_mean, w_in_std, w_out_mean, w_out_std = _connectivity_stats(
        learned_weights.flatten(), src, dst, n_neurons)
    W_learned = np.column_stack([w_in_mean, w_in_std, w_out_mean, w_out_std])

    w_in_mean_t, w_in_std_t, w_out_mean_t, w_out_std_t = _connectivity_stats(
        gt_w_np.flatten(), src, dst, n_neurons)
    W_true = np.column_stack([w_in_mean_t, w_in_std_t, w_out_mean_t, w_out_std_t])

    learned_combos = {
        'τ': learned_tau.reshape(-1, 1),
        'V': learned_V_rest.reshape(-1, 1),
        'W': W_learned,
        '(τ,V)': np.column_stack([learned_tau, learned_V_rest]),
        '(τ,V,W)': np.column_stack([learned_tau, learned_V_rest, W_learned]),
    }
    true_combos = {
        'τ': gt_taus_np.reshape(-1, 1),
        'V': gt_V_rest_np.reshape(-1, 1),
        'W': W_true,
        '(τ,V)': np.column_stack([gt_taus_np, gt_V_rest_np]),
        '(τ,V,W)': np.column_stack([gt_taus_np, gt_V_rest_np, W_true]),
    }

    learned_results = {}
    for name, feat in learned_combos.items():
        result = clustering_gmm(feat, type_list, n_components=75)
        learned_results[name] = result['accuracy']
        print(f"  learned {name}: {result['accuracy']:.3f}")
    true_results = {}
    for name, feat in true_combos.items():
        result = clustering_gmm(feat, type_list, n_components=75)
        true_results[name] = result['accuracy']
        print(f"  true {name}: {result['accuracy']:.3f}")

    # two-panel clustering bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    learned_order = list(learned_combos.keys())
    learned_vals = [learned_results[k] for k in learned_order]
    colors_l = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in learned_vals]
    ax1.barh(range(len(learned_order)), learned_vals, color=colors_l)
    ax1.set_yticks(range(len(learned_order)))
    ax1.set_yticklabels(learned_order, fontsize=11)
    ax1.set_xlabel('clustering accuracy', fontsize=12)
    ax1.set_title('learned features', fontsize=14)
    ax1.set_xlim([0, 1])
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    for i, v in enumerate(learned_vals):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)

    true_order = list(true_combos.keys())
    true_vals = [true_results[k] for k in true_order]
    colors_t = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in true_vals]
    ax2.barh(range(len(true_order)), true_vals, color=colors_t)
    ax2.set_yticks(range(len(true_order)))
    ax2.set_yticklabels(true_order, fontsize=11)
    ax2.set_xlabel('clustering accuracy', fontsize=12)
    ax2.set_title('true features', fontsize=14)
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    for i, v in enumerate(true_vals):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/clustering_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Augmented clustering: (tau, V_rest, W_stats) since no embeddings
    a_aug = np.column_stack([learned_tau, learned_V_rest, w_in_mean, w_in_std, w_out_mean, w_out_std])
    n_gmm = 100
    results = clustering_gmm(a_aug, type_list, n_components=n_gmm)
    cluster_acc = results['accuracy']
    print(f"GMM (n_components={n_gmm}): accuracy=\033[92m{cluster_acc:.3f}\033[0m, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")
    logger.info(f"GMM n_components={n_gmm}, accuracy={cluster_acc:.3f}, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")

    if log_file:
        log_file.write(f"cluster_accuracy: {cluster_acc:.4f}\n")

    # UMAP scatter
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    a_umap = reducer.fit_transform(a_aug)
    cluster_labels = GaussianMixture(n_components=n_gmm, random_state=42).fit_predict(a_aug)

    colors_65 = sns.color_palette("Set3", 12) * 6
    colors_65 = colors_65[:65]
    from matplotlib.colors import ListedColormap
    cmap_65 = ListedColormap(colors_65)

    plt.figure(figsize=(10, 9))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_alpha(0.75)
    plt.scatter(a_umap[:, 0], a_umap[:, 1], c=cluster_labels, s=24, cmap=cmap_65, alpha=0.8, edgecolors='none')
    plt.xlabel(r'UMAP$_1$', fontsize=48)
    plt.ylabel(r'UMAP$_2$', fontsize=48)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.text(0.05, 0.95, f"N: {n_neurons}\naccuracy: {cluster_acc:.2f}",
             transform=plt.gca().transAxes, fontsize=32, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/embedding_augmented_{config_indices}.png', dpi=300)
    plt.close()

    # Per-neuron type analysis
    analyze_neuron_type_reconstruction(
        config=config, model=model, edges=to_numpy(edges),
        true_weights=gt_w_np, gt_taus=gt_taus_np, gt_V_Rest=gt_V_rest_np,
        learned_weights=learned_weights, learned_tau=learned_tau,
        learned_V_rest=learned_V_rest, type_list=to_numpy(type_list),
        n_frames=sim.n_frames, dimension=sim.dimension,
        n_neuron_types=sim.n_neuron_types, device=device,
        log_dir=log_dir, dataset_name=config.dataset, logger=logger,
        index_to_name=INDEX_TO_NAME,
        r_squared=r_squared_W, slope_corrected=slope_W,
        r_squared_tau=r_squared_tau, r_squared_V_rest=r_squared_V_rest)


def plot_synaptic_flyvis(config, epoch_list, log_dir, logger, cc, style, extended, device, log_file=None):
    sim = config.simulation
    model_config = config.graph_model
    tc = config.training
    config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'evolution'


    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    config.simulation.max_radius if hasattr(config.simulation, 'max_radius') else 2.5

    results_log = os.path.join(log_dir, 'results.log')
    if os.path.exists(results_log):
        os.remove(results_log)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler only, no console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    file_handler = logging.FileHandler(results_log, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    # Load neuron group mapping for flyvis

    cmap = CustomColorMap(config=config)

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    time.sleep(0.5)
    print('\033[93mextracting parameters...\033[0m')
    x_path = graphs_data_path(config.dataset, 'x_list_train')
    if not os.path.exists(x_path):
        x_path = graphs_data_path(config.dataset, 'x_list_0')
    x_ts = load_simulation_data(x_path,
                                fields=['index', 'voltage', 'stimulus', 'neuron_type', 'group_type'])
    y_path = graphs_data_path(config.dataset, 'y_list_train')
    if not os.path.exists(y_path) and not os.path.exists(y_path + '.zarr'):
        y_path = graphs_data_path(config.dataset, 'y_list_0')
    y_data = load_raw_array(y_path)

    xnorm_path = os.path.join(log_dir, 'xnorm.pt')
    if os.path.exists(xnorm_path):
        xnorm = torch.load(xnorm_path, map_location=device, weights_only=False)
    else:
        xnorm = x_ts.xnorm.to(device)

    print(f'xnorm: {xnorm.item():0.3f}')
    logger.info(f'xnorm: {xnorm.item():0.3f}')

    type_list = x_ts.neuron_type.to(device)
    n_types = len(torch.unique(type_list))
    region_list = x_ts.group_type.to(device)
    n_region_types = len(torch.unique(region_list))
    n_neurons = x_ts.n_neurons

    ode_params = torch.load(graphs_data_path(config.dataset, 'ode_params.pt'), map_location=device, weights_only=False)
    gt_weights = ode_params['W']
    gt_taus = ode_params['tau_i']
    gt_V_Rest = ode_params['V_i_rest']
    edges = ode_params['edge_index']
    true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
    true_weights[edges[1], edges[0]] = gt_weights

    # Neuron type index to name mapping
    index_to_name = {
        0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
        10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
        18: 'Mi15', 19: 'Mi2', 20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
        27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
        36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1',
        44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
        52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14',
        59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
    }

    activity = x_ts.voltage.to(device).t()  # (N, T)
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)

    print(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    logger.info(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    os.makedirs(f'{log_dir}/results/', exist_ok=True)

    sorted_neuron_type_names = [index_to_name.get(i, f'Type{i}') for i in range(sim.n_neuron_types)]

    target_type_name_list = ['R1', 'R7', 'C2', 'Mi11', 'Tm1', 'Tm4', 'Tm30']
    activity_results = plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, sim.n_frames, sim.delta_t, f'{log_dir}/results/')
    plot_ground_truth_distributions(to_numpy(edges), to_numpy(gt_weights), to_numpy(gt_taus), to_numpy(gt_V_Rest), to_numpy(type_list), n_types, sorted_neuron_type_names, f'{log_dir}/results/')

    if ('Ising' in extended) | ('ising' in extended):
        analyze_ising_model(x_ts, sim.delta_t, log_dir, logger, to_numpy(edges))

    # Activity plots
    config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'evolution'
    neuron_types = to_numpy(type_list).astype(int).squeeze()

    # Get activity traces for all frames — voltage is (T, N), transpose to (N, T)
    activity_true = to_numpy(x_ts.voltage).T     # (n_neurons, sim.n_frames)
    visual_input_true = to_numpy(x_ts.stimulus).T  # (n_neurons, sim.n_frames)

    start_frame = 0

    # Create two figures: all types and selected types
    for fig_name, selected_types in [
        ("selected", [5, 15, 43, 39, 35, 31, 23, 19, 12, 55]),  # L1, Mi12, Tm1, T5a, T4a, T1, R1, Mi2, Mi1, Tm9
        ("all", np.arange(0, sim.n_neuron_types))
    ]:
        neuron_indices = []
        for stype in selected_types:
            indices = np.where(neuron_types == stype)[0]
            if len(indices) > 0:
                neuron_indices.append(indices[0])

        if len(neuron_indices) == 0:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        true_slice = activity_true[neuron_indices, start_frame:sim.n_frames]
        visual_input_slice = visual_input_true[neuron_indices, start_frame:sim.n_frames]
        step_v = 2.5
        lw = 1

        # Adjust fontsize based on number of neurons
        name_fontsize = 10 if len(selected_types) > 50 else 18

        for i in range(len(neuron_indices)):
            baseline = np.mean(true_slice[i])
            ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw, c='green', alpha=0.9,
                    label='activity' if i == 0 else None)
            # Plot visual input only for neuron_id = 0
            if (neuron_indices[i] == 0) and visual_input_slice[i].mean() > 0:
                ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=1, c='yellow', alpha=0.9,
                        linestyle='--', label='visual input')

        for i in range(len(neuron_indices)):
            type_idx = selected_types[i] if isinstance(selected_types, list) else selected_types[i]
            ax.text(-50, i * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color=mc)

        ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices)//50))])
        ax.set_yticks([])
        ax.set_xticks([0, 1000, 2000])
        ax.set_xticklabels([0, 1000, 2000], fontsize=16)
        ax.set_xlabel('frame', fontsize=20)
        ax.set_xlim([0, 2000])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(loc='upper right', fontsize=14)

        plt.tight_layout()
        if fig_name == "all":
            plt.savefig(f'{log_dir}/results/activity_{config_indices}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{log_dir}/results/activity_{config_indices}_selected.png', dpi=300, bbox_inches='tight')
        plt.close()

    if epoch_list[0] != 'all':
        config_indices = config.dataset.split('flyvis_')[1] if 'flyvis_' in config.dataset else 'evolution'
        files, file_id_list = get_training_files(log_dir, tc.n_runs)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'
            model = create_model(model_config.signal_model_name,
                                 aggr_type=model_config.aggr_type, config=config, device=device)
            state_dict = torch.load(net, map_location=device, weights_only=False)
            migrate_state_dict(state_dict)
            model.load_state_dict(state_dict['model_state_dict'])
            model.edges = edges

            logger.info(f'net: {net}')

            # --- Linear model branch ---
            if 'linear' in model_config.signal_model_name:
                _plot_synaptic_linear(
                    model, config, config_indices, log_dir, logger, mc,
                    edges, gt_weights, gt_taus, gt_V_Rest,
                    type_list, n_types, n_neurons, cmap, device,
                    extended, log_file, mu_activity, sigma_activity)
                continue

            # print learnable parameters table
            mlp0_params = sum(p.numel() for p in model.f_theta.parameters())
            mlp1_params = sum(p.numel() for p in model.g_phi.parameters())
            a_params = model.a.numel()
            w_params = get_model_W(model).numel()
            print('learnable parameters:')
            print(f'  MLP0 (f_theta): {mlp0_params:,}')
            print(f'  MLP1 (g_phi): {mlp1_params:,}')
            print(f'  a (embeddings): {a_params:,}')
            print(f'  W (connectivity): {w_params:,}')
            total_params = mlp0_params + mlp1_params + a_params + w_params
            if hasattr(model, 'NNR_f') and model.NNR_f is not None:
                nnr_f_params = sum(p.numel() for p in model.NNR_f.parameters())
                print(f'  INR (NNR_f): {nnr_f_params:,}')
                total_params += nnr_f_params
            print(f'  total: {total_params:,}')

            # Plot 1: Loss curve
            if os.path.exists(os.path.join(log_dir, 'loss.pt')):
                fig = plt.figure(figsize=(8, 6))
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_alpha(0.75)
                list_loss = torch.load(os.path.join(log_dir, 'loss.pt'), weights_only=False)
                plt.plot(list_loss, color=mc, linewidth=2)
                plt.xlim([0, len(list_loss)])
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.title('Training Loss')
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/loss.png', dpi=300)
                plt.close()

            # Plot 2: Embedding using model.a
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            for n in range(n_types):
                pos = torch.argwhere(type_list == n)
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=24, color=colors_65[n], alpha=0.8,
                            edgecolors='none')
            plt.xlabel(r'$a_{i0}$', fontsize=48)
            plt.ylabel(r'$a_{i1}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_{config_indices}.png', dpi=300)
            plt.close()

            n_pts = 1000
            post_fn = (lambda x: x ** 2) if model_config.g_phi_positive else None
            build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, model_config.signal_model_name)
            type_np = to_numpy(type_list).astype(int).ravel()

            # g_phi domain range: evaluate + slope extraction (vectorized)
            mu = to_numpy(mu_activity).astype(np.float32)
            sigma = to_numpy(sigma_activity).astype(np.float32)

            # Slope extraction uses positive domain (clamped to 0)
            valid_edge = (mu + sigma) > 0
            starts_edge_slope = np.maximum(mu - 2 * sigma, 0.0)
            ends_edge = mu + 2 * sigma
            starts_edge_slope[~valid_edge] = 0.0
            ends_edge[~valid_edge] = 1.0
            rr_domain_edge_slope = _vectorized_linspace(starts_edge_slope, ends_edge, n_pts, device)
            func_domain_edge_slope = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr_domain_edge_slope,
                                                 build_fn, device, post_fn=post_fn)
            slopes_edge, _ = _vectorized_linear_fit(rr_domain_edge_slope, func_domain_edge_slope)
            slopes_edge[~valid_edge] = 1.0
            slopes_g_phi_list = slopes_edge  # (N,) numpy array

            # Domain plot includes negative values to show g_phi → 0 for v < 0
            starts_edge_plot = mu - 2 * sigma
            starts_edge_plot[~valid_edge] = -0.5
            ends_edge_plot = ends_edge.copy()
            rr_domain_edge = _vectorized_linspace(starts_edge_plot, ends_edge_plot, n_pts, device)
            func_domain_edge = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr_domain_edge,
                                                 build_fn, device, post_fn=post_fn)

            rr_np = to_numpy(rr_domain_edge)
            func_np = to_numpy(func_domain_edge)

            # Ground truth g_phi: ReLU(v) — same for all neurons
            # The learned model plots g_phi(v)^2, and the true g_phi(v)^2 = ReLU(v)
            func_true_g_phi = np.maximum(rr_np, 0.0)

            # Side-by-side: true (left) vs learned (right)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
            _plot_curves_fast(ax1, rr_np[valid_edge], func_true_g_phi[valid_edge],
                              type_np[valid_edge], cmap, linewidth=1, alpha=0.1)
            ax1.set_xlabel('$v_j$', fontsize=48)
            ax1.set_ylabel(r'true $g_\phi(v_j)$', fontsize=48)
            ax1.tick_params(axis='both', which='major', labelsize=24)
            ax1.set_xlim([-1, 5])
            ax1.set_ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])

            _plot_curves_fast(ax2, rr_np[valid_edge], func_np[valid_edge],
                              type_np[valid_edge], cmap, linewidth=1, alpha=0.1)
            ax2.set_xlabel('$v_j$', fontsize=48)
            ax2.set_ylabel(r'learned $g_\phi(a_j, v_j)$', fontsize=48)
            ax2.tick_params(axis='both', which='major', labelsize=24)
            ax2.set_xlim([-1, 5])
            ax2.set_ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])

            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP1_{config_indices}_domain.png", dpi=300)
            plt.close()

            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            slopes_g_phi_array = np.array(slopes_g_phi_list)
            plt.scatter(np.arange(n_neurons), slopes_g_phi_array,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$r_j$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP1_slope_{config_indices}.png", dpi=300)
            plt.close()

            # f_theta domain range: evaluate + slope extraction (vectorized)
            starts_phi = mu - 2 * sigma
            ends_phi = mu + 2 * sigma
            rr_domain_phi = _vectorized_linspace(starts_phi, ends_phi, n_pts, device)
            func_domain_phi = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr_domain_phi,
                                                lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f), device)
            slopes_phi, offsets_phi = _vectorized_linear_fit(rr_domain_phi, func_domain_phi)
            slopes_f_theta_list = slopes_phi  # (N,) numpy array
            offsets_list = offsets_phi

            # Ground truth f_theta: f_true(v) = (-v + V_rest) / tau per neuron
            gt_taus_np = to_numpy(gt_taus[:n_neurons])
            gt_V_rest_np = to_numpy(gt_V_Rest[:n_neurons])
            rr_domain_phi_np = to_numpy(rr_domain_phi)
            func_true_f_theta = (-rr_domain_phi_np + gt_V_rest_np[:, None]) / gt_taus_np[:, None]

            # Side-by-side: true (left) vs learned (right)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
            _plot_curves_fast(ax1, rr_domain_phi_np, func_true_f_theta,
                              type_np, cmap, linewidth=1, alpha=0.1)
            ax1.set_xlim(config.plotting.xlim)
            ax1.set_ylim(config.plotting.ylim)
            ax1.set_xlabel('$v_i$', fontsize=48)
            ax1.set_ylabel(r'true $f_\theta(v_i)$', fontsize=48)
            ax1.tick_params(axis='both', which='major', labelsize=24)

            _plot_curves_fast(ax2, to_numpy(rr_domain_phi), to_numpy(func_domain_phi),
                              type_np, cmap, linewidth=1, alpha=0.1)
            ax2.set_xlim(config.plotting.xlim)
            ax2.set_ylim(config.plotting.ylim)
            ax2.set_xlabel('$v_i$', fontsize=48)
            ax2.set_ylabel(r'learned $f_\theta(a_i, v_i)$', fontsize=48)
            ax2.tick_params(axis='both', which='major', labelsize=24)

            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP0_{config_indices}_domain.png", dpi=300)
            plt.close()

            slopes_f_theta_array = np.array(slopes_f_theta_list)
            offsets_array = np.array(offsets_list)
            gt_taus = to_numpy(gt_taus[:n_neurons])
            learned_tau = derive_tau(slopes_f_theta_array, n_neurons)

            fig = plt.figure(figsize=(10, 9))
            plt.scatter(gt_taus, learned_tau, c=mc, s=1, alpha=0.3)
            r_squared_tau, slope_tau = compute_r_squared(gt_taus, learned_tau)
            plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {slope_tau:.2f}\nN: {sim.n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $\tau$', fontsize=48)
            plt.ylabel(r'learned $\tau$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([0, 0.35])
            plt.ylim([0, 0.35])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
            plt.close()


            # V_rest comparison (reconstructed vs ground truth)
            learned_V_rest = derive_vrest(slopes_f_theta_array, offsets_array, n_neurons)
            gt_V_rest = to_numpy(gt_V_Rest[:n_neurons])
            fig = plt.figure(figsize=(10, 9))
            plt.scatter(gt_V_rest, learned_V_rest, c=mc, s=1, alpha=0.3)
            r_squared_V_rest, slope_V_rest = compute_r_squared(gt_V_rest, learned_V_rest)
            plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {slope_V_rest:.2f}\nN: {sim.n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
            plt.xlabel(r'true $V_{rest}$', fontsize=48)
            plt.ylabel(r'learned $V_{rest}$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlim([0, 0.8])
            plt.ylim([0, 0.8])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
            plt.close()

            fig = plt.figure(figsize=(10, 9))
            ax = plt.subplot(2, 1, 1)
            plt.scatter(np.arange(n_neurons), learned_tau,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.ylabel(r'$\tau_i$', fontsize=48)
            plt.xticks([])   # no xticks for top plot
            plt.yticks(fontsize=24)
            ax = plt.subplot(2, 1, 2)
            plt.scatter(np.arange(n_neurons), learned_V_rest,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$V^{\mathrm{rest}}_i$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/MLP0_{config_indices}_params.png", dpi=300)
            plt.close()


            # Plot 4: Weight comparison using model.W and gt_weights
            # Check Dale's Law for learned weights
            # dale_results = check_dales_law(
            #     edges=edges,
            #     weights=model.W,
            #     type_list=type_list,
            #     n_neurons=n_neurons,
            #     verbose=False,
            #     logger=None
            # )

            fig = plt.figure(figsize=(10, 9))
            learned_weights = to_numpy(get_model_W(model).squeeze())
            true_weights = to_numpy(gt_weights)
            plt.scatter(true_weights, learned_weights, c=mc, s=0.1, alpha=0.1)
            r_squared, slope_raw = compute_r_squared(true_weights, learned_weights)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {slope_raw:.2f}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)

            # Add Dale's Law statistics
            # dale_text = (f"excitatory neurons (all W>0): {dale_results['n_excitatory']} "
            #              f"({100*dale_results['n_excitatory']/n_neurons:.1f}%)\n"
            #              f"inhibitory neurons (all W<0): {dale_results['n_inhibitory']} "
            #              f"({100*dale_results['n_inhibitory']/n_neurons:.1f}%)\n"
            #              f"mixed/zero neurons (violates Dale's Law): {dale_results['n_mixed']} "
            #              f"({100*dale_results['n_mixed']/n_neurons:.1f}%)")
            # plt.text(0.05, 0.05, dale_text, transform=plt.gca().transAxes,
            #          verticalalignment='bottom', fontsize=10)

            plt.xlabel(r'true $W_{ij}$', fontsize=48)
            plt.ylabel(r'learned $W_{ij}$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/weights_comparison_raw.png', dpi=300)
            plt.close()
            print(f"first weights fit R²: {r_squared:.2f}  slope: {np.round(slope_raw, 4)}")
            logger.info(f"first weights fit R²: {r_squared:.2f}  slope: {np.round(slope_raw, 4)}")

            # Corrected weights via metrics pipeline (replaces inline DataLoader +
            # gradient computation + correction formula — see metrics.py)
            corrected_W, ret_slopes_f, ret_slopes_g, ret_offsets = compute_all_corrected_weights(
                model, config, edges, x_ts, device, n_grad_frames=8)
            torch.save(corrected_W, f'{log_dir}/results/corrected_W.pt')

            learned_weights = to_numpy(corrected_W.squeeze())
            true_weights = to_numpy(gt_weights)

            # Outlier removal + R² via metrics
            r_squared, slope_corrected, mask = compute_r_squared_filtered(
                true_weights, learned_weights, outlier_threshold=5.0)
            residuals = learned_weights - true_weights
            true_in = true_weights[mask]
            learned_in = learned_weights[mask]

            if extended:
                # Partial correction (without g_phi factor) for diagnostic plot
                n_w = model.n_edges + model.n_extra_null_edges
                prior_ids = edges[0, :] % n_w
                slopes_g_t = torch.tensor(ret_slopes_g, dtype=torch.float32, device=device)
                corrected_W_ = corrected_W / slopes_g_t[prior_ids].unsqueeze(1)
                corrected_W_ = torch.nan_to_num(corrected_W_, nan=0.0, posinf=0.0, neginf=0.0)

                learned_in_ = to_numpy(corrected_W_.squeeze())
                learned_in_ = learned_in_[mask]

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(true_in, learned_in_, c=mc, s=0.1, alpha=0.1)
                r_squared_rj, slope_rj = compute_r_squared(true_in, learned_in_)
                plt.text(0.05, 0.95,
                        f'R²: {r_squared_rj:.3f}\nslope: {slope_rj:.2f}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)
                plt.xlabel(r'true $W_{ij}$', fontsize=48)
                plt.ylabel(r'learned $W_{ij}r_j$', fontsize=48)
                plt.xticks(fontsize = 24)
                plt.yticks(fontsize = 24)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/weights_comparison_rj.png', dpi=300)
                plt.close()

            fig = plt.figure(figsize=(10, 9))
            plt.scatter(true_in, learned_in, c=mc, s=0.5, alpha=0.06)
            plt.text(0.05, 0.95,
                     f'R²: {r_squared:.2f}\nslope: {slope_corrected:.2f}\nN: {sim.n_edges}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)

            plt.xlabel(r'true $W_{ij}$', fontsize=48)
            plt.ylabel(r'learned $W_{ij}^*$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.xlim([-1,2])
            plt.ylim([-1,2])
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/weights_comparison_corrected.png', dpi=300)
            plt.close()

            print(f"second weights fit R²: \033[92m{r_squared:.4f}\033[0m  slope: {np.round(slope_corrected, 4)}")
            logger.info(f"second weights fit R²: {r_squared:.4f}  slope: {np.round(slope_corrected, 4)}")

            # R² on only real (non-null) edges
            connectivity_r2_real = None
            if hasattr(model, 'n_extra_null_edges') and model.n_extra_null_edges > 0:
                n_real = model.n_edges
                try:
                    r2_real, _ = compute_r_squared(true_weights[:n_real], learned_weights[:n_real])
                    connectivity_r2_real = r2_real
                    print(f"connectivity R² (real edges only): \033[92m{r2_real:.4f}\033[0m")
                    logger.info(f"connectivity R² (real edges only): {r2_real:.4f}")
                except Exception:
                    pass
            print(f'median residuals: {np.median(residuals):.4f}')
            inlier_residuals = residuals[mask]
            print(f'inliers: {len(inlier_residuals)}  mean residual: {np.mean(inlier_residuals):.4f}  std: {np.std(inlier_residuals):.4f}  min,max: {np.min(inlier_residuals):.4f}, {np.max(inlier_residuals):.4f}')
            outlier_residuals = residuals[~mask]
            if len(outlier_residuals) > 0:
                print(
                    f'outliers: {len(outlier_residuals)}  mean residual: {np.mean(outlier_residuals):.4f}  std: {np.std(outlier_residuals):.4f}  min,max: {np.min(outlier_residuals):.4f}, {np.max(outlier_residuals):.4f}')
            else:
                print('outliers: 0  (no outliers detected)')
            print(f"tau reconstruction R²: \033[92m{r_squared_tau:.3f}\033[0m  slope: {slope_tau:.2f}")
            logger.info(f"tau reconstruction R²: {r_squared_tau:.3f}  slope: {slope_tau:.2f}")
            print(f"V_rest reconstruction R²: \033[92m{r_squared_V_rest:.3f}\033[0m  slope: {slope_V_rest:.2f}")
            logger.info(f"V_rest reconstruction R²: {r_squared_V_rest:.3f}  slope: {slope_V_rest:.2f}")

            # Write to analysis log file for Claude
            if log_file:
                log_file.write(f"connectivity_R2: {r_squared:.4f}\n")
                if connectivity_r2_real is not None:
                    log_file.write(f"connectivity_R2_real: {connectivity_r2_real:.4f}\n")
                log_file.write(f"tau_R2: {r_squared_tau:.4f}\n")
                log_file.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")


            # Plot connectivity matrix comparison (skipped — dense NxN heatmaps too slow)
            # eigenvalue and singular value analysis using sparse matrices
            print('plot eigenvalue spectrum and eigenvector comparison ...')

            # build sparse matrices for true and learned weights
            edges_np = to_numpy(edges)
            true_sparse = scipy.sparse.csr_matrix(
                (true_weights.flatten(), (edges_np[1], edges_np[0])),
                shape=(n_neurons, n_neurons)
            )
            learned_sparse = scipy.sparse.csr_matrix(
                (to_numpy(corrected_W.squeeze().flatten()), (edges_np[1], edges_np[0])),
                shape=(n_neurons, n_neurons)
            )

            # compute SVD using TruncatedSVD (for large sparse matrices)
            # 100 components captures dominant structure; 1000 was very slow for N>10000
            n_components = min(100, n_neurons - 1)
            svd_true = TruncatedSVD(n_components=n_components, random_state=42)
            svd_learned = TruncatedSVD(n_components=n_components, random_state=42)

            svd_true.fit(true_sparse)
            svd_learned.fit(learned_sparse)

            sv_true = svd_true.singular_values_
            sv_learned = svd_learned.singular_values_

            # get right singular vectors (V^T rows)
            V_true = svd_true.components_
            V_learned = svd_learned.components_

            # compute alignment matrix
            alignment = np.abs(V_true @ V_learned.T)
            best_alignment = np.max(alignment, axis=1)

            # compute eigenvalues using sparse eigensolver for complex plane plot
            # 200 largest-magnitude eigenvalues captures spectral structure;
            # 500 was very slow for N>10000 (ARPACK scales poorly with k)
            n_eigs = min(200, n_neurons - 2)
            try:
                eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
                eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
            except Exception:
                # fallback: use smaller k if convergence issues
                n_eigs = min(50, n_neurons - 2)
                eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
                eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')

            # create 2x3 figure
            fig, axes = plt.subplots(2, 3, figsize=(30, 20))

            # Row 1: Eigenvalues/Singular values
            # (0,0) eigenvalues in complex plane
            axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
            axes[0, 0].scatter(eig_learned.real, eig_learned.imag, s=100, c='black', alpha=0.7, label='learned')
            axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            axes[0, 0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            axes[0, 0].set_xlabel('real', fontsize=32)
            axes[0, 0].set_ylabel('imag', fontsize=32)
            axes[0, 0].legend(fontsize=20)
            axes[0, 0].tick_params(labelsize=20)
            axes[0, 0].set_title('eigenvalues in complex plane', fontsize=28)

            # (0,1) singular value magnitude comparison (scatter)
            n_compare = min(len(sv_true), len(sv_learned))
            axes[0, 1].scatter(sv_true[:n_compare], sv_learned[:n_compare], s=100, c='black', edgecolors='black', alpha=0.7)
            max_val = max(sv_true.max(), sv_learned.max())
            axes[0, 1].plot([0, max_val], [0, max_val], 'g--', linewidth=2)
            axes[0, 1].set_xlabel('true singular value', fontsize=32)
            axes[0, 1].set_ylabel('learned singular value', fontsize=32)
            axes[0, 1].tick_params(labelsize=20)
            axes[0, 1].set_title('singular value comparison', fontsize=28)

            # (0,2) singular value spectrum (log scale)
            axes[0, 2].plot(sv_true, color='green', linewidth=2, label='true')
            axes[0, 2].plot(sv_learned, color='black', linewidth=2, label='learned')
            axes[0, 2].set_xlabel('index', fontsize=32)
            axes[0, 2].set_ylabel('singular value', fontsize=32)
            axes[0, 2].set_yscale('log')
            axes[0, 2].legend(fontsize=20)
            axes[0, 2].tick_params(labelsize=20)
            axes[0, 2].set_title('singular value spectrum (log scale)', fontsize=28)

            # Row 2: Singular vectors
            # (1,0) right singular vector alignment matrix
            n_show = min(100, n_components)
            im = axes[1, 0].imshow(alignment[:n_show, :n_show], cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_xlabel('learned eigenvector index', fontsize=28)
            axes[1, 0].set_ylabel('true eigenvector index', fontsize=28)
            axes[1, 0].set_title('right eigenvector alignment', fontsize=28)
            axes[1, 0].tick_params(labelsize=16)
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

            # (1,1) left eigenvector alignment (placeholder - SVD doesn't give left eigenvectors directly)
            # For consistency with plot_signal, compute left singular vectors alignment
            U_true = svd_true.transform(true_sparse)[:, :n_show]
            U_learned = svd_learned.transform(learned_sparse)[:, :n_show]
            # Normalize columns
            U_true = U_true / (np.linalg.norm(U_true, axis=0, keepdims=True) + 1e-10)
            U_learned = U_learned / (np.linalg.norm(U_learned, axis=0, keepdims=True) + 1e-10)
            alignment_L = np.abs(U_true.T @ U_learned)
            best_alignment_L = np.max(alignment_L, axis=1)
            im_L = axes[1, 1].imshow(alignment_L, cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_xlabel('learned eigenvector index', fontsize=28)
            axes[1, 1].set_ylabel('true eigenvector index', fontsize=28)
            axes[1, 1].set_title('left eigenvector alignment', fontsize=28)
            axes[1, 1].tick_params(labelsize=16)
            plt.colorbar(im_L, ax=axes[1, 1], fraction=0.046)

            # (1,2) best alignment scores
            best_alignment_R = np.max(alignment[:n_show, :n_show], axis=1)
            axes[1, 2].scatter(range(len(best_alignment_R)), best_alignment_R, s=50, c='green', alpha=0.7, label=f'right (mean={np.mean(best_alignment_R):.2f})')
            axes[1, 2].scatter(range(len(best_alignment_L)), best_alignment_L, s=50, c='black', alpha=0.7, label=f'left (mean={np.mean(best_alignment_L):.2f})')
            axes[1, 2].axhline(y=1/np.sqrt(n_show), color='gray', linestyle='--', linewidth=2, label=f'random ({1/np.sqrt(n_show):.2f})')
            axes[1, 2].set_xlabel('eigenvector index (sorted by singular value)', fontsize=28)
            axes[1, 2].set_ylabel('best alignment score', fontsize=28)
            axes[1, 2].set_title('best alignment per eigenvector', fontsize=28)
            axes[1, 2].set_ylim([0, 1.05])
            axes[1, 2].legend(fontsize=20)
            axes[1, 2].tick_params(labelsize=16)

            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/eigen_comparison.png', dpi=87)
            plt.close()

            # print spectral analysis results (consistent with plot_signal)
            true_spectral_radius = np.max(np.abs(eig_true))
            learned_spectral_radius = np.max(np.abs(eig_learned))
            print(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
            logger.info(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
            print(f'eigenvector alignment - right: {np.mean(best_alignment_R):.3f}  left: {np.mean(best_alignment_L):.3f}')
            logger.info(f'eigenvector alignment - right: {np.mean(best_alignment_R):.3f}  left: {np.mean(best_alignment_L):.3f}')


            # plot analyze_neuron_type_reconstruction
            results_per_neuron = analyze_neuron_type_reconstruction(
                config=config,
                model=model,
                edges=to_numpy(edges),
                true_weights=true_weights,  #  ground truth weights
                gt_taus=gt_taus,  #  ground truth tau values
                gt_V_Rest=gt_V_rest,  #  ground truth V_rest values
                learned_weights=learned_weights,
                learned_tau = learned_tau,
                learned_V_rest=learned_V_rest, # Learned V_rest
                type_list=to_numpy(type_list),
                n_frames=sim.n_frames,
                dimension=sim.dimension,
                n_neuron_types=sim.n_neuron_types,
                device=device,
                log_dir=log_dir,
                dataset_name=config.dataset,
                logger=logger,
                index_to_name=index_to_name,
                r_squared=r_squared,
                slope_corrected=slope_corrected,
                r_squared_tau=r_squared_tau,
                r_squared_V_rest=r_squared_V_rest
            )

            print('alternative clustering methods...')


            # compute connectivity statistics (vectorized via bincount)
            print('computing connectivity statistics...')
            edges_np = to_numpy(edges)
            src, dst = edges_np[0], edges_np[1]

            def _connectivity_stats(w, src, dst, n):
                """Per-neuron mean/std of in-weights and out-weights."""
                # counts
                in_count = np.bincount(dst, minlength=n).astype(np.float64)
                out_count = np.bincount(src, minlength=n).astype(np.float64)
                # sums
                in_sum = np.bincount(dst, weights=w, minlength=n)
                out_sum = np.bincount(src, weights=w, minlength=n)
                # sum of squares
                in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
                out_sq = np.bincount(src, weights=w ** 2, minlength=n)
                # mean (0 where no edges)
                safe_in = np.where(in_count > 0, in_count, 1)
                safe_out = np.where(out_count > 0, out_count, 1)
                in_mean = in_sum / safe_in
                out_mean = out_sum / safe_out
                # std = sqrt(E[x^2] - E[x]^2), clamped to avoid negative from fp noise
                in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
                out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
                # zero out neurons with no edges
                in_mean[in_count == 0] = 0
                out_mean[out_count == 0] = 0
                in_std[in_count == 0] = 0
                out_std[out_count == 0] = 0
                return in_mean, in_std, out_mean, out_std

            w_in_mean_true, w_in_std_true, w_out_mean_true, w_out_std_true = \
                _connectivity_stats(true_weights.flatten(), src, dst, n_neurons)
            w_in_mean_learned, w_in_std_learned, w_out_mean_learned, w_out_std_learned = \
                _connectivity_stats(learned_weights.flatten(), src, dst, n_neurons)

            # all 4 connectivity stats combined
            W_learned = np.column_stack([w_in_mean_learned, w_in_std_learned,
                                        w_out_mean_learned, w_out_std_learned])
            W_true = np.column_stack([w_in_mean_true, w_in_std_true,
                                    w_out_mean_true, w_out_std_true])

            # learned combinations
            learned_combos = {
                'a': to_numpy(model.a),
                'τ': learned_tau.reshape(-1, 1),
                'V': learned_V_rest.reshape(-1, 1),
                'W': W_learned,
                '(τ,V)': np.column_stack([learned_tau, learned_V_rest]),
                '(τ,V,W)': np.column_stack([learned_tau, learned_V_rest, W_learned]),
                '(a,τ,V,W)': np.column_stack([to_numpy(model.a), learned_tau, learned_V_rest, W_learned]),
            }

            # true combinations
            true_combos = {
                'τ': gt_taus.reshape(-1, 1),
                'V': gt_V_rest.reshape(-1, 1),
                'W': W_true,
                '(τ,V)': np.column_stack([gt_taus, gt_V_rest]),
                '(τ,V,W)': np.column_stack([gt_taus, gt_V_rest, W_true]),
            }

            # cluster learned
            print('clustering learned features...')
            learned_results = {}
            for name, feat_array in learned_combos.items():
                result = clustering_gmm(feat_array, type_list, n_components=75)
                learned_results[name] = result['accuracy']
                print(f"{name}: {result['accuracy']:.3f}")

            # Cluster true
            print('clustering true features...')
            true_results = {}
            for name, feat_array in true_combos.items():
                result = clustering_gmm(feat_array, type_list, n_components=75)
                true_results[name] = result['accuracy']
                print(f"{name}: {result['accuracy']:.3f}")

            # Plot two-panel figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            # Learned features - fixed order
            learned_order = ['a', 'τ', 'V', 'W', '(τ,V)', '(τ,V,W)', '(a,τ,V,W)']
            learned_vals = [learned_results[k] for k in ['a', 'τ', 'V', 'W', '(τ,V)', '(τ,V,W)', '(a,τ,V,W)']]
            colors_l = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in learned_vals]
            ax1.barh(range(len(learned_order)), learned_vals, color=colors_l)
            ax1.set_yticks(range(len(learned_order)))
            ax1.set_yticklabels(learned_order, fontsize=11)
            ax1.set_xlabel('clustering accuracy', fontsize=12)
            ax1.set_title('learned features', fontsize=14)
            ax1.set_xlim([0, 1])
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            for i, v in enumerate(learned_vals):
                ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
            # True features - fixed order
            true_order = ['τ', 'V', 'W', '(τ,V)', '(τ,V,W)']
            true_vals = [true_results[k] for k in ['τ', 'V', 'W', '(τ,V)', '(τ,V,W)']]
            colors_t = ['#d62728' if v < 0.6 else '#ff7f0e' if v < 0.85 else '#2ca02c' for v in true_vals]
            ax2.barh(range(len(true_order)), true_vals, color=colors_t)
            ax2.set_yticks(range(len(true_order)))
            ax2.set_yticklabels(true_order, fontsize=11)
            ax2.set_xlabel('clustering accuracy', fontsize=12)
            ax2.set_title('true features', fontsize=14)
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            for i, v in enumerate(true_vals):
                ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/clustering_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()

            a_aug = np.column_stack([to_numpy(model.a), learned_tau, learned_V_rest,
                                    w_in_mean_learned, w_in_std_learned, w_out_mean_learned, w_out_std_learned])

            n_gmm = 100
            results = clustering_gmm(a_aug, type_list, n_components=n_gmm)
            cluster_acc = results['accuracy']
            print(f"GMM (n_components={n_gmm}): accuracy=\033[92m{cluster_acc:.3f}\033[0m, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")
            logger.info(f"GMM n_components={n_gmm}, accuracy={cluster_acc:.3f}, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")

            # Write cluster accuracy to analysis log file for Claude
            if log_file:
                log_file.write(f"cluster_accuracy: {cluster_acc:.4f}\n")

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            a_umap = reducer.fit_transform(a_aug)

            # Get cluster labels from GMM
            results = clustering_gmm(a_aug, type_list, n_components=n_gmm)
            cluster_labels = GaussianMixture(n_components=n_gmm, random_state=42).fit_predict(a_aug)

            plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            from matplotlib.colors import ListedColormap
            cmap_65 = ListedColormap(colors_65)
            plt.scatter(a_umap[:, 0], a_umap[:, 1], c=cluster_labels, s=24, cmap=cmap_65, alpha=0.8, edgecolors='none')


            plt.xlabel(r'UMAP$_1$', fontsize=48)
            plt.ylabel(r'UMAP$_2$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.text(0.05, 0.95, f"N: {n_neurons}\naccuracy: {cluster_acc:.2f}",
                    transform=plt.gca().transAxes, fontsize=32, verticalalignment='top')
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_augmented_{config_indices}.png', dpi=300)
            plt.close()

    # ---- Activity traces: clean vs noisy (measurement noise) ----
    if getattr(sim, 'measurement_noise_level', 0) > 0:
        try:
            import zarr as _zarr
            data_dir = graphs_data_path(config.dataset, 'x_list_train')
            voltage_path = os.path.join(data_dir, 'voltage.zarr')
            noise_path = os.path.join(data_dir, 'noise.zarr')
            if os.path.isdir(voltage_path) and os.path.isdir(noise_path):
                _n_traces = 20
                _frame_start, _frame_end = 5000, 5500
                _rng = np.random.RandomState(42)
                _voltage = _zarr.open(voltage_path, 'r')[_frame_start:_frame_end, :]
                _noise = _zarr.open(noise_path, 'r')[_frame_start:_frame_end, :]
                _n_neurons_total = _voltage.shape[1]
                _indices = np.sort(_rng.choice(_n_neurons_total, _n_traces, replace=False))
                _clean = _voltage[:, _indices].T
                _noisy = (_voltage[:, _indices] + _noise[:, _indices]).T
                _trace_range = np.median(np.ptp(_clean, axis=1))
                _spacing = _trace_range * 1.8
                _offsets = _spacing * np.arange(_n_traces)[:, None]
                _clean_off = _clean + _offsets
                _noisy_off = _noisy + _offsets
                _sigma = sim.measurement_noise_level
                _xvals = np.arange(_frame_start, _frame_end)

                # SNR for voltage: var(signal) / var(noise)
                _signal_var = np.var(_clean)
                _noise_var = np.var(_noise[:, _indices].T)
                _snr_v = _signal_var / _noise_var if _noise_var > 0 else float('inf')
                _snr_v_db = 10 * np.log10(_snr_v) if np.isfinite(_snr_v) else float('inf')

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='white', sharey=True)
                for ax, data, subtitle in [(ax1, _clean_off, 'without measurement noise'),
                                            (ax2, _noisy_off, 'with measurement noise')]:
                    ax.set_facecolor('white')
                    ax.plot(_xvals, data.T, linewidth=0.6, alpha=0.85, color='#333333')
                    ax.set_xlim([_frame_start, _frame_end])
                    ax.set_ylim([data[0].min() - _spacing, data[-1].max() + _spacing])
                    ax.set_yticks([])
                    ax.tick_params(axis='x', labelsize=9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlabel('time (frames)', fontsize=14)
                    ax.set_title(subtitle, fontsize=18, pad=12)
                ax1.set_ylabel(f'{_n_traces} / {_n_neurons_total} neurons', fontsize=14)
                ax2.text(0.97, 0.90, f'SNR(V) = {_snr_v_db:.1f} dB',
                         transform=ax2.transAxes, fontsize=18,
                         verticalalignment='top', horizontalalignment='right')
                fig.subplots_adjust(wspace=0.05)
                _out = graphs_data_path(config.dataset, 'activity_traces_noisy.png')
                plt.savefig(_out, dpi=300, facecolor='white', bbox_inches='tight')
                plt.close()
                logger.info(f'saved activity_traces_noisy.png')
        except Exception as _e:
            logger.warning(f'could not generate activity_traces_noisy: {_e}')


def analyze_neuron_type_reconstruction(config, model, edges, true_weights, gt_taus, gt_V_Rest,
                                       learned_weights, learned_tau, learned_V_rest, type_list, n_frames, dimension,
                                       n_neuron_types, device, log_dir, dataset_name, logger, index_to_name,
                                       r_squared=None, slope_corrected=None, r_squared_tau=None, r_squared_V_rest=None):

    print('stratified analysis by neuron type...')

    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    rmse_weights = []
    rmse_taus = []
    rmse_vrests = []
    n_connections = []

    for neuron_type in range(n_neuron_types):

        type_indices = np.where(type_list[edges[1,:]] == neuron_type)[0]
        gt_w_type = true_weights[type_indices]
        learned_w_type = learned_weights[type_indices]
        n_conn = len(type_indices)

        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        gt_vrest_type = gt_V_Rest[type_indices]

        learned_tau_type = learned_tau[type_indices]
        learned_vrest_type = learned_V_rest[type_indices]

        rmse_w = np.sqrt(np.mean((gt_w_type - learned_w_type)** 2))
        rmse_tau = np.sqrt(np.mean((gt_tau_type - learned_tau_type)** 2))
        rmse_vrest = np.sqrt(np.mean((gt_vrest_type - learned_vrest_type)** 2))

        rmse_weights.append(rmse_w)
        rmse_taus.append(rmse_tau)
        rmse_vrests.append(rmse_vrest)
        n_connections.append(n_conn)

    n_neurons = len(type_list)

    # Per-neuron RMSE for tau
    rmse_tau_per_neuron = np.abs(learned_tau - gt_taus)
    # Per-neuron RMSE for V_rest
    rmse_vrest_per_neuron = np.abs(learned_V_rest - gt_V_Rest)
    # Per-neuron RMSE for weights (incoming connections)
    rmse_weights_per_neuron = np.zeros(n_neurons)
    for neuron_idx in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == neuron_idx)[0]
        if len(incoming_edges) > 0:
            true_w = true_weights[incoming_edges]
            learned_w = learned_weights[incoming_edges]
            rmse_weights_per_neuron[neuron_idx] = np.sqrt(np.mean((learned_w - true_w)**2))

    # Convert to arrays
    rmse_weights = np.array(rmse_weights)
    rmse_taus = np.array(rmse_taus)
    rmse_vrests = np.array(rmse_vrests)

    unique_types_in_order = []
    seen_types = set()
    for i in range(len(type_list)):
        neuron_type_id = type_list[i].item() if hasattr(type_list[i], 'item') else int(type_list[i])
        if neuron_type_id not in seen_types:
            unique_types_in_order.append(neuron_type_id)
            seen_types.add(neuron_type_id)

    # Create neuron type names in the same order as they appear in data
    sorted_neuron_type_names = [index_to_name.get(type_id, f'Type{type_id}') for type_id in unique_types_in_order]
    unique_types_in_order = np.array(unique_types_in_order)
    sort_indices = unique_types_in_order.astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    x_pos = np.arange(len(sort_indices))

    # Plot weights RMSE
    ax1 = axes[0]
    ax1.bar(x_pos, rmse_weights[sort_indices], color='skyblue', alpha=0.7)
    ax1.set_ylabel('RMSE weights', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 2.5])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax1.grid(False)
    ax1.tick_params(axis='y', labelsize=12)

    for i, (tick, rmse_w) in enumerate(zip(ax1.get_xticklabels(), rmse_weights[sort_indices])):
        if rmse_w > 0.5:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 2 (tau)
    ax2 = axes[1]
    ax2.bar(x_pos, rmse_taus[sort_indices], color='lightcoral', alpha=0.7)
    ax2.set_ylabel(r'RMSE $\tau$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.3])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth taus per neuron type
    mean_gt_taus = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_tau_type = gt_taus[type_indices]
        mean_gt_taus.append(np.mean(np.abs(gt_tau_type)))

    mean_gt_taus = np.array(mean_gt_taus)

    for i, (tick, rmse_tau) in enumerate(zip(ax2.get_xticklabels(), rmse_taus[sort_indices])):
        if rmse_tau > 0.03:
            tick.set_color('red')
            tick.set_fontsize(8)

    # Panel 3 (V_rest)
    ax3 = axes[2]
    ax3.bar(x_pos, rmse_vrests[sort_indices], color='lightgreen', alpha=0.7)
    ax3.set_ylabel(r'RMSE $V_{rest}$', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.8])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
    ax3.grid(False)
    ax3.tick_params(axis='y', labelsize=12)

    # Calculate mean ground truth V_rest per neuron type
    mean_gt_vrests = []
    for neuron_type in range(n_neuron_types):
        type_indices = np.where(type_list == neuron_type)[0]
        gt_vrest_type = gt_V_Rest[type_indices]
        mean_gt_vrests.append(np.mean(np.abs(gt_vrest_type)))

    mean_gt_vrests = np.array(mean_gt_vrests)
    for i, (tick, rmse_vrest) in enumerate(zip(ax3.get_xticklabels(), rmse_vrests[sort_indices])):
        if rmse_vrest > 0.08:
            tick.set_color('red')
            tick.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'results', 'neuron_type_reconstruction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Log summary statistics
    logger.info(f"mean weights RMSE: {np.mean(rmse_weights):.3f} ± {np.std(rmse_weights):.3f}")
    logger.info(f"mean tau RMSE: {np.mean(rmse_taus):.3f} ± {np.std(rmse_taus):.3f}")
    logger.info(f"mean V_rest RMSE: {np.mean(rmse_vrests):.3f} ± {np.std(rmse_vrests):.3f}")

    # Write clean key-value metrics file for Notebook_04
    metrics_path = os.path.join(log_dir, 'results', 'metrics.txt')
    if r_squared is not None:
        with open(metrics_path, 'w') as mf:
            mf.write(f"W_corrected_R2: {r_squared:.4f}\n")
            mf.write(f"W_corrected_slope: {slope_corrected:.4f}\n")
            mf.write(f"tau_R2: {r_squared_tau:.4f}\n")
            mf.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")
    try:
        with open(metrics_path, 'a') as mf:
            mf.write(f"clustering_accuracy: {cluster_acc:.4f}\n")
    except NameError:
        pass

    # Return per-neuron results (NEW)
    return {
        'rmse_weights_per_neuron': rmse_weights_per_neuron,
        'rmse_tau_per_neuron': rmse_tau_per_neuron,
        'rmse_vrest_per_neuron': rmse_vrest_per_neuron,
        'rmse_weights_per_type': rmse_weights,
        'rmse_tau_per_type': rmse_taus,
        'rmse_vrest_per_type': rmse_vrests
    }
    pass  # Implement as needed


def plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, delta_t, output_path):

   # Calculate mean and std for each neuron
   mu_activity = torch.mean(activity, dim=1)
   sigma_activity = torch.std(activity, dim=1)

   # Create the plot (keeping original visualization)
   plt.figure(figsize=(16, 8))
   plt.errorbar(np.arange(n_neurons), to_numpy(mu_activity), yerr=to_numpy(sigma_activity),
                fmt='o', ecolor='lightgray', alpha=0.6, elinewidth=1, capsize=0,
                markersize=3, color='red')

   # Group neurons by type and add labels at type boundaries (similar to plot_ground_truth_distributions)
   type_boundaries = {}
   current_type = None
   for i in range(n_neurons):
       neuron_type_id = to_numpy(type_list[i]).item()
       if neuron_type_id != current_type:
           if current_type is not None:
               type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
           type_boundaries[neuron_type_id] = (i, i)
           current_type = neuron_type_id

   # Close the last type boundary
   if current_type is not None:
       type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

   # Add vertical lines and x-tick labels for each neuron type
   tick_positions = []
   tick_labels = []

   for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
       center_pos = (start_idx + end_idx) / 2
       neuron_type_name = index_to_name.get(neuron_type_id, f'Type{neuron_type_id}')

       tick_positions.append(center_pos)
       tick_labels.append(neuron_type_name)

       # Add vertical line at type boundary
       if start_idx > 0:
           plt.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.3)

   # Set x-ticks with neuron type names rotated 90 degrees
   plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
   plt.ylabel(r'neuron voltage $v_i(t)\quad\mu_i \pm \sigma_i$', fontsize=16)
   plt.yticks(fontsize=18)

   plt.tight_layout()
   plt.savefig(os.path.join(output_path, 'activity_mu_sigma.png'), dpi=300, bbox_inches='tight')
   plt.close()

   # Return per-neuron statistics (NEW)
   return {
       'mu_activity': to_numpy(mu_activity),
       'sigma_activity': to_numpy(sigma_activity)
   }


def plot_ground_truth_distributions(edges, true_weights, gt_taus, gt_V_Rest, type_list, n_neuron_types,
                                    sorted_neuron_type_names, output_path):
    """
    Create a 4-panel vertical figure showing ground truth parameter distributions per neuron type
    with neuron type names as x-axis labels
    """

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Get type boundaries for labels
    type_boundaries = {}
    current_type = None
    n_neurons = len(type_list)

    for i in range(n_neurons):
        neuron_type_id = int(type_list[i])
        if neuron_type_id != current_type:
            if current_type is not None:
                type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
            type_boundaries[neuron_type_id] = (i, i)
            current_type = neuron_type_id

    # Close the last type boundary
    if current_type is not None:
        type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

    def add_type_labels_and_setup_axes(ax, y_values, title):
        # Add mean line for each type and collect type positions
        type_positions = []
        type_names = []

        for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
            center_pos = (start_idx + end_idx) / 2
            type_positions.append(center_pos)
            neuron_type_name = sorted_neuron_type_names[int(neuron_type_id)] if int(neuron_type_id) < len(
                sorted_neuron_type_names) else f'Type{neuron_type_id}'
            type_names.append(neuron_type_name)

            # Add mean line for this type
            type_mean = np.mean(y_values[start_idx:end_idx + 1])
            ax.hlines(type_mean, start_idx, end_idx, colors='red', linewidth=3)

        # Set x-ticks to neuron type names
        ax.set_xticks(type_positions)
        ax.set_xticklabels(type_names, rotation=90, fontsize=8)
        ax.tick_params(axis='y', labelsize=16)

    # Panel 1: Scatter plot of true weights per connection with neuron index
    ax1 = axes[0]
    connection_targets = edges[1, :]
    connection_weights = true_weights

    ax1.scatter(connection_targets, connection_weights, c='white', s=0.1)
    ax1.set_ylabel('true weights', fontsize=16)

    # For weights, compute means per target neuron
    weight_means_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == i)[0]
        if len(incoming_edges) > 0:
            weight_means_per_neuron[i] = np.mean(true_weights[incoming_edges])

    add_type_labels_and_setup_axes(ax1, weight_means_per_neuron, 'distribution of true weights by neuron type')

    # Panel 2: Number of connections per neuron
    ax2 = axes[1]
    n_connections_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        n_connections_per_neuron[i] = np.sum(edges[1, :] == i)

    ax2.scatter(np.arange(n_neurons), n_connections_per_neuron, c='white', s=0.1)
    ax2.set_ylabel('number of connections', fontsize=16)
    add_type_labels_and_setup_axes(ax2, n_connections_per_neuron, 'number of incoming connections by neuron type')

    # Panel 3: Scatter plot of true tau values per neuron
    ax3 = axes[2]
    ax3.scatter(np.arange(n_neurons), gt_taus * 1000, c='white', s=0.1)
    ax3.set_ylabel(r'true $\tau$ values [ms]', fontsize=16)
    add_type_labels_and_setup_axes(ax3, gt_taus * 1000, r'distribution of true $\tau$ by neuron type')

    # Panel 4: Scatter plot of true V_rest values per neuron
    ax4 = axes[3]
    ax4.scatter(np.arange(n_neurons), gt_V_Rest, c='white', s=0.1)
    ax4.set_ylabel(r'true $v_{rest}$ values [a.u.]', fontsize=16)
    add_type_labels_and_setup_axes(ax4, gt_V_Rest, r'distribution of true $v_{rest}$ by neuron type')

    plt.tight_layout()
    plt.savefig(f'{output_path}/ground_truth_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return fig
    plt.close()


def data_plot(config, config_file, epoch_list, style, extended, device, apply_weight_correction=False, log_file=None):

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    fig_style.apply_globally()

    log_dir, logger = create_log_dir(config=config, erase=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)

    if epoch_list==['best']:
        files = glob.glob(f"{log_dir}/models/best_model_with_*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch_list=[filename]
        print(f'best model: {epoch_list}')
        logger.info(f'best model: {epoch_list}')

    if os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt', weights_only=False)
        fig, ax = fig_style.figure()
        plt.plot(loss, color=mc, linewidth=4)
        plt.xlim([0, 20])
        plt.ylabel('loss', fontsize=68)
        plt.xlabel('epochs', fontsize=68)
        plt.tight_layout()
        plt.savefig(f"{log_dir}/results/loss.png", dpi=170.7)
        plt.close()
        # Log final loss to analysis.log
        if log_file and len(loss) > 0:
            log_file.write(f"final_loss: {loss[-1]:.4e}\n")


    if 'fly' in config.dataset:
        if config.simulation.calcium_type != 'none':
            plot_synaptic_flyvis_calcium(config, epoch_list, log_dir, logger, 'viridis', style, extended, device) # noqa: F821
        else:
            plot_synaptic_flyvis(config, epoch_list, log_dir, logger, 'viridis', style, extended, device, log_file=log_file)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass


    config_list = ['signal_Claude']


    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(config_path(f'{config_file}.yaml'))
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        print(f'\033[94mconfig_file  {config.config_file}\033[0m')
        folder_name = log_path(pre_folder, 'tmp_results') + '/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(config=config, config_file=config_file, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)
        # data_plot(config=config, config_file=config_file, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=False)
        # data_plot(config=config, config_file=config_file, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=True)


    print("analysis completed")


