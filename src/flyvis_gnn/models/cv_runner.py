"""Cross-validation runner for flyvis-GNN.

Runs generate + train + test + plot for a given config across N seeds,
then saves a summary log and a bar plot (mean ± SD, dots per seed) to
the cv00 log folder (e.g. log/fly/<config_name>_cv00/).

Each CV fold uses a different simulation seed (for data generation) and a
different training seed (sim_seed + 1000), so data and model randomness are
independent.

Usage (run from repo root):
    python src/flyvis_gnn/models/cv_runner.py flyvis_noise_005 --n_seeds 5
    python src/flyvis_gnn/models/cv_runner.py flyvis_noise_005 --seeds 42,43,44,45,46
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Remove the script dir Python auto-inserts: src/flyvis_gnn/models/ contains
# flyvis_gnn.py which would shadow the flyvis_gnn package if left in sys.path.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

# repo root: src/flyvis_gnn/models/ -> src/flyvis_gnn/ -> src/ -> repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_repo_root, 'src'))  # for flyvis_gnn package
sys.path.insert(0, _repo_root)                        # for GNN_PlotFigure

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.generators.graph_data_generator import data_generate
from flyvis_gnn.models.graph_trainer import data_test, data_train
from flyvis_gnn.utils import add_pre_folder, graphs_data_path, log_path, set_device

from GNN_PlotFigure import data_plot  # noqa: E402, I001


METRICS = [
    ('W_corrected_R2',   '$R^2$ conn ($W$)'),
    ('tau_R2',           '$R^2$ $\\tau$'),
    ('V_rest_R2',        '$R^2$ $V^{\\mathrm{rest}}$'),
    ('clustering_accuracy', 'Clustering acc.'),
]


def parse_metrics(path):
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    pass
    return metrics


def _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done):
    x = np.arange(len(METRICS))
    labels = [lbl for _, lbl in METRICS]
    means, sds = [], []
    for key, _ in METRICS:
        vals = [v for v in all_metrics[key] if not np.isnan(v)]
        means.append(np.mean(vals) if vals else 0.0)
        sds.append(np.std(vals) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, means, yerr=sds, capsize=5, color='steelblue', alpha=0.7,
           error_kw=dict(elinewidth=1.5, ecolor='black'))

    rng = np.random.default_rng(0)
    for xi, (key, _) in enumerate(METRICS):
        vals = [v for v in all_metrics[key] if not np.isnan(v)]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jitter, vals, color='black', s=30, zorder=5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('$R^2$ / accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'CV results — {config_name} ({n_done}/{len(seeds)} seeds done)')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()

    plot_path = os.path.join(cv_out_dir, "cv_barplot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  bar plot updated: {plot_path}")


def run_cv(config_name, seeds):
    config_file, pre_folder = add_pre_folder(config_name)
    base_config = NeuralGraphConfig.from_yaml(f"./config/{config_file}.yaml")
    device = set_device(base_config.training.device)

    # Summary goes into the cv00 log folder
    cv_out_dir = log_path(pre_folder + f"{config_name}_cv00")
    os.makedirs(cv_out_dir, exist_ok=True)

    all_metrics = {key: [] for key, _ in METRICS}

    for i, seed in enumerate(seeds):
        run_name = f"{config_name}_cv{i:02d}"
        sim_seed = seed               # simulation / data-generation seed
        train_seed = seed + 1000      # training seed (different from sim)
        print()
        print("=" * 80)
        print(f"CV run {i+1}/{len(seeds)}  sim_seed={sim_seed}  train_seed={train_seed}  ({run_name})")
        print("=" * 80)

        # Per-run dataset and log dir
        config = NeuralGraphConfig.from_yaml(f"./config/{config_file}.yaml")
        config.simulation.seed = sim_seed       # each run generates its own data
        config.training.seed = train_seed       # different seed for training
        config.dataset = pre_folder + run_name  # per-run graphs_data dir
        config.config_file = pre_folder + run_name  # per-run log dir

        graphs_dir = graphs_data_path(config.dataset)

        # --- Generate ---
        data_exists = os.path.isdir(os.path.join(graphs_dir, 'x_list_train'))
        if data_exists:
            print(f"  data already exists at {graphs_dir}/  (skipping generation)")
        else:
            print(f"  generating data at {graphs_dir}/")
            data_generate(config, device=device, visualize=False, run_vizualized=0,
                          style="color", alpha=1, erase=False, save=True, step=100)

        # --- Train ---
        log_dir = log_path(config.config_file)
        model_dir = os.path.join(log_dir, "models")
        model_exists = (os.path.isdir(model_dir) and
                        any(f.startswith("best_model") for f in os.listdir(model_dir))
                        ) if os.path.isdir(model_dir) else False
        if model_exists:
            print(f"  trained model already present in {model_dir}/  (skipping training)")
        else:
            print(f"  training (train_seed={train_seed})...")
            data_train(config, device=device)

        # --- Test ---
        print("  testing...")
        data_test(config=config, visualize=True, style="color name continuous_slice",
                  verbose=False, best_model='best', run=0, step=10,
                  n_rollout_frames=250, device=device)

        # --- Plot / analyse ---
        print("  analysing...")
        data_plot(config=config, config_file=config.config_file,
                  epoch_list=['best'], style='color', extended='plots',
                  device=device)

        # --- Collect metrics ---
        m = parse_metrics(os.path.join(log_dir, "results", "metrics.txt"))
        for key, _ in METRICS:
            val = m.get(key, float('nan'))
            all_metrics[key].append(val)
            print(f"    {key}: {val:.4f}" if not np.isnan(val) else f"    {key}: —")

        # --- Update bar plot after every run ---
        _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done=i + 1)

        # --- Append per-run line to log immediately ---
        summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
        with open(summary_path, 'a') as f:
            if i == 0:
                f.write(f"CV log: {config_name}\n")
                f.write(f"seeds (sim / train): {seeds} / {[s+1000 for s in seeds]}\n")
                f.write("=" * 80 + "\n")
                header = f"{'run':<6} {'sim':<6} {'train':<6}" + "".join(f" {k:<22}" for k, _ in METRICS)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
            vals_str = "".join(
                f" {all_metrics[key][-1]:>22.4f}" if not np.isnan(all_metrics[key][-1])
                else f" {'—':>22}"
                for key, _ in METRICS
            )
            f.write(f"{i:<6} {sim_seed:<6} {train_seed:<6}{vals_str}\n")

    # --- Append summary statistics ---
    summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
    with open(summary_path, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{'Metric':<30} {'Mean':>8} {'SD':>8} {'CV%':>7} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 80 + "\n")
        for key, _ in METRICS:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                sd = np.std(vals)
                cv_pct = (sd / mean * 100) if mean != 0 else float('nan')
                mn, mx = np.min(vals), np.max(vals)
                f.write(f"{key:<30} {mean:>8.4f} {sd:>8.4f} {cv_pct:>6.1f}% {mn:>8.4f} {mx:>8.4f}\n")
            else:
                f.write(f"{key:<30} {'—':>8} {'—':>8} {'—':>7} {'—':>8} {'—':>8}\n")
    print(f"\nCV summary: {summary_path}")
    print(f"Bar plot:   {os.path.join(cv_out_dir, 'cv_barplot.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV benchmark for connectome-GNN")
    parser.add_argument("config_name", help="Config name, e.g. flyvis_noise_005")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds (uses 42, 43, ..., 42+N-1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list, e.g. 42,43,44 (overrides --n_seeds)")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = list(range(42, 42 + args.n_seeds))

    run_cv(args.config_name, seeds)
