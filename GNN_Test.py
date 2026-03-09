"""
Regression test for flyvis-gnn training pipeline.

Runs training + test_plot for a config, compares all metrics against archived
reference values, optionally calls Claude CLI for qualitative assessment,
and appends results with timestamp to a persistent test history log.

Usage:
    # Full local test
    python GNN_Test.py --config flyvis_62_1_gs

    # Full test on cluster
    python GNN_Test.py --config flyvis_62_1_gs --cluster

    # Skip training, only run test_plot + comparison
    python GNN_Test.py --config flyvis_62_1_gs --skip-train

    # Only compare existing results.log (no training, no plotting)
    python GNN_Test.py --config flyvis_62_1_gs --skip-train --skip-plot

    # Skip Claude assessment
    python GNN_Test.py --config flyvis_62_1_gs --no-claude
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

from flyvis_gnn.config import NeuralGraphConfig
from flyvis_gnn.utils import set_device, add_pre_folder, log_path


# ------------------------------------------------------------------ #
#  Metric parsing
# ------------------------------------------------------------------ #

def parse_results_log(path):
    """Parse results.log and extract key metrics.

    Regex patterns match the format written by plot_synaptic_flyvis()
    in GNN_PlotFigure.py (same patterns as compare_gnn_results()).
    """
    if not os.path.exists(path):
        print(f"warning: {path} not found")
        return {}

    with open(path, 'r') as f:
        content = f.read()

    metrics = {}
    patterns = {
        'raw_W_R2':             r'first weights fit\s+R²:\s*([\d.-]+)',
        'raw_W_slope':          r'first weights fit\s+R²:\s*[\d.-]+\s+slope:\s*([\d.-]+)',
        'corrected_W_R2':       r'second weights fit\s+R²:\s*([\d.-]+)',
        'corrected_W_slope':    r'second weights fit\s+R²:\s*[\d.-]+\s+slope:\s*([\d.-]+)',
        'tau_R2':               r'tau reconstruction R²:\s*([\d.-]+)',
        'tau_slope':            r'tau reconstruction R²:\s*[\d.-]+\s+slope:\s*([\d.-]+)',
        'V_rest_R2':            r'V_rest reconstruction R²:\s*([\d.-]+)',
        'V_rest_slope':         r'V_rest reconstruction R²:\s*[\d.-]+\s+slope:\s*([\d.-]+)',
        'spectral_radius_true': r'spectral radius - true:\s*([\d.-]+)',
        'spectral_radius_learned': r'spectral radius - true:\s*[\d.-]+\s+learned:\s*([\d.-]+)',
        'eigenvector_right':    r'eigenvector alignment - right:\s*([\d.-]+)',
        'eigenvector_left':     r'eigenvector alignment.*left:\s*([\d.-]+)',
        'GMM_accuracy':         r'accuracy=([\d.-]+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            metrics[key] = float(m.group(1))

    return metrics


def parse_rollout_log(path):
    """Parse results_rollout.log for rollout metrics."""
    if not os.path.exists(path):
        print(f"warning: {path} not found")
        return {}

    with open(path, 'r') as f:
        content = f.read()

    metrics = {}
    patterns = {
        'rollout_RMSE':    r'RMSE:\s*([\d.-]+)',
        'rollout_pearson': r'Pearson r:\s*([\d.-]+)',
        'rollout_FEVE':    r'FEVE:\s*([\d.-]+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            metrics[key] = float(m.group(1))

    return metrics


def parse_training_output(output_text):
    """Extract training R² and time from training stdout."""
    metrics = {}

    # tqdm line: R²=0.970
    m = re.search(r'R²=([\d.-]+)', output_text)
    if m:
        metrics['training_R2'] = float(m.group(1))

    # "training completed in 28.3 minutes"
    m = re.search(r'training completed in ([\d.]+) minutes', output_text)
    if m:
        metrics['training_time_min'] = float(m.group(1))

    return metrics


# ------------------------------------------------------------------ #
#  Comparison
# ------------------------------------------------------------------ #

def compare_metrics(current, reference, thresholds):
    """Compare current metrics against reference values.

    Returns list of dicts with: metric, reference, current, delta, status.
    """
    rows = []
    all_pass = True

    for key in sorted(set(list(reference.keys()) + list(current.keys()))):
        ref_val = reference.get(key)
        cur_val = current.get(key)

        if ref_val is None or cur_val is None:
            rows.append({
                'metric': key,
                'reference': ref_val,
                'current': cur_val,
                'delta': None,
                'status': 'N/A',
            })
            continue

        delta = cur_val - ref_val
        threshold = thresholds.get(key)

        if threshold is not None:
            # For RMSE, regression means increase; for R²/accuracy, regression means decrease
            if 'RMSE' in key:
                status = 'PASS' if delta <= threshold else 'FAIL'
            else:
                status = 'PASS' if delta >= -threshold else 'FAIL'
        else:
            status = 'INFO'

        if status == 'FAIL':
            all_pass = False

        rows.append({
            'metric': key,
            'reference': ref_val,
            'current': cur_val,
            'delta': delta,
            'status': status,
        })

    return rows, all_pass


def format_comparison_table(rows):
    """Format comparison rows as a markdown table."""
    lines = []
    lines.append("| Metric | Reference | Current | Delta | Status |")
    lines.append("|--------|-----------|---------|-------|--------|")

    for r in rows:
        ref = f"{r['reference']:.4f}" if r['reference'] is not None else "—"
        cur = f"{r['current']:.4f}" if r['current'] is not None else "—"
        delta = f"{r['delta']:+.4f}" if r['delta'] is not None else "—"
        lines.append(f"| {r['metric']} | {ref} | {cur} | {delta} | {r['status']} |")

    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Training execution
# ------------------------------------------------------------------ #

def run_training_local(config, device):
    """Run training locally (same as GNN_Main.py -o train)."""
    from flyvis_gnn.models.graph_trainer import data_train
    data_train(config=config, erase=False, best_model=None, style='color', device=device)


def _local_to_cluster(path, root_dir):
    """Map a local workspace path to the cluster path (same logic as GNN_LLM.py)."""
    cluster_home = "/groups/saalfeld/home/allierc"
    cluster_data_dir = f"{cluster_home}/GraphData"
    cluster_root_dir = f"{cluster_home}/GraphCluster/flyvis-gnn"
    for sub in ('config', 'log', 'graphs_data'):
        local_sub = os.path.join(root_dir, sub)
        if path.startswith(local_sub):
            return os.path.join(cluster_data_dir, sub) + path[len(local_sub):]
    return path.replace(root_dir, cluster_root_dir)


def run_training_cluster(config_name, root_dir, log_dir):
    """Submit training to cluster via SSH + bsub (pattern from GNN_LLM.py)."""
    cluster_home = "/groups/saalfeld/home/allierc"
    cluster_root_dir = f"{cluster_home}/GraphCluster/flyvis-gnn"

    config_file, pre_folder = add_pre_folder(config_name)

    # Build training command
    cluster_train_cmd = (
        f"python GNN_Main.py -o train {config_name}"
    )

    # Write cluster script
    cluster_script_path = os.path.join(log_dir, 'cluster_test_train.sh')
    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {cluster_root_dir}\n")
        f.write(f"conda run -n neural-graph {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = _local_to_cluster(cluster_script_path, root_dir)

    ssh_cmd = (
        f"ssh allierc@login1 \"cd {cluster_root_dir} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_h100 -W 6000 -K "
        f"'bash {cluster_script}'\""
    )

    print(f"\033[96mSubmitting training to cluster: {ssh_cmd}\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\033[91mCluster training failed:\033[0m")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError("Cluster training failed")

    print(f"\033[92mCluster training completed\033[0m")
    print(result.stdout)
    return result.stdout


def run_test_plot(config, config_file, device):
    """Run test + plot (same as GNN_Main.py -o test_plot)."""
    from flyvis_gnn.models.graph_trainer import data_test
    from GNN_PlotFigure import data_plot

    config.simulation.noise_model_level = 0.0
    data_test(
        config=config, visualize=False,
        style="color name continuous_slice", verbose=False,
        best_model='best', run=0, test_mode="",
        sample_embedding=False, step=1000, n_rollout_frames=10000,
        device=device, particle_of_interest=0,
        new_params=None, rollout_without_noise=False,
    )

    pre_folder = os.path.dirname(config.config_file)
    if pre_folder:
        pre_folder += '/'
    folder_name = log_path(pre_folder, 'tmp_results') + '/'
    os.makedirs(folder_name, exist_ok=True)
    data_plot(
        config=config, config_file=config_file,
        epoch_list=['best'], style='color',
        extended='plots', device=device,
        apply_weight_correction=True,
    )


def run_test_plot_cluster(config_name, root_dir, log_dir):
    """Submit test_plot to cluster via SSH + bsub."""
    cluster_home = "/groups/saalfeld/home/allierc"
    cluster_root_dir = f"{cluster_home}/GraphCluster/flyvis-gnn"

    cluster_cmd = f"python GNN_Main.py -o test_plot {config_name}"

    cluster_script_path = os.path.join(log_dir, 'cluster_test_plot.sh')
    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {cluster_root_dir}\n")
        f.write(f"conda run -n neural-graph {cluster_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = _local_to_cluster(cluster_script_path, root_dir)

    ssh_cmd = (
        f"ssh allierc@login1 \"cd {cluster_root_dir} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_h100 -W 6000 -K "
        f"'bash {cluster_script}'\""
    )

    print(f"\033[96mSubmitting test_plot to cluster: {ssh_cmd}\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\033[91mCluster test_plot failed:\033[0m")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError("Cluster test_plot failed")

    print(f"\033[92mCluster test_plot completed\033[0m")
    print(result.stdout)
    return result.stdout


# ------------------------------------------------------------------ #
#  Claude assessment
# ------------------------------------------------------------------ #

def get_claude_assessment(comparison_table, current_metrics, reference_metrics,
                          results_dir, root_dir):
    """Call Claude CLI to generate a qualitative assessment."""

    # Key plot images to reference
    plot_files = [
        'weights_comparison_corrected.png',
        'tau_comparison_*.png',
        'V_rest_comparison_*.png',
        'embedding_*.png',
    ]

    # Find actual plot paths
    import glob
    plot_paths = []
    for pattern in plot_files:
        matches = glob.glob(os.path.join(results_dir, pattern))
        if matches:
            plot_paths.append(matches[0])

    plot_list = '\n'.join(f"- {p}" for p in plot_paths) if plot_paths else "(no plots found)"

    prompt = f"""You are reviewing a regression test for the flyvis-gnn training pipeline.

Compare the current training results against the reference baseline and provide a brief assessment.

## Comparison Table
{comparison_table}

## Key Plot Files
{plot_list}

Please read the plot images listed above and provide:
1. A 2-3 sentence summary of whether results are consistent with the reference
2. Flag any concerning regressions or notable improvements
3. Overall verdict: PASS, WARNING, or FAIL

Keep your response concise (under 200 words)."""

    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', '5',
        '--allowedTools', 'Read',
    ]

    try:
        process = subprocess.Popen(
            claude_cmd, cwd=root_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)

        process.wait()

        output_text = ''.join(output_lines)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print("\033[91mClaude authentication error — skipping assessment\033[0m")
            return "(Claude assessment skipped — authentication error)"

        return output_text.strip()

    except FileNotFoundError:
        print("\033[93mClaude CLI not found — skipping assessment\033[0m")
        return "(Claude assessment skipped — CLI not available)"
    except Exception as e:
        print(f"\033[93mClaude assessment failed: {e}\033[0m")
        return f"(Claude assessment skipped — {e})"


# ------------------------------------------------------------------ #
#  Archival and history
# ------------------------------------------------------------------ #

def archive_results(log_dir, timestamp_str):
    """Copy current results files to archive/ with timestamp."""
    archive_dir = os.path.join(log_dir, 'archive')
    os.makedirs(archive_dir, exist_ok=True)

    for fname in ['results.log', 'results_rollout.log']:
        src = os.path.join(log_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(archive_dir, f"{timestamp_str}_{fname}")
            shutil.copy2(src, dst)
            print(f"archived: {dst}")


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def get_git_branch():
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def append_test_history(log_dir, timestamp_str, commit, branch,
                        comparison_table, all_pass, claude_assessment,
                        config_name=''):
    """Append test entry to log/fly/test_history.md."""
    fly_log_dir = os.path.join(os.path.dirname(log_dir))  # log/fly/
    history_path = os.path.join(fly_log_dir, 'test_history.md')

    # Create header if file doesn't exist
    if not os.path.exists(history_path):
        with open(history_path, 'w') as f:
            f.write("# Regression Test History\n\n")

    verdict = "PASS" if all_pass else "FAIL"

    with open(history_path, 'a') as f:
        f.write(f"## {timestamp_str} — {config_name} — commit {commit} ({branch}) — {verdict}\n\n")
        f.write(comparison_table)
        f.write("\n\n")
        if claude_assessment:
            f.write(f"**Claude assessment:**\n{claude_assessment}\n\n")
        f.write("---\n\n")

    print(f"test history appended: {history_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Regression test for flyvis-gnn')
    parser.add_argument('--config', type=str, default='flyvis_62_1_gs',
                        help='Config name (default: flyvis_62_1_gs)')
    parser.add_argument('--cluster', action='store_true',
                        help='Submit training to cluster via SSH+bsub')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, use existing model')
    parser.add_argument('--skip-plot', action='store_true',
                        help='Skip test_plot, use existing results.log')
    parser.add_argument('--no-claude', action='store_true',
                        help='Skip Claude CLI assessment')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference JSON (default: config/test_reference.json)')

    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(root_dir, 'config')
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')

    # Load config
    config_file, pre_folder = add_pre_folder(args.config)
    config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + args.config

    log_dir = os.path.join(root_dir, 'log', config_file)

    # Load reference
    ref_path = args.reference or os.path.join(config_root, 'test_reference.json')
    if not os.path.exists(ref_path):
        print(f"\033[91mReference file not found: {ref_path}\033[0m")
        sys.exit(1)

    with open(ref_path, 'r') as f:
        ref_data = json.load(f)

    reference_metrics = ref_data['metrics']
    thresholds = ref_data.get('thresholds', {})

    print(f"\033[94m{'='*60}\033[0m")
    print(f"\033[94mRegression Test: {args.config}\033[0m")
    print(f"\033[94mTimestamp: {timestamp_str}\033[0m")
    print(f"\033[94mCommit: {get_git_commit()} ({get_git_branch()})\033[0m")
    print(f"\033[94mReference: {ref_path} (date: {ref_data.get('date', '?')})\033[0m")
    print(f"\033[94m{'='*60}\033[0m")

    # Archive existing results
    archive_results(log_dir, timestamp_str)

    training_output = ""

    # Phase 1: Training
    if not args.skip_train:
        print(f"\n\033[93m--- Phase 1: Training ---\033[0m")
        # Erase metrics.log locally before training to avoid stale R² data
        # in loss.tif right panel (cluster erase may not sync reliably via NFS)
        metrics_log = os.path.join(log_dir, 'tmp_training', 'metrics.log')
        if os.path.exists(metrics_log):
            os.remove(metrics_log)
            print(f"erased stale {metrics_log}")
        if args.cluster:
            training_output = run_training_cluster(args.config, root_dir, log_dir)
        else:
            device = set_device('auto')
            run_training_local(config, device)

    # Phase 2: Test + Plot (always local — cluster test_plot is unreliable)
    if not args.skip_plot:
        print(f"\n\033[93m--- Phase 2: Test + Plot (local) ---\033[0m")
        if 'device' not in dir():
            device = set_device('auto')
        run_test_plot(config, config_file, device)

    # Phase 3: Parse metrics
    print(f"\n\033[93m--- Phase 3: Parse Metrics ---\033[0m")
    results_log_path = os.path.join(log_dir, 'results.log')
    rollout_log_path = os.path.join(log_dir, 'results_rollout.log')

    current_metrics = {}
    current_metrics.update(parse_results_log(results_log_path))
    current_metrics.update(parse_rollout_log(rollout_log_path))

    if training_output:
        current_metrics.update(parse_training_output(training_output))

    if not current_metrics:
        print(f"\033[91mNo metrics found — check that results.log exists at {results_log_path}\033[0m")
        sys.exit(1)

    print(f"Parsed {len(current_metrics)} metrics")

    # Phase 4: Compare
    print(f"\n\033[93m--- Phase 4: Compare ---\033[0m")
    rows, all_pass = compare_metrics(current_metrics, reference_metrics, thresholds)
    comparison_table = format_comparison_table(rows)
    print(comparison_table)

    if all_pass:
        print(f"\n\033[92mOverall: PASS\033[0m")
    else:
        print(f"\n\033[91mOverall: FAIL — some metrics regressed beyond threshold\033[0m")

    # Phase 5: Claude assessment
    claude_assessment = ""
    if not args.no_claude:
        print(f"\n\033[93m--- Phase 5: Claude Assessment ---\033[0m")
        results_dir = os.path.join(log_dir, 'results')
        claude_assessment = get_claude_assessment(
            comparison_table, current_metrics, reference_metrics,
            results_dir, root_dir,
        )

    # Phase 6: Append to history
    print(f"\n\033[93m--- Phase 6: Save Results ---\033[0m")
    commit = get_git_commit()
    branch = get_git_branch()
    append_test_history(log_dir, timestamp_str, commit, branch,
                        comparison_table, all_pass, claude_assessment,
                        config_name=args.config)

    # Summary
    print(f"\n\033[94m{'='*60}\033[0m")
    verdict = "\033[92mPASS\033[0m" if all_pass else "\033[91mFAIL\033[0m"
    print(f"Regression test complete: {verdict}")
    print(f"\033[94m{'='*60}\033[0m")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()

# python GNN_Test.py --config flyvis_noise_005 --cluster
# bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Test.py --config flyvis_noise_005 --cluster"
