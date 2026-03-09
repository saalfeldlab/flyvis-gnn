#!/usr/bin/env python3
"""
Analysis tool for Batch 11 (Iterations 41-44) — Understanding Exploration

Focus: Analyzing WHY architectural simplification destroys recurrent gains for Model 049,
why W_L1=5E-5 hurts Model 011 recurrent, and why recurrent_training hurts Model 041.

Key questions:
1. Model 049: What is different about the learned W when using simpler architecture?
2. Model 011: How does W_L1 strength interact with recurrent gradient aggregation?
3. Model 041: Why does recurrent training hurt near-collapsed activity?
4. Cross-model: Is recurrent benefit tied to per-neuron W correlation sign?
"""

import torch
import numpy as np
import os

# Configuration
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
DATA_DIR = 'graphs_data/fly'
LOG_DIR = 'log/fly'

def load_W_true(model_id):
    """Load ground truth connectivity weights."""
    path = f'{DATA_DIR}/flyvis_62_1_id_{model_id}/weights.pt'
    return torch.load(path, weights_only=True, map_location='cpu').numpy()

def load_W_learned(slot):
    """Load learned connectivity weights from trained model."""
    model_path = f'{LOG_DIR}/flyvis_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        return sd['model_state_dict']['W'].numpy().flatten()
    except Exception as e:
        print(f"Error loading slot {slot}: {e}")
        return None

def load_embeddings(slot):
    """Load learned neuron embeddings."""
    model_path = f'{LOG_DIR}/flyvis_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        return sd['model_state_dict']['a'].numpy()
    except:
        return None

def load_edge_index(model_id):
    """Load edge connectivity (source, target neuron indices)."""
    path = f'{DATA_DIR}/flyvis_62_1_id_{model_id}/edge_index.pt'
    return torch.load(path, weights_only=True, map_location='cpu').numpy()

def compute_per_neuron_W(W, edge_index, n_neurons=13741):
    """Compute per-neuron incoming and outgoing W sums."""
    src, tgt = edge_index[0], edge_index[1]
    incoming = np.zeros(n_neurons)
    outgoing = np.zeros(n_neurons)

    np.add.at(incoming, tgt, W)
    np.add.at(outgoing, src, W)

    return incoming, outgoing

def pearson_r(x, y):
    """Compute Pearson correlation."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return np.mean((x - mx) * (y - my)) / (sx * sy)

def analyze_W_recovery(W_true, W_learned):
    """Analyze W recovery metrics."""
    # Basic statistics
    pearson = pearson_r(W_true, W_learned)
    r2 = 1 - np.sum((W_true - W_learned)**2) / np.sum((W_true - W_true.mean())**2)

    # Sign analysis
    true_sign = np.sign(W_true)
    learned_sign = np.sign(W_learned)
    sign_match = np.mean(true_sign == learned_sign)

    # Magnitude analysis
    mag_true = np.abs(W_true).mean()
    mag_learned = np.abs(W_learned).mean()
    mag_ratio = mag_learned / mag_true if mag_true > 1e-10 else 0

    return {
        'pearson': pearson,
        'r2': r2,
        'sign_match': sign_match,
        'mag_true': mag_true,
        'mag_learned': mag_learned,
        'mag_ratio': mag_ratio
    }

def analyze_lin_edge_mlp(slot):
    """Analyze lin_edge MLP structure and activations."""
    model_path = f'{LOG_DIR}/flyvis_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(model_path, map_location='cpu', weights_only=False)

        # Count layers
        layer_keys = [k for k in sd['model_state_dict'].keys() if k.startswith('lin_edge.layers')]
        layer_nums = set(int(k.split('.')[2]) for k in layer_keys if '.weight' in k)
        n_layers = len(layer_nums)

        # Get layer statistics
        layer_stats = {}
        for i in sorted(layer_nums):
            w_key = f'lin_edge.layers.{i}.weight'
            if w_key in sd['model_state_dict']:
                w = sd['model_state_dict'][w_key].numpy()
                layer_stats[i] = {
                    'shape': w.shape,
                    'n_params': w.size,
                    'l2_norm': np.sqrt(np.sum(w**2)),
                    'frac_positive': (w > 0).mean(),
                    'frac_large': (np.abs(w) > 0.1).mean()
                }

        return n_layers, layer_stats
    except Exception as e:
        print(f"Error analyzing MLP for slot {slot}: {e}")
        return 0, {}

def main():
    print("=" * 70)
    print("BATCH 11 ANALYSIS: Architecture, W_L1, and Recurrent Training Effects")
    print("=" * 70)

    # ===== Section 1: Compare Model 049 Iter 33 vs Iter 41 =====
    print("\n" + "=" * 70)
    print("SECTION 1: Model 049 — Why Simpler Architecture Destroys Recurrent Gains")
    print("=" * 70)
    print("\nIter 33 (n_layers=4, emb=4): conn_R2=0.501 (BEST)")
    print("Iter 41 (n_layers=3, emb=2): conn_R2=0.150 (3.3x WORSE)")

    # Load current (Iter 41) W
    W_true_049 = load_W_true('049')
    W_learned_049 = load_W_learned(0)  # Slot 0 = Model 049
    edge_index_049 = load_edge_index('049')

    if W_learned_049 is not None:
        metrics = analyze_W_recovery(W_true_049, W_learned_049)
        print(f"\nIter 41 W metrics:")
        print(f"  Pearson:     {metrics['pearson']:.4f}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  Sign match:  {metrics['sign_match']*100:.1f}%")
        print(f"  Mag ratio:   {metrics['mag_ratio']:.2f}x")

        # Per-neuron analysis
        in_true, out_true = compute_per_neuron_W(W_true_049, edge_index_049)
        in_learned, out_learned = compute_per_neuron_W(W_learned_049, edge_index_049)

        in_pearson = pearson_r(in_true, in_learned)
        out_pearson = pearson_r(out_true, out_learned)

        print(f"\nPer-neuron W correlation (Iter 41):")
        print(f"  Incoming: {in_pearson:.4f}")
        print(f"  Outgoing: {out_pearson:.4f}")

        # Embedding analysis
        emb = load_embeddings(0)
        if emb is not None:
            print(f"\nEmbedding shape: {emb.shape}")
            emb_var = np.var(emb, axis=0)
            print(f"  Per-dim variance: {emb_var}")
            print(f"  Active dims (var>0.1): {np.sum(emb_var > 0.1)}/{emb.shape[1]}")

    # Analyze MLP structure
    n_layers, layer_stats = analyze_lin_edge_mlp(0)
    print(f"\nlin_edge MLP: {n_layers} layers")
    total_params = sum(s['n_params'] for s in layer_stats.values())
    print(f"  Total params: {total_params}")
    for i, stats in layer_stats.items():
        print(f"  Layer {i}: shape={stats['shape']}, L2={stats['l2_norm']:.3f}, frac_large={stats['frac_large']:.3f}")

    print("\n>>> HYPOTHESIS: Simpler architecture (n_layers=3, emb=2) cannot process")
    print("    temporal gradient aggregation effectively. Recurrent training needs")
    print("    MORE capacity to extract useful signal from accumulated gradients.")

    # ===== Section 2: Model 011 — W_L1 interaction with recurrent =====
    print("\n" + "=" * 70)
    print("SECTION 2: Model 011 — Why W_L1=5E-5 Hurts Recurrent Training")
    print("=" * 70)
    print("\nIter 38 (W_L1=3E-5, recurrent): conn_R2=0.810 (BEST)")
    print("Iter 42 (W_L1=5E-5, recurrent): conn_R2=0.732 (REGRESSION)")

    W_true_011 = load_W_true('011')
    W_learned_011 = load_W_learned(1)  # Slot 1 = Model 011
    edge_index_011 = load_edge_index('011')

    if W_learned_011 is not None:
        metrics = analyze_W_recovery(W_true_011, W_learned_011)
        print(f"\nIter 42 W metrics (W_L1=5E-5):")
        print(f"  Pearson:     {metrics['pearson']:.4f}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  Sign match:  {metrics['sign_match']*100:.1f}%")
        print(f"  Mag ratio:   {metrics['mag_ratio']:.2f}x")

        # Per-neuron analysis
        in_true, out_true = compute_per_neuron_W(W_true_011, edge_index_011)
        in_learned, out_learned = compute_per_neuron_W(W_learned_011, edge_index_011)

        in_pearson = pearson_r(in_true, in_learned)
        out_pearson = pearson_r(out_true, out_learned)

        print(f"\nPer-neuron W correlation (Iter 42):")
        print(f"  Incoming: {in_pearson:.4f}")
        print(f"  Outgoing: {out_pearson:.4f}")

        # W magnitude analysis
        print(f"\nW magnitude analysis:")
        print(f"  W_true:    mean={W_true_011.mean():.6f}, std={W_true_011.std():.6f}")
        print(f"  W_learned: mean={W_learned_011.mean():.6f}, std={W_learned_011.std():.6f}")
        print(f"  |W_true|>0.01:    {(np.abs(W_true_011) > 0.01).sum()}")
        print(f"  |W_learned|>0.01: {(np.abs(W_learned_011) > 0.01).sum()}")

    print("\n>>> HYPOTHESIS: Stronger W_L1 (5E-5) over-penalizes W during recurrent")
    print("    gradient aggregation. Recurrent training accumulates gradients over")
    print("    time, so the effective L1 penalty is multiplied. W_L1=3E-5 is optimal.")

    # ===== Section 3: Model 041 — Why recurrent hurts near-collapsed activity =====
    print("\n" + "=" * 70)
    print("SECTION 3: Model 041 — Why Recurrent Training Hurts Near-Collapsed Activity")
    print("=" * 70)
    print("\nIter 35 (recurrent=False): conn_R2=0.931 (BEST)")
    print("Iter 43 (recurrent=True):  conn_R2=0.869 (REGRESSION)")

    W_true_041 = load_W_true('041')
    W_learned_041 = load_W_learned(2)  # Slot 2 = Model 041
    edge_index_041 = load_edge_index('041')

    if W_learned_041 is not None:
        metrics = analyze_W_recovery(W_true_041, W_learned_041)
        print(f"\nIter 43 W metrics (recurrent=True):")
        print(f"  Pearson:     {metrics['pearson']:.4f}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  Sign match:  {metrics['sign_match']*100:.1f}%")
        print(f"  Mag ratio:   {metrics['mag_ratio']:.2f}x")

        # Per-neuron analysis
        in_true, out_true = compute_per_neuron_W(W_true_041, edge_index_041)
        in_learned, out_learned = compute_per_neuron_W(W_learned_041, edge_index_041)

        in_pearson = pearson_r(in_true, in_learned)
        out_pearson = pearson_r(out_true, out_learned)

        print(f"\nPer-neuron W correlation (Iter 43):")
        print(f"  Incoming: {in_pearson:.4f}")
        print(f"  Outgoing: {out_pearson:.4f}")

    print("\n>>> HYPOTHESIS: Near-collapsed activity (svd_rank=6) has very low-dimensional")
    print("    gradient signal. Per-frame training already extracts maximal information.")
    print("    Recurrent training adds temporal noise that degrades the signal.")

    # ===== Section 4: Model 003 — Confirm stable =====
    print("\n" + "=" * 70)
    print("SECTION 4: Model 003 — 11th Confirmation of SOLVED Status")
    print("=" * 70)
    print("\nIter 44 (recurrent=False): conn_R2=0.968 (STABLE)")

    W_true_003 = load_W_true('003')
    W_learned_003 = load_W_learned(3)  # Slot 3 = Model 003
    edge_index_003 = load_edge_index('003')

    if W_learned_003 is not None:
        metrics = analyze_W_recovery(W_true_003, W_learned_003)
        print(f"\nIter 44 W metrics:")
        print(f"  Pearson:     {metrics['pearson']:.4f}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  Sign match:  {metrics['sign_match']*100:.1f}%")
        print(f"  Mag ratio:   {metrics['mag_ratio']:.2f}x")

        # Per-neuron analysis
        in_true, out_true = compute_per_neuron_W(W_true_003, edge_index_003)
        in_learned, out_learned = compute_per_neuron_W(W_learned_003, edge_index_003)

        in_pearson = pearson_r(in_true, in_learned)
        out_pearson = pearson_r(out_true, out_learned)

        print(f"\nPer-neuron W correlation (Iter 44):")
        print(f"  Incoming: {in_pearson:.4f}")
        print(f"  Outgoing: {out_pearson:.4f}")

    # ===== Section 5: Cross-Model Summary =====
    print("\n" + "=" * 70)
    print("SECTION 5: Cross-Model Summary — Recurrent Training is Model-Dependent")
    print("=" * 70)

    print("\n| Model | svd_rank | Per-neuron W | Best Config | Recurrent Effect |")
    print("|-------|----------|--------------|-------------|------------------|")
    print("| 049   | 19       | NEGATIVE     | recurrent+complex | HELPS (0.16→0.50) |")
    print("| 011   | 45       | NEGATIVE     | recurrent+W_L1=3E-5 | HELPS (0.77→0.81) |")
    print("| 041   | 6        | MIXED        | per-frame   | HURTS (0.93→0.87) |")
    print("| 003   | 60       | POSITIVE     | per-frame   | NEUTRAL (0.97→0.97) |")

    print("\n>>> KEY INSIGHT: Recurrent training benefit is tied to per-neuron W correlation:")
    print("    - NEGATIVE per-neuron W → recurrent HELPS (temporal context disambiguates)")
    print("    - POSITIVE per-neuron W → recurrent NEUTRAL (already sufficient info)")
    print("    - Near-collapsed activity → recurrent HURTS (adds noise to weak signal)")

    print("\n>>> NEXT STEPS:")
    print("    - Model 049: Keep Iter 33 config, try slower lr_W=5E-4")
    print("    - Model 011: Keep Iter 38 config (W_L1=3E-5), try lr_W variations")
    print("    - Model 041: SOLVED at Iter 35, no further experiments")
    print("    - Model 003: SOLVED at Iter 4, no further experiments")

    print("\n" + "=" * 70)
    print("END OF ANALYSIS")
    print("=" * 70)

if __name__ == '__main__':
    main()
