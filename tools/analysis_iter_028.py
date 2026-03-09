#!/usr/bin/env python3
"""
Analysis tool for Batch 7 (iterations 25-28).

Key questions to answer:
1. WHY does n_layers=4 help Model 011? Analyze lin_edge layer structure.
2. WHY does embedding_dim=4 marginally help Model 049?
3. Verify lr_W=4E-4 vs 6E-4 effect on Model 041's W distribution.
4. Compare learned embeddings across models with different embedding_dim.

Model configurations tested:
- Model 049 (Iter 25): embedding_dim=4, input_size=5, input_size_update=7
- Model 011 (Iter 26): n_layers=4 (deeper edge MLP)
- Model 041 (Iter 27): lr_W=4E-4 (slower W learning)
- Model 003 (Iter 28): embedding_dim=4 (control)
"""

import numpy as np
import torch
import os

# Model mappings
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
SLOT_TO_MODEL = dict(zip(SLOTS, MODEL_IDS))

def load_model_state(slot):
    """Load trained model state dict for a slot."""
    path = f'log/fly/flyvis_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return sd['model_state_dict']
    except Exception as e:
        print(f"  Error loading slot {slot}: {e}")
        return None

def load_ground_truth(model_id):
    """Load ground truth W for a model."""
    path = f'graphs_data/fly/flyvis_62_1_id_{model_id}/weights.pt'
    try:
        W = torch.load(path, map_location='cpu', weights_only=True).numpy()
        return W
    except Exception as e:
        print(f"  Error loading W for model {model_id}: {e}")
        return None

def analyze_lin_edge_structure(state_dict, n_layers):
    """Analyze lin_edge MLP structure and weight statistics."""
    layer_stats = []
    for i in range(n_layers):
        w_key = f'lin_edge.layers.{i}.weight'
        b_key = f'lin_edge.layers.{i}.bias'
        if w_key in state_dict:
            w = state_dict[w_key].numpy()
            b = state_dict[b_key].numpy() if b_key in state_dict else None
            stats = {
                'layer': i,
                'shape': w.shape,
                'mean': float(np.mean(w)),
                'std': float(np.std(w)),
                'min': float(np.min(w)),
                'max': float(np.max(w)),
                'frac_positive': float(np.mean(w > 0)),
                'frac_large': float(np.mean(np.abs(w) > 0.1)),
                'l2_norm': float(np.linalg.norm(w)),
            }
            layer_stats.append(stats)
    return layer_stats

def analyze_embeddings(state_dict):
    """Analyze learned neuron embeddings."""
    if 'a' not in state_dict:
        return None
    a = state_dict['a'].numpy()
    stats = {
        'shape': a.shape,
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'mean_norm': float(np.mean(np.linalg.norm(a, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(a, axis=1))),
    }
    # Check variance per dimension
    var_per_dim = np.var(a, axis=0)
    stats['var_per_dim'] = var_per_dim.tolist()
    return stats

def compute_w_recovery_stats(W_true, W_learned):
    """Compute W recovery statistics."""
    if W_true is None or W_learned is None:
        return None

    # Basic stats
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    ss_res = np.sum((W_true - W_learned) ** 2)
    ss_tot = np.sum((W_true - np.mean(W_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Sign analysis
    sign_match = np.mean(np.sign(W_true) == np.sign(W_learned))

    # Magnitude analysis
    mag_ratio = np.mean(np.abs(W_learned)) / (np.mean(np.abs(W_true)) + 1e-10)

    return {
        'pearson': pearson,
        'r2': r2,
        'sign_match': sign_match,
        'mag_ratio': mag_ratio,
        'true_mean': float(np.mean(W_true)),
        'true_std': float(np.std(W_true)),
        'learned_mean': float(np.mean(W_learned)),
        'learned_std': float(np.std(W_learned)),
    }

def main():
    print("=" * 70)
    print("ANALYSIS: Batch 7 (Iterations 25-28) — Architectural Experiments")
    print("=" * 70)

    # Configuration for each slot this batch
    configs = {
        0: {'model': '049', 'change': 'embedding_dim=4', 'n_layers': 3, 'emb_dim': 4},
        1: {'model': '011', 'change': 'n_layers=4', 'n_layers': 4, 'emb_dim': 2},
        2: {'model': '041', 'change': 'lr_W=4E-4', 'n_layers': 3, 'emb_dim': 2},
        3: {'model': '003', 'change': 'embedding_dim=4 (control)', 'n_layers': 3, 'emb_dim': 4},
    }

    print("\n" + "=" * 70)
    print("1. LIN_EDGE LAYER ANALYSIS — WHY DOES n_layers=4 HELP MODEL 011?")
    print("=" * 70)

    for slot in SLOTS:
        model_id = configs[slot]['model']
        change = configs[slot]['change']
        n_layers = configs[slot]['n_layers']

        print(f"\n--- Slot {slot} (Model {model_id}): {change} ---")

        state = load_model_state(slot)
        if state is None:
            continue

        layer_stats = analyze_lin_edge_structure(state, n_layers)
        print(f"  n_layers: {n_layers}")
        for ls in layer_stats:
            print(f"  Layer {ls['layer']}: shape={ls['shape']}, mean={ls['mean']:.4f}, std={ls['std']:.4f}")
            print(f"    frac_positive={ls['frac_positive']:.3f}, frac_large={ls['frac_large']:.3f}, L2={ls['l2_norm']:.3f}")

        # Total parameter count in lin_edge
        total_params = sum(np.prod(ls['shape']) for ls in layer_stats)
        print(f"  Total lin_edge params: {total_params}")

    print("\n" + "=" * 70)
    print("2. EMBEDDING ANALYSIS — DOES embedding_dim=4 HELP DIFFERENTIATION?")
    print("=" * 70)

    for slot in SLOTS:
        model_id = configs[slot]['model']
        emb_dim = configs[slot]['emb_dim']

        print(f"\n--- Slot {slot} (Model {model_id}): embedding_dim={emb_dim} ---")

        state = load_model_state(slot)
        if state is None:
            continue

        emb_stats = analyze_embeddings(state)
        if emb_stats:
            print(f"  Shape: {emb_stats['shape']}")
            print(f"  Mean: {emb_stats['mean']:.4f}, Std: {emb_stats['std']:.4f}")
            print(f"  Mean norm: {emb_stats['mean_norm']:.4f}, Std norm: {emb_stats['std_norm']:.4f}")
            print(f"  Var per dim: {[f'{v:.4f}' for v in emb_stats['var_per_dim']]}")

            # Check if all dimensions are used
            var_per_dim = np.array(emb_stats['var_per_dim'])
            active_dims = np.sum(var_per_dim > 0.01)
            print(f"  Active dimensions (var > 0.01): {active_dims}/{len(var_per_dim)}")

    print("\n" + "=" * 70)
    print("3. W RECOVERY COMPARISON — ARCHITECTURAL EFFECTS")
    print("=" * 70)

    for slot in SLOTS:
        model_id = configs[slot]['model']
        change = configs[slot]['change']

        print(f"\n--- Slot {slot} (Model {model_id}): {change} ---")

        state = load_model_state(slot)
        W_true = load_ground_truth(model_id)

        if state is None or W_true is None:
            continue

        W_learned = state['W'].numpy().flatten()

        stats = compute_w_recovery_stats(W_true, W_learned)
        if stats:
            print(f"  Pearson: {stats['pearson']:.4f}, R²: {stats['r2']:.4f}")
            print(f"  Sign match: {stats['sign_match']:.3f}")
            print(f"  Mag ratio (learned/true): {stats['mag_ratio']:.3f}")
            print(f"  True W: mean={stats['true_mean']:.4f}, std={stats['true_std']:.4f}")
            print(f"  Learned W: mean={stats['learned_mean']:.4f}, std={stats['learned_std']:.4f}")

    print("\n" + "=" * 70)
    print("4. CROSS-MODEL COMPARISON: n_layers=4 vs n_layers=3")
    print("=" * 70)

    # Compare Model 011 (n_layers=4) vs Model 049 (n_layers=3)
    # Both have similar per-neuron W correlation issues

    state_011 = load_model_state(1)  # n_layers=4
    state_049 = load_model_state(0)  # n_layers=3

    if state_011 and state_049:
        # Compare lin_edge output capacity
        layers_011 = analyze_lin_edge_structure(state_011, 4)
        layers_049 = analyze_lin_edge_structure(state_049, 3)

        total_011 = sum(np.prod(ls['shape']) for ls in layers_011)
        total_049 = sum(np.prod(ls['shape']) for ls in layers_049)

        print(f"\n  Model 011 (n_layers=4): {total_011} lin_edge params")
        print(f"  Model 049 (n_layers=3): {total_049} lin_edge params")
        print(f"  Ratio: {total_011/total_049:.2f}x")

        # Check output layer weight magnitudes
        out_011 = layers_011[-1] if layers_011 else None
        out_049 = layers_049[-1] if layers_049 else None

        if out_011 and out_049:
            print(f"\n  Output layer comparison:")
            print(f"    Model 011: mean={out_011['mean']:.4f}, std={out_011['std']:.4f}, L2={out_011['l2_norm']:.3f}")
            print(f"    Model 049: mean={out_049['mean']:.4f}, std={out_049['std']:.4f}, L2={out_049['l2_norm']:.3f}")

    print("\n" + "=" * 70)
    print("5. KEY FINDINGS SUMMARY")
    print("=" * 70)

    print("\n  Model 011 (n_layers=4 → NEW BEST 0.769):")
    if state_011:
        W_true_011 = load_ground_truth('011')
        W_learned_011 = state_011['W'].numpy().flatten()
        stats_011 = compute_w_recovery_stats(W_true_011, W_learned_011)
        if stats_011:
            print(f"    - W Pearson: {stats_011['pearson']:.4f}")
            print(f"    - Sign match: {stats_011['sign_match']:.3f}")
            print(f"    - Deeper MLP provides more capacity for complex per-neuron mappings")

    print("\n  Model 049 (embedding_dim=4 → marginal 0.181):")
    if state_049:
        W_true_049 = load_ground_truth('049')
        W_learned_049 = state_049['W'].numpy().flatten()
        stats_049 = compute_w_recovery_stats(W_true_049, W_learned_049)
        if stats_049:
            print(f"    - W Pearson: {stats_049['pearson']:.4f}")
            print(f"    - Sign match: {stats_049['sign_match']:.3f}")
            print(f"    - Richer embeddings help marginally but fundamental limitation persists")

    state_041 = load_model_state(2)
    if state_041:
        W_true_041 = load_ground_truth('041')
        W_learned_041 = state_041['W'].numpy().flatten()
        stats_041 = compute_w_recovery_stats(W_true_041, W_learned_041)
        print("\n  Model 041 (lr_W=4E-4 → NEW BEST 0.919):")
        if stats_041:
            print(f"    - W Pearson: {stats_041['pearson']:.4f}")
            print(f"    - Sign match: {stats_041['sign_match']:.3f}")
            print(f"    - Slower W learning allows better convergence for near-collapsed activity")

    state_003 = load_model_state(3)
    if state_003:
        W_true_003 = load_ground_truth('003')
        W_learned_003 = state_003['W'].numpy().flatten()
        stats_003 = compute_w_recovery_stats(W_true_003, W_learned_003)
        print("\n  Model 003 (embedding_dim=4 control → stable 0.962):")
        if stats_003:
            print(f"    - W Pearson: {stats_003['pearson']:.4f}")
            print(f"    - Sign match: {stats_003['sign_match']:.3f}")
            print(f"    - Confirms embedding_dim=4 is neutral for SOLVED model")

    print("\n" + "=" * 70)
    print("6. RECOMMENDATIONS FOR BATCH 8")
    print("=" * 70)

    print("""
  Model 049: Try combining embedding_dim=4 WITH n_layers=4
    - Both architectural changes showed marginal/significant improvements
    - Combined effect might be greater than individual

  Model 011: Exploit n_layers=4, try n_layers_update=4 or fine-tune regularization
    - n_layers=4 is confirmed helpful
    - May benefit from deeper update MLP as well

  Model 041: Trade-off discovered — lr_W=4E-4 best for conn, lr_W=6E-4 best for tau
    - Consider whether connectivity or tau is priority
    - Already at 0.919, near ceiling

  Model 003: SOLVED, use slot for alternative experiments
    - Could test n_layers=4 as control to see if it helps SOLVED model
""")

    print("=" * 70)
    print("END ANALYSIS")
    print("=" * 70)

if __name__ == '__main__':
    main()
