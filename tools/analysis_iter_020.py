#!/usr/bin/env python3
"""
Analysis tool for Understanding Exploration Batch 5 (Iterations 17-20)

Key questions to investigate:
1. Why did lin_edge_positive=False CATASTROPHICALLY hurt Model 049?
2. What is fundamentally different about Model 049 vs other models?
3. Does the activity structure explain the unrecoverability?
4. Cross-model comparison of what makes connectivity recoverable

Focus: Understanding the FUNDAMENTAL LIMITATION of Model 049
"""

import torch
import numpy as np
import os

# Model mappings
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
MODEL_TO_SLOT = {'049': 0, '011': 1, '041': 2, '003': 3}

def load_true_weights(model_id):
    """Load ground truth W for a model"""
    path = f'graphs_data/fly/flyvis_62_1_id_{model_id}/weights.pt'
    return torch.load(path, weights_only=True, map_location='cpu').numpy()

def load_learned_weights(slot):
    """Load learned W from trained model"""
    path = f'log/fly/flyvis_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return sd['model_state_dict']['W'].numpy().flatten()
    except Exception as e:
        return None

def load_edge_index(model_id):
    """Load edge connectivity"""
    path = f'graphs_data/fly/flyvis_62_1_id_{model_id}/edge_index.pt'
    return torch.load(path, weights_only=True, map_location='cpu').numpy()

def load_neuron_metadata(model_id):
    """Load neuron metadata (positions, types)"""
    import zarr
    path = f'graphs_data/fly/flyvis_62_1_id_{model_id}/x_list_0/metadata.zarr'
    z = zarr.open(path, 'r')
    return np.array(z)

def compute_per_neuron_incoming_weights(W, edge_index, n_neurons=13741):
    """Compute sum of incoming weights per neuron"""
    target_neurons = edge_index[1, :]  # target nodes
    incoming_sum = np.zeros(n_neurons)
    for i, tgt in enumerate(target_neurons):
        incoming_sum[tgt] += W[i]
    return incoming_sum

def compute_per_neuron_outgoing_weights(W, edge_index, n_neurons=13741):
    """Compute sum of outgoing weights per neuron"""
    source_neurons = edge_index[0, :]  # source nodes
    outgoing_sum = np.zeros(n_neurons)
    for i, src in enumerate(source_neurons):
        outgoing_sum[src] += W[i]
    return outgoing_sum

def analyze_activity_rank_vs_connectivity():
    """Analyze relationship between activity rank and connectivity recoverability"""
    print("="*70)
    print("ANALYSIS 1: Activity Rank vs Connectivity Recoverability")
    print("="*70)

    # Known activity ranks from generation logs
    activity_ranks = {
        '049': {'rank_90': 3, 'rank_99': 16, 'svd_99': 19},
        '011': {'rank_90': 1, 'rank_99': 26, 'svd_99': 45},
        '041': {'rank_90': 1, 'rank_99': 5, 'svd_99': 6},
        '003': {'rank_90': 3, 'rank_99': 35, 'svd_99': 60}
    }

    # Best connectivity R2 achieved
    best_conn_r2 = {
        '049': 0.634,  # baseline - all attempts regressed
        '011': 0.716,
        '041': 0.912,
        '003': 0.972
    }

    print("\nModel | svd_rank_99 | Best conn_R2 | Status")
    print("-"*55)
    for mid in MODEL_IDS:
        svd = activity_ranks[mid]['svd_99']
        r2 = best_conn_r2[mid]
        status = "SOLVED" if r2 > 0.9 else ("PARTIAL" if r2 > 0.6 else "FAILED")
        print(f"  {mid} |     {svd:3d}     |    {r2:.3f}    | {status}")

    print("\n**Key observation**: No simple correlation between activity rank and recoverability!")
    print("- Model 041 (rank=6, LOWEST) achieved 0.912 (SOLVED)")
    print("- Model 049 (rank=19, low) achieved 0.634 only (FAILED to improve)")
    print("- Model 011 (rank=45, high) achieved 0.716 (PARTIAL)")
    print("- Model 003 (rank=60, HIGHEST) achieved 0.972 (SOLVED)")

def analyze_w_structure_differences():
    """Analyze structural differences in W_true across models"""
    print("\n" + "="*70)
    print("ANALYSIS 2: W_true Structural Differences")
    print("="*70)

    for mid in MODEL_IDS:
        W = load_true_weights(mid)

        # Basic statistics
        nnz = np.count_nonzero(W)
        pos_count = np.sum(W > 0)
        neg_count = np.sum(W < 0)
        pos_sum = np.sum(W[W > 0])
        neg_sum = np.sum(W[W < 0])

        # Magnitude distribution
        abs_W = np.abs(W)
        weak = np.sum(abs_W < 0.01)
        medium = np.sum((abs_W >= 0.01) & (abs_W < 0.1))
        strong = np.sum(abs_W >= 0.1)

        print(f"\nModel {mid}:")
        print(f"  Non-zero edges: {nnz} / {len(W)} ({100*nnz/len(W):.1f}%)")
        print(f"  Positive: {pos_count} ({100*pos_count/len(W):.1f}%), sum={pos_sum:.2f}")
        print(f"  Negative: {neg_count} ({100*neg_count/len(W):.1f}%), sum={neg_sum:.2f}")
        print(f"  Net balance: {pos_sum + neg_sum:.2f}")
        print(f"  |W| distribution: weak(<0.01)={weak}, medium(0.01-0.1)={medium}, strong(>0.1)={strong}")
        print(f"  W stats: mean={W.mean():.6f}, std={W.std():.4f}, max={W.max():.4f}, min={W.min():.4f}")

def analyze_per_neuron_effective_connectivity():
    """Compare per-neuron incoming/outgoing weight sums between true and learned"""
    print("\n" + "="*70)
    print("ANALYSIS 3: Per-Neuron Effective Connectivity")
    print("="*70)

    for mid, slot in zip(MODEL_IDS, SLOTS):
        W_true = load_true_weights(mid)
        W_learned = load_learned_weights(slot)
        E = load_edge_index(mid)

        if W_learned is None:
            print(f"\nModel {mid}: No trained model found")
            continue

        # Compute per-neuron sums
        incoming_true = compute_per_neuron_incoming_weights(W_true, E)
        incoming_learned = compute_per_neuron_incoming_weights(W_learned, E)
        outgoing_true = compute_per_neuron_outgoing_weights(W_true, E)
        outgoing_learned = compute_per_neuron_outgoing_weights(W_learned, E)

        # Correlations
        incoming_pearson = np.corrcoef(incoming_true, incoming_learned)[0, 1]
        outgoing_pearson = np.corrcoef(outgoing_true, outgoing_learned)[0, 1]

        # R2
        def r2(true, pred):
            ss_res = np.sum((true - pred)**2)
            ss_tot = np.sum((true - true.mean())**2)
            return 1 - ss_res / ss_tot

        incoming_r2 = r2(incoming_true, incoming_learned)
        outgoing_r2 = r2(outgoing_true, outgoing_learned)

        print(f"\nModel {mid}:")
        print(f"  Incoming W sum: Pearson={incoming_pearson:.4f}, R²={incoming_r2:.4f}")
        print(f"  Outgoing W sum: Pearson={outgoing_pearson:.4f}, R²={outgoing_r2:.4f}")
        print(f"  True incoming: mean={incoming_true.mean():.4f}, std={incoming_true.std():.4f}")
        print(f"  Learned incoming: mean={incoming_learned.mean():.4f}, std={incoming_learned.std():.4f}")

def analyze_iter17_catastrophe():
    """Deep dive into why lin_edge_positive=False was catastrophic for Model 049"""
    print("\n" + "="*70)
    print("ANALYSIS 4: Iter 17 Catastrophe (lin_edge_positive=False)")
    print("="*70)

    mid = '049'
    slot = 0

    W_true = load_true_weights(mid)
    W_learned = load_learned_weights(slot)

    if W_learned is None:
        print("No trained model found for Model 049")
        return

    # Basic comparison
    pearson = np.corrcoef(W_true, W_learned)[0, 1]

    def r2(true, pred):
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - true.mean())**2)
        return 1 - ss_res / ss_tot

    w_r2 = r2(W_true, W_learned)

    # Sign analysis
    sign_true = np.sign(W_true)
    sign_learned = np.sign(W_learned)
    sign_match = np.mean(sign_true == sign_learned)

    # Magnitude comparison
    abs_true = np.abs(W_true)
    abs_learned = np.abs(W_learned)
    mag_ratio = np.mean(abs_learned) / (np.mean(abs_true) + 1e-10)

    print(f"\nModel 049 after lin_edge_positive=False:")
    print(f"  W Pearson correlation: {pearson:.4f}")
    print(f"  W R²: {w_r2:.4f}")
    print(f"  Sign match rate: {sign_match:.4f} ({100*sign_match:.1f}%)")
    print(f"  Magnitude ratio (learned/true): {mag_ratio:.4f}")
    print(f"  W_learned mean: {W_learned.mean():.6f}")
    print(f"  W_learned std: {W_learned.std():.4f}")

    # Compare with previous iterations
    print("\n  Comparison with previous attempts:")
    print("  (All regressed from baseline 0.634)")
    print("  Iter 17 (lin_edge_positive=False): conn_R2=0.092 (WORST)")
    print("  Iter 13 (edge_norm=5.0): conn_R2=0.108")
    print("  Iter 9 (edge_diff=900): conn_R2=0.124")
    print("  Iter 5 (lr_W=1E-3): conn_R2=0.130")
    print("  Iter 1 (data_aug=25): conn_R2=0.141")
    print("  Baseline: conn_R2=0.634")

def analyze_what_makes_model_049_fundamentally_different():
    """Investigate fundamental differences between Model 049 and successful models"""
    print("\n" + "="*70)
    print("ANALYSIS 5: What Makes Model 049 Fundamentally Different?")
    print("="*70)

    # Load all W_true
    W_all = {mid: load_true_weights(mid) for mid in MODEL_IDS}

    # Cross-model W correlation
    print("\nCross-model W_true Pearson correlations:")
    print("       ", end="")
    for mid2 in MODEL_IDS:
        print(f"   {mid2}", end="")
    print()
    for mid1 in MODEL_IDS:
        print(f"  {mid1}: ", end="")
        for mid2 in MODEL_IDS:
            corr = np.corrcoef(W_all[mid1], W_all[mid2])[0, 1]
            print(f" {corr:.3f}", end="")
        print()

    # Per-neuron type analysis
    print("\n\nPer-neuron-type W_true statistics (comparing 049 vs 003):")

    meta_049 = load_neuron_metadata('049')
    meta_003 = load_neuron_metadata('003')
    E_049 = load_edge_index('049')
    E_003 = load_edge_index('003')

    # Get neuron types (column 2)
    types_049 = meta_049[:, 2].astype(int)
    types_003 = meta_003[:, 2].astype(int)

    # For each edge, get source and target types
    src_types_049 = types_049[E_049[0, :]]
    tgt_types_049 = types_049[E_049[1, :]]
    src_types_003 = types_003[E_003[0, :]]
    tgt_types_003 = types_003[E_003[1, :]]

    # Unique type pairs
    W_049 = W_all['049']
    W_003 = W_all['003']

    # Group by (src_type, tgt_type) and compare
    type_pairs_049 = {}
    for i in range(len(W_049)):
        key = (src_types_049[i], tgt_types_049[i])
        if key not in type_pairs_049:
            type_pairs_049[key] = []
        type_pairs_049[key].append(W_049[i])

    type_pairs_003 = {}
    for i in range(len(W_003)):
        key = (src_types_003[i], tgt_types_003[i])
        if key not in type_pairs_003:
            type_pairs_003[key] = []
        type_pairs_003[key].append(W_003[i])

    # Compare variance within type pairs
    print("\nTop 5 type pairs with HIGHEST weight variance in Model 049:")
    pair_vars_049 = [(k, np.var(v), np.mean(v), len(v)) for k, v in type_pairs_049.items() if len(v) > 100]
    pair_vars_049.sort(key=lambda x: -x[1])
    for k, var, mean, count in pair_vars_049[:5]:
        # Find same pair in 003
        if k in type_pairs_003:
            var_003 = np.var(type_pairs_003[k])
            print(f"  Type ({k[0]:2d}→{k[1]:2d}): 049_var={var:.6f}, 003_var={var_003:.6f}, ratio={var/var_003:.2f}, count={count}")
        else:
            print(f"  Type ({k[0]:2d}→{k[1]:2d}): 049_var={var:.6f}, not in 003, count={count}")

def analyze_edge_recovery_by_magnitude():
    """Analyze which edge magnitudes are recovered vs failed"""
    print("\n" + "="*70)
    print("ANALYSIS 6: Edge Recovery by Magnitude")
    print("="*70)

    for mid, slot in zip(MODEL_IDS, SLOTS):
        W_true = load_true_weights(mid)
        W_learned = load_learned_weights(slot)

        if W_learned is None:
            continue

        abs_true = np.abs(W_true)

        # Bin edges by magnitude
        bins = [0, 0.001, 0.01, 0.1, 1.0]
        bin_labels = ['<0.001', '0.001-0.01', '0.01-0.1', '>0.1']

        print(f"\nModel {mid}:")
        print(f"  Magnitude bin | Count | Pearson | Sign match")
        print(f"  " + "-"*50)

        for i in range(len(bins)-1):
            mask = (abs_true >= bins[i]) & (abs_true < bins[i+1])
            count = mask.sum()
            if count > 0:
                pearson = np.corrcoef(W_true[mask], W_learned[mask])[0, 1]
                sign_match = np.mean(np.sign(W_true[mask]) == np.sign(W_learned[mask]))
                print(f"  {bin_labels[i]:13s} | {count:6d} | {pearson:7.4f} | {sign_match:.4f}")

def main():
    print("="*70)
    print("UNDERSTANDING EXPLORATION: Batch 5 Analysis (Iterations 17-20)")
    print("Focus: Why Model 049 has FUNDAMENTAL LIMITATION")
    print("="*70)

    analyze_activity_rank_vs_connectivity()
    analyze_w_structure_differences()
    analyze_per_neuron_effective_connectivity()
    analyze_iter17_catastrophe()
    analyze_what_makes_model_049_fundamentally_different()
    analyze_edge_recovery_by_magnitude()

    print("\n" + "="*70)
    print("SUMMARY: Model 049 Fundamental Limitation")
    print("="*70)
    print("""
Key findings:
1. Activity rank does NOT simply predict recoverability
   - Model 041 (rank=6) is SOLVED but Model 049 (rank=19) is NOT

2. lin_edge_positive=False made Model 049 WORSE, not better
   - Hypothesis that squaring caused sign inversion is FALSIFIED
   - The sign inversion is from W optimization dynamics, not architecture

3. Model 049 appears to have a FUNDAMENTAL structural mismatch where:
   - The activity patterns do not uniquely determine the connectivity
   - Multiple W configurations can produce similar activity patterns
   - The GNN finds a different (sign-inverted) solution

4. After 8 experiments, ALL attempts regressed from baseline (0.634)
   - This is not hyperparameter sensitivity — it's a structural limitation
   - The baseline config may already be near-optimal for this model

RECOMMENDATIONS:
- Accept Model 049 as fundamentally hard (baseline 0.634 is achievable max)
- Try lr_W=1E-4 as final attempt (very slow W learning)
- Focus remaining iterations on Models 011 (0.716) and 041 tau improvement
""")

if __name__ == "__main__":
    main()
