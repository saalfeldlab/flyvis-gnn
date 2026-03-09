#!/usr/bin/env python
"""
Analysis tool for iterations 9-12 (Batch 3).

KEY QUESTION: The Model 049 PARADOX
- tau_R2=0.899 and V_rest_R2=0.666 are EXCELLENT
- BUT connectivity_R2=0.124 is CATASTROPHIC
- This means the GNN learns to predict dynamics correctly WITHOUT learning correct W
- How is this possible? What is W_learned doing?

Hypotheses to test:
1. W_learned may have different structure but similar EFFECTIVE connectivity (e.g., many weak edges vs few strong edges)
2. The tau/V_rest parameters may be compensating for wrong W
3. The activity pattern of Model 049 may be degenerate (multiple W solutions)
"""

import torch
import numpy as np
import os

# Configuration
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
BASE_DATA = 'graphs_data/fly/flyvis_62_1_id_{}'
BASE_LOG = 'log/fly/flyvis_62_1_understand_Claude_{:02d}'

print("=" * 70)
print("ANALYSIS ITER 012: The Model 049 Paradox")
print("=" * 70)
print("Why does WRONG connectivity produce CORRECT dynamics?")
print()

# Load all data
all_data = {}
for mid, slot in zip(MODEL_IDS, SLOTS):
    data_dir = BASE_DATA.format(mid)
    log_dir = BASE_LOG.format(slot)

    try:
        W_true = torch.load(f'{data_dir}/weights.pt', weights_only=True, map_location='cpu').numpy()
        tau_true = torch.load(f'{data_dir}/taus.pt', weights_only=True, map_location='cpu').numpy()
        V_rest_true = torch.load(f'{data_dir}/V_i_rest.pt', weights_only=True, map_location='cpu').numpy()
        edge_index = torch.load(f'{data_dir}/edge_index.pt', weights_only=True, map_location='cpu').numpy()

        model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'
        sd = torch.load(model_path, map_location='cpu', weights_only=False)['model_state_dict']
        W_learned = sd['W'].cpu().numpy().flatten()
        embeddings = sd['a'].cpu().numpy()

        all_data[mid] = {
            'W_true': W_true,
            'W_learned': W_learned,
            'tau_true': tau_true,
            'V_rest_true': V_rest_true,
            'edge_index': edge_index,
            'embeddings': embeddings
        }
        print(f"Loaded Model {mid}: W shape {W_true.shape}, edge_index shape {edge_index.shape}")
    except Exception as e:
        print(f"Error loading Model {mid}: {e}")
        all_data[mid] = None

print()

# =============================================================================
# 1. THE PARADOX: Compare W errors vs tau/V_rest metrics
# =============================================================================
print("=" * 70)
print("1. THE PARADOX: W is wrong but tau/V_rest are right")
print("=" * 70)
print()

# From the metrics logs:
metrics = {
    '049': {'conn_R2': 0.124, 'tau_R2': 0.899, 'V_rest_R2': 0.666},
    '011': {'conn_R2': 0.681, 'tau_R2': 0.103, 'V_rest_R2': 0.052},
    '041': {'conn_R2': 0.911, 'tau_R2': 0.253, 'V_rest_R2': 0.010},
    '003': {'conn_R2': 0.965, 'tau_R2': 0.849, 'V_rest_R2': 0.614}
}

print("Metric Summary (from training logs):")
print("-" * 60)
print(f"{'Model':<8} {'conn_R2':<12} {'tau_R2':<12} {'V_rest_R2':<12} {'Paradox?':<10}")
print("-" * 60)
for mid, m in metrics.items():
    paradox = "YES" if m['conn_R2'] < 0.3 and m['tau_R2'] > 0.5 else "no"
    print(f"{mid:<8} {m['conn_R2']:<12.3f} {m['tau_R2']:<12.3f} {m['V_rest_R2']:<12.3f} {paradox:<10}")
print()

# =============================================================================
# 2. W_learned vs W_true: Detailed comparison for Model 049
# =============================================================================
print("=" * 70)
print("2. W_learned vs W_true: Model 049 Deep Dive")
print("=" * 70)
print()

if all_data['049'] is not None:
    W_true = all_data['049']['W_true']
    W_learned = all_data['049']['W_learned']
    edge_index = all_data['049']['edge_index']

    # Basic statistics
    print("Basic Statistics:")
    print(f"  n_edges: {len(W_true)}")
    print(f"  W_true:    mean={W_true.mean():.6f}, std={W_true.std():.6f}")
    print(f"  W_learned: mean={W_learned.mean():.6f}, std={W_learned.std():.6f}")
    print()

    # Correlation analysis
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    r2 = 1 - np.sum((W_true - W_learned)**2) / np.sum((W_true - W_true.mean())**2)
    print(f"Correlation Analysis:")
    print(f"  Pearson correlation: {pearson:.4f}")
    print(f"  R²: {r2:.4f}")
    print()

    # Sign analysis
    pos_true = W_true > 0.01
    neg_true = W_true < -0.01
    pos_learned = W_learned > 0.01
    neg_learned = W_learned < -0.01

    print(f"Sign Distribution:")
    print(f"  W_true:    {pos_true.sum():6d} positive, {neg_true.sum():6d} negative, {len(W_true)-pos_true.sum()-neg_true.sum():6d} near-zero")
    print(f"  W_learned: {pos_learned.sum():6d} positive, {neg_learned.sum():6d} negative, {len(W_true)-pos_learned.sum()-neg_learned.sum():6d} near-zero")
    print()

    # Check sign flipping
    sign_match_pos = (pos_true & pos_learned).sum()
    sign_flip_pos_to_neg = (pos_true & neg_learned).sum()
    sign_match_neg = (neg_true & neg_learned).sum()
    sign_flip_neg_to_pos = (neg_true & pos_learned).sum()

    print(f"Sign Flipping Analysis:")
    print(f"  Positive true -> Positive learned: {sign_match_pos:6d} ({100*sign_match_pos/pos_true.sum():.1f}%)")
    print(f"  Positive true -> Negative learned: {sign_flip_pos_to_neg:6d} ({100*sign_flip_pos_to_neg/pos_true.sum():.1f}%)")
    print(f"  Negative true -> Negative learned: {sign_match_neg:6d} ({100*sign_match_neg/neg_true.sum():.1f}%)")
    print(f"  Negative true -> Positive learned: {sign_flip_neg_to_pos:6d} ({100*sign_flip_neg_to_pos/neg_true.sum():.1f}%)")
    print()

    # Magnitude analysis
    print(f"Magnitude Distribution (W_true nonzero edges):")
    nonzero_mask = np.abs(W_true) > 0.01
    print(f"  W_true magnitude:    mean={np.abs(W_true[nonzero_mask]).mean():.4f}")
    print(f"  W_learned magnitude: mean={np.abs(W_learned[nonzero_mask]).mean():.4f}")
    print(f"  Ratio (learned/true): {np.abs(W_learned[nonzero_mask]).mean() / np.abs(W_true[nonzero_mask]).mean():.4f}")
    print()

# =============================================================================
# 3. EFFECTIVE CONNECTIVITY: Does W_learned produce similar aggregated messages?
# =============================================================================
print("=" * 70)
print("3. EFFECTIVE CONNECTIVITY Analysis")
print("=" * 70)
print("Testing if W_learned produces similar per-neuron input despite wrong edges")
print()

for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue

    W_true = all_data[mid]['W_true']
    W_learned = all_data[mid]['W_learned']
    edge_index = all_data[mid]['edge_index']
    n_neurons = edge_index.max() + 1

    # Compute per-neuron total incoming weight (sum of W over incoming edges)
    incoming_true = np.zeros(n_neurons)
    incoming_learned = np.zeros(n_neurons)

    for i in range(len(W_true)):
        target = edge_index[1, i]
        incoming_true[target] += W_true[i]
        incoming_learned[target] += W_learned[i]

    # Correlation of per-neuron incoming weights
    pearson_incoming = np.corrcoef(incoming_true, incoming_learned)[0, 1]
    r2_incoming = 1 - np.sum((incoming_true - incoming_learned)**2) / np.sum((incoming_true - incoming_true.mean())**2)

    # Also compute absolute incoming (sum of |W|)
    abs_incoming_true = np.zeros(n_neurons)
    abs_incoming_learned = np.zeros(n_neurons)
    for i in range(len(W_true)):
        target = edge_index[1, i]
        abs_incoming_true[target] += np.abs(W_true[i])
        abs_incoming_learned[target] += np.abs(W_learned[i])

    pearson_abs = np.corrcoef(abs_incoming_true, abs_incoming_learned)[0, 1]

    print(f"Model {mid}:")
    print(f"  Per-neuron incoming W (sum): Pearson={pearson_incoming:.4f}, R²={r2_incoming:.4f}")
    print(f"  Per-neuron incoming |W| (sum): Pearson={pearson_abs:.4f}")
    print()

# =============================================================================
# 4. Per-Neuron-Type Analysis: Which types are learned correctly?
# =============================================================================
print("=" * 70)
print("4. PER-NEURON-TYPE W Recovery")
print("=" * 70)
print()

try:
    import zarr

    for mid in ['049', '003']:  # Compare failing vs solved
        if all_data[mid] is None:
            continue

        data_dir = BASE_DATA.format(mid)
        metadata = zarr.open(f'{data_dir}/x_list_0/metadata.zarr', 'r')[:]
        neuron_types = metadata[:, 2].astype(int)

        edge_index = all_data[mid]['edge_index']
        W_true = all_data[mid]['W_true']
        W_learned = all_data[mid]['W_learned']

        # Source neuron type for each edge
        src_types = neuron_types[edge_index[0]]

        # Per-type R²
        unique_types = np.unique(src_types)
        type_r2 = {}
        for t in unique_types:
            mask = src_types == t
            if mask.sum() < 10:
                continue
            w_t = W_true[mask]
            w_l = W_learned[mask]
            ss_res = np.sum((w_t - w_l)**2)
            ss_tot = np.sum((w_t - w_t.mean())**2)
            if ss_tot > 0:
                type_r2[t] = 1 - ss_res / ss_tot

        # Sort by R²
        sorted_types = sorted(type_r2.items(), key=lambda x: x[1])

        print(f"Model {mid}: Worst 10 types by R²")
        for t, r2 in sorted_types[:10]:
            mask = src_types == t
            print(f"  Type {t:3d}: R²={r2:+.4f}, n_edges={mask.sum():5d}")
        print()

except Exception as e:
    print(f"Type analysis failed: {e}")
    print()

# =============================================================================
# 5. Tau and V_rest Analysis
# =============================================================================
print("=" * 70)
print("5. TAU and V_REST: How are they computed?")
print("=" * 70)
print()

print("In the FlyVis GNN, tau and V_rest are LEARNED parameters (not derived from W).")
print("The model learns them via the lin_phi MLP that maps to dv/dt.")
print()
print("This explains the paradox: tau/V_rest can be correct even if W is wrong,")
print("because they are learned INDEPENDENTLY from W.")
print()

# Check if tau/V_rest are stored in model
for mid, slot in zip(MODEL_IDS, SLOTS):
    if all_data[mid] is None:
        continue

    log_dir = BASE_LOG.format(slot)
    try:
        sd = torch.load(f'{log_dir}/models/best_model_with_0_graphs_0.pt',
                        map_location='cpu', weights_only=False)['model_state_dict']

        keys = list(sd.keys())
        print(f"Model {mid} state dict keys: {len(keys)} total")

        # Check for tau and V_rest
        has_tau = any('tau' in k.lower() for k in keys)
        has_vrest = any('v_rest' in k.lower() or 'vrest' in k.lower() for k in keys)

        if 'tau' in sd:
            print(f"  tau shape: {sd['tau'].shape}")
        if 'V_rest' in sd:
            print(f"  V_rest shape: {sd['V_rest'].shape}")

        # The parameters are typically in:
        # - lin_phi: node update MLP (learns tau implicitly)
        # - lin_edge: edge message MLP (learns W contribution)
        print(f"  lin_phi layers: {sum(1 for k in keys if 'lin_phi' in k)}")
        print(f"  lin_edge layers: {sum(1 for k in keys if 'lin_edge' in k)}")
    except Exception as e:
        print(f"Model {mid}: {e}")
    print()

# =============================================================================
# 6. Activity Rank vs Learnability
# =============================================================================
print("=" * 70)
print("6. ACTIVITY RANK vs Learnability")
print("=" * 70)
print()

# Known from generation logs:
activity_ranks = {
    '049': {'svd_99': 19, 'activity_99': 16},
    '011': {'svd_99': 45, 'activity_99': 26},
    '041': {'svd_99': 6, 'activity_99': 5},
    '003': {'svd_99': 60, 'activity_99': 35}
}

print("Model activity ranks (from generation logs):")
print(f"{'Model':<8} {'svd_99':<10} {'activity_99':<12} {'conn_R2':<12}")
print("-" * 50)
for mid, ranks in activity_ranks.items():
    conn_r2 = metrics[mid]['conn_R2']
    print(f"{mid:<8} {ranks['svd_99']:<10} {ranks['activity_99']:<12} {conn_r2:<12.3f}")
print()

print("Observation: Activity rank does NOT directly predict connectivity recovery!")
print("- Model 041 (svd_99=6, LOWEST) achieved conn_R2=0.911")
print("- Model 049 (svd_99=19, moderate) achieved conn_R2=0.124")
print("- Model 011 (svd_99=45, HIGH) achieved conn_R2=0.681")
print("- Model 003 (svd_99=60, HIGHEST) achieved conn_R2=0.965")
print()

# =============================================================================
# 7. Cross-Model W_true Comparison
# =============================================================================
print("=" * 70)
print("7. W_TRUE Structure Comparison")
print("=" * 70)
print("Are the W_true matrices structurally different across models?")
print()

for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue

    W_true = all_data[mid]['W_true']

    # Compute SVD rank of W_true reshaped as adjacency matrix
    # But W is already sparse edge list, not adjacency matrix
    # Just compare distributions

    print(f"Model {mid}:")
    print(f"  W_true: min={W_true.min():.4f}, max={W_true.max():.4f}")
    print(f"  W_true std: {W_true.std():.4f}")
    print(f"  Positive edges: {(W_true > 0.01).sum()}")
    print(f"  Negative edges: {(W_true < -0.01).sum()}")
    print(f"  Near-zero: {(np.abs(W_true) <= 0.01).sum()}")

    # Histogram of W_true magnitudes
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf]
    hist, _ = np.histogram(np.abs(W_true), bins=bins)
    print(f"  |W| histogram: {list(hist)}")
    print()

# =============================================================================
# 8. HYPOTHESIS: Model 049 has degenerate activity
# =============================================================================
print("=" * 70)
print("8. HYPOTHESIS: Degeneracy in Model 049")
print("=" * 70)
print()

print("""
The Model 049 paradox suggests DEGENERACY: multiple W configurations
can produce the same (or similar) dynamics given Model 049's activity pattern.

Why this might happen:
1. Low activity rank (svd_99=19) means the activity lies in a low-dimensional subspace
2. Many edges may be INACTIVE (source neurons with zero activity)
3. The effective rank of the W-activity product may be much lower than W's rank
4. This creates a degenerate optimization landscape where many W solutions are equivalent

To test this, we need to check:
1. How many edges have INACTIVE source neurons?
2. What is the rank of the W*activity product?
""")

try:
    import zarr

    for mid in ['049', '003']:
        if all_data[mid] is None:
            continue

        data_dir = BASE_DATA.format(mid)
        x_zarr = zarr.open(f'{data_dir}/x_list_0/timeseries.zarr', 'r')

        # Load activity for first 1000 frames
        activity = x_zarr[:1000, :, 0]  # [frames, neurons]

        # Compute activity variance per neuron
        activity_var = np.var(activity, axis=0)

        edge_index = all_data[mid]['edge_index']
        W_true = all_data[mid]['W_true']

        # Source activity variance for each edge
        src_var = activity_var[edge_index[0]]

        # How many edges have near-zero source activity?
        low_var_threshold = 1e-6
        inactive_edges = (src_var < low_var_threshold).sum()

        print(f"Model {mid}:")
        print(f"  Total edges: {len(W_true)}")
        print(f"  Edges with inactive source (var<{low_var_threshold}): {inactive_edges} ({100*inactive_edges/len(W_true):.1f}%)")

        # Weighted by W magnitude
        weighted_inactive = (np.abs(W_true) * (src_var < low_var_threshold)).sum()
        total_W_magnitude = np.abs(W_true).sum()
        print(f"  |W|-weighted inactive fraction: {100*weighted_inactive/total_W_magnitude:.1f}%")

        # Activity variance correlation with W_learning error
        W_learned = all_data[mid]['W_learned']
        edge_error = np.abs(W_true - W_learned)
        corr = np.corrcoef(src_var, edge_error)[0, 1]
        print(f"  Correlation(source_var, edge_error): {corr:.4f}")
        print()

except Exception as e:
    print(f"Degeneracy analysis failed: {e}")
    print()

# =============================================================================
# SUMMARY AND RECOMMENDATIONS
# =============================================================================
print("=" * 70)
print("SUMMARY: Understanding the Failures")
print("=" * 70)
print()

print("""
MODEL STATUS:
  049: FAILING (conn_R2=0.124) - PARADOX: tau/V_rest correct but W wrong
  011: PARTIAL (conn_R2=0.681) - lr_W=1E-3 helped, W_L1=3E-5 optimal
  041: CONVERGED (conn_R2=0.911) - V_rest fundamentally limited
  003: SOLVED (conn_R2=0.965) - edge_diff=900 optimal

KEY INSIGHT FOR MODEL 049:
  The paradox (correct tau/V_rest, wrong W) suggests that:
  1. tau and V_rest are learned INDEPENDENTLY from W
  2. Model 049's low-rank activity creates DEGENERACY in W
  3. Multiple W configurations can produce similar dynamics
  4. The optimizer finds a local minimum with wrong W but correct dynamics

NEXT STEPS FOR MODEL 049:
  1. Try MUCH stronger W_L1 regularization (1E-4 or 5E-4) to constrain W
  2. Try reducing hidden_dim to limit model capacity
  3. Try recurrent training to force W to matter for dynamics
  4. Accept that this model may have fundamental W degeneracy
""")

print()
print("=" * 70)
print("END OF ANALYSIS")
print("=" * 70)
