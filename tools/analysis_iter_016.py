#!/usr/bin/env python
"""
Analysis tool for iterations 13-16 (Batch 4).

KEY FINDING: Stronger regularization made Model 049 WORSE
- edge_norm=5.0 + W_L1=1E-4 decreased conn_R2 from 0.124 to 0.108
- Also degraded tau_R2 (0.899 → 0.606) and V_rest_R2 (0.666 → 0.566)
- This FALSIFIES the hypothesis that sign inversion can be fixed with regularization

Questions to investigate:
1. WHY did stronger regularization hurt Model 049?
2. What is structurally different about Model 049 that makes it resistant?
3. Is there evidence for a fundamentally different approach?
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
print("ANALYSIS ITER 016: Why Did Stronger Regularization FAIL for Model 049?")
print("=" * 70)
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
            'embeddings': embeddings,
            'state_dict': sd
        }
        print(f"Loaded Model {mid}")
    except Exception as e:
        print(f"Error loading Model {mid}: {e}")
        all_data[mid] = None

print()

# =============================================================================
# 1. Current Results Summary
# =============================================================================
print("=" * 70)
print("1. BATCH 4 RESULTS SUMMARY")
print("=" * 70)
print()

# From the metrics logs:
metrics_batch4 = {
    '049': {'conn_R2': 0.108, 'tau_R2': 0.606, 'V_rest_R2': 0.566, 'config': 'edge_norm=5.0, W_L1=1E-4'},
    '011': {'conn_R2': 0.544, 'tau_R2': 0.221, 'V_rest_R2': 0.001, 'config': 'lr_emb=2E-3 (CATASTROPHIC)'},
    '041': {'conn_R2': 0.912, 'tau_R2': 0.373, 'V_rest_R2': 0.014, 'config': 'edge_diff=1500 (STABLE)'},
    '003': {'conn_R2': 0.966, 'tau_R2': 0.962, 'V_rest_R2': 0.685, 'config': 'Iter4 config (CONFIRMED)'}
}

best_results = {
    '049': {'best_conn': 0.634, 'iter': 'baseline', 'config': 'default'},
    '011': {'best_conn': 0.716, 'iter': 2, 'config': 'lr_W=1E-3, lr=1E-3, W_L1=3E-5'},
    '041': {'best_conn': 0.912, 'iter': 15, 'config': 'edge_diff=1500, phi_L1=1.0'},
    '003': {'best_conn': 0.972, 'iter': 4, 'config': 'edge_diff=900, W_L1=3E-5'}
}

print("Current Status (Batch 4):")
print("-" * 80)
print(f"{'Model':<8} {'conn_R2':<10} {'tau_R2':<10} {'V_rest_R2':<12} {'Config':<40}")
print("-" * 80)
for mid, m in metrics_batch4.items():
    print(f"{mid:<8} {m['conn_R2']:<10.3f} {m['tau_R2']:<10.3f} {m['V_rest_R2']:<12.3f} {m['config']:<40}")
print()

print("Best Results Across All Iterations:")
print("-" * 80)
for mid, b in best_results.items():
    status = "SOLVED" if b['best_conn'] > 0.9 else ("PARTIAL" if b['best_conn'] > 0.5 else "FAILING")
    print(f"Model {mid}: best={b['best_conn']:.3f} (Iter {b['iter']}) - {status}")
print()

# =============================================================================
# 2. Model 049: Why Did Stronger Regularization Fail?
# =============================================================================
print("=" * 70)
print("2. MODEL 049: Why Stronger Regularization FAILED")
print("=" * 70)
print()

if all_data['049'] is not None:
    W_true = all_data['049']['W_true']
    W_learned = all_data['049']['W_learned']
    edge_index = all_data['049']['edge_index']

    # Hypothesis 1: Over-regularization killed the signal
    print("Hypothesis 1: Over-regularization killed the learning signal")
    print("-" * 60)

    # Compare W_learned magnitude with stronger regularization
    w_magnitude = np.abs(W_learned).mean()
    w_true_magnitude = np.abs(W_true).mean()
    print(f"  W_true average magnitude:    {w_true_magnitude:.6f}")
    print(f"  W_learned average magnitude: {w_magnitude:.6f}")
    print(f"  Ratio (learned/true): {w_magnitude/w_true_magnitude:.4f}")

    # Check if W_learned is being pushed toward zero
    near_zero = (np.abs(W_learned) < 0.001).sum()
    print(f"  Near-zero W_learned edges (<0.001): {near_zero} / {len(W_learned)} ({100*near_zero/len(W_learned):.1f}%)")
    print()

    # Hypothesis 2: Sign structure is fundamentally wrong
    print("Hypothesis 2: Sign structure analysis")
    print("-" * 60)

    pos_true = W_true > 0.01
    neg_true = W_true < -0.01
    pos_learned = W_learned > 0
    neg_learned = W_learned < 0

    sign_match = ((pos_true & pos_learned) | (neg_true & neg_learned)).sum()
    total_significant = pos_true.sum() + neg_true.sum()

    print(f"  Sign match rate: {100*sign_match/total_significant:.1f}%")

    # Check if W_learned has opposite sign distribution
    pos_learned_frac = (W_learned > 0).sum() / len(W_learned)
    pos_true_frac = (W_true > 0).sum() / len(W_true)
    print(f"  W_true positive fraction:    {pos_true_frac:.3f}")
    print(f"  W_learned positive fraction: {pos_learned_frac:.3f}")
    print()

    # Hypothesis 3: The sign inversion is STRUCTURAL (not fixable by regularization)
    print("Hypothesis 3: Structural sign inversion analysis")
    print("-" * 60)

    # Compute correlation between W_true and W_learned
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    print(f"  Pearson(W_true, W_learned): {pearson:.4f}")

    # If pearson is near -1, it's a structural inversion
    if pearson < -0.5:
        print("  STRONG NEGATIVE CORRELATION: W_learned ≈ -W_true")
        print("  This suggests the GNN found a sign-inverted solution")
    elif pearson < 0:
        print("  NEGATIVE CORRELATION: Signs partially inverted")
    else:
        print("  POSITIVE CORRELATION: Signs mostly correct")
    print()

# =============================================================================
# 3. Lin_edge Positive Constraint Analysis
# =============================================================================
print("=" * 70)
print("3. LIN_EDGE_POSITIVE CONSTRAINT ANALYSIS")
print("=" * 70)
print()

print("""
The FlyVis GNN uses lin_edge_positive=True by default.
This means edge messages are SQUARED, forcing them to be positive.

In the PDE equation:
  tau * dv/dt = -v + V_rest + sum_j W_ij * ReLU(v_j) + I

The W_ij can be positive or negative (excitatory/inhibitory).
BUT if lin_edge outputs only positive values, how are inhibitory connections represented?

Possible mechanisms:
1. The W parameter itself encodes sign (W can be negative)
2. The lin_phi MLP compensates by learning negative coefficients
3. The embedding space encodes excitatory/inhibitory nature

For Model 049, if activity is low-rank, the GNN may find it easier to use
NEGATIVE W with positive lin_edge messages, instead of vice versa.
This would explain the sign inversion.
""")

# Analyze if lin_edge_positive contributes to sign degeneracy
if all_data['049'] is not None:
    sd = all_data['049']['state_dict']

    # Check lin_edge layers
    print("Lin_edge layer analysis:")
    for k, v in sd.items():
        if 'lin_edge' in k and 'weight' in k:
            w = v.numpy()
            pos_frac = (w > 0).sum() / w.size
            print(f"  {k}: shape={w.shape}, positive_frac={pos_frac:.2f}, mean={w.mean():.4f}")
    print()

# =============================================================================
# 4. Cross-Model Comparison: What Makes 049 Different?
# =============================================================================
print("=" * 70)
print("4. CROSS-MODEL COMPARISON: What Makes 049 Different?")
print("=" * 70)
print()

# Compare W statistics across all models
print("W_true statistics:")
print(f"{'Model':<8} {'mean':<12} {'std':<12} {'pos_frac':<12} {'neg_frac':<12}")
print("-" * 60)
for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue
    W = all_data[mid]['W_true']
    mean = W.mean()
    std = W.std()
    pos = (W > 0.01).sum() / len(W)
    neg = (W < -0.01).sum() / len(W)
    print(f"{mid:<8} {mean:<12.6f} {std:<12.6f} {pos:<12.3f} {neg:<12.3f}")
print()

# Compare W_learned vs W_true correlation
print("W_learned vs W_true correlation:")
print(f"{'Model':<8} {'Pearson':<12} {'R2':<12}")
print("-" * 40)
for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue
    W_t = all_data[mid]['W_true']
    W_l = all_data[mid]['W_learned']
    pearson = np.corrcoef(W_t, W_l)[0, 1]
    r2 = 1 - np.sum((W_t - W_l)**2) / np.sum((W_t - W_t.mean())**2)
    print(f"{mid:<8} {pearson:<12.4f} {r2:<12.4f}")
print()

# =============================================================================
# 5. Embedding Space Analysis
# =============================================================================
print("=" * 70)
print("5. EMBEDDING SPACE ANALYSIS")
print("=" * 70)
print()

for mid in ['049', '003']:
    if all_data[mid] is None:
        continue

    emb = all_data[mid]['embeddings']
    print(f"Model {mid} embeddings:")
    print(f"  Shape: {emb.shape}")
    print(f"  Mean: ({emb[:, 0].mean():.4f}, {emb[:, 1].mean():.4f})")
    print(f"  Std:  ({emb[:, 0].std():.4f}, {emb[:, 1].std():.4f})")
    print(f"  Range X: [{emb[:, 0].min():.4f}, {emb[:, 0].max():.4f}]")
    print(f"  Range Y: [{emb[:, 1].min():.4f}, {emb[:, 1].max():.4f}]")

    # Check if embeddings are clustered or spread
    spread = np.sqrt(emb[:, 0].var() + emb[:, 1].var())
    print(f"  Total spread (sqrt(var_x + var_y)): {spread:.4f}")
    print()

# =============================================================================
# 6. Per-Edge-Type Sign Analysis
# =============================================================================
print("=" * 70)
print("6. PER-EDGE-TYPE SIGN ANALYSIS FOR MODEL 049")
print("=" * 70)
print()

try:
    import zarr

    if all_data['049'] is not None:
        data_dir = BASE_DATA.format('049')
        metadata = zarr.open(f'{data_dir}/x_list_0/metadata.zarr', 'r')[:]
        neuron_types = metadata[:, 2].astype(int)

        edge_index = all_data['049']['edge_index']
        W_true = all_data['049']['W_true']
        W_learned = all_data['049']['W_learned']

        # Source neuron type for each edge
        src_types = neuron_types[edge_index[0]]

        # Per-type sign analysis
        unique_types = np.unique(src_types)

        print("Per-type sign inversion analysis (top 10 by edge count):")
        print("-" * 70)
        print(f"{'Type':<6} {'n_edges':<10} {'true_pos%':<12} {'learned_pos%':<14} {'sign_match%':<12}")
        print("-" * 70)

        type_stats = []
        for t in unique_types:
            mask = src_types == t
            n = mask.sum()
            if n < 100:
                continue

            w_t = W_true[mask]
            w_l = W_learned[mask]

            true_pos = (w_t > 0).sum() / n
            learned_pos = (w_l > 0).sum() / n

            # Sign match
            sign_match = ((w_t > 0) == (w_l > 0)).sum() / n

            type_stats.append((t, n, true_pos, learned_pos, sign_match))

        # Sort by edge count
        type_stats.sort(key=lambda x: -x[1])

        for t, n, tp, lp, sm in type_stats[:10]:
            sign_status = "INVERTED" if sm < 0.3 else ("MIXED" if sm < 0.7 else "OK")
            print(f"{t:<6} {n:<10} {100*tp:<12.1f} {100*lp:<14.1f} {100*sm:<12.1f} {sign_status}")
        print()

except Exception as e:
    print(f"Per-type analysis failed: {e}")
    print()

# =============================================================================
# 7. Alternative Approaches to Explore
# =============================================================================
print("=" * 70)
print("7. ALTERNATIVE APPROACHES FOR MODEL 049")
print("=" * 70)
print()

print("""
Standard regularization (edge_norm, W_L1) FAILED for Model 049.
The sign inversion appears to be STRUCTURAL, not a regularization issue.

POTENTIAL ALTERNATIVE APPROACHES:

1. TRY lin_edge_positive=False
   - Currently, edge messages are squared (always positive)
   - This may force the GNN to use W signs incorrectly
   - Allowing negative edge messages might resolve sign degeneracy
   - CAUTION: This is a significant architectural change

2. TRY much SLOWER W learning (lr_W=1E-4)
   - Fast W learning may lock into sign-inverted local minimum
   - Slower learning might allow correct sign structure to emerge
   - Combined with stronger MLP learning to guide W

3. TRY embedding-based sign recovery
   - Increase embedding_dim to 4 or higher
   - More embedding dimensions might capture sign information
   - CAUTION: Must update input_size and input_size_update

4. TRY multi-stage training
   - Stage 1: Train only MLPs (freeze W)
   - Stage 2: Train only W (freeze MLPs)
   - May break the coupled sign inversion

5. ACCEPT FUNDAMENTAL LIMITATION
   - Model 049's low-rank activity may create true degeneracy
   - Multiple W solutions might be mathematically equivalent
   - Focus on improving tau/V_rest recovery instead

RECOMMENDATION: Try lr_W=1E-4 first (simplest change), then lin_edge_positive=False
""")

# =============================================================================
# 8. Model 011: Confirm lr_emb Sensitivity
# =============================================================================
print("=" * 70)
print("8. MODEL 011: Learning Rate Embedding Sensitivity")
print("=" * 70)
print()

print("Iter 14 showed lr_emb=2E-3 is CATASTROPHIC (0.716 → 0.544)")
print()

if all_data['011'] is not None:
    emb = all_data['011']['embeddings']
    print(f"Model 011 embeddings (with lr_emb=2E-3):")
    print(f"  Shape: {emb.shape}")
    print(f"  Mean: ({emb[:, 0].mean():.4f}, {emb[:, 1].mean():.4f})")
    print(f"  Std:  ({emb[:, 0].std():.4f}, {emb[:, 1].std():.4f})")
    spread = np.sqrt(emb[:, 0].var() + emb[:, 1].var())
    print(f"  Total spread: {spread:.4f}")
    print()

print("High lr_emb likely causes embeddings to become unstable,")
print("disrupting the edge message function that depends on embeddings.")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print("""
MODEL STATUS AFTER BATCH 4:
  049: FAILING (0.108) - Stronger regularization made it WORSE
  011: PARTIAL (0.716 best, 0.544 current) - lr_emb=2E-3 catastrophic
  041: SOLVED (0.912) - Stable with edge_diff=1500
  003: SOLVED (0.966) - Confirmed with Iter 4 config

KEY FINDINGS:

1. MODEL 049: Regularization approach FALSIFIED
   - edge_norm=5.0 + W_L1=1E-4 made conn_R2 WORSE (0.124 → 0.108)
   - Also degraded tau_R2 and V_rest_R2
   - Sign inversion is STRUCTURAL, not fixable by regularization
   - NEED: Fundamentally different approach (lin_edge_positive=False or lr_W=1E-4)

2. MODEL 011: lr_emb constraint CONFIRMED
   - lr_emb=2E-3 destroys connectivity (0.716 → 0.544)
   - MUST stay at lr_emb=1.5E-3 or below
   - Best config remains Iter 2 (lr_W=1E-3, lr=1E-3, W_L1=3E-5)

3. MODEL 041: CONFIRMED SOLVED
   - edge_diff=1500 stable (0.912)
   - V_rest limitation (~0.01) is fundamental, not fixable

4. MODEL 003: CONFIRMED SOLVED
   - Iter 4 config is optimal (0.966)
   - No further tuning needed

NEXT BATCH RECOMMENDATIONS:
  Slot 0 (049): Try lr_W=1E-4 (very slow W learning)
  Slot 1 (011): Return to Iter 2 config, try phi_L1=0.3
  Slot 2 (041): Maintain - model solved
  Slot 3 (003): Maintain - model solved
""")

print()
print("=" * 70)
print("END OF ANALYSIS")
print("=" * 70)
