#!/usr/bin/env python3
"""
Analysis tool for Understanding Exploration Batch 1 (Iters 1-4)

KEY QUESTIONS TO ANSWER:
1. WHY did Model 049 regress from 0.634 to 0.141 with data_aug=25?
2. WHY do Models 011 and 041 have V_rest collapse while 003 doesn't?
3. What structural differences in W_true explain the different responses?
4. Are there specific neuron types that are systematically hard across models?
"""

import torch
import numpy as np
import os
from scipy import stats

# Configuration
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
DATA_DIR = 'graphs_data/fly'
LOG_DIR = 'log/fly'
BASELINES = {'049': 0.634, '011': 0.308, '041': 0.629, '003': 0.627}

print("=" * 70)
print("ANALYSIS TOOL: Batch 1 (Iters 1-4) - Understanding Difficult Models")
print("=" * 70)

# ============================================================================
# 1. Compute Connectivity R² and Compare to Baseline
# ============================================================================
print("\n=== CONNECTIVITY R² ANALYSIS ===\n")

connectivity_r2 = {}
w_true_data = {}
w_learned_data = {}

for mid, slot in zip(MODEL_IDS, SLOTS):
    dataset_dir = f'{DATA_DIR}/flyvis_62_1_id_{mid}'
    log_dir = f'{LOG_DIR}/flyvis_62_1_understand_Claude_{slot:02d}'

    W_true = torch.load(f'{dataset_dir}/weights.pt', weights_only=True, map_location='cpu').numpy()
    w_true_data[mid] = W_true

    model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
        W_learned = state_dict['model_state_dict']['W'].numpy().flatten()
        w_learned_data[mid] = W_learned

        ss_res = np.sum((W_true - W_learned) ** 2)
        ss_tot = np.sum((W_true - W_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        connectivity_r2[mid] = r2

        change = r2 - BASELINES[mid]
        status = "IMPROVED" if change > 0.05 else ("REGRESSED" if change < -0.05 else "SIMILAR")
        print(f"Model {mid}: R²={r2:.4f} (baseline={BASELINES[mid]:.4f}, change={change:+.4f}) [{status}]")
    else:
        print(f"Model {mid}: No trained model found")

# ============================================================================
# 2. W_TRUE Structure Analysis - Why are these models different?
# ============================================================================
print("\n\n=== W_TRUE STRUCTURE COMPARISON ===")
print("Looking for structural differences that explain model difficulty...\n")

print(f"{'Model':<8} {'Nonzero%':<10} {'Mean':<12} {'Std':<12} {'|W|<0.01%':<12} {'|W|>0.1%':<12} {'SVD_rank99':<12}")
print("-" * 80)

svd_ranks = {'049': 19, '011': 45, '041': 6, '003': 60}

for mid in MODEL_IDS:
    W = w_true_data[mid]
    nnz_pct = 100 * np.count_nonzero(W) / len(W)
    weak_pct = 100 * (np.abs(W) < 0.01).sum() / len(W)
    strong_pct = 100 * (np.abs(W) >= 0.1).sum() / len(W)
    print(f"{mid:<8} {nnz_pct:<10.2f} {W.mean():<12.6f} {W.std():<12.6f} {weak_pct:<12.1f} {strong_pct:<12.1f} {svd_ranks[mid]:<12}")

# ============================================================================
# 3. Investigate Model 049 Regression
# ============================================================================
print("\n\n=== MODEL 049 REGRESSION INVESTIGATION ===")
print("Why did data_aug=25 cause regression from 0.634 to 0.141?\n")

if '049' in w_learned_data:
    W_true = w_true_data['049']
    W_learned = w_learned_data['049']

    # Check for systematic bias
    print("W_learned statistics:")
    print(f"  Mean: {W_learned.mean():.6f} (true: {W_true.mean():.6f})")
    print(f"  Std:  {W_learned.std():.6f} (true: {W_true.std():.6f})")
    print(f"  Min:  {W_learned.min():.6f} (true: {W_true.min():.6f})")
    print(f"  Max:  {W_learned.max():.6f} (true: {W_true.max():.6f})")

    # Check if learned weights are collapsed
    learned_var = np.var(W_learned)
    true_var = np.var(W_true)
    print(f"  Variance ratio (learned/true): {learned_var/true_var:.4f}")

    # Check correlation
    corr, _ = stats.pearsonr(W_true, W_learned)
    print(f"  Pearson correlation: {corr:.4f}")

    # Error distribution by weight magnitude
    error = W_learned - W_true
    abs_W = np.abs(W_true)

    print("\nError by weight magnitude:")
    for lo, hi, label in [(0, 0.01, '|W|<0.01'), (0.01, 0.1, '0.01≤|W|<0.1'), (0.1, np.inf, '|W|≥0.1')]:
        mask = (abs_W >= lo) & (abs_W < hi)
        if mask.sum() > 0:
            print(f"  {label}: n={mask.sum():7d}, mean_error={error[mask].mean():+.6f}, std_error={error[mask].std():.6f}")

# ============================================================================
# 4. V_rest Collapse Investigation
# ============================================================================
print("\n\n=== V_REST COLLAPSE INVESTIGATION ===")
print("Why do Models 011/041 have V_rest collapse while 003 doesn't?\n")

for mid, slot in zip(MODEL_IDS, SLOTS):
    dataset_dir = f'{DATA_DIR}/flyvis_62_1_id_{mid}'
    log_dir = f'{LOG_DIR}/flyvis_62_1_understand_Claude_{slot:02d}'

    # Load true V_rest
    v_rest_path = f'{dataset_dir}/V_i_rest.pt'
    model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'

    if not os.path.exists(v_rest_path) or not os.path.exists(model_path):
        continue

    V_rest_true = torch.load(v_rest_path, weights_only=True, map_location='cpu').numpy()

    # V_rest is encoded in embeddings or in the model differently - check state dict keys
    state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
    model_sd = state_dict['model_state_dict']

    print(f"Model {mid}:")
    print(f"  V_rest_true: mean={V_rest_true.mean():.4f}, std={V_rest_true.std():.4f}, range=[{V_rest_true.min():.4f}, {V_rest_true.max():.4f}]")

    # Check embeddings
    embeddings = model_sd['a'].numpy()
    print(f"  Embeddings: shape={embeddings.shape}, mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    print(f"  Embedding dim 0: mean={embeddings[:,0].mean():.4f}, std={embeddings[:,0].std():.4f}")
    print(f"  Embedding dim 1: mean={embeddings[:,1].mean():.4f}, std={embeddings[:,1].std():.4f}")

    # Check for NaN/Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    if has_nan or has_inf:
        print(f"  WARNING: Has NaN={has_nan}, Has Inf={has_inf}")
    print()

# ============================================================================
# 5. SVD Analysis of W_true
# ============================================================================
print("\n=== SVD ANALYSIS OF W_TRUE ===")
print("Examining spectral structure of connectivity matrices...\n")

for mid in MODEL_IDS:
    W = w_true_data[mid]

    # Reshape to matrix form using edge_index
    edge_index_path = f'{DATA_DIR}/flyvis_62_1_id_{mid}/edge_index.pt'
    if os.path.exists(edge_index_path):
        edge_index = torch.load(edge_index_path, weights_only=True, map_location='cpu').numpy()

        # Get number of neurons
        n_neurons = max(edge_index.max() + 1, 13741)

        # Create sparse adjacency matrix
        W_matrix = np.zeros((n_neurons, n_neurons))
        W_matrix[edge_index[1], edge_index[0]] = W  # target, source

        # Compute SVD on a sample (full SVD is too expensive)
        # Use randomized SVD approximation
        try:
            from scipy.sparse.linalg import svds
            from scipy.sparse import csr_matrix

            W_sparse = csr_matrix(W_matrix)
            k = min(100, min(W_matrix.shape) - 1)
            U, s, Vt = svds(W_sparse, k=k)

            # Sort singular values in descending order
            s = s[::-1]

            # Compute cumulative variance explained
            total_var = (s**2).sum()
            cumvar = np.cumsum(s**2) / total_var

            rank_90 = np.searchsorted(cumvar, 0.90) + 1
            rank_99 = np.searchsorted(cumvar, 0.99) + 1

            print(f"Model {mid}:")
            print(f"  Top 5 singular values: {s[:5]}")
            print(f"  SVD rank at 90% var: {rank_90}")
            print(f"  SVD rank at 99% var: {rank_99}")
            print(f"  Condition number (s[0]/s[k-1]): {s[0]/s[-1]:.2f}")
            print()
        except Exception as e:
            print(f"Model {mid}: SVD failed - {e}")

# ============================================================================
# 6. Per-Type Recovery Analysis (brief)
# ============================================================================
print("\n=== PER-TYPE RECOVERY COMPARISON ===")
print("Are the same neuron types hard across all models?\n")

type_r2_by_model = {}

for mid, slot in zip(MODEL_IDS, SLOTS):
    if mid not in w_learned_data:
        continue

    dataset_dir = f'{DATA_DIR}/flyvis_62_1_id_{mid}'
    edge_index = torch.load(f'{dataset_dir}/edge_index.pt', weights_only=True, map_location='cpu').numpy()

    try:
        import zarr
        metadata = zarr.open(f'{dataset_dir}/x_list_0/metadata.zarr', 'r')[:]
        neuron_types = metadata[:, 2].astype(int)
    except:
        continue

    W_true = w_true_data[mid]
    W_learned = w_learned_data[mid]
    source_types = neuron_types[edge_index[0]]

    type_r2 = {}
    for t in np.unique(source_types):
        mask = source_types == t
        if mask.sum() < 100:
            continue
        w_t = W_true[mask]
        w_l = W_learned[mask]
        ss_res = np.sum((w_t - w_l) ** 2)
        ss_tot = np.sum((w_t - w_t.mean()) ** 2)
        if ss_tot > 1e-10:
            type_r2[t] = 1 - ss_res / ss_tot

    type_r2_by_model[mid] = type_r2

    # Find hardest types
    sorted_types = sorted(type_r2.items(), key=lambda x: x[1])
    print(f"Model {mid} - 3 hardest types: {[(t, f'{r:.3f}') for t, r in sorted_types[:3]]}")

# Check if same types are hard across models
if len(type_r2_by_model) >= 2:
    print("\nCross-model type difficulty correlation:")
    models = list(type_r2_by_model.keys())
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            common_types = set(type_r2_by_model[m1].keys()) & set(type_r2_by_model[m2].keys())
            if len(common_types) > 5:
                r2_1 = [type_r2_by_model[m1][t] for t in common_types]
                r2_2 = [type_r2_by_model[m2][t] for t in common_types]
                corr, _ = stats.pearsonr(r2_1, r2_2)
                print(f"  {m1} vs {m2}: correlation={corr:.3f} (n={len(common_types)} types)")

# ============================================================================
# 7. Summary and Hypotheses Update
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: INSIGHTS FOR HYPOTHESIS REFINEMENT")
print("=" * 70)

print("\n1. MODEL 049 REGRESSION:")
print("   - data_aug=25 caused R² drop from 0.634 to 0.141")
print("   - Hypothesis: Low-rank activity (svd_99=19) + high augmentation = signal dilution")
print("   - RECOMMENDATION: Try data_aug=15 or lr_W=1E-3 approach from Model 011")

print("\n2. V_REST COLLAPSE PATTERN:")
print("   - Models 011 (V_rest=0.004) and 041 (V_rest=0.0001) collapsed")
print("   - Model 003 (V_rest=0.725) did NOT collapse")
print("   - Difference: Model 003 used edge_diff=900 (vs 750 for others)")
print("   - Hypothesis: Higher edge_diff regularizes the embedding-V_rest interaction")

print("\n3. WHAT WORKED:")
print("   - Model 003: edge_diff=900 + W_L1=3E-5 -> R²=0.972 (excellent)")
print("   - Model 041: hidden_dim=64 + data_aug=30 -> R²=0.907 (for connectivity)")
print("   - Model 011: lr_W=1E-3 + lr=1E-3 + W_L1=3E-5 -> R²=0.716")

print("\n4. NEXT BATCH RECOMMENDATIONS:")
print("   - Slot 0 (049): Try lr_W=1E-3+lr=1E-3 (Model 011 approach) OR data_aug=15")
print("   - Slot 1 (011): Add edge_diff=900 to recover V_rest while keeping conn_R2")
print("   - Slot 2 (041): Add edge_diff=900 to try to recover V_rest")
print("   - Slot 3 (003): Minor tuning or try to push R² higher with edge_diff=1000")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)
