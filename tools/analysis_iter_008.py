#!/usr/bin/env python
"""
Analysis tool for iterations 5-8 (Batch 2).

Key questions:
1. WHY does Model 049 keep failing? Is there a structural issue in W_true or activity?
2. Why does lr_W=1E-3 work for Model 011 but fail for Model 049?
3. Why does edge_diff=900 help Model 003's V_rest but hurt Model 011's connectivity?
4. What differentiates the V_rest-collapsed models (011, 041) from the recovered model (003)?
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
print("ANALYSIS ITER 008: Why Model 049 Keeps Failing")
print("=" * 70)

# Load all data
all_data = {}
for mid, slot in zip(MODEL_IDS, SLOTS):
    data_dir = BASE_DATA.format(mid)
    log_dir = BASE_LOG.format(slot)

    try:
        W_true = torch.load(f'{data_dir}/weights.pt', weights_only=True).numpy()
        V_rest_true = torch.load(f'{data_dir}/V_i_rest.pt', weights_only=True).numpy()
        tau_true = torch.load(f'{data_dir}/taus.pt', weights_only=True).numpy()
        edge_index = torch.load(f'{data_dir}/edge_index.pt', weights_only=True).numpy()

        model_path = f'{log_dir}/models/best_model_with_0_graphs_0.pt'
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        W_learned = sd['model_state_dict']['W'].numpy().flatten()
        embeddings = sd['model_state_dict']['a'].numpy()

        all_data[mid] = {
            'W_true': W_true,
            'W_learned': W_learned,
            'V_rest_true': V_rest_true,
            'tau_true': tau_true,
            'edge_index': edge_index,
            'embeddings': embeddings
        }
    except Exception as e:
        print(f"Error loading Model {mid}: {e}")
        all_data[mid] = None

print("\n=== 1. W_true vs W_learned: Sign Analysis ===")
print("Checking if learned W has correct SIGN structure...\n")

for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue
    W_true = all_data[mid]['W_true']
    W_learned = all_data[mid]['W_learned']

    # Pearson correlation
    pearson = np.corrcoef(W_true, W_learned)[0, 1]

    # Sign agreement (for nonzero W_true)
    nonzero_mask = np.abs(W_true) > 0.001
    if nonzero_mask.sum() > 0:
        sign_true = np.sign(W_true[nonzero_mask])
        sign_learned = np.sign(W_learned[nonzero_mask])
        sign_agreement = (sign_true == sign_learned).mean()
    else:
        sign_agreement = 0

    # Magnitude correlation for positive and negative edges separately
    pos_mask = W_true > 0.01
    neg_mask = W_true < -0.01

    if pos_mask.sum() > 10:
        pos_pearson = np.corrcoef(W_true[pos_mask], W_learned[pos_mask])[0, 1]
    else:
        pos_pearson = np.nan

    if neg_mask.sum() > 10:
        neg_pearson = np.corrcoef(W_true[neg_mask], W_learned[neg_mask])[0, 1]
    else:
        neg_pearson = np.nan

    # R² calculation
    r2 = 1 - np.sum((W_true - W_learned)**2) / np.sum((W_true - W_true.mean())**2)

    print(f"Model {mid}:")
    print(f"  Overall Pearson: {pearson:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Sign agreement (|W|>0.001): {sign_agreement:.4f} ({nonzero_mask.sum()} edges)")
    print(f"  Pearson for POSITIVE edges (W>0.01): {pos_pearson:.4f} ({pos_mask.sum()} edges)")
    print(f"  Pearson for NEGATIVE edges (W<-0.01): {neg_pearson:.4f} ({neg_mask.sum()} edges)")
    print()

print("\n=== 2. W_learned Statistics: Range and Distribution ===")
print("Comparing learned W ranges across models...\n")

for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue
    W_learned = all_data[mid]['W_learned']
    W_true = all_data[mid]['W_true']

    print(f"Model {mid}:")
    print(f"  W_true:    min={W_true.min():.4f}, max={W_true.max():.4f}, std={W_true.std():.4f}")
    print(f"  W_learned: min={W_learned.min():.4f}, max={W_learned.max():.4f}, std={W_learned.std():.4f}")
    print(f"  Learned/True std ratio: {W_learned.std() / W_true.std():.4f}")
    print()

print("\n=== 3. Activity Correlation with W_true ===")
print("Testing if activity structure affects edge learnability...\n")

# Load activity traces for model 049 vs 003 comparison
try:
    import zarr
    for mid in ['049', '003']:  # Compare failing vs solved
        data_dir = BASE_DATA.format(mid)
        x_zarr = zarr.open(f'{data_dir}/x_list_0/timeseries.zarr', 'r')
        # Get sample activity (first 1000 frames)
        activity = x_zarr[:1000, :, 0]  # [frames, neurons]

        # Activity variance per neuron
        activity_var = activity.var(axis=0)

        # Edge source/target activity variance
        edge_index = all_data[mid]['edge_index']
        W_true = all_data[mid]['W_true']
        W_learned = all_data[mid]['W_learned']

        src_var = activity_var[edge_index[0]]
        tgt_var = activity_var[edge_index[1]]

        # Edge error
        edge_error = np.abs(W_true - W_learned)

        # Correlation between edge error and source activity variance
        corr_src = np.corrcoef(edge_error, src_var)[0, 1]
        corr_tgt = np.corrcoef(edge_error, tgt_var)[0, 1]

        print(f"Model {mid}:")
        print(f"  Activity variance: min={activity_var.min():.6f}, max={activity_var.max():.6f}")
        print(f"  Correlation(edge_error, source_activity_var): {corr_src:.4f}")
        print(f"  Correlation(edge_error, target_activity_var): {corr_tgt:.4f}")
        print()
except Exception as e:
    print(f"  Activity analysis skipped: {e}")

print("\n=== 4. V_rest Analysis: Why Some Models Collapse ===")
print("Comparing V_rest recovery across models...\n")

# Load V_rest from trained models if available
for mid, slot in zip(MODEL_IDS, SLOTS):
    if all_data[mid] is None:
        continue

    log_dir = BASE_LOG.format(slot)
    V_rest_true = all_data[mid]['V_rest_true']

    # Try to extract V_rest from embedding or other sources
    # The embedding 'a' is 2D, but V_rest is not directly stored
    # We need to look at the model's V_rest parameters if they exist

    try:
        sd = torch.load(f'{log_dir}/models/best_model_with_0_graphs_0.pt',
                        map_location='cpu', weights_only=False)['model_state_dict']

        # Check if V_rest is in the state dict
        if 'V_rest' in sd:
            V_rest_learned = sd['V_rest'].numpy().flatten()
            r2 = 1 - np.sum((V_rest_true - V_rest_learned)**2) / np.sum((V_rest_true - V_rest_true.mean())**2)
            pearson = np.corrcoef(V_rest_true, V_rest_learned)[0, 1]
            print(f"Model {mid}: V_rest R²={r2:.4f}, Pearson={pearson:.4f}")
        else:
            # V_rest may be computed from embeddings
            print(f"Model {mid}: V_rest not directly stored in model state dict")
    except Exception as e:
        print(f"Model {mid}: Could not analyze V_rest ({e})")

print("\n=== 5. Embedding Analysis: Per-Model Structure ===")
print("Comparing embedding distributions across models...\n")

for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue
    emb = all_data[mid]['embeddings']

    print(f"Model {mid}:")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  Dim 0: mean={emb[:, 0].mean():.4f}, std={emb[:, 0].std():.4f}")
    print(f"  Dim 1: mean={emb[:, 1].mean():.4f}, std={emb[:, 1].std():.4f}")
    print(f"  Embedding norm: mean={np.linalg.norm(emb, axis=1).mean():.4f}, std={np.linalg.norm(emb, axis=1).std():.4f}")
    print()

print("\n=== 6. Per-Neuron-Type W Recovery (Top 10 Hardest Types) ===")
print("Which neuron types are hardest to recover for each model?\n")

# Load metadata for neuron types
for mid in MODEL_IDS:
    if all_data[mid] is None:
        continue

    data_dir = BASE_DATA.format(mid)
    try:
        import zarr
        metadata = zarr.open(f'{data_dir}/x_list_0/metadata.zarr', 'r')[:]
        neuron_types = metadata[:, 2].astype(int)  # neuron_type is column 2

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

        print(f"Model {mid}: Top 10 HARDEST neuron types (by source type):")
        for t, r2 in sorted_types[:10]:
            mask = src_types == t
            print(f"  Type {t:3d}: R²={r2:+.4f}, n_edges={mask.sum():5d}, W_true_mean={W_true[mask].mean():.4f}")
        print()
    except Exception as e:
        print(f"Model {mid}: Type analysis failed ({e})")

print("\n=== 7. Model 049 vs 003 Comparison (Opposite Hard Types) ===")
print("Batch 1 found corr=-1.000 between 049 and 003 type difficulties.")
print("This means OPPOSITE types are hard. Why?\n")

if all_data['049'] is not None and all_data['003'] is not None:
    try:
        import zarr

        for mid in ['049', '003']:
            data_dir = BASE_DATA.format(mid)
            metadata = zarr.open(f'{data_dir}/x_list_0/metadata.zarr', 'r')[:]
            neuron_types = metadata[:, 2].astype(int)

            edge_index = all_data[mid]['edge_index']
            W_true = all_data[mid]['W_true']
            W_learned = all_data[mid]['W_learned']

            src_types = neuron_types[edge_index[0]]

            # Check type 0 specifically (hardest for 049)
            t0_mask = src_types == 0
            if t0_mask.sum() > 0:
                w_t = W_true[t0_mask]
                w_l = W_learned[t0_mask]
                ss_res = np.sum((w_t - w_l)**2)
                ss_tot = np.sum((w_t - w_t.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                pearson = np.corrcoef(w_t, w_l)[0, 1]

                print(f"Model {mid} - Type 0:")
                print(f"  n_edges={t0_mask.sum()}, W_true_mean={w_t.mean():.4f}, W_true_std={w_t.std():.4f}")
                print(f"  R²={r2:+.4f}, Pearson={pearson:+.4f}")
                print(f"  W_learned_mean={w_l.mean():.4f}, W_learned_std={w_l.std():.4f}")
                print()
    except Exception as e:
        print(f"Comparison failed: {e}")

print("\n=== 8. Model 049 Deeper Dive: Why Does lr_W=1E-3 Fail? ===")
print("Model 011 improved dramatically with lr_W=1E-3, but 049 got worse.\n")

if all_data['049'] is not None:
    W_true_049 = all_data['049']['W_true']
    W_learned_049 = all_data['049']['W_learned']

    # Distribution analysis
    print("Model 049 W distribution analysis:")
    print(f"  W_true range: [{W_true_049.min():.4f}, {W_true_049.max():.4f}]")
    print(f"  W_learned range: [{W_learned_049.min():.4f}, {W_learned_049.max():.4f}]")

    # Check if W_learned is near zero (underfitting) or divergent (overfitting)
    near_zero_learned = (np.abs(W_learned_049) < 0.001).sum()
    near_zero_true = (np.abs(W_true_049) < 0.001).sum()

    print(f"  W_true near-zero (|W|<0.001): {near_zero_true} ({100*near_zero_true/len(W_true_049):.1f}%)")
    print(f"  W_learned near-zero (|W|<0.001): {near_zero_learned} ({100*near_zero_learned/len(W_learned_049):.1f}%)")

    # Check large errors
    errors = np.abs(W_true_049 - W_learned_049)
    print(f"  Mean absolute error: {errors.mean():.4f}")
    print(f"  Max absolute error: {errors.max():.4f}")
    print(f"  Edges with error > 0.1: {(errors > 0.1).sum()}")

    # Check if learned W is inverted
    pos_true = W_true_049 > 0.01
    neg_learned_where_pos_true = (W_learned_049 < -0.01) & pos_true
    print(f"  Positive W_true edges: {pos_true.sum()}")
    print(f"  Of those, learned negative: {neg_learned_where_pos_true.sum()} ({100*neg_learned_where_pos_true.sum()/pos_true.sum():.1f}%)")

print("\n=== 9. Summary: Model Difficulty Categories ===\n")

print("MODEL DIFFICULTY ASSESSMENT:")
print("-" * 50)

status = {
    '049': ('FAILING', 0.130, 'lr_W=1E-3 fails; need different approach'),
    '011': ('PARTIAL', 0.674, 'lr_W=1E-3 helps conn but edge_diff=900 hurts'),
    '041': ('CONVERGED', 0.883, 'Connectivity OK but V_rest collapsed'),
    '003': ('SOLVED', 0.968, 'edge_diff=900 optimal, all metrics good')
}

for mid, (stat, r2, note) in status.items():
    print(f"Model {mid}: {stat:10s} (R²={r2:.3f}) - {note}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR NEXT BATCH:")
print("=" * 70)

print("""
Model 049: Try BASELINE lr (6E-4) + edge_diff=900 + W_L1=3E-5
  - lr_W=1E-3 fails for this model specifically
  - edge_diff=900 works for Model 003, may help stabilize 049
  - Focus on regularization, not learning rate

Model 011: Return to edge_diff=750, try W_L1=2E-5
  - edge_diff=900 hurt connectivity, revert
  - Lower W_L1 may help V_rest without connectivity loss

Model 041: Try edge_diff=1200 + phi_L1=1.0
  - Stronger regularization needed for V_rest
  - Connectivity is stable, focus on secondary metrics

Model 003: Maintain edge_diff=900, try phi_L1=0.6
  - Model is solved, minor improvements only
  - Test if phi_L1 can improve V_rest further
""")
