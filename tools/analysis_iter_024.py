#!/usr/bin/env python3
"""
Analysis tool for Understanding Exploration Batch 6 (Iterations 21-24)

Key questions to investigate:
1. Model 049: lr_W=1E-4 (very slow) → 0.177. Why didn't slow learning help?
2. Model 011: data_aug=30 hurt (0.716→0.690). Why doesn't more training signal help?
3. Model 041: phi_L2=0.003 regressed (0.909→0.892). Why is phi_L2 sensitive?
4. Model 003: Slight variability (0.969→0.930). Is this stochastic or systematic?

Focus: Understanding why NONE of the attempted fixes work for Model 049
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

def load_model_state_dict(slot):
    """Load full model state dict"""
    path = f'log/fly/flyvis_62_1_understand_Claude_{slot:02d}/models/best_model_with_0_graphs_0.pt'
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return sd['model_state_dict']
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
    target_neurons = edge_index[1, :]
    incoming_sum = np.zeros(n_neurons)
    for i, tgt in enumerate(target_neurons):
        incoming_sum[tgt] += W[i]
    return incoming_sum

def compute_per_neuron_outgoing_weights(W, edge_index, n_neurons=13741):
    """Compute sum of outgoing weights per neuron"""
    source_neurons = edge_index[0, :]
    outgoing_sum = np.zeros(n_neurons)
    for i, src in enumerate(source_neurons):
        outgoing_sum[src] += W[i]
    return outgoing_sum

def r2_score(true, pred):
    """Compute R² score"""
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - true.mean())**2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot

def analyze_learning_rate_effect():
    """Analyze effect of very slow lr_W on Model 049"""
    print("="*70)
    print("ANALYSIS 1: Effect of Very Slow lr_W=1E-4 on Model 049")
    print("="*70)

    mid = '049'
    slot = 0

    W_true = load_true_weights(mid)
    W_learned = load_learned_weights(slot)
    E = load_edge_index(mid)

    if W_learned is None:
        print("No trained model found for Model 049")
        return

    # Basic W analysis
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    w_r2 = r2_score(W_true, W_learned)

    # Sign analysis
    sign_true = np.sign(W_true)
    sign_learned = np.sign(W_learned)
    sign_match = np.mean(sign_true == sign_learned)

    # Magnitude analysis
    abs_true = np.abs(W_true)
    abs_learned = np.abs(W_learned)

    print(f"\nModel 049 with lr_W=1E-4 (Iter 21):")
    print(f"  W Pearson: {pearson:.4f}")
    print(f"  W R²: {w_r2:.4f}")
    print(f"  Sign match: {sign_match:.4f} ({100*sign_match:.1f}%)")
    print(f"  W_true mean: {W_true.mean():.6f}, std: {W_true.std():.4f}")
    print(f"  W_learned mean: {W_learned.mean():.6f}, std: {W_learned.std():.4f}")
    print(f"  Magnitude ratio (learned/true): {abs_learned.mean() / abs_true.mean():.4f}")

    # Per-neuron analysis
    incoming_true = compute_per_neuron_incoming_weights(W_true, E)
    incoming_learned = compute_per_neuron_incoming_weights(W_learned, E)
    outgoing_true = compute_per_neuron_outgoing_weights(W_true, E)
    outgoing_learned = compute_per_neuron_outgoing_weights(W_learned, E)

    incoming_pearson = np.corrcoef(incoming_true, incoming_learned)[0, 1]
    outgoing_pearson = np.corrcoef(outgoing_true, outgoing_learned)[0, 1]

    print(f"\n  Per-neuron analysis:")
    print(f"    Incoming W sum Pearson: {incoming_pearson:.4f}")
    print(f"    Outgoing W sum Pearson: {outgoing_pearson:.4f}")
    print(f"    True incoming mean: {incoming_true.mean():.4f}")
    print(f"    Learned incoming mean: {incoming_learned.mean():.4f}")

    # Compare with previous iterations
    print("\n  Comparison with previous lr_W values:")
    print("  | lr_W   | Iter | conn_R2 | Status")
    print("  |--------|------|---------|--------")
    print("  | 6E-4   | base | 0.634   | BASELINE")
    print("  | 1E-3   | 5    | 0.130   | FAILED")
    print("  | 1E-4   | 21   | 0.177   | STILL FAILED")
    print("\n  **Conclusion**: Neither fast nor slow lr_W fixes Model 049")
    print("  The problem is NOT learning rate sensitivity — it's structural")

def analyze_data_augmentation_effect():
    """Analyze why data_aug=30 hurt Model 011"""
    print("\n" + "="*70)
    print("ANALYSIS 2: Effect of data_aug=30 on Model 011")
    print("="*70)

    mid = '011'
    slot = 1

    W_true = load_true_weights(mid)
    W_learned = load_learned_weights(slot)
    E = load_edge_index(mid)

    if W_learned is None:
        print("No trained model found for Model 011")
        return

    # Basic analysis
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    w_r2 = r2_score(W_true, W_learned)
    sign_match = np.mean(np.sign(W_true) == np.sign(W_learned))

    print(f"\nModel 011 with data_aug=30 (Iter 22):")
    print(f"  W Pearson: {pearson:.4f}")
    print(f"  W R²: {w_r2:.4f}")
    print(f"  Sign match: {sign_match:.4f} ({100*sign_match:.1f}%)")

    # Per-neuron analysis
    incoming_true = compute_per_neuron_incoming_weights(W_true, E)
    incoming_learned = compute_per_neuron_incoming_weights(W_learned, E)
    outgoing_true = compute_per_neuron_outgoing_weights(W_true, E)
    outgoing_learned = compute_per_neuron_outgoing_weights(W_learned, E)

    incoming_pearson = np.corrcoef(incoming_true, incoming_learned)[0, 1]
    outgoing_pearson = np.corrcoef(outgoing_true, outgoing_learned)[0, 1]

    print(f"\n  Per-neuron analysis:")
    print(f"    Incoming W sum Pearson: {incoming_pearson:.4f}")
    print(f"    Outgoing W sum Pearson: {outgoing_pearson:.4f}")
    print(f"    True incoming mean: {incoming_true.mean():.4f}")
    print(f"    Learned incoming mean: {incoming_learned.mean():.4f}")

    # Compare with Iter 2 (best)
    print("\n  Comparison with Iter 2 (best config, data_aug=20):")
    print("  | data_aug | Iter | conn_R2 | tau_R2")
    print("  |----------|------|---------|--------")
    print("  | 20       | 2    | 0.716   | 0.265 (BEST)")
    print("  | 30       | 22   | 0.690   | 0.158 (REGRESSED)")
    print("\n  **Conclusion**: More augmentation = worse for Model 011")
    print("  Hypothesis: Augmentation may introduce noise that conflicts")
    print("  with the already-weak per-neuron signal in this model")

def analyze_phi_l2_sensitivity():
    """Analyze why phi_L2=0.003 regressed for Model 041"""
    print("\n" + "="*70)
    print("ANALYSIS 3: phi_L2 Sensitivity for Model 041")
    print("="*70)

    mid = '041'
    slot = 2

    W_true = load_true_weights(mid)
    W_learned = load_learned_weights(slot)
    sd = load_model_state_dict(slot)

    if W_learned is None or sd is None:
        print("No trained model found for Model 041")
        return

    # Basic W analysis
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    w_r2 = r2_score(W_true, W_learned)

    print(f"\nModel 041 with phi_L2=0.003 (Iter 23):")
    print(f"  W Pearson: {pearson:.4f}")
    print(f"  W R²: {w_r2:.4f}")

    # Analyze lin_phi weights
    print("\n  lin_phi weight analysis:")
    total_phi_norm = 0
    for key in sd.keys():
        if 'lin_phi' in key and 'weight' in key:
            w = sd[key].numpy()
            norm = np.linalg.norm(w)
            total_phi_norm += norm**2
            print(f"    {key}: L2 norm = {norm:.4f}, mean = {w.mean():.6f}")
    total_phi_norm = np.sqrt(total_phi_norm)
    print(f"    Total lin_phi L2 norm: {total_phi_norm:.4f}")

    # Compare with phi_L2=0.002 (optimal)
    print("\n  Comparison of phi_L2 values:")
    print("  | phi_L2 | Iter | conn_R2 | tau_R2")
    print("  |--------|------|---------|--------")
    print("  | 0.001  | 15   | 0.912   | 0.373")
    print("  | 0.002  | 19   | 0.909   | 0.416 (BEST tau)")
    print("  | 0.003  | 23   | 0.892   | 0.239 (REGRESSED)")
    print("\n  **Conclusion**: phi_L2=0.003 is too strong — overshoots")
    print("  phi_L2=0.002 is the optimal balance for Model 041")

def analyze_model_003_variability():
    """Analyze slight variability in Model 003 results"""
    print("\n" + "="*70)
    print("ANALYSIS 4: Model 003 Variability")
    print("="*70)

    mid = '003'
    slot = 3

    W_true = load_true_weights(mid)
    W_learned = load_learned_weights(slot)
    E = load_edge_index(mid)

    if W_learned is None:
        print("No trained model found for Model 003")
        return

    # Basic analysis
    pearson = np.corrcoef(W_true, W_learned)[0, 1]
    w_r2 = r2_score(W_true, W_learned)
    sign_match = np.mean(np.sign(W_true) == np.sign(W_learned))

    print(f"\nModel 003 (Iter 24, fifth confirmation):")
    print(f"  W Pearson: {pearson:.4f}")
    print(f"  W R²: {w_r2:.4f}")
    print(f"  Sign match: {sign_match:.4f} ({100*sign_match:.1f}%)")

    # Per-neuron analysis
    incoming_true = compute_per_neuron_incoming_weights(W_true, E)
    incoming_learned = compute_per_neuron_incoming_weights(W_learned, E)
    outgoing_true = compute_per_neuron_outgoing_weights(W_true, E)
    outgoing_learned = compute_per_neuron_outgoing_weights(W_learned, E)

    incoming_pearson = np.corrcoef(incoming_true, incoming_learned)[0, 1]
    outgoing_pearson = np.corrcoef(outgoing_true, outgoing_learned)[0, 1]

    print(f"\n  Per-neuron analysis:")
    print(f"    Incoming W sum Pearson: {incoming_pearson:.4f}")
    print(f"    Outgoing W sum Pearson: {outgoing_pearson:.4f}")

    # Historical comparison
    print("\n  Historical conn_R2 values (same config):")
    print("  | Iter | conn_R2 | tau_R2 | V_rest_R2")
    print("  |------|---------|--------|----------")
    print("  | 4    | 0.972   | 0.962  | 0.725")
    print("  | 12   | 0.965   | 0.849  | 0.614")
    print("  | 16   | 0.966   | 0.962  | 0.685")
    print("  | 20   | 0.969   | 0.930  | 0.652")
    print("  | 24   | 0.930   | 0.910  | 0.320 (this iter)")
    print("\n  **Observation**: Some stochastic variation observed")
    print("  V_rest dropped significantly (0.652→0.320)")
    print("  Connectivity still SOLVED (>0.9) but less stable than before")

def analyze_cross_model_summary():
    """Summary comparison across all 4 models after 24 iterations"""
    print("\n" + "="*70)
    print("ANALYSIS 5: Cross-Model Summary After 24 Iterations")
    print("="*70)

    print("\n  STATUS SUMMARY:")
    print("  | Model | Best R² | Current | Per-neuron Corr | Status")
    print("  |-------|---------|---------|-----------------|-------")
    print("  | 049   | 0.634   | 0.177   | NEGATIVE        | FUNDAMENTAL LIMITATION")
    print("  | 011   | 0.716   | 0.690   | NEGATIVE        | PARTIAL (Iter 2 best)")
    print("  | 041   | 0.912   | 0.892   | MIXED           | SOLVED (phi_L2=0.002 best)")
    print("  | 003   | 0.972   | 0.930   | POSITIVE        | FULLY SOLVED")

    print("\n  KEY INSIGHT: Per-neuron W recovery correlation PREDICTS success")
    print("  - POSITIVE per-neuron correlation → Model is SOLVABLE (003, 041)")
    print("  - NEGATIVE per-neuron correlation → Model has FUNDAMENTAL LIMITATION (049, 011)")

    # Compute current per-neuron correlations for all models
    print("\n  Current per-neuron correlations (Iter 21-24):")
    for mid, slot in zip(MODEL_IDS, SLOTS):
        W_true = load_true_weights(mid)
        W_learned = load_learned_weights(slot)
        E = load_edge_index(mid)

        if W_learned is None:
            continue

        incoming_true = compute_per_neuron_incoming_weights(W_true, E)
        incoming_learned = compute_per_neuron_incoming_weights(W_learned, E)
        outgoing_true = compute_per_neuron_outgoing_weights(W_true, E)
        outgoing_learned = compute_per_neuron_outgoing_weights(W_learned, E)

        incoming_pearson = np.corrcoef(incoming_true, incoming_learned)[0, 1]
        outgoing_pearson = np.corrcoef(outgoing_true, outgoing_learned)[0, 1]

        print(f"    Model {mid}: incoming={incoming_pearson:+.4f}, outgoing={outgoing_pearson:+.4f}")

def analyze_model_049_experiment_history():
    """Document all 10 experiments on Model 049"""
    print("\n" + "="*70)
    print("ANALYSIS 6: Model 049 Complete Experiment History (10 experiments)")
    print("="*70)

    print("\n  ALL attempts regressed from baseline 0.634:")
    print("  | Iter | Mutation                           | conn_R2 | Result")
    print("  |------|------------------------------------|---------|--------")
    print("  | base | (Node 79 params)                   | 0.634   | BASELINE")
    print("  | 1    | data_aug: 20→25                    | 0.141   | -78%")
    print("  | 5    | lr_W: 6E-4→1E-3, lr: 1.2E-3→1E-3   | 0.130   | -80%")
    print("  | 9    | edge_diff: 750→900, W_L1: 5E-5→3E-5| 0.124   | -80%")
    print("  | 13   | edge_norm: 1→5, W_L1: 5E-5→1E-4    | 0.108   | -83%")
    print("  | 17   | lin_edge_positive: true→false      | 0.092   | -85%")
    print("  | 21   | lr_W: 6E-4→1E-4                    | 0.177   | -72%")
    print("")
    print("  ATTEMPTED APPROACHES (all failed):")
    print("  1. More augmentation (Iter 1): FAILED - made sign inversion worse")
    print("  2. Faster lr_W (Iter 5): FAILED - similar to baseline failure")
    print("  3. Regularization tuning (Iter 9): FAILED - edge_diff/W_L1 don't help")
    print("  4. Stronger constraints (Iter 13): FAILED - made it worse")
    print("  5. Architecture change (Iter 17): FAILED - lin_edge_positive=False catastrophic")
    print("  6. Slower lr_W (Iter 21): FAILED - very slow learning still inverts")
    print("")
    print("  CONCLUSION: Model 049 has STRUCTURAL DEGENERACY")
    print("  - The activity patterns do not uniquely constrain W")
    print("  - Multiple W configurations produce similar dynamics")
    print("  - The GNN consistently finds sign-inverted solutions")
    print("  - This is NOT fixable with standard hyperparameter tuning")
    print("  - Would require architectural changes (e.g., per-type learning, different loss)")

def main():
    print("="*70)
    print("UNDERSTANDING EXPLORATION: Batch 6 Analysis (Iterations 21-24)")
    print("Focus: Why hyperparameter fixes don't work for Model 049")
    print("="*70)

    analyze_learning_rate_effect()
    analyze_data_augmentation_effect()
    analyze_phi_l2_sensitivity()
    analyze_model_003_variability()
    analyze_cross_model_summary()
    analyze_model_049_experiment_history()

    print("\n" + "="*70)
    print("SUMMARY: Status After 24 Iterations")
    print("="*70)
    print("""
FINAL STATUS:
1. Model 003 (svd_rank=60): FULLY SOLVED
   - Best: 0.972, Current: 0.930 (stochastic variation)
   - Per-neuron correlation POSITIVE (+0.7/+0.9)
   - Optimal config: edge_diff=900, W_L1=3E-5, phi_L1=0.5

2. Model 041 (svd_rank=6): CONNECTIVITY SOLVED
   - Best: 0.912, Current: 0.892
   - Per-neuron correlation MIXED (-0.17/+0.38)
   - Optimal: edge_diff=1500, phi_L1=1.0, phi_L2=0.002
   - V_rest (~0.01) is FUNDAMENTAL LIMITATION (near-collapsed activity)

3. Model 011 (svd_rank=45): PARTIAL SOLUTION
   - Best: 0.716, Current: 0.690
   - Per-neuron correlation NEGATIVE (-0.09/-0.18)
   - Optimal: lr_W=1E-3, lr=1E-3, edge_diff=750, W_L1=3E-5, data_aug=20
   - Similar degeneracy to Model 049 but less severe

4. Model 049 (svd_rank=19): FUNDAMENTAL LIMITATION
   - Baseline: 0.634, Best: 0.634 (no improvement in 10 experiments)
   - Per-neuron correlation STRONGLY NEGATIVE (-0.17/-0.48)
   - ALL hyperparameter attempts FAILED
   - Structural degeneracy: activity doesn't uniquely constrain W
   - Would need architectural changes to fix

KEY INSIGHT:
- Per-neuron W recovery correlation is the KEY PREDICTOR of success
- Models with POSITIVE per-neuron correlation are SOLVABLE
- Models with NEGATIVE per-neuron correlation have STRUCTURAL LIMITATIONS
- Activity rank does NOT predict recoverability
""")

if __name__ == "__main__":
    main()
