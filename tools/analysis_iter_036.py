#!/usr/bin/env python3
"""
Analysis tool for Batch 9 (Iterations 33-36)

Key questions to investigate:
1. Why does recurrent_training=True help Model 049 so dramatically (0.166→0.501)?
2. Why does hidden_dim=96 hurt Model 011 (0.769→0.593)?
3. What makes lr_W=5E-4 optimal for Model 041 (0.931)?
4. Compare W recovery patterns across all models

Focus: Understanding the recurrent_training breakthrough for Model 049
"""

import os
import numpy as np
import torch

# Model configurations
MODEL_IDS = ['049', '011', '041', '003']
SLOTS = [0, 1, 2, 3]
DATA_DIR = 'graphs_data/fly'
LOG_DIR = 'log/fly'

def load_model_weights(slot):
    """Load trained model weights for a slot."""
    log_path = f'{LOG_DIR}/flyvis_62_1_understand_Claude_{slot:02d}'
    model_path = f'{log_path}/models/best_model_with_0_graphs_0.pt'
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint['model_state_dict']
    except Exception as e:
        return None

def load_true_weights(model_id):
    """Load ground truth weights for a model."""
    data_path = f'{DATA_DIR}/flyvis_62_1_id_{model_id}'
    W_true = torch.load(f'{data_path}/weights.pt', weights_only=True).cpu().numpy()
    return W_true

def compute_w_metrics(W_true, W_learned):
    """Compute detailed W recovery metrics."""
    # Flatten
    W_true_flat = W_true.flatten()
    W_learned_flat = W_learned.flatten()

    # Basic stats
    pearson = np.corrcoef(W_true_flat, W_learned_flat)[0, 1]
    r2 = 1 - np.sum((W_true_flat - W_learned_flat)**2) / np.sum((W_true_flat - W_true_flat.mean())**2)

    # Sign analysis
    sign_match = np.mean(np.sign(W_true_flat) == np.sign(W_learned_flat))

    # Magnitude ratio
    nonzero_mask = np.abs(W_true_flat) > 1e-6
    if nonzero_mask.sum() > 0:
        mag_ratio = np.mean(np.abs(W_learned_flat[nonzero_mask]) / np.abs(W_true_flat[nonzero_mask]))
    else:
        mag_ratio = np.nan

    # Per-sign analysis
    pos_mask = W_true_flat > 0.01
    neg_mask = W_true_flat < -0.01

    if pos_mask.sum() > 0:
        pos_pearson = np.corrcoef(W_true_flat[pos_mask], W_learned_flat[pos_mask])[0, 1]
    else:
        pos_pearson = np.nan

    if neg_mask.sum() > 0:
        neg_pearson = np.corrcoef(W_true_flat[neg_mask], W_learned_flat[neg_mask])[0, 1]
    else:
        neg_pearson = np.nan

    return {
        'pearson': pearson,
        'r2': r2,
        'sign_match': sign_match,
        'mag_ratio': mag_ratio,
        'pos_pearson': pos_pearson,
        'neg_pearson': neg_pearson,
        'mean_true': W_true_flat.mean(),
        'mean_learned': W_learned_flat.mean(),
        'std_true': W_true_flat.std(),
        'std_learned': W_learned_flat.std()
    }

def analyze_mlp_structure(state_dict, prefix='lin_edge'):
    """Analyze MLP structure and weight statistics."""
    layer_info = {}
    for key, value in state_dict.items():
        if key.startswith(prefix) and 'weight' in key:
            layer_num = key.split('.')[2]
            w = value.numpy()
            layer_info[layer_num] = {
                'shape': w.shape,
                'l2_norm': np.linalg.norm(w),
                'mean_abs': np.abs(w).mean(),
                'frac_large': (np.abs(w) > 0.1).mean()
            }
    return layer_info

def load_edge_index(model_id):
    """Load edge index for a model."""
    data_path = f'{DATA_DIR}/flyvis_62_1_id_{model_id}'
    edge_index = torch.load(f'{data_path}/edge_index.pt', weights_only=True).cpu().numpy()
    return edge_index

def per_neuron_w_analysis(W_true, W_learned, edge_index, n_neurons=13741):
    """Analyze per-neuron W recovery."""
    # Compute per-neuron incoming and outgoing W sums
    true_incoming = np.zeros(n_neurons)
    true_outgoing = np.zeros(n_neurons)
    learned_incoming = np.zeros(n_neurons)
    learned_outgoing = np.zeros(n_neurons)

    src, dst = edge_index[0], edge_index[1]

    for i, (s, d) in enumerate(zip(src, dst)):
        true_outgoing[s] += W_true[i]
        true_incoming[d] += W_true[i]
        learned_outgoing[s] += W_learned[i]
        learned_incoming[d] += W_learned[i]

    # Correlations
    in_pearson = np.corrcoef(true_incoming, learned_incoming)[0, 1]
    out_pearson = np.corrcoef(true_outgoing, learned_outgoing)[0, 1]

    return {
        'incoming_pearson': in_pearson,
        'outgoing_pearson': out_pearson,
        'true_in_mean': true_incoming.mean(),
        'learned_in_mean': learned_incoming.mean(),
        'true_out_mean': true_outgoing.mean(),
        'learned_out_mean': learned_outgoing.mean()
    }

def main():
    print("=" * 70)
    print("ANALYSIS ITER 036: Recurrent Training Breakthrough Investigation")
    print("=" * 70)

    # Iteration info
    iter_info = {
        '049': {'iter': 33, 'conn_r2': 0.501, 'recurrent': True, 'config': 'n_layers=4, emb=4, recurrent=True'},
        '011': {'iter': 34, 'conn_r2': 0.593, 'recurrent': False, 'config': 'hidden_dim=96, n_layers=4'},
        '041': {'iter': 35, 'conn_r2': 0.931, 'recurrent': False, 'config': 'lr_W=5E-4'},
        '003': {'iter': 36, 'conn_r2': 0.962, 'recurrent': False, 'config': 'Iter 4 baseline'}
    }

    print("\n=== Batch 9 Results Summary ===")
    print(f"{'Model':<8} {'Iter':<6} {'conn_R2':<10} {'Config':<40}")
    print("-" * 65)
    for mid, info in iter_info.items():
        print(f"{mid:<8} {info['iter']:<6} {info['conn_r2']:<10.4f} {info['config']:<40}")

    # Load and compare W recovery
    print("\n=== W Recovery Analysis ===")
    print(f"{'Model':<8} {'Pearson':<10} {'R²':<10} {'Sign%':<10} {'MagRatio':<10} {'Pos_Pear':<10} {'Neg_Pear':<10}")
    print("-" * 70)

    w_metrics = {}
    for mid, slot in zip(MODEL_IDS, SLOTS):
        state_dict = load_model_weights(slot)
        W_true = load_true_weights(mid)

        if state_dict is not None and 'W' in state_dict:
            W_learned = state_dict['W'].numpy().flatten()
            metrics = compute_w_metrics(W_true, W_learned)
            w_metrics[mid] = metrics

            print(f"{mid:<8} {metrics['pearson']:<10.4f} {metrics['r2']:<10.4f} "
                  f"{metrics['sign_match']*100:<10.1f} {metrics['mag_ratio']:<10.3f} "
                  f"{metrics['pos_pearson']:<10.4f} {metrics['neg_pearson']:<10.4f}")
        else:
            print(f"{mid:<8} NO MODEL LOADED")

    # Compare recurrent vs non-recurrent for Model 049
    print("\n=== Model 049: Recurrent Training Effect ===")
    print("Iter 33 (recurrent=True):  conn_R2=0.501 (BREAKTHROUGH)")
    print("Iter 29 (recurrent=False): conn_R2=0.166")
    print("Improvement: 0.166 → 0.501 = +0.335 (3x improvement)")

    if '049' in w_metrics:
        m = w_metrics['049']
        print(f"\nW recovery with recurrent_training=True:")
        print(f"  Pearson:    {m['pearson']:.4f}")
        print(f"  R²:         {m['r2']:.4f}")
        print(f"  Sign match: {m['sign_match']*100:.1f}%")
        print(f"  Mag ratio:  {m['mag_ratio']:.3f}")
        print(f"  Mean(true): {m['mean_true']:.6f}, Mean(learned): {m['mean_learned']:.6f}")

    # Per-neuron analysis
    print("\n=== Per-Neuron W Recovery ===")
    print(f"{'Model':<8} {'In_Pearson':<12} {'Out_Pearson':<12} {'In_Mean_T':<12} {'In_Mean_L':<12}")
    print("-" * 60)

    per_neuron = {}
    for mid, slot in zip(MODEL_IDS, SLOTS):
        state_dict = load_model_weights(slot)
        W_true = load_true_weights(mid)
        edge_index = load_edge_index(mid)

        if state_dict is not None and 'W' in state_dict:
            W_learned = state_dict['W'].numpy().flatten()
            pn = per_neuron_w_analysis(W_true, W_learned, edge_index)
            per_neuron[mid] = pn

            print(f"{mid:<8} {pn['incoming_pearson']:<12.4f} {pn['outgoing_pearson']:<12.4f} "
                  f"{pn['true_in_mean']:<12.4f} {pn['learned_in_mean']:<12.4f}")

    # MLP analysis
    print("\n=== lin_edge MLP Analysis ===")
    for mid, slot in zip(MODEL_IDS, SLOTS):
        state_dict = load_model_weights(slot)
        if state_dict is not None:
            print(f"\nModel {mid}:")
            layer_info = analyze_mlp_structure(state_dict, 'lin_edge')
            total_params = 0
            for layer_num, info in sorted(layer_info.items()):
                params = info['shape'][0] * info['shape'][1]
                total_params += params
                print(f"  Layer {layer_num}: shape={info['shape']}, L2={info['l2_norm']:.3f}, "
                      f"mean_abs={info['mean_abs']:.4f}, frac_large={info['frac_large']:.3f}")
            print(f"  Total lin_edge params: {total_params}")

    # Why does recurrent help Model 049?
    print("\n" + "=" * 70)
    print("=== KEY FINDING: Why Recurrent Training Helps Model 049 ===")
    print("=" * 70)

    if '049' in w_metrics and '049' in per_neuron:
        m = w_metrics['049']
        pn = per_neuron['049']

        print(f"""
Model 049 with recurrent_training=True:
  - conn_R2 improved from 0.166 to 0.501 (3x improvement)
  - Per-neuron incoming Pearson: {pn['incoming_pearson']:.4f}
  - Per-neuron outgoing Pearson: {pn['outgoing_pearson']:.4f}
  - W Pearson: {m['pearson']:.4f}
  - Sign match: {m['sign_match']*100:.1f}%

HYPOTHESIS: Recurrent training provides temporal context that enables:
  1. Longer-range gradient flow through time
  2. More stable W updates by seeing multiple frames in sequence
  3. Better disambiguation of degenerate W solutions

The model's low activity rank (svd_rank=19) means per-frame gradients
are weak and ambiguous. Recurrent training aggregates information
across multiple timesteps, providing stronger and more consistent
gradient signal for W learning.
""")

    # Model 011 hidden_dim analysis
    print("\n=== Model 011: Why hidden_dim=96 Hurts ===")
    if '011' in w_metrics:
        m = w_metrics['011']
        pn = per_neuron['011']
        print(f"""
Model 011 with hidden_dim=96 (vs 80):
  - conn_R2 regressed from 0.769 to 0.593
  - Per-neuron incoming Pearson: {pn['incoming_pearson']:.4f}
  - Per-neuron outgoing Pearson: {pn['outgoing_pearson']:.4f}
  - W Pearson: {m['pearson']:.4f}

HYPOTHESIS: Excess capacity causes overfitting:
  - n_layers=4 helps by providing depth (more compositional features)
  - But hidden_dim=96 adds width (more parameters per layer)
  - Width without depth leads to overfitting/poor generalization
  - Optimal: n_layers=4 + hidden_dim=80
""")

    # Model 041 lr_W analysis
    print("\n=== Model 041: lr_W=5E-4 is Optimal ===")
    if '041' in w_metrics:
        m = w_metrics['041']
        print(f"""
Model 041 lr_W sensitivity:
  - lr_W=3E-4: conn_R2=0.888 (too slow)
  - lr_W=4E-4: conn_R2=0.919
  - lr_W=5E-4: conn_R2=0.931 (NEW BEST)
  - lr_W=6E-4: conn_R2=0.629 (baseline)

With lr_W=5E-4:
  - W Pearson: {m['pearson']:.4f}
  - Mag ratio: {m['mag_ratio']:.3f}

Near-collapsed activity (svd_rank=6) provides weak gradient signal.
lr_W=5E-4 is the optimal trade-off: fast enough to exploit weak signal,
slow enough to avoid overshooting.
""")

    # Compare all models
    print("\n=== Cross-Model Comparison: Status After 36 Iterations ===")
    print("""
| Model | Best R² | Status                    | Key Factor                        |
|-------|---------|---------------------------|-----------------------------------|
| 003   | 0.972   | FULLY SOLVED (9 conf)     | Per-neuron W correlation POSITIVE |
| 041   | 0.931   | CONNECTIVITY SOLVED       | lr_W=5E-4 optimal for low rank    |
| 011   | 0.769   | PARTIAL                   | n_layers=4 helps, width hurts     |
| 049   | 0.501*  | BREAKTHROUGH              | recurrent_training enables W      |

* Model 049 baseline was 0.634, best without recurrent was 0.181.
  With recurrent_training=True: 0.501. Path forward identified!
""")

    print("\n=== Recommendations for Batch 10 ===")
    print("""
1. Model 049: Try recurrent_training + edge_diff=900 + W_L1=3E-5
   - Combine recurrent breakthrough with Model 003's optimal regularization
   - Target: Exceed baseline 0.634

2. Model 011: Try recurrent_training=True
   - Test if temporal context helps like Model 049
   - If breakthrough, could improve from 0.769

3. Model 041: Maintain lr_W=5E-4 config
   - Connectivity SOLVED at 0.931
   - May try phi_L2 tuning for tau improvement

4. Model 003: Continue maintenance runs
   - FULLY SOLVED, no changes needed
""")

    print("\n" + "=" * 70)
    print("END OF ANALYSIS")
    print("=" * 70)

if __name__ == '__main__':
    main()
