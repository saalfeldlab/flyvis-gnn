# FlyVis GNN Training Exploration — flyvis_noise_05

## Goal

Explore the **optimization landscape** of GNN training for the **Drosophila visual system** at high noise ($\sigma{=}0.5$, DAVIS input).

### Context: Noise as a Leveler

At $\sigma{=}0.5$, the GNN already achieves near-perfect parameter recovery with the default configuration:
- $R^2_W \approx 0.99$, $R^2_\tau \approx 1.00$, $R^2_{V^{\text{rest}}} \approx 0.85$, clustering accuracy $> 0.86$

This is dramatically better than the noise_005 ($\sigma{=}0.05$) condition, where 220 iterations of careful optimization were needed to reach $R^2_W{=}0.982$.  The mechanism is clear: high intrinsic noise widens the distribution of membrane voltages explored during training, providing richer coverage of the system's dynamical regime.  The GNN learns a **noise-free** dynamical model — when evaluated on clean test data, it tracks the deterministic ground truth with high fidelity.

**This changes the nature of the exploration.**  We are NOT trying to "make it work" — it already works.  Instead, we ask:

### Three Research Questions

**Q1 — How wide is the basin of attraction?**
At noise_005, the optimization landscape is narrow and treacherous: a single step in `data_augmentation_loop` (35→36) causes a 30x CV increase; all 3 LR components sit at sharp optima where ±11-17% perturbations degrade performance.  Does noise_05 flatten this landscape?  Can we use "bad" hyperparameters that catastrophically fail at noise_005 and still succeed at noise_05?

**Approach**: Deliberately test configurations that FAILED at noise_005 and measure whether noise_05 rescues them.  Known failure modes from noise_005:
- `n_layers=4`: FRAGILE (0.837, all seeds < 0.9)
- `embedding_dim=4`: Partial (0.914, high variance)
- `coeff_f_theta_msg_diff=100`: Catastrophic (0.662)
- `n_epochs=2`: Consistently harmful (FRAGILE)
- LR schedulers: Catastrophic failures
- `batch_size=2` with default LRs: High variance (CV=10.6%)

If noise_05 rescues these, we learn that noise acts as an implicit regularizer that smooths the optimization landscape.  If some still fail, we learn which failure modes are fundamental vs. noise-dependent.

**Q2 — What is the minimal configuration?**
With noise providing such strong signal, how much can we strip away?  Can we:
- Remove all regularization (`coeff_g_phi_diff=0`, all L1/L2=0)?
- Use tiny networks (`hidden_dim=32`, `n_layers=2`)?
- Use minimal augmentation (`data_augmentation_loop=5` or even 1)?
- Use a single batch_size=1?

This maps the **necessary vs. sufficient** conditions for circuit recovery under high noise.

**Q3 — Can we push V_rest beyond 0.85?**
$V^{\text{rest}}$ recovery is the weakest metric even at noise_05.  Is this a fundamental limit of the 1-epoch training paradigm, or can we push it higher with targeted optimization?  The noise_005 exploration found that V_rest is orthogonal to connectivity optimization and bimodally distributed — is the same true at noise_05, or does the wider voltage exploration help?

### Metrics and Assessment

Primary metric: **connectivity_R2** — but since it's already near 0.99, the interesting signal is in the **secondary metrics** and **failure modes**.

| Rating | Criterion | Interpretation |
|--------|-----------|----------------|
| **Robust** | All 4 seeds conn_R2 > 0.95, CV < 3% | Normal for noise_05 — expected baseline |
| **Degraded** | conn_R2 > 0.9 but < 0.95, or CV > 3% | Interesting — this config stresses the system |
| **Broken** | Any seed conn_R2 < 0.9 | Very interesting — found a failure mode that survives noise |
| **Catastrophic** | Mean conn_R2 < 0.8 | Found a fundamental failure — noise can't rescue this |

**The most scientifically valuable results are "Degraded" and "Broken"** — they map the boundaries of the basin.

For each iteration, also report: tau_R2, V_rest_R2, cluster_accuracy, training_time.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction about the noise_05 landscape
2. **Design experiment**: Choose a mutation that tests the hypothesis — change ONE parameter at a time
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Compare to both the noise_05 baseline AND the noise_005 result for the same config
5. **Update understanding**: Build a comparative landscape map

**CRITICAL**: You can only hypothesize.  Only training results can validate or falsify.  The most interesting findings will be SURPRISES — configs that fail despite the noise advantage, or configs that work despite being "wrong".

**Evidence hierarchy:**

| Level | Criterion | Action |
|-------|-----------|--------|
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent across seeds | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations/seeds | Note in Open Questions |

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):

- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t) + sigma * xi_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.5** (10x higher than noise_005)
- 64,000 frames, delta_t=0.02

## GNN Architecture

Two MLPs learn the neural dynamics:

- **g_phi** (MLP1): Edge message function. Maps (v_j, a_j) → message. `g_phi_positive=true` squares output to enforce positivity.
- **f_theta** (MLP0): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) → dv_i/dt.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

Architecture parameters (explorable):

- `hidden_dim` / `n_layers`: g_phi MLP width/depth (default: 80 / 3)
- `hidden_dim_update` / `n_layers_update`: f_theta MLP width/depth (default: 80 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim` (v_j + a_j for g_phi)
- `input_size_update = 3 + embedding_dim` (v_i + a_i + msg + I_i for f_theta)

## Regularization Parameters

| Config parameter | Role | Default | Annealed? |
|-----------------|------|---------|-----------|
| `coeff_g_phi_diff` | Monotonicity penalty on g_phi | 750 | No |
| `coeff_g_phi_norm` | Normalization penalty at saturation | 0.9 | No |
| `coeff_g_phi_weight_L1` | L1 on g_phi MLP weights | 0.28 | **Yes** |
| `coeff_g_phi_weight_L2` | L2 on g_phi MLP weights | 0 | **Yes** |
| `coeff_f_theta_weight_L1` | L1 on f_theta MLP weights | 0.5 | **Yes** |
| `coeff_f_theta_weight_L2` | L2 on f_theta MLP weights | 0.001 | **Yes** |
| `coeff_f_theta_msg_diff` | Monotonicity of f_theta w.r.t. messages | 0 | No |
| `coeff_W_L1` | L1 sparsity penalty on W | 7.5e-5 | **Yes** |
| `coeff_W_L2` | L2 penalty on W | 1.5e-6 | **Yes** |

### Regularization Annealing

**Formula**: `effective_coeff = coeff * (1 - exp(-rate * epoch))`

**CRITICAL — 1-epoch training**: With `n_epochs=1`, training only runs epoch 0, where the annealing multiplier is exactly **0.00**.  ALL L1/L2 regularizers are completely inactive.  Only the non-annealed coefficients (`coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_msg_diff`) apply.

This means the default config relies entirely on the monotonicity constraint (`coeff_g_phi_diff=750`) and the noise itself as regularizers.

## Training Parameters (explorable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate_W_start` | 6e-4 | LR for connectivity W |
| `learning_rate_start` | 1.2e-3 | LR for g_phi and f_theta |
| `learning_rate_embedding_start` | 1.55e-3 | LR for embeddings |
| `n_epochs` | 1 | Epochs (explore 2-3 for V_rest) |
| `batch_size` | 2 | Batch size |
| `data_augmentation_loop` | 20 | Data augmentation multiplier |
| `w_init_mode` | randn_scaled | W initialization |
| `regul_annealing_rate` | 0.5 | Annealing rate (irrelevant at 1 epoch) |

## Training Time Constraint

Target: **≤ 60 min/iteration** (1 epoch).  The noise_005 champion at aug=35 took ~69 min; noise_05 with aug=20 should be well within budget.

## Parallel Mode — 4 Slots Per Batch

You receive **4 results per batch** and propose **4 mutations** for the next batch.
Each slot runs with a **different random seed** for robustness assessment.

### Robustness Assessment

| Rating | Criterion | Action |
|--------|-----------|--------|
| **Robust** | All 4 seeds > 0.95, CV < 3% | Expected baseline for noise_05 |
| **Degraded** | 0.9 < mean < 0.95 or CV > 3% | Interesting — stress point found |
| **Broken** | Any seed < 0.9 | Very interesting — noise can't rescue |
| **Catastrophic** | Mean < 0.8 | Fundamental failure |

### Slot Strategy

All 4 slots should run the **same config** (different seeds are applied automatically).

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- **All 4 configs should be identical** (only seeds differ)
- Only modify `training:` and `graph_model:` parameters
- **DO NOT change `simulation:` parameters**

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12, i.e. 3 batches of 4).

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `connectivity_R2`: R² of learned vs true W (**PRIMARY METRIC**)
- `tau_R2`: R² of learned vs true time constants
- `V_rest_R2`: R² of learned vs true resting potentials
- `cluster_accuracy`: neuron type clustering accuracy from embeddings
- `training_time_min`: training duration

**Dual comparison**: For each config tested, compare to:
1. The noise_05 baseline (how much did this perturbation degrade/improve?)
2. The noise_005 result for the same config (how much does noise rescue?)

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/degraded/broken/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: [key changes from baseline]
Slot 0: conn_R2=A, tau_R2=B, V_rest_R2=C, cluster_acc=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, tau_R2=B, V_rest_R2=C, cluster_acc=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, tau_R2=B, V_rest_R2=C, cluster_acc=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, tau_R2=B, V_rest_R2=C, cluster_acc=D, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
noise_005 comparison: [same config at noise_005 gave X — noise rescued by Y%]
Mutation: [param]: [old] -> [new]
Verdict: [robust/degraded/broken] — [one line interpretation]
Next: parent=P
```

### Step 4: Acknowledge User Input (if any)

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Partition — PRESCRIBED Configs (Blocks 1-4)

Blocks 1-4 are **prescribed**: you MUST run these exact configs in order. This ensures a direct, systematic comparison with the noise_005 exploration. After block 4, you are free to explore based on findings.

### Block 1 (Iters 1-12): Baseline + Architecture Stress Tests

| Batch | Config | noise_005 result | What we learn |
|-------|--------|-----------------|---------------|
| 1-4 | **Baseline** (default config, no changes) | 0.888±0.094 (Partial) | noise_05 starting point |
| 5-8 | **n_layers=4** (4-layer MLPs) | 0.837±0.030 (FRAGILE 0/4) | Does noise rescue depth failure? |
| 9-12 | **embedding_dim=4** (input_size=5, input_size_update=7) | 0.914±0.071 (Partial 3/4) | Does noise rescue over-parameterization? |

### Block 2 (Iters 13-24): Training Regime Stress Tests

| Batch | Config | noise_005 result | What we learn |
|-------|--------|-----------------|---------------|
| 13-16 | **n_epochs=2, aug=10** (same total training time) | 0.831±0.181 (FRAGILE 2/4) | Does noise rescue epoch boundary? |
| 17-20 | **f_theta_msg_diff=100** (f_theta monotonicity) | 0.662±0.194 (Catastrophic 0/4) | Does noise rescue wrong constraint? |
| 21-24 | **cosine_warm_restarts** LR scheduler | 0.909±0.097 (Partial 3/4) | Does noise rescue LR scheduling? |

### Block 3 (Iters 25-36): Minimal Configuration Tests

| Batch | Config | noise_005 result | What we learn |
|-------|--------|-----------------|---------------|
| 25-28 | **g_phi_diff=0** (remove monotonicity entirely) | Not tested at noise_005 | Is monotonicity necessary at high noise? |
| 29-32 | **hidden_dim=32, n_layers=2** (minimal network) | Not tested at noise_005 | Minimum network for recovery |
| 33-36 | **aug=5** (minimal augmentation) | Not tested at noise_005 | Minimum data for recovery |

### Block 4 (Iters 37-48): LR Landscape + V_rest

| Batch | Config | noise_005 result | What we learn |
|-------|--------|-----------------|---------------|
| 37-40 | **1.5x LRs** (noise_005 champion LRs: lr_W=9e-4, lr=1.8e-3, lr_emb=2.325e-3) | 0.982±0.003 (CHAMPION) | noise_005 champion at noise_05 |
| 41-44 | **aug=36** (cliff edge at noise_005) | 0.898±0.089 (DISQUALIFIED) | Does noise eliminate the cliff? |
| 45-48 | **n_epochs=2, aug=10, 1.5x LRs** | ~0.945±0.063 (Partial) | V_rest push via multi-epoch |

### Blocks 5+ (Iters 49+): Free Exploration — CV Elimination Priority

**PRIMARY OBJECTIVE**: Eliminate the bimodal catastrophic failure mode. ~25% of seeds fail with conn_R2 < 0.20 under the default config. Find a configuration where ALL 5 seeds achieve conn_R2 > 0.90 with CV < 10%.

Known failed interventions: n_layers=4 (replication failure), zeros init, 1.5x LRs, augmentation changes, LR schedulers. Explore fundamentally different approaches: gradient clipping, alternative initializations (xavier, kaiming), LR warmup, higher embedding_dim, larger batch sizes, stronger g_phi_diff.

Secondary priorities:
- Deepen understanding of any SURPRISING results (configs that failed or succeeded unexpectedly)
- Map the boundary between "rescued" and "not rescued" more precisely
- Push V_rest_R2 beyond 0.85 using the best approach found

## Complete noise_005 Reference Table (220 iterations, 45 perturbations)

Use this table to look up the noise_005 result for any config you test. The **noise_005 champion** is: aug=35, bs=4, 1.5x LRs, hidden=80, 3L, emb=2, W_L1=1.5e-4 → conn_R2=0.982±0.003.

| Config | noise_005 conn_R2 | CV% | min | Status | Category |
|--------|-------------------|-----|-----|--------|----------|
| defaults (bs=2) | 0.938 | N/A | — | single seed | baseline |
| 1.5x LRs (bs=2) | 0.971 | N/A | — | single seed | LR |
| g_phi_diff=1500 | 0.924 | N/A | — | single seed | regularization |
| bs=4, W_L1=1.5e-4, default LRs | 0.888±0.094 | 10.6 | 0.766 | Partial (2/4) | batch+sparsity |
| **bs=4, W_L1=1.5e-4, 1.5x LRs** | **0.952±0.022** | 2.3 | 0.927 | **ROBUST** | LR compensation |
| bs=4, W_L1=1.5e-4, 2x LRs | 0.949±0.020 | 2.1 | 0.921 | ROBUST | LR |
| lr_W=2x, lr/emb=1.5x | 0.963±0.010 | 1.1 | 0.953 | ROBUST | LR differential |
| lr_W=1.5x, lr/emb=2x | 0.953±0.022 | 2.3 | 0.921 | ROBUST | LR differential |
| g_phi_L1=0.14 (halved) | 0.912±0.050 | 5.5 | 0.841 | Partial (3/4) | regularization |
| g_phi_norm=1.8 (doubled) | 0.945±0.028 | 3.0 | 0.902 | ROBUST | regularization |
| g_phi_L1=0.42 (1.5x) | 0.942±0.022 | 2.3 | 0.917 | ROBUST | regularization |
| **n_epochs=2, bs=4, 1.5x LRs** | **0.831±0.181** | **21.8** | **0.566** | **FRAGILE (2/4)** | **epoch boundary** |
| **f_theta_msg_diff=100** | **0.662±0.194** | **29.4** | **0.398** | **CATASTROPHIC (0/4)** | **wrong constraint** |
| W_L1=7.5e-5 (default) | 0.867±0.091 | 10.5 | 0.777 | Partial (2/4) | regularization |
| W_L1=3e-4 (doubled) | 0.939±0.020 | 2.1 | 0.908 | ROBUST | regularization |
| W_L2=1.5e-5 (10x) | 0.937±0.010 | 1.1 | 0.922 | ROBUST | regularization |
| hidden_dim=120 | 0.951±0.011 | 1.2 | 0.935 | ROBUST | architecture |
| **n_layers=4** | **0.837±0.030** | **3.6** | **0.796** | **FRAGILE (0/4)** | **architecture** |
| **embedding_dim=4** | **0.914±0.071** | **7.8** | **0.809** | **Partial (3/4)** | **architecture** |
| w_init_mode=zeros | 0.957±0.023 | 2.4 | 0.924 | ROBUST | initialization |
| f_theta_L1=0.025 (halved) | 0.937±0.054 | 5.8 | 0.858 | Partial (3/4) | regularization |
| f_theta_L2=0.002 (doubled) | 0.928±0.050 | 5.4 | 0.849 | Partial (3/4) | regularization |
| regul_anneal=2.0 | 0.949±0.044 | 4.6 | 0.884 | Partial (3/4) | annealing |
| regul_anneal=1.0 | 0.942±0.034 | 3.6 | 0.890 | Partial (3/4) | annealing |
| **cosine_warm_restarts** | **0.909±0.097** | **10.7** | **0.742** | **Partial (3/4)** | **LR scheduler** |
| **linear_warmup_cosine** | **0.853±0.186** | **21.8** | **0.575** | **Partial (3/4)** | **LR scheduler** |
| data_aug=25 | 0.962±0.015 | 1.6 | 0.950 | ROBUST | augmentation |
| data_aug=30 | 0.969±0.011 | 1.2 | 0.956 | ROBUST | augmentation |
| **data_aug=35 (CHAMPION)** | **0.982±0.003** | **0.3** | **0.977** | **CHAMPION** | **augmentation** |
| **data_aug=36 (cliff)** | **0.898±0.089** | **9.9** | **0.782** | **DISQUALIFIED** | **augmentation cliff** |
| data_aug=40 | 0.928±0.072 | 7.7 | 0.804 | Partial (3/4) | augmentation |
| data_aug=34 | 0.978±0.002 | 0.3 | 0.975 | Stable-Robust | augmentation plateau |
| lr_W=1.0e-3 (+11%) at aug=35 | 0.951±0.031 | 3.3 | 0.917 | Robust | LR fine-tuning |
| lr_W=8e-4 (-11%) at aug=35 | 0.972±0.012 | 1.3 | 0.958 | Stable-Robust | LR fine-tuning |
| **lr=2.1e-3 (+17%) at aug=35** | **0.904±0.058** | **6.4** | **0.817** | **DISQUALIFIED** | **LR overshoot** |
| lr=1.5e-3 (-17%) at aug=35 | 0.955±0.028 | 2.9 | 0.915 | Robust | LR fine-tuning |
| lr_emb=2.72e-3 (+17%) | 0.963±0.021 | 2.1 | 0.942 | Robust (degraded) | LR fine-tuning |
| **lr_emb=1.93e-3 (-17%)** | **0.927±0.076** | **8.1** | **0.797** | **DISQUALIFIED** | **embedding bottleneck** |
| g_phi_diff=500 at aug=35 | 0.963±0.014 | 1.4 | 0.950 | Stable-Robust (degraded) | regularization |
| batch_size=8 at aug=35 | 0.968±0.009 | 1.0 | 0.955 | Stable-Robust (degraded) | batch size |
| g_phi_norm=0.45 (halved) | 0.962±0.005 | 0.5 | 0.955 | Stable-Robust (degraded) | regularization |

**Bold rows** = failure modes prescribed for testing in blocks 1-4.

**Sibling memory file**: `./log/Claude_exploration/LLM_flyvis_noise_005/flyvis_noise_005_Claude_memory.md` — read at block boundaries for full context.

## Start Call

When prompt says `PARALLEL START`:

- Read base config
- Set all 4 configs **identically** to the baseline
- First iteration = baseline — do not change hyperparameters
- Baseline hypothesis: "The default noise_05 config achieves connectivity_R2 > 0.95 on all 4 seeds with CV < 2%, establishing a robust baseline that exceeds noise_005 performance with simpler hyperparameters"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_05

## Paper Summary (update at every block boundary)

- **Noise_05 landscape**: [...]
- **LLM-driven exploration**: [...]
- **Comparison with noise_005**: [...]

## Knowledge Base

### Comparative Landscape Table

| Iter | Config summary | noise_05 conn_R2 (mean±std) | CV% | min | noise_005 result | Rescued? | Rating | Question tested |
| ---- | -------------- | --------------------------- | --- | --- | ---------------- | -------- | ------ | --------------- |
| 1 | baseline | ? | ? | ? | 0.938 (baseline) | N/A | ? | Q1 baseline |

### Established Principles

[What noise_05 reveals about the optimization landscape vs. noise_005]

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summary

---

## Current Block

### Block Info

### Current Hypothesis

**Hypothesis**: [specific prediction about noise_05 landscape]
**Rationale**: [why — reference noise_005 comparison]
**Test**: [config change]
**Expected outcome**: [what would support vs falsify]
**Status**: untested / supported / falsified

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
