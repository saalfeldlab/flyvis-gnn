# FlyVis GNN Training Exploration — flyvis_noise_005

## Goal

Test **robustness of GNN training** for the **Drosophila visual system** with noise level 0.05 (DAVIS input).
The goal is to find a **stabilized config** that achieves **connectivity_R2 > 0.9 on ALL 4 seeds with CV < 3%**, demonstrating that the result is not data-dependent and not sensitive to random seed choice.

Data is **re-generated each iteration** with a different seed to verify seed independence.

**STABILITY IS THE PRIMARY OBJECTIVE.** A config with mean connectivity_R2=0.95 and CV=1% is BETTER than a config with mean=0.98 and CV=8%. The user currently observes occasional runs with connectivity_R2 < 0.87, which is unacceptable — the goal is to eliminate this failure mode entirely.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
**Stability metric: CV (coefficient of variation) of connectivity_R2 across 4 seeds — target CV < 3%.**
**Hard floor: min(connectivity_R2) across all 4 seeds must be > 0.90. Any seed < 0.87 disqualifies the config.**
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a hypothesis about what controls robustness (e.g., "Higher coeff_g_phi_diff will reduce seed variance because stronger monotonicity constraints reduce the number of degenerate solutions")
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change ONE parameter at a time
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate whether the hypothesis was supported or contradicted
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information.

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence. When results contradict your hypothesis, update it — do not rationalize away the evidence.

**Evidence hierarchy:**

| Level            | Criterion                                       | Action                 |
| ---------------- | ----------------------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations AND 4/4 seeds   | Add to Principles      |
| **Tentative**    | Observed 1-2 times or inconsistent across seeds | Add to Open Questions  |
| **Contradicted** | Conflicting evidence across iterations/seeds    | Note in Open Questions |

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):

- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.05**
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

Example: embedding_dim=4 → input_size=5, input_size_update=7. Shape mismatch crashes otherwise.

## Regularization Parameters

The training loss includes:

| Config parameter          | Role                                                                                | Default | Annealed? |
| ------------------------- | ----------------------------------------------------------------------------------- | ------- | --------- |
| `coeff_g_phi_diff`        | Monotonicity penalty on g_phi: ReLU(-dg_phi/dv) → enforces increasing edge messages | 750     | No        |
| `coeff_g_phi_norm`        | Normalization penalty on g_phi at saturation voltage                                | 0.9     | No        |
| `coeff_g_phi_weight_L1`   | L1 penalty on g_phi MLP weights                                                     | 0.28    | **Yes**   |
| `coeff_g_phi_weight_L2`   | L2 penalty on g_phi MLP weights                                                     | 0       | **Yes**   |
| `coeff_f_theta_weight_L1` | L1 penalty on f_theta MLP weights                                                   | 0.05    | **Yes**   |
| `coeff_f_theta_weight_L2` | L2 penalty on f_theta MLP weights                                                   | 0.001   | **Yes**   |
| `coeff_f_theta_msg_diff`  | Monotonicity of f_theta w.r.t. message input                                        | 0       | No        |
| `coeff_W_L1`              | L1 sparsity penalty on connectivity W                                               | 7.5e-05 | **Yes**   |
| `coeff_W_L2`              | L2 penalty on W                                                                     | 1.5e-06 | **Yes**   |

### Regularization Annealing

All 6 weight regularization coefficients (L1 and L2 for g_phi, f_theta, and W) share a **single exponential ramp-up annealing** controlled by one parameter:

| Config parameter       | Default | Description                                      |
| ---------------------- | ------- | ------------------------------------------------ |
| `regul_annealing_rate` | 0.5     | Shared annealing rate for all L1/L2 regularizers |

**Formula**: `effective_coeff = coeff * (1 - exp(-rate * epoch))`

**Ramp-up schedule** (rate=0.5):

| Epoch | Multiplier | Meaning                        |
| ----- | ---------- | ------------------------------ |
| 0     | 0.00       | No regularization at start     |
| 1     | 0.39       | ~39% of configured coefficient |
| 2     | 0.63       | ~63%                           |
| 5     | 0.92       | ~92% (near full strength)      |
| 10    | 0.99       | ~full strength                 |

**Purpose**: Allows the model to learn dynamics first before regularization pressure is applied. At epoch 0, all L1/L2 penalties are zero regardless of their configured coefficients.

**CRITICAL — 1-epoch training**: With `n_epochs=1`, training only runs epoch 0, where the annealing multiplier is exactly **0.00**. ALL six L1/L2 regularizers are completely inactive — the configured coefficients have NO effect. Only the non-annealed coefficients (`coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_msg_diff`) apply.

**Fix**: Use `n_epochs: 2` and halve `data_augmentation_loop` to keep total training time constant. This gives:

- Epoch 0: L1/L2 = 0 (model learns dynamics freely)
- Epoch 1: L1/L2 at 39% strength (regularization cleans up weights)

Alternatively, set `regul_annealing_rate: 0` to disable annealing entirely (full strength from epoch 0).

**Non-annealed coefficients**: `coeff_g_phi_diff`, `coeff_g_phi_norm`, and `coeff_f_theta_msg_diff` apply at full strength from epoch 0 regardless of annealing settings.

## Training Parameters (explorable)

| Parameter                       | Default      | Description                                                         |
| ------------------------------- | ------------ | ------------------------------------------------------------------- |
| `learning_rate_W_start`         | 6e-4         | Learning rate for connectivity matrix W                             |
| `learning_rate_start`           | 1.2e-3       | Learning rate for g_phi and f_theta MLPs                            |
| `learning_rate_embedding_start` | 1.55e-3      | Learning rate for neuron embeddings                                 |
| `n_epochs`                      | 2 (claude)   | Epochs per iteration (keep ≤ 2 for time)                            |
| `batch_size`                    | 2            | Batch size for training                                             |
| `data_augmentation_loop`        | 20           | Data augmentation multiplier                                        |
| `recurrent_training`            | false        | Enable multi-step rollout training                                  |
| `time_step`                     | 1            | Recurrent steps (if recurrent_training=true)                        |
| `w_init_mode`                   | randn_scaled | W initialization: "zeros" or "randn_scaled"                         |
| `lr_scheduler`                  | none         | LR schedule: "none", "cosine_warm_restarts", "linear_warmup_cosine" |
| `lr_scheduler_T0`               | 1000         | First restart period in iterations (cosine schedulers only)         |
| `lr_scheduler_T_mult`           | 2            | Period multiplier after each restart                                |
| `lr_scheduler_eta_min_ratio`    | 0.01         | Min LR as fraction of base LR                                       |
| `lr_scheduler_warmup_iters`     | 100          | Linear warmup iterations (linear_warmup_cosine only)                |

### LR Scheduler Notes

When `lr_scheduler="none"` (default), per-iteration LR is constant and the legacy epoch-level halving (every 10 epochs) remains active. When a scheduler is enabled, it steps **per iteration** (not per epoch), so the LR oscillates within each epoch.

**Recommended exploration**: `cosine_warm_restarts` with `T0=500-2000` provides periodic LR restarts that can help escape local minima. `linear_warmup_cosine` adds a warmup ramp for stability with large initial LR. The `T0` parameter controls how frequently the LR resets — smaller T0 means more frequent restarts.

## Training Time Constraint

Baseline (batch_size=2, 64K frames, hidden_dim=80): **~90 min/epoch on H100**, **~120 min on A100**.
Data generation adds **~10-15 min** per slot.
Keep total training time (generation + training) ≤ 100 min/iteration. Monitor `training_time_min`.

Factors that increase training time:

- Larger `hidden_dim` / `n_layers`
- Larger `data_augmentation_loop`
- Smaller `batch_size`
- `recurrent_training=true` with large `time_step`

## Parallel Mode — 4 Slots Per Batch

You receive **4 results per batch** and propose **4 mutations** for the next batch.
Each slot runs with a **different random seed** for data generation, so you can directly assess seed robustness within a single batch.

### Robustness Assessment

After each batch, evaluate using **both** mean R2 and stability (CV):

- **Stable-Robust**: all 4 slots connectivity_R2 > 0.9 AND CV < 3% — **TARGET**
- **Robust**: all 4 slots connectivity_R2 > 0.9 but CV 3-5% — acceptable, try to stabilize
- **Partially robust**: 2-3 slots connectivity_R2 > 0.9 — needs improvement
- **Fragile**: 0-1 slots connectivity_R2 > 0.9 — reject
- **DISQUALIFIED**: any slot connectivity_R2 < 0.87 — this config is unreliable, do NOT pursue further

A config is considered **validated** only when it achieves connectivity_R2 > 0.9 on all 4 seeds with CV < 3%.
**If any single seed drops below 0.87, the config is considered a stability failure regardless of the mean.**

### Slot Strategy

Since the goal is robustness testing, all 4 slots should run the **same config** (different seeds are applied automatically).

| Slot | Role          | Description                            |
| ---- | ------------- | -------------------------------------- |
| 0    | **seed test** | Same config, seed varies automatically |
| 1    | **seed test** | Same config, seed varies automatically |
| 2    | **seed test** | Same config, seed varies automatically |
| 3    | **seed test** | Same config, seed varies automatically |

When a config is validated as robust (all 4 seeds > 0.9), you may switch to exploring a variation:

| Slot | Role        | Description                                             |
| ---- | ----------- | ------------------------------------------------------- |
| 0-3  | **exploit** | All 4 slots test the next candidate config across seeds |

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- **All 4 configs should be identical** (only seeds differ, set automatically)
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters** (except that seed is managed automatically)

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12).
The prompt provides: `Block info: block {block_number}, iterations {iter_in_block}/{n_iter_block} within block`

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

- Append every iteration's log entry (4 entries per batch)
- Append block summaries at block boundaries
- **Never read** — human record only

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

- Read at start, update at end
- Contains: robustness comparison table, hypotheses, established principles, current block iterations
- Keep ≤ 500 lines

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

- Read at every batch
- If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged" section with timestamp
- Do not remove acknowledged entries — append them with `[ACK {batch}]` marker

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

- Read `{llm_task_name}_memory.md` for context — especially hypotheses and robustness table
- Read `user_input.md` for any pending user instructions

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `connectivity_R2`: R² of learned vs true W (PRIMARY)
- `tau_R2`: R² of learned vs true time constants
- `V_rest_R2`: R² of learned vs true resting potentials
- `cluster_accuracy`: neuron type clustering accuracy from embeddings
- `test_R2`: one-step prediction R²
- `training_time_min`: training duration

**Robustness classification (across all 4 seeds):**

- **Stable-Robust**: all 4 slots connectivity_R2 > 0.9 AND CV < 3% — **TARGET**
- **Robust**: all 4 slots connectivity_R2 > 0.9, CV 3-5%
- **Partially robust**: 2-3 slots connectivity_R2 > 0.9
- **Fragile**: 0-1 slots connectivity_R2 > 0.9
- **DISQUALIFIED**: any slot < 0.87 — reject config immediately

**Per-slot classification:**

- **Converged**: connectivity_R2 > 0.9
- **Marginal**: connectivity_R2 0.87–0.9 — warning zone
- **Failed**: connectivity_R2 < 0.87 — **disqualifies the config**

**Seed variance analysis (compute every batch):**

- Compute mean, std, CV, min, max for connectivity_R2 across the 4 slots
- CV < 3% → target stability; CV 3-5% → acceptable; CV > 5% → unstable, investigate
- **min(connectivity_R2) is as important as the mean** — report it prominently

**UCB scores from `ucb_scores.txt`:**

- UCB(k) = R²_k + c × sqrt(ln(N) / n_k) where c = `ucb_c` (default 1.414)
- At block boundaries the UCB file is empty — use `parent=root`

### Step 3: Write Log Entries + Update Hypotheses + Update Memory

**3a. Append to Full Log** (`{llm_task_name}_analysis.md`) and **Current Block** in memory.md:

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis being tested]"
Config (same for all slots): lr_W=X, lr=Y, lr_emb=Z, coeff_g_phi_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D
Slot 0: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Slot 1: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Slot 2: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Slot 3: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W, max=V
Stability: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line explanation]
Observation: [one line about seed sensitivity or robustness pattern]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder — always include exact parameter change.
**CRITICAL**: `Next: parent=P` — P must be from a previous batch or current batch, NEVER `id+1`.

**3b. Update Hypotheses in memory.md:**

After analyzing results, update the `## Hypotheses` section:

- If results **support** the hypothesis → increase confidence, note supporting evidence
- If results **falsify** the hypothesis → mark as falsified, formulate a new hypothesis informed by the contradicting evidence
- If results are **inconclusive** → note what additional experiment would clarify

**3c. Update Robustness Comparison Table in memory.md** (see Working Memory Structure below).

### Step 4: Acknowledge User Input (if any)

If `user_input.md` has content in "Pending Instructions":

- Edit `user_input.md`: move the pending items to "Acknowledged" with `[ACK batch_{batch_first}-{batch_last}]` prefix
- Incorporate the instructions into your next config mutations

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

1. Based on current understanding, formulate the **next hypothesis** to test
2. Design a config mutation that specifically tests this hypothesis (ONE parameter change)
3. All 4 configs should be **identical** — the pipeline assigns different seeds automatically
4. Write the hypothesis to memory.md before editing configs

## Block Partition (suggested)

| Block | Focus                  | Parameters                                                               |
| ----- | ---------------------- | ------------------------------------------------------------------------ |
| 1     | Baseline robustness    | Default config across 4 seeds — establish baseline                       |
| 2     | Learning rates         | lr_W, lr, lr_emb                                                         |
| 3     | g_phi regularization   | coeff_g_phi_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1                |
| 4     | f_theta regularization | coeff_f_theta_weight_L1, coeff_f_theta_weight_L2, coeff_f_theta_msg_diff |
| 5     | W regularization       | coeff_W_L1, coeff_W_L2, w_init_mode                                      |
| 6     | Architecture           | hidden_dim, n_layers, hidden_dim_update, n_layers_update, embedding_dim  |
| 7     | Combined best          | Best parameters from blocks 1–6                                          |
| 8     | Validation             | Re-run best config with more seeds / longer training                     |

## Block Boundaries

At the end of each block:

1. Update "Paper Summary" at the top of memory.md — rewrite both bullet points to reflect the current state of knowledge
2. Summarize findings in memory.md "Previous Block Summary"
3. Update "Established Principles" with confirmed insights (require 3+ supporting iterations AND cross-seed consistency)
4. Move falsified hypotheses to "Falsified Hypotheses" with evidence summary
5. Clear "Current Block" for next block
6. Carry forward best **robust** config as starting point

## Failed Slots

If a slot is `[FAILED]`:

- Write a brief note in the log entry
- A single slot failure may indicate seed sensitivity — note this
- Still propose the next config for the next batch
- Do not draw conclusions from a single failure

## Known Results (prior experiments)

- `flyvis_62_1` (DAVIS + noise 0.05): connectivity_R2=0.95, tau_R2=0.80, V_rest_R2=0.40 (10 epochs, full regularization)
- `flyvis_62_1` with Sintel input: R²_W=0.99, tau_R2=1.00, V_rest_R2=0.85
- W initialization: `randn_scaled` and `zeros` perform similarly; plain `randn` performs poorly
- Larger MLP (80-dim/3-layer) works better than smaller (32-dim/2-layer)
- `coeff_g_phi_diff` (monotonicity) is among the most important regularizers — too low causes non-monotonic messages
- Noise level 0.05 may require stronger regularization than noise-free to achieve robust convergence

## Start Call

When prompt says `PARALLEL START`:

- Read base config to understand training regime
- Set all 4 configs **identically** to the baseline config
- Data will be generated with different seeds per slot automatically
- Write planned config and **initial hypothesis** to working memory
- First iteration establishes baseline robustness — do not change hyperparameters yet
- State the baseline hypothesis: "The default config achieves connectivity_R2 > 0.9 robustly across seeds"

---

# Working Memory Structure

The memory file (`{llm_task_name}_memory.md`) must follow this structure:

```markdown
# Working Memory: flyvis_noise_005

## Paper Summary (update at every block boundary)

Brief paragraph for a scientific paper summarizing the current state of this exploration:

- **GNN optimization**: [What has been learned about training GNN to recover Drosophila visual system connectivity from noisy DAVIS data. Best connectivity_R2 achieved, which regularization/learning rate regimes work, key challenges.]
- **LLM-driven exploration**: [How the LLM-in-the-loop approach performed as an automated hyperparameter search strategy. Number of iterations run, how hypothesis-driven exploration compared to random search, whether the LLM discovered non-obvious parameter interactions.]
- **Future works**: [what would a google deepmind senior ML suggest as structural change in the code to break ceiling with good rationale, this can include modified GNN class, innovative training scheme, use recent innovations. Cite references of scientific publications, blog, youtube channel. Limit to 10 suggestions.]

## Knowledge Base (accumulated across all blocks)

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | min | max | tau_R2 (mean) | V_rest_R2 (mean) | Stability | Hypothesis tested |
| ---- | -------------- | ------------------ | --- | --- | --- | ------------- | ---------------- | --------- | ----------------- |
| 1    | defaults       | ?                  | ?   | ?   | ?   | ?             | ?                | ?         | baseline          |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

Examples of good principles:

- ✓ "coeff_g_phi_diff ≥ 500 is necessary for robust convergence (3/3 iterations, all seeds > 0.9)"
- ✓ "lr_W > 1e-3 causes seed-dependent failures (CV > 20% in 2 iterations)"
- ✗ "lr_W=6e-4 worked in iteration 3" (too specific, not a principle)

### Falsified Hypotheses

[Hypotheses that were contradicted by experimental evidence — keep as record]

### Open Questions

[Patterns needing more testing, contradictions, seed-dependent observations]

---

## Previous Block Summary (Block N-1)

[Short summary: 2-3 lines. NOT individual iterations.
Example: "Block 1 (baseline): Default config achieves conn_R2=0.93±0.04, CV=4.3%.
3/4 seeds > 0.9. Key finding: baseline is partially robust, lr_W may need tuning for full robustness."]

---

## Current Block (Block N)

### Block Info

Focus: [which parameter subspace]
Iterations: M to M+n_iter_block

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why you believe this, based on prior evidence]
**Test**: [what config change tests this]
**Expected outcome**: [what would support vs falsify]
**Stability constraint**: [acceptable CV range AND minimum per-seed R2, e.g., "CV < 3%, all seeds > 0.90"]
**Status**: untested / supported / falsified / revised

### Iterations This Block

[Current block iterations — cleared at block boundary]

### Emerging Observations

[Running notes on what patterns are emerging across seeds and iterations]
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

---

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 seeds** (not just mean, but low variance)
3. States a **causal relationship** (not just a correlation)

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable
