# %% [raw]
# ---
# title: "Agentic Hyper-Parameter Optimization: Addressing Identifiability"
# author: "Allier, Lappalainen, Saalfeld"
# categories:
#   - FlyVis
#   - GNN
#   - Agentic Optimization
#   - Identifiability
# execute:
#   echo: false
# image: "assets/Fig_agentic_loop.svg"
# description: "Agentic hyper-parameter optimization addresses the ill-posedness of circuit recovery across three noise regimes, discovering that identifiability fails for different reasons at each level and finding distinct solutions through hypothesis-driven reasoning."
# ---

# %% [markdown]
# ## Agentic Hyper-Parameter Optimization
#
# The inverse problem solved by the GNN is **ill-posed**: recovering five coupled
# components ($\widehat{W}$, $\tau$, $V^{\text{rest}}$, $f_\theta$,
# $g_\phi$) from voltage traces alone is under-determined.  Many
# different parameter combinations can produce indistinguishable
# voltage predictions.  This degeneracy manifests as **seed
# dependence**: slight differences in random initialization can push
# the optimizer toward a different solution on the degenerate
# manifold.  A GNN that accurately forecasts neural activity may
# nonetheless have recovered the wrong connectivity.
#
# The combined space of architecture, regularization, and training
# hyperparameters (~20 coupled parameters) is too large to explore
# exhaustively.  Rather than grid search, we deployed a closed-loop
# system where Claude Code interpreted experiment results,
# maintained a structured research summary, and proposed the next intervention.
# At each iteration the agent selected parent configurations to mutate
# using an Upper Confidence Bound (UCB) tree search that balances
# exploitation of high-performing branches with exploration of
# under-visited regions.  The system implemented a form of automated
# scientific reasoning: testable hypotheses were drawn, repeatable
# experiments validated or falsified them, and causal understanding
# progressively emerged [1].
#
# <object type="image/svg+xml" data="assets/Fig_agentic_loop.svg" width="700"></object>
#
# The primary optimization target is not prediction accuracy (which
# is easy) but **identifiability**: the coefficient of variation
# (CV) of connectivity $R^2$ across random seeds measures how
# consistently a configuration escapes the degenerate solution landscape.
# Across five explorations (600 iterations), the agent established
# transferable principles and falsified hypotheses.  At each
# noise level, identifiability fails for a different reason and
# requires a different solution.

# %% [markdown]
# ## Identifiability Across Noise Regimes
#
# The three noise conditions reveal three distinct faces of the
# ill-posedness problem.  At $\sigma{=}0$ the bottleneck is
# geometric capacity of the representation space.  At
# $\sigma{=}0.05$ it is the sharpness of the optimization
# landscape.  At $\sigma{=}0.5$ it is the non-convexity of the
# loss surface itself.  The agent discovered each mechanism through
# hypothesis-driven reasoning chains spanning tens to hundreds of
# iterations.

# %% [markdown]
# ### Noise-free ($\sigma = 0$)
# *160 iterations, 14 blocks*
#
# The default configuration yields a mean connectivity $R^2$ of
# $0.64 \pm 0.17$ across five seeds, with two seeds falling below
# $0.6$.  The LLM agent first identified that weight L1
# regularization, designed for noisy data, severely degrades
# noise-free performance ($R^2 = 0.76$ vs.\ $0.90$ without).
# It then found that increasing the embedding dimension from 2 to 4
# raised connectivity to $0.927$, but an 8-seed validation revealed
# a 37% catastrophic failure rate with batch size 2.  The agent
# hypothesized that noisy early gradients trap the connectivity
# matrix $W$ in bad basins and that larger batches would smooth
# these gradients.  Switching to batch size 4 eliminated all
# catastrophic failures but reduced the mean to $0.894$ due to
# fewer gradient steps per epoch.  A systematic augmentation sweep
# identified $\text{aug}{=}30$ as optimal, restoring the mean to
# $0.923 \pm 0.008$ with a CV of $0.82\%$ and all seeds above $0.9$.
#
# **Best config**: `embedding_dim=4`, `batch_size=4`,
# `g_phi_diff=1500`, `aug_loop=30`, `n_epochs=1`.
# **Result**: $R^2 = 0.92 \pm 0.01$, **CV = 0.82%** (4/4 seeds
# $> 0.91$).

# %% [markdown]
# ### Low noise ($\sigma = 0.05$)
# *256 iterations, 22 blocks*
#
# The default configuration achieves a mean connectivity $R^2$ of
# $0.79 \pm 0.18$, with one seed collapsing to $0.45$.  The LLM
# agent found that uniformly scaling all learning rates by
# $1.5\times$ produced the single largest improvement
# ($0.938 \to 0.971$).  It combined this with batch size 4 and
# doubled $W$ sparsity ($\lambda_{L1} = 1.5 \times 10^{-4}$),
# then performed a fine-grained augmentation sweep.  This sweep
# revealed a narrow plateau at $\text{aug} \in [34, 35]$ with a
# sharp performance cliff at $\text{aug}{=}36$, a structure that
# would be difficult to find without systematic search.  The
# resulting best configuration achieves $R^2 = 0.98 \pm 0.004$
# on five seeds.  Extended validation at 12 seeds uncovered a
# structural ${\sim}8\%$ catastrophic failure rate, indicating that
# the 1-epoch training framework has an irreducible seed-dependent
# bifurcation.
#
# **Best config**: `batch_size=4`, $1.5\times$ learning rates,
# `aug_loop=35`, `W_L1=1.5e-4`, `n_epochs=1`.
# **Result**: $R^2 = 0.98 \pm 0.004$, **CV = 0.3%** (5/5 seeds
# $> 0.97$).

# %% [markdown]
# ### High noise ($\sigma = 0.5$)
# *204 iterations, 17 blocks*
#
# The default configuration exhibits a **bimodal** landscape:
# ${\sim}25\%$ of seeds catastrophically fail ($R^2 \approx 0.20$)
# while the rest achieve near-perfect recovery ($R^2 >: 0.99$).
# The agent found that this failure rate is invariant to architecture depth,
# embedding dimension, and epoch count.  Scaling all learning rates
# by $1.5\times$ was the single intervention that eliminated
# catastrophic failures (4/4 seeds robust, CV${=}0.64\%$).
# The agent then found a **non-monotonic augmentation landscape**:
# $\text{aug}{=}5$ and $\text{aug}{=}20$ both give 0\% failure at
# $1.5\times$ LRs, but $\text{aug}{=}10$ triggers 50\% catastrophic
# failure.  This destructive interference between augmentation 
# and learning rate was confirmed by replication (8 seeds).
# The champion configuration ($\text{aug}{=}15$, $1.5\times$ LRs)
# sits on the safe side of this valley.  Extended validation at 12
# seeds revealed a ${\sim}8\%$ irreducible failure rate, lower than
# the 25\% baseline but not fully eliminated.
#
# **Best config**: `batch_size=2`, $1.5\times$ learning rates,
# `aug_loop=15`, `n_epochs=1`.
# **Result**: $R^2 = 0.99 \pm 0.01$, **CV = 0.98\%** (5/5 seeds
# $> 0.97$).

# %% [markdown]
# ### Joint GNN + SIREN ($\sigma = 0.05$): 24 iterations, 2 blocks
#
# **SIREN depth $\geq 4$ eliminates catastrophic failures** (3-layer:
# 25% failure rate).  4 layers gives best connectivity ($R^2 = 0.94$,
# CV = 0.90%); 5 layers gives best field reconstruction
# ($R^2 = 0.83$, higher GNN variance).
# `omega=4096` outperforms 1024 in joint training — opposite of
# standalone SIREN.

# %% [markdown]
# ### Standalone SIREN ($\sigma = 0.05$): 152 iterations, 13 blocks
#
# Best field $R^2 = 0.90$ (`hidden_dim=768`, 7 layers,
# `omega=1750`, `lr=1.5e-7`, 60k steps).  The dominant lever is
# `omega` (initial frequency scale): increasing it from 30 to 1024
# yielded $+0.24$ $R^2$, exceeding all other tuning combined.
# Higher omega ($\geq 2000$) suffers from late-stage collapse,
# making 1750 optimal.  Learning rate scales inversely with omega
# ($\omega \times \text{lr} \approx 2.5 \times 10^{-4}$).

# %%
#| output: false
import os
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Running the Agentic Pipeline
#
# **Prerequisites.** A Claude API subscription is required.
# We recommend running the pipeline inside a sandboxed environment
# such as a Dev Container to avoid unintended side effects.
#
# The command line:
#
#     python GNN_Agentic.py -o train_test_plot_Claude flyvis_noise_005_agentic iterations=48
#
# The pipeline takes two inputs:
#
# - **Config file** (`config/fly/flyvis_noise_005_agentic.yaml`):
#   defines the GNN architecture, training hyperparameters, and
#   the `claude:` section controlling the exploration
#   (block size, UCB constant, training budget, case study brief).
# - **Instruction file** (`LLM/instruction_flyvis_noise_005_agentic.md`):
#   system prompt describing the problem, the available
#   hyperparameters, and the metrics the agent should optimize.
#
# All outputs are written to
# `log/Claude_exploration/LLM_flyvis_noise_005_agentic/`:
#
# | Path | Contents |
# |:--|:--|
# | `configs/` | YAML snapshots of every configuration tested |
# | `results/` | Per-iteration metrics and plots |
# | `tree/` | UCB search-tree visualizations |
# | `*_Claude_analysis.md` | **Research summary** — the agent's running synthesis of all experiments, hypotheses tested, and current best configurations |
# | `*_Claude_memory.md` | **Research memory** — compact factual records the agent carries across iterations |
# | `*_Claude_reasoning.log` | Full reasoning trace for every batch |
# | `*_Claude_ucb_scores.txt` | UCB scores guiding parent selection |
#


# %% [markdown]
# ## References
#
# [1] C. Allier, L. Heinrich, M. Schneider, S. Saalfeld, "Graph
# neural networks uncover structure and functions underlying the
# activity of simulated neural assemblies," *arXiv:2602.13325*,
# 2026.
# [doi:10.48550/arXiv.2602.13325](https://doi.org/10.48550/arXiv.2602.13325)
#
# [2] B. Romera-Paredes et al., "Mathematical discoveries from
# program search with large language models," *Nature*, 2024.
#
# [3] A. Novikov et al., "AlphaEvolve: A coding agent for
# scientific and algorithmic exploration," 2025.

# %%
