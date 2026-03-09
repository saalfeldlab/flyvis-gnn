"""Two-panel plot: null edges (all vs real-only R²) and edge removal.

Usage:
    python plot_two_panel.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

REMOVAL_CSV = os.path.join('log', 'edge_removal_pc_results.csv')
NULL_CSV = os.path.join('log', 'null_edges_pc_results.csv')
BENCH_CSV = os.path.join('log', 'benchmark_results.csv')
OUT_PATH = os.path.join('log', 'two_panel_conn_r2.png')

# ── Load baseline (noise_005 optimized) ──────────────────────────

baseline_all = []   # conn_r2 (all edges)
baseline_real = []  # test_conn_r2 (real edges only)
with open(BENCH_CSV, newline='') as f:
    for r in csv.DictReader(f):
        if r['label'] == 'optimized' and float(r['noise']) == 0.05:
            v = r.get('conn_r2', '')
            if v:
                baseline_all.append(float(v))
            v2 = r.get('test_conn_r2', '')
            if v2:
                baseline_real.append(float(v2))

# ── Load null edges data ─────────────────────────────────────────

null_all = defaultdict(list)   # conn_r2
null_real = defaultdict(list)  # test_conn_r2
with open(NULL_CSV, newline='') as f:
    for r in csv.DictReader(f):
        pct = float(r['null_pct'])
        v = r.get('conn_r2', '')
        if v:
            null_all[pct].append(float(v))
        v2 = r.get('test_conn_r2', '')
        if v2:
            null_real[pct].append(float(v2))

# ── Load edge removal data ───────────────────────────────────────

removal_groups = defaultdict(list)
with open(REMOVAL_CSV, newline='') as f:
    for r in csv.DictReader(f):
        pct = float(r['removal_pct']) * 100
        v = r.get('conn_r2', '')
        if v:
            removal_groups[pct].append(float(v))

# ── Plot ─────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)

# --- Panel 1: Null edges — all vs real-only ---
pcts_null = sorted(null_all.keys())

# All edges line (red)
means_all = [np.mean(null_all[p]) for p in pcts_null]
stds_all = [np.std(null_all[p]) for p in pcts_null]
x_all = [0.0] + list(pcts_null)
y_all = [np.mean(baseline_all)] + means_all
e_all = [np.std(baseline_all)] + stds_all

ax1.errorbar(x_all, y_all, yerr=e_all,
             fmt='s-', color='#2CA02C', capsize=5, markersize=8,
             linewidth=2, markeredgewidth=0,
             label='All edges')
for p in pcts_null:
    vals = null_all[p]
    ax1.scatter([p] * len(vals), vals, s=25, color='#2CA02C', alpha=0.35, zorder=2)
ax1.scatter([0.0] * len(baseline_all), baseline_all, s=25,
            color='#2CA02C', alpha=0.35, zorder=2)

# Real edges only line
means_real = [np.mean(null_real[p]) for p in pcts_null]
stds_real = [np.std(null_real[p]) for p in pcts_null]
x_real = [0.0] + list(pcts_null)
y_real = [np.mean(baseline_real)] + means_real
e_real = [np.std(baseline_real)] + stds_real

ax1.errorbar(x_real, y_real, yerr=e_real,
             fmt='o--', color='#66BB6A', capsize=5, markersize=8,
             linewidth=2, markeredgewidth=0,
             label='Real edges only')
for p in pcts_null:
    vals = null_real[p]
    ax1.scatter([p] * len(vals), vals, s=25, color='#66BB6A', alpha=0.35, zorder=2)
ax1.scatter([0.0] * len(baseline_real), baseline_real, s=25,
            color='#66BB6A', alpha=0.35, zorder=2)

ax1.set_xlabel('Null edges added (%)', fontsize=16)
ax1.set_ylabel('Connectivity R²', fontsize=16)
ax1.axhline(y=0.9, color='#999999', linestyle='--', linewidth=0.7, alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(-0.05, 1.1)
ax1.legend(fontsize=12, frameon=False)

# --- Panel 2: Edge removal ---
pcts_rm = sorted(removal_groups.keys())
means_rm = [np.mean(removal_groups[p]) for p in pcts_rm]
stds_rm = [np.std(removal_groups[p]) for p in pcts_rm]

x_rm = [0.0] + list(pcts_rm)
y_rm = [np.mean(baseline_all)] + means_rm
e_rm = [np.std(baseline_all)] + stds_rm

ax2.errorbar(x_rm, y_rm, yerr=e_rm,
             fmt='o-', color='#D62728', capsize=5, markersize=8,
             linewidth=2, markeredgewidth=0)
for p in pcts_rm:
    vals = removal_groups[p]
    ax2.scatter([p] * len(vals), vals, s=25, color='#D62728', alpha=0.4, zorder=2)
ax2.scatter([0.0] * len(baseline_all), baseline_all, s=25,
            color='#D62728', alpha=0.4, zorder=2)

ax2.set_xlabel('Edges removed (%)', fontsize=16)
ax2.set_ylabel('Connectivity R²', fontsize=16)
ax2.axhline(y=0.9, color='#999999', linestyle='--', linewidth=0.7, alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved: {OUT_PATH}')
