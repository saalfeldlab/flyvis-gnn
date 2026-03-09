"""Generate the agentic optimization loop figure (Fig_agentic_loop.svg).

Run once to create the static asset; Notebook_08 and index.qmd display the SVG.
Boxes are clickable in the SVG (embedded hyperlinks).
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # embed text as <text>, not paths
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 6.0)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Palette ──────────────────────────────────────────────────────
c_exp  = "#3B6EA5"   # experiment — deep blue
c_agt  = "#E07B39"   # agent — warm orange
c_mem  = "#3A9A5B"   # research summary — green
c_ucb  = "#C44E52"   # UCB tree — red
c_grey = "#888888"
c_lgrey = "#BBBBBB"

# ── Helpers ──────────────────────────────────────────────────────
def rounded_box(x, y, w, h, color, alpha=0.18, url=None):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35",
        facecolor=color, edgecolor="none", alpha=alpha, linewidth=0)
    if url:
        box.set_url(url)
    ax.add_patch(box)
    return box

def arrow(x1, y1, x2, y2, color, label=None, label_side="above"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                mutation_scale=18, color=color))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.25 if label_side == "above" else -0.25
        ax.text(mx, my + offset, label, fontsize=8, ha="center",
                color="#555", fontstyle="italic")

def panel_label(x, y, text):
    ax.text(x, y, text, fontsize=14, fontweight="bold", va="top", ha="left",
            color="#333")

# ════════════════════════════════════════════════════════════════
#  (a) Closed-loop optimization
# ════════════════════════════════════════════════════════════════
panel_label(-0.3, 5.9, "a")
ax.text(2.0, 5.8, "Agentic hyper-parameter optimization",
        fontsize=12, va="top", fontstyle="italic", color="#444")

# ── Experiment box ──
rounded_box(0.1, 2.6, 2.6, 1.6, c_exp, url="../Notebook_01.html")
ax.text(1.4, 3.7, "Experiment", fontsize=12, ha="center", va="center",
        fontweight="bold", color=c_exp)
ax.text(1.4, 3.15, "GNN training\n(4 seeds, GPU)", fontsize=8.5,
        ha="center", va="center", color=c_exp, linespacing=1.3)

# ── Agent box ──
rounded_box(4.0, 2.6, 2.6, 1.6, c_agt, url="../Notebook_09.html")
ax.text(5.3, 3.7, "Claude Code", fontsize=12, ha="center", va="center",
        fontweight="bold", color=c_agt)
ax.text(5.3, 3.15, "interpret, hypothesize\n& configure", fontsize=8.5,
        ha="center", va="center", color=c_agt, linespacing=1.3)

# ── Research Summary box ──
rounded_box(4.0, -0.1, 2.6, 1.6, c_mem, url="../Notebook_09.html#research-summary")
ax.text(5.3, 1.0, "Research Summary", fontsize=11, ha="center",
        va="center", fontweight="bold", color=c_mem)
ax.text(5.3, 0.5, "principles, hypotheses\n& experiment history",
        fontsize=8.5, ha="center", va="center", color=c_mem,
        linespacing=1.3)

# ── Arrows: loop ──
# Experiment → Agent (results)
arrow(2.7, 3.65, 4.0, 3.65, "#444444", "R\u00b2, metrics", "above")
# Agent → Experiment (next config)
arrow(4.0, 3.1, 2.7, 3.1, "#444444", "next config", "below")
# Agent ↔ Research Summary
arrow(5.05, 2.6, 5.05, 1.5, "#444444")
arrow(5.55, 1.5, 5.55, 2.6, "#444444")
ax.text(6.15, 2.0, "read / write", fontsize=8, ha="left", color="#555",
        fontstyle="italic")

# ════════════════════════════════════════════════════════════════
#  (b) UCB tree search
# ════════════════════════════════════════════════════════════════
panel_label(7.5, 5.9, "b")
ax.text(8.3, 5.8, "UCB tree search", fontsize=12, va="top",
        fontstyle="italic", color="#444")

def tree_node(x, y, score, best=False):
    color = c_ucb if best else c_grey
    alpha = 1.0 if best else 0.45
    circle = plt.Circle((x, y), 0.26, facecolor=color, edgecolor="white",
                         linewidth=2, alpha=alpha, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, f"{score}", fontsize=7.5, ha="center", va="center",
            color="white", fontweight="bold", zorder=4)

def tree_edge(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#BBBBBB", lw=1.8, zorder=1)

# Root
rx, ry = 9.5, 4.8
tree_node(rx, ry, ".94")

# Level 1
for x, y in [(8.5, 3.7), (9.5, 3.7), (10.5, 3.7)]:
    tree_edge(rx, ry - 0.26, x, y + 0.26)
tree_node(8.5, 3.7, ".95")
tree_node(9.5, 3.7, ".97")
tree_node(10.5, 3.7, ".91")

# Level 2 (under .97)
for x, y in [(9.0, 2.6), (10.0, 2.6)]:
    tree_edge(9.5, 3.7 - 0.26, x, y + 0.26)
tree_node(9.0, 2.6, ".96")
tree_node(10.0, 2.6, ".98", best=True)

# Level 3 (under .98 — current frontier)
for x, y in [(9.6, 1.5), (10.4, 1.5)]:
    tree_edge(10.0, 2.6 - 0.26, x, y + 0.26)
tree_node(9.6, 1.5, ".97")
tree_node(10.4, 1.5, ".98", best=True)

# Under .95
tree_edge(8.5, 3.7 - 0.26, 8.5, 2.86)
tree_node(8.5, 2.6, ".92")

# UCB label
ax.text(9.5, 0.7, "exploit best branches\n+ explore uncertain ones",
        fontsize=8.5, ha="center", va="center", color=c_ucb,
        fontstyle="italic", linespacing=1.3)

plt.tight_layout()
out_path = "/workspace/flyvis-gnn/assets/Fig_agentic_loop.svg"
fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
plt.close()

# Post-process: add target="_top" so links navigate the parent page
with open(out_path) as f:
    svg = f.read()
svg = svg.replace('xlink:href=', 'target="_top" xlink:href=')
with open(out_path, 'w') as f:
    f.write(svg)
