"""Fig. 3: CAST method overview (schematic, not measured data).

Three panels matching the four steps in sec:field-update:
  (1) Propose  -- frozen base maps noise to a cloud of candidate actions
  (2) Nudge + Leash -- critic gradient, clipped; anchor pulls back strays
  (3) Imitate  -- regress the residual head onto the frozen targets
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

rng = np.random.default_rng(0)
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#777777", "#009E73"

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
for ax in axes:
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.15, 1.25)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")

# data-manifold band, shared by all panels (same shape family as Fig. 1)
gx = np.linspace(-1.30, 1.30, 240)
gy = 0.55 * gx**2 - 0.45
HALF = 0.20

a0 = np.array([-0.15, 0.55 * (-0.15)**2 - 0.45])                       # base action pi_0(s,z)
_cx = -0.15 + rng.normal(0, 0.20, size=6)
cand = np.stack([_cx, 0.55 * _cx**2 - 0.45 + rng.normal(0, 0.055, size=6)], 1)      # K sampled candidates

# ---------------- (1) Propose ----------------
ax = axes[0]
ax.fill_between(gx, gy - HALF, gy + HALF, color=BLUE, alpha=0.10, lw=0)
ax.plot(gx, gy, color=BLUE, lw=1.1, alpha=0.55)
ax.text(-1.28, -1.05, "actions the demonstrations support", fontsize=7.5, color=BLUE)
ax.scatter(*cand.T, s=34, color=GREY, zorder=3, edgecolor="white", lw=0.6)
ax.scatter(*a0, s=70, marker="*", color="black", zorder=4)
ax.annotate(r"$\pi_0(s,z_i)$", xy=a0, xytext=(a0[0] - 1.05, a0[1] + 0.75),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
ax.set_title("1. Propose\n" + r"frozen base $\to$ $K$ candidates", fontsize=9)

# ---------------- (2) Nudge + Leash ----------------
ax = axes[1]
ax.fill_between(gx, gy - HALF, gy + HALF, color=BLUE, alpha=0.10, lw=0)
ax.plot(gx, gy, color=BLUE, lw=1.1, alpha=0.55)
rho = 0.28
ax.add_patch(Circle(a0, rho, fill=False, ls=(0, (3, 2)), ec=BLUE, lw=1.2))
ax.text(a0[0] + rho * 0.72, a0[1] + rho * 0.80, r"leash $\rho$", fontsize=7.5, color=BLUE)

# critic points up-right, off the band: raw ask vs what CAST allows
raw = np.array([0.30, 1.45])
for c in cand:
    want = c + raw                                   # unclipped critic ask
    step = raw / np.linalg.norm(raw) * 0.30          # clipped to delta_Q
    tgt = c + step
    d = tgt - a0
    if np.linalg.norm(d) > rho:                      # anchor pulls it back
        tgt = a0 + d / np.linalg.norm(d) * rho
        ax.add_patch(FancyArrowPatch(c + step, tgt, arrowstyle="->",
                                     color=GREEN, lw=1.3, mutation_scale=8, zorder=5))
    ax.add_patch(FancyArrowPatch(c, want, arrowstyle="->", color=ORANGE, lw=0.7,
                                 alpha=0.35, ls=(0, (2, 2)), mutation_scale=6))
    ax.add_patch(FancyArrowPatch(c, c + step, arrowstyle="->", color=ORANGE,
                                 lw=1.5, mutation_scale=8, zorder=4))
    ax.scatter(*tgt, s=30, color=BLUE, zorder=6, edgecolor="white", lw=0.6)
ax.scatter(*cand.T, s=30, color=GREY, zorder=3, edgecolor="white", lw=0.6)
ax.scatter(*a0, s=70, marker="*", color="black", zorder=7)
ax.text(-1.28, 1.05, "critic asks (unbounded)", fontsize=7.5, color=ORANGE, alpha=0.8)
ax.text(-1.28, 0.90, r"clipped to $\delta_Q$", fontsize=7.5, color=ORANGE)
ax.text(-1.28, -0.98, "anchor pulls strays back", fontsize=7.5, color=GREEN)
ax.set_title("2. Nudge, then leash\n" + r"critic sets heading, we set stride", fontsize=9)

# ---------------- (3) Imitate ----------------
ax = axes[2]
ax.fill_between(gx, gy - HALF, gy + HALF, color=BLUE, alpha=0.10, lw=0)
ax.plot(gx, gy, color=BLUE, lw=1.1, alpha=0.55)
tgts = []
for c in cand:
    step = raw / np.linalg.norm(raw) * 0.30
    t = c + step; d = t - a0
    if np.linalg.norm(d) > rho:
        t = a0 + d / np.linalg.norm(d) * rho
    tgts.append(t)
tgts = np.array(tgts)
ax.scatter(*cand.T, s=30, color=GREY, alpha=0.45, zorder=3, edgecolor="white", lw=0.6)
ax.scatter(*tgts.T, s=34, color=BLUE, zorder=5, edgecolor="white", lw=0.6)
for c, t in zip(cand, tgts):
    ax.add_patch(FancyArrowPatch(c, t, arrowstyle="->", color=GREY, lw=0.8,
                                 alpha=0.55, mutation_scale=7))
ax.text(-1.28, 1.08, r"targets are frozen: $\mathrm{sg}[\cdot]$", fontsize=7.5, color="black")
ax.add_patch(FancyArrowPatch(cand.mean(0), tgts.mean(0), arrowstyle="-|>", color=BLUE,
                             lw=2.6, mutation_scale=15, zorder=8))
ax.text(tgts.mean(0)[0] + 0.10, tgts.mean(0)[1] + 0.12, "policy moves\nthis far, and stops",
        fontsize=7.5, color=BLUE, fontweight="bold")
ax.text(-1.28, -0.98, r"train $r_\theta$ by regression", fontsize=7.5, color=BLUE)
ax.scatter(*a0, s=70, marker="*", color="black", zorder=7)
ax.set_title("3. Imitate\n" + r"targets are data, not a graph", fontsize=9)

fig.tight_layout()
out = "iclr2026/figures/method_overview.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
