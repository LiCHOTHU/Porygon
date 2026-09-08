"""Fig. 3: CAST method overview (schematic, not measured data).

Four panels, one per named step in sec:field-update: Propose / Nudge / Leash /
Imitate. The previous version compressed the whole mechanism into a corner of
a wide manifold view -- the arrows overlapped into an unreadable knot -- and
drew three panels for four steps. This one zooms on the action region, keeps
the data band as context, and traces ONE highlighted candidate through the
pipeline so each panel adds a single visible operation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

rng = np.random.default_rng(4)
BLUE, ORANGE, GREY, GREEN, INK, MUT = "#0072B2", "#D55E00", "#8a8a8a", "#009E73", "#0b0b0b", "#52514e"

fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.9))
XL, XH, YL, YH = -0.80, 0.62, -0.86, 0.42          # zoomed on the action
for ax in axes:
    ax.set_xlim(XL, XH); ax.set_ylim(YL, YH)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color("#d9d8d4")

gx = np.linspace(XL, XH, 200)
gy = 0.55 * gx**2 - 0.45
HALF = 0.16
def band(ax):
    ax.fill_between(gx, gy - HALF, gy + HALF, color=BLUE, alpha=0.10, lw=0)
    ax.plot(gx, gy, color=BLUE, lw=1.0, alpha=0.5)

a0 = np.array([-0.15, 0.55 * (-0.15) ** 2 - 0.45])
_cx = np.array([-0.52, -0.34, -0.02, 0.13, 0.30, -0.20])
cand = np.stack([_cx, 0.55 * _cx**2 - 0.45 + rng.normal(0, 0.045, 6)], 1)
HERO = 4                                            # the candidate the reader follows
rho, dQ = 0.34, 0.22
raw = np.array([0.28, 1.30])                        # critic's raw (off-band) ask
unit = raw / np.linalg.norm(raw)
nudged = cand + unit * dQ
leashed = nudged.copy()
for i, t in enumerate(nudged):
    d = t - a0
    if np.linalg.norm(d) > rho:
        leashed[i] = a0 + d / np.linalg.norm(d) * rho

def dots(ax, pts, color, s=42, alpha=1.0, z=4):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, color=color, alpha=alpha, zorder=z,
               edgecolor="white", lw=0.7)

def title(ax, t, sub):
    ax.set_title(t, fontsize=11, loc="left", fontweight="bold", color=INK, pad=30)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.2, color=MUT, va="bottom")

# ------------------------------- 1. Propose -------------------------------
ax = axes[0]; band(ax)
dots(ax, cand, GREY)
ax.scatter(*a0, s=90, marker="*", color=INK, zorder=6)
for c in cand:
    ax.plot([a0[0], c[0]], [a0[1], c[1]], color=GREY, lw=0.6, alpha=0.35, zorder=2)
ax.annotate(r"$K$ candidates $\pi_0(s,z_i)$", xy=cand[1], xytext=(XL + 0.05, YH - 0.16),
            fontsize=8.4, color=INK,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
ax.text(XL + 0.05, YL + 0.05, "sampled from the frozen base;\nall on the data band",
        fontsize=7.8, color=BLUE)
title(ax, "1. Propose", "draw $K$ noise samples through\nthe frozen base")

# ------------------------------- 2. Nudge ---------------------------------
ax = axes[1]; band(ax)
dots(ax, cand, GREY, alpha=0.85)
for i, c in enumerate(cand):
    hero = (i == HERO)
    ax.add_patch(FancyArrowPatch(c, c + unit * 0.62, arrowstyle="->", color=ORANGE,
                                 lw=1.0 if hero else 0.7, alpha=0.45 if hero else 0.25,
                                 ls=(0, (2, 2)), mutation_scale=7))
    ax.add_patch(FancyArrowPatch(c, c + unit * dQ, arrowstyle="->", color=ORANGE,
                                 lw=2.4 if hero else 1.4, mutation_scale=11 if hero else 8,
                                 zorder=5))
dots(ax, nudged, ORANGE, s=30, z=6)
h = cand[HERO]
ax.annotate("the critic's full ask\n(off the band)", xy=h + unit * 0.60,
            xytext=(XL + 0.05, YH - 0.16), fontsize=7.8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7, alpha=0.6))
ax.text(h[0] + 0.14, h[1] + 0.02, r"step clipped to $\delta_Q$",
        fontsize=8.2, color=ORANGE, fontweight="bold")
title(ax, "2. Nudge", "the critic sets the heading;\nwe set the stride")

# ------------------------------- 3. Leash ---------------------------------
ax = axes[2]; band(ax)
ax.add_patch(Circle(a0, rho, fill=False, ls=(0, (4, 2)), ec=GREEN, lw=1.5, zorder=3))
ax.text(a0[0] - rho - 0.03, a0[1] + rho * 0.55, "dead zone\nradius $\\rho$",
        fontsize=8.2, color=GREEN, ha="right")
ax.scatter(*a0, s=90, marker="*", color=INK, zorder=7)
dots(ax, nudged, ORANGE, s=30, alpha=0.8)
for i, (t, l) in enumerate(zip(nudged, leashed)):
    moved = np.linalg.norm(t - l) > 1e-9
    if moved:
        ax.add_patch(FancyArrowPatch(t, l, arrowstyle="->", color=GREEN,
                                     lw=2.2 if i == HERO else 1.3,
                                     mutation_scale=11 if i == HERO else 8, zorder=6))
dots(ax, leashed, BLUE, s=36, z=7)
ax.text(XH - 0.03, YH - 0.16, "outside $\\rho$: pulled back to the rim\ninside $\\rho$: left alone",
        fontsize=7.8, color=GREEN, ha="right")
title(ax, "3. Leash", "restore only what strayed;\nthe interior is free")

# ------------------------------- 4. Imitate -------------------------------
ax = axes[3]; band(ax)
dots(ax, cand, GREY, s=30, alpha=0.45)
dots(ax, leashed, BLUE, s=40, z=6)
for i, (c, t) in enumerate(zip(cand, leashed)):
    ax.add_patch(FancyArrowPatch(c, t, arrowstyle="->", color=GREY,
                                 lw=1.6 if i == HERO else 0.8, alpha=0.7,
                                 mutation_scale=9 if i == HERO else 7, zorder=4))
ax.text(XL + 0.05, YH - 0.16, r"targets detached: $\mathrm{sg}[\tilde a_i]$",
        fontsize=8.2, color=INK)
ax.text(XL + 0.05, YL + 0.05,
        "residual head $r_\\theta$ regresses onto\nthe targets; no critic gradient\nflows into the policy",
        fontsize=7.8, color=BLUE)
title(ax, "4. Imitate", "the update is a regression,\nnot a backprop through $Q$")

fig.tight_layout()
out = "iclr2026/figures/method_overview.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
print("wrote", out)
