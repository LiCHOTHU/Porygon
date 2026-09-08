"""Figure 2: the two-regime toy, drawn for a reader rather than a debugger.

The old figure was eight raw-matplotlib panels with internal arm names
(GRAD_free, FIELD_tilted) and no axis labels. The paper's point needs two
panels: in the easy regime backprop wins, in the hard regime it collapses
while its own Q-estimate explodes and the bounded field update holds the base.
Curves come from toy2_curves.json (3-seed means dumped by the toy itself).
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BAD, C_BAD2, C_GOOD, C_GOOD2, C_INK, C_MUT = "#eb6834", "#c44514", "#1baf7a", "#0f7a55", "#0b0b0b", "#52514e"
LABEL = {"GRAD_free": "backprop $-Q$", "GRAD_bc": "backprop $-Q$ + BC anchor",
         "FIELD_tilted": "bounded field (tilted)", "FIELD_topk": "bounded field (top-$k$)"}
COLOR = {"GRAD_free": C_BAD, "GRAD_bc": C_BAD2, "FIELD_tilted": C_GOOD, "FIELD_topk": C_GOOD2}

d = json.load(open("/storage/scratch1/8/lwang831/toy2/toy2_curves.json"))
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))
for ax, (key, title, sub) in zip(axes, [
    ("EASY_2d", "(a) easy regime: the critic's gradient generalizes",
     "2-D actions, dense critic data — backprop is the right tool"),
    ("HARD_16d", "(b) hard regime: the same update collapses",
     "16-D actions, 60 critic points — the regime post-training lives in")]):
    r = d[key]; t = r["t"]
    for nm in ("GRAD_free", "GRAD_bc", "FIELD_tilted", "FIELD_topk"):
        ax.plot(t, r[nm]["true_mean"], c=COLOR[nm], lw=2.0, label=LABEL[nm])
    ax.axhline(r["base"]["true_mean"], color=C_INK, lw=1.0, ls="--")
    ax.text(t[-1], r["base"]["true_mean"] + .012, "frozen base", fontsize=8,
            color=C_INK, ha="right")
    ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold", color=C_INK, pad=20)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.2, color=C_MUT, va="bottom")
    ax.set_xlabel("update step", fontsize=9, color=C_MUT)
    ax.set_ylabel("true reward", fontsize=9, color=C_MUT)
    ax.tick_params(labelsize=8, colors=C_MUT)
    ax.set_ylim(-0.03, None)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#d9d8d4")
# the smoking gun on the hard panel: the collapsing arm's own Q-estimate explodes
axh = axes[1]
ax2 = axh.twinx()
r = d["HARD_16d"]
ax2.plot(r["t"], r["GRAD_free"]["q_mean"], c=C_BAD, lw=1.2, ls=":")
ax2.set_ylabel("its own $\\hat{Q}$-estimate (dotted)", fontsize=8, color=C_BAD)
ax2.tick_params(labelsize=7, colors=C_BAD)
ax2.spines["right"].set_color(C_BAD); ax2.spines["top"].set_visible(False)
axh.text(.45, .45, "true reward $\\rightarrow 0$\nwhile $\\hat{Q}$ explodes",
         transform=axh.transAxes, fontsize=8.6, color=C_BAD, fontweight="bold")
axes[0].legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.05))
fig.tight_layout()
fig.savefig("iclr2026/figures/toy_regimes.pdf", bbox_inches="tight", dpi=300)
print("wrote iclr2026/figures/toy_regimes.pdf")
