"""fig:containment, two panels.

(a) D_base(t): how far the policy walks from its frozen base during
    fine-tuning, per constraint arm, read from the residual_norm printed by
    dice_train.py.
(b) what that distance buys: final distance vs powered success, for both
    tasks. The point of the figure is the pairing -- arms that stay inside
    the trust radius keep (or improve) the base, arms that leave it score
    0.000, and the step clip alone does NOT keep them inside.
"""
import glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/hydra"
RHO = 0.05
RE = re.compile(r"\[iter (\d+)\] update.*?residual_norm=([0-9.]+)")

# arm -> (label, color, linestyle, glob per task, powered success per task)
ARMS = [
    ("no constraint",            "#D55E00", "-",  "*BNONE_{t}*_1220*",  {"t65": 0.000, "t32": 0.000}),
    ("step clip only",           "#E69F00", "--", "*BCLIP_{t}*_1220*",  {"t65": 0.000, "t32": 0.000}),
    ("dead-zone anchor only",    "#56B4E9", "-.", "*BANC_{t}*_1220*",   {"t65": 0.810, "t32": 0.387}),
    ("full CAST (clip+anchor)",  "#0072B2", "-",  "field_st_B_{t}_s1000*", {"t65": 0.781, "t32": 0.684}),
    ("hinge in the loss (DICE)", "#009E73", ":",  "*A_{t}_hinge*",      {"t65": 0.837, "t32": 0.700}),
]
BASE = {"t65": 0.573, "t32": 0.610}
TASKNAME = {"t65": "red mug onto plate", "t32": "ketchup into drawer"}


def curve(pattern):
    xs = {}
    for d in sorted(glob.glob(os.path.join(H, pattern))):
        for lf in glob.glob(os.path.join(d, "*.log")):
            for m in RE.finditer(open(lf, errors="ignore").read()):
                xs[int(m.group(1))] = float(m.group(2))
    return sorted(xs.items())


fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.5),
                               gridspec_kw=dict(width_ratios=[1.15, 1]))

# ---- (a) trajectories on the red-mug task ----
for label, c, ls, pat, _ in ARMS:
    pts = curve(pat.format(t="t65"))
    if not pts:
        print("MISSING", pat); continue
    axL.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=1.9, label=label)
    print(f"{label:26s} n={len(pts):3d} last={pts[-1][1]:.4f}")
axL.axhspan(3e-3, RHO, color="#0072B2", alpha=0.07, lw=0)
axL.axhline(RHO, color="#333333", lw=1.0, ls=(0, (2, 2)))
axL.text(1, RHO * 1.25, r"trust radius $\rho=0.05$", fontsize=8, color="#333333")
axL.text(20, 6e-3, "inside the leash:\nsuccess preserved", fontsize=7.5,
         color="#0072B2", va="center")
axL.set_yscale("log")
axL.set_ylim(4e-3, 60)
axL.set_xlabel("RL iteration")
axL.set_ylabel(r"distance from frozen base  $D_{\mathrm{base}}$")
axL.set_title("(a) how far the policy walks", fontsize=10)
axL.spines[["top", "right"]].set_visible(False)
axL.legend(fontsize=7.2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.30))

# ---- (b) distance vs success, both tasks ----
for label, c, ls, pat, succ in ARMS:
    for t, mk in (("t65", "o"), ("t32", "s")):
        pts = curve(pat.format(t=t))
        if not pts:
            continue
        axR.scatter(max(pts[-1][1], 1e-3), succ[t], s=62, marker=mk, color=c,
                    edgecolor="white", lw=0.8, zorder=4)
for t, mk, ls in (("t65", "o", "-"), ("t32", "s", "--")):
    axR.axhline(BASE[t], color="#888888", lw=0.9, ls=ls)
    axR.text(1.0, BASE[t] + 0.012, f"base, {TASKNAME[t]}", fontsize=7, color="#888888", ha="left")
axR.axvspan(6e-3, RHO, color="#0072B2", alpha=0.07, lw=0)
axR.axvline(RHO, color="#333333", lw=1.0, ls=(0, (2, 2)))
axR.text(RHO * 1.6, 0.20, "outside the leash:\n" + r"success collapses to $0.000$",
         fontsize=8, color="#D55E00")
axR.text(7e-3, 0.90, "inside the leash", fontsize=8, color="#0072B2")
axR.set_xscale("log")
axR.set_xlabel(r"final distance from base  $D_{\mathrm{base}}$")
axR.set_ylabel("powered success")
axR.set_ylim(-0.04, 0.95)
axR.set_title("(b) what that distance costs", fontsize=10)
axR.spines[["top", "right"]].set_visible(False)
axR.scatter([], [], marker="o", color="#555555", label="red mug onto plate")
axR.scatter([], [], marker="s", color="#555555", label="ketchup into drawer")
axR.legend(fontsize=7.2, frameon=False, loc="lower left")

fig.tight_layout()
out = "iclr2026/figures/containment_t65.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
