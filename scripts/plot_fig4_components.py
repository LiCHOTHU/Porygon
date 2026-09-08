"""Fig. 4, two panels.

(a) robomimic square learning curves: frozen base vs DICE-RL's actor vs CAST's
    actor. Same base, critic, data, replay and budget -- only the update differs.
(b) why each guard is needed: the LIBERO constraint factorial (complete, both
    tasks). Removing the anchor is fatal; the clip alone is fatal; the anchor
    alone survives but drops below base on one task; only the pair is reliable.
"""
import csv, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGDIR = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/imitation/logs"
EVAL_RE = re.compile(r"Eval at step (\d+): reward=([\d.]+)")
BASE_SQ = 0.382
XMAX, BUDGET = 40_000, 20_000
BLUE, ORANGE, GREY, RED, GREEN = "#0072B2", "#E69F00", "#666666", "#D55E00", "#009E73"

SRC = {"A": ["sq_ext_A_s42_*", "sq_ext_A_s43_*", "sq_ext_A_s44_*"],
       "B": ["sq_swp_B_qstep05_*", "sq_conf_Bq05_s43_*", "sq_conf_Bq05_s44_*"]}

def curve(pat):
    best = {}
    for path in glob.glob(os.path.join(LOGDIR, pat + ".out")):
        ev = {}
        for line in open(path, errors="ignore"):
            m = EVAL_RE.search(line)
            if m: ev[int(m.group(1))] = float(m.group(2))
        if ev and (not best or max(ev) > max(best)): best = ev
    return best

def agg(cs):
    steps = sorted({s for c in cs for s in c if 0 < s <= XMAX})
    m = [np.mean([c[s] for c in cs if s in c]) for s in steps]
    lo = [min(c[s] for c in cs if s in c) for s in steps]
    hi = [max(c[s] for c in cs if s in c) for s in steps]
    return np.array(steps), np.array(m), np.array(lo), np.array(hi)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.7),
                               gridspec_kw=dict(width_ratios=[1.05, 1]))

# ---------------- (a) head-to-head ----------------
for k, col, ls, lab in [("A", ORANGE, "--", "DICE-RL actor (backprop $-Q$)"),
                        ("B", BLUE, "-", "CAST actor (ours)")]:
    cs = [c for c in (curve(p) for p in SRC[k]) if c]
    if not cs: print("MISSING", k); continue
    x, m, lo, hi = agg(cs)
    axA.plot(x/1000, m, color=col, ls=ls, lw=2.2, label=lab, zorder=3)
    axA.fill_between(x/1000, lo, hi, color=col, alpha=0.15, lw=0)
    if k == "B":
        axA.fill_between(x/1000, BASE_SQ, m, where=(m > BASE_SQ), color=BLUE, alpha=0.08, lw=0)
axA.axhline(BASE_SQ, color="black", lw=1.1, ls=(0, (2, 2)))
axA.text(0.6, BASE_SQ+0.02, f"frozen base = {BASE_SQ:.2f}", fontsize=8)
axA.axvline(BUDGET/1000, color="#aaa", lw=0.9, ls="--")
axA.text(BUDGET/1000+0.4, 0.03, "matched budget", fontsize=7.5, rotation=90, color="#888")
axA.set_xlim(0, XMAX/1000); axA.set_ylim(0, 1.02)
axA.set_xlabel("training steps (K)"); axA.set_ylabel("success rate (300 episodes)")
axA.set_title("(a) same critic, same budget: only the update differs", fontsize=9.5)
kA = int(np.argmin(np.abs(x - BUDGET)))
axA.annotate("", xy=(BUDGET/1000, 0.924), xytext=(BUDGET/1000, 0.880),
             arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
axA.text(BUDGET/1000 - 0.6, 0.902,
         "at the matched budget\n0.924 vs 0.880 (+4.4 pts),\nand ahead on every seed",
         fontsize=7.5, color=BLUE, ha="right", va="center", fontweight="bold",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
axA.legend(fontsize=8, frameon=False, loc="lower right")
axA.spines[["top","right"]].set_visible(False); axA.grid(axis="y", color="#eee", lw=0.7); axA.set_axisbelow(True)

# ---------------- (b) the constraint factorial, all five tasks ----------------
# Bars: mean change from each task's own base across the five tasks of
# tab:constraint. Dots: the five per-task values. The earlier version showed
# two tasks, used since-corrected full-CAST numbers (0.781/0.684 -> 0.600/0.627),
# and was titled "each guard is necessary" -- a claim the completed factorial
# does not support. What it does support: full CAST is the only arm above base
# on every task.
arms  = ["no\nconstraint", "clip\nonly", "anchor\nonly", "clip + anchor\n(full CAST)"]
TASKS = ["bowl", "juice", "right caddy", "red mug", "ketchup"]
base  = dict(zip(TASKS, [0.818, 0.785, 0.578, 0.573, 0.610]))
vals  = {
    "no\nconstraint":            dict(zip(TASKS, [0.000]*5)),
    "clip\nonly":                dict(zip(TASKS, [0.000]*5)),
    "anchor\nonly":              dict(zip(TASKS, [0.807, 0.860, 0.423, 0.810, 0.387])),
    "clip + anchor\n(full CAST)":dict(zip(TASKS, [0.840, 0.820, 0.630, 0.600, 0.627])),
}
x = np.arange(len(arms))
deltas = {a: [vals[a][t] - base[t] for t in TASKS] for a in arms}
means  = [np.mean(deltas[a]) for a in arms]
cols   = [RED if m <= 0 else BLUE for m in means]
axB.bar(x, means, 0.58, color=cols, alpha=0.35, edgecolor="none", zorder=2)
rngB = np.random.default_rng(3)
for xi, a in enumerate(arms):
    for d in deltas[a]:
        axB.scatter(xi + rngB.uniform(-0.13, 0.13), d, s=26, zorder=4,
                    color=(RED if d <= 0 else BLUE), edgecolor="white", lw=0.6)
    axB.text(xi, means[xi] + (0.022 if means[xi] > 0 else -0.03),
             f"mean {means[xi]:+.3f}", ha="center",
             va=("bottom" if means[xi] > 0 else "top"),
             fontsize=7.2, color=(BLUE if means[xi] > 0 else RED), fontweight="bold")
axB.axhline(0, color="black", lw=1.2)
axB.text(-0.45, 0.015, "base level", fontsize=7.5, color="black", ha="left")
axB.annotate("above base on all five tasks\n($+0.017$ to $+0.052$)",
             xy=(3, 0.052), xytext=(1.75, 0.24), fontsize=7.6, color=BLUE,
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
axB.text(2.0, -0.45,
         "anchor alone swings $-0.223$ to $+0.237$\ndepending on the task",
         fontsize=7.2, color=RED, ha="center", va="center",
         bbox=dict(fc="white", ec=RED, lw=0.6, alpha=0.95, pad=2.5))
axB.set_xticks(x); axB.set_xticklabels(arms, fontsize=8)
axB.set_ylim(-0.9, 0.34)
axB.set_ylabel("change from that task's own base")
axB.set_title("(b) only the full pair clears base on every task (5 tasks)", fontsize=9.5)
axB.spines[["top","right"]].set_visible(False)
axB.grid(axis="y", color="#eee", lw=0.7); axB.set_axisbelow(True)

fig.tight_layout()
out = "iclr2026/figures/robomimic_curves.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
